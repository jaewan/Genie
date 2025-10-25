"""
Phase 1: Realistic Workload Evaluation (SOSP Focus).

This script runs the realistic production-scale workloads that address the
peer review critique: microbenchmarks with 140ms overhead on 6ms workloads
mask semantic optimizations.

Key changes from microbenchmark evaluation:
- 5-10 second workloads (not 6ms)
- Real batch sizes (8-64 items, not 1)
- Realistic models (GPT-J 6B, BERT-large, ResNet-152)
- Expected overhead amortizes to 5-20% (not 95%)
- Semantic optimizations become visible

Run: python -m benchmarks.realistic_evaluation --output paper_results_realistic/
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import time
import numpy as np
import torch

# Import realistic workloads
from benchmarks.workloads_detailed import (
    RealisticLLMDecodeWorkload,
    RealisticLLMPrefillWorkload,
    RealisticVisionCNNWorkload,
)

# Import baselines (same as before)
from benchmarks.baselines import (
    LocalPyTorchBaseline,
    GenieCaptureOnlyBaseline,
    GenieLocalRemoteBaseline,
    GenieNoSemanticsBaseline,
    GenieFullBaseline,
)


class RealisticEvaluation:
    """Runs production-scale workload evaluation for SOSP."""

    def __init__(self, output_dir: str = "paper_results_realistic"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Key baselines (reduced set for realistic evaluation)
        # Skip PyTorch RPC and Ray as they're misconfigured in current evaluation
        self.baselines = {
            '1_local_pytorch': LocalPyTorchBaseline(),
            '2_genie_capture': GenieCaptureOnlyBaseline(),
            '3_genie_no_semantics': GenieNoSemanticsBaseline(),
            '4_genie_full': GenieFullBaseline(),
        }

        # Realistic workloads (production scale)
        self.workloads = {
            'realistic_llm_decode': RealisticLLMDecodeWorkload(
                model_name="gpt2-xl",
                max_new_tokens=512,
                batch_size=8
            ),
            'realistic_llm_prefill': RealisticLLMPrefillWorkload(
                model_name="bert-large-uncased",
                batch_size=32,
                max_length=2048
            ),
            'realistic_vision_cnn': RealisticVisionCNNWorkload(
                model_name="resnet152",
                batch_size=64,
                image_size=384
            ),
        }

        self.results = []
        self.successful_runs = 0
        self.failed_runs = 0

    def run_sync(self, num_runs: int = 3, num_warmup: int = 1):
        """Run all baseline × workload combinations synchronously."""
        
        num_baselines = len(self.baselines)
        num_workloads = len(self.workloads)
        total_experiments = num_baselines * num_workloads * num_runs

        print(f"\n{'='*80}")
        print(f"🎯 REALISTIC WORKLOAD EVALUATION (Phase 1)")
        print(f"{'='*80}")
        print(f"📊 Configurations: {num_baselines} baselines × {num_workloads} workloads")
        print(f"📈 Total experiments: {total_experiments} ({num_runs} runs each)")
        print(f"🔥 Warm-up runs: {num_warmup} per configuration")
        print(f"📁 Output: {self.output_dir}")
        print(f"\n📋 Expected scale improvements:")
        print(f"   LLM Decode:   140ms overhead on 10sec workload = 1.4% overhead (was 95%)")
        print(f"   LLM Prefill:  140ms overhead on 4sec workload  = 3.5% overhead (was 95%)")
        print(f"   Vision CNN:   140ms overhead on 2.5sec workload= 5.6% overhead (was 95%)")
        print(f"{'='*80}\n")

        completed = 0

        for baseline_name, baseline in self.baselines.items():
            print(f"\n{'='*60}")
            print(f"📌 Baseline: {baseline_name}")
            print(f"{'='*60}")

            for workload_name, workload in self.workloads.items():
                print(f"\n  📌 Workload: {workload_name}")
                print(f"  Metadata: {workload.get_metadata()}")
                
                latencies_ms = []

                for run_num in range(num_runs + num_warmup):
                    try:
                        # Warm-up runs (not counted)
                        if run_num < num_warmup:
                            print(f"    ⏳ Warm-up run {run_num + 1}/{num_warmup}...", end=" ")
                            _ = baseline.run(workload.model, workload.get_sample_inputs())
                            print("✓")
                            continue

                        # Measurement run
                        actual_run = run_num - num_warmup + 1
                        print(f"    ⏱️  Measurement run {actual_run}/{num_runs}...", end=" ")

                        # Clear GPU cache
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize() if torch.cuda.is_available() else None

                        # Run and measure
                        start_time = time.perf_counter()
                        sample_inputs = workload.get_sample_inputs()
                        # Debug: print what we got
                        if baseline_name == '1_local_pytorch' and workload_name == 'realistic_llm_decode':
                            print(f"\n    DEBUG: sample_inputs type: {type(sample_inputs)}, len: {len(sample_inputs)}")
                            if len(sample_inputs) > 0:
                                print(f"    DEBUG: sample_inputs[0] type: {type(sample_inputs[0])}")
                        result = baseline.run(workload.model, sample_inputs)
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        end_time = time.perf_counter()

                        latency_ms = (end_time - start_time) * 1000

                        # Verify output correctness
                        reference = workload.run_reference()
                        if not self._outputs_match(reference, result):
                            print(f"❌ Output mismatch!")
                            self.failed_runs += 1
                            continue

                        latencies_ms.append(latency_ms)
                        print(f"✓ {latency_ms:.2f}ms")
                        self.successful_runs += 1
                        completed += 1

                    except Exception as e:
                        print(f"❌ Error: {e}")
                        self.failed_runs += 1

                # Analyze results for this configuration
                if latencies_ms:
                    mean = np.mean(latencies_ms)
                    std = np.std(latencies_ms)
                    min_lat = np.min(latencies_ms)
                    max_lat = np.max(latencies_ms)

                    print(f"\n  📊 Results:")
                    print(f"     Mean: {mean:.2f}ms (σ={std:.2f}ms)")
                    print(f"     Range: {min_lat:.2f}ms - {max_lat:.2f}ms")

                    # Compute speedup vs local PyTorch
                    if baseline_name != '1_local_pytorch' and latencies_ms:
                        local_key = '1_local_pytorch'
                        local_results = [r for r in self.results 
                                       if r['baseline'] == local_key and r['workload'] == workload_name]
                        if local_results:
                            local_mean = np.mean(local_results[0]['latencies_ms'])
                            speedup = local_mean / mean
                            print(f"     Speedup vs local: {speedup:.2f}x")

                    # Store results
                    self.results.append({
                        'baseline': baseline_name,
                        'workload': workload_name,
                        'latencies_ms': latencies_ms,
                        'mean_ms': mean,
                        'std_ms': std,
                        'min_ms': min_lat,
                        'max_ms': max_lat,
                    })

        print(f"\n{'='*80}")
        print(f"✅ EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"✓ Successful runs: {self.successful_runs}")
        print(f"❌ Failed runs: {self.failed_runs}")
        print(f"Success rate: {100 * self.successful_runs / (self.successful_runs + self.failed_runs):.1f}%")

        # Generate report
        self._generate_report()
        self._save_results()

    def _outputs_match(self, ref: torch.Tensor, test: torch.Tensor, rtol: float = 1e-2, atol: float = 1e-3) -> bool:
        """Check if outputs match (with tolerance)."""
        if not isinstance(test, torch.Tensor):
            return False
        if ref.shape != test.shape:
            return False
        return torch.allclose(ref, test, rtol=rtol, atol=atol)

    def _generate_report(self):
        """Generate comparison report."""
        print(f"\n{'='*80}")
        print(f"📊 PERFORMANCE ANALYSIS")
        print(f"{'='*80}\n")

        try:
            import pandas as pd
            
            # Create summary table
            summary_data = []
            for result in self.results:
                summary_data.append({
                    'Baseline': result['baseline'],
                    'Workload': result['workload'],
                    'Mean (ms)': f"{result['mean_ms']:.2f}",
                    'Std (ms)': f"{result['std_ms']:.2f}",
                    'Min (ms)': f"{result['min_ms']:.2f}",
                    'Max (ms)': f"{result['max_ms']:.2f}",
                })
            
            df = pd.DataFrame(summary_data)
            print(df.to_string(index=False))

            # Compute speedups
            print(f"\n{'='*80}")
            print(f"⚡ SPEEDUP ANALYSIS (vs Local PyTorch)")
            print(f"{'='*80}\n")

            for workload_name in self.workloads.keys():
                print(f"📌 {workload_name}:")
                
                workload_results = [r for r in self.results if r['workload'] == workload_name]
                local_results = [r for r in workload_results if r['baseline'] == '1_local_pytorch']
                
                if not local_results:
                    print("  (No local baseline results)")
                    continue
                
                local_mean = local_results[0]['mean_ms']
                
                for result in workload_results:
                    if result['baseline'] != '1_local_pytorch':
                        speedup = local_mean / result['mean_ms']
                        improvement = (1 - speedup) * 100
                        status = "✓" if speedup > 0.9 else "✗"
                        print(f"  {status} {result['baseline']}: {speedup:.2f}x ({improvement:+.1f}%)")

            # Semantic benefit analysis
            print(f"\n{'='*80}")
            print(f"🎯 SEMANTIC OPTIMIZATION IMPACT")
            print(f"{'='*80}\n")

            for workload_name in self.workloads.keys():
                print(f"📌 {workload_name}:")
                
                workload_results = [r for r in self.results if r['workload'] == workload_name]
                no_semantics = [r for r in workload_results if r['baseline'] == '3_genie_no_semantics']
                full_semantics = [r for r in workload_results if r['baseline'] == '4_genie_full']
                
                if no_semantics and full_semantics:
                    no_sem_mean = no_semantics[0]['mean_ms']
                    full_sem_mean = full_semantics[0]['mean_ms']
                    speedup = no_sem_mean / full_sem_mean
                    improvement = (1 - 1/speedup) * 100
                    
                    print(f"  No Semantics: {no_sem_mean:.2f}ms")
                    print(f"  Full Semantics: {full_sem_mean:.2f}ms")
                    print(f"  Speedup from Semantics: {speedup:.2f}x ({improvement:+.1f}%)")
                    
                    if speedup > 1.1:
                        print(f"  ✓ SIGNIFICANT improvement!")
                    else:
                        print(f"  ⚠️  Minimal improvement (needs investigation)")

        except ImportError:
            print("⚠️  pandas not available - skipping formatted report")

    def _save_results(self):
        """Save results to JSON."""
        output_file = self.output_dir / "realistic_results.json"
        
        # Prepare serializable results
        serializable_results = []
        for result in self.results:
            serializable_results.append({
                'baseline': result['baseline'],
                'workload': result['workload'],
                'mean_ms': float(result['mean_ms']),
                'std_ms': float(result['std_ms']),
                'min_ms': float(result['min_ms']),
                'max_ms': float(result['max_ms']),
            })
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n📁 Results saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Realistic workload evaluation")
    parser.add_argument("--output", default="paper_results_realistic", help="Output directory")
    parser.add_argument("--runs", type=int, default=3, help="Number of measurement runs")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warm-up runs")
    
    args = parser.parse_args()
    
    eval_engine = RealisticEvaluation(output_dir=args.output)
    eval_engine.run_sync(num_runs=args.runs, num_warmup=args.warmup)
