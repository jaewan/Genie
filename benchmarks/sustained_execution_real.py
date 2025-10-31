"""
Sustained Execution Benchmarks with REAL Workloads
===================================================

Measures performance over many sequential requests using REAL models:
- GPT-2-XL (1.5B parameters) for LLM workloads
- ResNet-50 (25M parameters) for vision workloads

This captures production behavior:
- Cache warming effects
- Connection pooling benefits
- Memory stability over time
- Performance evolution (cold → warm → steady state)
"""

import time
import json
import psutil
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import argparse

# Import baselines
from benchmarks.baselines import (
    LocalPyTorchBaseline,
    NaiveDisaggregationBaseline,
    GenieFullBaseline,
)

# Import REAL workloads
from benchmarks.workloads_detailed import (
    RealisticLLMDecodeWorkload,
    RealisticVisionCNNWorkload,
)


@dataclass
class SustainedExecutionMetrics:
    """Metrics collected during sustained execution"""
    request_num: int
    latency_ms: float
    memory_mb: float
    cache_stats: Dict[str, Any]
    timestamp: float


class CacheStatsCollector:
    """Collects cache statistics from Genie components"""
    
    @staticmethod
    def get_cache_stats() -> Dict[str, Any]:
        """Get current cache statistics"""
        stats = {
            'shape_cache_hits': 0,
            'shape_cache_misses': 0,
            'pattern_cache_hits': 0,
            'pattern_cache_misses': 0,
            'universal_dispatch_success': 0,
            'special_handler_used': 0,
        }
        
        try:
            # Get shape inference cache stats
            from genie.core.shape_inference import ShapeInference
            if hasattr(ShapeInference, '_cache'):
                stats['shape_cache_size'] = len(ShapeInference._cache)
        except Exception:
            pass
        
        try:
            # Get universal dispatcher stats
            from genie.core import executor as executor_module
            executor = executor_module._executor
            if hasattr(executor, 'universal_dispatcher'):
                dispatcher_stats = executor.universal_dispatcher.get_stats()
                stats.update(dispatcher_stats)
        except Exception:
            pass
        
        return stats
    
    @staticmethod
    def reset_cache_stats():
        """Reset cache statistics"""
        try:
            from genie.core import executor as executor_module
            executor = executor_module._executor
            if hasattr(executor, 'universal_dispatcher'):
                executor.universal_dispatcher.reset_stats()
        except Exception:
            pass


class SustainedExecutionBenchmarkReal:
    """Runs sustained execution benchmarks with REAL models"""
    
    def __init__(self, 
                 output_dir: str = "sustained_execution_real_results",
                 num_requests: int = 100,
                 measure_every: int = 10,
                 workloads: List[str] = None):
        """
        Initialize sustained execution benchmark with real models.
        
        Args:
            output_dir: Output directory for results
            num_requests: Number of requests to run (default: 100)
            measure_every: Measure metrics every N requests (default: 10)
            workloads: List of workloads to run ['llm', 'vision'] (default: both)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.num_requests = num_requests
        self.measure_every = measure_every
        
        # Initialize baselines (subset for sustained execution)
        print("🔧 Initializing baselines...")
        self.baselines = {
            'genie_full': GenieFullBaseline(),
            'local_pytorch': LocalPyTorchBaseline(),
        }
        
        # Initialize REAL workloads
        self.workload_names = workloads or ['llm', 'vision']
        self._load_workloads()
        
        self.results = []
        self.process = psutil.Process()
    
    def _load_workloads(self):
        """Load REAL workloads"""
        print("📚 Loading REAL models (this may take a minute)...")
        self.workloads = {}
        
        if 'llm' in self.workload_names:
            print("  Loading GPT-2-XL (1.5B parameters)...")
            self.workloads['llm_decode'] = RealisticLLMDecodeWorkload(use_real_model=True)
            print("  ✓ GPT-2-XL loaded")
        
        if 'vision' in self.workload_names:
            print("  Loading ResNet-50 (25M parameters)...")
            self.workloads['vision_cnn'] = RealisticVisionCNNWorkload()
            print("  ✓ ResNet-50 loaded")
        
        print(f"✓ Loaded {len(self.workloads)} real workloads")
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            mem_info = self.process.memory_info()
            return mem_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0
    
    def run_sustained_execution(self, baseline_name: str, baseline: Any,
                               workload_name: str, workload: Any) -> Dict[str, Any]:
        """
        Run sustained execution for a single baseline × workload combination.
        
        Returns:
            Dictionary with metrics over time
        """
        print(f"\n{'='*80}")
        print(f"Running sustained execution: {baseline_name} × {workload_name}")
        metadata = workload.get_metadata()
        print(f"  Model: {metadata.get('model', 'Unknown')}")
        print(f"  Requests: {self.num_requests}")
        print(f"  Measuring every: {self.measure_every}")
        print(f"{'='*80}")
        
        metrics_over_time = []
        latencies = []
        
        # Reset cache stats
        CacheStatsCollector.reset_cache_stats()
        
        # Clear GPU cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        for i in range(self.num_requests):
            # Get sample inputs for this request
            sample_inputs = workload.get_sample_inputs()
            
            # Run single request
            request_start = time.perf_counter()
            try:
                # Synchronize before timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                result = baseline.run(workload.model, sample_inputs)
                
                # Synchronize after execution
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                latency = (time.perf_counter() - request_start) * 1000  # Convert to ms
                latencies.append(latency)
                success = True
            except Exception as e:
                print(f"  ⚠️ Request {i} failed: {e}")
                latency = 0.0
                success = False
            
            # Measure metrics periodically
            if i % self.measure_every == 0 or i == 0:
                memory_mb = self.get_memory_usage_mb()
                cache_stats = CacheStatsCollector.get_cache_stats()
                
                metrics = SustainedExecutionMetrics(
                    request_num=i,
                    latency_ms=latency,
                    memory_mb=memory_mb,
                    cache_stats=cache_stats,
                    timestamp=time.time() - start_time
                )
                metrics_over_time.append(asdict(metrics))
                
                # Print progress
                print(f"  Request {i:4d}: {latency:7.2f}ms | "
                      f"Memory: {memory_mb:6.1f}MB | "
                      f"Cache hits: {cache_stats.get('universal_dispatch_success', 0)}")
        
        # Analyze results
        analysis = self._analyze_results(baseline_name, workload_name, latencies, metrics_over_time)
        
        # Print summary
        print(f"\n📊 Summary for {baseline_name} × {workload_name}:")
        print(f"  Cold start (first 10):  {analysis['cold_start_ms']:.2f}ms")
        print(f"  Warm (mid-point):       {analysis['warm_ms']:.2f}ms")
        print(f"  Steady state (last 10): {analysis['steady_state_ms']:.2f}ms")
        print(f"  Improvement:            {analysis['improvement_pct']:.1f}%")
        print(f"  Memory stable:          {analysis['memory_stable']}")
        print(f"  Mean latency:           {analysis['mean_latency_ms']:.2f}ms")
        print(f"  Std deviation:          {analysis['std_latency_ms']:.2f}ms")
        
        return analysis
    
    def _analyze_results(self, baseline_name: str, workload_name: str,
                        latencies: List[float], metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze sustained execution results"""
        
        if not latencies:
            return {
                'baseline': baseline_name,
                'workload': workload_name,
                'num_requests': 0,
                'cold_start_ms': 0,
                'warm_ms': 0,
                'steady_state_ms': 0,
                'improvement_pct': 0,
                'mean_latency_ms': 0,
                'std_latency_ms': 0,
                'min_latency_ms': 0,
                'max_latency_ms': 0,
                'memory_start_mb': 0,
                'memory_end_mb': 0,
                'memory_growth_mb': 0,
                'memory_stable': False,
                'cache_stats_final': {},
                'all_latencies': [],
                'metrics_over_time': [],
            }
        
        # Calculate latency statistics
        cold_start_latencies = latencies[:10] if len(latencies) >= 10 else latencies
        mid_point = len(latencies) // 2
        warm_latencies = latencies[mid_point:mid_point+10] if len(latencies) >= mid_point+10 else latencies[-10:]
        steady_latencies = latencies[-10:] if len(latencies) >= 10 else latencies
        
        cold_start_ms = np.mean(cold_start_latencies)
        warm_ms = np.mean(warm_latencies)
        steady_state_ms = np.mean(steady_latencies)
        
        improvement_pct = ((cold_start_ms - steady_state_ms) / cold_start_ms * 100) if cold_start_ms > 0 else 0
        
        # Calculate memory stability
        memory_values = [m['memory_mb'] for m in metrics]
        memory_start = memory_values[0] if memory_values else 0
        memory_end = memory_values[-1] if memory_values else 0
        memory_growth_mb = memory_end - memory_start
        memory_stable = abs(memory_growth_mb) < 500  # Less than 500MB growth is acceptable for real models
        
        # Calculate cache statistics
        cache_stats_final = metrics[-1]['cache_stats'] if metrics else {}
        
        return {
            'baseline': baseline_name,
            'workload': workload_name,
            'num_requests': len(latencies),
            'cold_start_ms': cold_start_ms,
            'warm_ms': warm_ms,
            'steady_state_ms': steady_state_ms,
            'improvement_pct': improvement_pct,
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'memory_start_mb': memory_start,
            'memory_end_mb': memory_end,
            'memory_growth_mb': memory_growth_mb,
            'memory_stable': memory_stable,
            'cache_stats_final': cache_stats_final,
            'all_latencies': latencies,
            'metrics_over_time': metrics,
        }
    
    def run_all(self) -> Dict[str, Any]:
        """Run sustained execution for all baseline × workload combinations"""
        print("\n" + "="*80)
        print("SUSTAINED EXECUTION BENCHMARKS (REAL MODELS)")
        print("="*80)
        print(f"Configuration:")
        print(f"  Baselines: {len(self.baselines)}")
        print(f"  Workloads: {len(self.workloads)}")
        print(f"  Requests per combination: {self.num_requests}")
        print(f"  Total requests: {len(self.baselines) * len(self.workloads) * self.num_requests}")
        print(f"  Output directory: {self.output_dir}")
        print(f"")
        print(f"⚠️  This will take a while with real models!")
        print(f"   Estimated time: ~{len(self.baselines) * len(self.workloads) * self.num_requests * 0.5 / 60:.0f} minutes")
        print("")
        
        all_results = []
        
        for baseline_name, baseline in self.baselines.items():
            for workload_name, workload in self.workloads.items():
                try:
                    result = self.run_sustained_execution(
                        baseline_name, baseline, workload_name, workload
                    )
                    all_results.append(result)
                    
                    # Save intermediate results after each run
                    self._save_results(all_results)
                    
                except Exception as e:
                    print(f"❌ Failed: {baseline_name} × {workload_name}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Print final summary
        self._print_summary(all_results)
        
        return {
            'results': all_results,
            'num_successful': len(all_results),
            'output_dir': str(self.output_dir),
        }
    
    def _save_results(self, results: List[Dict[str, Any]]):
        """Save results to JSON"""
        output_file = self.output_dir / "sustained_execution_real_results.json"
        
        # Remove large arrays for summary file
        summary_results = []
        for r in results:
            summary = {k: v for k, v in r.items() 
                      if k not in ['all_latencies', 'metrics_over_time']}
            summary_results.append(summary)
        
        with open(output_file, 'w') as f:
            json.dump({
                'summary': summary_results,
                'config': {
                    'num_requests': self.num_requests,
                    'measure_every': self.measure_every,
                    'use_real_models': True,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
            }, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        # Save detailed results
        detailed_file = self.output_dir / "sustained_execution_real_detailed.json"
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Detailed results saved to: {detailed_file}")
    
    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print summary of all results"""
        print("\n" + "="*80)
        print("SUSTAINED EXECUTION SUMMARY (REAL MODELS)")
        print("="*80)
        
        print(f"\n{'Baseline':<20} {'Workload':<15} {'Cold Start':<15} {'Steady State':<15} {'Improvement':<12}")
        print("-" * 80)
        
        for r in results:
            print(f"{r['baseline']:<20} {r['workload']:<15} "
                  f"{r['cold_start_ms']:>12.2f}ms {r['steady_state_ms']:>12.2f}ms "
                  f"{r['improvement_pct']:>10.1f}%")
        
        print("="*80)
        
        # Print additional insights
        print("\n📊 Key Insights:")
        for r in results:
            if 'genie' in r['baseline'].lower():
                print(f"\n{r['baseline']} × {r['workload']}:")
                print(f"  Mean latency: {r['mean_latency_ms']:.2f}ms ± {r['std_latency_ms']:.2f}ms")
                print(f"  P50: {r['p50_latency_ms']:.2f}ms | P95: {r['p95_latency_ms']:.2f}ms | P99: {r['p99_latency_ms']:.2f}ms")
                print(f"  Memory: {r['memory_start_mb']:.1f}MB → {r['memory_end_mb']:.1f}MB (Δ{r['memory_growth_mb']:+.1f}MB)")
                print(f"  Stable: {'✅' if r['memory_stable'] else '❌'}")


def main():
    """Main entry point for sustained execution benchmarks with real models"""
    parser = argparse.ArgumentParser(description='Run sustained execution benchmarks with REAL models')
    parser.add_argument('--num-requests', type=int, default=100,
                       help='Number of requests per combination (default: 100)')
    parser.add_argument('--measure-every', type=int, default=10,
                       help='Measure metrics every N requests (default: 10)')
    parser.add_argument('--workloads', nargs='+', choices=['llm', 'vision'], default=['llm', 'vision'],
                       help='Workloads to run (default: both)')
    parser.add_argument('--output-dir', type=str, default='sustained_execution_real_results',
                       help='Output directory (default: sustained_execution_real_results)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SUSTAINED EXECUTION BENCHMARKS WITH REAL MODELS")
    print("="*80)
    print(f"Configuration:")
    print(f"  Requests per combination: {args.num_requests}")
    print(f"  Workloads: {', '.join(args.workloads)}")
    print(f"  Output directory: {args.output_dir}")
    print("="*80)
    
    benchmark = SustainedExecutionBenchmarkReal(
        output_dir=args.output_dir,
        num_requests=args.num_requests,
        measure_every=args.measure_every,
        workloads=args.workloads
    )
    
    results = benchmark.run_all()
    
    print(f"\n✅ Sustained execution benchmarks complete!")
    print(f"   Successful runs: {results['num_successful']}")
    print(f"   Results saved to: {results['output_dir']}")


if __name__ == "__main__":
    main()

