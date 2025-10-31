"""
Sustained Execution Benchmarks
================================

Measures performance over many sequential requests to capture:
- Cache warming effects
- Connection pooling benefits
- Memory stability
- Production behavior patterns

This complements the single-shot benchmarks by showing how performance
evolves over time in production scenarios.
"""

import time
import json
import psutil
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Import baselines
from benchmarks.baselines import (
    LocalPyTorchBaseline,
    NaiveDisaggregationBaseline,
    GenieCaptureOnlyBaseline,
    GenieLocalRemoteBaseline,
    GenieNoSemanticsBaseline,
    GenieFullBaseline,
    PyTorchRPCBaseline,
    RayBaseline
)

# Import workloads
from benchmarks.workloads_detailed import (
    RealisticLLMDecodeWorkload,
    RealisticVisionCNNWorkload,
    MicrobenchmarkWorkload
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
            # Get semantic annotator cache stats
            from genie.semantic.annotator import SemanticAnnotator
            # Add cache stats if available
            pass
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
        """Reset cache statistics (for testing)"""
        try:
            from genie.core import executor as executor_module
            executor = executor_module._executor
            if hasattr(executor, 'universal_dispatcher'):
                executor.universal_dispatcher.reset_stats()
        except Exception:
            pass


class SustainedExecutionBenchmark:
    """Runs sustained execution benchmarks"""
    
    def __init__(self, output_dir: str = "sustained_execution_results", 
                 num_requests: int = 100, 
                 measure_every: int = 10,
                 use_real_models: bool = False):
        """
        Initialize sustained execution benchmark.
        
        Args:
            output_dir: Output directory for results
            num_requests: Number of requests to run (default: 100 for quick test)
            measure_every: Measure metrics every N requests (default: 10)
            use_real_models: Use real models (slower) or mock models (faster)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.num_requests = num_requests
        self.measure_every = measure_every
        self.use_real_models = use_real_models
        
        # Initialize baselines (subset for quick testing)
        self.baselines = {
            'local_pytorch': LocalPyTorchBaseline(),
            'naive_disaggregation': NaiveDisaggregationBaseline(),
            'genie_full': GenieFullBaseline(),
        }
        
        # Initialize workloads
        self._load_workloads()
        
        self.results = []
        self.process = psutil.Process()
    
    def _load_workloads(self):
        """Load workloads for testing"""
        if self.use_real_models:
            print("📚 Using REAL models (slower but realistic)")
            self.workloads = {
                'llm_decode': RealisticLLMDecodeWorkload(use_real_model=True),
                'vision_cnn': RealisticVisionCNNWorkload(),
            }
        else:
            print("📦 Using MOCK models (faster for validation)")
            self.workloads = {
                'microbenchmark': MicrobenchmarkWorkload(),
            }
    
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
        print(f"  Requests: {self.num_requests}")
        print(f"  Measuring every: {self.measure_every}")
        print(f"{'='*80}")
        
        metrics_over_time = []
        latencies = []
        
        # Reset cache stats
        CacheStatsCollector.reset_cache_stats()
        
        start_time = time.time()
        
        for i in range(self.num_requests):
            # Get sample inputs for this request
            sample_inputs = workload.get_sample_inputs()
            
            # Run single request
            request_start = time.perf_counter()
            try:
                result = baseline.run(workload.model, sample_inputs)
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
                if i % (self.measure_every * 10) == 0:
                    print(f"  Request {i:4d}: {latency:7.2f}ms | "
                          f"Memory: {memory_mb:6.1f}MB | "
                          f"Cache hits: {cache_stats.get('universal_dispatch_success', 0)}")
        
        # Analyze results
        analysis = self._analyze_results(baseline_name, workload_name, latencies, metrics_over_time)
        
        # Print summary
        print(f"\n📊 Summary for {baseline_name} × {workload_name}:")
        print(f"  Cold start (first 10):  {analysis['cold_start_ms']:.2f}ms")
        print(f"  Warm (100-110):         {analysis['warm_ms']:.2f}ms")
        print(f"  Steady state (last 10): {analysis['steady_state_ms']:.2f}ms")
        print(f"  Improvement:            {analysis['improvement_pct']:.1f}%")
        print(f"  Memory stable:          {analysis['memory_stable']}")
        
        return analysis
    
    def _analyze_results(self, baseline_name: str, workload_name: str,
                        latencies: List[float], metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze sustained execution results"""
        
        # Calculate latency statistics
        cold_start_latencies = latencies[:10] if len(latencies) >= 10 else latencies
        warm_latencies = latencies[100:110] if len(latencies) >= 110 else latencies[-10:]
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
        memory_stable = abs(memory_growth_mb) < 100  # Less than 100MB growth
        
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
        print("SUSTAINED EXECUTION BENCHMARKS")
        print("="*80)
        print(f"Configuration:")
        print(f"  Baselines: {len(self.baselines)}")
        print(f"  Workloads: {len(self.workloads)}")
        print(f"  Requests per combination: {self.num_requests}")
        print(f"  Total requests: {len(self.baselines) * len(self.workloads) * self.num_requests}")
        print(f"  Output directory: {self.output_dir}")
        
        all_results = []
        
        for baseline_name, baseline in self.baselines.items():
            for workload_name, workload in self.workloads.items():
                try:
                    result = self.run_sustained_execution(
                        baseline_name, baseline, workload_name, workload
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"❌ Failed: {baseline_name} × {workload_name}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Save results
        self._save_results(all_results)
        
        # Print final summary
        self._print_summary(all_results)
        
        return {
            'results': all_results,
            'num_successful': len(all_results),
            'output_dir': str(self.output_dir),
        }
    
    def _save_results(self, results: List[Dict[str, Any]]):
        """Save results to JSON"""
        output_file = self.output_dir / "sustained_execution_results.json"
        
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
                    'use_real_models': self.use_real_models,
                }
            }, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        # Save detailed results
        detailed_file = self.output_dir / "sustained_execution_detailed.json"
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Detailed results saved to: {detailed_file}")
    
    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print summary of all results"""
        print("\n" + "="*80)
        print("SUSTAINED EXECUTION SUMMARY")
        print("="*80)
        
        print(f"\n{'Baseline':<25} {'Workload':<15} {'Cold Start':<12} {'Steady State':<12} {'Improvement':<12}")
        print("-" * 80)
        
        for r in results:
            print(f"{r['baseline']:<25} {r['workload']:<15} "
                  f"{r['cold_start_ms']:>10.2f}ms {r['steady_state_ms']:>10.2f}ms "
                  f"{r['improvement_pct']:>10.1f}%")
        
        print("="*80)


def main():
    """Main entry point for sustained execution benchmarks"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run sustained execution benchmarks')
    parser.add_argument('--num-requests', type=int, default=100,
                       help='Number of requests per combination (default: 100)')
    parser.add_argument('--measure-every', type=int, default=10,
                       help='Measure metrics every N requests (default: 10)')
    parser.add_argument('--real-models', action='store_true',
                       help='Use real models (slower but realistic)')
    parser.add_argument('--output-dir', type=str, default='sustained_execution_results',
                       help='Output directory (default: sustained_execution_results)')
    
    args = parser.parse_args()
    
    benchmark = SustainedExecutionBenchmark(
        output_dir=args.output_dir,
        num_requests=args.num_requests,
        measure_every=args.measure_every,
        use_real_models=args.real_models
    )
    
    results = benchmark.run_all()
    
    print(f"\n✅ Sustained execution benchmarks complete!")
    print(f"   Successful runs: {results['num_successful']}")
    print(f"   Results saved to: {results['output_dir']}")


if __name__ == "__main__":
    main()

