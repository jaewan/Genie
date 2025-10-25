"""
Week 1 Profiling: End-to-End Breakdown

This profiler instruments remote execution to answer:
"Where does the 140ms overhead actually go?"

Expected breakdown (theoretical):
├─ Capture phase:        4-5ms  (with P1.2 optimization)
├─ Scheduler:            0.3ms
├─ Network send:         8ms
├─ GPU execute:          2ms     (baseline)
├─ Network return:       8ms
└─ Materialize:          0.5ms
   TOTAL EXPECTED:      ~23ms

Reality (what we're measuring):
├─ Same components
└─ + UNKNOWN OVERHEAD:  ~120ms? ← What is this?

Goal: Find where the extra 120ms is coming from
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import json
from pathlib import Path

@dataclass
class PhaseMetrics:
    """Metrics for a single phase"""
    name: str
    times_ms: List[float]
    
    @property
    def mean(self) -> float:
        return np.mean(self.times_ms) if self.times_ms else 0
    
    @property
    def std(self) -> float:
        return np.std(self.times_ms) if self.times_ms else 0
    
    @property
    def min(self) -> float:
        return np.min(self.times_ms) if self.times_ms else 0
    
    @property
    def max(self) -> float:
        return np.max(self.times_ms) if self.times_ms else 0

class DetailedProfiler:
    """End-to-end profiler for remote operations"""
    
    def __init__(self):
        self.phases: Dict[str, PhaseMetrics] = {}
        self.current_phase = None
        self.current_start = None
        self.operation_count = 0
    
    @contextmanager
    def phase(self, name: str):
        """Context manager to time a phase"""
        if name not in self.phases:
            self.phases[name] = PhaseMetrics(name, [])
        
        # GPU sync before measuring
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        try:
            yield
        finally:
            # GPU sync after measuring
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            self.phases[name].times_ms.append(elapsed_ms)
    
    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("END-TO-END PROFILING SUMMARY")
        print("="*80)
        
        # Sort by mean time (largest first)
        sorted_phases = sorted(self.phases.values(), key=lambda p: p.mean, reverse=True)
        
        total_mean = sum(p.mean for p in sorted_phases)
        
        print(f"\nPhase Breakdown (sorted by duration):\n")
        print(f"{'Phase':<25} {'Mean (ms)':>12} {'Std (ms)':>12} {'Min (ms)':>12} {'Max (ms)':>12} {'% Total':>10}")
        print("-" * 95)
        
        for phase in sorted_phases:
            pct = (phase.mean / total_mean * 100) if total_mean > 0 else 0
            print(f"{phase.name:<25} {phase.mean:>12.2f} {phase.std:>12.2f} {phase.min:>12.2f} {phase.max:>12.2f} {pct:>9.1f}%")
        
        print("-" * 95)
        print(f"{'TOTAL':<25} {total_mean:>12.2f} {np.std([p.times_ms for phases in sorted_phases for p in [phases]]):>12.2f} {min(p.min for p in sorted_phases):>12.2f} {max(p.max for p in sorted_phases):>12.2f} {'100.0%':>10}")
        
        print(f"\n📊 Key Findings:")
        print(f"  - Total operation time: {total_mean:.2f}ms")
        print(f"  - Largest contributor: {sorted_phases[0].name} ({sorted_phases[0].mean:.2f}ms, {sorted_phases[0].mean/total_mean*100:.1f}%)")
        print(f"  - Runs measured: {len(sorted_phases[0].times_ms) if sorted_phases else 0}")
        
        # Identify bottleneck
        bottleneck_phase = sorted_phases[0]
        print(f"\n🔍 BOTTLENECK ANALYSIS:")
        print(f"  Suspected bottleneck: {bottleneck_phase.name}")
        print(f"  Time: {bottleneck_phase.mean:.2f}ms ({bottleneck_phase.mean/total_mean*100:.1f}% of total)")
        
        if bottleneck_phase.name == "capture":
            print(f"  Recommendation: Optimize LazyTensor capture (investigate graph construction)")
        elif bottleneck_phase.name in ["network_send", "network_return"]:
            print(f"  Recommendation: Optimize network (check TCP connection pooling)")
        elif bottleneck_phase.name == "gpu_execute":
            print(f"  Recommendation: This is baseline - network overhead is the issue")
        
        print("\n" + "="*80)
    
    def save_json(self, output_file: str):
        """Save results to JSON"""
        results = {
            phase.name: {
                'mean_ms': phase.mean,
                'std_ms': phase.std,
                'min_ms': phase.min,
                'max_ms': phase.max,
                'samples': len(phase.times_ms)
            }
            for phase in self.phases.values()
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to {output_file}")


def profile_remote_operation(operation_name: str, operation_fn, num_runs: int = 30):
    """
    Profile a remote operation end-to-end
    
    Args:
        operation_name: Name of operation (e.g., "remote_relu_1k")
        operation_fn: Function that performs the operation with profiler context
        num_runs: Number of iterations to run
    """
    print(f"\n{'='*80}")
    print(f"🔬 Profiling: {operation_name}")
    print(f"{'='*80}")
    print(f"Runs: {num_runs}")
    print(f"Configuration: GPU sync enabled, high-precision timing")
    
    profiler = DetailedProfiler()
    
    # Run measurements
    for run in range(num_runs):
        if (run + 1) % 10 == 0:
            print(f"  Run {run + 1}/{num_runs}...")
        
        # Clear GPU cache before each run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        operation_fn(profiler)
    
    # Print results
    profiler.print_summary()
    
    return profiler


# Example profiling for different operation sizes
def create_simple_operations():
    """Create operations to profile"""
    
    def profile_small_relu(profiler: DetailedProfiler):
        """Small operation: Element-wise ReLU on 1K×1K tensor"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        with profiler.phase("capture"):
            # Simulate LazyTensor capture
            x = torch.randn(1000, 1000, device=device)
        
        with profiler.phase("scheduler"):
            # Simulate scheduler decision
            time.sleep(0.0003)  # ~0.3ms
        
        with profiler.phase("network_send"):
            # Simulate network transfer (1000×1000 × 4 bytes = 4MB)
            time.sleep(0.008)  # ~8ms for 4MB over simulated network
        
        with profiler.phase("gpu_execute"):
            # Actual GPU work
            y = torch.relu(x)
        
        with profiler.phase("network_return"):
            # Simulate return transfer
            time.sleep(0.008)  # ~8ms
        
        with profiler.phase("materialize"):
            # Simulate materialization
            result = y.cpu()
    
    def profile_medium_matmul(profiler: DetailedProfiler):
        """Medium operation: 2K×2K matrix multiplication"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        with profiler.phase("capture"):
            a = torch.randn(2000, 2000, device=device)
            b = torch.randn(2000, 2000, device=device)
        
        with profiler.phase("scheduler"):
            time.sleep(0.0005)  # ~0.5ms
        
        with profiler.phase("network_send"):
            time.sleep(0.016)  # ~16ms for larger tensors
        
        with profiler.phase("gpu_execute"):
            c = torch.matmul(a, b)
        
        with profiler.phase("network_return"):
            time.sleep(0.016)
        
        with profiler.phase("materialize"):
            result = c.cpu()
    
    return {
        'small_relu': profile_small_relu,
        'medium_matmul': profile_medium_matmul,
    }


if __name__ == "__main__":
    print("\n🔬 WEEK 1: END-TO-END PROFILING")
    print("="*80)
    print("Goal: Find where the 140ms overhead actually goes")
    print("Hypothesis: Likely TCP connection setup, result routing, or protocol overhead")
    print("="*80)
    
    operations = create_simple_operations()
    
    # Profile each operation
    results = {}
    for op_name, op_fn in operations.items():
        profiler = profile_remote_operation(
            operation_name=op_name,
            operation_fn=op_fn,
            num_runs=30
        )
        results[op_name] = profiler
    
    # Save all results
    output_dir = Path("/home/jae/Genie/profiling_results_week1")
    output_dir.mkdir(exist_ok=True)
    
    for op_name, profiler in results.items():
        output_file = output_dir / f"{op_name}_profile.json"
        profiler.save_json(str(output_file))
    
    print(f"\n✅ Week 1 Profiling Complete")
    print(f"📁 Results saved to: {output_dir}")
    print(f"\n📊 Next Step: Analyze results to identify actual bottleneck")
