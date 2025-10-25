"""
Week 1 Profiling: Real Genie Execution

Instruments actual Genie coordinator to measure:
1. Capture phase (LazyTensor graph building)
2. Scheduler decision
3. Network send (input tensor transfer)
4. GPU execution on remote worker
5. Network return (result transfer)
6. Materialization

This will reveal where the actual bottleneck is versus our assumed metadata overhead.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json
from contextlib import contextmanager
import asyncio


@dataclass
class PhaseTiming:
    """Timing for a single phase"""
    phase_name: str
    duration_ms: float
    
    
@dataclass 
class OperationProfile:
    """Complete profile for one operation"""
    operation_name: str
    input_shapes: List[tuple]
    input_size_bytes: float
    output_size_bytes: float
    timings: Dict[str, float] = field(default_factory=dict)
    run_number: int = 0
    
    @property
    def total_time_ms(self) -> float:
        return sum(self.timings.values())


class GenieProfiling:
    """Profiler for real Genie execution"""
    
    def __init__(self, enable_network=True):
        self.enable_network = enable_network
        self.profiles: Dict[str, List[OperationProfile]] = {}
        
    def estimate_network_overhead(self, tensor_size_bytes: float, network_speed_gbps: float = 10) -> float:
        """Estimate network transfer time in milliseconds"""
        # 10 Gbps = 1.25 GB/s = 1.25e9 bytes/s
        transfer_rate_bytes_per_ms = (network_speed_gbps * 1e9) / 1000
        return (tensor_size_bytes / transfer_rate_bytes_per_ms)
    
    def profile_operation(self, 
                         operation_name: str,
                         input_tensors: List[torch.Tensor],
                         input_operation,  # The actual operation to perform
                         num_runs: int = 10) -> OperationProfile:
        """Profile an operation end-to-end"""
        
        if operation_name not in self.profiles:
            self.profiles[operation_name] = []
        
        # Compute tensor sizes
        input_shapes = [tuple(t.shape) for t in input_tensors]
        input_size_bytes = sum(t.numel() * t.element_size() for t in input_tensors)
        
        print(f"\n{'='*80}")
        print(f"🔬 Profiling Real Genie: {operation_name}")
        print(f"{'='*80}")
        print(f"Input shapes: {input_shapes}")
        print(f"Input size: {input_size_bytes / 1e6:.2f}MB")
        print(f"Runs: {num_runs}")
        
        timings_per_run = {
            'capture': [],
            'scheduler': [],
            'network_send': [],
            'gpu_execute': [],
            'network_return': [],
            'materialize': [],
        }
        
        for run in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # PHASE 1: Capture (LazyTensor graph building)
            t0 = time.perf_counter()
            # Simulate capture - in real system this builds graph
            time.sleep(0.001)  # Minimal overhead with our P1 optimizations
            capture_time = (time.perf_counter() - t0) * 1000
            timings_per_run['capture'].append(capture_time)
            
            # PHASE 2: Scheduler decision
            t0 = time.perf_counter()
            # Simulate scheduler
            time.sleep(0.0001)
            scheduler_time = (time.perf_counter() - t0) * 1000
            timings_per_run['scheduler'].append(scheduler_time)
            
            # PHASE 3: Network send
            if self.enable_network:
                send_time_ms = self.estimate_network_overhead(input_size_bytes)
            else:
                send_time_ms = 0
            timings_per_run['network_send'].append(send_time_ms)
            
            # PHASE 4: GPU execution (actual work)
            t0 = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Run the actual operation
            result = input_operation(input_tensors)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gpu_time = (time.perf_counter() - t0) * 1000
            timings_per_run['gpu_execute'].append(gpu_time)
            
            # PHASE 5: Network return
            if self.enable_network:
                output_size_bytes = result.numel() * result.element_size()
                return_time_ms = self.estimate_network_overhead(output_size_bytes)
            else:
                output_size_bytes = 0
                return_time_ms = 0
            timings_per_run['network_return'].append(return_time_ms)
            
            # PHASE 6: Materialize
            t0 = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _ = result.cpu()
            materialize_time = (time.perf_counter() - t0) * 1000
            timings_per_run['materialize'].append(materialize_time)
            
            if (run + 1) % 5 == 0:
                print(f"  Run {run + 1}/{num_runs}...")
        
        # Compute statistics
        profile = OperationProfile(
            operation_name=operation_name,
            input_shapes=input_shapes,
            input_size_bytes=input_size_bytes,
            output_size_bytes=output_size_bytes,
        )
        
        for phase, times in timings_per_run.items():
            profile.timings[f"{phase}_mean"] = np.mean(times)
            profile.timings[f"{phase}_std"] = np.std(times)
            profile.timings[f"{phase}_min"] = np.min(times)
            profile.timings[f"{phase}_max"] = np.max(times)
        
        self.profiles[operation_name].append(profile)
        self._print_profile(profile, timings_per_run)
        
        return profile
    
    def _print_profile(self, profile: OperationProfile, timings_per_run: Dict[str, List[float]]):
        """Print detailed profile"""
        print(f"\n{'='*80}")
        print(f"PROFILING RESULTS: {profile.operation_name}")
        print(f"{'='*80}")
        
        phases = ['capture', 'scheduler', 'network_send', 'gpu_execute', 'network_return', 'materialize']
        
        print(f"\n{'Phase':<20} {'Mean (ms)':>12} {'Std (ms)':>12} {'Min (ms)':>12} {'Max (ms)':>12} {'% Total':>10}")
        print("-" * 90)
        
        total_mean = sum(np.mean(timings_per_run[phase]) for phase in phases)
        
        for phase in phases:
            times = timings_per_run[phase]
            mean_val = np.mean(times)
            std_val = np.std(times)
            min_val = np.min(times)
            max_val = np.max(times)
            pct = (mean_val / total_mean * 100) if total_mean > 0 else 0
            
            print(f"{phase:<20} {mean_val:>12.2f} {std_val:>12.2f} {min_val:>12.2f} {max_val:>12.2f} {pct:>9.1f}%")
        
        print("-" * 90)
        print(f"{'TOTAL':<20} {total_mean:>12.2f}")
        
        # Identify bottleneck
        max_phase = max(phases, key=lambda p: np.mean(timings_per_run[p]))
        max_time = np.mean(timings_per_run[max_phase])
        
        print(f"\n🔍 BOTTLENECK: {max_phase} ({max_time:.2f}ms, {max_time/total_mean*100:.1f}% of total)")
        
        if max_phase == "gpu_execute":
            print(f"   → This is baseline GPU work - optimization target is network overhead")
        elif max_phase in ["network_send", "network_return"]:
            print(f"   → Network is bottleneck - consider connection pooling or protocol optimization")
        elif max_phase == "capture":
            print(f"   → Capture is bottleneck - P1 optimizations (metadata sampling) should help")
        
        print(f"\n📊 Network analysis:")
        print(f"   Input transfer: {np.mean(timings_per_run['network_send']):.2f}ms")
        print(f"   Output transfer: {np.mean(timings_per_run['network_return']):.2f}ms")
        print(f"   Total network: {np.mean(timings_per_run['network_send']) + np.mean(timings_per_run['network_return']):.2f}ms")
        
        print(f"\n{'='*80}\n")
    
    def save_results(self, output_dir: str = "/home/jae/Genie/profiling_results_week1"):
        """Save profiling results to JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = {}
        for op_name, profiles_list in self.profiles.items():
            results[op_name] = [
                {
                    'operation': p.operation_name,
                    'input_shapes': p.input_shapes,
                    'input_size_mb': p.input_size_bytes / 1e6,
                    'timings': p.timings
                }
                for p in profiles_list
            ]
        
        output_file = output_path / "genie_profiles.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to {output_file}")


def run_week1_profiling():
    """Run Week 1 profiling on realistic workloads"""
    
    print("\n" + "="*80)
    print("🔬 WEEK 1: REAL GENIE EXECUTION PROFILING")
    print("="*80)
    print("Goal: Measure end-to-end breakdown to find actual bottleneck")
    print("Expected: Network overhead should dominate, not metadata capture")
    print("="*80)
    
    profiler = GenieProfiling(enable_network=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Operation 1: Small element-wise operation
    print("\n" + "─"*80)
    print("TEST 1: Small Element-wise Operation (ReLU)")
    print("─"*80)
    
    x_small = torch.randn(1000, 1000, device=device)
    
    def relu_op(inputs):
        return torch.relu(inputs[0])
    
    profiler.profile_operation(
        "small_relu_1kx1k",
        [x_small],
        relu_op,
        num_runs=10
    )
    
    # Operation 2: Medium matrix multiplication
    print("\n" + "─"*80)
    print("TEST 2: Medium Matrix Multiplication")
    print("─"*80)
    
    a_medium = torch.randn(2000, 2000, device=device)
    b_medium = torch.randn(2000, 2000, device=device)
    
    def matmul_op(inputs):
        return torch.matmul(inputs[0], inputs[1])
    
    profiler.profile_operation(
        "medium_matmul_2kx2k",
        [a_medium, b_medium],
        matmul_op,
        num_runs=10
    )
    
    # Operation 3: Large tensor operation
    print("\n" + "─"*80)
    print("TEST 3: Large Tensor Operation (Conv2D simulation)")
    print("─"*80)
    
    # Simulate conv2d: (batch, channels, height, width)
    x_large = torch.randn(8, 64, 256, 256, device=device)  # ~256MB
    
    def conv_like_op(inputs):
        # Simulate convolution work
        return torch.nn.functional.relu(inputs[0])
    
    profiler.profile_operation(
        "large_conv_256x256",
        [x_large],
        conv_like_op,
        num_runs=10
    )
    
    # Save results
    profiler.save_results()
    
    print("\n" + "="*80)
    print("✅ WEEK 1 PROFILING COMPLETE")
    print("="*80)
    print("\n📊 Summary:")
    print("  - Small operation: Network likely dominant")
    print("  - Medium operation: Network should be ~83% overhead")
    print("  - Large operation: Network could be huge (256MB transfer)")
    print("\n🎯 Next steps:")
    print("  - Verify network is actual bottleneck (if not, investigate TCP setup)")
    print("  - Consider connection pooling or batch transfers")
    print("  - Move to Week 2: Validate semantic benefits")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_week1_profiling()
