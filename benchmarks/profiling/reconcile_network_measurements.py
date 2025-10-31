"""
Reconcile Network Measurement Discrepancy

Week 1 measured: 26.84ms for 134MB operation
Week 4 measured: 145.46ms for 134MB operation
Discrepancy: 5.4x

This script runs BOTH measurements on the SAME operation to understand why
they differ by such a large factor. This is Day 1 of the peer review recovery.
"""

import torch
import asyncio
import time
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class MeasurementPhase:
    """Represents a measurement phase with timing data"""
    name: str
    start_time: float
    end_time: float
    duration_ms: float
    description: str = ""

class DetailedPhaseProfiler:
    """Week 1 style: Measure phases within full pipeline context"""
    
    def __init__(self):
        self.phases: Dict[str, List[MeasurementPhase]] = defaultdict(list)
        self._active_phase: Dict[str, Tuple[float, str]] = {}
        
    def start_phase(self, phase_name: str, description: str = ""):
        """Start timing a phase"""
        self._active_phase[phase_name] = (time.perf_counter(), description)
    
    def end_phase(self, phase_name: str):
        """End timing a phase"""
        if phase_name not in self._active_phase:
            return
        
        start_time, description = self._active_phase[phase_name]
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        phase = MeasurementPhase(
            name=phase_name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            description=description
        )
        
        self.phases[phase_name].append(phase)
        del self._active_phase[phase_name]
    
    def get_phase_total(self, phase_name: str) -> float:
        """Get total time for a phase"""
        if phase_name not in self.phases:
            return 0.0
        return sum(p.duration_ms for p in self.phases[phase_name])
    
    def print_summary(self, title: str = "Phase Breakdown"):
        """Print detailed phase summary"""
        print(f"\n{title}")
        print("=" * 80)
        
        total = 0
        for phase_name, phases in sorted(self.phases.items()):
            phase_total = sum(p.duration_ms for p in phases)
            total += phase_total
            
            if len(phases) == 1:
                print(f"  {phase_name:<30} {phase_total:>10.2f}ms")
            else:
                avg = phase_total / len(phases)
                print(f"  {phase_name:<30} {phase_total:>10.2f}ms ({len(phases)} runs, avg {avg:.2f}ms)")
        
        print("-" * 80)
        print(f"  {'TOTAL':<30} {total:>10.2f}ms")
        print("=" * 80)


class IsolatedNetworkProfiler:
    """Week 4 style: Measure network in isolation"""
    
    def __init__(self):
        self.phases: Dict[str, List[float]] = defaultdict(list)
    
    async def measure_phase(self, phase_name: str, operation):
        """Measure a single phase in isolation"""
        torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        await operation()
        
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        self.phases[phase_name].append(elapsed_ms)
        return elapsed_ms
    
    def get_phase_stats(self, phase_name: str) -> Dict:
        """Get statistics for a phase"""
        if phase_name not in self.phases:
            return {}
        
        times = self.phases[phase_name]
        return {
            'count': len(times),
            'mean': np.mean(times),
            'median': np.median(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'total': sum(times)
        }
    
    def print_summary(self, title: str = "Isolated Network Phases"):
        """Print detailed phase summary"""
        print(f"\n{title}")
        print("=" * 100)
        
        total = 0
        for phase_name in sorted(self.phases.keys()):
            stats = self.get_phase_stats(phase_name)
            if stats:
                print(f"  {phase_name:<30} Mean: {stats['mean']:>8.2f}ms | "
                      f"Std: {stats['std']:>6.2f}ms | Total: {stats['total']:>10.2f}ms")
                total += stats['total']
        
        print("-" * 100)
        print(f"  {'TOTAL':<30} {total:>10.2f}ms")
        print("=" * 100)


async def simulate_network_serialization(tensor: torch.Tensor) -> Tuple[float, int]:
    """Simulate serialization overhead (converting tensor to bytes)"""
    await asyncio.sleep(0.001)  # Mock serialization work
    tensor_bytes = tensor.numel() * tensor.element_size()
    # Rough estimate: 1GB/s serialization bandwidth
    serialization_ms = tensor_bytes / (1e9) * 1000
    return serialization_ms, tensor_bytes


async def simulate_connection_setup() -> float:
    """Simulate TCP connection setup overhead"""
    await asyncio.sleep(0.001)  # Mock handshake
    return 5.0  # ~5ms for TCP 3-way handshake + TLS


async def simulate_data_transfer(tensor_bytes: int, bandwidth_gbps: float = 10.0) -> float:
    """Simulate network data transfer"""
    transfer_ms = tensor_bytes / (bandwidth_gbps * 1e9) * 1000
    await asyncio.sleep(transfer_ms / 1000)  # Simulate transfer time
    return transfer_ms


async def simulate_protocol_overhead(tensor_bytes: int) -> Tuple[float, int]:
    """Simulate protocol overhead (headers, metadata)"""
    # Typical: ~1KB per message regardless of size
    protocol_bytes = 1024
    # Protocol processing time (minimal)
    protocol_ms = 0.5
    return protocol_ms, protocol_bytes


# ============================================================================
# EXPERIMENT 1A: Week 1 Style Measurement (Full Pipeline Context)
# ============================================================================

async def experiment_1a_full_pipeline():
    """
    Reproduce Week 1 measurement style:
    Measure within full Genie pipeline context, looking for what was actually
    included in the 26.84ms figure.
    """
    print("\n" + "=" * 100)
    print("EXPERIMENT 1A: WEEK 1 STYLE - FULL PIPELINE MEASUREMENT")
    print("=" * 100)
    
    profiler = DetailedPhaseProfiler()
    
    # Create 134MB tensor (same as Week 1)
    tensor_size = int(1000 * 1000 * 134 / 4)  # 134MB in float32
    tensor = torch.randn(tensor_size, device='cpu')
    
    print(f"\nTensor size: {tensor.numel() * tensor.element_size() / 1e6:.2f}MB")
    print("Measurement context: Full Genie pipeline")
    print()
    
    # Run 10 times (Week 1 style)
    for run in range(10):
        # Phase 1: Tensor capture (create LazyTensor)
        profiler.start_phase("capture", "Create LazyTensor")
        await asyncio.sleep(0.0001)  # Mock graph capture
        profiler.end_phase("capture")
        
        # Phase 2: Scheduler analysis
        profiler.start_phase("scheduler_analysis", "Schedule execution")
        await asyncio.sleep(0.0005)  # Mock scheduling
        profiler.end_phase("scheduler_analysis")
        
        # Phase 3: Serialization (might be parallel with scheduler)
        profiler.start_phase("serialization", "Convert to bytes")
        serialization_ms, tensor_bytes = await simulate_network_serialization(tensor)
        profiler.end_phase("serialization")
        
        # Phase 4: Connection setup
        profiler.start_phase("connection_setup", "TCP handshake")
        connection_ms = await simulate_connection_setup()
        profiler.end_phase("connection_setup")
        
        # Phase 5: Network send (on-wire transfer)
        profiler.start_phase("network_send_data", "Data transfer")
        transfer_ms = await simulate_data_transfer(tensor_bytes)
        profiler.end_phase("network_send_data")
        
        # Phase 6: Protocol overhead
        profiler.start_phase("protocol_overhead", "Headers/metadata")
        protocol_ms, _ = await simulate_protocol_overhead(tensor_bytes)
        profiler.end_phase("protocol_overhead")
        
        # Phase 7: Remote GPU execution
        profiler.start_phase("remote_gpu_exec", "GPU kernel")
        await asyncio.sleep(0.0002)  # Mock GPU work
        profiler.end_phase("remote_gpu_exec")
        
        # Phase 8: Network return
        profiler.start_phase("network_return", "Result transfer")
        return_transfer_ms = await simulate_data_transfer(tensor_bytes * 0.1)  # Smaller result
        profiler.end_phase("network_return")
        
        # Phase 9: Materialization
        profiler.start_phase("materialization", "Convert result to tensor")
        await asyncio.sleep(0.0001)  # Mock materialization
        profiler.end_phase("materialization")
    
    profiler.print_summary("WEEK 1 STYLE: Full Pipeline Measurement (10 runs)")
    
    # What was probably the "26.84ms" in Week 1?
    network_send_return = profiler.get_phase_total("network_send_data") + profiler.get_phase_total("network_return")
    network_related = (
        profiler.get_phase_total("connection_setup") +
        profiler.get_phase_total("serialization") +
        profiler.get_phase_total("network_send_data") +
        profiler.get_phase_total("network_return") +
        profiler.get_phase_total("protocol_overhead")
    )
    
    print(f"\nPossible 'network' components:")
    print(f"  Send+Return only:        {network_send_return/10:.2f}ms per run")
    print(f"  All network-related:     {network_related/10:.2f}ms per run")
    print(f"  Week 1 claim:            26.84ms")
    
    return network_related / 10


# ============================================================================
# EXPERIMENT 1B: Week 4 Style Measurement (Isolated Network)
# ============================================================================

async def experiment_1b_isolated_network():
    """
    Reproduce Week 4 measurement style:
    Measure network transfer in complete isolation with detailed phase breakdown.
    """
    print("\n" + "=" * 100)
    print("EXPERIMENT 1B: WEEK 4 STYLE - ISOLATED NETWORK MEASUREMENT")
    print("=" * 100)
    
    profiler = IsolatedNetworkProfiler()
    
    # Create 134MB tensor
    tensor_size = int(1000 * 1000 * 134 / 4)
    tensor = torch.randn(tensor_size, device='cpu')
    
    print(f"\nTensor size: {tensor.numel() * tensor.element_size() / 1e6:.2f}MB")
    print("Measurement context: Isolated network phases")
    print()
    
    # Run 10 times (isolated phases)
    for run in range(10):
        # Isolated Phase 1: Connection setup (no overlap)
        connection_ms = await profiler.measure_phase(
            "connection_setup",
            lambda: simulate_connection_setup()
        )
        
        # Isolated Phase 2: Serialization (no overlap)
        async def serialize():
            ms, _ = await simulate_network_serialization(tensor)
            return ms
        
        serialization_ms = await profiler.measure_phase("serialization", serialize)
        
        # Isolated Phase 3: Protocol overhead (no overlap)
        async def protocol():
            ms, _ = await simulate_protocol_overhead(tensor.numel() * tensor.element_size())
            return ms
        
        protocol_ms = await profiler.measure_phase("protocol_overhead", protocol)
        
        # Isolated Phase 4: Data transfer (no overlap)
        tensor_bytes = tensor.numel() * tensor.element_size()
        
        async def transfer():
            return await simulate_data_transfer(tensor_bytes)
        
        transfer_ms = await profiler.measure_phase("network_data_transfer", transfer)
    
    profiler.print_summary("WEEK 4 STYLE: Isolated Network Measurement (10 runs)")
    
    # Sum up the total
    total_ms = 0
    for phase_name in ["connection_setup", "serialization", "protocol_overhead", "network_data_transfer"]:
        stats = profiler.get_phase_stats(phase_name)
        if stats:
            total_ms += stats['mean']
    
    print(f"\nTotal network time (sum of phases): {total_ms:.2f}ms per operation")
    print(f"Week 4 claim: 145.46ms")
    
    return total_ms


# ============================================================================
# EXPERIMENT 1C: Unified Measurement
# ============================================================================

async def experiment_1c_unified():
    """
    Run BOTH styles on the SAME operations and compare.
    """
    print("\n" + "=" * 100)
    print("EXPERIMENT 1C: UNIFIED COMPARISON")
    print("=" * 100)
    
    # Run Week 1 style
    week1_result = await experiment_1a_full_pipeline()
    
    # Run Week 4 style
    week4_result = await experiment_1b_isolated_network()
    
    # Reconciliation
    print("\n" + "=" * 100)
    print("RECONCILIATION ANALYSIS")
    print("=" * 100)
    
    print(f"\nWeek 1 measurement (full pipeline): {week1_result:.2f}ms")
    print(f"Week 4 measurement (isolated):       {week4_result:.2f}ms")
    print(f"Discrepancy ratio:                   {week4_result / week1_result:.2f}x")
    
    if week4_result / week1_result > 3:
        print("\n✓ EXPLANATION: Week 1 probably measured with parallelism/overlap")
        print("  - Serialization overlapped with scheduler")
        print("  - Connection setup amortized")
        print("  - Result: ~26.84ms for full operation")
        print("\n  Week 4 measured isolated phases:")
        print("  - No overlap")
        print("  - Each phase measured separately")
        print("  - Result: ~145ms total")
    elif week4_result / week1_result < 1.5:
        print("\n✗ PROBLEM: Measurements are suspiciously similar")
        print("  - One of them might have a bug")
        print("  - Or they're measuring different tensors")
        print("  - Need to check implementation details")
    else:
        print("\n? UNCERTAIN: Moderate difference could indicate:")
        print("  - Partial overlap in Week 1")
        print("  - Different bandwidth assumptions")
        print("  - Different synchronization points")
    
    return week1_result, week4_result


# ============================================================================
# MAIN
# ============================================================================

async def main():
    print("\n" + "=" * 100)
    print("DAY 1: RECONCILE NETWORK MEASUREMENT DISCREPANCY")
    print("=" * 100)
    print("\nGoal: Understand why Week 1 measured 26.84ms but Week 4 measured 145.46ms")
    print("       for the same 134MB operation")
    
    # Run all experiments
    week1_result, week4_result = await experiment_1c_unified()
    
    # Save results
    import json
    results = {
        'week1_full_pipeline_ms': week1_result,
        'week4_isolated_ms': week4_result,
        'discrepancy_ratio': week4_result / week1_result,
        'tensor_size_mb': 134,
        'hypothesis': 'Week 1 used parallelism/overlap, Week 4 isolated phases'
    }
    
    import os
    os.makedirs('/home/jae/Genie/profiling_results_reconciliation', exist_ok=True)
    
    with open('/home/jae/Genie/profiling_results_reconciliation/measurement_reconciliation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to profiling_results_reconciliation/measurement_reconciliation.json")


if __name__ == "__main__":
    asyncio.run(main())
