"""
Comprehensive profiling framework for Genie operations.

Tracks:
- End-to-end latency breakdown
- Per-component timing (serialize, transfer, execute, deserialize)
- Resource utilization (GPU %, network bandwidth, CPU %)
- Queue depths and contention
- Memory usage patterns
"""

import time
import threading
import logging
from contextlib import contextmanager
from typing import Dict, List, Any, Optional
import torch
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_util: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0


@dataclass
class OperationTiming:
    """Detailed timing breakdown for a single operation."""
    operation: str
    total_latency: float
    serialize_time: float = 0.0
    network_send_time: float = 0.0
    wait_result_time: float = 0.0
    deserialize_time: float = 0.0
    queue_wait_time: float = 0.0
    scheduler_time: float = 0.0
    metadata: Dict[str, Any] = None


class GPUMonitor:
    """Monitors GPU utilization during operation."""

    def __init__(self, sampling_interval: float = 0.01):
        self.sampling_interval = sampling_interval
        self.samples = []
        self.memory_samples = []
        self.running = False
        self.thread = None
        self.peak_memory = 0

    def start(self):
        """Start monitoring."""
        self.running = True
        self.samples = []
        self.memory_samples = []
        self.peak_memory = 0
        self.thread = threading.Thread(target=self._sample_loop)
        self.thread.start()

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join()

    def _sample_loop(self):
        """Background sampling loop."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            while self.running:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    self.samples.append(util.gpu)
                    self.memory_samples.append(mem_info.used / mem_info.total * 100)
                    self.peak_memory = max(self.peak_memory, mem_info.used)

                    time.sleep(self.sampling_interval)
                except Exception as e:
                    logger.debug(f"GPU sampling error: {e}")
                    break

            pynvml.nvmlShutdown()

        except ImportError:
            logger.warning("pynvml not available - GPU monitoring disabled")
        except Exception as e:
            logger.warning(f"GPU monitoring failed: {e}")

    def get_samples(self) -> List[float]:
        """Get collected utilization samples."""
        return self.samples

    def get_memory_samples(self) -> List[float]:
        """Get collected memory samples."""
        return self.memory_samples

    def get_peak_memory(self) -> int:
        """Get peak memory usage in bytes."""
        return self.peak_memory

    def get_average_utilization(self) -> float:
        """Get average GPU utilization."""
        if not self.samples:
            return 0.0
        return np.mean(self.samples)


class NetworkMonitor:
    """Monitors network activity during operation."""

    def __init__(self):
        self.start_bytes_sent = 0
        self.start_bytes_recv = 0
        self.end_bytes_sent = 0
        self.end_bytes_recv = 0

    def start_monitoring(self):
        """Start network monitoring."""
        try:
            import psutil
            net_io = psutil.net_io_counters()
            self.start_bytes_sent = net_io.bytes_sent
            self.start_bytes_recv = net_io.bytes_recv
        except ImportError:
            logger.warning("psutil not available - network monitoring disabled")
        except Exception as e:
            logger.warning(f"Network monitoring failed: {e}")

    def stop_monitoring(self) -> Dict[str, int]:
        """Stop monitoring and return network usage."""
        try:
            import psutil
            net_io = psutil.net_io_counters()
            self.end_bytes_sent = net_io.bytes_sent
            self.end_bytes_recv = net_io.bytes_recv

            return {
                'bytes_sent': self.end_bytes_sent - self.start_bytes_sent,
                'bytes_recv': self.end_bytes_recv - self.start_bytes_recv
            }
        except Exception as e:
            logger.warning(f"Network monitoring error: {e}")
            return {'bytes_sent': 0, 'bytes_recv': 0}


class GenieProfiler:
    """
    Comprehensive profiling for Genie operations.

    Tracks:
    - End-to-end latency breakdown
    - Per-component timing (serialize, transfer, execute, deserialize)
    - Resource utilization (GPU %, network bandwidth, CPU %)
    - Queue depths and contention
    """

    def __init__(self):
        self.measurements = []
        self.gpu_util_samples = []
        self.network_samples = []
        self.resource_snapshots = []

    @contextmanager
    def profile_operation(self, operation: str, metadata: Dict):
        """Profile single operation with detailed breakdown."""
        start_time = time.perf_counter()

        measurement = {
            'operation': operation,
            'metadata': metadata,
            'start_time': start_time,
            'timings': {},
            'resource_snapshots': [],
            'network_usage': {},
        }

        # Start resource monitoring
        gpu_monitor = None
        network_monitor = NetworkMonitor()
        network_monitor.start_monitoring()

        if torch.cuda.is_available():
            gpu_monitor = GPUMonitor(sampling_interval=0.01)
            gpu_monitor.start()

        try:
            # Capture pre-operation resource state
            self._capture_resource_snapshot(measurement)
            yield measurement

        finally:
            # Capture post-operation timing and resources
            measurement['end_time'] = time.perf_counter()
            measurement['total_latency'] = measurement['end_time'] - start_time

            # Stop GPU monitoring
            if gpu_monitor:
                gpu_monitor.stop()
                measurement['gpu_util'] = gpu_monitor.get_samples()
                measurement['gpu_memory_samples'] = gpu_monitor.get_memory_samples()
                measurement['gpu_memory_peak'] = gpu_monitor.get_peak_memory()
                measurement['gpu_avg_util'] = gpu_monitor.get_average_utilization()

            # Stop network monitoring
            measurement['network_usage'] = network_monitor.stop_monitoring()

            # Capture final resource snapshot
            self._capture_resource_snapshot(measurement)

            self.measurements.append(measurement)

    def _capture_resource_snapshot(self, measurement: Dict):
        """Capture current system resource usage."""
        try:
            import psutil

            snapshot = ResourceSnapshot(
                timestamp=time.perf_counter(),
                cpu_percent=psutil.cpu_percent(),
                memory_percent=psutil.virtual_memory().percent
            )

            if torch.cuda.is_available():
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    snapshot.gpu_util = util.gpu
                    snapshot.gpu_memory_percent = (mem_info.used / mem_info.total) * 100

                    pynvml.nvmlShutdown()
                except Exception as e:
                    logger.debug(f"GPU snapshot failed: {e}")

            measurement['resource_snapshots'].append(snapshot)

        except ImportError:
            logger.warning("psutil not available - resource monitoring disabled")
        except Exception as e:
            logger.debug(f"Resource snapshot failed: {e}")

    def generate_report(self) -> str:
        """Generate human-readable profiling report."""
        if not self.measurements:
            return "No measurements collected"

        report = []
        report.append("="*80)
        report.append("Genie Profiling Report")
        report.append("="*80)
        report.append(f"Total operations: {len(self.measurements)}")
        report.append("")

        # Latency breakdown
        report.append("Latency Breakdown (ms):")
        report.append("-"*80)

        # Group by operation type
        by_operation = {}
        for m in self.measurements:
            op = m['operation']
            if op not in by_operation:
                by_operation[op] = []
            by_operation[op].append(m)

        for op, measurements in by_operation.items():
            latencies = [m['total_latency'] * 1000 for m in measurements]
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)

            report.append(f"{op:30s}: p50={p50:6.2f}ms  p95={p95:6.2f}ms  p99={p99:6.2f}ms  (n={len(latencies)})")

        report.append("")

        # Component breakdown (if available)
        if any('timings' in m and m['timings'] for m in self.measurements):
            report.append("Component Breakdown (average, ms):")
            report.append("-"*80)

            # Aggregate timing components
            component_times = {}
            for m in self.measurements:
                for component, duration in m.get('timings', {}).items():
                    if component not in component_times:
                        component_times[component] = []
                    component_times[component].append(duration * 1000)

            for component, times in sorted(component_times.items()):
                avg = np.mean(times)
                p95 = np.percentile(times, 95)
                report.append(f"  {component:25s}: {avg:6.2f}ms (p95: {p95:6.2f}ms)")

        report.append("")

        # Resource utilization
        gpu_utils = []
        memory_utils = []
        network_sent = []
        network_recv = []

        for m in self.measurements:
            # GPU utilization
            if 'gpu_util' in m and m['gpu_util']:
                gpu_utils.extend(m['gpu_util'])

            # Network usage
            net_usage = m.get('network_usage', {})
            if net_usage.get('bytes_sent', 0) > 0:
                network_sent.append(net_usage['bytes_sent'])
            if net_usage.get('bytes_recv', 0) > 0:
                network_recv.append(net_usage['bytes_recv'])

        if gpu_utils:
            avg_gpu = np.mean(gpu_utils)
            report.append(f"GPU Utilization: {avg_gpu:.1f}% average")

        if network_sent:
            total_sent = sum(network_sent)
            avg_sent = np.mean(network_sent)
            report.append(f"Network Sent: {total_sent / 1024 / 1024:.1f}MB total, {avg_sent / 1024:.1f}KB avg per op")

        if network_recv:
            total_recv = sum(network_recv)
            avg_recv = np.mean(network_recv)
            report.append(f"Network Recv: {total_recv / 1024 / 1024:.1f}MB total, {avg_recv / 1024:.1f}KB avg per op")

        # Connection pool statistics (if available)
        if hasattr(self, '_connection_pool_stats'):
            report.append("")
            report.append("Connection Pool Performance:")
            report.append("-"*80)
            stats = self._connection_pool_stats
            report.append(f"  Hit rate: {stats['hit_rate']:.1%}")
            report.append(f"  Created: {stats['created']}, Reused: {stats['reused']}")
            report.append(f"  Errors: {stats['errors']}")

        report.append("="*80)

        return "\n".join(report)

    def save_report(self, filename: str):
        """Save detailed profiling data to file."""
        import json

        # Prepare serializable data
        serializable_measurements = []
        for m in self.measurements:
            serializable = {
                'operation': m['operation'],
                'metadata': m['metadata'],
                'total_latency': m['total_latency'],
                'timings': m.get('timings', {}),
                'network_usage': m.get('network_usage', {}),
                'gpu_avg_util': m.get('gpu_avg_util', 0),
                'gpu_memory_peak': m.get('gpu_memory_peak', 0),
            }

            # Convert numpy types to native Python types
            for key, value in serializable.items():
                if isinstance(value, np.integer):
                    serializable[key] = int(value)
                elif isinstance(value, np.floating):
                    serializable[key] = float(value)

            serializable_measurements.append(serializable)

        data = {
            'measurements': serializable_measurements,
            'summary': {
                'total_operations': len(self.measurements),
                'timestamp': time.time(),
                'profiling_version': '1.0'
            }
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Profiling data saved to {filename}")

    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """Analyze bottlenecks from collected measurements."""
        if not self.measurements:
            return {'error': 'No measurements available'}

        # Calculate component contributions to total latency
        total_time = sum(m['total_latency'] for m in self.measurements)

        component_times = {}
        for m in self.measurements:
            for component, duration in m.get('timings', {}).items():
                component_times[component] = component_times.get(component, 0) + duration

        # Calculate percentages
        bottlenecks = {}
        for component, time_spent in component_times.items():
            percentage = (time_spent / total_time) * 100
            bottlenecks[component] = {
                'time_ms': time_spent * 1000,
                'percentage': percentage,
                'is_bottleneck': percentage > 20  # 20% threshold for bottleneck
            }

        # Network analysis
        network_analysis = {'total_bytes': 0, 'avg_per_op': 0}
        network_total = 0
        for m in self.measurements:
            net_usage = m.get('network_usage', {})
            network_total += net_usage.get('bytes_sent', 0) + net_usage.get('bytes_recv', 0)

        if self.measurements:
            network_analysis['total_bytes'] = network_total
            network_analysis['avg_per_op'] = network_total / len(self.measurements)
            network_analysis['throughput_mbps'] = (network_total / total_time) / 1024 / 1024

        return {
            'component_bottlenecks': bottlenecks,
            'network_analysis': network_analysis,
            'recommendations': self._generate_optimization_recommendations(bottlenecks, network_analysis)
        }

    def _generate_optimization_recommendations(self, bottlenecks: Dict, network: Dict) -> List[str]:
        """Generate optimization recommendations based on profiling data."""
        recommendations = []

        # Component-based recommendations
        for component, data in bottlenecks.items():
            if data['is_bottleneck']:
                if component == 'network_send':
                    recommendations.append("Network transfer is bottleneck - consider DPDK zero-copy implementation")
                elif component == 'serialize':
                    recommendations.append("Serialization is bottleneck - investigate zero-copy tensor transfer")
                elif component == 'wait_result':
                    recommendations.append("Result waiting is bottleneck - check server-side execution efficiency")
                elif component == 'scheduler_time':
                    recommendations.append("Scheduler overhead is significant - optimize placement decisions")

        # Network-based recommendations
        if network['throughput_mbps'] < 1000:  # Less than 1 Gbps
            recommendations.append("Network bandwidth is low - consider faster interconnect or compression")

        return recommendations

    def reset(self):
        """Reset all measurements."""
        self.measurements.clear()
        self.gpu_util_samples.clear()
        self.network_samples.clear()
        self.resource_snapshots.clear()
        logger.info("Profiling measurements reset")


class DetailedComponentProfiler:
    """
    P0 FIX: Fine-grained profiling for each component.
    
    Tracks where the 140ms overhead goes:
    - Graph construction
    - Metadata annotation
    - Pattern matching
    - Serialization
    - Network transfer
    - Remote execution
    - Deserialization
    """
    
    def __init__(self):
        self.component_timings: Dict[str, List[float]] = defaultdict(list)
        self.active_components: Dict[threading.Thread, List[str]] = defaultdict(list)
        self.lock = threading.Lock()
    
    @contextmanager
    def profile_component(self, component_name: str):
        """
        Profile a single component.
        
        Usage:
            profiler = DetailedComponentProfiler()
            with profiler.profile_component("graph_construction"):
                build_graph(model)
            
            stats = profiler.get_component_stats("graph_construction")
            # stats = {"mean": 45.2, "std": 3.1, "min": 42, "max": 52, "count": 5}
        """
        start_time = time.perf_counter()
        thread_id = threading.current_thread()
        
        with self.lock:
            self.active_components[thread_id].append(component_name)
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1000
            
            with self.lock:
                self.component_timings[component_name].append(elapsed_ms)
                self.active_components[thread_id].pop()
    
    def get_component_stats(self, component_name: str) -> Dict[str, float]:
        """Get statistics for a component."""
        timings = self.component_timings.get(component_name, [])
        if not timings:
            return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0}
        
        return {
            "count": len(timings),
            "mean": np.mean(timings),
            "std": np.std(timings),
            "min": np.min(timings),
            "max": np.max(timings),
            "sum": np.sum(timings),
        }
    
    def print_summary(self):
        """Print summary of all component timings."""
        print("\n" + "="*80)
        print("COMPONENT PROFILING SUMMARY (Identifies the 140ms overhead bottleneck)")
        print("="*80)
        
        total_time = 0
        components_sorted = sorted(
            self.component_timings.items(),
            key=lambda x: np.sum(x[1]),
            reverse=True
        )
        
        for component_name, timings in components_sorted:
            stats = self.get_component_stats(component_name)
            total_time += stats["sum"]
            
            print(f"\n{component_name}:")
            print(f"  Count:  {stats['count']}")
            print(f"  Mean:   {stats['mean']:.2f}ms")
            print(f"  Std:    {stats['std']:.2f}ms")
            print(f"  Range:  {stats['min']:.2f}ms - {stats['max']:.2f}ms")
            print(f"  Total:  {stats['sum']:.2f}ms")
        
        print(f"\nTotal overhead across all components: {total_time:.2f}ms")
        print("="*80 + "\n")
    
    def clear(self):
        """Clear all recorded timings."""
        with self.lock:
            self.component_timings.clear()
            self.active_components.clear()


# Global instance for easy access
_global_profiler = DetailedComponentProfiler()


def get_detailed_profiler() -> DetailedComponentProfiler:
    """Get the global detailed profiler instance."""
    return _global_profiler
