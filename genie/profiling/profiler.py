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
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

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
            logger.debug("pynvml not available - GPU monitoring disabled")
        except Exception as e:
            logger.debug(f"GPU monitoring failed: {e}")

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
            logger.debug("psutil not available - network monitoring disabled")
        except Exception as e:
            logger.debug(f"Network monitoring failed: {e}")

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
            logger.debug(f"Network monitoring error: {e}")
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
            logger.debug("psutil not available - resource monitoring disabled")
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
        report.append("="*80)

        return "\n".join(report)

    def reset(self):
        """Reset all measurements."""
        self.measurements.clear()
        self.gpu_util_samples.clear()
        self.network_samples.clear()
        self.resource_snapshots.clear()
        logger.info("Profiling measurements reset")


class DetailedComponentProfiler:
    """
    Fine-grained profiling for each component.
    
    Tracks where overhead is spent:
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
        
        # OPTIMIZATION: Thread-local storage for lock-free fast path
        import threading as tls_module
        self._thread_local = tls_module.local()
    
    @contextmanager
    def profile_component(self, component_name: str):
        """
        Profile a single component.
        
        OPTIMIZATION: Uses thread-local storage to avoid locks in the fast path.
        
        Usage:
            profiler = DetailedComponentProfiler()
            with profiler.profile_component("graph_construction"):
                build_graph(model)
            
            stats = profiler.get_component_stats("graph_construction")
        """
        start_time = time.perf_counter()
        
        # OPTIMIZATION: Try thread-local append first (fast path, no lock)
        try:
            if not hasattr(self._thread_local, 'stack'):
                self._thread_local.stack = []
            self._thread_local.stack.append(component_name)
        except:
            # Fallback to locked path if thread-local fails
            thread_id = threading.current_thread()
            with self.lock:
                self.active_components[thread_id].append(component_name)
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1000
            
            # OPTIMIZATION: Store timing directly without lock (list append is atomic in CPython)
            self.component_timings[component_name].append(elapsed_ms)
            
            # OPTIMIZATION: Pop from thread-local first (fast path)
            try:
                if hasattr(self._thread_local, 'stack'):
                    self._thread_local.stack.pop()
                else:
                    raise AttributeError  # Fall back to locked path
            except:
                # Fallback to locked path
                thread_id = threading.current_thread()
                with self.lock:
                    if thread_id in self.active_components:
                        self.active_components[thread_id].pop()
    
    def get_component_stats(self, component_name: str) -> Dict[str, float]:
        """Get statistics for a component."""
        timings = self.component_timings.get(component_name, [])
        if not timings:
            return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0, "sum": 0}
        
        return {
            "count": len(timings),
            "mean": np.mean(timings),
            "std": np.std(timings),
            "min": np.min(timings),
            "max": np.max(timings),
            "sum": np.sum(timings),
        }
    
    def reset(self):
        """
        Reset profiler state completely.
        
        OPTIMIZATION FIX: Clear thread-local storage to prevent regressions
        when profiling multiple workloads sequentially. This addresses the
        factory_randn regression issue where profiler state accumulated.
        
        Call this between profiling sessions to ensure clean state.
        """
        self.component_timings.clear()
        self.active_components.clear()
        
        # Clear thread-local storage completely
        if hasattr(self, '_thread_local'):
            try:
                if hasattr(self._thread_local, 'stack'):
                    del self._thread_local.stack
            except (AttributeError, TypeError):
                pass
        
        # Re-initialize thread-local
        import threading as tls_module
        self._thread_local = tls_module.local()
    
    def print_summary(self):
        """Print summary of all component timings."""
        print("\n" + "="*80)
        print("COMPONENT PROFILING SUMMARY")
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
            print(f"  Mean:   {stats['mean']:.3f}ms")
            print(f"  Std:    {stats['std']:.3f}ms")
            print(f"  Range:  {stats['min']:.3f}ms - {stats['max']:.3f}ms")
            print(f"  Total:  {stats['sum']:.2f}ms")
        
        print(f"\nTotal overhead across all components: {total_time:.2f}ms")
        print("="*80 + "\n")


# Global instance for easy access
_global_profiler = DetailedComponentProfiler()


def get_detailed_profiler() -> DetailedComponentProfiler:
    """Get the global detailed profiler instance."""
    return _global_profiler
