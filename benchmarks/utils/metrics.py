"""
Metrics and measurement utilities for benchmarks.

Provides standardized metrics collection, calculation, and reporting
functions used across different benchmark types.
"""

import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict
import json
from pathlib import Path


@dataclass
class LatencyMetrics:
    """Latency measurements and statistics."""
    latencies: List[float] = field(default_factory=list)

    def add_measurement(self, latency_ms: float):
        """Add a latency measurement."""
        self.latencies.append(latency_ms)

    def get_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.latencies:
            return {
                'count': 0,
                'avg_ms': 0.0,
                'min_ms': 0.0,
                'max_ms': 0.0,
                'p50_ms': 0.0,
                'p95_ms': 0.0,
                'p99_ms': 0.0,
                'std_ms': 0.0,
            }

        sorted_latencies = sorted(self.latencies)

        return {
            'count': len(self.latencies),
            'avg_ms': statistics.mean(self.latencies),
            'min_ms': min(self.latencies),
            'max_ms': max(self.latencies),
            'p50_ms': statistics.median(sorted_latencies),
            'p95_ms': sorted_latencies[int(0.95 * len(sorted_latencies))],
            'p99_ms': sorted_latencies[int(0.99 * len(sorted_latencies))],
            'std_ms': statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0.0,
        }


@dataclass
class ThroughputMetrics:
    """Throughput measurements and statistics."""
    total_samples: int = 0
    total_time_sec: float = 0.0
    measurements: List[Dict[str, float]] = field(default_factory=list)

    def add_measurement(self, samples: int, time_sec: float):
        """Add a throughput measurement."""
        self.total_samples += samples
        self.total_time_sec += time_sec
        self.measurements.append({
            'samples': samples,
            'time_sec': time_sec,
            'throughput_samples_per_sec': samples / time_sec,
        })

    def get_stats(self) -> Dict[str, float]:
        """Get throughput statistics."""
        if not self.measurements:
            return {
                'total_samples': 0,
                'total_time_sec': 0.0,
                'avg_throughput_samples_per_sec': 0.0,
                'min_throughput_samples_per_sec': 0.0,
                'max_throughput_samples_per_sec': 0.0,
            }

        throughputs = [m['throughput_samples_per_sec'] for m in self.measurements]

        return {
            'total_samples': self.total_samples,
            'total_time_sec': self.total_time_sec,
            'avg_throughput_samples_per_sec': statistics.mean(throughputs),
            'min_throughput_samples_per_sec': min(throughputs),
            'max_throughput_samples_per_sec': max(throughputs),
            'overall_throughput_samples_per_sec': self.total_samples / self.total_time_sec,
        }


@dataclass
class MemoryMetrics:
    """Memory usage measurements and statistics."""
    measurements: List[Dict[str, float]] = field(default_factory=list)

    def add_measurement(self, gpu_memory_mb: float, cpu_memory_mb: Optional[float] = None,
                       gpu_utilization: Optional[float] = None):
        """Add a memory measurement."""
        measurement = {
            'timestamp': time.time(),
            'gpu_memory_mb': gpu_memory_mb,
            'gpu_utilization': gpu_utilization,
        }
        if cpu_memory_mb is not None:
            measurement['cpu_memory_mb'] = cpu_memory_mb
        self.measurements.append(measurement)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self.measurements:
            return {
                'count': 0,
                'avg_gpu_memory_mb': 0.0,
                'max_gpu_memory_mb': 0.0,
                'avg_gpu_utilization': 0.0,
                'max_gpu_utilization': 0.0,
            }

        gpu_memories = [m['gpu_memory_mb'] for m in self.measurements]
        gpu_utils = [m['gpu_utilization'] for m in self.measurements if m['gpu_utilization'] is not None]

        stats = {
            'count': len(self.measurements),
            'avg_gpu_memory_mb': statistics.mean(gpu_memories),
            'max_gpu_memory_mb': max(gpu_memories),
            'min_gpu_memory_mb': min(gpu_memories),
        }

        if gpu_utils:
            stats.update({
                'avg_gpu_utilization': statistics.mean(gpu_utils),
                'max_gpu_utilization': max(gpu_utils),
                'min_gpu_utilization': min(gpu_utils),
            })

        return stats


class BenchmarkMetrics:
    """Comprehensive metrics collection for benchmarks."""

    def __init__(self):
        self.latency = LatencyMetrics()
        self.throughput = ThroughputMetrics()
        self.memory = MemoryMetrics()
        self.custom_metrics = defaultdict(list)
        self.start_time = None
        self.end_time = None

    def start_benchmark(self):
        """Mark the start of benchmark execution."""
        self.start_time = time.time()

    def end_benchmark(self):
        """Mark the end of benchmark execution."""
        self.end_time = time.time()

    def add_latency_measurement(self, latency_ms: float):
        """Add a latency measurement."""
        self.latency.add_measurement(latency_ms)

    def add_throughput_measurement(self, samples: int, time_sec: float):
        """Add a throughput measurement."""
        self.throughput.add_measurement(samples, time_sec)

    def add_memory_measurement(self, gpu_memory_mb: float, cpu_memory_mb: Optional[float] = None,
                              gpu_utilization: Optional[float] = None):
        """Add a memory measurement."""
        self.memory.add_measurement(gpu_memory_mb, cpu_memory_mb, gpu_utilization)

    def add_custom_metric(self, name: str, value: Any):
        """Add a custom metric."""
        self.custom_metrics[name].append(value)

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        total_time = self.end_time - self.start_time if self.start_time and self.end_time else 0

        return {
            'total_execution_time_sec': total_time,
            'latency': self.latency.get_stats(),
            'throughput': self.throughput.get_stats(),
            'memory': self.memory.get_stats(),
            'custom_metrics': dict(self.custom_metrics),
        }

    def save_to_json(self, filepath: str):
        """Save metrics to JSON file."""
        summary = self.get_summary()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)


def measure_function_latency(func: Callable, *args, num_runs: int = 10, warmup_runs: int = 3) -> LatencyMetrics:
    """Measure function latency with warmup."""
    # Warmup
    for _ in range(warmup_runs):
        func(*args)

    # Measurement
    metrics = LatencyMetrics()
    for _ in range(num_runs):
        start = time.time()
        result = func(*args)
        latency = (time.time() - start) * 1000  # Convert to ms
        metrics.add_measurement(latency)

    return metrics


def calculate_efficiency_metrics(latency_ms: float, throughput_samples_per_sec: float,
                               memory_mb: float) -> Dict[str, float]:
    """Calculate efficiency metrics combining latency, throughput, and memory."""
    return {
        'latency_efficiency': 1.0 / latency_ms,  # Higher is better
        'throughput_efficiency': throughput_samples_per_sec / memory_mb,  # Samples/sec per MB
        'memory_efficiency': throughput_samples_per_sec / latency_ms,  # Throughput per latency unit
        'overall_efficiency_score': (throughput_samples_per_sec / latency_ms) * (1.0 / memory_mb),
    }


def format_metrics_for_display(metrics: Dict[str, Any]) -> str:
    """Format metrics dictionary for human-readable display."""
    lines = ["Benchmark Results", "=" * 50]

    def format_value(value):
        if isinstance(value, float):
            if value < 1.0:
                return f"{value:.4f}"
            elif value < 10.0:
                return f"{value:.2f}"
            else:
                return f"{value:,.0f}"
        return str(value)

    def format_section(title, data):
        if not data:
            return []
        lines = [f"\n{title}:"]
        for key, value in data.items():
            if isinstance(value, dict):
                lines.extend(format_section(f"  {key}", value))
            else:
                lines.append(f"  {key}: {format_value(value)}")
        return lines

    for section_name, section_data in metrics.items():
        if isinstance(section_data, dict):
            lines.extend(format_section(section_name.replace('_', ' ').title(), section_data))
        else:
            lines.append(f"{section_name}: {format_value(section_data)}")

    return '\n'.join(lines)
