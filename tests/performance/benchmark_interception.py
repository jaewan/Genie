"""
Performance benchmarks for interception overhead.

Measures the overhead of:
1. LazyTensor creation and operation interception
2. Factory function wrapping
3. Capture context signaling
4. End-to-end graph capture

Targets:
- <1μs per-operation overhead
- <100ns capture context overhead
- <100ms for 1000 operations
"""

import time
import statistics
import torch
import numpy as np

import genie
from genie.core.lazy_tensor import LazyTensor
from genie.core.factory_interceptor import FactoryInterceptor


class InterceptionBenchmark:
    """Benchmark suite for interception performance."""

    def __init__(self):
        self.results = {}

    def benchmark_native_operations(self, iterations=10000):
        """Benchmark native PyTorch operations."""
        times = []

        # Warmup
        for _ in range(100):
            x = torch.randn(100, 100)
            y = x + 1

        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            x = torch.randn(100, 100)
            y = x + 1
            end = time.perf_counter()
            times.append(end - start)

        return statistics.mean(times), statistics.stdev(times)

    def benchmark_lazy_tensor_operations(self, iterations=10000):
        """Benchmark LazyTensor operations (context-based API)."""
        times = []

        # Warmup
        for _ in range(100):
            with genie.capture():
                x = torch.randn(100, 100)
                y = x + 1

        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            with genie.capture():
                x = torch.randn(100, 100)
                y = x + 1
            end = time.perf_counter()
            times.append(end - start)

        return statistics.mean(times), statistics.stdev(times)

    def benchmark_device_based_operations(self, iterations=10000):
        """Benchmark device-based API operations."""
        times = []

        # Warmup
        for _ in range(100):
            x = torch.randn(100, 100, device='remote_accelerator:0')
            y = x + 1

        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            x = torch.randn(100, 100, device='remote_accelerator:0')
            y = x + 1
            end = time.perf_counter()
            times.append(end - start)

        return statistics.mean(times), statistics.stdev(times)

    def benchmark_capture_context_overhead(self, iterations=10000):
        """Benchmark overhead of capture context signaling."""
        times = []

        # Benchmark without capture context
        for _ in range(iterations):
            start = time.perf_counter()
            x = torch.randn(100, 100)
            end = time.perf_counter()
            times.append(end - start)

        no_capture_time = statistics.mean(times)

        # Benchmark with capture context (but no LazyTensor creation)
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            with genie.capture():
                x = torch.randn(100, 100)  # Creates LazyTensor
            end = time.perf_counter()
            times.append(end - start)

        capture_time = statistics.mean(times)

        return capture_time - no_capture_time

    def benchmark_factory_interceptor_overhead(self, iterations=10000):
        """Benchmark factory interceptor overhead."""
        interceptor = FactoryInterceptor()

        # Benchmark without interceptor
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            x = torch.randn(100, 100)
            end = time.perf_counter()
            times.append(end - start)

        no_interceptor_time = statistics.mean(times)

        # Benchmark with interceptor
        interceptor.wrap()
        try:
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                x = torch.randn(100, 100)
                end = time.perf_counter()
                times.append(end - start)

            with_interceptor_time = statistics.mean(times)
        finally:
            interceptor.unwrap()

        return with_interceptor_time - no_interceptor_time

    def benchmark_complex_graph_capture(self, operations=1000):
        """Benchmark capture of complex computation graph."""
        # Create a model-like computation
        def complex_computation():
            x = torch.randn(64, 128)
            for i in range(operations // 10):
                x = x @ torch.randn(128, 128)
                x = torch.relu(x)
                x = x + torch.randn_like(x) * 0.1
            return x

        # Benchmark native execution
        start = time.perf_counter()
        for _ in range(10):
            result = complex_computation()
        native_time = time.perf_counter() - start

        # Benchmark with capture
        start = time.perf_counter()
        for _ in range(10):
            with genie.capture():
                result = complex_computation()
        capture_time = time.perf_counter() - start

        return capture_time - native_time

    def run_all_benchmarks(self):
        """Run all benchmarks and return results."""
        print("Running interception performance benchmarks...")

        # Basic operation benchmarks
        native_mean, native_std = self.benchmark_native_operations()
        lazy_mean, lazy_std = self.benchmark_lazy_tensor_operations()
        device_mean, device_std = self.benchmark_device_based_operations()

        print(f"Native operations: {native_mean*1e9".1f"}ns ± {native_std*1e9".1f"}ns")
        print(f"LazyTensor (context): {lazy_mean*1e9".1f"}ns ± {lazy_std*1e9".1f"}ns")
        print(f"LazyTensor (device): {device_mean*1e9".1f"}ns ± {device_std*1e9".1f"}ns")

        # Context overhead
        context_overhead = self.benchmark_capture_context_overhead()
        print(f"Capture context overhead: {context_overhead*1e9".1f"}ns")

        # Factory interceptor overhead
        factory_overhead = self.benchmark_factory_interceptor_overhead()
        print(f"Factory interceptor overhead: {factory_overhead*1e9".1f"}ns")

        # Complex graph benchmark
        complex_overhead = self.benchmark_complex_graph_capture()
        print(f"Complex graph capture overhead: {complex_overhead*1000".1f"}ms")

        # Store results
        self.results = {
            'native_time_ns': native_mean * 1e9,
            'lazy_context_time_ns': lazy_mean * 1e9,
            'lazy_device_time_ns': device_mean * 1e9,
            'context_overhead_ns': context_overhead * 1e9,
            'factory_overhead_ns': factory_overhead * 1e9,
            'complex_overhead_ms': complex_overhead * 1000,
        }

        return self.results

    def validate_targets(self):
        """Validate that benchmarks meet performance targets."""
        results = self.results

        # Target: <1μs per-operation overhead
        lazy_overhead = results['lazy_context_time_ns'] - results['native_time_ns']
        print(f"Per-operation overhead: {lazy_overhead".1f"}ns")

        if lazy_overhead > 1000:
            raise AssertionError(f"Per-operation overhead too high: {lazy_overhead:.1f}ns > 1000ns")

        # Target: <100ns capture context overhead
        context_overhead = results['context_overhead_ns']
        print(f"Capture context overhead: {context_overhead".1f"}ns")

        if context_overhead > 100:
            raise AssertionError(f"Capture context overhead too high: {context_overhead:.1f}ns > 100ns")

        # Target: <100ms for 1000 operations
        complex_overhead = results['complex_overhead_ms']
        print(f"Complex graph overhead: {complex_overhead".1f"}ms")

        if complex_overhead > 100:
            raise AssertionError(f"Complex graph overhead too high: {complex_overhead:.1f}ms > 100ms")

        print("✅ All performance targets met!")


def benchmark_interception_overhead():
    """Main benchmark function for per-operation overhead."""
    benchmark = InterceptionBenchmark()
    results = benchmark.run_all_benchmarks()
    benchmark.validate_targets()
    return results


def benchmark_capture_context_overhead():
    """Benchmark capture context overhead specifically."""
    benchmark = InterceptionBenchmark()
    overhead = benchmark.benchmark_capture_context_overhead()
    print(f"Capture context overhead: {overhead*1e9".1f"}ns")

    if overhead * 1e9 > 100:
        raise AssertionError(f"Capture context overhead too high: {overhead*1e9:.1f}ns > 100ns")

    return overhead


if __name__ == "__main__":
    print("Starting interception benchmarks...")
    print("=" * 50)

    try:
        benchmark_interception_overhead()
        print("✅ All benchmarks passed!")
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        raise
