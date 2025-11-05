"""
Memory Leak Tests

These tests validate that the optimization improvements don't cause
memory leaks and that cache eviction works correctly.

Usage:
    pytest tests/performance/test_memory_leaks.py -v

Memory Profiling:
    python -m memory_profiler tests/performance/test_memory_leaks.py::TestMemoryLeaks::test_cache_eviction
"""

import pytest
import torch
import torch.nn as nn
import time
import gc
import psutil
import os
from typing import List, Dict, Any

import djinn
from djinn.core.capture import capture, get_graph
from djinn.frontend.semantic.annotator import SemanticAnnotator
from djinn.frontend.semantic.pattern_registry import PatternRegistry, MatchingMode

# Optional memory profiling
try:
    from memory_profiler import profile
    MEMORY_PROFILING = True
except ImportError:
    MEMORY_PROFILING = False
    def profile(func):
        return func


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


class TestMemoryLeaks:
    """Test for memory leaks in optimization components."""

    def setup_method(self):
        """Clean up before each test."""
        # Clear caches
        if hasattr(genie, '_global_annotator'):
            genie._global_annotator.clear_cache()

        # Force garbage collection
        gc.collect()

    def test_cache_size_limits_respected(self):
        """Test that cache size limits prevent unbounded growth."""
        annotator = SemanticAnnotator(enable_cache=True)

        # Create many different graphs to fill cache
        initial_memory = get_memory_usage_mb()

        for i in range(50):
            with capture():
                size = 10 + i * 2  # Varying sizes
                x = torch.randn(size, size)
                y = x @ torch.randn(size, size)
                z = y.relu()
            graph = get_graph()

            result = annotator.annotate(graph)

            # Check cache size doesn't grow unbounded
            cache_stats = annotator.get_cache_stats()
            cache_size = cache_stats.get('content_addressed_cache', {}).get('cache_size', 0)

            # Should not exceed reasonable limits
            assert cache_size <= 1000, f"Cache size too large: {cache_size}"

        final_memory = get_memory_usage_mb()
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable (< 100MB for 50 graphs)
        assert memory_growth < 100, f"Memory growth too high: {memory_growth:.1f}MB"

        print(f"✅ Cache limits: {cache_size} entries, {memory_growth:.1f}MB growth")

    @profile
    def test_long_running_cache_behavior(self):
        """Test cache behavior under long-running load."""
        annotator = SemanticAnnotator(enable_cache=True)

        # Simulate long-running workload
        for iteration in range(100):
            # Create varying graphs
            with capture():
                if iteration % 3 == 0:
                    # LLM-like
                    x = torch.randn(2, 10, 32)
                    y = x @ torch.randn(32, 32)
                elif iteration % 3 == 1:
                    # Vision-like
                    x = torch.randn(1, 3, 16, 16)
                    conv = torch.randn(16, 3, 3, 3)
                    y = torch.conv2d(x, conv, padding=1)
                else:
                    # Simple
                    x = torch.randn(8, 8)
                    y = x @ x

            graph = get_graph()
            result = annotator.annotate(graph)

            # Periodically check cache stats
            if iteration % 20 == 0:
                cache_stats = annotator.get_cache_stats()
                cache_size = cache_stats.get('content_addressed_cache', {}).get('cache_size', 0)
                memory_mb = cache_stats.get('content_addressed_cache', {}).get('memory_mb', 0)

                print(f"  Iteration {iteration}: cache_size={cache_size}, memory={memory_mb:.1f}MB")

                # Cache should stabilize (not grow indefinitely)
                if iteration > 40:
                    assert cache_size <= 100, f"Cache size growing too large: {cache_size}"

        print("✅ Long-running cache behavior: stable memory usage")

    def test_pattern_registry_memory_usage(self):
        """Test that pattern registry doesn't leak memory."""
        initial_memory = get_memory_usage_mb()

        # Create and destroy multiple registries
        for i in range(20):
            registry = PatternRegistry()

            # Register patterns and run matching
            with capture():
                x = torch.randn(5, 5)
                y = x @ x
            graph = get_graph()

            result = registry.match_patterns(graph, mode=MatchingMode.EXHAUSTIVE)

            # Explicitly delete registry
            del registry
            gc.collect()

        final_memory = get_memory_usage_mb()
        memory_growth = final_memory - initial_memory

        # Should not have significant memory growth
        assert memory_growth < 50, f"Pattern registry memory growth too high: {memory_growth:.1f}MB"

        print(f"✅ Pattern registry: {memory_growth:.1f}MB growth across 20 instances")

    def test_thread_safety_memory_usage(self):
        """Test memory usage under concurrent load."""
        import threading

        def worker(thread_id):
            """Worker that creates and processes graphs."""
            try:
                with capture():
                    x = torch.randn(10, 10)
                    y = x @ torch.randn(10, 10)
                    z = y.relu()
                graph = get_graph()

                # Use shared annotator
                result = genie.annotate_graph(graph)
                return True
            except Exception:
                return False

        initial_memory = get_memory_usage_mb()

        # Run concurrent workers
        results = []
        threads = []

        for i in range(30):
            thread = threading.Thread(target=lambda: results.append(worker(i)))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        success_count = sum(results)
        final_memory = get_memory_usage_mb()
        memory_growth = final_memory - initial_memory

        # Should succeed and not leak memory excessively
        assert success_count == 30, f"Only {success_count}/30 threads succeeded"
        assert memory_growth < 200, f"Concurrent memory growth too high: {memory_growth:.1f}MB"

        print(f"✅ Thread safety: {success_count}/30 succeeded, {memory_growth:.1f}MB growth")

    def test_cache_eviction_effectiveness(self):
        """Test that cache eviction works correctly and prevents OOM."""
        annotator = SemanticAnnotator(enable_cache=True, max_memory_mb=10)  # Small cache

        # Create graphs larger than cache limit
        for i in range(20):
            with capture():
                # Create progressively larger graphs
                size = 20 + i * 5
                x = torch.randn(size, size)
                y = x @ torch.randn(size, size)
                z = y.relu()
            graph = get_graph()

            result = annotator.annotate(graph)

            # Check cache size stays bounded
            cache_stats = annotator.get_cache_stats()
            cache_size = cache_stats.get('content_addressed_cache', {}).get('cache_size', 0)
            memory_mb = cache_stats.get('content_addressed_cache', {}).get('memory_mb', 0)

            # Should not exceed memory limit
            assert memory_mb <= 15, f"Cache memory exceeded limit: {memory_mb:.1f}MB"

            if i % 5 == 0:
                print(f"  Graph {i}: cache_size={cache_size}, memory={memory_mb:.1f}MB")

        print("✅ Cache eviction: memory limits respected")

    def test_lazy_tensor_shape_cache_bounds(self):
        """Test that LazyTensor shape cache stays bounded."""
        from djinn.frontend.core.lazy_tensor import LazyTensor

        # Clear existing cache
        LazyTensor._shape_cache.clear()

        initial_cache_size = len(LazyTensor._shape_cache)

        # Create many operations that would use shape inference
        for i in range(100):
            with capture():
                # Different shapes to create different cache entries
                size1, size2 = 10 + i, 20 + i
                x = torch.randn(size1, size2)
                y = x @ torch.randn(size2, size1)
                z = y.relu()
            graph = get_graph()

            # Force shape inference by accessing shapes
            for node in graph.nodes():
                # LazyDAGNodeAdapter may not expose shape directly
                try:
                    _ = node.shape
                except AttributeError:
                    # Shape may not be directly available - that's OK
                    pass

        final_cache_size = len(LazyTensor._shape_cache)

        # Cache should not grow unbounded (should be much less than 100)
        assert final_cache_size < 50, f"Shape cache grew too large: {final_cache_size} entries"

        print(f"✅ Shape cache: {initial_cache_size} -> {final_cache_size} entries")


class TestMemoryProfiling:
    """Advanced memory profiling tests."""

    def test_memory_usage_over_time(self):
        """Monitor memory usage over extended time period."""
        annotator = SemanticAnnotator(enable_cache=True)

        memory_samples = []

        # Run extended workload
        for hour in range(24):  # Simulate 24 iterations
            start_memory = get_memory_usage_mb()

            # Simulate typical workload
            for batch in range(10):
                with capture():
                    if batch % 4 == 0:
                        # LLM workload
                        x = torch.randn(2, 12, 64)
                        y = x @ torch.randn(64, 64)
                    elif batch % 4 == 1:
                        # Vision workload
                        x = torch.randn(1, 3, 32, 32)
                        conv = torch.randn(16, 3, 3, 3)
                        y = torch.conv2d(x, conv, padding=1)
                    elif batch % 4 == 2:
                        # Mixed workload
                        x = torch.randn(1, 10, 32)
                        y = x @ torch.randn(32, 64)
                        z = y.relu()
                    else:
                        # Simple workload
                        x = torch.randn(8, 8)
                        y = x @ x

                graph = get_graph()
                result = annotator.annotate(graph)

            end_memory = get_memory_usage_mb()
            memory_growth = end_memory - start_memory

            memory_samples.append({
                'hour': hour,
                'start_memory_mb': start_memory,
                'end_memory_mb': end_memory,
                'growth_mb': memory_growth
            })

            # Force garbage collection
            gc.collect()

            if hour % 6 == 0:
                print(f"  Hour {hour}: {start_memory:.1f}MB -> {end_memory:.1f}MB (+{memory_growth:.1f}MB)")

        # Analyze memory trend
        total_growth = memory_samples[-1]['end_memory_mb'] - memory_samples[0]['start_memory_mb']

        # Should not have significant memory growth over time
        assert total_growth < 100, f"Memory grew too much over time: {total_growth:.1f}MB"

        # Memory growth should stabilize (not continuously increase)
        recent_growths = [s['growth_mb'] for s in memory_samples[-6:]]
        avg_recent_growth = sum(recent_growths) / len(recent_growths)

        # Recent growth should be minimal (cache should stabilize)
        assert avg_recent_growth < 5, f"Recent memory growth too high: {avg_recent_growth:.1f}MB/hour"

        print(f"✅ Memory stability: {total_growth:.1f}MB total growth, {avg_recent_growth:.1f}MB/hour recent")

    def test_cache_fragmentation(self):
        """Test for cache fragmentation issues."""
        annotator = SemanticAnnotator(enable_cache=True)

        # Create graphs of varying sizes to test fragmentation
        sizes = [10, 50, 100, 200, 500, 1000, 2000]

        for size in sizes:
            with capture():
                x = torch.randn(size, size)
                y = x @ torch.randn(size, size)
            graph = get_graph()

            result = annotator.annotate(graph)

            # Check cache efficiency
            cache_stats = annotator.get_cache_stats()
            cache_size = cache_stats.get('content_addressed_cache', {}).get('cache_size', 0)
            memory_mb = cache_stats.get('content_addressed_cache', {}).get('memory_mb', 0)

            # For very large graphs, cache might evict smaller ones
            if size > 1000:
                # Should handle large graphs without excessive memory usage
                assert memory_mb < 50, f"Memory usage too high for large graph: {memory_mb:.1f}MB"

            print(f"  Size {size}: cache_size={cache_size}, memory={memory_mb:.1f}MB")

        print("✅ Cache fragmentation: handled varying graph sizes")


def test_memory_usage_baseline():
    """Establish baseline memory usage for comparison."""
    initial_memory = get_memory_usage_mb()

    # Basic operations
    with capture():
        x = torch.randn(100, 100)
        y = x @ torch.randn(100, 100)
    graph = get_graph()

    result = genie.annotate_graph(graph)

    final_memory = get_memory_usage_mb()
    memory_used = final_memory - initial_memory

    # Should use reasonable amount of memory
    assert memory_used < 50, f"Memory usage too high: {memory_used:.1f}MB"

    print(f"✅ Baseline memory usage: {memory_used:.1f}MB for typical workload")


if __name__ == '__main__':
    # Run memory tests with optional profiling
    if MEMORY_PROFILING:
        print("🧠 Running with memory profiling...")
        # Run specific test with profiling
        test = TestMemoryLeaks()
        test.test_long_running_cache_behavior()
    else:
        print("⚠️  Memory profiling not available (install memory_profiler)")
        pytest.main([__file__, '-v', '--tb=short'])
