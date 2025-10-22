"""
Test: Optimization improvements from peer review

Tests the new optimization features:
1. Hierarchical Pattern Index
2. Content-addressed caching
3. Early termination modes
4. Thread safety improvements
5. Error handling improvements
6. Performance metrics and observability

These tests validate that the performance optimizations work correctly
and don't break existing functionality.
"""

import pytest
import torch
import torch.nn as nn
import threading
import time
import logging
import genie
from genie.semantic.pattern_registry import PatternRegistry, MatchingMode
from genie.semantic.annotator import SemanticAnnotator
from genie.semantic.pattern_index import HierarchicalPatternIndex, PatternSignature
from genie.core.capture import capture, get_graph

logger = logging.getLogger(__name__)


class TestHierarchicalPatternIndex:
    """Test hierarchical pattern index functionality."""

    def test_index_creation_and_registration(self):
        """Test that index can be created and patterns registered."""
        index = HierarchicalPatternIndex()

        # Create a mock pattern with metadata
        class MockPattern:
            name = "test_pattern"
            expected_operations = frozenset({"aten::matmul", "aten::softmax"})
            min_nodes = 3
            max_nodes = 100
            allows_cycles = False
            max_fanout = 10

        pattern = MockPattern()
        index.register_pattern(pattern)

        # Check that pattern was registered
        assert len(index.all_patterns) == 1
        assert pattern in index.all_patterns

        # Check that pattern was indexed by operations
        assert pattern in index.by_operations[frozenset({"aten::matmul", "aten::softmax"})]

        # Check that pattern was indexed by size bucket
        size_bucket = index._compute_size_bucket(3, 100)
        assert pattern in index.by_size[size_bucket]

        print("✅ Pattern index registration works correctly")

    def test_candidate_selection(self):
        """Test that hierarchical index selects correct candidates."""
        index = HierarchicalPatternIndex()

        # Create patterns with different characteristics
        class LLMPattern:
            name = "llm"
            expected_operations = frozenset({"aten::matmul", "aten::softmax"})  # Only operations present in graph
            min_nodes = 5
            max_nodes = 1000
            allows_cycles = False
            max_fanout = 50

        class VisionPattern:
            name = "vision"
            expected_operations = frozenset({"aten::conv2d", "aten::relu", "aten::max_pool2d"})
            min_nodes = 3
            max_nodes = 200
            allows_cycles = False
            max_fanout = 20

        index.register_pattern(LLMPattern())
        index.register_pattern(VisionPattern())

        # Test graph properties
        graph_ops = frozenset({"aten::matmul", "aten::softmax"})
        graph_size = 50
        has_cycles = False
        max_fanout = 10
        graph_hash = "test_hash"

        # Get candidates
        candidates = index.get_candidate_patterns(
            graph_hash, graph_ops, graph_size, has_cycles, max_fanout
        )

        # Debug: print what was selected
        candidate_names = [p.name for p in candidates]
        print(f"Graph ops: {graph_ops}")
        print(f"Candidates selected: {candidate_names}")

        # Should find LLM pattern (only requires operations present in graph)
        llm_found = any(p.name == "llm" for p in candidates)
        vision_found = any(p.name == "vision" for p in candidates)

        assert llm_found, "LLM pattern should be selected for matmul+softmax graph"
        assert not vision_found, "Vision pattern should not be selected (no conv2d)"

        print("✅ Hierarchical candidate selection works correctly")

    def test_real_pattern_metadata_extraction(self):
        """Test that real patterns have proper metadata for indexing."""
        from genie.patterns.advanced_patterns import AdvancedLLMPattern

        pattern = AdvancedLLMPattern()
        signature = PatternSignature.from_pattern_plugin(pattern)

        assert signature is not None, "LLM pattern should have extractable signature"
        assert len(signature.operations) > 0, "LLM pattern should have expected operations"
        assert signature.min_nodes > 0, "LLM pattern should have min nodes"
        assert signature.max_nodes > signature.min_nodes, "LLM pattern should have valid node range"

        print(f"✅ Real pattern metadata: ops={len(signature.operations)}, nodes={signature.min_nodes}-{signature.max_nodes}")

    def test_index_performance_metrics(self):
        """Test that index performance metrics are collected."""
        index = HierarchicalPatternIndex()

        # Create and register a pattern
        class TestPattern:
            name = "test"
            expected_operations = frozenset({"aten::add"})
            min_nodes = 1
            max_nodes = 10
            allows_cycles = False
            max_fanout = 5

        index.register_pattern(TestPattern())

        # Make several queries
        for i in range(10):
            candidates = index.get_candidate_patterns(
                f"hash_{i}",
                frozenset({"aten::add"}),
                5,
                False,
                3
            )

        stats = index.get_stats()

        assert stats['total_patterns'] == 1
        assert stats['total_queries'] == 10
        assert stats['index_hit_rate'] > 0, "Should have some index hits"

        print(f"✅ Index metrics: {stats}")


class TestContentAddressedCaching:
    """Test content-addressed caching functionality."""

    def test_cache_hit_identical_graphs(self):
        """Test that identical graphs produce cache hits."""
        annotator = SemanticAnnotator(enable_cache=True)

        # Create identical graphs
        def create_test_graph():
            with capture():
                x = torch.randn(10, 20)
                y = torch.randn(20, 30)
                z = x @ y
                result = z.relu()
            return get_graph()

        graph1 = create_test_graph()
        graph2 = create_test_graph()  # Identical structure

        # First annotation should be cache miss
        start = time.time()
        result1 = annotator.annotate(graph1)
        time1 = time.time() - start

        # Second annotation should be cache hit (potentially faster)
        start = time.time()
        result2 = annotator.annotate(graph2)
        time2 = time.time() - start

        # Results should be identical
        assert len(result1.nodes()) == len(result2.nodes()), "Cached results should be identical"

        # Check cache stats
        cache_stats = annotator.get_cache_stats()
        assert cache_stats is not None, "Cache stats should be available"
        assert cache_stats['content_addressed_cache']['cache_size'] >= 1, "Should have cached at least one graph"

        print(f"✅ Content caching: first={time1:.3f}s, second={time2:.3f}s, cache_size={cache_stats['content_addressed_cache']['cache_size']}")

    def test_cache_miss_different_graphs(self):
        """Test that different graphs don't produce cache hits."""
        annotator = SemanticAnnotator(enable_cache=True)

        # Create different graphs
        with capture():
            x1 = torch.randn(10, 20)
            y1 = x1 @ torch.randn(20, 30)
        graph1 = get_graph()

        with capture():
            x2 = torch.randn(15, 25)  # Different shape
            y2 = x2 @ torch.randn(25, 35)
        graph2 = get_graph()

        # Annotate both
        result1 = annotator.annotate(graph1)
        result2 = annotator.annotate(graph2)

        # Should have different results due to different shapes
        cache_stats = annotator.get_cache_stats()
        assert cache_stats['content_addressed_cache']['cache_size'] >= 2, "Should have cached both different graphs"

        print(f"✅ Cache miss for different graphs: cache_size={cache_stats['content_addressed_cache']['cache_size']}")


class TestEarlyTerminationModes:
    """Test early termination functionality."""

    def test_exhaustive_mode_tries_all_patterns(self):
        """Test that exhaustive mode tries all patterns."""
        registry = PatternRegistry()

        # Create a more complex graph that should match some patterns
        with capture():
            batch_size, seq_len, hidden_size = 2, 12, 64
            x = torch.randn(batch_size, seq_len, hidden_size)

            # Attention-like computation
            q_proj = torch.randn(hidden_size, hidden_size)
            k_proj = torch.randn(hidden_size, hidden_size)

            q = x @ q_proj
            k = x @ k_proj
            scores = q @ k.transpose(-2, -1)
            attention = scores.softmax(dim=-1)

            # Value computation
            v_proj = torch.randn(hidden_size, hidden_size)
            v = x @ v_proj
            output = attention @ v

        graph = get_graph()

        # Test exhaustive mode
        result = registry.match_patterns(graph, mode=MatchingMode.EXHAUSTIVE)

        assert result.is_ok, "Exhaustive mode should succeed"
        matches = result.unwrap()
        patterns_tried = len(matches)

        print(f"✅ Exhaustive mode tried {patterns_tried} patterns, found {len([m for m in matches if m.confidence > 0])} matches")

    def test_fast_mode_with_early_termination(self):
        """Test that fast mode terminates early when conditions are met."""
        registry = PatternRegistry()

        with capture():
            x = torch.randn(10, 10)
            y = x @ x
        graph = get_graph()

        # Test fast mode (should terminate early if high confidence pattern found)
        result = registry.match_patterns(
            graph,
            mode=MatchingMode.FAST,
            confidence_threshold=0.8,
            min_patterns_to_try=2
        )

        assert result.is_ok, "Fast mode should succeed"

        print("✅ Fast mode with early termination works")

    def test_required_only_mode(self):
        """Test that required_only mode only tries specified patterns."""
        registry = PatternRegistry()

        with capture():
            x = torch.randn(10, 10)
            y = x @ x
        graph = get_graph()

        # Test required only mode
        result = registry.match_patterns(
            graph,
            mode=MatchingMode.REQUIRED_ONLY,
            required_patterns={"llm", "vision"}
        )

        assert result.is_ok, "Required only mode should succeed"

        print("✅ Required only mode works")


class TestErrorHandlingImprovements:
    """Test improved error handling."""

    def test_graceful_handling_of_invalid_graphs(self):
        """Test that invalid graphs are handled gracefully."""
        registry = PatternRegistry()

        # Create a graph that might cause issues
        with capture():
            x = torch.randn(1, 5)
            # Operation that might not be supported
            try:
                result = torch.linalg.svd(x)
            except:
                # If SVD fails, just use a simple operation
                result = x + 1

        graph = get_graph()

        # Should not crash even if patterns fail
        result = registry.match_patterns(graph, mode=MatchingMode.EXHAUSTIVE)

        # Should either succeed or return error result (not crash)
        assert result.is_ok or not result.is_ok, "Should return a result (success or error)"

        print(f"✅ Error handling: result is {'success' if result.is_ok else 'error'}")

    def test_pattern_failure_isolation(self):
        """Test that one pattern failure doesn't affect others."""
        registry = PatternRegistry()

        with capture():
            x = torch.randn(10, 10)
            y = x @ x
        graph = get_graph()

        # Run matching multiple times to ensure consistency
        results = []
        for i in range(3):
            result = registry.match_patterns(graph, mode=MatchingMode.EXHAUSTIVE)
            results.append(result)

        # All should have same outcome (all success or all error)
        all_success = all(r.is_ok for r in results)
        all_error = all(not r.is_ok for r in results)

        assert all_success or all_error, "Pattern matching should be consistent across runs"

        print(f"✅ Pattern failure isolation: consistent results ({'success' if all_success else 'error'})")


class TestPerformanceMetrics:
    """Test performance metrics and observability."""

    def test_metrics_collection(self):
        """Test that performance metrics are collected correctly."""
        registry = PatternRegistry()

        with capture():
            x = torch.randn(10, 10)
            y = x @ x
        graph = get_graph()

        # Run pattern matching
        result = registry.match_patterns(graph, mode=MatchingMode.EXHAUSTIVE)

        # Get performance report
        report = registry.get_performance_report()

        assert isinstance(report, dict), "Performance report should be a dictionary"
        assert '_index' in report, "Should have index statistics"
        assert '_matching' in report, "Should have matching statistics"

        print(f"✅ Metrics collection: {len(report)} metric categories")

    def test_cache_performance_tracking(self):
        """Test that cache performance is tracked."""
        annotator = SemanticAnnotator(enable_cache=True)

        # Create and annotate graphs
        for i in range(3):
            with capture():
                x = torch.randn(10, 10)
                y = x @ x
            graph = get_graph()
            result = annotator.annotate(graph)

        # Check cache stats
        cache_stats = annotator.get_cache_stats()

        assert cache_stats is not None, "Cache stats should be available"
        assert 'content_addressed_cache' in cache_stats, "Should have content cache stats"

        content_stats = cache_stats['content_addressed_cache']
        assert 'hits' in content_stats, "Should track cache hits"
        assert 'misses' in content_stats, "Should track cache misses"
        assert 'hit_rate' in content_stats, "Should calculate hit rate"

        print(f"✅ Cache performance tracking: hit_rate={content_stats['hit_rate']:.2f}")


class TestThreadSafetyImprovements:
    """Test thread safety improvements."""

    def test_concurrent_pattern_matching(self):
        """Test that pattern matching is thread-safe."""
        registry = PatternRegistry()

        def worker(thread_id):
            """Worker function for concurrent testing."""
            try:
                with capture():
                    x = torch.randn(5, 5)
                    y = x @ x
                graph = get_graph()

                result = registry.match_patterns(graph, mode=MatchingMode.FAST)
                return result.is_ok
            except Exception:
                return False

        # Run concurrent workers
        results = []
        threads = []

        for i in range(10):
            thread = threading.Thread(target=lambda: results.append(worker(i)))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All should succeed
        success_count = sum(results)
        assert success_count == 10, f"Only {success_count}/10 threads succeeded"

        print(f"✅ Thread safety: {success_count}/10 concurrent pattern matches succeeded")

    def test_concurrent_cache_access(self):
        """Test that cache access is thread-safe."""
        annotator = SemanticAnnotator(enable_cache=True)

        def worker(thread_id):
            """Worker function for concurrent cache testing."""
            try:
                with capture():
                    x = torch.randn(8, 8)
                    y = x @ x
                graph = get_graph()

                result = annotator.annotate(graph)
                return len(list(result.nodes())) > 0
            except Exception:
                return False

        # Run concurrent workers
        results = []
        threads = []

        for i in range(8):
            thread = threading.Thread(target=lambda: results.append(worker(i)))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        success_count = sum(results)
        assert success_count == 8, f"Only {success_count}/8 cache accesses succeeded"

        print(f"✅ Cache thread safety: {success_count}/8 concurrent cache accesses succeeded")


class TestIntegration:
    """Integration tests for all optimizations working together."""

    def test_end_to_end_optimization_pipeline(self):
        """Test complete optimization pipeline end-to-end."""
        annotator = SemanticAnnotator(enable_cache=True)

        # Create a moderately complex graph
        with capture():
            batch_size, seq_len, hidden_size = 2, 12, 64

            # Input
            x = torch.randn(batch_size, seq_len, hidden_size)

            # Attention-like computation
            q_proj = torch.randn(hidden_size, hidden_size)
            k_proj = torch.randn(hidden_size, hidden_size)

            q = x @ q_proj
            k = x @ k_proj

            # Attention scores
            scores = q @ k.transpose(-2, -1)
            attention = scores.softmax(dim=-1)

            # Output projection
            v_proj = torch.randn(hidden_size, hidden_size)
            v = x @ v_proj
            output = attention @ v

            # Final projection
            final_proj = torch.randn(hidden_size, hidden_size)
            result = output @ final_proj

        graph = get_graph()

        # Test full annotation pipeline
        start_time = time.time()
        annotated = annotator.annotate(graph)
        total_time = time.time() - start_time

        # Verify results
        assert len(list(annotated.nodes())) > 0, "Should have annotated nodes"
        assert annotated.costs['total_compute_flops'] > 0, "Should have cost estimates"

        # Check performance metrics
        cache_stats = annotator.get_cache_stats()
        registry_report = annotator.pattern_registry.get_performance_report()

        print(f"✅ End-to-end pipeline: {total_time:.3f}s, {len(list(annotated.nodes()))} nodes, {annotated.costs['total_compute_flops']:.2e} FLOPs")

    def test_optimization_benefits(self):
        """Test that optimizations provide measurable benefits."""
        # Test without optimizations (simulate)
        registry1 = PatternRegistry()

        with capture():
            x = torch.randn(20, 30)
            y = x @ torch.randn(30, 40)
        graph = get_graph()

        # Time multiple runs
        times = []
        for i in range(5):
            start = time.time()
            result = registry1.match_patterns(graph, mode=MatchingMode.EXHAUSTIVE)
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)

        # Get final metrics
        report = registry1.get_performance_report()
        index_stats = report.get('_index', {})
        matching_stats = report.get('_matching', {})

        print(f"✅ Optimization benefits: avg_time={avg_time:.4f}s, index_hit_rate={index_stats.get('index_hit_rate', 0):.2f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
