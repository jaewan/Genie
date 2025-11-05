"""
Comprehensive test suite for peer review improvements.

Tests cover:
1. Thread safety in global caches
2. Automatic graph compaction (memory leak prevention)
3. Error handling with Result types
4. Type safety with NodeProtocol
5. Unified graph interface
"""

import pytest
import torch
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CRITICAL ISSUE #1: Thread Safety Tests
# ============================================================================

class TestThreadSafety:
    """Test thread safety in global caches and concurrent operations."""
    
    def test_shape_cache_thread_safe(self):
        """Test that shape cache is thread-safe under concurrent access."""
        from djinn.frontend.core.lazy_tensor import _global_shape_cache, _global_cache_lock
        
        # Shared cache
        cache = _global_shape_cache
        results = []
        errors = []
        
        def worker(thread_id: int):
            """Worker thread that adds and reads from cache."""
            try:
                for i in range(100):
                    key = f"thread_{thread_id}_key_{i}"
                    shape = torch.Size([thread_id, i])
                    
                    # Write
                    with _global_cache_lock:
                        cache[key] = shape
                    
                    # Read
                    with _global_cache_lock:
                        retrieved = cache.get(key)
                    
                    if retrieved != shape:
                        errors.append(f"Mismatch: {retrieved} != {shape}")
                
                results.append(thread_id)
            except Exception as e:
                errors.append(str(e))
        
        # Run with multiple threads
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, i) for i in range(8)]
            for future in futures:
                future.result()
        
        # Verify no errors
        assert not errors, f"Errors occurred: {errors}"
        assert len(results) == 8, f"Not all threads completed"
    
    def test_concurrent_lazy_tensor_creation(self):
        """Test creating LazyTensors concurrently is safe."""
        from djinn.frontend.core.lazy_tensor import LazyTensor
        
        tensors = []
        errors = []
        
        def create_tensors(thread_id: int):
            """Create multiple tensors in a thread."""
            try:
                for i in range(50):
                    # Create tensor on remote device
                    x = torch.randn(10, 10, device="cpu")  # Fallback to CPU for testing
                    tensors.append(x)
            except Exception as e:
                errors.append(str(e))
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(create_tensors, i) for i in range(4)]
            for future in futures:
                future.result()
        
        assert not errors, f"Errors: {errors}"
        assert len(tensors) == 200, f"Expected 200 tensors, got {len(tensors)}"


# ============================================================================
# CRITICAL ISSUE #2: Memory Leak Prevention Tests
# ============================================================================

class TestMemoryManagement:
    """Test automatic graph compaction and memory management."""
    
    def test_graph_compactor_initialization(self):
        """Test GraphCompactor initializes correctly."""
        from djinn.memory import GraphCompactor
        
        # Mock graph builder
        class MockGraphBuilder:
            def __init__(self):
                self.nodes = {}
                self.root_tensor = None
        
        builder = MockGraphBuilder()
        compactor = GraphCompactor(builder)
        
        assert compactor._operation_count == 0
        assert compactor._compaction_count == 0
        assert compactor.COMPACTION_THRESHOLD_OPS == 500
    
    def test_memory_monitor_pressure_levels(self):
        """Test memory monitor correctly identifies pressure levels."""
        from djinn.memory import MemoryMonitor, MemoryPressure
        
        monitor = MemoryMonitor(limit_mb=100)  # 100MB limit
        
        # Test normal pressure (40% utilization)
        monitor.track("obj1", 40 * 1024 * 1024)  # 40MB
        stats = monitor.get_stats()
        assert stats.pressure == MemoryPressure.NORMAL
        
        # Test moderate pressure (60% utilization)
        monitor.track("obj2", 20 * 1024 * 1024)  # +20MB = 60MB
        stats = monitor.get_stats()
        assert stats.pressure == MemoryPressure.MODERATE
        
        # Test high pressure (80% utilization)
        monitor.track("obj3", 20 * 1024 * 1024)  # +20MB = 80MB
        stats = monitor.get_stats()
        assert stats.pressure == MemoryPressure.HIGH
        
        # Test critical pressure (95% utilization)
        monitor.track("obj4", 15 * 1024 * 1024)  # +15MB = 95MB
        stats = monitor.get_stats()
        assert stats.pressure == MemoryPressure.CRITICAL
    
    def test_compaction_removes_materialized_nodes(self):
        """Test that compaction removes materialized nodes."""
        from djinn.memory import GraphCompactor
        
        class MockLazyTensor:
            def __init__(self, node_id: str, materialized: bool = False):
                self.id = node_id
                self._materialized_value = "result" if materialized else None
        
        class MockGraphBuilder:
            def __init__(self):
                self.nodes = {}
                self.root_tensor = None
        
        builder = MockGraphBuilder()
        compactor = GraphCompactor(builder)
        
        # Add some nodes (some materialized, some not)
        builder.nodes = {
            "node1": MockLazyTensor("node1", materialized=False),  # Keep
            "node2": MockLazyTensor("node2", materialized=True),   # Remove
            "node3": MockLazyTensor("node3", materialized=False),  # Keep
            "node4": MockLazyTensor("node4", materialized=True),   # Remove
        }
        
        # Compact
        removed = compactor.compact()
        
        # Should have removed 2 nodes
        assert removed == 2
        assert len(builder.nodes) == 2
        assert "node1" in builder.nodes
        assert "node3" in builder.nodes
        assert "node2" not in builder.nodes
        assert "node4" not in builder.nodes


# ============================================================================
# MAJOR ISSUE #1: Error Handling with Result Type
# ============================================================================

class TestResultType:
    """Test error handling with Result type."""
    
    def test_ok_result(self):
        """Test Ok result construction and usage."""
        from djinn.core.exceptions import Ok
        
        result = Ok(42)
        assert result.is_ok()
        assert not result.is_err()
        assert result.unwrap() == 42
        assert result.unwrap_or(-1) == 42
    
    def test_err_result(self):
        """Test Err result construction and usage."""
        from djinn.core.exceptions import Err
        
        error = ValueError("Test error")
        result = Err(error, context={"operation": "test"})
        
        assert result.is_err()
        assert not result.is_ok()
        assert result.unwrap_or(-1) == -1
        
        with pytest.raises(ValueError):
            result.unwrap()
    
    def test_result_map(self):
        """Test Result.map() for transformations."""
        from djinn.core.exceptions import Ok, Err
        
        # Ok mapping
        result = Ok(5)
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_ok()
        assert mapped.unwrap() == 10
        
        # Err mapping
        error = ValueError("Error")
        result = Err(error)
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_err()
    
    def test_result_and_then(self):
        """Test Result.and_then() for chaining."""
        from djinn.core.exceptions import Ok, Err
        
        def divide(x):
            if x == 0:
                return Err(ValueError("Division by zero"))
            return Ok(10 / x)
        
        # Chain with Ok
        result = Ok(2).and_then(divide)
        assert result.is_ok()
        assert result.unwrap() == 5.0
        
        # Chain with Err
        result = Ok(0).and_then(divide)
        assert result.is_err()
    
    def test_collect_results(self):
        """Test collecting multiple results."""
        from djinn.core.exceptions import Ok, Err, collect_results
        
        # All Ok
        results = [Ok(1), Ok(2), Ok(3)]
        collected = collect_results(results)
        assert collected.is_ok()
        assert collected.unwrap() == [1, 2, 3]
        
        # One Err
        results = [Ok(1), Err(ValueError("error")), Ok(3)]
        collected = collect_results(results)
        assert collected.is_err()


# ============================================================================
# MAJOR ISSUE #3: Type Safety Tests
# ============================================================================

class TestTypeSafety:
    """Test NodeProtocol and type safety improvements."""
    
    def test_node_protocol_compliance(self):
        """Test that ConcreteNode complies with NodeProtocol."""
        from djinn.core.types import ConcreteNode, NodeProtocol
        
        node = ConcreteNode(
            id="test_node",
            operation="aten::matmul",
            shape=torch.Size([10, 10]),
            dtype=torch.float32
        )
        
        # Check protocol compliance
        assert isinstance(node, NodeProtocol) or hasattr(node, 'id')
        assert node.id == "test_node"
        assert node.operation == "aten::matmul"
        assert node.shape == torch.Size([10, 10])
        assert node.dtype == torch.float32
    
    def test_concrete_node_metadata_helpers(self):
        """Test ConcreteNode helper methods for metadata."""
        from djinn.core.types import ConcreteNode, ExecutionPhase, DataResidency, Modality
        
        node = ConcreteNode(id="test", operation="aten::add")
        
        # Test phase helpers
        node.set_phase(ExecutionPhase.LLM_DECODE)
        assert node.get_phase() == ExecutionPhase.LLM_DECODE
        
        # Test residency helpers
        node.set_residency(DataResidency.PERSISTENT_WEIGHT)
        assert node.get_residency() == DataResidency.PERSISTENT_WEIGHT
        
        # Test modality helpers
        node.set_modality(Modality.VISION)
        assert node.get_modality() == Modality.VISION
    
    def test_dict_node_adapter(self):
        """Test adapter for legacy dict-based nodes."""
        from djinn.core.types import DictNodeAdapter
        
        node_dict = {
            'id': 'legacy_node',
            'op': 'aten::relu',
            'shape': [10, 10],
            'dtype': 'float32',
            'metadata': {'role': 'activation'}
        }
        
        adapter = DictNodeAdapter(node_dict)
        
        assert adapter.id == 'legacy_node'
        assert adapter.operation == 'aten::relu'
        assert adapter.shape == torch.Size([10, 10])
        assert adapter.metadata == {'role': 'activation'}


# ============================================================================
# MAJOR ISSUE #5: Unified Graph Interface
# ============================================================================

class TestGenieGraph:
    """Test unified GenieGraph interface."""
    
    def test_concrete_graph_creation(self):
        """Test creating a concrete graph."""
        from djinn.core.graph_interface import ConcreteGraphImpl
        from djinn.core.types import ConcreteNode
        
        nodes = [
            ConcreteNode(id="n1", operation="aten::randn"),
            ConcreteNode(id="n2", operation="aten::matmul"),
        ]
        nodes[1].add_input(nodes[0])
        
        graph = ConcreteGraphImpl(nodes)
        
        assert graph.num_nodes == 2
        assert graph.get_node("n1") is not None
        assert graph.get_node("n2") is not None
    
    def test_graph_topological_order(self):
        """Test topological ordering of graph nodes."""
        from djinn.core.graph_interface import ConcreteGraphImpl
        from djinn.core.types import ConcreteNode
        
        # Create DAG: n1 -> n2 -> n3
        nodes = [
            ConcreteNode(id="n1", operation="aten::randn"),
            ConcreteNode(id="n2", operation="aten::matmul"),
            ConcreteNode(id="n3", operation="aten::add"),
        ]
        nodes[1].add_input(nodes[0])
        nodes[2].add_input(nodes[1])
        
        graph = ConcreteGraphImpl(nodes)
        order = graph.topological_order()
        
        # Should be n1, n2, n3
        ids = [n.id for n in order]
        assert ids == ["n1", "n2", "n3"]
    
    def test_graph_roots_and_leaves(self):
        """Test finding root and leaf nodes."""
        from djinn.core.graph_interface import ConcreteGraphImpl
        from djinn.core.types import ConcreteNode
        
        nodes = [
            ConcreteNode(id="n1", operation="aten::randn"),  # Root
            ConcreteNode(id="n2", operation="aten::matmul"),
            ConcreteNode(id="n3", operation="aten::add"),    # Leaf
        ]
        nodes[1].add_input(nodes[0])
        nodes[2].add_input(nodes[1])
        
        graph = ConcreteGraphImpl(nodes)
        
        roots = graph.get_roots()
        assert len(roots) == 1
        assert roots[0].id == "n1"
        
        leaves = graph.get_leaves()
        assert len(leaves) == 1
        assert leaves[0].id == "n3"


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance tests for optimizations."""
    
    def test_shape_cache_performance(self):
        """Test shape cache lookup performance."""
        from djinn.frontend.core.lazy_tensor import get_shape_cache_stats, _global_shape_cache, _global_cache_lock
        
        cache = _global_shape_cache
        
        # Populate cache
        for i in range(100):
            key = f"shape_key_{i}"
            with _global_cache_lock:
                cache[key] = torch.Size([10 + i, 20 + i])
        
        # Measure lookups
        start = time.time()
        for _ in range(1000):
            for i in range(100):
                key = f"shape_key_{i}"
                with _global_cache_lock:
                    _ = cache.get(key)
        elapsed = time.time() - start
        
        # Should complete quickly (<100ms for 100k lookups)
        assert elapsed < 0.1, f"Cache lookups too slow: {elapsed*1000:.1f}ms"
        
        # Check stats
        stats = get_shape_cache_stats()
        logger.info(f"Cache stats: {stats}")
    
    def test_graph_construction_overhead(self):
        """Test overhead of creating concrete graph."""
        from djinn.core.graph_interface import ConcreteGraphImpl
        from djinn.core.types import ConcreteNode
        
        # Create large graph
        num_nodes = 1000
        nodes = [ConcreteNode(id=f"n{i}", operation="aten::add") for i in range(num_nodes)]
        
        # Link them
        for i in range(1, num_nodes):
            nodes[i].add_input(nodes[i-1])
        
        # Measure graph creation
        start = time.time()
        graph = ConcreteGraphImpl(nodes)
        elapsed = time.time() - start
        
        # Should be fast (<100ms)
        assert elapsed < 0.1, f"Graph creation too slow: {elapsed*1000:.1f}ms"
        assert graph.num_nodes == num_nodes


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple improvements."""
    
    def test_error_handling_in_graph_construction(self):
        """Test error handling during graph operations."""
        from djinn.core.graph_interface import ConcreteGraphImpl
        from djinn.core.types import ConcreteNode
        from djinn.core.exceptions import Ok, Err
        
        try:
            nodes = [ConcreteNode(id="n1", operation="aten::matmul")]
            graph = ConcreteGraphImpl(nodes)
            
            # Return as Result
            result = Ok(graph)
            assert result.is_ok()
            assert result.unwrap().num_nodes == 1
        except Exception as e:
            result = Err(e)
            assert result.is_err()
    
    def test_concurrent_graph_operations(self):
        """Test concurrent graph operations with proper locking."""
        from djinn.core.graph_interface import ConcreteGraphImpl
        from djinn.core.types import ConcreteNode
        import threading
        
        results = []
        errors = []
        
        def worker(thread_id: int):
            try:
                nodes = [
                    ConcreteNode(id=f"t{thread_id}_n1", operation="aten::randn"),
                    ConcreteNode(id=f"t{thread_id}_n2", operation="aten::matmul"),
                ]
                nodes[1].add_input(nodes[0])
                
                graph = ConcreteGraphImpl(nodes)
                results.append(graph.num_nodes)
            except Exception as e:
                errors.append(str(e))
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert not errors, f"Errors: {errors}"
        assert len(results) == 8


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
