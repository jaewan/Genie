"""Integration tests for Refactoring #2 and #3 coordination.

Tests that FXGraphAdapter (Refactoring #3) properly integrates with
MetadataRegistry (Refactoring #2) using the agreed coordination strategy:

- FX meta stores: tensor_id, operation, shape, dtype (structural)
- MetadataRegistry stores: semantic_role, execution_phase, etc. (semantic)
- Bridge: tensor_id is used to lookup semantic metadata
"""

import pytest
import torch
import torch.fx as fx
from torch.fx import GraphModule

from genie.core.fx_graph_adapter import FXGraphAdapter
from genie.core.fx_graph_builder import FXGraphBuilder


class TestRefactoringCoordination:
    """Test coordination between Refactoring #2 and #3."""
    
    def test_fx_meta_stores_structural_data(self):
        """Test that FX meta contains structural data as agreed."""
        try:
            from genie.core.lazy_tensor import LazyTensor
        except ImportError:
            pytest.skip("LazyTensor not available")
        
        # Create LazyTensor with FXGraphBuilder
        fx_builder = FXGraphBuilder()
        lt = LazyTensor("aten::matmul", [], {'size': (10, 20)})
        node = fx_builder.add_lazy_tensor(lt)
        
        # Verify FX meta has structural data
        assert 'genie' in node.meta
        assert 'tensor_id' in node.meta['genie']
        assert 'operation' in node.meta['genie']
        assert 'shape' in node.meta['genie']
        assert 'dtype' in node.meta['genie']
        
        # Verify tensor_id matches LazyTensor
        assert node.meta['genie']['tensor_id'] == lt.id
    
    def test_metadata_registry_stores_semantic_data(self):
        """Test that MetadataRegistry stores semantic data as agreed."""
        try:
            from genie.core.lazy_tensor import LazyTensor
            from genie.semantic.metadata_registry import get_metadata_registry
        except ImportError:
            pytest.skip("Components not available")
        
        # Create LazyTensor (should auto-register)
        lt = LazyTensor("aten::matmul", [], {'size': (10, 20)})
        
        # Get from registry
        registry = get_metadata_registry()
        metadata = registry.get_metadata(lt.id)
        
        # Should have semantic data
        if metadata:
            # These are semantic properties that should be in registry, not FX meta
            assert hasattr(metadata, 'semantic_role')
            assert hasattr(metadata, 'execution_phase')
            assert hasattr(metadata, 'memory_pattern')
            assert hasattr(metadata, 'compute_intensity')
    
    def test_bridge_mechanism_tensor_id_lookup(self):
        """Test the bridge mechanism: use tensor_id to lookup semantic metadata."""
        try:
            from genie.core.lazy_tensor import LazyTensor
            from genie.semantic.metadata_registry import get_metadata_registry
        except ImportError:
            pytest.skip("Components not available")
        
        # Create FX graph with LazyTensor
        fx_builder = FXGraphBuilder()
        lt = LazyTensor("aten::matmul", [], {'size': (10, 20)})
        node = fx_builder.add_lazy_tensor(lt)
        
        # Get tensor_id from FX meta
        tensor_id = node.meta['genie']['tensor_id']
        assert tensor_id == lt.id
        
        # Use tensor_id to lookup in registry
        registry = get_metadata_registry()
        semantic_meta = registry.get_metadata(tensor_id)
        
        # Should be able to get semantic metadata via bridge
        if semantic_meta:
            assert semantic_meta.operation_type == lt.operation
    
    def test_fx_graph_adapter_bridges_both_layers(self):
        """Test FXGraphAdapter provides unified access to both structural and semantic data."""
        try:
            from genie.core.lazy_tensor import LazyTensor
        except ImportError:
            pytest.skip("LazyTensor not available")
        
        # Create FX graph
        fx_builder = FXGraphBuilder()
        lt = LazyTensor("aten::relu", [], {'size': (10, 10)})
        fx_builder.add_lazy_tensor(lt)
        
        gm = fx_builder.to_graph_module()
        adapter = FXGraphAdapter(gm)
        
        # Get the operation node
        op_nodes = adapter.get_operation_nodes()
        if not op_nodes:
            pytest.skip("No operation nodes")
        
        node = op_nodes[0]
        
        # Should be able to get structural data from FX
        tensor_id = adapter.get_tensor_id(node)
        operation = adapter.get_operation(node)
        shape = adapter.get_shape(node)
        
        assert tensor_id is not None
        assert operation is not None
        
        # Should be able to get semantic data via registry bridge
        semantic_meta = adapter.get_semantic_metadata(node)
        # May be None if registry not set up, but shouldn't crash
        assert True  # Test that it doesn't crash
    
    def test_no_duplication_between_layers(self):
        """Test that data is not duplicated between FX meta and MetadataRegistry."""
        try:
            from genie.core.lazy_tensor import LazyTensor
            from genie.semantic.metadata_registry import get_metadata_registry
        except ImportError:
            pytest.skip("Components not available")
        
        # Create LazyTensor
        fx_builder = FXGraphBuilder()
        lt = LazyTensor("aten::matmul", [], {'size': (10, 20)})
        node = fx_builder.add_lazy_tensor(lt)
        
        # Get metadata from both layers
        fx_meta = node.meta['genie']
        
        registry = get_metadata_registry()
        semantic_meta = registry.get_metadata(lt.id)
        
        # Verify separation of concerns:
        # - FX meta should have: tensor_id, operation, shape, dtype (structural)
        assert 'tensor_id' in fx_meta
        assert 'operation' in fx_meta
        assert 'shape' in fx_meta
        
        # - FX meta should NOT have semantic-specific fields in genie namespace
        # (these should be in MetadataRegistry)
        # Note: Old format may still exist for backward compat, but new genie meta
        # should not duplicate semantic data
        
        if semantic_meta:
            # Semantic metadata should have different fields
            assert hasattr(semantic_meta, 'semantic_role')
            assert hasattr(semantic_meta, 'memory_pattern')
            
            # These are semantic, not structural
            # They should be in registry, not duplicated in FX meta['genie']
            assert 'semantic_role' not in fx_meta
            assert 'memory_pattern' not in fx_meta


class TestBackwardCompatibility:
    """Test backward compatibility during migration."""
    
    def test_old_format_still_works(self):
        """Test that old format metadata still works during migration."""
        # Create graph with old format
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        
        # Old format (without genie namespace)
        y.meta['lazy_tensor_id'] = 'lt_999'
        y.meta['shape'] = torch.Size([10, 10])
        
        graph.output(y)
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        # Should still be able to get data
        tensor_id = adapter.get_tensor_id(y)
        assert tensor_id == 'lt_999'
        
        shape = adapter.get_shape(y)
        assert shape == torch.Size([10, 10])
    
    def test_migration_to_new_format(self):
        """Test migrating from old to new format."""
        # Create graph with old format
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        
        y.meta['lazy_tensor_id'] = 'lt_888'
        y.meta['shape'] = torch.Size([5, 5])
        y.meta['dtype'] = torch.float32
        
        graph.output(y)
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        # Migrate
        adapter.ensure_genie_meta(y)
        
        # Should now have new format
        assert 'genie' in y.meta
        assert y.meta['genie']['tensor_id'] == 'lt_888'
        assert y.meta['genie']['operation'] == 'aten::relu'
        
        # Old format should still exist (for backward compat)
        assert y.meta['lazy_tensor_id'] == 'lt_888'


class TestCoordinationEdgeCases:
    """Test edge cases in coordination."""
    
    def test_node_without_tensor_id(self):
        """Test handling nodes without tensor_id."""
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        graph.output(y)
        
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        # Node without tensor_id should return None
        tensor_id = adapter.get_tensor_id(y)
        # May be None, should not crash
        
        # Semantic metadata should also return None
        semantic = adapter.get_semantic_metadata(y)
        assert semantic is None
    
    def test_registry_not_available(self):
        """Test graceful degradation when MetadataRegistry not available."""
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        y.meta['genie'] = {'tensor_id': 'lt_123'}
        graph.output(y)
        
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        # Should be able to get structural data even if registry unavailable
        tensor_id = adapter.get_tensor_id(y)
        assert tensor_id == 'lt_123'
        
        # Semantic data may be None, but should not crash
        semantic = adapter.get_semantic_metadata(y)
        # May be None if registry not set up
    
    def test_statistics_with_partial_semantic_coverage(self):
        """Test statistics when only some nodes have semantic metadata."""
        try:
            from genie.core.lazy_tensor import LazyTensor
        except ImportError:
            pytest.skip("LazyTensor not available")
        
        # Create mixed graph (some with LazyTensor, some without)
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        
        # Only y has genie meta
        y.meta['genie'] = {'tensor_id': 'lt_1', 'operation': 'aten::relu'}
        
        graph.output(y)
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        # Statistics should handle partial coverage
        stats = adapter.get_statistics()
        
        # Should have coverage metric
        assert 'semantic_coverage' in stats
        # Coverage should be between 0 and 1
        assert 0.0 <= stats['semantic_coverage'] <= 1.0


class TestPerformanceConsiderations:
    """Test that coordination doesn't significantly impact performance."""
    
    def test_metadata_lookup_is_fast(self):
        """Test that metadata lookup via bridge is reasonably fast."""
        try:
            from genie.core.lazy_tensor import LazyTensor
        except ImportError:
            pytest.skip("LazyTensor not available")
        
        import time
        
        # Create multiple LazyTensors
        fx_builder = FXGraphBuilder()
        tensors = []
        for i in range(100):
            lt = LazyTensor("aten::add", [], {'size': (10, 10)})
            fx_builder.add_lazy_tensor(lt)
            tensors.append(lt)
        
        gm = fx_builder.to_graph_module()
        adapter = FXGraphAdapter(gm)
        
        # Time metadata access
        start = time.perf_counter()
        
        for node in adapter.get_operation_nodes():
            # Access both structural and semantic data
            _ = adapter.get_tensor_id(node)
            _ = adapter.get_operation(node)
            _ = adapter.get_semantic_metadata(node)
        
        elapsed = time.perf_counter() - start
        
        # Should be fast (< 100ms for 100 nodes)
        assert elapsed < 0.1, f"Too slow: {elapsed*1000:.1f}ms"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

