"""Tests for FXGraphAdapter (Refactoring #3).

Tests the unified graph representation and coordination with Refactoring #2's
MetadataRegistry.
"""

import pytest
import torch
import torch.fx as fx
from torch.fx import GraphModule

from genie.core.fx_graph_adapter import FXGraphAdapter, get_current_fx_adapter
from genie.core.fx_graph_builder import FXGraphBuilder


class TestFXGraphAdapterBasics:
    """Test basic FXGraphAdapter functionality."""
    
    def test_adapter_creation(self):
        """Test creating adapter with GraphModule."""
        # Create simple FX graph
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        graph.output(y)
        
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        assert adapter.graph_module == gm
        assert len(adapter.get_all_nodes()) == 3  # placeholder, relu, output
    
    def test_adapter_empty(self):
        """Test adapter without GraphModule."""
        adapter = FXGraphAdapter()
        
        with pytest.raises(ValueError, match="No GraphModule"):
            _ = adapter.graph
    
    def test_get_all_nodes(self):
        """Test getting all nodes."""
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        z = graph.call_function(torch.sigmoid, (y,))
        graph.output(z)
        
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        nodes = adapter.get_all_nodes()
        assert len(nodes) == 4  # placeholder, relu, sigmoid, output
    
    def test_get_operation_nodes(self):
        """Test getting only operation nodes."""
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        z = graph.call_function(torch.sigmoid, (y,))
        graph.output(z)
        
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        op_nodes = adapter.get_operation_nodes()
        assert len(op_nodes) == 2  # relu, sigmoid (not placeholder or output)


class TestFXGraphAdapterMeta:
    """Test metadata access methods."""
    
    def test_get_tensor_id_new_format(self):
        """Test getting tensor_id from new genie meta format."""
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        
        # Add new format metadata
        y.meta['genie'] = {'tensor_id': 'lt_123'}
        
        graph.output(y)
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        tensor_id = adapter.get_tensor_id(y)
        assert tensor_id == 'lt_123'
    
    def test_get_tensor_id_old_format(self):
        """Test backward compatibility with old format."""
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        
        # Add old format metadata
        y.meta['lazy_tensor_id'] = 'lt_456'
        
        graph.output(y)
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        tensor_id = adapter.get_tensor_id(y)
        assert tensor_id == 'lt_456'
    
    def test_get_operation_new_format(self):
        """Test getting operation from new format."""
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        
        y.meta['genie'] = {'operation': 'aten::relu'}
        
        graph.output(y)
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        operation = adapter.get_operation(y)
        assert operation == 'aten::relu'
    
    def test_get_operation_fallback(self):
        """Test operation inference from node target."""
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        
        graph.output(y)
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        operation = adapter.get_operation(y)
        assert operation == 'aten::relu'
    
    def test_get_shape(self):
        """Test getting shape from meta."""
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        
        y.meta['genie'] = {'shape': torch.Size([10, 20])}
        
        graph.output(y)
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        shape = adapter.get_shape(y)
        assert shape == torch.Size([10, 20])
    
    def test_get_dtype(self):
        """Test getting dtype from meta."""
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        
        y.meta['genie'] = {'dtype': torch.float32}
        
        graph.output(y)
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        dtype = adapter.get_dtype(y)
        assert dtype == torch.float32


class TestFXGraphAdapterSemanticIntegration:
    """Test integration with MetadataRegistry (Refactoring #2)."""
    
    def test_metadata_registry_available(self):
        """Test that MetadataRegistry can be accessed."""
        adapter = FXGraphAdapter()
        
        # Should be able to get registry (may be None if not installed)
        registry = adapter.metadata_registry
        # Just check it doesn't crash
        assert True
    
    def test_get_semantic_metadata_no_tensor_id(self):
        """Test semantic metadata access without tensor_id."""
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        graph.output(y)
        
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        # Should return None if no tensor_id
        semantic = adapter.get_semantic_metadata(y)
        assert semantic is None
    
    def test_has_semantic_metadata(self):
        """Test checking for semantic metadata."""
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        graph.output(y)
        
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        # Should be False without tensor_id
        has_meta = adapter.has_semantic_metadata(y)
        assert has_meta is False


class TestFXGraphAdapterAnalysis:
    """Test graph analysis methods."""
    
    def test_get_nodes_by_operation(self):
        """Test filtering nodes by operation."""
        graph = fx.Graph()
        x = graph.placeholder('x')
        y1 = graph.call_function(torch.relu, (x,))
        y2 = graph.call_function(torch.relu, (y1,))
        z = graph.call_function(torch.sigmoid, (y2,))
        graph.output(z)
        
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        relu_nodes = adapter.get_nodes_by_operation('aten::relu')
        assert len(relu_nodes) == 2
        
        sigmoid_nodes = adapter.get_nodes_by_operation('aten::sigmoid')
        assert len(sigmoid_nodes) == 1
    
    def test_get_execution_order(self):
        """Test getting execution order."""
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        z = graph.call_function(torch.sigmoid, (y,))
        graph.output(z)
        
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        order = adapter.get_execution_order()
        assert len(order) == 2  # relu, sigmoid
        
        # Check order is correct (relu before sigmoid)
        ops = [adapter.get_operation(node) for node in order]
        assert ops == ['aten::relu', 'aten::sigmoid']
    
    def test_get_dependencies(self):
        """Test getting node dependencies."""
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        z = graph.call_function(torch.add, (y, x))  # Depends on both y and x
        graph.output(z)
        
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        deps = adapter.get_dependencies(z)
        assert len(deps) == 2
        assert x in deps
        assert y in deps
    
    def test_get_users(self):
        """Test getting node users."""
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        z1 = graph.call_function(torch.sigmoid, (y,))
        z2 = graph.call_function(torch.tanh, (y,))
        out = graph.call_function(torch.add, (z1, z2))
        graph.output(out)
        
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        # y is used by both z1 and z2
        users = adapter.get_users(y)
        assert len(users) == 2
    
    def test_get_statistics(self):
        """Test graph statistics."""
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        z = graph.call_function(torch.relu, (y,))
        w = graph.call_function(torch.sigmoid, (z,))
        graph.output(w)
        
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        stats = adapter.get_statistics()
        
        assert stats['num_nodes'] == 5  # placeholder, relu, relu, sigmoid, output
        assert stats['num_operations'] == 3  # relu, relu, sigmoid
        assert stats['operation_counts']['aten::relu'] == 2
        assert stats['operation_counts']['aten::sigmoid'] == 1


class TestFXGraphAdapterMigration:
    """Test migration helpers."""
    
    def test_ensure_genie_meta_new_format(self):
        """Test ensure_genie_meta with already-new format."""
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        
        # Already has new format
        y.meta['genie'] = {'tensor_id': 'lt_123'}
        
        graph.output(y)
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        # Should not change
        adapter.ensure_genie_meta(y)
        assert y.meta['genie']['tensor_id'] == 'lt_123'
    
    def test_ensure_genie_meta_from_old_format(self):
        """Test migrating from old format."""
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        
        # Old format
        y.meta['lazy_tensor_id'] = 'lt_456'
        y.meta['shape'] = torch.Size([10, 10])
        y.meta['dtype'] = torch.float32
        
        graph.output(y)
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        # Migrate
        adapter.ensure_genie_meta(y)
        
        # Should now have new format
        assert 'genie' in y.meta
        assert y.meta['genie']['tensor_id'] == 'lt_456'
        assert y.meta['genie']['operation'] == 'aten::relu'
    
    def test_migrate_all_nodes(self):
        """Test migrating all nodes in graph."""
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        z = graph.call_function(torch.sigmoid, (y,))
        
        # Old format
        y.meta['lazy_tensor_id'] = 'lt_1'
        z.meta['lazy_tensor_id'] = 'lt_2'
        
        graph.output(z)
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        # Migrate
        migrated = adapter.migrate_all_nodes()
        
        # Both operation nodes should be migrated
        assert migrated == 2
        assert 'genie' in y.meta
        assert 'genie' in z.meta


class TestFXGraphAdapterIntegrationWithFXGraphBuilder:
    """Test integration with FXGraphBuilder."""
    
    def test_fx_builder_creates_new_format(self):
        """Test that FXGraphBuilder creates new genie meta format."""
        try:
            from genie.core.lazy_tensor import LazyTensor
        except ImportError:
            pytest.skip("LazyTensor not available")
        
        fx_builder = FXGraphBuilder()
        
        # Create simple LazyTensor
        lt = LazyTensor("aten::randn", [], {'size': (10, 10)})
        
        # Add to FX builder
        node = fx_builder.add_lazy_tensor(lt)
        
        # Should have new format
        assert 'genie' in node.meta
        assert node.meta['genie']['tensor_id'] == lt.id
        assert node.meta['genie']['operation'] == 'aten::randn'
    
    def test_adapter_from_fx_builder(self):
        """Test creating adapter from FXGraphBuilder."""
        try:
            from genie.core.lazy_tensor import LazyTensor
        except ImportError:
            pytest.skip("LazyTensor not available")
        
        fx_builder = FXGraphBuilder()
        
        # Create chain of operations
        x = LazyTensor("aten::randn", [], {'size': (10, 10)})
        y = LazyTensor("aten::relu", [x])
        
        fx_builder.add_lazy_tensor(x)
        fx_builder.add_lazy_tensor(y)
        
        # Convert to GraphModule and adapter
        gm = fx_builder.to_graph_module()
        adapter = FXGraphAdapter(gm)
        
        # Check stats
        stats = adapter.get_statistics()
        assert stats['num_operations'] >= 2


class TestFXGraphAdapterRepresentation:
    """Test string representation."""
    
    def test_repr(self):
        """Test __repr__ output."""
        graph = fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(torch.relu, (x,))
        graph.output(y)
        
        gm = GraphModule(torch.nn.Module(), graph)
        adapter = FXGraphAdapter(gm)
        
        repr_str = repr(adapter)
        assert 'FXGraphAdapter' in repr_str
        assert 'nodes=' in repr_str
        assert 'operations=' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

