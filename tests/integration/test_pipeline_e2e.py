"""
Test: Full pipeline integration

Validates the complete flow:
1. Capture operations
2. Build graph
3. Semantic analysis
4. Scheduling
5. Execution

This is the most important integration test.
"""

import torch
import torch.nn as nn
import pytest
import logging
import genie

logger = logging.getLogger(__name__)


class TestPipelineIntegration:
    """Test complete pipeline integration."""
    
    def test_simple_model_full_pipeline(self):
        """Test complete pipeline on simple operations."""
        
        # Step 2: Capture - use simple operations instead of model
        with genie.capture():
            x = torch.randn(32, 128)
            y = torch.randn(128, 256)
            z = x @ y  # matmul
            output = torch.relu(z)
        
        # Step 3: Get graph
        graph = genie.get_graph()
        assert graph is not None, "Graph is None!"
        
        nodes = list(graph.nodes())
        assert len(nodes) > 0, "Graph has no nodes!"
        
        print(f"✅ Step 1-2: Captured {len(nodes)} operations")
        
        # Step 4: Semantic analysis
        try:
            annotated = genie.annotate_graph(graph)
        except Exception as e:
            pytest.fail(f"Semantic analysis failed: {e}")
        
        assert annotated is not None, "Annotated graph is None!"
        assert annotated.costs['total_compute_flops'] > 0, \
            "Cost estimation failed!"
        
        print(f"✅ Step 3: Semantic analysis completed")
        print(f"   Total FLOPs: {annotated.costs['total_compute_flops']:.2e}")
        
        # Step 5: Scheduling
        try:
            schedule = genie.schedule(annotated.base_graph)
        except Exception as e:
            pytest.fail(f"Scheduling failed: {e}")
        
        assert schedule is not None, "Schedule is None!"
        assert schedule.total_stages > 0, "Schedule has no stages!"
        
        print(f"✅ Step 4: Scheduling completed")
        print(f"   Stages: {schedule.total_stages}")
        
        # Step 6: Execution (local)
        try:
            result = output.cpu()
        except Exception as e:
            pytest.fail(f"Execution failed: {e}")
        
        # Validate result
        assert result.shape == torch.Size([32, 256]), \
            f"Wrong output shape: {result.shape}"
        
        assert not torch.isnan(result).any(), \
            "Output contains NaN!"
        
        assert not torch.isinf(result).any(), \
            "Output contains Inf!"
        
        print(f"✅ Step 5: Execution completed")
        print(f"   Output shape: {result.shape}")
        print(f"   Output range: [{result.min():.2f}, {result.max():.2f}]")
        
        print(f"\n🎉 FULL PIPELINE SUCCESS")
    
    def test_pipeline_with_control_flow(self):
        """Test pipeline with dynamic control flow (FX fallback)."""
        
        class DynamicModel(nn.Module):
            def forward(self, x):
                if x.sum() > 0:  # Data-dependent branch
                    return torch.relu(x)
                else:
                    return torch.tanh(x)
        
        model = DynamicModel()
        
        with genie.capture():
            x = torch.randn(10, 10)
            output = model(x)
        
        # Should use LazyDAG fallback
        graph = genie.get_graph()
        assert graph.backend_type == 'lazy_dag', \
            f"Expected lazy_dag for control flow, got {graph.backend_type}"
        
        # Pipeline should still work
        annotated = genie.annotate_graph(graph)
        assert annotated is not None
        
        schedule = genie.schedule(annotated.base_graph)
        assert schedule is not None
        
        result = output.cpu()
        assert result.shape == torch.Size([10, 10])
        
        print("✅ Pipeline handles control flow with LazyDAG fallback")
    
    def test_metadata_propagation(self):
        """Test metadata flows through pipeline stages."""
        
        with genie.capture():
            x = torch.randn(10, 20)
            y = torch.randn(20, 30)
            z = x @ y
        
        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)
        
        # Find matmul node
        matmul_nodes = [n for n in graph.nodes() 
                       if 'matmul' in n.operation.lower()]
        
        assert len(matmul_nodes) > 0, "Matmul node not captured!"
        
        matmul_node = matmul_nodes[0]
        
        # Check metadata exists after annotation
        metadata = annotated.get_metadata(matmul_node.id)
        assert metadata is not None, "No metadata for matmul node!"
        
        # Check metadata has expected fields
        assert metadata.compute_flops > 0, \
            "No compute cost in metadata!"
        assert metadata.memory_bytes > 0, \
            "No memory cost in metadata!"
        assert metadata.operational_intensity > 0, \
            "No operational intensity!"
        
        print("✅ Metadata propagates correctly through pipeline")
        print(f"   Matmul FLOPs: {metadata.compute_flops:.2e}")
        print(f"   Memory: {metadata.memory_bytes:.2e} bytes")
        print(f"   Intensity: {metadata.operational_intensity:.2f} FLOPs/byte")
    
    def test_multiple_captures_independent(self):
        """Test multiple capture contexts are independent."""
        
        # First capture
        with genie.capture():
            x1 = torch.randn(5, 5)
            y1 = x1 + 1
        
        graph1 = genie.get_graph()
        nodes1 = list(graph1.nodes())
        ops1 = [n.operation for n in nodes1]
        
        # Second capture (should be independent)
        with genie.capture():
            x2 = torch.randn(10, 10)
            y2 = x2 @ x2
        
        graph2 = genie.get_graph()
        nodes2 = list(graph2.nodes())
        ops2 = [n.operation for n in nodes2]
        
        # Operations should be different
        has_add_in_1 = any('add' in op for op in ops1)
        has_matmul_in_2 = any('matmul' in op or 'mm' in op for op in ops2)
        
        assert has_add_in_1, "First graph missing add operation!"
        assert has_matmul_in_2, "Second graph missing matmul operation!"
        
        print("✅ Multiple captures are independent")
    
    def test_nested_captures_handled(self):
        """Test nested capture contexts."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            
            # Nested capture
            with genie.capture():
                y = torch.randn(5, 5)
                z = y @ y
            
            w = x + 1
        
        # Should complete without error
        graph = genie.get_graph()
        assert graph is not None
        
        result = w.cpu()
        assert result.shape == torch.Size([5, 5])
        
        print("✅ Nested captures handled")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
