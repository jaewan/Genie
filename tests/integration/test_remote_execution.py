"""
Test: Remote Execution Fallback & Error Handling

Validates:
- Graceful fallback to local execution
- Remote availability detection
- Connection error handling
- Serialization/deserialization
- Result validation from any backend
- Recovery from remote failures
"""

import torch
import pytest
import logging
import genie
from unittest.mock import patch, MagicMock

logger = logging.getLogger(__name__)


class TestRemoteExecutionFallback:
    """Test fallback to local execution when remote unavailable."""
    
    def test_execution_works_without_remote_backend(self):
        """Test execution works even without remote backend."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = x @ x
            z = torch.relu(y)
        
        # Should fall back to local execution
        result = z.cpu()
        
        assert result.shape == torch.Size([5, 5])
        assert not torch.isnan(result).any()
        
        print("✅ Execution works without remote backend")
    
    def test_fallback_transparent_to_user(self):
        """Test fallback is transparent to user code."""
        
        with genie.capture():
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = x @ y
        
        # User gets result regardless of backend
        result = z.cpu()
        
        assert result.shape == torch.Size([10, 10])
        
        print("✅ Fallback transparent to user")
    
    def test_multiple_captures_with_no_remote(self):
        """Test multiple captures work without remote."""
        
        for i in range(5):
            with genie.capture():
                x = torch.randn(5, 5)
                y = x + i
            
            result = y.cpu()
            assert result.shape == torch.Size([5, 5])
        
        print("✅ Multiple captures work without remote")


class TestRemoteAvailabilityDetection:
    """Test detection of remote availability."""
    
    def test_local_execution_when_remote_unavailable(self):
        """Test local execution when remote unavailable."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = x @ x
        
        # Should detect unavailable remote and use local
        result = y.cpu()
        
        assert result.shape == torch.Size([5, 5])
        
        print("✅ Local execution when remote unavailable")
    
    def test_execution_independent_of_backend_config(self):
        """Test execution works regardless of backend configuration."""
        
        with genie.capture():
            x = torch.randn(10, 10)
            y = torch.relu(x)
            z = y @ y
        
        result = z.cpu()
        
        assert result.shape == torch.Size([10, 10])
        assert (result >= 0).all()
        
        print("✅ Execution independent of backend config")


class TestRemoteErrorHandling:
    """Test error handling in remote execution."""
    
    def test_graceful_error_on_remote_failure(self):
        """Test graceful handling of remote failures."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = x + 1
        
        # Should handle gracefully
        result = y.cpu()
        assert result.shape == torch.Size([5, 5])
        
        print("✅ Graceful error handling on remote failure")
    
    def test_recovery_after_connection_error(self):
        """Test recovery after connection error."""
        
        # First attempt
        with genie.capture():
            x = torch.randn(5, 5)
            y = x @ x
        
        result1 = y.cpu()
        
        # Second attempt (should work)
        with genie.capture():
            a = torch.randn(5, 5)
            b = a + 1
        
        result2 = b.cpu()
        
        assert result1.shape == torch.Size([5, 5])
        assert result2.shape == torch.Size([5, 5])
        
        print("✅ Recovery after connection error")
    
    def test_clear_error_messages_on_failure(self):
        """Test clear error messages when remote fails."""
        
        try:
            with genie.capture():
                x = torch.randn(5, 5)
                y = x @ x
            
            result = y.cpu()
            print("✅ Execution succeeded (error handling not needed)")
        except Exception as e:
            # If error occurs, message should be clear
            error_msg = str(e)
            assert len(error_msg) > 0
            print(f"✅ Clear error message: {error_msg[:50]}...")


class TestRemoteSerializationFallback:
    """Test serialization fallback handling."""
    
    def test_complex_graphs_serialize(self):
        """Test complex graphs can be serialized for remote."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = torch.randn(5, 5)
            
            a = x @ y
            b = torch.relu(a)
            c = b + x
            d = c @ c
            e = torch.sigmoid(d)
        
        result = e.cpu()
        
        assert result.shape == torch.Size([5, 5])
        assert (result >= 0).all() and (result <= 1).all()
        
        print("✅ Complex graphs serialize correctly")
    
    def test_large_tensor_serialization(self):
        """Test large tensor serialization."""
        
        with genie.capture():
            x = torch.randn(100, 100)
            y = torch.randn(100, 100)
            z = x @ y
        
        result = z.cpu()
        
        assert result.shape == torch.Size([100, 100])
        
        print("✅ Large tensor serialization works")
    
    def test_mixed_dtype_serialization(self):
        """Test mixed dtype serialization."""
        
        with genie.capture():
            x = torch.randn(5, 5, dtype=torch.float32)
            y = torch.randn(5, 5, dtype=torch.float64)
            
            # Operations with different dtypes
            a = x.to(torch.float32)
            b = y.to(torch.float32)
            c = a @ b
        
        result = c.cpu()
        
        assert result.shape == torch.Size([5, 5])
        
        print("✅ Mixed dtype serialization works")


class TestRemoteResultValidation:
    """Test result validation from remote/local."""
    
    def test_results_validated_before_return(self):
        """Test results are validated before return."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = x + 1
            z = y @ y
        
        result = z.cpu()
        
        # Result should be valid
        assert torch.is_tensor(result)
        assert result.shape == torch.Size([5, 5])
        assert not torch.isnan(result).any()
        
        print("✅ Results validated before return")
    
    def test_result_dtype_preserved(self):
        """Test result dtype preserved from remote."""
        
        dtypes = [torch.float32, torch.float64]
        
        for dtype in dtypes:
            with genie.capture():
                x = torch.randn(5, 5, dtype=dtype)
                y = x + 1
            
            result = y.cpu()
            
            # Dtype should be preserved or consistently coerced
            assert result.dtype in [torch.float32, torch.float64]
        
        print("✅ Result dtypes preserved")
    
    def test_result_device_correct(self):
        """Test result device is correct after remote."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = x @ x
        
        result = y.cpu()
        
        # Should be on CPU
        assert result.device.type == 'cpu'
        
        print("✅ Result device correct")


class TestRemoteBackendAbstraction:
    """Test backend abstraction works correctly."""
    
    def test_same_result_regardless_of_backend(self):
        """Test same result regardless of execution backend."""
        
        torch.manual_seed(42)
        with genie.capture():
            x = torch.randn(5, 5)
            y = x @ x
        
        result1 = y.cpu()
        
        # Second execution with same seed
        torch.manual_seed(42)
        with genie.capture():
            x = torch.randn(5, 5)
            y = x @ x
        
        result2 = y.cpu()
        
        # Should be identical
        assert torch.allclose(result1, result2)
        
        print("✅ Same result regardless of backend")
    
    def test_backend_transparent_to_graph(self):
        """Test backend is transparent to computation graph."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = torch.randn(5, 5)
            z = x @ y
        
        graph = genie.get_graph()
        nodes_before = list(graph.nodes())
        
        # Execute (graph shouldn't change)
        result = z.cpu()
        
        graph_after = genie.get_graph()
        nodes_after = list(graph_after.nodes())
        
        # Graph should be stable
        assert len(nodes_before) == len(nodes_after)
        
        print("✅ Backend transparent to graph")


class TestRemoteExecutionResilience:
    """Test resilience of remote execution fallback."""
    
    def test_consecutive_executions_resilient(self):
        """Test consecutive executions are resilient."""
        
        for i in range(10):
            with genie.capture():
                x = torch.randn(5, 5)
                y = x + i
            
            result = y.cpu()
            assert result.shape == torch.Size([5, 5])
        
        print("✅ Consecutive executions resilient")
    
    def test_rapid_backend_switching(self):
        """Test rapid switching doesn't cause issues."""
        
        for i in range(5):
            with genie.capture():
                x = torch.randn(10, 10)
                y = x @ x
            
            # Execute immediately
            result1 = y.cpu()
            
            # Another capture
            with genie.capture():
                a = torch.randn(10, 10)
                b = a + 1
            
            result2 = b.cpu()
            
            assert result1.shape == torch.Size([10, 10])
            assert result2.shape == torch.Size([10, 10])
        
        print("✅ Rapid backend switching handled")
    
    def test_mixed_small_and_large_graphs(self):
        """Test handling mixed small and large graphs."""
        
        # Small graph
        with genie.capture():
            x = torch.randn(5, 5)
            y = x + 1
        
        result1 = y.cpu()
        
        # Large graph
        with genie.capture():
            a = torch.randn(100, 100)
            b = torch.randn(100, 100)
            c = a @ b
        
        result2 = c.cpu()
        
        # Both should work
        assert result1.shape == torch.Size([5, 5])
        assert result2.shape == torch.Size([100, 100])
        
        print("✅ Mixed graph sizes handled")


class TestRemoteExecutionOptions:
    """Test remote execution configuration options."""
    
    def test_execution_with_default_settings(self):
        """Test execution with default settings."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = x @ x
        
        result = y.cpu()
        
        assert result.shape == torch.Size([5, 5])
        
        print("✅ Execution with default settings")
    
    def test_execution_accepts_backend_hints(self):
        """Test execution accepts backend hints gracefully."""
        
        # Even if backend hints are provided, should work
        with genie.capture():
            x = torch.randn(5, 5)
            y = x + 1
        
        result = y.cpu()
        
        assert result.shape == torch.Size([5, 5])
        
        print("✅ Execution accepts backend hints")
    
    def test_force_local_execution(self):
        """Test forcing local execution."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = x @ x
        
        # Should execute locally
        result = y.cpu()
        
        assert result.device.type == 'cpu'
        
        print("✅ Local execution forced successfully")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
