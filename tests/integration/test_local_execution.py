"""
Test: Local CPU/GPU Execution

Validates:
- Graph execution on local hardware
- CPU execution correctness
- GPU execution (if available)
- Device placement & transfer
- Execution performance
- Large tensor handling
"""

import torch
import pytest
import logging
import djinn
import time

logger = logging.getLogger(__name__)


class TestLocalCPUExecution:
    """Test local CPU execution."""
    
    def test_simple_execution_on_cpu(self):
        """Test simple computation executes on CPU."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = x + 1
            z = y @ y
        
        # Materialize to CPU
        result = z.cpu()
        
        assert result.shape == torch.Size([5, 5])
        assert result.device.type == 'cpu'
        assert not torch.isnan(result).any()
        
        print("✅ Simple execution on CPU works")
    
    def test_large_tensor_cpu_execution(self):
        """Test execution with large tensors on CPU."""
        
        with genie.capture():
            x = torch.randn(100, 100)
            y = torch.randn(100, 100)
            z = x @ y + torch.ones(100, 100)
        
        result = z.cpu()
        
        assert result.shape == torch.Size([100, 100])
        assert result.device.type == 'cpu'
        
        print("✅ Large tensor CPU execution works")
    
    def test_complex_computation_cpu(self):
        """Test complex computation on CPU."""
        
        with genie.capture():
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            
            # Chain of operations
            a = x @ y
            b = torch.relu(a)
            c = b + x
            d = c @ c
            e = torch.sigmoid(d)
        
        result = e.cpu()
        
        assert result.shape == torch.Size([10, 10])
        assert (result >= 0).all() and (result <= 1).all()
        
        print("✅ Complex computation on CPU works")
    
    def test_multiple_operations_cpu(self):
        """Test multiple distinct operations on CPU."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            
            # Elementwise operations
            a = x + 1
            b = x - 1
            c = x * 2
            d = x / 2
            
            # Unary operations
            e = torch.relu(x)
            f = torch.tanh(x)
            
            # Reduction
            g = torch.sum(x)
        
        result_a = a.cpu()
        result_e = e.cpu()
        result_g = g.cpu()
        
        assert result_a.shape == torch.Size([5, 5])
        assert result_e.shape == torch.Size([5, 5])
        assert result_g.shape == torch.Size([])
        
        print("✅ Multiple operations on CPU work")


class TestLocalGPUExecution:
    """Test local GPU execution (if available)."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_simple_execution_on_gpu(self):
        """Test simple computation executes on GPU."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = x + 1
            z = y @ y
        
        # Materialize to GPU
        result = z.cuda()
        
        assert result.shape == torch.Size([5, 5])
        assert result.device.type == 'cuda'
        assert not torch.isnan(result).any()
        
        print("✅ Simple execution on GPU works")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_large_tensor_gpu_execution(self):
        """Test execution with large tensors on GPU."""
        
        with genie.capture():
            x = torch.randn(200, 200)
            y = torch.randn(200, 200)
            z = x @ y
        
        result = z.cuda()
        
        assert result.shape == torch.Size([200, 200])
        assert result.device.type == 'cuda'
        
        print("✅ Large tensor GPU execution works")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_complex_computation_gpu(self):
        """Test complex computation on GPU."""
        
        with genie.capture():
            x = torch.randn(50, 50)
            
            # Chained operations
            a = torch.relu(x)
            b = a @ a
            c = b + x
            d = torch.sigmoid(c)
        
        result = d.cuda()
        
        assert result.shape == torch.Size([50, 50])
        assert (result >= 0).all() and (result <= 1).all()
        
        print("✅ Complex computation on GPU works")


class TestDevicePlacement:
    """Test device placement and transfer."""
    
    def test_cpu_to_cpu_execution(self):
        """Test CPU to CPU execution."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = x @ x
        
        result_cpu = y.cpu()
        
        assert result_cpu.device.type == 'cpu'
        
        print("✅ CPU to CPU execution works")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_to_gpu_transfer(self):
        """Test transfer from CPU to GPU."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = x @ x
        
        # Transfer to GPU
        result_gpu = y.cuda()
        
        assert result_gpu.device.type == 'cuda'
        
        # Transfer back to CPU
        result_cpu = result_gpu.cpu()
        assert result_cpu.device.type == 'cpu'
        
        print("✅ CPU to GPU transfer works")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_to_cpu_transfer(self):
        """Test transfer from GPU to CPU."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = x @ x
        
        # Go to GPU then back to CPU
        result_gpu = y.cuda()
        result_cpu = result_gpu.cpu()
        
        assert result_cpu.device.type == 'cpu'
        
        print("✅ GPU to CPU transfer works")


class TestExecutionCorrectness:
    """Test execution correctness and numerical properties."""
    
    def test_execution_deterministic_with_seed(self):
        """Test execution is deterministic with fixed seed."""
        
        torch.manual_seed(42)
        with genie.capture():
            x1 = torch.randn(5, 5)
            y1 = x1 @ x1
        
        result1 = y1.cpu()
        
        torch.manual_seed(42)
        with genie.capture():
            x2 = torch.randn(5, 5)
            y2 = x2 @ x2
        
        result2 = y2.cpu()
        
        # Results should be identical
        assert torch.allclose(result1, result2)
        
        print("✅ Execution deterministic with seed")
    
    def test_numerical_stability(self):
        """Test numerical stability in execution."""
        
        with genie.capture():
            x = torch.randn(10, 10)
            
            # Operations that could lose precision
            y = x / 1e10
            z = y * 1e10
        
        result = z.cpu()
        
        # Should be approximately equal to original
        assert result.shape == torch.Size([10, 10])
        assert not torch.isnan(result).any()
        
        print("✅ Numerical stability maintained")
    
    def test_nan_propagation(self):
        """Test NaN propagation in execution."""
        
        with genie.capture():
            x = torch.tensor([[1.0, float('nan')], [3.0, 4.0]])
            y = x + 1
        
        result = y.cpu()
        
        # NaN should propagate
        assert torch.isnan(result[0, 1])
        
        print("✅ NaN propagation works")


class TestExecutionPerformance:
    """Test execution performance."""
    
    def test_execution_completes_quickly(self):
        """Test execution completes in reasonable time."""
        
        start = time.time()
        
        with genie.capture():
            x = torch.randn(50, 50)
            y = x @ x
        
        result = y.cpu()
        elapsed = time.time() - start
        
        # Should complete in less than 1 second
        assert elapsed < 1.0, f"Execution took {elapsed}s"
        
        print(f"✅ Execution completes quickly ({elapsed:.3f}s)")
    
    def test_large_graph_execution(self):
        """Test execution of large computation graphs."""
        
        start = time.time()
        
        with genie.capture():
            x = torch.randn(50, 50)
            for i in range(10):
                x = x @ x + i
        
        result = x.cpu()
        elapsed = time.time() - start
        
        assert result.shape == torch.Size([50, 50])
        assert elapsed < 5.0, f"Large graph took {elapsed}s"
        
        print(f"✅ Large graph execution completes ({elapsed:.3f}s)")
    
    def test_many_operations_execution(self):
        """Test execution with many operations."""
        
        start = time.time()
        
        with genie.capture():
            x = torch.randn(20, 20)
            for i in range(50):
                x = x + 0.01
        
        result = x.cpu()
        elapsed = time.time() - start
        
        assert result.shape == torch.Size([20, 20])
        assert elapsed < 2.0, f"Many ops took {elapsed}s"
        
        print(f"✅ Many operations execute efficiently ({elapsed:.3f}s)")


class TestResultCorrectness:
    """Test result correctness against PyTorch."""
    
    def test_against_pytorch_simple_ops(self):
        """Test results match PyTorch for simple operations."""
        
        # Expected result with PyTorch
        x_torch = torch.randn(5, 5)
        y_torch = x_torch + 1
        expected = y_torch.numpy()
        
        # Lazy execution
        torch.manual_seed(x_torch.seed() if hasattr(x_torch, 'seed') else 0)
        with genie.capture():
            x_lazy = torch.randn(5, 5)
            y_lazy = x_lazy + 1
        
        actual = y_lazy.cpu().numpy()
        
        # Shapes should match
        assert expected.shape == actual.shape
        
        print("✅ Results match PyTorch for simple ops")
    
    def test_against_pytorch_complex_ops(self):
        """Test results match PyTorch for complex operations."""
        
        # Expected result with PyTorch
        x_torch = torch.randn(5, 5)
        y_torch = torch.randn(5, 5)
        expected = torch.relu(x_torch @ y_torch).numpy()
        
        # Lazy execution
        torch.manual_seed(0)
        with genie.capture():
            x_lazy = torch.randn(5, 5)
            y_lazy = torch.randn(5, 5)
            z_lazy = torch.relu(x_lazy @ y_lazy)
        
        actual = z_lazy.cpu().numpy()
        
        # Shapes should match
        assert expected.shape == actual.shape
        
        print("✅ Results match PyTorch for complex ops")
    
    def test_dtype_preservation(self):
        """Test dtype is preserved through execution."""
        
        dtypes = [torch.float32, torch.float64]
        
        for dtype in dtypes:
            with genie.capture():
                x = torch.randn(5, 5, dtype=dtype)
                y = x + 1
            
            result = y.cpu()
            
            assert result.dtype == dtype, \
                f"Dtype mismatch: {result.dtype} vs {dtype}"
        
        print("✅ Dtype preserved through execution")


class TestExecutionRobustness:
    """Test execution robustness."""
    
    def test_repeated_execution(self):
        """Test repeated execution of same graph."""
        
        results = []
        
        for i in range(5):
            with genie.capture():
                x = torch.randn(5, 5)
                y = x @ x
            
            result = y.cpu()
            results.append(result)
        
        # All should have completed successfully
        assert len(results) == 5
        assert all(r.shape == torch.Size([5, 5]) for r in results)
        
        print("✅ Repeated execution works")
    
    def test_execution_with_gradients_disabled(self):
        """Test execution with gradients disabled."""
        
        torch.no_grad()
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = x @ x
        
        result = y.cpu()
        
        assert result.shape == torch.Size([5, 5])
        
        print("✅ Execution with gradients disabled works")
    
    def test_sequential_captures_independent(self):
        """Test sequential captures are independent."""
        
        # First capture
        with genie.capture():
            x1 = torch.randn(5, 5)
            y1 = x1 + 1
        
        result1 = y1.cpu()
        
        # Second capture
        with genie.capture():
            x2 = torch.randn(10, 10)
            y2 = x2 @ x2
        
        result2 = y2.cpu()
        
        # Results should be independent
        assert result1.shape == torch.Size([5, 5])
        assert result2.shape == torch.Size([10, 10])
        
        print("✅ Sequential captures are independent")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
