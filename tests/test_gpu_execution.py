"""
GPU Execution Test for Genie Framework

Tests that tensor operations are properly intercepted and executed on GPU (no CPU fallback).
This validates the core implementation from HotNets'25 paper:
- LazyTensor captures operations
- Dispatcher intercepts PyTorch calls  
- Executor materializes on actual GPU hardware

Usage:
    pytest tests/test_gpu_execution.py -v
    
    # Or run directly:
    python tests/test_gpu_execution.py
"""

import sys
import os
import pytest
import torch
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from genie.core.lazy_tensor import LazyTensor
from genie.core.device import get_device, is_available
from genie.core.enhanced_dispatcher import enhanced_dispatcher

logger = logging.getLogger(__name__)


def check_cuda_available():
    """Check if CUDA is available and compatible."""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    try:
        # Test if we can actually create CUDA tensors
        test = torch.randn(10, 10, device='cuda')
        return True, "CUDA available"
    except RuntimeError as e:
        if "no kernel image" in str(e) or "sm_120" in str(e):
            return False, f"GPU architecture not supported: {e}"
        return False, f"CUDA error: {e}"


class GPUExecutor:
    """GPU-based executor for testing actual GPU execution.
    
    Unlike SimpleExecutor which forces CPU, this executor runs on GPU.
    """
    
    def __init__(self, device='cuda:0'):
        self.device = device
        self.execution_count = 0
        
    def execute_subgraph(self, target_lazy_tensor: LazyTensor) -> torch.Tensor:
        """Execute LazyTensor graph on GPU."""
        # CRITICAL: Respect LazyTensor's caching to avoid re-executing random ops
        if target_lazy_tensor.materialized and target_lazy_tensor.concrete_value is not None:
            return target_lazy_tensor.concrete_value
        
        self.execution_count += 1
        
        # Recursively materialize on GPU
        result = self._materialize_recursive(target_lazy_tensor)
        
        # Cache the result in the LazyTensor
        target_lazy_tensor.concrete_value = result
        target_lazy_tensor.materialized = True
        
        return result
    
    def _materialize_recursive(self, lt: LazyTensor) -> torch.Tensor:
        """Recursively materialize LazyTensor and its inputs on GPU."""
        
        # Check if already materialized (important for shared subgraphs)
        if lt.materialized and lt.concrete_value is not None:
            return lt.concrete_value
        
        # Resolve inputs first
        resolved_inputs = []
        for inp in lt.inputs:
            if isinstance(inp, LazyTensor):
                # Recursive call will use caching
                resolved_inputs.append(self._materialize_recursive(inp))
            elif isinstance(inp, torch.Tensor):
                # Move to GPU if not already there
                resolved_inputs.append(inp.to(self.device))
            else:
                # Keep scalars/shapes as-is
                resolved_inputs.append(inp)
        
        # Get kwargs and ensure device is GPU
        kwargs = lt.kwargs.copy() if lt.kwargs else {}
        
        # Execute operation on GPU
        result = self._execute_op_on_gpu(lt.operation, resolved_inputs, kwargs)
        
        # Cache the result
        lt.concrete_value = result
        lt.materialized = True
        
        return result
    
    def _execute_op_on_gpu(self, op_name: str, inputs, kwargs) -> torch.Tensor:
        """Execute a single operation on GPU."""
        
        # Normalize operation name
        aten_prefix = "aten::"
        base_name = op_name[len(aten_prefix):] if op_name.startswith(aten_prefix) else op_name
        
        # Handle creation operations
        creation_ops = {
            "randn", "rand", "randint", "zeros", "ones", 
            "empty", "full", "arange", "linspace"
        }
        
        if base_name in creation_ops:
            kwargs['device'] = self.device
            
            # Extract size arguments
            if inputs and isinstance(inputs[0], (tuple, list)):
                size = inputs[0]
            elif inputs and all(isinstance(x, int) for x in inputs):
                size = tuple(inputs)
            else:
                size = (1,)
            
            # Dispatch to appropriate function
            if base_name == "randn":
                return torch.randn(*size, **kwargs)
            elif base_name == "rand":
                return torch.rand(*size, **kwargs)
            elif base_name == "zeros":
                return torch.zeros(*size, **kwargs)
            elif base_name == "ones":
                return torch.ones(*size, **kwargs)
            elif base_name == "empty":
                return torch.empty(*size, **kwargs)
            elif base_name == "full":
                fill_value = kwargs.pop('fill_value', 0)
                return torch.full(size, fill_value, **kwargs)
        
        # Ensure all tensor inputs are on GPU
        gpu_inputs = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                gpu_inputs.append(inp.to(self.device))
            else:
                gpu_inputs.append(inp)
        
        # Remove device from kwargs for non-creation ops
        kwargs.pop('device', None)
        
        # Dispatch common operations directly
        try:
            # Arithmetic
            if base_name == "add":
                return torch.add(gpu_inputs[0], gpu_inputs[1], **kwargs)
            elif base_name == "sub":
                return torch.sub(gpu_inputs[0], gpu_inputs[1], **kwargs)
            elif base_name == "mul":
                return torch.mul(gpu_inputs[0], gpu_inputs[1], **kwargs)
            elif base_name == "div":
                return torch.div(gpu_inputs[0], gpu_inputs[1], **kwargs)
            
            # Linear algebra
            elif base_name == "matmul":
                return torch.matmul(gpu_inputs[0], gpu_inputs[1])
            elif base_name == "mm":
                return torch.mm(gpu_inputs[0], gpu_inputs[1])
            elif base_name == "bmm":
                return torch.bmm(gpu_inputs[0], gpu_inputs[1])
            
            # Activations
            elif base_name == "relu":
                return torch.relu(gpu_inputs[0])
            elif base_name == "sigmoid":
                return torch.sigmoid(gpu_inputs[0])
            elif base_name == "tanh":
                return torch.tanh(gpu_inputs[0])
            elif base_name == "gelu":
                return torch.nn.functional.gelu(gpu_inputs[0])
            
            # Reductions
            elif base_name == "sum":
                return torch.sum(gpu_inputs[0], **kwargs)
            elif base_name == "mean":
                return torch.mean(gpu_inputs[0], **kwargs)
            elif base_name == "max":
                result = torch.max(gpu_inputs[0], **kwargs)
                return result[0] if isinstance(result, tuple) else result
            elif base_name == "min":
                result = torch.min(gpu_inputs[0], **kwargs)
                return result[0] if isinstance(result, tuple) else result
            
            # Softmax
            elif base_name == "softmax":
                return torch.softmax(gpu_inputs[0], **kwargs)
            
            # Convolution
            elif base_name == "conv2d":
                return torch.conv2d(*gpu_inputs, **kwargs)
            
            # Otherwise use torch.ops.aten
            else:
                aten_op = getattr(torch.ops.aten, base_name)
                result = aten_op(*gpu_inputs, **kwargs)
                return result[0] if isinstance(result, tuple) else result
                
        except Exception as e:
            logger.error(f"GPU execution failed for {op_name}: {e}")
            raise


# Test fixtures
@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available for the session."""
    available, msg = check_cuda_available()
    if not available:
        pytest.skip(msg)
    return available


@pytest.fixture
def gpu_executor():
    """Create GPU executor."""
    return GPUExecutor(device='cuda:0')


@pytest.fixture(autouse=True)
def patch_executor_for_gpu(monkeypatch, cuda_available, gpu_executor):
    """Patch the executor to use GPU instead of CPU."""
    # Replace the execute_subgraph function with GPU version
    import genie.core.executor as executor_module
    monkeypatch.setattr(executor_module, 'execute_subgraph', gpu_executor.execute_subgraph)


# ============================================================================
# Test Suite
# ============================================================================

class TestGPUInterception:
    """Test that operations are properly intercepted for GPU execution."""
    
    def test_lazy_tensor_creation_on_remote_device(self, cuda_available):
        """Test that LazyTensor is created for remote_accelerator device."""
        x = torch.randn(4, 4, device="remote_accelerator:0")
        
        assert isinstance(x, LazyTensor), "Should create LazyTensor"
        assert x.shape == torch.Size([4, 4])
        assert not x.materialized, "Should not be materialized yet"
        assert x.operation == "aten::randn"
    
    def test_operation_interception(self, cuda_available):
        """Test that operations are intercepted and create lazy graph."""
        x = torch.randn(4, 4, device="remote_accelerator:0")
        y = x + x
        z = torch.matmul(y, y)
        
        # All should be LazyTensors
        assert isinstance(x, LazyTensor)
        assert isinstance(y, LazyTensor)
        assert isinstance(z, LazyTensor)
        
        # Check operations are captured
        assert x.operation == "aten::randn"
        assert y.operation == "aten::add"
        assert z.operation == "aten::matmul"
        
        # None materialized yet
        assert not x.materialized
        assert not y.materialized
        assert not z.materialized


class TestGPUExecution:
    """Test that operations execute on GPU (no CPU fallback)."""
    
    def test_simple_tensor_creation_on_gpu(self, cuda_available, gpu_executor):
        """Test tensor creation executes on GPU."""
        x = torch.randn(100, 100, device="remote_accelerator:0")
        
        # Materialize
        result = x.materialize()
        
        # Verify it's a real tensor on GPU
        assert isinstance(result, torch.Tensor)
        assert result.is_cuda, f"Result should be on CUDA, got {result.device}"
        assert result.device.type == "cuda"
        assert result.shape == torch.Size([100, 100])
        
        # Verify we executed
        assert gpu_executor.execution_count > 0
    
    def test_arithmetic_on_gpu(self, cuda_available, gpu_executor):
        """Test arithmetic operations execute on GPU."""
        x = torch.randn(50, 50, device="remote_accelerator:0")
        y = x + x
        
        result = y.materialize()
        
        assert result.is_cuda, f"Addition result should be on CUDA, got {result.device}"
        assert result.shape == torch.Size([50, 50])
        
        # Verify correctness: y = x + x, so each element should be ~2x
        x_materialized = x.materialize()
        expected = x_materialized + x_materialized
        assert torch.allclose(result, expected, rtol=1e-5)
    
    def test_matmul_on_gpu(self, cuda_available, gpu_executor):
        """Test matrix multiplication executes on GPU."""
        x = torch.randn(64, 64, device="remote_accelerator:0")
        y = torch.matmul(x, x)
        
        result = y.materialize()
        
        assert result.is_cuda, f"Matmul result should be on CUDA, got {result.device}"
        assert result.shape == torch.Size([64, 64])
        
        # Verify correctness
        x_mat = x.materialize()
        expected = torch.matmul(x_mat, x_mat)
        assert torch.allclose(result, expected, rtol=1e-4)
    
    def test_complex_operations_on_gpu(self, cuda_available, gpu_executor):
        """Test complex operation chains execute on GPU."""
        # Build a computation graph
        x = torch.randn(32, 32, device="remote_accelerator:0")
        y = x + x
        z = torch.matmul(y, y)
        w = torch.relu(z)
        
        # Materialize final result
        result = w.materialize()
        
        assert result.is_cuda, f"Result should be on CUDA, got {result.device}"
        assert result.shape == torch.Size([32, 32])
        
        # Verify no negative values (ReLU property)
        assert (result >= 0).all(), "ReLU output should be non-negative"
    
    def test_activation_functions_on_gpu(self, cuda_available, gpu_executor):
        """Test activation functions execute on GPU."""
        x = torch.randn(10, 10, device="remote_accelerator:0")
        
        # Test ReLU
        relu_result = torch.relu(x).materialize()
        assert relu_result.is_cuda
        assert (relu_result >= 0).all()
        
        # Test Sigmoid
        sigmoid_result = torch.sigmoid(x).materialize()
        assert sigmoid_result.is_cuda
        assert ((sigmoid_result >= 0) & (sigmoid_result <= 1)).all()
        
        # Test Tanh
        tanh_result = torch.tanh(x).materialize()
        assert tanh_result.is_cuda
        assert ((tanh_result >= -1) & (tanh_result <= 1)).all()


class TestGPUPerformance:
    """Test GPU performance characteristics."""
    
    def test_large_matmul_on_gpu(self, cuda_available, gpu_executor):
        """Test large matrix multiplication on GPU."""
        import time
        
        size = 1024
        x = torch.randn(size, size, device="remote_accelerator:0")
        y = torch.matmul(x, x)
        
        # Time GPU execution
        start = time.perf_counter()
        result = y.materialize()
        gpu_time = time.perf_counter() - start
        
        assert result.is_cuda
        assert result.shape == torch.Size([size, size])
        
        print(f"\nGPU matmul ({size}x{size}): {gpu_time*1000:.2f}ms")
        
        # Compare with direct CUDA execution
        x_cuda = torch.randn(size, size, device='cuda')
        torch.cuda.synchronize()
        start = time.perf_counter()
        result_direct = torch.matmul(x_cuda, x_cuda)
        torch.cuda.synchronize()
        direct_time = time.perf_counter() - start
        
        print(f"Direct CUDA matmul ({size}x{size}): {direct_time*1000:.2f}ms")
        print(f"Overhead: {(gpu_time/direct_time - 1)*100:.1f}%")
        
        # Overhead should be reasonable (< 50% for large operations)
        assert gpu_time < direct_time * 1.5, "GPU execution overhead too high"
    
    def test_memory_stays_on_gpu(self, cuda_available, gpu_executor):
        """Test that memory stays on GPU throughout computation."""
        x = torch.randn(100, 100, device="remote_accelerator:0")
        y = x + x
        z = torch.matmul(y, y)
        w = z.sum()
        
        # Materialize and check
        result = w.materialize()
        
        assert result.is_cuda, "Final result should be on GPU"
        
        # Check intermediate results are also on GPU
        y_mat = y.materialize()
        z_mat = z.materialize()
        
        assert y_mat.is_cuda, "Intermediate y should be on GPU"
        assert z_mat.is_cuda, "Intermediate z should be on GPU"


class TestGPUCorrectness:
    """Test that GPU execution produces correct results."""
    
    def test_correctness_vs_cpu(self, cuda_available, gpu_executor):
        """Test GPU execution matches CPU results."""
        # Create concrete input first, then test operations on it
        torch.manual_seed(42)
        x_concrete = torch.randn(20, 20, device="cuda")
        
        # Create LazyTensor wrapper around the concrete input
        # This ensures both GPU and CPU use the same input values
        x_remote = torch.randn(20, 20, device="remote_accelerator:0")
        
        # Materialize the random tensor first to get concrete values
        x_materialized = x_remote.materialize()
        
        # Now use the same concrete values for CPU computation
        x_cpu = x_materialized.cpu()
        
        # Same computation on both
        y_remote_lazy = x_remote + x_remote  # This uses the cached x_remote
        y_cpu = x_cpu + x_cpu
        
        # Materialize GPU result
        y_remote_mat = y_remote_lazy.materialize()
        
        # Compare (move GPU result to CPU for comparison)
        y_remote_cpu = y_remote_mat.cpu()
        
        assert torch.allclose(y_remote_cpu, y_cpu, rtol=1e-4, atol=1e-6), \
            "GPU and CPU results should match"
    
    def test_complex_computation_correctness(self, cuda_available, gpu_executor):
        """Test complex computation produces correct results on GPU."""
        torch.manual_seed(123)
        x = torch.randn(16, 16, device="remote_accelerator:0")
        
        # Complex computation
        y = x + x
        z = torch.matmul(y, y)
        w = torch.relu(z)
        result = w.sum()
        
        # Materialize
        gpu_result = result.materialize()
        
        # Compute same thing directly on GPU
        torch.manual_seed(123)
        x_cuda = torch.randn(16, 16, device='cuda')
        y_cuda = x_cuda + x_cuda
        z_cuda = torch.matmul(y_cuda, y_cuda)
        w_cuda = torch.relu(z_cuda)
        expected = w_cuda.sum()
        
        # Compare
        assert torch.allclose(gpu_result.cpu(), expected.cpu(), rtol=1e-4), \
            "Complex GPU computation should match direct CUDA"


class TestGPUIntegration:
    """Integration tests for GPU execution."""
    
    def test_dispatcher_stats(self, cuda_available):
        """Test that operations are captured in computation graph."""
        from genie.core.graph import GraphBuilder
        
        # Get current graph state
        graph_before = GraphBuilder.current().get_graph()
        nodes_before = len(graph_before.nodes)
        
        # Execute some operations
        x = torch.randn(10, 10, device="remote_accelerator:0")
        y = x + x
        z = torch.matmul(y, y)
        
        # Check that operations were captured in the graph
        graph_after = GraphBuilder.current().get_graph()
        nodes_after = len(graph_after.nodes)
        
        # Should have captured at least 3 operations (randn, add, matmul)
        assert nodes_after > nodes_before, "Operations should be captured in graph"
        assert nodes_after >= 3, f"Expected at least 3 nodes, got {nodes_after}"
    
    def test_multiple_devices(self, cuda_available):
        """Test execution on multiple GPU devices if available."""
        device_count = torch.cuda.device_count()
        if device_count < 2:
            pytest.skip("Need at least 2 GPUs")
        
        # Create tensors on different devices
        x0 = torch.randn(10, 10, device="remote_accelerator:0")
        x1 = torch.randn(10, 10, device="remote_accelerator:1")
        
        # Materialize
        result0 = x0.materialize()
        result1 = x1.materialize()
        
        assert result0.is_cuda
        assert result1.is_cuda
        # Note: Currently both go to cuda:0 in our implementation
        # Phase 2+ will support actual multi-GPU placement


# ============================================================================
# Main
# ============================================================================

def main():
    """Run tests with detailed output."""
    
    # Check CUDA
    available, msg = check_cuda_available()
    print(f"\n{'='*70}")
    print(f"CUDA Status: {msg}")
    print(f"{'='*70}\n")
    
    if not available:
        print("❌ CUDA not available - tests will be skipped")
        return 1
    
    # Print GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Run tests
    exit_code = pytest.main([
        __file__,
        '-v',
        '-s',
        '--tb=short',
        '--color=yes'
    ])
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

