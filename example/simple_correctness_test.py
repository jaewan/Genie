"""
Simple correctness test for Genie LazyTensor implementation.

This tests basic operations that should work with our current implementation.
"""
from __future__ import annotations

import torch
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import genie
from genie.core.lazy_tensor import LazyTensor


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_basic_operations():
    """Test basic operations that should work."""
    print_header("Basic LazyTensor Operations Test")
    
    # Enable lazy mode
    genie.set_lazy_mode(True)
    
    print("1. Testing tensor creation...")
    try:
        # Create tensors on remote_accelerator device
        x = torch.randn(4, 4, device="remote_accelerator:0")
        y = torch.randn(4, 4, device="remote_accelerator:0")
        
        print(f"   ✅ Created LazyTensor x: {type(x).__name__}, shape: {x.shape}")
        print(f"   ✅ Created LazyTensor y: {type(y).__name__}, shape: {y.shape}")
        
        # Test that they are LazyTensors
        assert isinstance(x, LazyTensor), f"Expected LazyTensor, got {type(x)}"
        assert isinstance(y, LazyTensor), f"Expected LazyTensor, got {type(y)}"
        
    except Exception as e:
        print(f"   ❌ Tensor creation failed: {e}")
        return False
    
    print("\n2. Testing basic arithmetic via __torch_function__...")
    try:
        # Test addition via __torch_function__
        z = torch.add(x, y)
        print(f"   ✅ torch.add(x, y) -> {type(z).__name__}, shape: {z.shape}")
        assert isinstance(z, LazyTensor), f"Expected LazyTensor, got {type(z)}"
        
        # Test multiplication
        w = torch.mul(x, y)
        print(f"   ✅ torch.mul(x, y) -> {type(w).__name__}, shape: {w.shape}")
        assert isinstance(w, LazyTensor), f"Expected LazyTensor, got {type(w)}"
        
    except Exception as e:
        print(f"   ❌ Arithmetic operations failed: {e}")
        return False
    
    print("\n3. Testing materialization...")
    try:
        # Test materialization
        z_cpu = z.cpu()
        print(f"   ✅ Materialized z to CPU: {type(z_cpu).__name__}, shape: {z_cpu.shape}")
        assert isinstance(z_cpu, torch.Tensor), f"Expected torch.Tensor, got {type(z_cpu)}"
        assert z_cpu.device.type == "cpu", f"Expected CPU device, got {z_cpu.device}"
        
        # Test that materialized result has correct shape
        assert z_cpu.shape == (4, 4), f"Expected shape (4, 4), got {z_cpu.shape}"
        
    except Exception as e:
        print(f"   ❌ Materialization failed: {e}")
        return False
    
    print("\n4. Testing correctness with simple case...")
    try:
        # Create simple tensors for correctness check
        a_cpu = torch.ones(2, 2)
        b_cpu = torch.ones(2, 2) * 2
        
        # Eager computation
        eager_result = torch.add(a_cpu, b_cpu)
        
        # Lazy computation
        a_lazy = torch.ones(2, 2, device="remote_accelerator:0")
        b_lazy = torch.ones(2, 2, device="remote_accelerator:0") * 2
        lazy_result = torch.add(a_lazy, b_lazy).cpu()
        
        # Compare results
        if torch.allclose(eager_result, lazy_result):
            print(f"   ✅ Correctness check passed!")
            print(f"      Eager result: {eager_result.flatten()}")
            print(f"      Lazy result:  {lazy_result.flatten()}")
        else:
            print(f"   ❌ Correctness check failed!")
            print(f"      Eager result: {eager_result.flatten()}")
            print(f"      Lazy result:  {lazy_result.flatten()}")
            return False
            
    except Exception as e:
        print(f"   ❌ Correctness check failed: {e}")
        return False
    
    print("\n5. Testing tensor methods...")
    try:
        # Test tensor methods that should work
        x_t = x.transpose(0, 1)
        print(f"   ✅ x.transpose(0, 1) -> {type(x_t).__name__}, shape: {x_t.shape}")
        assert isinstance(x_t, LazyTensor), f"Expected LazyTensor, got {type(x_t)}"
        
        # Test unsqueeze
        x_unsq = x.unsqueeze(0)
        print(f"   ✅ x.unsqueeze(0) -> {type(x_unsq).__name__}, shape: {x_unsq.shape}")
        assert isinstance(x_unsq, LazyTensor), f"Expected LazyTensor, got {type(x_unsq)}"
        
    except Exception as e:
        print(f"   ❌ Tensor methods failed: {e}")
        return False
    
    print("\n🎉 All basic tests passed! LazyTensor is working correctly.")
    return True


def main():
    """Run the simple correctness test."""
    print("Genie LazyTensor - Simple Correctness Test")
    print(f"PyTorch version: {torch.__version__}")
    
    success = test_basic_operations()
    
    if success:
        print("\n✅ SUCCESS: Basic LazyTensor functionality is working!")
        return 0
    else:
        print("\n❌ FAILURE: Some basic functionality is not working.")
        return 1


if __name__ == "__main__":
    exit(main())
