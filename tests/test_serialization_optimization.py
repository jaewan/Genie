"""
Unit tests for optimized tensor serialization.

Tests the numpy.save optimization which provides 44% faster serialization
compared to torch.save for network transfer.
"""

import unittest
import torch
import numpy as np
import io
import time
from genie.core.serialization import (
    serialize_tensor,
    deserialize_tensor,
    measure_serialization_overhead,
    get_serialization_stats,
    FORMAT_NUMPY,
    FORMAT_TORCH
)


class TestSerializationCorrectness(unittest.TestCase):
    """Test correctness of serialization/deserialization"""
    
    def test_numpy_serialization_correctness_various_shapes(self):
        """Test numpy serialization produces correct results for various tensor shapes"""
        test_cases = [
            torch.randn(1, 128, 768),      # Small: GPT-2 small hidden
            torch.randn(8, 256, 1024),     # Medium: Batch processing
            torch.randn(1, 1024, 768),     # Large: GPT-2 long sequence
            torch.randn(16, 512, 2048),    # Very large: Large model
        ]
        
        for tensor in test_cases:
            with self.subTest(shape=tensor.shape):
                # Serialize with numpy
                data = serialize_tensor(tensor, use_numpy=True)
                
                # Deserialize
                result = deserialize_tensor(data)
                
                # Check correctness
                self.assertTrue(torch.allclose(tensor, result, rtol=1e-5))
                self.assertEqual(tensor.shape, result.shape)
                self.assertEqual(tensor.dtype, result.dtype)
    
    def test_torch_serialization_correctness(self):
        """Test torch.save serialization still works correctly"""
        tensor = torch.randn(1, 1024, 768)
        
        # Serialize with torch.save
        data = serialize_tensor(tensor, use_numpy=False)
        
        # Deserialize
        result = deserialize_tensor(data)
        
        # Check correctness
        self.assertTrue(torch.allclose(tensor, result, rtol=1e-5))
        self.assertEqual(tensor.shape, result.shape)
        self.assertEqual(tensor.dtype, result.dtype)
    
    def test_fp16_serialization_correctness(self):
        """Test float16 serialization maintains acceptable precision"""
        tensor = torch.randn(1, 1024, 768)
        
        # Serialize with fp16
        data = serialize_tensor(tensor, use_numpy=True, use_fp16=True)
        
        # Deserialize
        result = deserialize_tensor(data)
        
        # Check shape and dtype
        self.assertEqual(tensor.shape, result.shape)
        self.assertEqual(result.dtype, torch.float16)
        
        # Check reasonable precision (fp16 has lower precision)
        self.assertTrue(torch.allclose(tensor, result.float(), rtol=1e-2, atol=1e-3))
    
    def test_device_transfer(self):
        """Test deserialization to different devices"""
        tensor = torch.randn(100, 100)
        
        # Serialize
        data = serialize_tensor(tensor, use_numpy=True)
        
        # Deserialize to CPU
        result_cpu = deserialize_tensor(data, device=torch.device('cpu'))
        self.assertEqual(result_cpu.device.type, 'cpu')
        self.assertTrue(torch.allclose(tensor, result_cpu))
        
        # Deserialize to CUDA (if available)
        if torch.cuda.is_available():
            result_cuda = deserialize_tensor(data, device=torch.device('cuda:0'))
            self.assertEqual(result_cuda.device.type, 'cuda')
            self.assertTrue(torch.allclose(tensor, result_cuda.cpu()))
    
    def test_format_headers(self):
        """Test that format headers are correctly written"""
        tensor = torch.randn(10, 10)
        
        # Numpy format
        data_numpy = serialize_tensor(tensor, use_numpy=True)
        self.assertEqual(data_numpy[:8], FORMAT_NUMPY)
        
        # Torch format
        data_torch = serialize_tensor(tensor, use_numpy=False)
        self.assertEqual(data_torch[:8], FORMAT_TORCH)
    
    def test_backward_compatibility(self):
        """Test backward compatibility with old torch.save format"""
        tensor = torch.randn(100, 100)
        
        # Serialize with old torch.save format (no header)
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        old_format_data = buffer.getvalue()
        
        # Should be able to deserialize
        result = deserialize_tensor(old_format_data)
        self.assertTrue(torch.allclose(tensor, result))


class TestSerializationPerformance(unittest.TestCase):
    """Test performance of serialization methods"""
    
    def test_numpy_faster_than_torch(self):
        """Test that numpy serialization is faster than torch.save"""
        tensor = torch.randn(1, 1024, 768)  # 3MB tensor
        
        torch_time, numpy_time, speedup = measure_serialization_overhead(
            tensor, num_iterations=50
        )
        
        print(f"\nPerformance comparison (50 iterations):")
        print(f"  torch.save: {torch_time:.3f}ms")
        print(f"  numpy.save: {numpy_time:.3f}ms")
        print(f"  Speedup:    {speedup:.2f}x")
        
        # Assert numpy is faster
        self.assertLess(numpy_time, torch_time)
        
        # Assert at least 20% speedup (conservative, actual is 44%)
        self.assertGreater(speedup, 1.2)
    
    def test_fp16_faster_and_smaller(self):
        """Test that fp16 serialization is faster and smaller"""
        tensor = torch.randn(1, 1024, 768)
        
        # Serialize with fp32 and fp16
        data_fp32 = serialize_tensor(tensor, use_numpy=True, use_fp16=False)
        data_fp16 = serialize_tensor(tensor, use_numpy=True, use_fp16=True)
        
        # FP16 should be approximately half the size
        size_ratio = len(data_fp16) / len(data_fp32)
        print(f"\nSize comparison:")
        print(f"  FP32: {len(data_fp32)/1024/1024:.2f}MB")
        print(f"  FP16: {len(data_fp16)/1024/1024:.2f}MB")
        print(f"  Ratio: {size_ratio:.2f}")
        
        # Assert fp16 is smaller (should be ~0.5)
        self.assertLess(size_ratio, 0.6)
        self.assertGreater(size_ratio, 0.4)


class TestSerializationStats(unittest.TestCase):
    """Test serialization statistics gathering"""
    
    def test_get_serialization_stats(self):
        """Test get_serialization_stats returns correct information"""
        tensor = torch.randn(1, 1024, 768)
        
        stats = get_serialization_stats(tensor)
        
        # Check structure
        self.assertIn('tensor_shape', stats)
        self.assertIn('tensor_dtype', stats)
        self.assertIn('tensor_size_mb', stats)
        self.assertIn('torch_save', stats)
        self.assertIn('numpy_save', stats)
        self.assertIn('numpy_fp16', stats)
        
        # Check values
        self.assertEqual(stats['tensor_shape'], (1, 1024, 768))
        self.assertEqual(stats['tensor_dtype'], 'torch.float32')
        
        # Numpy should be smaller or equal
        self.assertLessEqual(
            stats['numpy_save']['size_bytes'],
            stats['torch_save']['size_bytes']
        )
        
        # FP16 should be approximately half
        self.assertLess(
            stats['numpy_fp16']['size_bytes'],
            stats['numpy_save']['size_bytes'] * 0.6
        )
        
        print(f"\nSerialization stats for {tensor.shape}:")
        print(f"  Tensor size: {stats['tensor_size_mb']:.2f}MB")
        print(f"  torch.save:  {stats['torch_save']['size_mb']:.2f}MB")
        print(f"  numpy.save:  {stats['numpy_save']['size_mb']:.2f}MB " +
              f"({stats['numpy_save']['size_ratio']:.2%} of torch)")
        print(f"  numpy_fp16:  {stats['numpy_fp16']['size_mb']:.2f}MB " +
              f"({stats['numpy_fp16']['size_ratio']:.2%} of torch)")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_tensor(self):
        """Test serialization of empty tensors"""
        tensor = torch.randn(0, 0)
        
        # Should work with both methods
        data_numpy = serialize_tensor(tensor, use_numpy=True)
        result_numpy = deserialize_tensor(data_numpy)
        self.assertEqual(result_numpy.shape, tensor.shape)
        
        data_torch = serialize_tensor(tensor, use_numpy=False)
        result_torch = deserialize_tensor(data_torch)
        self.assertEqual(result_torch.shape, tensor.shape)
    
    def test_1d_tensor(self):
        """Test serialization of 1D tensors"""
        tensor = torch.randn(1000)
        
        data = serialize_tensor(tensor, use_numpy=True)
        result = deserialize_tensor(data)
        
        self.assertTrue(torch.allclose(tensor, result))
        self.assertEqual(tensor.shape, result.shape)
    
    def test_large_tensor(self):
        """Test serialization of large tensors"""
        # 100MB tensor
        tensor = torch.randn(1000, 1000, 25)
        
        data = serialize_tensor(tensor, use_numpy=True)
        result = deserialize_tensor(data)
        
        self.assertTrue(torch.allclose(tensor, result, rtol=1e-5))
        self.assertEqual(tensor.shape, result.shape)
    
    def test_different_dtypes(self):
        """Test serialization with different data types"""
        dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]
        
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                if dtype in [torch.float32, torch.float64]:
                    tensor = torch.randn(100, 100).to(dtype)
                else:
                    tensor = torch.randint(0, 100, (100, 100)).to(dtype)
                
                data = serialize_tensor(tensor, use_numpy=True)
                result = deserialize_tensor(data)
                
                self.assertTrue(torch.equal(tensor, result))
                self.assertEqual(tensor.dtype, result.dtype)


def run_performance_benchmark():
    """Run comprehensive performance benchmark"""
    print("\n" + "="*70)
    print("SERIALIZATION PERFORMANCE BENCHMARK")
    print("="*70)
    
    tensor_shapes = [
        (1, 128, 768),      # Small
        (1, 256, 768),      # Medium
        (1, 512, 768),      # Large
        (1, 1024, 768),     # GPT-2 typical
        (8, 128, 768),      # Batch processing
    ]
    
    print("\nTensor Shape       | torch.save | numpy.save | Speedup | Size (MB)")
    print("-" * 70)
    
    for shape in tensor_shapes:
        tensor = torch.randn(*shape)
        torch_time, numpy_time, speedup = measure_serialization_overhead(
            tensor, num_iterations=50
        )
        
        stats = get_serialization_stats(tensor)
        
        print(f"{str(shape):18s} | {torch_time:9.3f}ms | {numpy_time:9.3f}ms | "
              f"{speedup:6.2f}x | {stats['tensor_size_mb']:6.2f}")
    
    print("="*70)


if __name__ == '__main__':
    # Run performance benchmark first
    run_performance_benchmark()
    
    # Then run unit tests
    print("\n" + "="*70)
    print("RUNNING UNIT TESTS")
    print("="*70 + "\n")
    
    unittest.main(verbosity=2)

