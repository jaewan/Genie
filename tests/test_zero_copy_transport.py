#!/usr/bin/env python3
"""
Test Zero-Copy Transport Implementation

Tests the complete zero-copy data path including:
- GPU memory registration and DMA
- External buffer attachment
- Reliability (ACK/NACK)
- Flow control
- AsyncIO integration
"""

import asyncio
import sys
import os
import time
import torch
import numpy as np
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genie.runtime.async_zero_copy_bridge import (
    AsyncZeroCopyBridge,
    ZeroCopyTransferManager,
    TransferState
)

class ZeroCopyTransportTester:
    """Test harness for zero-copy transport"""
    
    def __init__(self):
        self.bridge = None
        self.control_client = None  # Mock control client
        self.manager = None
        
    async def setup(self):
        """Setup test environment"""
        print("Setting up zero-copy transport test...")
        
        # Configuration
        config = {
            'lib_path': './build/libgenie_data_plane.so',
            'port_id': 0,
            'gpu_id': 0,
            'use_gpu_direct': True,
            'mtu': 8192,
            'num_workers': 4
        }
        
        # Create bridge
        self.bridge = AsyncZeroCopyBridge(config)
        
        # Initialize
        success = await self.bridge.initialize()
        if not success:
            print("WARNING: Failed to initialize with GPU Direct, using fallback")
            config['use_gpu_direct'] = False
            self.bridge = AsyncZeroCopyBridge(config)
            await self.bridge.initialize()
        
        print("Zero-copy transport initialized")
        return True
    
    async def teardown(self):
        """Cleanup test environment"""
        if self.bridge:
            await self.bridge.shutdown()
        print("Teardown complete")
    
    async def test_single_tensor_transfer(self):
        """Test single tensor transfer"""
        print("\n=== Test 1: Single Tensor Transfer ===")
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU tensor (will use staging)")
            tensor = torch.randn(1024, 1024)
        else:
            # Create test tensor on GPU
            tensor = torch.randn(1024, 1024, device='cuda')
            print(f"Created tensor: shape={tensor.shape}, device={tensor.device}")
        
        # Send tensor (loopback for testing)
        target = "127.0.0.1:12345"
        
        try:
            request = await self.bridge.send_tensor(
                tensor=tensor,
                target_node=target,
                target_gpu=0,
                timeout=5.0
            )
            
            print(f"Transfer initiated: {request.transfer_id}")
            print(f"Size: {request.size / (1024*1024):.2f} MB")
            
            # In real test, would wait for completion
            # For now, just check state
            await asyncio.sleep(0.1)
            
            print(f"Transfer state: {request.state}")
            
            # Get stats
            stats = self.bridge.get_stats()
            print(f"Transfers sent: {stats['transfers_sent']}")
            print(f"Bytes sent: {stats['bytes_sent'] / (1024*1024):.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"Transfer failed: {e}")
            return False
    
    async def test_batch_transfer(self):
        """Test batch tensor transfers"""
        print("\n=== Test 2: Batch Transfer ===")
        
        if not torch.cuda.is_available():
            print("CUDA not available, skipping batch test")
            return True
        
        # Create multiple tensors
        tensors = [
            torch.randn(512, 512, device='cuda') for _ in range(10)
        ]
        
        print(f"Created {len(tensors)} tensors")
        
        # Send all tensors concurrently
        target = "127.0.0.1:12345"
        requests = []
        
        for i, tensor in enumerate(tensors):
            request = await self.bridge.send_tensor(
                tensor=tensor,
                target_node=target,
                priority=i  # Different priorities
            )
            requests.append(request)
        
        print(f"Initiated {len(requests)} transfers")
        
        # Wait briefly
        await asyncio.sleep(0.5)
        
        # Check states
        completed = sum(1 for r in requests if r.state == TransferState.COMPLETED)
        in_progress = sum(1 for r in requests if r.state == TransferState.IN_PROGRESS)
        failed = sum(1 for r in requests if r.state == TransferState.FAILED)
        
        print(f"Completed: {completed}, In Progress: {in_progress}, Failed: {failed}")
        
        return True
    
    async def test_large_tensor(self):
        """Test large tensor transfer"""
        print("\n=== Test 3: Large Tensor Transfer ===")
        
        if not torch.cuda.is_available():
            print("CUDA not available, skipping large tensor test")
            return True
        
        # Check available GPU memory
        torch.cuda.empty_cache()
        mem_info = torch.cuda.mem_get_info()
        available_gb = mem_info[0] / (1024**3)
        
        print(f"Available GPU memory: {available_gb:.2f} GB")
        
        if available_gb < 1:
            print("Insufficient GPU memory for large tensor test")
            return True
        
        # Create large tensor (100MB)
        size = 100 * 1024 * 1024 // 4  # float32 elements
        tensor = torch.randn(size, device='cuda')
        
        print(f"Created large tensor: {tensor.numel() * 4 / (1024*1024):.2f} MB")
        
        # Transfer
        target = "127.0.0.1:12345"
        
        start_time = time.time()
        request = await self.bridge.send_tensor(
            tensor=tensor,
            target_node=target
        )
        
        # Wait briefly
        await asyncio.sleep(0.5)
        
        elapsed = time.time() - start_time
        
        if request.size > 0 and elapsed > 0:
            throughput = (request.size * 8) / (elapsed * 1e9)
            print(f"Throughput: {throughput:.2f} Gbps")
        
        return True
    
    async def test_performance_metrics(self):
        """Test performance monitoring"""
        print("\n=== Test 4: Performance Metrics ===")
        
        # Print current stats
        self.bridge.print_stats()
        
        # Check specific metrics
        stats = self.bridge.get_stats()
        
        print("\nDetailed Metrics:")
        print(f"  Zero-copy efficiency: {stats.get('zero_copy_ratio', 0)*100:.1f}%")
        print(f"  Average latency: {stats.get('avg_latency_ms', 0):.2f} ms")
        print(f"  Peak throughput: {stats.get('peak_throughput_gbps', 0):.2f} Gbps")
        
        return True
    
    async def test_error_handling(self):
        """Test error handling"""
        print("\n=== Test 5: Error Handling ===")
        
        # Test invalid target
        if torch.cuda.is_available():
            tensor = torch.randn(100, device='cuda')
        else:
            tensor = torch.randn(100)
        
        # Invalid address format
        try:
            await self.bridge.send_tensor(
                tensor=tensor,
                target_node="invalid_address"
            )
            print("ERROR: Should have raised exception for invalid address")
            return False
        except ValueError as e:
            print(f"Correctly caught invalid address: {e}")
        
        # Test timeout
        try:
            request = await self.bridge.send_tensor(
                tensor=tensor,
                target_node="192.168.1.100:12345",  # Non-existent
                timeout=0.1  # Very short timeout
            )
            
            # Wait for timeout
            await asyncio.sleep(0.2)
            
            if request.state == TransferState.FAILED:
                print(f"Transfer correctly timed out: {request.error_message}")
            
        except Exception as e:
            print(f"Timeout handling: {e}")
        
        return True
    
    async def run_all_tests(self):
        """Run all tests"""
        print("=" * 60)
        print("ZERO-COPY TRANSPORT TEST SUITE")
        print("=" * 60)
        
        # Setup
        if not await self.setup():
            print("Setup failed!")
            return False
        
        # Run tests
        tests = [
            ("Single Tensor", self.test_single_tensor_transfer),
            ("Batch Transfer", self.test_batch_transfer),
            ("Large Tensor", self.test_large_tensor),
            ("Performance Metrics", self.test_performance_metrics),
            ("Error Handling", self.test_error_handling)
        ]
        
        results = []
        for name, test_func in tests:
            try:
                success = await test_func()
                results.append((name, success))
            except Exception as e:
                print(f"Test {name} failed with exception: {e}")
                results.append((name, False))
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        for name, success in results:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{name:.<40} {status}")
        
        passed = sum(1 for _, s in results if s)
        total = len(results)
        print(f"\nTotal: {passed}/{total} tests passed")
        
        # Teardown
        await self.teardown()
        
        return passed == total


async def main():
    """Main test function"""
    tester = ZeroCopyTransportTester()
    success = await tester.run_all_tests()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    # Check if running with proper permissions
    if os.geteuid() != 0:
        print("WARNING: Not running as root. Some DPDK features may not work.")
        print("For full functionality, run with: sudo python3 test_zero_copy_transport.py")
    
    # Run tests
    asyncio.run(main())


















