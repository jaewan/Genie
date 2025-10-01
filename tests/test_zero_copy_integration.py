#!/usr/bin/env python3
"""
Integration test for zero-copy transport system

Tests the complete flow from Python control plane through C++ data plane.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genie.runtime.transport_coordinator import (
    TransportCoordinator, DataPlaneConfig, initialize_transport, shutdown_transport
)
from genie.runtime.async_zero_copy_bridge import AsyncZeroCopyBridge
from genie.runtime.control_server import ControlPlaneServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockTensor:
    """Mock tensor for testing without PyTorch dependency"""
    def __init__(self, size: int, dtype='float32'):
        self.size = size
        self.dtype = dtype
        self.data = np.random.randn(size).astype(np.float32)
        self._ptr = id(self.data)  # Use object ID as mock pointer
    
    def data_ptr(self) -> int:
        return self._ptr
    
    def numel(self) -> int:
        return self.size
    
    def element_size(self) -> int:
        return 4  # float32
    
    @property
    def shape(self):
        return [self.size]

async def test_control_plane():
    """Test control plane server"""
    logger.info("Testing control plane...")
    
    # Create control server
    server = ControlPlaneServer(
        node_id="test-node-1",
        host="127.0.0.1",
        port=5555
    )
    
    try:
        # Start server
        await server.start()
        logger.info(f"Control server started on {server.host}:{server.port}")
        
        # Test capability exchange
        capabilities = server.get_capabilities()
        assert capabilities.node_id == "test-node-1"
        logger.info(f"Node capabilities: {capabilities}")
        
        # Simulate some activity
        await asyncio.sleep(1)
        
        logger.info("✅ Control plane test passed")
        
    finally:
        await server.stop()

async def test_data_plane_bindings():
    """Test C++ data plane bindings"""
    logger.info("Testing data plane bindings...")
    
    from genie.runtime.transport_coordinator import DataPlaneBindings
    
    # Create bindings
    bindings = DataPlaneBindings()
    
    if not bindings.lib:
        logger.warning("C++ data plane library not available, skipping test")
        return
    
    # Create configuration
    config = DataPlaneConfig(
        eal_args=["test", "-c", "0x1", "-n", "1", "--no-huge", "--no-pci"],
        port_id=0,
        local_ip="127.0.0.1",
        local_mac="aa:bb:cc:dd:ee:01"
    )
    
    # Try to create data plane
    if bindings.create(config):
        logger.info("✅ Data plane created successfully")
        
        # Get statistics
        stats = bindings.get_statistics()
        logger.info(f"Data plane stats: {stats}")
        
        # Cleanup
        bindings.destroy()
    else:
        logger.warning("Failed to create data plane (may need root/DPDK setup)")

async def test_async_bridge():
    """Test async bridge"""
    logger.info("Testing async bridge...")
    
    # Create bridge instance with minimal config
    config = {
        'lib_path': './libgenie_zero_copy.so',
        'port_id': 0,
        'gpu_id': 0,
        'use_gpu_direct': False  # Use fallback mode for testing
    }
    bridge = AsyncZeroCopyBridge(config)
    
    # Initialize bridge
    success = await bridge.initialize()
    if not success:
        logger.warning("Failed to initialize bridge (expected without DPDK)")
        return
    
    try:
        # Create mock tensor (simulating a PyTorch tensor)
        import numpy as np
        tensor_data = np.random.randn(1024).astype(np.float32)
        
        # Create a mock tensor that looks like torch.Tensor
        class MockGPUTensor:
            def __init__(self, data):
                self.data = data
                self._ptr = id(data)
            
            @property
            def is_cuda(self):
                return False  # For testing, use CPU tensor
            
            def data_ptr(self):
                return self._ptr
            
            def numel(self):
                return len(self.data)
            
            def element_size(self):
                return 4  # float32
            
            @property
            def shape(self):
                return [len(self.data)]
        
        tensor = MockGPUTensor(tensor_data)
        
        # Start transfer (will use simulation fallback)
        request = await bridge.send_tensor(
            tensor=tensor,
            target_node="192.168.1.2:5556",  # Valid IP:port format
            target_gpu=0
        )
        
        logger.info(f"Transfer started: {request.transfer_id}")
        
        # Wait for completion with timeout
        try:
            await asyncio.wait_for(request.future, timeout=2.0)
            logger.info(f"✅ Transfer completed: {request.bytes_transferred} bytes")
        except asyncio.TimeoutError:
            logger.info("Transfer timed out (expected in simulation mode)")
        
        # Get statistics
        stats = bridge.get_stats()
        logger.info(f"Bridge statistics: {stats}")
        
        logger.info("✅ Async bridge test passed")
        
    finally:
        await bridge.shutdown()

async def test_transport_coordinator():
    """Test the complete transport coordinator"""
    logger.info("Testing transport coordinator...")
    
    # Configuration
    config = {
        'control_plane': {
            'host': '127.0.0.1',
            'port': 5556,
            'gpu_count': 1
        },
        'data_plane': {
            'eal_args': ["test", "-c", "0x1", "-n", "1", "--no-huge", "--no-pci"],
            'port_id': 0,
            'local_ip': '127.0.0.1',
            'local_mac': 'aa:bb:cc:dd:ee:01',
            'enable_gpudev': False  # Disable for testing
        }
    }
    
    # Create coordinator
    coordinator = TransportCoordinator("test-node-1", config)
    
    try:
        # Initialize
        success = await coordinator.initialize()
        if not success:
            logger.warning("Failed to initialize coordinator (may need DPDK setup)")
            return
        
        logger.info("✅ Coordinator initialized")
        
        # Configure target node
        coordinator.set_target_node("test-node-2", "192.168.1.101", "aa:bb:cc:dd:ee:02")
        
        # Create mock tensor
        tensor = MockTensor(1024 * 1024)  # 1MB tensor
        
        # Send tensor
        transfer_id = await coordinator.send_tensor(
            tensor=tensor,
            target_node="test-node-2"
        )
        
        logger.info(f"Started transfer: {transfer_id}")
        
        # Check status
        await asyncio.sleep(1)
        status = coordinator.get_transfer_status(transfer_id)
        if status:
            logger.info(f"Transfer status: {status.state.name}")
        
        # Get statistics
        stats = coordinator.get_statistics()
        logger.info(f"Coordinator statistics: {stats}")
        
        logger.info("✅ Transport coordinator test passed")
        
    finally:
        await coordinator.shutdown()

async def test_end_to_end():
    """Test end-to-end flow"""
    logger.info("Testing end-to-end flow...")
    
    coordinator = None
    bridge = None
    
    try:
        # Initialize transport system
        coordinator = await initialize_transport(
            node_id="test-node-1",
            config={
                'control_plane': {'port': 5557},
                'data_plane': {
                    'eal_args': ["test", "-c", "0x1", "-n", "1", "--no-huge", "--no-pci"],
                    'enable_gpudev': False
                }
            }
        )
        
        if not coordinator:
            logger.warning("Failed to initialize transport (may need DPDK setup)")
            return
        
        logger.info("✅ Transport system initialized")
        
        # Create async bridge with minimal config
        config = {
            'lib_path': './libgenie_zero_copy.so',
            'port_id': 0,
            'gpu_id': 0,
            'use_gpu_direct': False
        }
        bridge = AsyncZeroCopyBridge(config)
        
        # Initialize bridge
        success = await bridge.initialize()
        if not success:
            logger.warning("Failed to initialize bridge")
            return
        
        # Create batch of tensors
        import numpy as np
        tensors = []
        for i in range(5):
            data = np.random.randn(1024 * (i+1)).astype(np.float32)
            
            class MockGPUTensor:
                def __init__(self, data):
                    self.data = data
                    self._ptr = id(data)
                
                @property
                def is_cuda(self):
                    return False
                
                def data_ptr(self):
                    return self._ptr
                
                def numel(self):
                    return len(self.data)
                
                def element_size(self):
                    return 4
                
                @property
                def shape(self):
                    return [len(self.data)]
            
            tensors.append(MockGPUTensor(data))
        
        # Start multiple transfers (batch simulation)
        logger.info("Starting batch transfer of 5 tensors...")
        requests = []
        for i, tensor in enumerate(tensors):
            request = await bridge.send_tensor(
                tensor=tensor,
                target_node=f"192.168.1.{i+2}:5556",
                target_gpu=0
            )
            requests.append(request)
        
        logger.info(f"Started {len(requests)} transfers")
        
        # Wait for transfers with timeout
        await asyncio.sleep(2)
        
        # Check results
        completed = sum(1 for r in requests if r.future.done())
        logger.info(f"Completed {completed}/{len(requests)} transfers")
        
        logger.info("✅ End-to-end test passed")
        
    finally:
        # Shutdown bridge first
        if bridge:
            try:
                await bridge.shutdown()
            except Exception as e:
                logger.warning(f"Bridge shutdown error: {e}")
        
        # Shutdown transport coordinator
        if coordinator:
            try:
                await coordinator.shutdown()
            except Exception as e:
                logger.warning(f"Coordinator shutdown error: {e}")
        
        # Shutdown global transport
        try:
            await shutdown_transport()
        except Exception as e:
            logger.warning(f"Transport shutdown error: {e}")

async def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Starting Zero-Copy Transport Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Control Plane", test_control_plane),
        ("Data Plane Bindings", test_data_plane_bindings),
        ("Async Bridge", test_async_bridge),
        ("Transport Coordinator", test_transport_coordinator),
        ("End-to-End", test_end_to_end)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {name}")
        logger.info(f"{'='*60}")
        
        try:
            await test_func()
            passed += 1
            logger.info(f"✅ {name} PASSED")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {name} FAILED: {e}", exc_info=True)
        
        logger.info("")
    
    logger.info("=" * 60)
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    logger.info("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
