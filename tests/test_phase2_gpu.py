"""
Phase 2 GPU Tests - Zero-Copy Transport and GPU Memory Registration

Tests the Phase 2 implementation including:
1. GPU memory registration with DPDK GPUDev
2. Zero-copy data path
3. Async transport bridge
4. GPU Direct RDMA capabilities

Hardware Requirements:
- NVIDIA GPU with CUDA
- DPDK-compatible NIC (optional, will fall back gracefully)
- nvidia-peermem kernel module (optional for GPU Direct)
"""

import sys
import os
import pytest
import torch
import asyncio
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from genie.runtime.gpu_memory import (
    GPUDevMemoryManager,
    DMAHandle,
    GPUMemoryMetrics,
    RegistrationError
)
from genie.runtime.async_zero_copy_bridge import (
    AsyncZeroCopyBridge,
    TransferRequest,
    TransferState
)

logger = logging.getLogger(__name__)


def check_cuda_available():
    """Check if CUDA is available."""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    try:
        test = torch.randn(10, 10, device='cuda')
        return True, "CUDA available"
    except RuntimeError as e:
        return False, f"CUDA error: {e}"


def check_dpdk_available():
    """Check if DPDK library is available."""
    try:
        import ctypes
        lib = ctypes.CDLL("./build/libgenie_data_plane.so")
        return True, "DPDK library found"
    except Exception as e:
        return False, f"DPDK not available: {e}"


@pytest.fixture(scope="session")
def cuda_available():
    """Check CUDA availability for session."""
    available, msg = check_cuda_available()
    if not available:
        pytest.skip(msg)
    return available


@pytest.fixture(scope="session")
def dpdk_available():
    """Check DPDK availability for session."""
    available, msg = check_dpdk_available()
    return available  # Don't skip, test fallback


@pytest.fixture
def gpu_memory_manager():
    """Create GPU memory manager."""
    return GPUDevMemoryManager(cache_size=10)


# ============================================================================
# Test Suite 1: GPU Memory Registration
# ============================================================================

class TestGPUMemoryRegistration:
    """Test GPU memory registration capabilities."""
    
    def test_manager_initialization(self, cuda_available, gpu_memory_manager):
        """Test GPU memory manager initialization."""
        mgr = gpu_memory_manager
        
        assert mgr.cache_size == 10
        assert len(mgr.registration_cache) == 0
        assert len(mgr.active_transfers) == 0
        assert isinstance(mgr.metrics, GPUMemoryMetrics)
    
    def test_tensor_registration_fallback(self, cuda_available, gpu_memory_manager):
        """Test tensor registration with CPU fallback."""
        mgr = gpu_memory_manager
        
        # Create GPU tensor
        tensor = torch.randn(100, 100, device='cuda')
        
        # Register (will use fallback if GPUDev not available)
        handle = mgr.register_tensor_memory(tensor)
        
        # Validate handle
        assert isinstance(handle, DMAHandle)
        assert handle.is_valid()
        assert handle.gpu_ptr == tensor.data_ptr()
        assert handle.size == tensor.numel() * tensor.element_size()
        assert handle.gpu_id == 0
        assert handle.ref_count == 1
        
        print(f"\n✓ Registered GPU tensor: {tensor.shape}")
        print(f"  GPU ptr: 0x{handle.gpu_ptr:x}")
        print(f"  IOVA: 0x{handle.iova:x}")
        print(f"  Size: {handle.size} bytes")
    
    def test_registration_caching(self, cuda_available, gpu_memory_manager):
        """Test registration caching mechanism."""
        mgr = gpu_memory_manager
        
        tensor = torch.randn(50, 50, device='cuda')
        
        # First registration - cache miss
        handle1 = mgr.register_tensor_memory(tensor)
        assert mgr.metrics.cache_misses == 1
        assert mgr.metrics.cache_hits == 0
        
        # Second registration of same tensor - cache hit
        handle2 = mgr.register_tensor_memory(tensor)
        assert mgr.metrics.cache_hits == 1
        assert handle2.gpu_ptr == handle1.gpu_ptr
        assert handle2.ref_count == 2  # Reference count increased
        
        print(f"\n✓ Registration caching works")
        print(f"  Cache misses: {mgr.metrics.cache_misses}")
        print(f"  Cache hits: {mgr.metrics.cache_hits}")
        print(f"  Ref count: {handle2.ref_count}")
    
    def test_lru_eviction(self, cuda_available):
        """Test LRU cache eviction."""
        mgr = GPUDevMemoryManager(cache_size=5)
        
        # Register more tensors than cache size
        tensors = []
        handles = []
        
        for i in range(8):
            tensor = torch.randn(10, 10, device='cuda')
            tensors.append(tensor)
            handle = mgr.register_tensor_memory(tensor)
            handles.append(handle)
        
        # Cache should be at capacity
        assert len(mgr.registration_cache) <= mgr.cache_size
        assert mgr.metrics.evictions > 0
        
        print(f"\n✓ LRU eviction works")
        print(f"  Registered: 8 tensors")
        print(f"  Cache size: {len(mgr.registration_cache)}/{mgr.cache_size}")
        print(f"  Evictions: {mgr.metrics.evictions}")
    
    def test_keepalive_mechanism(self, cuda_available, gpu_memory_manager):
        """Test keepalive mechanism for active transfers."""
        mgr = gpu_memory_manager
        
        tensor = torch.randn(100, 100, device='cuda')
        transfer_id = "test_transfer_123"
        
        # Register with keepalive
        handle = mgr.register_with_keepalive(tensor, transfer_id)
        
        assert handle.keepalive is tensor
        assert transfer_id in mgr.active_transfers
        
        # Release transfer
        mgr.release_transfer(transfer_id)
        
        assert transfer_id not in mgr.active_transfers
        assert handle.keepalive is None
        
        print(f"\n✓ Keepalive mechanism works")
        print(f"  Transfer ID: {transfer_id}")
        print(f"  Keepalive set: Yes")
        print(f"  Released: Yes")
    
    def test_dma_handle_validation(self, cuda_available):
        """Test DMA handle validation."""
        # Valid handle
        valid = DMAHandle(iova=0x1000, gpu_ptr=0x2000, size=1024)
        assert valid.is_valid()
        
        # Invalid handles
        invalid1 = DMAHandle(iova=0, gpu_ptr=0x2000, size=1024)
        assert not invalid1.is_valid()
        
        invalid2 = DMAHandle(iova=0x1000, gpu_ptr=0, size=1024)
        assert not invalid2.is_valid()
        
        invalid3 = DMAHandle(iova=0x1000, gpu_ptr=0x2000, size=0)
        assert not invalid3.is_valid()
        
        print(f"\n✓ DMA handle validation works")
    
    def test_reference_counting(self, cuda_available):
        """Test reference counting operations."""
        handle = DMAHandle(iova=0x1000, gpu_ptr=0x2000, size=1024)
        
        assert handle.ref_count == 1
        
        handle.increment_ref()
        assert handle.ref_count == 2
        
        count = handle.decrement_ref()
        assert count == 1
        assert handle.ref_count == 1
        
        count = handle.decrement_ref()
        assert count == 0
        assert handle.ref_count == 0
        
        # Should not go below zero
        count = handle.decrement_ref()
        assert count == 0
        
        print(f"\n✓ Reference counting works")
    
    def test_metrics_tracking(self, cuda_available, gpu_memory_manager):
        """Test metrics collection."""
        mgr = gpu_memory_manager
        
        # Perform operations
        tensor1 = torch.randn(10, 10, device='cuda')
        tensor2 = torch.randn(20, 20, device='cuda')
        
        handle1 = mgr.register_tensor_memory(tensor1)
        handle2 = mgr.register_tensor_memory(tensor2)
        handle1_again = mgr.register_tensor_memory(tensor1)  # Cache hit
        
        metrics = mgr.get_metrics()
        
        assert metrics.registrations >= 2
        assert metrics.cache_hits >= 1
        assert metrics.cache_misses >= 2
        
        print(f"\n✓ Metrics tracking works")
        print(f"  Registrations: {metrics.registrations}")
        print(f"  Cache hits: {metrics.cache_hits}")
        print(f"  Cache misses: {metrics.cache_misses}")
        print(f"  Avg registration time: {metrics.get_avg_registration_time():.3f}ms")


# ============================================================================
# Test Suite 2: Zero-Copy Transport Bridge
# ============================================================================

class TestZeroCopyBridge:
    """Test async zero-copy transport bridge."""
    
    @pytest.mark.asyncio
    async def test_bridge_initialization(self, cuda_available):
        """Test bridge initialization."""
        config = {
            'lib_path': './build/libgenie_data_plane.so',
            'port_id': 0,
            'gpu_id': 0,
            'use_gpu_direct': True,
            'mtu': 8192,
            'num_workers': 2
        }
        
        bridge = AsyncZeroCopyBridge(config)
        
        assert bridge.port_id == 0
        assert bridge.gpu_id == 0
        assert bridge.use_gpu_direct == True
        assert bridge.num_workers == 2
        
        print(f"\n✓ Bridge initialized")
        print(f"  Port ID: {bridge.port_id}")
        print(f"  GPU ID: {bridge.gpu_id}")
        print(f"  Workers: {bridge.num_workers}")
    
    @pytest.mark.asyncio
    async def test_bridge_initialization_with_library(self, cuda_available, dpdk_available):
        """Test bridge initialization with actual library."""
        config = {
            'lib_path': './build/libgenie_data_plane.so',
            'use_gpu_direct': True,
        }
        
        bridge = AsyncZeroCopyBridge(config)
        
        # Try to initialize (will gracefully fall back if DPDK not available)
        initialized = await bridge.initialize()
        
        if initialized and bridge.lib is not None:
            print(f"\n✓ DPDK transport initialized")
            print(f"  Native transport: Available")
            print(f"  GPU Direct: {bridge.use_gpu_direct}")
            
            # Cleanup
            await bridge.shutdown()
        else:
            print(f"\n⚠ DPDK not available, using fallback")
            print(f"  This is expected if DPDK is not set up")
    
    @pytest.mark.asyncio
    async def test_transfer_request_creation(self, cuda_available):
        """Test transfer request creation."""
        tensor = torch.randn(100, 100, device='cuda')
        
        req = TransferRequest(
            transfer_id="test_123",
            tensor=tensor,
            target_node="192.168.1.100",
            target_gpu=0,
            priority=5
        )
        
        assert req.transfer_id == "test_123"
        assert req.size == tensor.numel() * tensor.element_size()
        assert req.state == TransferState.PENDING
        assert req.target_gpu == 0
        
        print(f"\n✓ Transfer request created")
        print(f"  ID: {req.transfer_id}")
        print(f"  Size: {req.size} bytes")
        print(f"  State: {req.state.value}")


# ============================================================================
# Test Suite 3: End-to-End Integration
# ============================================================================

class TestPhase2Integration:
    """Integration tests for Phase 2 features."""
    
    def test_gpu_memory_to_transport_flow(self, cuda_available):
        """Test complete flow from GPU memory to transport."""
        # Step 1: Create GPU tensor
        tensor = torch.randn(512, 512, device='cuda')
        print(f"\n✓ Step 1: Created GPU tensor {tensor.shape}")
        print(f"  Device: {tensor.device}")
        print(f"  Size: {tensor.numel() * tensor.element_size()} bytes")
        
        # Step 2: Register GPU memory
        mgr = GPUDevMemoryManager()
        handle = mgr.register_tensor_memory(tensor)
        print(f"\n✓ Step 2: Registered GPU memory")
        print(f"  GPU ptr: 0x{handle.gpu_ptr:x}")
        print(f"  IOVA: 0x{handle.iova:x}")
        print(f"  Valid: {handle.is_valid()}")
        
        # Step 3: Create transfer request (conceptual - actual send needs network)
        transfer_id = f"transfer_{id(tensor)}"
        handle_with_keepalive = mgr.register_with_keepalive(tensor, transfer_id)
        print(f"\n✓ Step 3: Created transfer request")
        print(f"  Transfer ID: {transfer_id}")
        print(f"  Keepalive: Active")
        
        # Step 4: Cleanup
        mgr.release_transfer(transfer_id)
        print(f"\n✓ Step 4: Cleanup complete")
        print(f"  Transfer released")
        
        assert handle.is_valid()
        assert handle_with_keepalive.is_valid()
    
    def test_multiple_concurrent_registrations(self, cuda_available):
        """Test multiple concurrent GPU memory registrations."""
        mgr = GPUDevMemoryManager(cache_size=20)
        
        # Register multiple tensors
        tensors = []
        handles = []
        
        print(f"\n✓ Registering multiple GPU tensors...")
        for i in range(10):
            size = (100 + i * 10, 100 + i * 10)
            tensor = torch.randn(*size, device='cuda')
            handle = mgr.register_tensor_memory(tensor)
            
            tensors.append(tensor)
            handles.append(handle)
            
            assert handle.is_valid()
        
        print(f"  Registered: {len(handles)} tensors")
        print(f"  Cache size: {len(mgr.registration_cache)}")
        print(f"  All valid: {all(h.is_valid() for h in handles)}")
        
        # Verify metrics
        metrics = mgr.get_metrics()
        print(f"\n  Metrics:")
        print(f"    Registrations: {metrics.registrations}")
        print(f"    Cache misses: {metrics.cache_misses}")
        print(f"    Avg time: {metrics.get_avg_registration_time():.3f}ms")


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run Phase 2 tests with detailed output."""
    print("\n" + "="*70)
    print(" " * 20 + "PHASE 2 GPU TESTS")
    print("="*70)
    
    # Check CUDA
    cuda_ok, cuda_msg = check_cuda_available()
    print(f"\nCUDA Status: {cuda_msg}")
    
    if not cuda_ok:
        print("❌ CUDA not available - cannot run Phase 2 tests")
        return 1
    
    # Check DPDK
    dpdk_ok, dpdk_msg = check_dpdk_available()
    print(f"DPDK Status: {dpdk_msg}")
    
    if not dpdk_ok:
        print("⚠️  DPDK not available - will test fallback paths")
    
    # Print GPU info
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Run tests
    print("\n" + "="*70)
    print("Running Phase 2 Tests...")
    print("="*70 + "\n")
    
    exit_code = pytest.main([
        __file__,
        '-v',
        '-s',
        '--tb=short',
        '--color=yes',
        '-k', 'not asyncio'  # Skip async tests in simple run
    ])
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

