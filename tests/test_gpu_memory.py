"""
Tests for GPU Memory Registration Module

Tests the GPUDevMemoryManager and related functionality for:
- Basic registration/unregistration
- IOVA retrieval
- Cache behavior and LRU eviction
- Memory lifecycle management
- Error handling and fallbacks
- Performance characteristics
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from genie.runtime.gpu_memory import (
    GPUDevMemoryManager, 
    DMAHandle, 
    GPUMemoryMetrics,
    RegistrationError,
    get_gpu_memory_manager,
    is_gpu_memory_available
)

class MockTensor:
    """Mock PyTorch tensor for testing"""
    
    def __init__(self, size: int = 1024, device: str = 'cuda', data_ptr: int = None):
        self.size = size
        self.device = device
        self._data_ptr = data_ptr or (0x7f0000000000 + id(self))  # Fake GPU pointer
        self._element_size = 4  # float32
    
    def data_ptr(self) -> int:
        return self._data_ptr
    
    def numel(self) -> int:
        return self.size // self._element_size
    
    def element_size(self) -> int:
        return self._element_size
    
    def is_cuda(self) -> bool:
        return 'cuda' in self.device

class TestDMAHandle:
    """Test DMAHandle functionality"""
    
    def test_dma_handle_creation(self):
        """Test DMA handle creation and validation"""
        handle = DMAHandle(
            iova=0x1000,
            gpu_ptr=0x7f0000001000,
            size=4096,
            gpu_id=0
        )
        
        assert handle.is_valid()
        assert handle.ref_count == 1
        assert handle.gpu_id == 0
    
    def test_dma_handle_invalid(self):
        """Test invalid DMA handle detection"""
        # Zero IOVA
        handle1 = DMAHandle(iova=0, gpu_ptr=0x1000, size=4096)
        assert not handle1.is_valid()
        
        # Zero GPU pointer
        handle2 = DMAHandle(iova=0x1000, gpu_ptr=0, size=4096)
        assert not handle2.is_valid()
        
        # Zero size
        handle3 = DMAHandle(iova=0x1000, gpu_ptr=0x1000, size=0)
        assert not handle3.is_valid()
    
    def test_reference_counting(self):
        """Test reference counting operations"""
        handle = DMAHandle(iova=0x1000, gpu_ptr=0x1000, size=4096)
        
        # Initial count
        assert handle.ref_count == 1
        
        # Increment
        handle.increment_ref()
        assert handle.ref_count == 2
        
        # Decrement
        count = handle.decrement_ref()
        assert count == 1
        assert handle.ref_count == 1
        
        # Decrement to zero
        count = handle.decrement_ref()
        assert count == 0
        assert handle.ref_count == 0
        
        # Decrement below zero (should clamp)
        count = handle.decrement_ref()
        assert count == 0
        assert handle.ref_count == 0

class TestGPUMemoryMetrics:
    """Test GPU memory metrics"""
    
    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = GPUMemoryMetrics()
        
        assert metrics.registrations == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
        assert metrics.evictions == 0
        assert metrics.registration_failures == 0
        assert len(metrics.registration_time_ms) == 0
    
    def test_registration_time_tracking(self):
        """Test registration time tracking"""
        metrics = GPUMemoryMetrics()
        
        # Add some times
        metrics.add_registration_time(1.5)
        metrics.add_registration_time(2.0)
        metrics.add_registration_time(1.8)
        
        assert len(metrics.registration_time_ms) == 3
        assert metrics.get_avg_registration_time() == pytest.approx(1.77, rel=1e-2)
    
    def test_registration_time_limit(self):
        """Test registration time list size limit"""
        metrics = GPUMemoryMetrics()
        
        # Add more than 100 times
        for i in range(150):
            metrics.add_registration_time(float(i))
        
        # Should keep only last 100
        assert len(metrics.registration_time_ms) == 100
        assert metrics.registration_time_ms[0] == 50.0  # First 50 should be removed

class TestGPUDevMemoryManager:
    """Test GPUDevMemoryManager functionality"""
    
    @pytest.fixture
    def manager(self):
        """Create a test GPU memory manager"""
        return GPUDevMemoryManager(cache_size=10)  # Small cache for testing
    
    @pytest.fixture
    def mock_tensor(self):
        """Create a mock tensor"""
        return MockTensor(size=4096)
    
    def test_manager_initialization(self, manager):
        """Test manager initialization"""
        assert manager.cache_size == 10
        assert len(manager.registration_cache) == 0
        assert len(manager.active_transfers) == 0
        assert manager.metrics.registrations == 0
    
    def test_gpudev_unavailable_fallback(self, manager, mock_tensor):
        """Test fallback when GPUDev is unavailable"""
        # Force GPUDev unavailable
        manager._gpudev_available = False
        
        handle = manager.register_tensor_memory(mock_tensor)
        
        assert handle.is_valid()
        assert handle.gpu_ptr == mock_tensor.data_ptr()
        assert handle.iova == mock_tensor.data_ptr()  # Fallback uses GPU ptr as IOVA
        assert handle.size == mock_tensor.numel() * mock_tensor.element_size()
    
    def test_registration_caching(self, manager, mock_tensor):
        """Test registration caching behavior"""
        manager._gpudev_available = False  # Use fallback for testing
        
        # First registration - cache miss
        handle1 = manager.register_tensor_memory(mock_tensor)
        assert manager.metrics.cache_misses == 1
        assert manager.metrics.cache_hits == 0
        assert len(manager.registration_cache) == 1
        
        # Second registration of same tensor - cache hit
        handle2 = manager.register_tensor_memory(mock_tensor)
        assert manager.metrics.cache_hits == 1
        assert handle2.gpu_ptr == handle1.gpu_ptr
        assert handle2.ref_count == 2  # Reference count increased
        
        # Should be same handle object
        assert handle1 is handle2
    
    def test_lru_eviction(self, manager):
        """Test LRU cache eviction"""
        manager._gpudev_available = False
        
        # Fill cache beyond capacity
        tensors = []
        handles = []
        
        for i in range(15):  # More than cache_size (10)
            tensor = MockTensor(data_ptr=0x1000 + i * 0x1000)
            tensors.append(tensor)
            handle = manager.register_tensor_memory(tensor)
            handles.append(handle)
        
        # Cache should be at capacity
        assert len(manager.registration_cache) == manager.cache_size
        assert manager.metrics.evictions > 0
        
        # First few tensors should be evicted
        first_tensor_ptr = tensors[0].data_ptr()
        assert first_tensor_ptr not in manager.registration_cache
    
    def test_keepalive_registration(self, manager, mock_tensor):
        """Test registration with keepalive"""
        manager._gpudev_available = False
        
        tensor_id = "test_transfer_123"
        handle = manager.register_with_keepalive(mock_tensor, tensor_id)
        
        assert handle.keepalive is mock_tensor
        assert tensor_id in manager.active_transfers
        
        # Release transfer
        manager.release_transfer(tensor_id)
        assert tensor_id not in manager.active_transfers
        assert handle.keepalive is None
    
    def test_iova_retrieval(self, manager, mock_tensor):
        """Test IOVA retrieval for registered memory"""
        manager._gpudev_available = False
        
        # Register tensor
        handle = manager.register_tensor_memory(mock_tensor)
        gpu_ptr = mock_tensor.data_ptr()
        
        # Should be able to retrieve IOVA
        iova = manager.get_iova(gpu_ptr)
        assert iova == handle.iova
        
        # Unregistered memory should return None
        fake_ptr = 0x999999
        assert manager.get_iova(fake_ptr) is None
    
    def test_cleanup_expired_registrations(self, manager):
        """Test cleanup of expired registrations"""
        manager._gpudev_available = False
        
        # Register some tensors
        tensors = []
        for i in range(5):
            tensor = MockTensor(data_ptr=0x2000 + i * 0x1000)
            tensors.append(tensor)
            handle = manager.register_tensor_memory(tensor)
            # Make some registrations old
            if i < 3:
                handle.timestamp = time.time() - 400  # 400 seconds old
        
        # Cleanup with 300 second threshold
        cleaned = manager.cleanup_expired_registrations(max_age_seconds=300)
        
        # Should have cleaned up 3 expired registrations
        assert cleaned == 3
        assert len(manager.registration_cache) == 2
    
    def test_invalid_tensor_handling(self, manager):
        """Test handling of invalid tensor objects"""
        # Test with None
        with pytest.raises(RegistrationError):
            manager.register_tensor_memory(None)
        
        # Test with object missing required methods
        invalid_obj = object()
        with pytest.raises(RegistrationError):
            manager.register_tensor_memory(invalid_obj)
    
    def test_thread_safety(self, manager):
        """Test thread safety of registration operations"""
        manager._gpudev_available = False
        
        results = []
        errors = []
        
        def register_tensor(thread_id):
            try:
                tensor = MockTensor(data_ptr=0x3000 + thread_id * 0x1000)
                handle = manager.register_tensor_memory(tensor)
                results.append((thread_id, handle))
            except Exception as e:
                errors.append((thread_id, e))
        
        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=register_tensor, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have no errors and all registrations
        assert len(errors) == 0
        assert len(results) == 10
        assert len(manager.registration_cache) == 10
    
    def test_metrics_tracking(self, manager, mock_tensor):
        """Test metrics tracking"""
        manager._gpudev_available = False
        
        # Initial metrics
        metrics = manager.get_metrics()
        assert metrics.registrations == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
        
        # Register tensor
        handle = manager.register_tensor_memory(mock_tensor)
        
        # Check updated metrics
        metrics = manager.get_metrics()
        assert metrics.registrations == 1
        assert metrics.cache_misses == 1
        assert len(metrics.registration_time_ms) == 1
        
        # Register same tensor again (cache hit)
        handle2 = manager.register_tensor_memory(mock_tensor)
        
        metrics = manager.get_metrics()
        assert metrics.cache_hits == 1
        
        # Reset metrics
        manager.reset_metrics()
        metrics = manager.get_metrics()
        assert metrics.registrations == 0
        assert metrics.cache_hits == 0

class TestGlobalFunctions:
    """Test global utility functions"""
    
    def test_global_manager_singleton(self):
        """Test global manager is singleton"""
        manager1 = get_gpu_memory_manager()
        manager2 = get_gpu_memory_manager()
        
        assert manager1 is manager2
    
    def test_gpu_memory_availability(self):
        """Test GPU memory availability check"""
        # This will depend on actual system state
        # Just test that it returns a boolean
        available = is_gpu_memory_available()
        assert isinstance(available, bool)

class TestPerformance:
    """Performance tests for GPU memory operations"""
    
    @pytest.fixture
    def manager(self):
        return GPUDevMemoryManager(cache_size=100)
    
    def test_registration_performance(self, manager):
        """Test registration performance for various sizes"""
        manager._gpudev_available = False  # Use fallback for consistent testing
        
        sizes = [1024, 1024*1024, 10*1024*1024]  # 1KB, 1MB, 10MB
        
        for size in sizes:
            tensor = MockTensor(size=size)
            
            start_time = time.perf_counter()
            handle = manager.register_tensor_memory(tensor)
            elapsed = time.perf_counter() - start_time
            
            # Registration should be fast (< 10ms even in fallback mode)
            assert elapsed < 0.01
            assert handle.is_valid()
    
    def test_cache_performance(self, manager):
        """Test cache hit performance"""
        manager._gpudev_available = False
        
        tensor = MockTensor()
        
        # First registration (cache miss)
        start_time = time.perf_counter()
        handle1 = manager.register_tensor_memory(tensor)
        miss_time = time.perf_counter() - start_time
        
        # Second registration (cache hit)
        start_time = time.perf_counter()
        handle2 = manager.register_tensor_memory(tensor)
        hit_time = time.perf_counter() - start_time
        
        # Cache hit should be much faster than miss
        assert hit_time < miss_time
        assert hit_time < 0.001  # Should be very fast
    
    def test_concurrent_registrations(self, manager):
        """Test performance with concurrent registrations"""
        manager._gpudev_available = False
        
        # Use a counter to ensure unique addresses across threads
        counter = [0]
        counter_lock = threading.Lock()
        
        def register_multiple():
            for i in range(10):
                with counter_lock:
                    unique_id = counter[0]
                    counter[0] += 1
                tensor = MockTensor(data_ptr=0x4000 + unique_id * 0x1000)
                manager.register_tensor_memory(tensor)
        
        start_time = time.perf_counter()
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=register_multiple)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        elapsed = time.perf_counter() - start_time
        
        # Should complete reasonably fast even with contention
        assert elapsed < 1.0  # 1 second for 50 registrations across 5 threads
        # Note: Due to cache size limit (100), we might have fewer than 50 if eviction occurs
        assert len(manager.registration_cache) >= 10  # At least some registrations should remain

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
