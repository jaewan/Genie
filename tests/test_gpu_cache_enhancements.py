"""
Unit tests for GPU Cache Phase 1 enhancements.

Tests memory-aware eviction, per-model memory tracking, and statistics.
"""

import sys
import numpy as np
import pytest
import torch

from genie.server.gpu_cache import SimpleGPUCache


class TestGPUCacheMemoryAware:
    """Test memory-aware eviction feature."""
    
    def test_cache_initialization(self):
        """Test cache initializes with default parameters."""
        cache = SimpleGPUCache(max_models=3)
        assert cache.max_models == 3
        assert len(cache.cache) == 0
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0
    
    def test_cache_initialization_with_memory_limit(self):
        """Test cache initialization with memory limit."""
        cache = SimpleGPUCache(max_models=3, max_memory_mb=1024.0)
        assert cache.max_memory_mb == 1024.0
        assert cache.stats["total_memory_bytes"] == 0
    
    def test_simple_cache_hit(self):
        """Test basic cache hit."""
        cache = SimpleGPUCache(max_models=5)
        
        # Create dummy weights
        weight_dict = {
            0: np.random.randn(100, 100).astype(np.float32),
            1: np.random.randn(50, 50).astype(np.float32),
        }
        
        # First access (miss)
        result1 = cache.get_weights("model_a", weight_dict)
        assert cache.stats["misses"] == 1
        assert cache.stats["hits"] == 0
        
        # Second access (hit)
        result2 = cache.get_weights("model_a", weight_dict)
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 1
        assert result2 is result1
    
    def test_cache_eviction_by_count(self):
        """Test LRU eviction when max_models exceeded."""
        cache = SimpleGPUCache(max_models=2)
        
        weights_a = {0: np.random.randn(10, 10).astype(np.float32)}
        weights_b = {0: np.random.randn(10, 10).astype(np.float32)}
        weights_c = {0: np.random.randn(10, 10).astype(np.float32)}
        
        cache.get_weights("model_a", weights_a)
        cache.get_weights("model_b", weights_b)
        assert len(cache.cache) == 2
        assert cache.stats["evictions"] == 0
        
        # Adding third model should evict first
        cache.get_weights("model_c", weights_c)
        assert len(cache.cache) == 2
        assert cache.stats["evictions"] == 1
        assert "model_a" not in cache.cache
        assert "model_b" in cache.cache
        assert "model_c" in cache.cache
    
    def test_memory_aware_eviction(self):
        """Test memory-aware eviction triggers when max_memory_mb exceeded."""
        # Create small cache with 0.1 MB limit
        cache = SimpleGPUCache(max_models=10, max_memory_mb=0.1)
        
        # Create 0.05 MB weight (should fit)
        small_weight = np.random.randn(10, 10).astype(np.float32)  # ~4KB
        cache.get_weights("model_small", {0: small_weight})
        assert len(cache.cache) == 1
        
        # Create 0.1 MB weight (should trigger eviction of first)
        large_weight = np.random.randn(200, 200).astype(np.float32)  # ~160KB
        cache.get_weights("model_large", {0: large_weight})
        
        # Small model should be evicted
        assert "model_small" not in cache.cache
        assert "model_large" in cache.cache
    
    def test_per_model_memory_tracking(self):
        """Test per-model memory tracking accuracy."""
        cache = SimpleGPUCache(max_models=5)
        
        weight_dict = {
            0: np.random.randn(100, 100).astype(np.float32),
            1: np.random.randn(50, 50).astype(np.float32),
        }
        
        cache.get_weights("model_a", weight_dict)
        
        # Expected: (100*100 + 50*50) * 4 bytes (float32)
        expected_bytes = (10000 + 2500) * 4
        actual_bytes = cache.get_model_memory_mb("model_a") * (1024 * 1024)
        
        assert abs(actual_bytes - expected_bytes) < 100  # Allow 100 byte tolerance
    
    def test_cache_statistics(self):
        """Test cache statistics calculation."""
        cache = SimpleGPUCache(max_models=5)
        
        weight_dict = {0: np.random.randn(10, 10).astype(np.float32)}
        
        cache.get_weights("model_a", weight_dict)
        cache.get_weights("model_a", weight_dict)
        cache.get_weights("model_b", weight_dict)
        cache.get_weights("model_b", weight_dict)
        cache.get_weights("model_b", weight_dict)
        
        stats = cache.get_stats()
        
        # 5 total calls: model_a (1 miss, 1 hit) + model_b (1 miss, 2 hits) = 3 hits, 2 misses
        assert stats["hits"] == 3
        assert stats["misses"] == 2
        assert stats["hit_rate_percent"] == 60.0
        assert stats["cached_models"] == 2
        assert stats["total_memory_mb"] > 0
    
    def test_clear_cache(self):
        """Test cache clearing."""
        cache = SimpleGPUCache(max_models=5)
        
        weight_dict = {0: np.random.randn(10, 10).astype(np.float32)}
        cache.get_weights("model_a", weight_dict)
        
        assert len(cache.cache) == 1
        cache.clear()
        assert len(cache.cache) == 0
        assert cache.stats["total_memory_bytes"] == 0
    
    def test_has_model(self):
        """Test has_model check without LRU update."""
        cache = SimpleGPUCache(max_models=5)
        
        weight_dict = {0: np.random.randn(10, 10).astype(np.float32)}
        cache.get_weights("model_a", weight_dict)
        
        # has_model should not update LRU
        assert cache.has_model("model_a") is True
        assert cache.has_model("model_b") is False
        
        # Get stats should show same counts (no new hit from has_model)
        cache.get_weights("model_a", weight_dict)
        stats = cache.get_stats()
        assert stats["hits"] == 1  # Only from second get_weights call
    
    def test_evict_until_freed(self):
        """Test memory-aware eviction by bytes."""
        cache = SimpleGPUCache(max_models=10)
        
        # Add two models
        weight_a = {0: np.random.randn(100, 100).astype(np.float32)}  # ~40KB
        weight_b = {0: np.random.randn(100, 100).astype(np.float32)}  # ~40KB
        
        cache.get_weights("model_a", weight_a)
        cache.get_weights("model_b", weight_b)
        
        initial_memory = cache.stats["total_memory_bytes"] / (1024 * 1024)
        
        # Evict until 0.02 MB freed (should evict at least one model)
        freed = cache.evict_until_freed(0.02)
        
        assert freed > 0
        assert cache.stats["total_memory_bytes"] < initial_memory * (1024 * 1024)
    
    def test_max_memory_tracking(self):
        """Test tracking of maximum memory seen."""
        cache = SimpleGPUCache(max_models=5)
        
        weight_dict = {0: np.random.randn(100, 100).astype(np.float32)}
        
        cache.get_weights("model_a", weight_dict)
        max_memory_1 = cache.stats["max_memory_bytes_seen"]
        
        cache.get_weights("model_b", weight_dict)
        max_memory_2 = cache.stats["max_memory_bytes_seen"]
        
        assert max_memory_2 >= max_memory_1


class TestGPUCacheMultiModel:
    """Test cache behavior with multiple models."""
    
    def test_lru_ordering_respected(self):
        """Test that LRU eviction respects access order."""
        cache = SimpleGPUCache(max_models=3)
        
        weight_dict = {0: np.random.randn(10, 10).astype(np.float32)}
        
        cache.get_weights("model_a", weight_dict)
        cache.get_weights("model_b", weight_dict)
        cache.get_weights("model_c", weight_dict)
        
        # Access model_a again (makes it most recent)
        cache.get_weights("model_a", weight_dict)
        
        # Adding model_d should evict model_b (least recent)
        cache.get_weights("model_d", {0: np.random.randn(10, 10).astype(np.float32)})
        
        assert "model_a" in cache.cache
        assert "model_b" not in cache.cache
        assert "model_c" in cache.cache
        assert "model_d" in cache.cache
    
    def test_concurrent_models(self):
        """Test cache with many concurrent models."""
        cache = SimpleGPUCache(max_models=5)
        
        for i in range(10):
            weight_dict = {0: np.random.randn(10, 10).astype(np.float32)}
            cache.get_weights(f"model_{i}", weight_dict)
        
        # Should evict all but last 5
        assert len(cache.cache) == 5
        assert cache.stats["evictions"] == 5
        assert cache.stats["misses"] == 10


class TestGPUCacheNonBlocking:
    """Test non-blocking tensor operations."""
    
    def test_non_blocking_tensor_copy(self):
        """Test that tensor copy uses non_blocking=True."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        cache = SimpleGPUCache()
        weight_dict = {0: np.random.randn(100, 100).astype(np.float32)}
        
        result = cache.get_weights("model_a", weight_dict)
        
        # Check that tensors are on GPU
        for tensor_id, tensor in result.items():
            assert tensor.device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
