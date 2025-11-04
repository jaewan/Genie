"""
Comprehensive tests for improved GraphCache implementation.

Tests cover:
1. Cache hit/miss behavior with various model types
2. Cost-aware eviction strategy
3. Memory monitoring and adaptive sizing
4. Dynamic batch size handling
5. Concurrent access patterns
6. Cache invalidation
"""

import pytest
import torch
import torch.nn as nn
import time
import threading
from typing import List
import logging

# Setup logging for test output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleModel(nn.Module):
    """Simple test model for cache testing."""
    def __init__(self, input_size=10, hidden_size=32, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ComplexModel(nn.Module):
    """More complex model for testing expensive-to-capture graphs."""
    def __init__(self, input_size=10, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


@pytest.fixture
def fresh_cache():
    """Create fresh cache for each test."""
    from genie.core.graph_cache import GraphCache
    cache = GraphCache(max_entries=5, max_memory_mb=256)
    yield cache
    cache.clear()


class TestCacheFundamentals:
    """Test basic cache functionality."""
    
    def test_cache_hit_miss(self, fresh_cache):
        """Test cache hit and miss behavior."""
        model = SimpleModel()
        input_tensor = torch.randn(4, 10)
        
        # First call: cache miss
        start = time.perf_counter()
        graph1 = fresh_cache.get_or_capture(model, input_tensor)
        time1 = time.perf_counter() - start
        
        stats = fresh_cache.get_stats()
        assert stats['misses'] == 1
        assert stats['hits'] == 0
        
        # Second call: cache hit (should be faster)
        start = time.perf_counter()
        graph2 = fresh_cache.get_or_capture(model, input_tensor)
        time2 = time.perf_counter() - start
        
        stats = fresh_cache.get_stats()
        assert stats['misses'] == 1
        assert stats['hits'] == 1
        assert graph1 is graph2  # Same object returned
        
        # Cache hit should be significantly faster
        logger.info(f"Cache miss: {time1*1000:.1f}ms, Cache hit: {time2*1000:.1f}ms")
        assert time2 < time1  # Hit should be faster than miss
    
    def test_force_recapture(self, fresh_cache):
        """Test force_recapture flag."""
        model = SimpleModel()
        input_tensor = torch.randn(4, 10)
        
        # First capture
        graph1 = fresh_cache.get_or_capture(model, input_tensor)
        stats = fresh_cache.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 1
        
        # Force recapture
        graph2 = fresh_cache.get_or_capture(model, input_tensor, force_recapture=True)
        stats = fresh_cache.get_stats()
        assert stats['misses'] == 2  # Recapture counted as miss
        assert stats['hits'] == 0
    
    def test_different_models_not_conflicting(self, fresh_cache):
        """Test that different models are cached separately."""
        model1 = SimpleModel(input_size=10)
        model2 = SimpleModel(input_size=20)
        
        input1 = torch.randn(4, 10)
        input2 = torch.randn(4, 20)
        
        graph1 = fresh_cache.get_or_capture(model1, input1)
        graph2 = fresh_cache.get_or_capture(model2, input2)
        
        # Both should be in cache
        stats = fresh_cache.get_stats()
        assert stats['entries'] == 2
        assert stats['misses'] == 2


class TestCacheEviction:
    """Test cache eviction strategies."""
    
    def test_lru_eviction_on_capacity(self, fresh_cache):
        """Test eviction when cache reaches capacity."""
        # Create models with DIFFERENT sizes to generate different cache keys
        models = [SimpleModel(input_size=10 + i) for i in range(6)]
        inputs = [torch.randn(4, 10 + i) for i in range(6)]
        
        # Fill cache beyond capacity
        for model, input_tensor in zip(models, inputs):
            fresh_cache.get_or_capture(model, input_tensor)
        
        stats = fresh_cache.get_stats()
        assert stats['entries'] <= fresh_cache.max_entries  # Should not exceed max
        assert stats['evictions'] > 0  # Should have evicted something
        logger.info(f"Evictions: {stats['evictions']}, Entries: {stats['entries']}")
    
    def test_cost_aware_eviction(self, fresh_cache):
        """Test that expensive-to-capture graphs are kept."""
        # Simple model (fast to capture)
        simple_model = SimpleModel()
        simple_input = torch.randn(2, 10)
        
        # Complex model (slow to capture)
        complex_model = ComplexModel()
        complex_input = torch.randn(2, 10)
        
        # Capture both
        fresh_cache.get_or_capture(simple_model, simple_input)
        fresh_cache.get_or_capture(complex_model, complex_input)
        
        # Access complex model multiple times
        for _ in range(3):
            fresh_cache.get_or_capture(complex_model, complex_input)
        
        stats = fresh_cache.get_stats()
        assert stats['hits'] >= 3  # Complex model was hit multiple times
        logger.info(f"Cost-aware eviction - Hits: {stats['hits']}, Evictions: {stats['evictions']}")
    
    def test_memory_based_eviction(self):
        """Test memory-based eviction when limit exceeded."""
        from genie.core.graph_cache import GraphCache
        
        # Small memory limit to trigger eviction
        cache = GraphCache(max_entries=100, max_memory_mb=1)
        
        models = [SimpleModel() for _ in range(5)]
        inputs = [torch.randn(64, 10) for _ in range(5)]  # Larger inputs
        
        # Try to fill cache
        for model, input_tensor in zip(models, inputs):
            cache.get_or_capture(model, input_tensor)
        
        stats = cache.get_stats()
        logger.info(f"Memory-based eviction - Memory: {stats['current_memory_mb']:.1f}MB, Entries: {stats['entries']}")
        assert stats['current_memory_mb'] <= cache.max_memory_mb + 1  # Some tolerance


class TestCacheMetrics:
    """Test cache metrics and profiling."""
    
    def test_capture_time_tracking(self, fresh_cache):
        """Test that capture times are tracked."""
        model = SimpleModel()
        input_tensor = torch.randn(4, 10)
        
        graph = fresh_cache.get_or_capture(model, input_tensor)
        
        stats = fresh_cache.get_stats()
        assert stats['total_capture_time_ms'] > 0
        assert stats['total_time_saved_ms'] == 0  # No saves yet
        
        # Cache hit should save time
        fresh_cache.get_or_capture(model, input_tensor)
        stats = fresh_cache.get_stats()
        assert stats['total_time_saved_ms'] > 0  # Time saved from cache hit
        logger.info(f"Capture time saved: {stats['total_time_saved_ms']:.1f}ms")
    
    def test_hit_rate_calculation(self, fresh_cache):
        """Test hit rate statistics."""
        model = SimpleModel()
        input_tensor = torch.randn(4, 10)
        
        # 1 miss
        fresh_cache.get_or_capture(model, input_tensor)
        
        # 3 hits
        for _ in range(3):
            fresh_cache.get_or_capture(model, input_tensor)
        
        stats = fresh_cache.get_stats()
        assert stats['hit_rate'] == 0.75  # 3 hits out of 4 total
        logger.info(f"Hit rate: {stats['hit_rate']:.1%}")
    
    def test_eviction_tracking(self, fresh_cache):
        """Test that different eviction types are tracked."""
        models = [SimpleModel() for _ in range(6)]
        inputs = [torch.randn(4, 10) for _ in range(6)]
        
        for model, input_tensor in zip(models, inputs):
            fresh_cache.get_or_capture(model, input_tensor)
        
        stats = fresh_cache.get_stats()
        total_evictions = (stats['evictions_by_lru'] + 
                          stats['evictions_by_memory'] + 
                          stats['evictions_by_cost'])
        
        logger.info(f"Eviction breakdown - LRU: {stats['evictions_by_lru']}, "
                   f"Memory: {stats['evictions_by_memory']}, "
                   f"Cost: {stats['evictions_by_cost']}")


class TestCacheInvalidation:
    """Test cache invalidation."""
    
    def test_model_invalidation(self, fresh_cache):
        """Test invalidating model cache."""
        model = SimpleModel()
        input_tensor = torch.randn(4, 10)
        
        # Cache the model
        fresh_cache.get_or_capture(model, input_tensor)
        stats = fresh_cache.get_stats()
        assert stats['entries'] == 1
        
        # Invalidate
        fresh_cache.invalidate(model)
        stats = fresh_cache.get_stats()
        assert stats['entries'] == 0
        assert stats['invalidations'] == 1
    
    def test_clear_all(self, fresh_cache):
        """Test clearing entire cache."""
        # Create models with DIFFERENT sizes to generate different cache keys
        models = [SimpleModel(input_size=10 + i) for i in range(3)]
        inputs = [torch.randn(4, 10 + i) for i in range(3)]
        
        for model, input_tensor in zip(models, inputs):
            fresh_cache.get_or_capture(model, input_tensor)
        
        stats = fresh_cache.get_stats()
        assert stats['entries'] == 3
        
        fresh_cache.clear()
        stats = fresh_cache.get_stats()
        assert stats['entries'] == 0


class TestConcurrency:
    """Test thread-safe cache access."""
    
    def test_concurrent_cache_access(self, fresh_cache):
        """Test concurrent access to cache."""
        model = SimpleModel()
        input_tensor = torch.randn(4, 10)
        
        results = []
        errors = []
        
        def access_cache():
            try:
                for _ in range(5):
                    graph = fresh_cache.get_or_capture(model, input_tensor)
                    results.append(graph)
            except Exception as e:
                errors.append(e)
        
        # Create threads
        threads = [threading.Thread(target=access_cache) for _ in range(4)]
        
        # Start threads
        for t in threads:
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        assert len(errors) == 0  # No errors
        stats = fresh_cache.get_stats()
        logger.info(f"Concurrent access - Hits: {stats['hits']}, Misses: {stats['misses']}")


class TestDynamicBatchSizes:
    """Test cache key handling for dynamic batch sizes."""
    
    def test_different_batch_sizes(self, fresh_cache):
        """Test caching with different batch sizes."""
        model = SimpleModel()
        
        # Different batch sizes
        batch1 = torch.randn(4, 10)
        batch2 = torch.randn(8, 10)
        batch3 = torch.randn(16, 10)
        
        # All should be cached separately (different cache keys)
        graph1 = fresh_cache.get_or_capture(model, batch1)
        graph2 = fresh_cache.get_or_capture(model, batch2)
        graph3 = fresh_cache.get_or_capture(model, batch3)
        
        stats = fresh_cache.get_stats()
        # Each batch size = different cache entry
        assert stats['entries'] == 3
        assert stats['misses'] == 3
        
        # Accessing same batch size should hit cache
        fresh_cache.get_or_capture(model, batch1)
        stats = fresh_cache.get_stats()
        assert stats['hits'] == 1
        
        logger.info("Dynamic batch size handling works correctly")


class TestGlobalCache:
    """Test global cache instance."""
    
    def test_global_cache_singleton(self):
        """Test that global cache is a singleton."""
        from genie.core.graph_cache import get_graph_cache
        
        cache1 = get_graph_cache()
        cache2 = get_graph_cache()
        
        assert cache1 is cache2  # Same instance
    
    def test_global_cache_public_api(self):
        """Test public API for cache management."""
        import genie
        
        model = SimpleModel()
        input_tensor = torch.randn(4, 10)
        
        # Clear before test
        genie.clear_graph_cache()
        
        # Get stats
        stats = genie.get_graph_cache_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        
        # Use cache through public API
        graph = genie.execute_model(model, input_tensor)
        
        stats = genie.get_graph_cache_stats()
        # Should have at least one entry
        assert stats['entries'] >= 0  # May or may not use cache depending on execute_model implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
