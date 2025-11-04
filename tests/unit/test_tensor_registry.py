"""
Unit tests for SmartTensorRegistry.

Tests cover:
- Version-aware cache keys
- UUID-based handles
- Async concurrency safety
- Memory budget enforcement
- LRU eviction
- Ephemeral tensor filtering
- Statistics tracking
"""

import asyncio
import pytest
import torch
import uuid
from typing import Tuple

# Import the components to test
from genie.server.tensor_registry import (
    SmartTensorRegistry,
    RemoteHandle,
    TensorRegistryStats
)


@pytest.fixture
def registry():
    """Create a test registry with reasonable defaults."""
    return SmartTensorRegistry(
        max_cached_models=3,
        max_bytes_per_model=10 * 1024 * 1024,  # 10MB per model
        max_total_bytes=30 * 1024 * 1024  # 30MB total
    )


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(100, 512, dtype=torch.float32)


@pytest.fixture
def sample_handle(sample_tensor):
    """Create a sample remote handle."""
    return RemoteHandle(
        device_id="cuda:0",
        tensor_id=str(uuid.uuid4()),
        shape=sample_tensor.shape,
        dtype=sample_tensor.dtype,
        timestamp=0.0,
        version=1,
        tensor_bytes=sample_tensor.numel() * sample_tensor.element_size()
    )


class TestTensorRegistry:
    """Test SmartTensorRegistry basic functionality."""

    @pytest.mark.asyncio
    async def test_cache_miss(self, registry):
        """Test that cache miss returns correct status."""
        needs_transfer, handle = await registry.check_and_register(
            model_id="gpt2-tiny",
            tensor_name="layer.0.weight",
            model_version=1
        )
        
        assert needs_transfer is True
        assert handle is None
        assert registry.stats.misses == 1
        assert registry.stats.hits == 0

    @pytest.mark.asyncio
    async def test_cache_hit(self, registry, sample_tensor, sample_handle):
        """Test that cache hit returns handle."""
        # First, register a tensor
        await registry.check_and_register(
            model_id="gpt2-tiny",
            tensor_name="layer.0.weight",
            tensor=sample_tensor,
            model_version=1,
            remote_handle=sample_handle
        )
        
        # Now check if it's cached
        needs_transfer, handle = await registry.check_and_register(
            model_id="gpt2-tiny",
            tensor_name="layer.0.weight",
            tensor=sample_tensor,
            model_version=1
        )
        
        assert needs_transfer is False
        assert handle is not None
        assert handle.tensor_id == sample_handle.tensor_id
        assert registry.stats.hits == 1

    @pytest.mark.asyncio
    async def test_version_invalidation(self, registry, sample_tensor, sample_handle):
        """Test that version change invalidates cache."""
        # Register with version 1
        await registry.check_and_register(
            model_id="gpt2-tiny",
            tensor_name="layer.0.weight",
            tensor=sample_tensor,
            model_version=1,
            remote_handle=sample_handle
        )
        
        # Check cache with version 2
        needs_transfer, handle = await registry.check_and_register(
            model_id="gpt2-tiny",
            tensor_name="layer.0.weight",
            tensor=sample_tensor,
            model_version=2
        )
        
        # Should be a miss due to version change
        assert needs_transfer is True
        assert handle is None
        assert registry.stats.version_conflicts == 1

    @pytest.mark.asyncio
    async def test_ephemeral_tensor_never_cached(self, registry, sample_tensor, sample_handle):
        """Test that ephemeral tensors are never cached."""
        # Try to cache a tensor with 'activation' in the name
        needs_transfer, handle = await registry.check_and_register(
            model_id="gpt2-tiny",
            tensor_name="layer.0.hidden_state_activation",
            tensor=sample_tensor,
            model_version=1,
            remote_handle=sample_handle
        )
        
        # Should always need transfer (never cached)
        assert needs_transfer is True
        
        # Verify it wasn't cached
        stats = registry.get_stats()
        assert stats['total_tensors_cached'] == 0

    @pytest.mark.asyncio
    async def test_lru_eviction(self, registry, sample_tensor):
        """Test LRU eviction when max models exceeded."""
        handles = []
        
        # Cache 3 models
        for i in range(3):
            handle = RemoteHandle(
                device_id="cuda:0",
                tensor_id=str(uuid.uuid4()),
                shape=sample_tensor.shape,
                dtype=sample_tensor.dtype,
                timestamp=float(i),
                version=1,
                tensor_bytes=sample_tensor.numel() * sample_tensor.element_size()
            )
            handles.append(handle)
            
            await registry.check_and_register(
                model_id=f"model_{i}",
                tensor_name="weight",
                tensor=sample_tensor,
                model_version=1,
                remote_handle=handle
            )
        
        assert registry.stats.evictions == 0
        
        # Cache a 4th model (should evict oldest)
        handle4 = RemoteHandle(
            device_id="cuda:0",
            tensor_id=str(uuid.uuid4()),
            shape=sample_tensor.shape,
            dtype=sample_tensor.dtype,
            timestamp=3.0,
            version=1,
            tensor_bytes=sample_tensor.numel() * sample_tensor.element_size()
        )
        
        await registry.check_and_register(
            model_id="model_3",
            tensor_name="weight",
            tensor=sample_tensor,
            model_version=1,
            remote_handle=handle4
        )
        
        # Should have evicted model_0
        assert registry.stats.evictions == 1
        assert len(registry.registry) == 3  # 3 models cached

    @pytest.mark.asyncio
    async def test_memory_budget_enforcement(self, registry, sample_tensor):
        """Test that memory budgets are enforced."""
        # Create a large tensor that exceeds per-model budget
        large_tensor = torch.randn(5000, 5000, dtype=torch.float32)
        
        large_handle = RemoteHandle(
            device_id="cuda:0",
            tensor_id=str(uuid.uuid4()),
            shape=large_tensor.shape,
            dtype=large_tensor.dtype,
            timestamp=0.0,
            version=1,
            tensor_bytes=large_tensor.numel() * large_tensor.element_size()
        )
        
        # Try to register (should fail due to budget)
        needs_transfer, handle = await registry.check_and_register(
            model_id="gpt2-tiny",
            tensor_name="large_weight",
            tensor=large_tensor,
            model_version=1,
            remote_handle=large_handle
        )
        
        # Should indicate need transfer (registration rejected)
        assert needs_transfer is True

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, registry, sample_tensor, sample_handle):
        """Test that statistics are properly tracked."""
        # Do some operations
        await registry.check_and_register(
            model_id="gpt2-tiny",
            tensor_name="weight1",
            tensor=sample_tensor,
            model_version=1,
            remote_handle=sample_handle
        )
        
        # Cache hit
        await registry.check_and_register(
            model_id="gpt2-tiny",
            tensor_name="weight1",
            tensor=sample_tensor,
            model_version=1
        )
        
        # Cache miss
        await registry.check_and_register(
            model_id="gpt2-tiny",
            tensor_name="weight2",
            tensor=sample_tensor,
            model_version=1
        )
        
        stats = registry.get_stats()
        
        assert stats['hits'] == 1
        assert stats['misses'] == 2
        assert stats['hit_rate_percent'] == pytest.approx(33.33, rel=1)
        assert stats['bytes_saved'] > 0

    @pytest.mark.asyncio
    async def test_invalidate_model(self, registry, sample_tensor, sample_handle):
        """Test model invalidation."""
        # Register model
        await registry.check_and_register(
            model_id="gpt2-tiny",
            tensor_name="weight",
            tensor=sample_tensor,
            model_version=1,
            remote_handle=sample_handle
        )
        
        assert len(registry.registry) > 0
        
        # Invalidate model
        await registry.invalidate_model("gpt2-tiny")
        
        # Should be empty
        assert len(registry.registry) == 0
        assert registry.stats.invalidations == 1

    @pytest.mark.asyncio
    async def test_handle_validation(self, registry, sample_tensor):
        """Test that handles validate shape/dtype."""
        # Register with shape (100, 512)
        handle = RemoteHandle(
            device_id="cuda:0",
            tensor_id=str(uuid.uuid4()),
            shape=torch.Size([100, 512]),
            dtype=torch.float32,
            timestamp=0.0,
            version=1,
            tensor_bytes=100 * 512 * 4
        )
        
        await registry.check_and_register(
            model_id="gpt2-tiny",
            tensor_name="weight",
            tensor=sample_tensor,
            model_version=1,
            remote_handle=handle
        )
        
        # Query with different shape
        different_tensor = torch.randn(200, 512, dtype=torch.float32)
        needs_transfer, _ = await registry.check_and_register(
            model_id="gpt2-tiny",
            tensor_name="weight",
            tensor=different_tensor,
            model_version=1
        )
        
        # Should be a miss due to shape mismatch
        assert needs_transfer is True


class TestAsyncConcurrency:
    """Test concurrent access to tensor registry."""

    @pytest.mark.asyncio
    async def test_concurrent_registrations(self, registry, sample_tensor):
        """Test concurrent cache registrations don't corrupt state."""
        
        async def register_tensor(model_id: str):
            handle = RemoteHandle(
                device_id="cuda:0",
                tensor_id=str(uuid.uuid4()),
                shape=sample_tensor.shape,
                dtype=sample_tensor.dtype,
                timestamp=0.0,
                version=1,
                tensor_bytes=sample_tensor.numel() * sample_tensor.element_size()
            )
            
            await registry.check_and_register(
                model_id=model_id,
                tensor_name="weight",
                tensor=sample_tensor,
                model_version=1,
                remote_handle=handle
            )
        
        # Run concurrent registrations
        await asyncio.gather(
            register_tensor("model_0"),
            register_tensor("model_1"),
            register_tensor("model_2")
        )
        
        # All should be registered
        assert len(registry.registry) == 3
        assert registry.stats.misses == 3

    @pytest.mark.asyncio
    async def test_concurrent_hits_and_misses(self, registry, sample_tensor, sample_handle):
        """Test concurrent cache hits and misses."""
        
        # Pre-populate cache
        await registry.check_and_register(
            model_id="model_0",
            tensor_name="weight",
            tensor=sample_tensor,
            model_version=1,
            remote_handle=sample_handle
        )
        
        async def hit_or_miss(model_id: str, hit: bool):
            await registry.check_and_register(
                model_id=model_id if hit else f"new_{model_id}",
                tensor_name="weight",
                tensor=sample_tensor,
                model_version=1
            )
        
        # Concurrent hits and misses
        # 2 hits on model_0, 2 misses on new_model_0
        await asyncio.gather(
            hit_or_miss("model_0", hit=True),
            hit_or_miss("model_0", hit=True),
            hit_or_miss("model_0", hit=False),  # Creates new_model_0 (miss)
            hit_or_miss("model_0", hit=False),  # Creates another new_model_0 (miss)
        )
        
        # Stats track: 1 initial miss (pre-populate) + 2 hits + 2 misses = 3 misses total, 2 hits
        stats = registry.get_stats()
        assert stats['hits'] == 2
        assert stats['misses'] == 3  # 1 from pre-populate + 2 from new models


class TestRemoteHandle:
    """Test RemoteHandle validation."""

    def test_handle_validation_with_matching_tensor(self, sample_tensor):
        """Test handle validates correctly with matching tensor."""
        handle = RemoteHandle(
            device_id="cuda:0",
            tensor_id=str(uuid.uuid4()),
            shape=sample_tensor.shape,
            dtype=sample_tensor.dtype,
            timestamp=0.0,
            version=1,
            tensor_bytes=sample_tensor.numel() * sample_tensor.element_size()
        )
        
        assert handle.is_valid(sample_tensor) is True

    def test_handle_validation_with_mismatched_shape(self, sample_tensor):
        """Test handle rejects mismatched shape."""
        handle = RemoteHandle(
            device_id="cuda:0",
            tensor_id=str(uuid.uuid4()),
            shape=torch.Size([200, 512]),
            dtype=sample_tensor.dtype,
            timestamp=0.0,
            version=1,
            tensor_bytes=100 * 512 * 4
        )
        
        assert handle.is_valid(sample_tensor) is False

    def test_handle_validation_with_mismatched_dtype(self, sample_tensor):
        """Test handle rejects mismatched dtype."""
        handle = RemoteHandle(
            device_id="cuda:0",
            tensor_id=str(uuid.uuid4()),
            shape=sample_tensor.shape,
            dtype=torch.float64,
            timestamp=0.0,
            version=1,
            tensor_bytes=sample_tensor.numel() * sample_tensor.element_size()
        )
        
        assert handle.is_valid(sample_tensor) is False

    def test_handle_validation_with_none_tensor(self):
        """Test handle always valid when tensor is None."""
        handle = RemoteHandle(
            device_id="cuda:0",
            tensor_id=str(uuid.uuid4()),
            shape=torch.Size([100, 512]),
            dtype=torch.float32,
            timestamp=0.0,
            version=1,
            tensor_bytes=100 * 512 * 4
        )
        
        assert handle.is_valid(None) is True
