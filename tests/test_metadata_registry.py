"""Tests for Metadata Registry (Refactoring #2)."""

import pytest
import gc
from genie.semantic.metadata_registry import (
    MetadataRegistry, SemanticMetadata, get_metadata_registry
)


def test_registry_singleton():
    """Test registry is a singleton."""
    reg1 = MetadataRegistry()
    reg2 = MetadataRegistry()
    assert reg1 is reg2


def test_register_and_get_metadata():
    """Test registering and retrieving metadata."""
    registry = get_metadata_registry()
    registry.clear()
    
    # Create metadata
    metadata = SemanticMetadata(
        operation_type="aten::matmul",
        tensor_shape=(10, 10),
        dtype="float32",
        semantic_role="attention"
    )
    
    # Register (using a simple object since LazyTensor might not be available)
    class FakeTensor:
        def __init__(self, id):
            self.id = id
    
    tensor = FakeTensor("lt_1")
    registry.register(tensor.id, tensor)
    registry.set_metadata(tensor.id, metadata)
    
    # Retrieve
    retrieved = registry.get_metadata(tensor.id)
    assert retrieved is not None
    assert retrieved.operation_type == "aten::matmul"
    assert retrieved.semantic_role == "attention"
    assert retrieved.tensor_shape == (10, 10)


def test_metadata_garbage_collection():
    """Test metadata is cleaned up when tensor is GC'd."""
    registry = get_metadata_registry()
    registry.clear()
    
    class FakeTensor:
        def __init__(self, id):
            self.id = id
    
    # Create and register tensor
    tensor = FakeTensor("lt_gc_test")
    metadata = SemanticMetadata(
        operation_type="aten::add",
        tensor_shape=(5, 5),
        dtype="float32"
    )
    
    registry.register(tensor.id, tensor)
    registry.set_metadata(tensor.id, metadata)
    
    assert registry.get_metadata(tensor.id) is not None
    
    # Delete tensor
    tensor_id = tensor.id
    del tensor
    
    # Force garbage collection
    gc.collect()
    
    # Cleanup should remove it
    cleaned = registry.cleanup()
    assert cleaned >= 1
    assert registry.get_metadata(tensor_id) is None


def test_registry_stats():
    """Test registry statistics."""
    registry = get_metadata_registry()
    registry.clear()
    
    # Add some metadata
    for i in range(5):
        metadata = SemanticMetadata(
            operation_type=f"aten::op_{i}",
            tensor_shape=(10, 10),
            dtype="float32"
        )
        registry.set_metadata(f"lt_{i}", metadata)
    
    stats = registry.stats()
    assert stats['total_entries'] == 5


def test_metadata_update():
    """Test metadata can be updated."""
    registry = get_metadata_registry()
    registry.clear()
    
    metadata = SemanticMetadata(
        operation_type="aten::matmul",
        tensor_shape=(10, 10),
        dtype="float32",
        priority=5
    )
    
    registry.set_metadata("lt_update", metadata)
    
    # Update metadata
    metadata.priority = 10
    metadata.update()
    
    retrieved = registry.get_metadata("lt_update")
    assert retrieved.priority == 10
    assert retrieved.last_updated > retrieved.created_at


def test_remove_metadata():
    """Test explicit metadata removal."""
    registry = get_metadata_registry()
    registry.clear()
    
    metadata = SemanticMetadata(
        operation_type="aten::add",
        tensor_shape=(5, 5),
        dtype="float32"
    )
    
    registry.set_metadata("lt_remove", metadata)
    assert registry.get_metadata("lt_remove") is not None
    
    registry.remove("lt_remove")
    assert registry.get_metadata("lt_remove") is None


def test_clear_registry():
    """Test clearing all registry data."""
    registry = get_metadata_registry()
    
    # Add some data
    for i in range(3):
        metadata = SemanticMetadata(
            operation_type=f"aten::op_{i}",
            tensor_shape=(10, 10),
            dtype="float32"
        )
        registry.set_metadata(f"lt_clear_{i}", metadata)
    
    # Clear
    registry.clear()
    
    # Verify empty
    stats = registry.stats()
    assert stats['total_entries'] == 0
    assert stats['active_tensors'] == 0


def test_metadata_properties():
    """Test semantic metadata properties."""
    metadata = SemanticMetadata(
        operation_type="aten::conv2d",
        tensor_shape=(1, 64, 224, 224),
        dtype="float32",
        semantic_role="feature_extraction",
        memory_pattern="streaming",
        compute_intensity=8.0,
        can_parallelize=True,
        priority=7
    )
    
    assert metadata.operation_type == "aten::conv2d"
    assert metadata.tensor_shape == (1, 64, 224, 224)
    assert metadata.semantic_role == "feature_extraction"
    assert metadata.memory_pattern == "streaming"
    assert metadata.compute_intensity == 8.0
    assert metadata.can_parallelize is True
    assert metadata.priority == 7
    assert metadata.metadata_version == "2.0"


def test_none_metadata():
    """Test retrieving non-existent metadata returns None."""
    registry = get_metadata_registry()
    registry.clear()
    
    result = registry.get_metadata("nonexistent_id")
    assert result is None

