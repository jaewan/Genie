"""
Metadata registry for managing semantic metadata.

Separates metadata lifecycle from LazyTensor execution (Refactoring #2).
Coordinates with Refactoring #3's FX Graph via tensor_id bridge.
"""
from typing import Dict, Optional
import weakref
from dataclasses import dataclass, field
from genie.core.exceptions import Result, SemanticException
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class SemanticMetadata:
    """
    Rich semantic metadata for operations (HotNets'25 §3.1).
    
    Moved from LazyTensor to separate the concerns of
    execution (LazyTensor) and semantics (this class).
    """
    
    # Basic properties
    operation_type: str
    tensor_shape: Optional[tuple] = None
    dtype: Optional[str] = None
    device_hint: Optional[str] = None
    
    # Semantic enrichment (HotNets'25 §3.1)
    semantic_role: Optional[str] = None
    model_module: Optional[str] = None
    execution_phase: Optional[str] = None
    data_lineage: Optional[dict] = None
    
    # Memory patterns (HotNets'25 §3.2)
    memory_pattern: Optional[str] = None
    compute_intensity: float = 0.0
    kv_cache_related: bool = False
    
    # Additional context
    layer_depth: Optional[int] = None
    is_activation: bool = False
    
    # Scheduling hints
    can_parallelize: bool = False
    priority: int = 5  # 0-10 scale
    colocation_group: Optional[str] = None
    
    # Metadata versioning
    metadata_version: str = "2.0"
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    def update(self) -> None:
        """Update timestamp when metadata changes."""
        self.last_updated = time.time()


class MetadataRegistry:
    """
    Global registry for tensor metadata.
    
    Uses weak references to avoid memory leaks when tensors are GC'd.
    Coordinates with Refactoring #3: FX Graph stores tensor_id in meta,
    which is used to lookup metadata here.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # Only initialize once (singleton pattern)
        if self._initialized:
            return
            
        self._metadata: Dict[str, SemanticMetadata] = {}
        self._tensor_refs: Dict[str, weakref.ref] = {}
        self._initialized = True
    
    @classmethod
    def register(cls, tensor_id: str, tensor_ref) -> None:
        """
        Register a tensor for metadata tracking.
        
        Args:
            tensor_id: Unique identifier for tensor (e.g., "lt_42")
            tensor_ref: Reference to tensor (for weak ref and GC)
        """
        instance = cls()
        # Store weak reference to allow GC
        try:
            instance._tensor_refs[tensor_id] = weakref.ref(tensor_ref, lambda ref: cls._cleanup_callback(tensor_id))
        except TypeError:
            # If object doesn't support weak references, store None
            instance._tensor_refs[tensor_id] = None
    
    @classmethod
    def _cleanup_callback(cls, tensor_id: str) -> None:
        """Callback when tensor is garbage collected."""
        instance = cls()
        instance.remove(tensor_id)
        logger.debug(f"Cleaned up metadata for {tensor_id}")
    
    @classmethod
    def get_metadata(cls, tensor_id: str) -> Optional[SemanticMetadata]:
        """
        Get metadata for a tensor.
        
        Args:
            tensor_id: Tensor identifier
            
        Returns:
            Metadata if available, None otherwise
        """
        instance = cls()
        return instance._metadata.get(tensor_id)
    
    @classmethod
    def set_metadata(cls, tensor_id: str, metadata: SemanticMetadata) -> None:
        """
        Store metadata for a tensor.
        
        Args:
            tensor_id: Tensor identifier
            metadata: Semantic metadata
        """
        instance = cls()
        instance._metadata[tensor_id] = metadata
        logger.debug(f"Stored metadata for {tensor_id}: {metadata.operation_type}")
    
    @classmethod
    def remove(cls, tensor_id: str) -> None:
        """Remove metadata when tensor is garbage collected."""
        instance = cls()
        instance._metadata.pop(tensor_id, None)
        instance._tensor_refs.pop(tensor_id, None)
    
    @classmethod
    def cleanup(cls) -> int:
        """
        Clean up metadata for garbage collected tensors.
        
        Returns:
            Number of entries cleaned up
        """
        instance = cls()
        dead_ids = []
        for tensor_id, ref in instance._tensor_refs.items():
            if ref is None:
                # Object doesn't support weak refs, skip
                continue
            if ref() is None:  # Tensor was GC'd
                dead_ids.append(tensor_id)
        
        for tensor_id in dead_ids:
            cls.remove(tensor_id)
        
        return len(dead_ids)
    
    @classmethod
    def clear(cls) -> None:
        """Clear all metadata (for testing)."""
        instance = cls()
        instance._metadata.clear()
        instance._tensor_refs.clear()
    
    @classmethod
    def stats(cls) -> dict:
        """Get registry statistics."""
        instance = cls()
        active_count = 0
        dead_count = 0
        
        for ref in instance._tensor_refs.values():
            if ref is None:
                continue  # Object doesn't support weak refs
            if ref() is not None:
                active_count += 1
            else:
                dead_count += 1
        
        return {
            'total_entries': len(instance._metadata),
            'active_tensors': active_count,
            'dead_tensors': dead_count,
        }


# Convenience function
def get_metadata_registry() -> MetadataRegistry:
    """Get the global metadata registry."""
    return MetadataRegistry()

