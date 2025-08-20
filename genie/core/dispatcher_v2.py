"""
Enhanced dispatcher integration using PyTorch's structured library approach.

This module replaces the manual dispatcher registration with proper torch.library
integration, providing better performance and compatibility.
"""
import logging
from typing import Any, Dict, Set, Optional

import torch

from .library import genie_lib, set_lazy_mode, get_library_stats

logger = logging.getLogger(__name__)


class StructuredDispatcher:
    """
    Enhanced dispatcher using PyTorch's structured library approach.
    
    This replaces the manual operation registration with proper torch.library
    integration for better performance and compatibility.
    """
    
    def __init__(self):
        self.library = genie_lib
        self._initialized = False
        self._lazy_mode = True
        
    def initialize(self) -> None:
        """Initialize the structured dispatcher."""
        if self._initialized:
            return
            
        try:
            # The library operations are already registered during import
            # We just need to ensure the backend is properly set up
            self._initialized = True
            logger.info("Structured dispatcher initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize structured dispatcher: {e}")
            raise
    
    def set_lazy_mode(self, enabled: bool) -> None:
        """Enable or disable lazy execution mode."""
        self._lazy_mode = enabled
        set_lazy_mode(enabled)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dispatcher statistics."""
        stats = get_library_stats()
        stats.update({
            "initialized": self._initialized,
            "library_name": "aten (FRAGMENT)",
            "backend_key": "PrivateUse1"
        })
        return stats
    
    def is_lazy_mode(self) -> bool:
        """Check if lazy mode is enabled."""
        return self._lazy_mode
    
    def get_registered_operations(self) -> Set[str]:
        """Get set of registered operation names."""
        # This would require introspection of the library
        # For now, return the operations we know we registered
        return {
            "add.Tensor", "sub.Tensor", "mul.Tensor", "div.Tensor",
            "matmul", "mm", "bmm", "addmm", "linear",
            "randn", "zeros", "ones",
            "relu", "sigmoid", "tanh", "softmax.int", "log_softmax.int",
            "dropout", "batch_norm",
            "conv2d", "view", "transpose.int", "permute", "cat", "stack"
        }
    
    def create_lazy_tensor(self, op_name: str, *args, **kwargs):
        """Create LazyTensor for deferred execution."""
        # Route through library to update operation_count stats
        from .library import create_lazy_tensor
        return create_lazy_tensor(op_name, *args, **kwargs)


# Global structured dispatcher instance
structured_dispatcher = StructuredDispatcher()


def initialize_dispatcher() -> None:
    """Initialize the structured dispatcher."""
    structured_dispatcher.initialize()


def set_dispatcher_lazy_mode(enabled: bool) -> None:
    """Set lazy mode for the dispatcher."""
    structured_dispatcher.set_lazy_mode(enabled)


def get_dispatcher_stats() -> Dict[str, Any]:
    """Get dispatcher statistics."""
    return structured_dispatcher.get_stats()


def is_dispatcher_lazy_mode() -> bool:
    """Check if dispatcher is in lazy mode."""
    return structured_dispatcher.is_lazy_mode()


# Compatibility functions for existing code
def create_lazy_tensor_for_device_op(op_name: str, args: tuple, kwargs: dict):
    """Create LazyTensor for device operations (compatibility function)."""
    return structured_dispatcher.create_lazy_tensor(op_name, *args, **kwargs)


# Auto-initialize on import
try:
    initialize_dispatcher()
except Exception as e:
    logger.warning(f"Failed to auto-initialize dispatcher: {e}")


# Backward compatibility with old dispatcher interface
class DispatcherIntegration:
    """Backward compatibility wrapper for the old dispatcher interface."""
    
    def __init__(self):
        self._structured = structured_dispatcher
    
    def set_lazy_mode(self, enabled: bool) -> None:
        """Enable or disable lazy execution mode."""
        self._structured.set_lazy_mode(enabled)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dispatcher statistics."""
        stats = self._structured.get_stats()
        # Convert to old format for compatibility
        return {
            "registered_ops": stats.get("registered_ops", 0),
            "fallback_ops": stats.get("failed_ops", 0),
            "operation_count": stats.get("operation_count", 0),
            "lazy_mode": stats.get("lazy_mode", True)
        }
    
    def register_op(self, op_name: str):
        """Deprecated: Use structured library registration instead."""
        logger.warning(f"register_op is deprecated. Operation {op_name} should be registered in library.py")
        
        def decorator(func):
            logger.warning(f"Skipping manual registration of {op_name}, using structured approach")
            return func
        return decorator


# Provide backward compatibility instance
dispatcher = DispatcherIntegration()


logger.info("Enhanced structured dispatcher loaded successfully")
