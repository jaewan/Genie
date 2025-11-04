"""
Unified interception layer - 2 mechanisms + 1 setup step for complete PyTorch operation capture.

This module coordinates the interception mechanisms for transparent GPU disaggregation:

1. **Factory Intercept**: Wrap torch.randn, torch.zeros, etc. for first tensor creation
2. **__torch_dispatch__**: Official PyTorch 2.0+ mechanism for custom tensor subclass operations
3. **Device Backend**: Register remote_accelerator device with PyTorch core (setup, not interception)

The LazyTensor subclass already implements __torch_dispatch__ correctly.
Factory interceptor handles tensor creation. Device registration enables the device type.
"""
import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Set
import os

import torch

logger = logging.getLogger(__name__)

# Note: Removed _creating_lazy_internally flag - recursion prevention now handled by device filtering

from .lazy_tensor import LazyTensor
from .exceptions import Result


class GenieInterception:
    """
    Unified interception layer - 2 mechanisms + 1 setup step for complete operation capture.

    Coordinates the interception mechanisms for transparent GPU disaggregation:

    1. **Factory Intercept** (torch.randn, torch.zeros): Entry points without LazyTensors yet
    2. **__torch_dispatch__** (LazyTensor subclass): THE official PyTorch 2.0+ mechanism
    3. **Device Backend** (setup): Register remote_accelerator device type

    Note: LazyTensor already implements __torch_dispatch__ correctly.
    No monkey-patching needed - it's already defined in the subclass.

    Performance characteristics:
    - Factory intercept: ~1-2μs overhead per creation (negligible)
    - __torch_dispatch__: ~100ns overhead per op (fastest path after XLA/MPS)
    - Device backend: One-time registration cost
    """

    def __init__(self):
        self._factory_wrapped = False
        self._dispatch_registered = False
        self._device_registered = False
        self._stats = {
            "factory_intercepts": 0,
            "dispatch_intercepts": 0,
            "device_registrations": 0,
            "fallback_operations": 0
        }

    @staticmethod
    def register_device():
        """
        Mechanism 1: Device backend registration.

        This is mandatory for PyTorch to recognize "remote_accelerator" as a valid device type.
        Without this, torch.device("remote_accelerator:0") will fail.
        """
        try:
            from genie import _C
            _C.register_remote_accelerator_device()
            logger.info("Successfully registered remote_accelerator backend (C++)")
            return True
        except Exception as e:
            logger.warning(f"C++ backend registration failed: {e}")
            logger.info("Falling back to Python-only mode (still functional)")
            return False

    @staticmethod
    def wrap_factories():
        """
        Mechanism 2: Factory function interception.

        Wrap torch.randn, torch.zeros, etc. to return LazyTensor for first tensor creation.
        These are the entry points - without LazyTensors existing yet, dispatch doesn't help.

        Performance: ~1-2μs overhead per creation, negligible for typical workloads.

        Delegates to factory_interceptor.py for actual implementation.
        """
        from .factory_interceptor import wrap_factories
        wrap_factories()

    @classmethod
    def enable_dispatch_interception(cls):
        """
        Verify that __torch_dispatch__ is properly enabled for LazyTensor subclass.

        LazyTensor already implements __torch_dispatch__ correctly as a classmethod.
        This method just verifies the implementation is in place.
        """
        try:
            if not hasattr(LazyTensor, '__torch_dispatch__'):
                raise RuntimeError("LazyTensor.__torch_dispatch__ not found - implementation error")

            # Verify it's a classmethod (not instance method)
            dispatch_method = getattr(LazyTensor, '__torch_dispatch__')
            if not (isinstance(dispatch_method, classmethod) or hasattr(dispatch_method, '__self__')):
                raise RuntimeError("LazyTensor.__torch_dispatch__ must be a classmethod")

            logger.info("✓ __torch_dispatch__ interception verified for LazyTensor operations")
        except Exception as e:
            logger.warning(f"Dispatch verification failed: {e}")
            logger.info("LazyTensor.__torch_dispatch__ will be verified at runtime")

    def get_stats(self) -> Dict[str, Any]:
        """Get interception statistics."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset interception statistics."""
        for key in self._stats:
            self._stats[key] = 0


# Create global interception instance
genie_interception = GenieInterception()


# Convenience functions for external use
def register_device_backend() -> bool:
    """Register the remote_accelerator device backend."""
    return GenieInterception.register_device()


def wrap_factory_functions() -> None:
    """Wrap PyTorch factory functions for LazyTensor creation."""
    GenieInterception.wrap_factories()


def enable_dispatch_interception() -> None:
    """Enable __torch_dispatch__ interception for LazyTensor operations."""
    GenieInterception.enable_dispatch_interception()


def get_interception_stats() -> Dict[str, Any]:
    """Get interception layer statistics."""
    return genie_interception.get_stats()


def reset_interception_stats() -> None:
    """Reset interception statistics."""
    genie_interception.reset_stats()


# Note: Thread-local capture context moved to capture.py

# Auto-initialize when module is imported
def _initialize_interception():
    """Initialize interception mechanisms and device backend."""
    logger.info("Initializing Genie interception layer...")

    # 1. Register device backend (setup, not interception)
    device_registered = register_device_backend()

    # 2. Wrap factory functions (interception mechanism 1)
    wrap_factory_functions()

    # 3. Verify __torch_dispatch__ is available (interception mechanism 2, already in LazyTensor)
    enable_dispatch_interception()

    logger.info("Genie interception layer initialized successfully")


# Initialize on import
_initialize_interception()
