"""
Factory function interceptor for tensor creation operations.

This module wraps torch.randn, torch.zeros, etc. to return LazyTensors
when device='remote_accelerator' is specified OR inside capture context.

Why necessary: Factory functions don't have LazyTensor arguments, so
__torch_dispatch__ won't be called. We need explicit namespace wrapping.

Performance: ~1-2μs overhead per creation, negligible for typical workloads.

Example:
    >>> interceptor = FactoryInterceptor()
    >>> interceptor.wrap()
    >>> with genie.capture():
    ...     x = torch.randn(10, 10)  # Returns LazyTensor
    >>> interceptor.unwrap()
"""

import torch
import functools
import logging
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# OPTIMIZATION: Module-level profiler caching to reduce per-call overhead
# ============================================================================
_profiler = None
_profiler_lock = threading.Lock()

def _get_profiler_cached():
    """Get cached profiler instance (only import once per module load)."""
    global _profiler, _profiler_lock
    
    # Fast path: profiler already cached
    if _profiler is not None:
        return _profiler
    
    # Slow path: first call - import and cache
    with _profiler_lock:
        if _profiler is not None:
            return _profiler
        
        try:
            from genie.profiling import get_detailed_profiler
            _profiler = get_detailed_profiler() or False  # False = inactive profiler
        except (ImportError, Exception):
            _profiler = False  # Mark as no profiler available
        
        return _profiler if _profiler is not False else None


# Thread-local storage to track when we're inside a LazyTensor factory method
# This prevents infinite recursion when factory interceptor calls LazyTensor methods
_in_lazy_factory = threading.local()

# Module-level import (avoids hot path import)
_executor_module = None
try:
    from . import executor as _executor_module
except ImportError:
    pass


class FactoryInterceptor:
    """
    Intercepts PyTorch tensor creation functions.

    This module wraps torch.randn, torch.zeros, etc. to return LazyTensors
    when device='remote_accelerator' is specified OR inside capture context.

    **Why necessary:** Factory functions don't have LazyTensor arguments, so
    __torch_dispatch__ won't be called. We need explicit namespace wrapping
    to intercept tensor creation operations.

    **Performance:** ~1-2μs overhead per creation, negligible for typical workloads.
    This overhead is amortized over model inference where hundreds of operations
    are performed.

    **Thread Safety:** Each thread maintains its own capture state via thread-local
    storage. Factory wrapping is thread-safe and doesn't interfere across threads.

    Example:
        >>> interceptor = FactoryInterceptor()
        >>> interceptor.wrap()
        >>>
        >>> # Device-based API (paper API)
        >>> x = torch.randn(10, 10, device='remote_accelerator:0')
        >>> isinstance(x, LazyTensor)  # True
        >>>
        >>> # Context-based API (convenience API)
        >>> with genie.capture():
        ...     y = torch.randn(10, 10)  # No device needed
        >>> isinstance(y, LazyTensor)  # True
        >>>
        >>> interceptor.unwrap()  # Restore for testing
    """

    # Functions to intercept (complete list)
    FACTORY_FUNCTIONS = [
        # Basic creation
        'randn', 'rand', 'randint', 'randn_like', 'rand_like', 'randint_like',
        'zeros', 'ones', 'empty', 'full',
        'zeros_like', 'ones_like', 'empty_like', 'full_like',

        # Data conversion
        'tensor', 'as_tensor', 'from_numpy',

        # Special constructors
        'eye', 'arange', 'linspace', 'logspace',

        # Random distributions
        'normal', 'randperm',
    ]

    def __init__(self):
        self._wrapped = False
        self._original_functions: Dict[str, Callable] = {}

    def wrap(self):
        """Wrap all factory functions in torch namespace."""
        if self._wrapped:
            logger.warning("Factory functions already wrapped")
            return

        for func_name in self.FACTORY_FUNCTIONS:
            if not hasattr(torch, func_name):
                logger.debug(f"torch.{func_name} not found, skipping")
                continue

            # Store original
            self._original_functions[func_name] = getattr(torch, func_name)

            # Create wrapper
            wrapper = self._create_wrapper(func_name)

            # Replace in torch namespace
            setattr(torch, func_name, wrapper)

        self._wrapped = True
        logger.info(f"Wrapped {len(self._original_functions)} factory functions")

    def unwrap(self):
        """Restore original factory functions (for testing)."""
        if not self._wrapped:
            return

        for func_name, original_func in self._original_functions.items():
            setattr(torch, func_name, original_func)

        self._wrapped = False
        self._original_functions.clear()

    def _create_wrapper(self, func_name: str) -> Callable:
        """Create wrapper function for a factory function."""
        original_func = self._original_functions[func_name]

        @functools.wraps(original_func)
        def wrapper(*args, **kwargs):
            # TODO(Jae) Delete profiling hooks later
            # OPTIMIZATION: Cache profiler at module level to reduce import overhead
            profiler = _get_profiler_cached()
            profiler_active = profiler is not None
            
            if profiler_active:
                component_name = f"factory_{func_name}"
                profiler_context = profiler.profile_component(component_name)
                profiler_context.__enter__()
            else:
                profiler_context = None
            
            try:
                device = kwargs.get('device')

                # CRITICAL FIX: Don't intercept meta or cpu devices
                # These are used internally by PyTorch and LazyTensor for shape inference
                if (device is not None and
                    (device == 'meta' or device == 'cpu' or
                     (isinstance(device, torch.device) and device.type in ('meta', 'cpu')))):
                    # Convert torch.device to string for original function
                    if isinstance(device, torch.device):
                        fixed_kwargs = kwargs.copy()
                        fixed_kwargs['device'] = device.type
                        return original_func(*args, **fixed_kwargs)
                    else:
                        return original_func(*args, **kwargs)

                # Check if we're inside executor materialization (skip interception in that case)
                if _executor_module:
                    executor_active = getattr(_executor_module._in_executor, 'active', False)
                    if executor_active:
                        # Inside executor - don't return LazyTensor, create concrete tensor
                        # For remote devices, create on CPU instead
                        if self._is_remote_device(device):
                            # Create the tensor on CPU instead of remote device
                            materialized_kwargs = kwargs.copy()
                            materialized_kwargs['device'] = 'cpu'
                            return original_func(*args, **materialized_kwargs)
                        else:
                            # For other devices, call original function
                            return original_func(*args, **kwargs)

                # Check if we're already inside a LazyTensor factory method (prevent recursion)
                if getattr(_in_lazy_factory, 'active', False):
                    # For remote devices, we need to handle them specially to avoid torch errors
                    if device is not None and ('remote_accelerator' in str(device) or 'privateuseone' in str(device)):
                        # Return a placeholder - the LazyTensor constructor will handle this properly
                        return torch.empty(1, device='meta')  # This will be replaced
                    # For meta devices during LazyTensor creation, don't intercept
                    if device == 'meta' or (isinstance(device, torch.device) and device.type == 'meta'):
                        return original_func(*args, **kwargs)
                    return original_func(*args, **kwargs)

                # Check if should return LazyTensor
                from .capture import is_capturing
                if self._is_remote_device(device) or is_capturing():
                    from .lazy_tensor import LazyTensor
                    
                    # ✅ TRIGGER ASYNC INIT: First Genie API call
                    # This is one of the earliest points where Genie code is invoked
                    # Initialize runtime on first remote tensor creation or capture
                    from ..runtime.initialization import _ensure_async_init
                    _ensure_async_init()

                    # Set flag to prevent recursion
                    _in_lazy_factory.active = True
                    try:
                        # Call LazyTensor factory method (FIX: Simplified - factory methods now handle size conversion)
                        lazy_factory = getattr(LazyTensor, func_name, None)
                        if lazy_factory is not None:
                            return lazy_factory(*args, **kwargs)
                        else:
                            # Final fallback: generic LazyTensor creation with metadata
                            from genie.core.metadata_capture import get_metadata_capture
                            metadata = get_metadata_capture().capture_metadata(
                                operation=f'aten::{func_name}',
                                inputs=list(args),
                                kwargs=kwargs
                            )
                            return LazyTensor(
                                operation=f'aten::{func_name}',
                                inputs=list(args),
                                kwargs=kwargs,
                                metadata=metadata  # ✅ NEW: Pass semantic metadata
                            )
                    finally:
                        # Always clear the flag
                        _in_lazy_factory.active = False
                # ✅ Only reached if NOT capturing AND NOT remote device
                return original_func(*args, **kwargs)
            finally:
                # TODO(Jae) Delete profiling hooks later
                if profiler_context is not None:
                    profiler_context.__exit__(None, None, None)

        return wrapper

    @staticmethod
    def _is_remote_device(device: Any) -> bool:
        """Check if device is remote_accelerator."""
        if device is None:
            return False

        device_str = str(device)
        return ('remote_accelerator' in device_str or
                'privateuseone' in device_str)


# Global interceptor instance
_factory_interceptor = FactoryInterceptor()


def wrap_factories():
    """Wrap factory functions (call once at initialization)."""
    _factory_interceptor.wrap()


def unwrap_factories():
    """Unwrap factory functions (for testing)."""
    _factory_interceptor.unwrap()


def get_factory_interceptor() -> FactoryInterceptor:
    """Get the global factory interceptor instance."""
    return _factory_interceptor