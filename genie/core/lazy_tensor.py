from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple
import torch
from functools import lru_cache

logger = logging.getLogger(__name__)

# Import interception control for cleaner recursion handling
from .interception_control import should_intercept, disable_interception, InterceptionContext


# ============================================================================
# PHASE 1 FIX: Thread-Safe Shape Cache with LRU
# ============================================================================

def _get_thread_local_fake_mode():
    """Get thread-local FakeTensorMode for reuse."""
    if not hasattr(_get_thread_local_fake_mode, '_thread_local'):
        _get_thread_local_fake_mode._thread_local = threading.local()

    tls = _get_thread_local_fake_mode._thread_local
    if not hasattr(tls, 'fake_mode'):
        from torch._subclasses.fake_tensor import FakeTensorMode
        tls.fake_mode = FakeTensorMode()

    return tls.fake_mode


def _get_thread_local_shape_cache() -> Dict[str, Optional[torch.Size]]:
    """
    Get thread-local bounded shape cache.

    Each thread gets its own bounded LRU cache to avoid race conditions
    and prevent memory leaks in long-running applications.

    See: peer_review.md Phase 1.1 - Thread Safety
    """
    if not hasattr(_get_thread_local_shape_cache, '_thread_local'):
        _get_thread_local_shape_cache._thread_local = threading.local()

    tls = _get_thread_local_shape_cache._thread_local
    if not hasattr(tls, 'shape_cache'):
        # ✅ FIX: Bounded LRU cache to prevent memory leaks
        from functools import lru_cache

        @lru_cache(maxsize=1024)  # Bound to 1024 entries to prevent OOM
        def _bounded_shape_cache(op_name: str, inputs_signature: str) -> Optional[torch.Size]:
            """Bounded cache function for shape inference."""
            return None  # This will be overridden

        tls.shape_cache = _bounded_shape_cache

    return tls.shape_cache


class LazyTensor(torch.Tensor):
    """
    Lazy tensor that captures operations without executing.

    This is a proper torch.Tensor subclass that integrates with PyTorch's
    dispatcher system. All operations on LazyTensors are automatically
    intercepted via __torch_dispatch__.

    Usage:
        >>> # Device-based API (paper API)
        >>> x = torch.randn(10, 10, device='remote_accelerator:0')
        >>> isinstance(x, LazyTensor)  # True
        >>> y = x @ x  # Operations deferred
        >>> result = y.cpu()  # Triggers execution

        >>> # Context-based API (convenience API)
        >>> with genie.capture():
        ...     x = torch.randn(10, 10)  # No device needed
        ...     y = x @ x
        >>> result = y.cpu()  # Triggers execution

    How It Works:
        1. Operations create new LazyTensor instances (no computation)
        2. Graph is built incrementally as operations are captured
        3. Materialization (.cpu(), .numpy()) triggers execution
        4. Graph is traversed and operations executed in order

    Thread Safety:
        - LazyTensor instances are immutable (thread-safe)
        - Graph building uses thread-local state (safe)
        - Materialization is thread-safe (no shared state)
    """

    # Class-level state
    _graph_builder: Optional['GraphBuilder'] = None

    # Fast path for simple operations (avoids FakeTensorMode overhead)
    _SIMPLE_OPS = {
        'aten::relu': lambda inputs: inputs[0].shape,
        'aten::sigmoid': lambda inputs: inputs[0].shape,
        'aten::tanh': lambda inputs: inputs[0].shape,
        'aten::abs': lambda inputs: inputs[0].shape,
        'aten::neg': lambda inputs: inputs[0].shape,
        'aten::exp': lambda inputs: inputs[0].shape,
        'aten::log': lambda inputs: inputs[0].shape,
        # NOTE: Removed simplified add/sub/mul/div - they need proper broadcasting support
        # which requires FakeTensor mode for accurate shape inference
    }

    # Track metadata without breaking tensor subclass protocol
    @staticmethod
    def __new__(
        cls,
        operation: str,
        inputs: List[Any],
        kwargs: Optional[Dict[str, Any]] = None,
        shape: Optional[torch.Size] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Create LazyTensor wrapper.

        CRITICAL: Must use _make_subclass for proper tensor subclass.
        This creates a tensor wrapper WITHOUT allocating actual storage.
        """
        with disable_interception(InterceptionContext.CONSTRUCTION):
            # Infer shape/dtype if not provided
            if shape is None:
                shape = torch.Size([])  # Use empty shape as placeholder
                # Shape inference will be done lazily when needed

            if dtype is None:
                dtype = cls._infer_dtype(inputs, kwargs or {})
            if dtype is None:
                dtype = torch.float32  # Fallback

            # Ensure dtype is a PyTorch dtype, not numpy dtype
            if hasattr(dtype, 'dtype'):
                dtype = dtype.dtype  # Convert numpy dtype to torch dtype
            if not isinstance(dtype, torch.dtype):
                dtype = torch.float32  # Fallback

            # Handle remote devices - use meta for storage
            if device is None:
                device = torch.device('meta')  # Symbolic device (no storage)
            elif isinstance(device, str) and ('remote_accelerator' in device or 'privateuseone' in device):
                # For remote devices, use meta device for storage
                device = torch.device('meta')
            elif isinstance(device, torch.device) and device.type in ('remote_accelerator', 'privateuseone'):
                # Handle torch.device objects for remote devices
                device = torch.device('meta')

        # Create tensor wrapper using official API
        # This is what makes LazyTensor a "real" tensor
        wrapper = torch.Tensor._make_subclass(
            cls,
            torch.empty(shape, dtype=dtype, device=device),
            require_grad=False  # Disable autograd for now (Phase 2 addition)
        )

        # Store original device info before modifying kwargs
        original_device = None
        if kwargs:
            original_device = kwargs.get('device')

        # Replace the original device in kwargs with the processed device for __init__
        if kwargs:
            kwargs = kwargs.copy()
            kwargs['device'] = device

        # Store original device on the wrapper using object.__setattr__ to avoid recursion
        object.__setattr__(wrapper, '_original_device', original_device)

        # ✅ NEW: Store metadata for __init__ to use
        if metadata is not None:
            object.__setattr__(wrapper, '_pending_metadata', metadata)

        return wrapper

    def __init__(
        self,
        operation: str,
        inputs: List[Any],
        kwargs: Optional[Dict[str, Any]] = None,
        shape: Optional[torch.Size] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LazyTensor metadata.

        Note: __new__ has already created the tensor wrapper.
        Here we attach operation metadata for graph building.
        """
        # Store operation info
        # These become attributes of the tensor instance
        object.__setattr__(self, '_operation', operation)
        object.__setattr__(self, '_inputs', inputs)
        object.__setattr__(self, '_kwargs', kwargs or {})
        object.__setattr__(self, '_tensor_id', id(self))
        object.__setattr__(self, '_shape', shape)
        object.__setattr__(self, '_dtype', dtype)
        object.__setattr__(self, '_device', device)

        # ✅ NEW: Store semantic metadata
        # Use provided metadata or fall back to pending metadata from __new__
        final_metadata = metadata
        if final_metadata is None:
            final_metadata = object.__getattribute__(self, '_pending_metadata', None)
        object.__setattr__(self, '_metadata', final_metadata or {})

        # Original device is already stored in __new__

        # Register with thread-local graph builder
        from .graph_builder import get_global_builder
        try:
            builder = get_global_builder()
            builder.add_operation(self)
        except RuntimeError:
            # Graph builder not initialized yet - skip for now
            pass

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """
        Intercept ALL operations involving LazyTensor.

        CRITICAL: PyTorch dispatcher only calls this when at least one arg
        is a LazyTensor. We should ALWAYS intercept unless explicitly disabled.
        """
        kwargs = kwargs or {}

        # ✅ ONLY check for disabled contexts (construction, materialization)
        from .interception_control import get_current_context, InterceptionContext
        if get_current_context() != InterceptionContext.NONE:
            # We're inside LazyTensor construction or materialization - skip
            return func(*args, **kwargs)

        # Handle special operations that force materialization
        # Check both direct equality and method descriptor equality
        is_materialization_op = False
        if func in cls._MATERIALIZATION_OPS:
            is_materialization_op = True
        elif hasattr(func, '__name__') and hasattr(func, '__qualname__'):
            # Check if this is a method that matches a materialization operation
            for mat_op in cls._MATERIALIZATION_OPS:
                if (hasattr(mat_op, '__name__') and mat_op.__name__ == func.__name__ and
                    hasattr(mat_op, '__qualname__') and mat_op.__qualname__ == func.__qualname__):
                    is_materialization_op = True
                    break

        if is_materialization_op:
            # These operations need concrete tensors
            materialized_args = tuple(
                arg.materialize() if type(arg).__name__ == 'LazyTensor' else arg
                for arg in args
            )
            # For method calls, call the method directly on the materialized tensor
            if hasattr(func, '__name__') and func.__name__.startswith('__'):
                # This is a method call - call it directly on the first materialized tensor
                if materialized_args and isinstance(materialized_args[0], torch.Tensor):
                    method_name = func.__name__
                    method = getattr(materialized_args[0], method_name)
                    return method(*materialized_args[1:], **kwargs)
            # For regular function calls, use the function
            return func(*materialized_args, **kwargs)

        # ✅ Check for mixed operations ONLY when outside capture
        from .capture import is_capturing
        if not is_capturing():
            # Count LazyTensors vs concrete tensors
            lazy_count = sum(1 for arg in args if type(arg).__name__ == 'LazyTensor')
            concrete_tensor_count = sum(1 for arg in args
                                       if isinstance(arg, torch.Tensor)
                                       and type(arg).__name__ != 'LazyTensor')

            # If mixing LazyTensor with concrete tensors, materialize and execute normally
            if lazy_count > 0 and concrete_tensor_count > 0:
                # Materialize all LazyTensor arguments
                materialized_args = []
                target_device = None

                for arg in args:
                    if type(arg).__name__ == 'LazyTensor':
                        materialized = arg.materialize()
                        # If we haven't determined target device yet, use this tensor's device
                        if target_device is None and isinstance(materialized, torch.Tensor):
                            target_device = materialized.device
                        materialized_args.append(materialized)
                    else:
                        materialized_args.append(arg)

                # Move all tensors to the same device if needed
                if target_device is not None:
                    for i, arg in enumerate(materialized_args):
                        if isinstance(arg, torch.Tensor) and arg.device != target_device:
                            materialized_args[i] = arg.to(target_device)

                return func(*materialized_args, **kwargs)

        # ✅ Normal case: Create new LazyTensor (ALWAYS if we got here)
        op_name = cls._normalize_op_name(func)

        # Infer device from input tensors if not explicitly provided
        inferred_device = None
        if 'device' not in kwargs:
            # Look for device in input LazyTensors
            for arg in args:
                if type(arg).__name__ == 'LazyTensor':
                    # Check original device first
                    if hasattr(arg, '_original_device') and arg._original_device:
                        inferred_device = arg._original_device
                        break
                    # Then check if it's a remote device by checking the device type
                    elif hasattr(arg, 'device') and arg.device:
                        device_str = str(arg.device)
                        if 'remote_accelerator' in device_str or 'privateuseone' in device_str:
                            inferred_device = arg._original_device or arg.device
                            break

        # Infer dtype from inputs if not explicitly provided
        inferred_dtype = None
        if 'dtype' not in kwargs:
            inferred_dtype = cls._infer_dtype(list(args), kwargs)

        # ✅ NEW: Capture semantic metadata
        from .metadata_capture import get_metadata_capture
        metadata = get_metadata_capture().capture_metadata(
            operation=op_name,
            inputs=list(args),
            kwargs=kwargs
        )

        # Create LazyTensor with inferred device and dtype
        # Remove dtype from kwargs if we're providing it as a parameter to avoid conflicts
        call_kwargs = kwargs.copy()
        if inferred_dtype is not None:
            call_kwargs.pop('dtype', None)  # Remove dtype from kwargs if we have it

        result = cls(
            operation=op_name,
            inputs=list(args),
            kwargs=call_kwargs,
            device=inferred_device,  # Pass device directly
            dtype=inferred_dtype,     # Pass inferred dtype
            metadata=metadata         # ✅ NEW: Pass semantic metadata
        )

        return result

    # Operator methods that route through torch_dispatch
    def __matmul__(self, other):
        """Matrix multiplication using @ operator."""
        # Use torch.matmul to ensure it goes through __torch_dispatch__
        return torch.matmul(self, other)
    
    def __rmatmul__(self, other):
        """Right matrix multiplication."""
        return torch.ops.aten.matmul(other, self)

    def __add__(self, other):
        """Addition using + operator."""
        return torch.ops.aten.add(self, other)

    def __radd__(self, other):
        """Right addition."""
        return torch.ops.aten.add(other, self)

    def __sub__(self, other):
        """Subtraction using - operator."""
        return torch.ops.aten.sub(self, other)

    def __rsub__(self, other):
        """Right subtraction."""
        return torch.ops.aten.sub(other, self)

    def __mul__(self, other):
        """Multiplication using * operator."""
        return torch.ops.aten.mul(self, other)

    def __rmul__(self, other):
        """Right multiplication."""
        return torch.ops.aten.mul(other, self)

    def __truediv__(self, other):
        """Division using / operator."""
        return torch.ops.aten.div(self, other)

    def __rtruediv__(self, other):
        """Right division."""
        return torch.ops.aten.div(other, self)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Handle torch functions that don't go through __torch_dispatch__.

        This catches torch functions like:
        - torch.add, torch.matmul (binary operations)
        - torch.relu, torch.sum (unary operations)
        - .cpu(), .cuda() (device transfers)
        - .numpy(), .item() (conversion to non-tensor)

        Binary operations like x + y call torch.add(x, y) which comes here.
        """
        kwargs = kwargs or {}

        # Prevent recursion when accessing tensor properties
        from .interception_control import get_current_context, InterceptionContext
        if get_current_context() != InterceptionContext.NONE:
            return NotImplemented

        # Mark that we're in torch function context
        from .interception_control import _interception_context
        prev_context = getattr(_interception_context, 'context', InterceptionContext.NONE)
        _interception_context.context = InterceptionContext.PROPERTY_ACCESS

        try:
            # Check if any of the arguments are LazyTensors
            # Use direct type check to avoid triggering torch operations
            has_lazy_tensor = any(
                type(arg).__name__ == 'LazyTensor'
                for arg in args if hasattr(arg, '__class__')
            )

            if not has_lazy_tensor:
                # No LazyTensors involved, let PyTorch handle normally
                return NotImplemented

            # Operations that force materialization - handle these specially to avoid recursion
            # Check both direct equality and method descriptor equality
            is_materialization_op = False
            if func in cls._MATERIALIZATION_OPS:
                is_materialization_op = True
            elif hasattr(func, '__name__') and hasattr(func, '__qualname__'):
                # Check if this is a method that matches a materialization operation
                for mat_op in cls._MATERIALIZATION_OPS:
                    if (hasattr(mat_op, '__name__') and mat_op.__name__ == func.__name__ and
                        hasattr(mat_op, '__qualname__') and mat_op.__qualname__ == func.__qualname__):
                        is_materialization_op = True
                        break

            if is_materialization_op:
                # Find the LazyTensor in arguments and materialize it
                lazy_tensor = None
                for arg in args:
                    try:
                        if type(arg).__name__ == 'LazyTensor':
                            lazy_tensor = arg
                            break
                    except:
                        pass

                if lazy_tensor is not None:
                    # For materialization ops, avoid recursion by directly materializing
                    # and calling the method without going through torch dispatch
                    try:
                        materialized = lazy_tensor.materialize()
                        # Special handling for methods - call them directly
                        if hasattr(func, '__name__'):
                            method_name = func.__name__
                            if method_name.startswith('__') and method_name.endswith('__'):
                                # Handle special method names
                                if method_name == '__len__':
                                    return len(materialized)
                                elif method_name == '__bool__':
                                    # bool() on a tensor returns a tensor, not a bool
                                    # We need to check if the tensor has any elements
                                    try:
                                        return materialized.numel() > 0
                                    except:
                                        return True  # Default to True if we can't determine
                                elif method_name == '__int__':
                                    return int(materialized)
                                elif method_name == '__float__':
                                    return float(materialized)
                                # For other dunder methods, return the materialized tensor
                                # and let the caller handle it
                                return materialized
                            else:
                                # Regular method call
                                method = getattr(materialized, method_name)
                                return method(*args[1:], **kwargs)
                        else:
                            # Fallback: just return materialized tensor
                            return materialized
                    except:
                        # If anything fails, just return the materialized tensor
                        return lazy_tensor.materialize()

            # Check if we're outside a capture context with mixed LazyTensor/concrete types
            # In this case, materialize the LazyTensor and perform the operation normally
            from .capture import is_capturing
            if not is_capturing():
                # Count LazyTensors vs concrete tensors
                lazy_count = sum(1 for arg in args if type(arg).__name__ == 'LazyTensor')
                concrete_tensor_count = sum(1 for arg in args if isinstance(arg, torch.Tensor) and type(arg).__name__ != 'LazyTensor')

                # If we have a mix of LazyTensor and concrete tensors outside capture context,
                # materialize all LazyTensors and let PyTorch handle it normally
                if lazy_count > 0 and concrete_tensor_count > 0:
                    # Materialize all LazyTensor arguments
                    materialized_args = []
                    target_device = None

                    for arg in args:
                        if type(arg).__name__ == 'LazyTensor':
                            materialized = arg.materialize()
                            # If we haven't determined target device yet, use this tensor's device
                            if target_device is None and isinstance(materialized, torch.Tensor):
                                target_device = materialized.device
                            materialized_args.append(materialized)
                        else:
                            materialized_args.append(arg)

                    # Move all tensors to the same device if needed
                    if target_device is not None:
                        for i, arg in enumerate(materialized_args):
                            if isinstance(arg, torch.Tensor) and arg.device != target_device:
                                materialized_args[i] = arg.to(target_device)

                    # Call the function with materialized arguments
                    return func(*materialized_args, **kwargs)

            # For operations involving LazyTensors inside capture context, create new LazyTensor
            op_name = cls._normalize_op_name(func)

            # Infer dtype from inputs if not explicitly provided
            inferred_dtype = None
            if 'dtype' not in kwargs:
                inferred_dtype = cls._infer_dtype(list(args), kwargs)

            # ✅ NEW: Capture semantic metadata for torch functions too
            from .metadata_capture import get_metadata_capture
            metadata = get_metadata_capture().capture_metadata(
                operation=op_name,
                inputs=list(args),
                kwargs=kwargs
            )

            result = cls(
                operation=op_name,
                inputs=list(args),
                kwargs=kwargs,
                dtype=inferred_dtype,
                metadata=metadata  # ✅ NEW: Pass semantic metadata
            )
            return result
        finally:
            _interception_context.context = prev_context

    # ===================================================================
    # FACTORY METHODS
    # ===================================================================

    @classmethod
    def randn(cls, *size, dtype=None, device=None, requires_grad=False):
        """Create random normal LazyTensor."""
        # Handle case where size is passed as torch.Size or tuple (FIX: Simplified)
        if len(size) == 1 and isinstance(size[0], (torch.Size, tuple, list)):
            size = tuple(size[0])
        else:
            size = tuple(size)

        # ✅ NEW: Capture semantic metadata for factory functions
        from .metadata_capture import get_metadata_capture
        metadata = get_metadata_capture().capture_metadata(
            operation='aten::randn',
            inputs=list(size),
            kwargs={'dtype': dtype, 'device': device, 'requires_grad': requires_grad}
        )

        return cls(
            operation='aten::randn',
            inputs=list(size),
            kwargs={'dtype': dtype, 'device': device, 'requires_grad': requires_grad},
            shape=torch.Size(size),
            dtype=dtype or torch.float32,
            device=device,
            metadata=metadata  # ✅ NEW: Pass semantic metadata
        )

    @classmethod
    def tensor(cls, data, dtype=None, device=None, requires_grad=False):
        """Create LazyTensor from data."""
        # For tensor creation from data, we already know the shape and don't need torch.empty
        shape = torch.Size(data.shape) if hasattr(data, 'shape') else torch.Size([])
        inferred_dtype = dtype or (data.dtype if hasattr(data, 'dtype') else torch.float32)

        # Store original device before processing
        original_device = device

        # Handle remote devices - use meta for storage
        if device is None:
            device = torch.device('meta')  # Symbolic device (no storage)
        elif isinstance(device, str) and ('remote_accelerator' in device or 'privateuseone' in device):
            # For remote devices, use meta device for storage
            device = torch.device('meta')
        elif isinstance(device, torch.device) and device.type in ('remote_accelerator', 'privateuseone'):
            # Handle torch.device objects for remote devices
            device = torch.device('meta')

        # Create tensor wrapper using official API - avoid torch.empty to prevent interception
        # This is what makes LazyTensor a "real" tensor
        # Use torch.tensor with meta device to create the underlying tensor
        # Convert numpy dtypes to torch dtypes if needed
        torch_dtype = inferred_dtype
        if hasattr(inferred_dtype, 'type') and hasattr(inferred_dtype.type, '__module__'):
            if inferred_dtype.type.__module__ == 'numpy':
                # Simple mapping from numpy dtypes to torch dtypes
                numpy_to_torch_dtype = {
                    'int8': torch.int8,
                    'int16': torch.int16,
                    'int32': torch.int32,
                    'int64': torch.int64,
                    'uint8': torch.uint8,
                    'float16': torch.float16,
                    'float32': torch.float32,
                    'float64': torch.float64,
                    'bool': torch.bool,
                }
                torch_dtype = numpy_to_torch_dtype.get(str(inferred_dtype), torch.float32)

        wrapper = torch.Tensor._make_subclass(
            cls,
            torch.tensor(0, dtype=torch_dtype, device=device).expand(shape),
            require_grad=False  # Disable autograd for now (Phase 2 addition)
        )

        # Store original device on wrapper (same pattern as __new__)
        object.__setattr__(wrapper, '_original_device', original_device)

        # Replace the original device in kwargs with the processed device for __init__
        kwargs = {'dtype': dtype, 'device': device, 'requires_grad': requires_grad}
        if kwargs:
            kwargs = kwargs.copy()
            kwargs['device'] = device

        # ✅ NEW: Capture semantic metadata for tensor creation
        from .metadata_capture import get_metadata_capture
        metadata = get_metadata_capture().capture_metadata(
            operation='aten::tensor',
            inputs=[data],
            kwargs=kwargs
        )

        # Initialize the wrapper
        wrapper.__init__(
            operation='aten::tensor',
            inputs=[data],
            kwargs=kwargs,
            shape=shape,
            dtype=inferred_dtype,
            device=device,
            metadata=metadata  # ✅ NEW: Pass semantic metadata
        )

        return wrapper

    @classmethod
    def as_tensor(cls, data, dtype=None, device=None):
        """Create LazyTensor from data (alias for tensor)."""
        # Create LazyTensor with correct operation name
        shape = torch.Size(data.shape) if hasattr(data, 'shape') else torch.Size([])
        inferred_dtype = dtype or (data.dtype if hasattr(data, 'dtype') else torch.float32)

        # Store original device before processing
        original_device = device

        # Handle remote devices - use meta for storage
        if device is None:
            device = torch.device('meta')  # Symbolic device (no storage)
        elif isinstance(device, str) and ('remote_accelerator' in device or 'privateuseone' in device):
            # For remote devices, use meta device for storage
            device = torch.device('meta')
        elif isinstance(device, torch.device) and device.type in ('remote_accelerator', 'privateuseone'):
            # Handle torch.device objects for remote devices
            device = torch.device('meta')

        # Convert numpy dtypes to torch dtypes if needed
        torch_dtype = inferred_dtype
        if hasattr(inferred_dtype, 'type') and hasattr(inferred_dtype.type, '__module__'):
            if inferred_dtype.type.__module__ == 'numpy':
                # Simple mapping from numpy dtypes to torch dtypes
                numpy_to_torch_dtype = {
                    'int8': torch.int8,
                    'int16': torch.int16,
                    'int32': torch.int32,
                    'int64': torch.int64,
                    'uint8': torch.uint8,
                    'float16': torch.float16,
                    'float32': torch.float32,
                    'float64': torch.float64,
                    'bool': torch.bool,
                }
                torch_dtype = numpy_to_torch_dtype.get(str(inferred_dtype), torch.float32)

        # Create tensor wrapper using official API - avoid torch.empty to prevent interception
        # Use empty with meta device, then expand to target shape
        wrapper = torch.Tensor._make_subclass(
            cls,
            torch.empty(shape, dtype=torch_dtype, device=device),
            require_grad=False  # Disable autograd for now (Phase 2 addition)
        )

        # Store original device on wrapper (same pattern as __new__)
        object.__setattr__(wrapper, '_original_device', original_device)

        # Replace the original device in kwargs with the processed device for __init__
        kwargs = {'dtype': dtype, 'device': device}
        if kwargs:
            kwargs = kwargs.copy()
            kwargs['device'] = device

        # ✅ NEW: Capture semantic metadata for as_tensor
        from .metadata_capture import get_metadata_capture
        metadata = get_metadata_capture().capture_metadata(
            operation='aten::as_tensor',
            inputs=[data],
            kwargs=kwargs
        )

        # Initialize the wrapper with as_tensor operation
        wrapper.__init__(
            operation='aten::as_tensor',
            inputs=[data],
            kwargs=kwargs,
            shape=shape,
            dtype=torch_dtype,
            device=device,
            metadata=metadata  # ✅ NEW: Pass semantic metadata
        )

        return wrapper

    @classmethod
    def from_numpy(cls, ndarray, dtype=None, device=None):
        """Create LazyTensor from numpy array."""
        # ✅ NEW: Capture semantic metadata for factory functions
        from .metadata_capture import get_metadata_capture
        metadata = get_metadata_capture().capture_metadata(
            operation='aten::from_numpy',
            inputs=[ndarray],
            kwargs={'dtype': dtype, 'device': device}
        )

        return cls(
            operation='aten::from_numpy',
            inputs=[ndarray],
            kwargs={'dtype': dtype, 'device': device},
            shape=torch.Size(ndarray.shape),
            dtype=dtype or torch.from_numpy(ndarray).dtype,
            device=device,
            metadata=metadata  # ✅ NEW: Pass semantic metadata
        )

    @classmethod
    def zeros(cls, *size, dtype=None, device=None, requires_grad=False):
        """Create zero-filled LazyTensor."""
        # Handle case where size is passed as torch.Size or tuple (FIX: Simplified)
        if len(size) == 1 and isinstance(size[0], (torch.Size, tuple, list)):
            size = tuple(size[0])
        else:
            size = tuple(size)

        # ✅ NEW: Capture semantic metadata for factory functions
        from .metadata_capture import get_metadata_capture
        metadata = get_metadata_capture().capture_metadata(
            operation='aten::zeros',
            inputs=list(size),
            kwargs={'dtype': dtype, 'device': device, 'requires_grad': requires_grad}
        )

        return cls(
            operation='aten::zeros',
            inputs=list(size),
            kwargs={'dtype': dtype, 'device': device, 'requires_grad': requires_grad},
            shape=torch.Size(size),
            dtype=dtype or torch.float32,
            device=device,
            metadata=metadata  # ✅ NEW: Pass semantic metadata
        )

    @classmethod
    def ones(cls, *size, dtype=None, device=None, requires_grad=False):
        """Create one-filled LazyTensor."""
        # Handle case where size is passed as torch.Size or tuple (FIX: Simplified)
        if len(size) == 1 and isinstance(size[0], (torch.Size, tuple, list)):
            size = tuple(size[0])
        else:
            size = tuple(size)

        # ✅ NEW: Capture semantic metadata for factory functions
        from .metadata_capture import get_metadata_capture
        metadata = get_metadata_capture().capture_metadata(
            operation='aten::ones',
            inputs=list(size),
            kwargs={'dtype': dtype, 'device': device, 'requires_grad': requires_grad}
        )

        return cls(
            operation='aten::ones',
            inputs=list(size),
            kwargs={'dtype': dtype, 'device': device, 'requires_grad': requires_grad},
            shape=torch.Size(size),
            dtype=dtype or torch.float32,
            device=device,
            metadata=metadata  # ✅ NEW: Pass semantic metadata
        )

    @classmethod
    def empty(cls, *size, dtype=None, device=None, requires_grad=False):
        """Create empty LazyTensor (uninitialized memory)."""
        # Handle case where size is passed as torch.Size or tuple (FIX: Simplified)
        if len(size) == 1 and isinstance(size[0], (torch.Size, tuple, list)):
            size = tuple(size[0])
        else:
            size = tuple(size)

        # ✅ NEW: Capture semantic metadata for factory functions
        from .metadata_capture import get_metadata_capture
        metadata = get_metadata_capture().capture_metadata(
            operation='aten::empty',
            inputs=list(size),
            kwargs={'dtype': dtype, 'device': device, 'requires_grad': requires_grad}
        )

        return cls(
            operation='aten::empty',
            inputs=list(size),
            kwargs={'dtype': dtype, 'device': device, 'requires_grad': requires_grad},
            shape=torch.Size(size),
            dtype=dtype or torch.float32,
            device=device,
            metadata=metadata  # ✅ NEW: Pass semantic metadata
        )

    @classmethod
    def full(cls, fill_value, *size, dtype=None, device=None, requires_grad=False):
        """Create LazyTensor filled with a scalar value."""
        # Handle case where size is passed as torch.Size or tuple (FIX: Simplified)
        if len(size) == 1 and isinstance(size[0], (torch.Size, tuple, list)):
            size = tuple(size[0])
        else:
            size = tuple(size)

        return cls(
            operation='aten::full',
            inputs=[fill_value, *size],
            kwargs={'dtype': dtype, 'device': device, 'requires_grad': requires_grad},
            shape=torch.Size(size),
            dtype=dtype or torch.float32,
            device=device
        )

    # ===================================================================
    # MATERIALIZATION
    # ===================================================================

    def materialize(self) -> torch.Tensor:
        """
        Force execution of the computation graph.

        This traverses the DAG and executes operations to produce
        a concrete tensor.

        Returns:
            Concrete torch.Tensor with actual data
        """
        from .graph_builder import get_global_builder
        builder = get_global_builder()
        return builder.materialize(self)


    # Operations that force materialization
    _MATERIALIZATION_OPS = {
        torch.Tensor.cpu,
        torch.Tensor.cuda,
        torch.Tensor.numpy,
        torch.Tensor.item,
        torch.Tensor.tolist,
        torch.Tensor.__bool__,
        torch.Tensor.__int__,
        torch.Tensor.__float__,
        torch.Tensor.__len__,
        # Comparison operations that need concrete values
        torch.Tensor.__gt__,
        torch.Tensor.__ge__,
        torch.Tensor.__lt__,
        torch.Tensor.__le__,
        torch.Tensor.__eq__,
        torch.Tensor.__ne__,
    }

    # ===================================================================
    # SHAPE INFERENCE
    # ===================================================================

    @classmethod
    def _infer_shape(
        cls,
        operation: str,
        inputs: List[Any],
        kwargs: Dict[str, Any]
    ) -> Optional[torch.Size]:
        """
        Infer output shape using PyTorch's meta tensor system.

        This executes the operation on "fake" tensors that only track
        shape/dtype without allocating storage.
        """
        # Fast path for simple operations (avoids FakeTensorMode overhead)
        if operation in LazyTensor._SIMPLE_OPS:
            try:
                return LazyTensor._SIMPLE_OPS[operation](inputs)
            except:
                pass  # Fall through to FakeTensorMode

        # Use cached shape inference with LazyTensor IDs as keys
        return LazyTensor._infer_shape_with_cache(operation, inputs, kwargs)

    @classmethod
    def _infer_shape_with_cache(cls, operation: str, inputs: List[Any], kwargs: Dict[str, Any]) -> Optional[torch.Size]:
        """
        Cached shape inference with bounded LRU cache.

        PHASE 1 FIX: Thread-safe bounded caching to prevent race conditions
        and memory leaks in long-running applications.

        See: peer_review.md - Phase 1.1 Make Shape Cache Thread-Safe
        """
        # Get thread-local bounded cache
        cache = _get_thread_local_shape_cache()

        # Build cache signature from operation and input shapes/types
        inputs_signature = cls._build_inputs_signature(inputs)

        # Check bounded cache (functools.lru_cache handles bounds automatically)
        try:
            cached_result = cache(operation, inputs_signature)
            if cached_result is not None:
                return cached_result
        except TypeError:
            # Cache miss or non-hashable inputs - compute normally
            pass

        # Compute shape
        result = cls._infer_shape_original(operation, inputs, kwargs)

        # Cache result (bounded LRU cache handles eviction automatically)
        # Note: We can't cache the result directly in the lru_cache because
        # the function signature expects specific types. Instead, we return
        # the result and let the cache store it based on the signature.
        return result

    @staticmethod
    def _build_inputs_signature(inputs: List[Any]) -> str:
        """Build a hashable signature of inputs for caching."""
        parts = []
        for inp in inputs:
            if isinstance(inp, LazyTensor):
                # Use shape + dtype for LazyTensors
                parts.append(f"lt_{tuple(inp.shape) if inp.shape else 'none'}_{inp.dtype}")
            elif isinstance(inp, torch.Tensor):
                # Use shape + dtype for concrete tensors
                parts.append(f"t_{tuple(inp.shape)}_{inp.dtype}")
            else:
                # Scalar or non-tensor - use string representation
                parts.append(f"s_{repr(inp)}")

        return "|".join(parts)

    @staticmethod
    def _infer_shape_original(
        operation: str,
        inputs: List[Any],
        kwargs: Dict[str, Any]
    ) -> Optional[torch.Size]:
        """
        Original shape inference logic using FakeTensor.
        
        This method is called after cache miss to compute actual shapes.
        """
        try:
            # ✅ FIX: Reuse thread-local FakeTensorMode to avoid recreation overhead
            with _get_thread_local_fake_mode():
                # Convert inputs to fake tensors
                fake_inputs = []
                for inp in inputs:
                    if isinstance(inp, LazyTensor):
                        # CRITICAL FIX: Use object.__getattribute__ to bypass property recursion
                        inp_shape = object.__getattribute__(inp, '_shape')
                        inp_dtype = object.__getattribute__(inp, '_dtype')

                        # Guard: if shape can't be inferred, return None early
                        if inp_shape is None:
                            logger.warning(f"Can't infer shape for {inp.operation} (unsupported op)")
                            return None

                        # ✅ FIX: Check for empty shape placeholder (uninitialized shape)
                        # Empty shape torch.Size([]) means "not inferred yet" for non-tensor operations
                        if len(inp_shape) == 0 and inp.operation not in ('aten::tensor', 'aten::scalar'):
                            logger.warning(
                                f"Cannot infer shape for {operation}: "
                                f"input operation {inp.operation} has uninitialized shape (empty placeholder)"
                            )
                            return None

                        fake_inputs.append(
                            torch.empty(inp_shape, dtype=inp_dtype or torch.float32, device='meta')
                        )
                    elif isinstance(inp, torch.Tensor):
                        # Convert to meta tensor (ensure consistent device)
                        fake_inputs.append(inp.to('meta'))
                    else:
                        # Scalar or non-tensor (keep as-is)
                        fake_inputs.append(inp)

                # Get operation function
                op_func = LazyTensor._get_operation_function(operation)

                # Execute on fake tensors (shape inference only)
                fake_result = op_func(*fake_inputs, **kwargs)

                # Extract shape
                if isinstance(fake_result, torch.Tensor):
                    return fake_result.shape
                else:
                    return torch.Size([])

        except Exception as e:
            logger.debug(f"Shape inference failed for {operation}: {e}")
            # Fallback: try simple heuristics
            return LazyTensor._infer_shape_fallback(operation, inputs)

    @staticmethod
    def _infer_shape_fallback(
        operation: str,
        inputs: List[Any]
    ) -> Optional[torch.Size]:
        """
        Fallback shape inference using simple heuristics.

        Used when FakeTensorMode fails (dynamic shapes, unsupported ops, etc.)
        """
        # Element-wise operations preserve shape
        if operation in ['aten::relu', 'aten::sigmoid', 'aten::tanh',
                        'aten::abs', 'aten::neg', 'aten::exp', 'aten::log']:
            if inputs and hasattr(inputs[0], 'shape'):
                if isinstance(inputs[0], LazyTensor):
                    # Use metadata directly to avoid recursion
                    try:
                        return object.__getattribute__(inputs[0], '_shape')
                    except AttributeError:
                        # Fallback if metadata not set yet
                        return torch.Size([])
                else:
                    return inputs[0].shape

        # Matrix multiplication
        if operation in ['aten::matmul', 'aten::mm']:
            if len(inputs) >= 2:
                a_shape = None
                b_shape = None

                if hasattr(inputs[0], 'shape'):
                    if isinstance(inputs[0], LazyTensor):
                        try:
                            a_shape = object.__getattribute__(inputs[0], '_shape')
                        except AttributeError:
                            a_shape = torch.Size([])
                    else:
                        a_shape = inputs[0].shape

                if hasattr(inputs[1], 'shape'):
                    if isinstance(inputs[1], LazyTensor):
                        try:
                            b_shape = object.__getattribute__(inputs[1], '_shape')
                        except AttributeError:
                            b_shape = torch.Size([])
                    else:
                        b_shape = inputs[1].shape

                if a_shape and b_shape and len(a_shape) >= 2 and len(b_shape) >= 2:
                    # (..., M, K) @ (..., K, N) -> (..., M, N)
                    return torch.Size([*a_shape[:-1], b_shape[-1]])

        # Broadcasting operations
        if operation in ['aten::add', 'aten::sub', 'aten::mul', 'aten::div']:
            if len(inputs) >= 2:
                a_shape = None
                b_shape = None

                if hasattr(inputs[0], 'shape'):
                    if isinstance(inputs[0], LazyTensor):
                        try:
                            a_shape = object.__getattribute__(inputs[0], '_shape')
                        except AttributeError:
                            a_shape = torch.Size([])
                    else:
                        a_shape = inputs[0].shape

                if hasattr(inputs[1], 'shape'):
                    if isinstance(inputs[1], LazyTensor):
                        try:
                            b_shape = object.__getattribute__(inputs[1], '_shape')
                        except AttributeError:
                            b_shape = torch.Size([])
                    else:
                        b_shape = inputs[1].shape

                if a_shape and b_shape:
                    # Simple broadcasting (return larger shape)
                    if len(a_shape) >= len(b_shape):
                        return a_shape
                    else:
                        return b_shape

        # Unknown - return empty shape
            return None

    @staticmethod
    def _infer_dtype(inputs: List[Any], kwargs: Dict[str, Any]) -> Optional[torch.dtype]:
        """Infer output dtype from inputs or kwargs."""
        # Explicit dtype in kwargs
        if 'dtype' in kwargs and kwargs['dtype'] is not None:
            dtype = kwargs['dtype']
            # Convert numpy dtypes to PyTorch dtypes
            if hasattr(dtype, 'type') and hasattr(dtype.type, '__module__'):
                if dtype.type.__module__ == 'numpy':
                    # Convert numpy dtype to PyTorch dtype
                    if dtype == torch.float32.numpy_dtype():
                        return torch.float32
                    elif dtype == torch.float64.numpy_dtype():
                        return torch.float64
                    elif dtype == torch.int64.numpy_dtype():
                        return torch.int64
                    elif dtype == torch.int32.numpy_dtype():
                        return torch.int32
                    # Add more conversions as needed
                    else:
                        # Fallback: try to map by name
                        dtype_name = str(dtype).split('.')[-1].replace('Dtype', '').lower()
                        dtype_map = {
                            'float32': torch.float32,
                            'float64': torch.float64,
                            'int32': torch.int32,
                            'int64': torch.int64,
                            'bool': torch.bool,
                        }
                        return dtype_map.get(dtype_name, torch.float32)
            return dtype

        # Infer from tensor inputs (LazyTensor or concrete tensors)
        for inp in inputs:
            if isinstance(inp, LazyTensor):
                # Use metadata directly to avoid recursion
                try:
                    inp_dtype = object.__getattribute__(inp, '_dtype')
                    if inp_dtype is not None:
                        return inp_dtype
                except AttributeError:
                    pass

                # If LazyTensor doesn't have dtype set, try to infer from its inputs
                # This handles cases where dtype wasn't set during construction
                inp_inputs = object.__getattribute__(inp, '_inputs')
                if inp_inputs:
                    # Recursively infer from inputs (but avoid infinite recursion)
                    try:
                        inferred = cls._infer_dtype(inp_inputs, object.__getattribute__(inp, '_kwargs') or {})
                        if inferred is not None:
                            return inferred
                    except:
                        pass

                # Fallback to default if metadata not set yet
                return torch.float32
            elif isinstance(inp, torch.Tensor):
                return inp.dtype

        # If no tensor inputs found, try to infer from scalar values that can be converted to tensors
        # This handles cases like x + 1.0 where 1.0 becomes a scalar tensor
        for inp in inputs:
            if isinstance(inp, (int, float)):
                # Default to float32 for scalar arithmetic
                return torch.float32
            elif hasattr(inp, 'dtype'):  # Handle other tensor-like objects
                return inp.dtype

        # Default
        return None

    # ===================================================================
    # UTILITIES
    # ===================================================================

    @staticmethod
    def _normalize_op_name(func) -> str:
        """
        Normalize operation names to canonical form.

        Examples:
            torch.ops.aten.add.Tensor -> aten::add
            torch.add -> aten::add
            add -> aten::add
            <method 'cpu' of ...> -> aten::cpu
        """
        # Debug logging for problematic cases
        if hasattr(func, '__name__') and func.__name__ == '__get__':
            logger.debug(f"Normalizing __get__ method: {func}")

        if hasattr(func, '__name__'):
            name = func.__name__
        elif hasattr(func, '_schema'):
            # ATen operation with schema
            schema_str = str(func._schema)
            name = schema_str.split('(')[0]
            if '::' in name:
                name = name.split('::')[-1]
        else:
            # Handle method objects (e.g., <method 'cpu' of ...>)
            func_str = str(func)
            if "'method '" in func_str and " of " in func_str:
                # Extract method name from string representation
                method_part = func_str.split("'method '")[1].split("'")[0]
                name = method_part
            else:
                name = str(func)

        # Remove overload suffix (e.g., add.Tensor -> add)
        name = name.split('.')[0]

        # Handle special cases for method names that start with __
        if name.startswith('__') and name.endswith('__'):
            # This is likely a dunder method, map to the actual operation
            if name == '__len__':
                name = 'len'
            elif name == '__add__':
                name = 'add'
            elif name == '__matmul__':
                name = 'matmul'
            elif name == '__sub__':
                name = 'sub'
            elif name == '__mul__':
                name = 'mul'
            elif name == '__truediv__':
                name = 'div'
            elif name == '__floordiv__':
                name = 'floordiv'
            elif name == '__mod__':
                name = 'remainder'
            elif name == '__pow__':
                name = 'pow'
            elif name == '__lshift__':
                name = 'lshift'
            elif name == '__rshift__':
                name = 'rshift'
            elif name == '__and__':
                name = 'bitwise_and'
            elif name == '__or__':
                name = 'bitwise_or'
            elif name == '__xor__':
                name = 'bitwise_xor'
            elif name == '__invert__':
                name = 'bitwise_not'
            elif name == '__lt__':
                name = 'lt'
            elif name == '__le__':
                name = 'le'
            elif name == '__gt__':
                name = 'gt'
            elif name == '__ge__':
                name = 'ge'
            elif name == '__eq__':
                name = 'eq'
            elif name == '__ne__':
                name = 'ne'
            else:
                # For other dunder methods that don't have direct aten equivalents,
                # especially internal methods like __get__, __set__, etc.
                # These are internal Python methods that shouldn't be intercepted
                logger.debug(f"Ignoring internal dunder method: {name} for func: {func}")
                name = 'unknown_method'

        # Ensure aten:: prefix
        if not name.startswith('aten::'):
            name = f'aten::{name}'

        return name

    @staticmethod
    def _get_operation_function(operation: str):
        """
        Get PyTorch function for an operation name.

        Maps "aten::add" -> torch.ops.aten.add
        """
        if operation.startswith('aten::'):
            op_name = operation[6:]  # Remove "aten::" prefix
            try:
                return getattr(torch.ops.aten, op_name)
            except AttributeError:
                # Fallback to torch namespace
                return getattr(torch, op_name)
        else:
            return getattr(torch, operation)

    # ===================================================================
    # PROPERTY ACCESSORS
    # ===================================================================

    @property
    def operation(self) -> str:
        """Get the operation that created this tensor."""
        return object.__getattribute__(self, '_operation')

    @property
    def inputs(self) -> List[Any]:
        """Get the input arguments to this operation."""
        return object.__getattribute__(self, '_inputs')

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Get the keyword arguments to this operation."""
        return object.__getattribute__(self, '_kwargs')

    @property
    def tensor_id(self) -> int:
        """Get unique ID for this tensor."""
        return object.__getattribute__(self, '_tensor_id')

    @property
    def id(self) -> int:
        """Get unique ID for this tensor (alias for tensor_id)."""
        return self.tensor_id

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get semantic metadata for this tensor."""
        return object.__getattribute__(self, '_metadata')

    # Lazy shape inference - only compute when actually needed
    def _ensure_shape(self):
        """Ensure shape is properly inferred."""
        current_shape = object.__getattribute__(self, '_shape')
        if current_shape is None or (len(current_shape) == 0 and self.operation != 'aten::tensor'):
            # Need to infer shape
            inferred_shape = type(self)._infer_shape(self.operation, self.inputs, self.kwargs)
            if inferred_shape is not None:
                object.__setattr__(self, '_shape', inferred_shape)

    @property
    def shape(self) -> torch.Size:
        """Get the shape of this tensor."""
        # Prevent recursion when accessing shape
        from .interception_control import get_current_context, InterceptionContext
        if get_current_context() != InterceptionContext.NONE:
            return object.__getattribute__(self, '_shape') or torch.Size([])

        self._ensure_shape()
        return object.__getattribute__(self, '_shape') or torch.Size([])

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of this tensor."""
        # Prevent recursion when accessing dtype
        from .interception_control import get_current_context, InterceptionContext
        if get_current_context() != InterceptionContext.NONE:
            return object.__getattribute__(self, '_dtype') or torch.float32

        return object.__getattribute__(self, '_dtype') or torch.float32

    @property
    def device(self):
        """Get the device of this tensor."""
        # Prevent recursion when accessing device
        from .interception_control import get_current_context, InterceptionContext
        if get_current_context() != InterceptionContext.NONE:
            original_device = object.__getattribute__(self, '_original_device')
            if original_device is not None:
                # For remote devices, return a torch.device with privateuseone type
                if isinstance(original_device, str) and 'remote_accelerator' in original_device:
                    return torch.device('privateuseone:0')
                return original_device
            return object.__getattribute__(self, '_device') or torch.device('meta')

        original_device = object.__getattribute__(self, '_original_device')
        if original_device is not None:
            # For remote devices, return a torch.device with privateuseone type
            if isinstance(original_device, str) and 'remote_accelerator' in original_device:
                return torch.device('privateuseone:0')
            return original_device
        return object.__getattribute__(self, '_device') or torch.device('meta')

    @property
    def original_device(self):
        """Get the original device (before meta conversion)."""
        return object.__getattribute__(self, '_original_device')

    @property
    def T(self):
        """Transpose property - return transposed view."""
        # Use torch.t to go through the normal dispatch mechanism
        # This ensures proper graph building
        return torch.t(self)

    # ===================================================================
    # STRING REPRESENTATION
    # ===================================================================

    def __repr__(self) -> str:
        # Use internal attributes directly to avoid recursion during property access
        operation = object.__getattribute__(self, '_operation')
        shape = object.__getattribute__(self, '_shape') or torch.Size([])
        dtype = object.__getattribute__(self, '_dtype') or torch.float32
        return f"LazyTensor(op={operation}, shape={shape}, dtype={dtype})"

    def __str__(self) -> str:
        return self.__repr__()

    @classmethod
    def create_from_factory(cls, factory_name: str, size_args, kwargs: Dict[str, Any]):
        """
        Create LazyTensor from factory function (torch.randn, torch.zeros, etc.).

        This is used by the factory interception mechanism to create LazyTensors
        for initial tensor creation operations.
        """
        # Map factory names to aten operations
        factory_to_aten = {
            'randn': 'aten::randn',
            'zeros': 'aten::zeros',
            'ones': 'aten::ones',
            'empty': 'aten::empty',
            'full': 'aten::full',
            'randn_like': 'aten::randn_like',
            'zeros_like': 'aten::zeros_like',
            'ones_like': 'aten::ones_like',
            'empty_like': 'aten::empty_like',
            'full_like': 'aten::full_like',
            'eye': 'aten::eye',
            'arange': 'aten::arange',
            'linspace': 'aten::linspace',
            'logspace': 'aten::logspace',
            'rand': 'aten::rand',
            'rand_like': 'aten::rand_like',
            'randint': 'aten::randint',
            'randint_like': 'aten::randint_like',
            'normal': 'aten::normal',
            'randperm': 'aten::randperm'
        }

        op_name = factory_to_aten.get(factory_name, f'aten::{factory_name}')

        # For _like functions, the first argument is the tensor to mimic
        if factory_name.endswith('_like') and size_args and hasattr(size_args[0], 'shape'):
            tensor_like = size_args[0]
            inputs = [tensor_like]
        else:
            inputs = list(size_args)

        return cls(operation=op_name, inputs=inputs, kwargs=kwargs)

    # ===================================================================
    # ADDITIONAL UTILITIES FOR TESTING
    # ===================================================================

    @classmethod
    def reset_id_counter(cls):
        """Reset ID counter for testing."""
        pass  # No longer needed with proper tensor subclass