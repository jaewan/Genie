from __future__ import annotations

import logging
import os
import threading
from itertools import count
from typing import Any, Dict, List, Optional, Tuple
import torch

logger = logging.getLogger(__name__)

# Thread-local storage to track when we're inside __torch_function__
# This prevents infinite recursion when accessing tensor properties
_in_torch_function = threading.local()


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
    _shape_cache: Dict[Tuple, Optional[torch.Size]] = {}

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
    ):
        """
        Create LazyTensor wrapper.

        CRITICAL: Must use _make_subclass for proper tensor subclass.
        This creates a tensor wrapper WITHOUT allocating actual storage.
        """
        # Infer shape/dtype if not provided
        if shape is None:
            shape = torch.Size([])  # Use empty shape as placeholder
            # Shape inference will be done lazily when needed

        if dtype is None:
            dtype = cls._infer_dtype(inputs, kwargs or {})
        if dtype is None:
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

        # Replace the original device in kwargs with the processed device for __init__
        if kwargs:
            kwargs = kwargs.copy()
            kwargs['device'] = device

        return wrapper

    def __init__(
        self,
        operation: str,
        inputs: List[Any],
        kwargs: Optional[Dict[str, Any]] = None,
        shape: Optional[torch.Size] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
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

        # Store original device info (for remote devices) - need to extract from original kwargs before processing
        # The device parameter here is already processed (meta), so we need to look at the original kwargs
        original_device = None
        if hasattr(self, '_kwargs') and self._kwargs:
            original_device = self._kwargs.get('device')

        object.__setattr__(self, '_original_device', original_device)

        # Register with hybrid graph builder
        if LazyTensor._graph_builder is not None:
            LazyTensor._graph_builder.add_operation(self)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """
        Intercept ALL operations involving LazyTensor.

        This is automatically called by PyTorch's dispatcher for any
        operation that involves a LazyTensor operand.

        Args:
            func: ATen operation (e.g., torch.ops.aten.add.Tensor)
            types: Tuple of types involved (contains LazyTensor)
            args: Positional arguments (default empty tuple)
            kwargs: Keyword arguments (default None)

        Returns:
            LazyTensor representing the deferred operation
        """
        kwargs = kwargs or {}

        # Handle special operations that force materialization
        if func in cls._MATERIALIZATION_OPS:
            # These operations need concrete tensors
            materialized_args = tuple(
                arg.materialize() if type(arg).__name__ == 'LazyTensor' else arg
                for arg in args
            )
            return func(*materialized_args, **kwargs)

        # Check if we're outside a capture context with LazyTensors
        # In this case, materialize all LazyTensors and let PyTorch handle it
        from .capture import is_capturing
        if not is_capturing():
            # Count LazyTensors vs concrete tensors
            lazy_count = sum(1 for arg in args if type(arg).__name__ == 'LazyTensor')
            concrete_tensor_count = sum(1 for arg in args if isinstance(arg, torch.Tensor) and type(arg).__name__ != 'LazyTensor')
            
            # If we have LazyTensors with concrete tensors outside capture context,
            # materialize all LazyTensors and let PyTorch handle it normally
            if lazy_count > 0 and concrete_tensor_count > 0:
                # Materialize all LazyTensor arguments
                materialized_args = tuple(
                    arg.materialize() if type(arg).__name__ == 'LazyTensor' else arg
                    for arg in args
                )
                return func(*materialized_args, **kwargs)

        # Normalize operation name
        op_name = cls._normalize_op_name(func)

        # Create new LazyTensor for the result
        result = cls(
            operation=op_name,
            inputs=list(args),
            kwargs=kwargs
        )

        return result

    # Operator methods that route through torch_dispatch
    def __matmul__(self, other):
        """Matrix multiplication using @ operator."""
        return torch.matmul(self, other)
    
    def __rmatmul__(self, other):
        """Right matrix multiplication."""
        return torch.matmul(other, self)
    
    def __add__(self, other):
        """Addition using + operator."""
        return torch.add(self, other)
    
    def __radd__(self, other):
        """Right addition."""
        return torch.add(other, self)
    
    def __sub__(self, other):
        """Subtraction using - operator."""
        return torch.sub(self, other)
    
    def __rsub__(self, other):
        """Right subtraction."""
        return torch.sub(other, self)
    
    def __mul__(self, other):
        """Multiplication using * operator."""
        return torch.mul(self, other)
    
    def __rmul__(self, other):
        """Right multiplication."""
        return torch.mul(other, self)
    
    def __truediv__(self, other):
        """Division using / operator."""
        return torch.div(self, other)
    
    def __rtruediv__(self, other):
        """Right division."""
        return torch.div(other, self)

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
        if getattr(_in_torch_function, 'active', False):
            return NotImplemented

        _in_torch_function.active = True
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
            if func in cls._MATERIALIZATION_OPS:
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
                        # Special handling for __len__
                        if func.__name__ == '__len__':
                            return len(materialized)
                        # Call the method directly on the materialized tensor
                        method_name = func.__name__
                        method = getattr(materialized, method_name)
                        return method(*args[1:], **kwargs)
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
                    for arg in args:
                        if type(arg).__name__ == 'LazyTensor':
                            materialized_args.append(arg.materialize())
                        else:
                            materialized_args.append(arg)
                    # Call the function with materialized arguments
                    return func(*materialized_args, **kwargs)

            # For operations involving LazyTensors inside capture context, create new LazyTensor
            op_name = cls._normalize_op_name(func)
            result = cls(
                operation=op_name,
                inputs=list(args),
                kwargs=kwargs
            )
            return result
        finally:
            _in_torch_function.active = False

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

        return cls(
            operation='aten::randn',
            inputs=list(size),
            kwargs={'dtype': dtype, 'device': device, 'requires_grad': requires_grad},
            shape=torch.Size(size),
            dtype=dtype or torch.float32,
            device=device
        )

    @classmethod
    def tensor(cls, data, dtype=None, device=None, requires_grad=False):
        """Create LazyTensor from data."""
        # For tensor creation from data, we already know the shape and don't need torch.empty
        shape = torch.Size(data.shape) if hasattr(data, 'shape') else torch.Size([])
        inferred_dtype = dtype or (data.dtype if hasattr(data, 'dtype') else torch.float32)

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

        # Replace the original device in kwargs with the processed device for __init__
        kwargs = {'dtype': dtype, 'device': device, 'requires_grad': requires_grad}
        if kwargs:
            kwargs = kwargs.copy()
            kwargs['device'] = device

        # Initialize the wrapper
        wrapper.__init__(
            operation='aten::tensor',
            inputs=[data],
            kwargs=kwargs,
            shape=shape,
            dtype=inferred_dtype,
            device=device
        )

        return wrapper

    @classmethod
    def as_tensor(cls, data, dtype=None, device=None):
        """Create LazyTensor from data (alias for tensor)."""
        # Create LazyTensor with correct operation name
        shape = torch.Size(data.shape) if hasattr(data, 'shape') else torch.Size([])
        inferred_dtype = dtype or (data.dtype if hasattr(data, 'dtype') else torch.float32)

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
        wrapper = torch.Tensor._make_subclass(
            cls,
            torch.tensor(0, dtype=torch_dtype, device=device).expand(shape),
            require_grad=False  # Disable autograd for now (Phase 2 addition)
        )

        # Replace the original device in kwargs with the processed device for __init__
        kwargs = {'dtype': dtype, 'device': device}
        if kwargs:
            kwargs = kwargs.copy()
            kwargs['device'] = device

        # Initialize the wrapper with as_tensor operation
        wrapper.__init__(
            operation='aten::as_tensor',
            inputs=[data],
            kwargs=kwargs,
            shape=shape,
            dtype=torch_dtype,
            device=device
        )

        return wrapper

    @classmethod
    def from_numpy(cls, ndarray, dtype=None, device=None):
        """Create LazyTensor from numpy array."""
        return cls(
            operation='aten::from_numpy',
            inputs=[ndarray],
            kwargs={'dtype': dtype, 'device': device},
            shape=torch.Size(ndarray.shape),
            dtype=dtype or torch.from_numpy(ndarray).dtype,
            device=device
        )

    @classmethod
    def zeros(cls, *size, dtype=None, device=None, requires_grad=False):
        """Create zero-filled LazyTensor."""
        # Handle case where size is passed as torch.Size or tuple (FIX: Simplified)
        if len(size) == 1 and isinstance(size[0], (torch.Size, tuple, list)):
            size = tuple(size[0])
        else:
            size = tuple(size)

        return cls(
            operation='aten::zeros',
            inputs=list(size),
            kwargs={'dtype': dtype, 'device': device, 'requires_grad': requires_grad},
            shape=torch.Size(size),
            dtype=dtype or torch.float32,
            device=device
        )

    @classmethod
    def ones(cls, *size, dtype=None, device=None, requires_grad=False):
        """Create one-filled LazyTensor."""
        # Handle case where size is passed as torch.Size or tuple (FIX: Simplified)
        if len(size) == 1 and isinstance(size[0], (torch.Size, tuple, list)):
            size = tuple(size[0])
        else:
            size = tuple(size)

        return cls(
            operation='aten::ones',
            inputs=list(size),
            kwargs={'dtype': dtype, 'device': device, 'requires_grad': requires_grad},
            shape=torch.Size(size),
            dtype=dtype or torch.float32,
            device=device
        )

    @classmethod
    def empty(cls, *size, dtype=None, device=None, requires_grad=False):
        """Create empty LazyTensor (uninitialized memory)."""
        # Handle case where size is passed as torch.Size or tuple (FIX: Simplified)
        if len(size) == 1 and isinstance(size[0], (torch.Size, tuple, list)):
            size = tuple(size[0])
        else:
            size = tuple(size)

        return cls(
            operation='aten::empty',
            inputs=list(size),
            kwargs={'dtype': dtype, 'device': device, 'requires_grad': requires_grad},
            shape=torch.Size(size),
            dtype=dtype or torch.float32,
            device=device
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
        if LazyTensor._graph_builder is None:
            raise RuntimeError("No graph builder registered")

        return LazyTensor._graph_builder.materialize(self)


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

        # Create cache key from operation and input shapes
        input_shapes = []
        for inp in inputs:
            if hasattr(inp, 'shape'):
                input_shapes.append(tuple(inp.shape))
            else:
                input_shapes.append(None)

        cache_key = (operation, tuple(input_shapes))

        # Check cache first
        if cache_key in LazyTensor._shape_cache:
            return LazyTensor._shape_cache[cache_key]

        try:
            # Use PyTorch's FakeTensor mode for accurate shape inference
            from torch._subclasses.fake_tensor import FakeTensorMode

            with FakeTensorMode():
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
                        # Convert to meta tensor
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
                    result_shape = fake_result.shape
                else:
                    result_shape = torch.Size([])

                # Cache result
                LazyTensor._shape_cache[cache_key] = result_shape
                return result_shape

        except Exception as e:
            logger.debug(f"Shape inference failed for {operation}: {e}")
            # Fallback: try simple heuristics
            result = cls._infer_shape_fallback(operation, inputs)
            # Cache fallback result too
            LazyTensor._shape_cache[cache_key] = result
            return result

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
            return kwargs['dtype']

        # Infer from first tensor input
        for inp in inputs:
            if isinstance(inp, LazyTensor):
                # Use metadata directly to avoid recursion
                try:
                    return object.__getattribute__(inp, '_dtype')
                except AttributeError:
                    # Fallback to default if metadata not set yet
                    return torch.float32
            elif isinstance(inp, torch.Tensor):
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
        """
        if hasattr(func, '__name__'):
            name = func.__name__
        elif hasattr(func, '_schema'):
            # ATen operation with schema
            schema_str = str(func._schema)
            name = schema_str.split('(')[0]
            if '::' in name:
                name = name.split('::')[-1]
        else:
            name = str(func)

        # Remove overload suffix (e.g., add.Tensor -> add)
        name = name.split('.')[0]

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
        if getattr(_in_torch_function, 'active', False):
            return object.__getattribute__(self, '_shape') or torch.Size([])

        self._ensure_shape()
        return object.__getattribute__(self, '_shape') or torch.Size([])

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of this tensor."""
        # Prevent recursion when accessing dtype
        if getattr(_in_torch_function, 'active', False):
            return object.__getattribute__(self, '_dtype') or torch.float32

        return object.__getattribute__(self, '_dtype') or torch.float32

    @property
    def device(self):
        """Get the device of this tensor."""
        # Prevent recursion when accessing device
        if getattr(_in_torch_function, 'active', False):
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