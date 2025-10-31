"""
Automatic Dispatch System for LazyTensor Operations.

This module implements Phase 2 of the strategic plan: automatic operation dispatch
using PyTorch's meta tensor system. This eliminates the need for manual shape inference
and special handlers for most operations.

Key Features:
1. Automatic shape inference via meta tensors
2. Handles ALL PyTorch operations (2000+)
3. No manual handlers needed (except truly special cases)
4. Scalable and maintainable

Architecture:
    LazyTensor → Meta Tensor → PyTorch Op → Meta Result → LazyTensor
"""

import torch
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable

logger = logging.getLogger(__name__)


class AutomaticDispatch:
    """
    Automatic dispatch system for LazyTensor operations.
    
    Uses PyTorch's meta tensor system to automatically infer shapes, dtypes,
    and other metadata for operations without manual intervention.
    """
    
    # Operations that require special handling (cannot use meta tensors)
    SPECIAL_OPS = {
        # Materialization operations
        'aten::item',
        'aten::numpy',
        'aten::to_numpy',
        'aten::cpu',
        'aten::cuda',
        'aten::to',
        
        # Operations that return non-tensors
        'aten::size',
        'aten::numel',
        'aten::dim',
        'aten::is_contiguous',
        
        # Operations with side effects
        'aten::print',
        'aten::save',
        'aten::load',
    }
    
    @classmethod
    def should_use_automatic_dispatch(cls, func: Callable, args: Tuple, kwargs: Dict) -> bool:
        """
        Determine if we should use automatic dispatch for this operation.
        
        Returns:
            bool: True if automatic dispatch should be used, False if special handling needed
        """
        # Get operation name
        op_name = cls._get_op_name(func)
        
        # Check if it's a special operation
        if op_name in cls.SPECIAL_OPS:
            return False
        
        # Check if function name suggests special handling
        if hasattr(func, '__name__'):
            func_name = func.__name__
            # Materialization operations
            if func_name in ('item', 'numpy', 'tolist', '__len__', '__bool__', '__int__', '__float__'):
                return False
        
        return True
    
    @classmethod
    def dispatch(cls, func: Callable, args: Tuple, kwargs: Dict, lazy_tensor_class: type) -> Any:
        """
        Automatically dispatch operation using meta tensors.
        
        Args:
            func: PyTorch function to call
            args: Function arguments (may contain LazyTensors)
            kwargs: Function keyword arguments (may contain LazyTensors)
            lazy_tensor_class: LazyTensor class for creating result
            
        Returns:
            LazyTensor or tuple of LazyTensors with inferred metadata
        """
        try:
            # Step 1: Convert all LazyTensors to meta tensors
            meta_args, arg_mapping = cls._to_meta_tensors(args)
            meta_kwargs, kwarg_mapping = cls._to_meta_tensors(kwargs)
            
            # Step 2: Call function with meta tensors (automatic shape inference!)
            with torch.device('meta'):
                meta_result = func(*meta_args, **meta_kwargs)
            
            # Step 3: Convert meta result back to LazyTensor(s)
            result = cls._from_meta_result(
                meta_result,
                func,
                args,
                kwargs,
                lazy_tensor_class,
                arg_mapping,
                kwarg_mapping
            )
            
            return result
            
        except Exception as e:
            # If automatic dispatch fails, log and return None (caller will use fallback)
            logger.debug(f"Automatic dispatch failed for {cls._get_op_name(func)}: {e}")
            return None
    
    @classmethod
    def _to_meta_tensors(cls, obj: Any) -> Tuple[Any, Dict[int, Any]]:
        """
        Convert LazyTensors to meta tensors recursively.
        
        Args:
            obj: Object to convert (can be tensor, list, tuple, dict, etc.)
            
        Returns:
            Tuple of (converted_obj, mapping) where mapping tracks original LazyTensors
        """
        mapping = {}
        
        def convert(x, path=()):
            if type(x).__name__ == 'LazyTensor':
                # Convert LazyTensor to meta tensor
                obj_id = id(x)
                mapping[obj_id] = x
                
                # Get shape, dtype, device from LazyTensor
                shape = x.shape if hasattr(x, 'shape') else torch.Size([])
                dtype = x.dtype if hasattr(x, 'dtype') else torch.float32
                
                # Create meta tensor with same metadata
                # Disable interception to avoid detach() issues
                from .interception_control import disable_interception, InterceptionContext
                with disable_interception(InterceptionContext.CONSTRUCTION):
                    meta_tensor = torch.empty(shape, dtype=dtype, device='meta')
                return meta_tensor
                
            elif isinstance(x, torch.Tensor):
                # Regular tensor - convert to meta
                return torch.empty(x.shape, dtype=x.dtype, device='meta')
                
            elif isinstance(x, (list, tuple)):
                converted = [convert(item, path + (i,)) for i, item in enumerate(x)]
                return type(x)(converted)
                
            elif isinstance(x, dict):
                return {k: convert(v, path + (k,)) for k, v in x.items()}
                
            else:
                # Non-tensor types (scalars, strings, etc.) - return as-is
                return x
        
        converted = convert(obj)
        return converted, mapping
    
    @classmethod
    def _from_meta_result(
        cls,
        meta_result: Any,
        func: Callable,
        original_args: Tuple,
        original_kwargs: Dict,
        lazy_tensor_class: type,
        arg_mapping: Dict[int, Any],
        kwarg_mapping: Dict[int, Any]
    ) -> Any:
        """
        Convert meta tensor result back to LazyTensor(s).
        
        Args:
            meta_result: Result from calling func with meta tensors
            func: Original function
            original_args: Original arguments (with LazyTensors)
            original_kwargs: Original kwargs (with LazyTensors)
            lazy_tensor_class: LazyTensor class
            arg_mapping: Mapping of LazyTensor IDs to original LazyTensors
            kwarg_mapping: Mapping of LazyTensor IDs to original LazyTensors
            
        Returns:
            LazyTensor or structure of LazyTensors matching meta_result structure
        """
        op_name = cls._get_op_name(func)
        
        def convert(meta_obj):
            if isinstance(meta_obj, torch.Tensor):
                # Meta tensor result - convert to LazyTensor
                
                # Infer device from input LazyTensors
                inferred_device = cls._infer_device(original_args, original_kwargs)
                
                # Get metadata from meta tensor
                shape = meta_obj.shape
                dtype = meta_obj.dtype
                
                # Create LazyTensor with inferred metadata
                from .metadata_capture import get_metadata_capture
                metadata = get_metadata_capture().capture_metadata(
                    operation=op_name,
                    inputs=list(original_args),
                    kwargs=original_kwargs
                )
                
                return lazy_tensor_class(
                    operation=op_name,
                    inputs=list(original_args),
                    kwargs=original_kwargs,
                    shape=shape,
                    dtype=dtype,
                    device=inferred_device,
                    metadata=metadata
                )
                
            elif isinstance(meta_obj, (list, tuple)):
                converted = [convert(item) for item in meta_obj]
                return type(meta_obj)(converted)
                
            elif isinstance(meta_obj, dict):
                return {k: convert(v) for k, v in meta_obj.items()}
                
            else:
                # Non-tensor result (scalar, None, etc.)
                return meta_obj
        
        return convert(meta_result)
    
    @classmethod
    def _infer_device(cls, args: Tuple, kwargs: Dict) -> torch.device:
        """
        Infer device from input arguments.
        
        Args:
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            torch.device: Inferred device
        """
        # Check kwargs first
        if 'device' in kwargs:
            device = kwargs['device']
            if isinstance(device, str):
                return torch.device(device)
            return device
        
        # Check args for LazyTensors
        def find_device(obj):
            if type(obj).__name__ == 'LazyTensor':
                if hasattr(obj, '_logical_device') and obj._logical_device:
                    return obj._logical_device
                if hasattr(obj, 'device') and obj.device:
                    return obj.device
            elif isinstance(obj, torch.Tensor):
                return obj.device
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    device = find_device(item)
                    if device:
                        return device
            elif isinstance(obj, dict):
                for value in obj.values():
                    device = find_device(value)
                    if device:
                        return device
            return None
        
        # Search args
        for arg in args:
            device = find_device(arg)
            if device:
                return device
        
        # Search kwargs
        for value in kwargs.values():
            device = find_device(value)
            if device:
                return device
        
        # Default to CPU
        return torch.device('cpu')
    
    @classmethod
    def _get_op_name(cls, func: Callable) -> str:
        """
        Get normalized operation name from function.
        
        Args:
            func: PyTorch function
            
        Returns:
            str: Normalized operation name (e.g., 'aten::add')
        """
        if hasattr(func, '__name__'):
            name = func.__name__
            
            # Handle torch.ops.aten.* format
            if hasattr(func, '__module__') and 'torch._ops' in func.__module__:
                return f'aten::{name}'
            
            # Handle torch.nn.functional.* format
            if hasattr(func, '__module__') and 'torch.nn.functional' in func.__module__:
                return f'aten::{name}'
            
            # Default format
            return f'aten::{name}'
        
        # Fallback
        return str(func)


def create_automatic_dispatcher() -> AutomaticDispatch:
    """
    Factory function to create an AutomaticDispatch instance.
    
    Returns:
        AutomaticDispatch: Configured dispatcher
    """
    return AutomaticDispatch()


# Global dispatcher instance
_global_dispatcher = None


def get_automatic_dispatcher() -> AutomaticDispatch:
    """
    Get the global automatic dispatcher instance.
    
    Returns:
        AutomaticDispatch: Global dispatcher
    """
    global _global_dispatcher
    if _global_dispatcher is None:
        _global_dispatcher = create_automatic_dispatcher()
    return _global_dispatcher

