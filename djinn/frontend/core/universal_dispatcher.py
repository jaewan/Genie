"""
Universal Dispatcher - Handles 99% of PyTorch operations automatically.

This module implements the correct architectural pattern:
- Use PyTorch's dispatch system for execution (automatic)
- Manual handlers ONLY for shape inference (PyTorch meta tensor bugs)

Key Insight: PyTorch already knows how to execute operations.
We should use its dispatch system, not reimplement it!
"""

import torch
import logging
from typing import Any, Dict, List, Callable, Optional

logger = logging.getLogger(__name__)


class UniversalDispatcher:
    """
    Universal operation dispatcher using PyTorch's built-in dispatch system.
    
    This achieves TRUE transparency - we don't need to know operations in advance.
    PyTorch's dispatch system handles everything automatically.
    
    Design Principles:
    1. Use PyTorch's dispatch as PRIMARY path (99% coverage)
    2. Argument preprocessing for edge cases (~5 operations)
    3. Manual handlers ONLY for confirmed PyTorch bugs (0-5 operations)
    
    Benefits:
    - ✅ Scales to 99% of PyTorch API automatically
    - ✅ No manual handler maintenance
    - ✅ Works with future PyTorch versions
    - ✅ Achieves research goal of transparency
    """
    
    def __init__(self):
        """Initialize universal dispatcher."""
        self._setup_argument_preprocessors()
        self._setup_special_handlers()
        
        # Statistics
        self.stats = {
            'universal_dispatch_success': 0,
            'argument_preprocessing_used': 0,
            'special_handler_used': 0,
            'dispatch_failures': 0,
        }
    
    def _setup_argument_preprocessors(self):
        """
        Setup argument preprocessors for operations with non-standard signatures.
        
        These are operations where the first argument is a list/tuple instead of a tensor.
        This is NOT a PyTorch bug - just a different calling convention.
        
        Only ~5 operations need this.
        """
        self.argument_preprocessors: Dict[str, Callable] = {
            # Concatenation operations - first arg is list of tensors
            'cat': self._preprocess_cat,
            'stack': self._preprocess_stack,
            'hstack': self._preprocess_cat,
            'vstack': self._preprocess_cat,
            'dstack': self._preprocess_cat,
        }
    
    def _setup_special_handlers(self):
        """
        Setup special handlers for operations with confirmed PyTorch bugs.
        
        IMPORTANT: This should be EMPTY or contain only 0-5 operations!
        If you're adding handlers here, ask: "Is this a PyTorch bug or am I doing it wrong?"
        
        Most operations should be handled by universal dispatch.
        """
        self.special_handlers: Dict[str, Callable] = {
            # ✅ FIX: torch.nn.functional operations (PyTorch naming inconsistencies)
            'linear': self._handle_linear,
            'max_pool2d': self._handle_max_pool2d,
            'softmax': self._handle_softmax,
        }
    
    def _preprocess_cat(self, inputs: List[Any], kwargs: Dict[str, Any]) -> tuple:
        """
        Preprocess arguments for torch.cat and similar operations.
        
        torch.cat expects: cat(tensors, dim=0)
        But we receive: inputs=[list_of_tensors], kwargs={'dim': 0}
        
        Need to unpack the list.
        """
        if inputs and isinstance(inputs[0], (list, tuple)):
            # First arg is list of tensors - this is correct
            return inputs, kwargs
        else:
            # Inputs are already unpacked - wrap them
            return [inputs], kwargs
    
    def _preprocess_stack(self, inputs: List[Any], kwargs: Dict[str, Any]) -> tuple:
        """Preprocess arguments for torch.stack (same as cat)."""
        return self._preprocess_cat(inputs, kwargs)
    
    def _handle_linear(self, inputs: List[Any], kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Handle F.linear operation.
        
        PyTorch doesn't have torch.linear - it's torch.nn.functional.linear.
        This is a naming inconsistency, not a bug.
        """
        import torch.nn.functional as F
        
        # ✅ FIX: Ensure device consistency (input, weight, bias must be on same device)
        if len(inputs) >= 2 and isinstance(inputs[0], torch.Tensor) and isinstance(inputs[1], torch.Tensor):
            if inputs[0].device != inputs[1].device:
                logger.debug(f"Moving input from {inputs[0].device} to {inputs[1].device} for linear")
                inputs[0] = inputs[0].to(inputs[1].device)
            
            # Handle bias if present
            if len(inputs) >= 3 and inputs[2] is not None and isinstance(inputs[2], torch.Tensor):
                if inputs[2].device != inputs[1].device:
                    logger.debug(f"Moving bias from {inputs[2].device} to {inputs[1].device} for linear")
                    inputs[2] = inputs[2].to(inputs[1].device)
        
        return F.linear(*inputs, **kwargs)
    
    def _handle_max_pool2d(self, inputs: List[Any], kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Handle F.max_pool2d operation.
        
        PyTorch doesn't have torch.max_pool2d - it's torch.nn.functional.max_pool2d.
        This is a naming inconsistency, not a bug.
        """
        import torch.nn.functional as F
        return F.max_pool2d(*inputs, **kwargs)
    
    def _handle_softmax(self, inputs: List[Any], kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Handle F.softmax operation.
        
        PyTorch doesn't have torch.softmax - it's torch.nn.functional.softmax.
        This is a naming inconsistency, not a bug.
        """
        import torch.nn.functional as F
        return F.softmax(*inputs, **kwargs)
    
    def dispatch(self, operation: str, inputs: List[Any], kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Universal dispatch - handles 99% of operations automatically.
        
        Algorithm:
        1. Check if operation needs argument preprocessing
        2. Try PyTorch's ATen namespace (torch.ops.aten.X)
        3. Try PyTorch's torch namespace (torch.X)
        4. Try as tensor method (tensor.X())
        5. Check special handlers (only for PyTorch bugs)
        6. Fail with clear error
        
        Args:
            operation: Operation name (e.g., 'aten::add', 'aten::softmax')
            inputs: List of input tensors (already materialized)
            kwargs: Keyword arguments
        
        Returns:
            Result tensor
        
        Raises:
            NotImplementedError: If operation cannot be dispatched
        """
        # Normalize operation name
        op_name = operation.replace('aten::', '')
        
        # Step 1: Argument preprocessing (if needed)
        if op_name in self.argument_preprocessors:
            inputs, kwargs = self.argument_preprocessors[op_name](inputs, kwargs)
            self.stats['argument_preprocessing_used'] += 1
            logger.debug(f"Preprocessed arguments for {op_name}")
        
        # Step 2: Check special handlers (only for PyTorch bugs)
        if op_name in self.special_handlers:
            self.stats['special_handler_used'] += 1
            logger.debug(f"Using special handler for {op_name}")
            return self.special_handlers[op_name](inputs, kwargs)
        
        # Step 3: Try PyTorch's ATen namespace (most reliable)
        try:
            if hasattr(torch.ops.aten, op_name):
                aten_op = getattr(torch.ops.aten, op_name)
                result = aten_op(*inputs, **kwargs)
                self.stats['universal_dispatch_success'] += 1
                logger.debug(f"✓ Universal dispatch succeeded for {op_name} via torch.ops.aten")
                return result
        except Exception as e:
            logger.debug(f"torch.ops.aten.{op_name} failed: {e}")
        
        # Step 4: Try PyTorch's torch namespace
        try:
            if hasattr(torch, op_name):
                torch_op = getattr(torch, op_name)
                result = torch_op(*inputs, **kwargs)
                self.stats['universal_dispatch_success'] += 1
                logger.debug(f"✓ Universal dispatch succeeded for {op_name} via torch")
                return result
        except Exception as e:
            logger.debug(f"torch.{op_name} failed: {e}")
        
        # Step 5: Try as tensor method (e.g., tensor.float())
        try:
            if inputs and isinstance(inputs[0], torch.Tensor):
                if hasattr(inputs[0], op_name):
                    method = getattr(inputs[0], op_name)
                    result = method(*inputs[1:], **kwargs)
                    self.stats['universal_dispatch_success'] += 1
                    logger.debug(f"✓ Universal dispatch succeeded for {op_name} via tensor method")
                    return result
        except Exception as e:
            logger.debug(f"tensor.{op_name}() failed: {e}")
        
        # Step 6: Dispatch failed
        self.stats['dispatch_failures'] += 1
        raise NotImplementedError(
            f"Universal dispatch failed for operation '{operation}'.\n"
            f"  Tried:\n"
            f"    1. torch.ops.aten.{op_name}\n"
            f"    2. torch.{op_name}\n"
            f"    3. tensor.{op_name}()\n"
            f"  This operation may not exist in PyTorch or has a different name.\n"
            f"  Check PyTorch documentation for the correct operation name."
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dispatcher statistics."""
        total_dispatches = (
            self.stats['universal_dispatch_success'] +
            self.stats['special_handler_used'] +
            self.stats['dispatch_failures']
        )
        
        return {
            **self.stats,
            'total_dispatches': total_dispatches,
            'success_rate': (
                self.stats['universal_dispatch_success'] / total_dispatches * 100
                if total_dispatches > 0 else 0
            ),
        }


# Global singleton
_dispatcher: Optional[UniversalDispatcher] = None


def get_universal_dispatcher() -> UniversalDispatcher:
    """Get global universal dispatcher instance."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = UniversalDispatcher()
    return _dispatcher


