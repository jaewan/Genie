"""
Shared operation registry for Genie framework.

Provides a centralized registry of PyTorch operations that can be executed
on both client and server sides. Eliminates code duplication between
executor.py and subgraph_executor.py.

Design principles:
- Single source of truth for supported operations
- Consistent execution across client and server
- Extensible for new operations
- Proper error handling and fallbacks
"""

import torch
import torch.nn.functional as F
from typing import Dict, Callable, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class OperationRegistry:
    """Centralized registry for PyTorch operations."""

    _instance: Optional['OperationRegistry'] = None
    _registry: Optional[Dict[str, Callable]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._build_registry()
        return cls._instance

    def _build_registry(self) -> None:
        """Build the operation registry with all supported operations."""
        self._registry = {
            # Arithmetic operations
            'aten::add': lambda inputs, kwargs: torch.add(inputs[0], inputs[1], **kwargs),
            'aten::sub': lambda inputs, kwargs: torch.sub(inputs[0], inputs[1], **kwargs),
            'aten::mul': lambda inputs, kwargs: torch.mul(inputs[0], inputs[1], **kwargs),
            'aten::div': lambda inputs, kwargs: torch.div(inputs[0], inputs[1], **kwargs),

            # Linear algebra operations
            'aten::matmul': lambda inputs, kwargs: torch.matmul(inputs[0], inputs[1]),
            'aten::mm': lambda inputs, kwargs: torch.mm(inputs[0], inputs[1]),
            'aten::bmm': lambda inputs, kwargs: torch.bmm(inputs[0], inputs[1]),

            # Transpose operations
            'aten::t': lambda inputs, kwargs: torch.t(inputs[0]),
            'aten::transpose': lambda inputs, kwargs: torch.transpose(inputs[0], **kwargs),

            # Activation functions
            'aten::relu': lambda inputs, kwargs: torch.relu(inputs[0]),
            'aten::sigmoid': lambda inputs, kwargs: torch.sigmoid(inputs[0]),
            'aten::tanh': lambda inputs, kwargs: torch.tanh(inputs[0]),
            'aten::gelu': lambda inputs, kwargs: F.gelu(inputs[0]),
            'aten::leaky_relu': lambda inputs, kwargs: F.leaky_relu(inputs[0], **kwargs),
            'aten::elu': lambda inputs, kwargs: F.elu(inputs[0], **kwargs),

            # Element-wise operations
            'aten::abs': lambda inputs, kwargs: torch.abs(inputs[0]),
            'aten::neg': lambda inputs, kwargs: torch.neg(inputs[0]),
            'aten::exp': lambda inputs, kwargs: torch.exp(inputs[0]),
            'aten::log': lambda inputs, kwargs: torch.log(inputs[0]),
            'aten::sqrt': lambda inputs, kwargs: torch.sqrt(inputs[0]),
            'aten::square': lambda inputs, kwargs: torch.square(inputs[0]),
            'aten::pow': lambda inputs, kwargs: torch.pow(inputs[0], inputs[1] if len(inputs) > 1 else 2),

            # Reduction operations
            'aten::sum': lambda inputs, kwargs: torch.sum(inputs[0], **kwargs),
            'aten::mean': lambda inputs, kwargs: torch.mean(inputs[0], **kwargs),
            'aten::softmax': lambda inputs, kwargs: torch.softmax(inputs[0], **kwargs),
            'aten::log_softmax': lambda inputs, kwargs: torch.log_softmax(inputs[0], **kwargs),

            # Convolution operations
            'aten::conv2d': lambda inputs, kwargs: torch.ops.aten.conv2d(*inputs, **kwargs),
            'aten::conv1d': lambda inputs, kwargs: torch.ops.aten.conv1d(*inputs, **kwargs),
            'aten::conv3d': lambda inputs, kwargs: torch.ops.aten.conv3d(*inputs, **kwargs),

            # Pooling operations
            'aten::max_pool2d': lambda inputs, kwargs: torch.ops.aten.max_pool2d(inputs[0], **kwargs),
            'aten::avg_pool2d': lambda inputs, kwargs: torch.ops.aten.avg_pool2d(inputs[0], **kwargs),
            'aten::max_pool1d': lambda inputs, kwargs: torch.ops.aten.max_pool1d(inputs[0], **kwargs),
            'aten::avg_pool1d': lambda inputs, kwargs: torch.ops.aten.avg_pool1d(inputs[0], **kwargs),

            # Adaptive pooling
            'aten::adaptive_avg_pool2d': lambda inputs, kwargs: F.adaptive_avg_pool2d(inputs[0], **kwargs),
            'aten::adaptive_max_pool2d': lambda inputs, kwargs: F.adaptive_max_pool2d(inputs[0], **kwargs),
            'aten::adaptive_avg_pool1d': lambda inputs, kwargs: F.adaptive_avg_pool1d(inputs[0], **kwargs),

            # Normalization
            'aten::batch_norm': lambda inputs, kwargs: torch.ops.aten.batch_norm(*inputs, **kwargs),
            'aten::layer_norm': lambda inputs, kwargs: F.layer_norm(inputs[0], **kwargs),
            'aten::instance_norm': lambda inputs, kwargs: F.instance_norm(inputs[0], **kwargs),

            # Dropout
            'aten::dropout': lambda inputs, kwargs: F.dropout(inputs[0], **kwargs),
            'aten::dropout2d': lambda inputs, kwargs: F.dropout2d(inputs[0], **kwargs),

            # Interpolation
            'aten::interpolate': lambda inputs, kwargs: F.interpolate(inputs[0], **kwargs),
            'aten::upsample': lambda inputs, kwargs: F.interpolate(inputs[0], **kwargs),

            # Tensor manipulation
            'aten::reshape': lambda inputs, kwargs: inputs[0].reshape(*inputs[1:] if len(inputs) > 1 else kwargs.get('shape', inputs[0].shape)),
            'aten::view': lambda inputs, kwargs: inputs[0].view(*inputs[1:] if len(inputs) > 1 else kwargs.get('shape', inputs[0].shape)),
            'aten::flatten': lambda inputs, kwargs: inputs[0].flatten(**kwargs),
            'aten::squeeze': lambda inputs, kwargs: inputs[0].squeeze(**kwargs),
            'aten::unsqueeze': lambda inputs, kwargs: inputs[0].unsqueeze(**kwargs),
            'aten::split': lambda inputs, kwargs: torch.split(*inputs, **kwargs),
            'aten::chunk': lambda inputs, kwargs: torch.chunk(*inputs, **kwargs),
            'aten::cat': lambda inputs, kwargs: torch.cat(inputs, **kwargs),
            'aten::stack': lambda inputs, kwargs: torch.stack(inputs, **kwargs),

            # Special operations
            'aten::alias': lambda inputs, kwargs: inputs[0],  # Pass-through
            'aten::clone': lambda inputs, kwargs: inputs[0].clone(),
            'aten::detach': lambda inputs, kwargs: inputs[0].detach(),
            'aten::contiguous': lambda inputs, kwargs: inputs[0].contiguous(),

            # Linear layer operations
            'aten::linear': lambda inputs, kwargs: torch.nn.functional.linear(*inputs, **kwargs),

            # Device operations (should not appear in subgraphs, but handle gracefully)
            'aten::cpu': lambda inputs, kwargs: inputs[0].cpu(),
            'aten::cuda': lambda inputs, kwargs: inputs[0].cuda(**kwargs),
            'aten::to': lambda inputs, kwargs: inputs[0].to(**kwargs),

            # Additional operations that tests might expect
            'aten::zeros': lambda inputs, kwargs: torch.zeros(*inputs, **kwargs),
            'aten::ones': lambda inputs, kwargs: torch.ones(*inputs, **kwargs),
            'aten::randn': lambda inputs, kwargs: torch.randn(*inputs, **kwargs),
            'aten::eye': lambda inputs, kwargs: torch.eye(*inputs, **kwargs),
            'aten::arange': lambda inputs, kwargs: torch.arange(*inputs, **kwargs),
        }

        # Add aliases for common operations
        self._add_aliases()

        logger.info(f"Operation registry built with {len(self._registry)} operations")

    def _add_aliases(self) -> None:
        """Add common aliases for operations."""
        aliases = {
            # PyTorch functional aliases
            'aten::F.relu': self._registry['aten::relu'],
            'aten::F.sigmoid': self._registry['aten::sigmoid'],
            'aten::F.tanh': self._registry['aten::tanh'],
            'aten::F.gelu': self._registry['aten::gelu'],

            # NumPy-style aliases
            'aten::add_': self._registry['aten::add'],
            'aten::sub_': self._registry['aten::sub'],
            'aten::mul_': self._registry['aten::mul'],
            'aten::div_': self._registry['aten::div'],

            # Common variations
            'aten::matmul_': self._registry['aten::matmul'],
            'aten::mm_': self._registry['aten::mm'],
            'aten::bmm_': self._registry['aten::bmm'],

            # Additional aliases that tests might expect
            'aten::zeros_like': lambda inputs, kwargs: torch.zeros_like(inputs[0], **kwargs),
            'aten::ones_like': lambda inputs, kwargs: torch.ones_like(inputs[0], **kwargs),
            'aten::empty': lambda inputs, kwargs: torch.empty(*inputs, **kwargs),
            'aten::full': lambda inputs, kwargs: torch.full(*inputs, **kwargs),
        }

        self._registry.update(aliases)

    def execute(self, operation: str, inputs: List[torch.Tensor], kwargs: Optional[Dict] = None) -> torch.Tensor:
        """
        Execute an operation with the given inputs.

        Args:
            operation: Operation name (e.g., 'aten::relu', 'aten::matmul')
            inputs: List of input tensors
            kwargs: Optional keyword arguments for the operation

        Returns:
            Result tensor

        Raises:
            NotImplementedError: If operation is not supported
        """
        if kwargs is None:
            kwargs = {}

        # Check registry first
        op_func = self._registry.get(operation)
        if op_func:
            try:
                return op_func(inputs, kwargs)
            except Exception as e:
                logger.error(f"Registry execution failed for {operation}: {e}")
                raise

        # Fallback to dynamic dispatch
        return self._execute_dynamic(operation, inputs, kwargs)

    def _execute_dynamic(self, operation: str, inputs: List[torch.Tensor], kwargs: Dict) -> torch.Tensor:
        """Fallback: dynamic operation dispatch using torch namespaces."""
        # Normalize operation name
        op_name = operation.replace('aten::', '')

        # Try torch namespace first
        try:
            torch_func = getattr(torch, op_name, None)
            if torch_func:
                return torch_func(*inputs, **kwargs)
        except Exception as e:
            logger.debug(f"torch.{op_name} failed: {e}")

        # Try torch.nn.functional
        try:
            func = getattr(F, op_name, None)
            if func:
                return func(*inputs, **kwargs)
        except Exception as e:
            logger.debug(f"torch.nn.functional.{op_name} failed: {e}")

        # Try torch.ops.aten
        try:
            aten_func = getattr(torch.ops.aten, op_name, None)
            if aten_func:
                return aten_func(*inputs, **kwargs)
        except Exception as e:
            logger.debug(f"torch.ops.aten.{op_name} failed: {e}")

        # Last resort: raise error with helpful message
        available_ops = sorted(self._registry.keys())[:20]  # Show first 20
        raise NotImplementedError(
            f"Operation '{operation}' not supported. "
            f"Available operations: {available_ops}... "
            f"Total supported: {len(self._registry)} operations."
        )

    def is_supported(self, operation: str) -> bool:
        """Check if an operation is supported."""
        return operation in self._registry

    def get_supported_operations(self) -> List[str]:
        """Get list of all supported operations."""
        return list(self._registry.keys())

    def __len__(self) -> int:
        """Return number of supported operations."""
        return len(self._registry)

    def __contains__(self, operation: str) -> bool:
        """Check if operation is supported."""
        return operation in self._registry

    def __getitem__(self, operation: str) -> Callable:
        """Get operation function by name."""
        return self._registry[operation]

    def __iter__(self):
        """Iterate over operation names."""
        return iter(self._registry)

    def keys(self):
        """Get operation names."""
        return self._registry.keys()

    def values(self):
        """Get operation functions."""
        return self._registry.values()

    def items(self):
        """Get operation name-function pairs."""
        return self._registry.items()

    def register_operation(self, operation: str, func: Callable) -> None:
        """Register a new operation (for extensibility)."""
        self._registry[operation] = func
        logger.info(f"Registered new operation: {operation}")


def get_operation_registry() -> OperationRegistry:
    """Get the global operation registry instance."""
    return OperationRegistry()


def execute_operation(operation: str, inputs: List[torch.Tensor], kwargs: Optional[Dict] = None) -> torch.Tensor:
    """
    Convenience function to execute an operation.

    Args:
        operation: Operation name
        inputs: List of input tensors
        kwargs: Optional keyword arguments

    Returns:
        Result tensor
    """
    registry = get_operation_registry()
    return registry.execute(operation, inputs, kwargs)
