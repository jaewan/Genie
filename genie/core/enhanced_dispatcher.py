"""
Enhanced dispatcher that improves upon the original while maintaining compatibility.

This approach focuses on practical improvements rather than complex torch.library integration
that may not work reliably across PyTorch versions.
"""
import functools
import logging
from typing import Any, Callable, Dict, Set, List
import torch

logger = logging.getLogger(__name__)


class EnhancedDispatcher:
    """
    Enhanced dispatcher with improved operation coverage and better error handling.
    
    This maintains the working approach from the original dispatcher while adding:
    - Better operation coverage
    - Improved error handling
    - Performance tracking
    - More structured registration
    """

    def __init__(self) -> None:
        self.registered_ops: Dict[str, Callable[..., Any]] = {}
        self.lazy_mode: bool = True
        self.fallback_ops: Set[str] = set()
        self.operation_count: int = 0
        self.successful_registrations: Set[str] = set()
        self.failed_registrations: Set[str] = set()

    def register_op(self, op_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Enhanced operation registration with better error handling."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any):
                self.operation_count += 1
                
                if self.lazy_mode:
                    return self._create_lazy_tensor(op_name, args, kwargs)
                else:
                    # Fallback to eager execution
                    return self._execute_eagerly(func, args, kwargs)

            try:
                # Try to register with PyTorch dispatcher
                # Use the working approach from the original implementation
                torch.library.impl(op_name, "PrivateUse1")(wrapper)
                self.registered_ops[op_name] = wrapper
                self.successful_registrations.add(op_name)
                logger.debug(f"Successfully registered operation: {op_name}")
            except Exception as e:
                logger.debug(f"PyTorch registration failed for {op_name}: {e}")
                # Store as fallback operation
                self.fallback_ops.add(op_name)
                self.failed_registrations.add(op_name)
                self.registered_ops[op_name] = wrapper
            
            return wrapper

        return decorator

    def _create_lazy_tensor(self, op_name: str, args: Any, kwargs: Any):
        """Create LazyTensor for deferred execution."""
        from .lazy_tensor import LazyTensor
        # Increment operation count when creating lazy tensors
        self.operation_count += 1
        return LazyTensor(operation=op_name, inputs=list(args), kwargs=kwargs or {})

    def _execute_eagerly(self, func: Callable[..., Any], args: Any, kwargs: Any) -> Any:
        """Fallback to eager execution."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Eager execution failed: {e}")
            raise

    def set_lazy_mode(self, enabled: bool) -> None:
        """Enable or disable lazy execution mode."""
        self.lazy_mode = enabled
        logger.info(f"Enhanced dispatcher lazy mode {'enabled' if enabled else 'disabled'}")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive dispatcher statistics."""
        return {
            "registered_ops": len(self.registered_ops),
            "successful_registrations": len(self.successful_registrations),
            "failed_registrations": len(self.failed_registrations),
            "fallback_ops": len(self.fallback_ops),
            "operation_count": self.operation_count,
            "lazy_mode": self.lazy_mode,
            "coverage_percentage": self._calculate_coverage()
        }

    def _calculate_coverage(self) -> float:
        """Calculate operation coverage percentage."""
        total_attempted = len(self.successful_registrations) + len(self.failed_registrations)
        if total_attempted == 0:
            return 0.0
        return (len(self.successful_registrations) / total_attempted) * 100

    def get_registered_operations(self) -> List[str]:
        """Get list of all registered operations."""
        return list(self.registered_ops.keys())

    def get_successful_operations(self) -> List[str]:
        """Get list of successfully registered operations."""
        return list(self.successful_registrations)

    def get_failed_operations(self) -> List[str]:
        """Get list of failed operation registrations."""
        return list(self.failed_registrations)


# Create enhanced dispatcher instance
enhanced_dispatcher = EnhancedDispatcher()


# Register comprehensive set of operations
# Core arithmetic operations
@enhanced_dispatcher.register_op("aten::add")
def _add_impl(x, y, *, alpha=1):
    return torch.add(x, y, alpha=alpha)

@enhanced_dispatcher.register_op("aten::sub")
def _sub_impl(x, y, *, alpha=1):
    return torch.sub(x, y, alpha=alpha)

@enhanced_dispatcher.register_op("aten::mul")
def _mul_impl(x, y):
    return torch.mul(x, y)

@enhanced_dispatcher.register_op("aten::div")
def _div_impl(x, y):
    return torch.div(x, y)

@enhanced_dispatcher.register_op("aten::pow")
def _pow_impl(x, y):
    return torch.pow(x, y)

# Linear algebra operations
@enhanced_dispatcher.register_op("aten::matmul")
def _matmul_impl(x, y):
    return torch.matmul(x, y)

@enhanced_dispatcher.register_op("aten::mm")
def _mm_impl(x, y):
    return torch.mm(x, y)

@enhanced_dispatcher.register_op("aten::bmm")
def _bmm_impl(x, y):
    return torch.bmm(x, y)

@enhanced_dispatcher.register_op("aten::addmm")
def _addmm_impl(bias, x, y, *, beta=1, alpha=1):
    return torch.addmm(bias, x, y, beta=beta, alpha=alpha)

@enhanced_dispatcher.register_op("aten::linear")
def _linear_impl(input, weight, bias=None):
    return torch.nn.functional.linear(input, weight, bias)

# Tensor creation operations
@enhanced_dispatcher.register_op("aten::randn")
def _randn_impl(*size, dtype=None, device=None, requires_grad=False):
    return torch.randn(*size, dtype=dtype, device=device, requires_grad=requires_grad)

@enhanced_dispatcher.register_op("aten::zeros")
def _zeros_impl(*size, dtype=None, device=None, requires_grad=False):
    return torch.zeros(*size, dtype=dtype, device=device, requires_grad=requires_grad)

@enhanced_dispatcher.register_op("aten::ones")
def _ones_impl(*size, dtype=None, device=None, requires_grad=False):
    return torch.ones(*size, dtype=dtype, device=device, requires_grad=requires_grad)

@enhanced_dispatcher.register_op("aten::empty")
def _empty_impl(*size, dtype=None, device=None, requires_grad=False):
    return torch.empty(*size, dtype=dtype, device=device, requires_grad=requires_grad)

@enhanced_dispatcher.register_op("aten::full")
def _full_impl(size, fill_value, dtype=None, device=None, requires_grad=False):
    return torch.full(size, fill_value, dtype=dtype, device=device, requires_grad=requires_grad)

# Activation functions
@enhanced_dispatcher.register_op("aten::relu")
def _relu_impl(x):
    return torch.relu(x)

@enhanced_dispatcher.register_op("aten::sigmoid")
def _sigmoid_impl(x):
    return torch.sigmoid(x)

@enhanced_dispatcher.register_op("aten::tanh")
def _tanh_impl(x):
    return torch.tanh(x)

@enhanced_dispatcher.register_op("aten::softmax")
def _softmax_impl(x, dim, dtype=None):
    return torch.softmax(x, dim=dim, dtype=dtype)

@enhanced_dispatcher.register_op("aten::log_softmax")
def _log_softmax_impl(x, dim, dtype=None):
    return torch.log_softmax(x, dim=dim, dtype=dtype)

@enhanced_dispatcher.register_op("aten::gelu")
def _gelu_impl(x):
    return torch.nn.functional.gelu(x)

@enhanced_dispatcher.register_op("aten::leaky_relu")
def _leaky_relu_impl(x, negative_slope=0.01):
    return torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)

# Convolution operations
@enhanced_dispatcher.register_op("aten::conv2d")
def _conv2d_impl(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return torch.conv2d(input, weight, bias, stride, padding, dilation, groups)

@enhanced_dispatcher.register_op("aten::conv1d")
def _conv1d_impl(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return torch.conv1d(input, weight, bias, stride, padding, dilation, groups)

# Normalization operations
@enhanced_dispatcher.register_op("aten::batch_norm")
def _batch_norm_impl(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled):
    return torch.batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled)

@enhanced_dispatcher.register_op("aten::layer_norm")
def _layer_norm_impl(input, normalized_shape, weight=None, bias=None, eps=1e-5, cudnn_enable=True):
    return torch.layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enable)

@enhanced_dispatcher.register_op("aten::dropout")
def _dropout_impl(input, p=0.5, training=True):
    return torch.dropout(input, p=p, train=training)

# Tensor manipulation operations
@enhanced_dispatcher.register_op("aten::view")
def _view_impl(x, size):
    return x.view(size)

@enhanced_dispatcher.register_op("aten::reshape")
def _reshape_impl(x, shape):
    return torch.reshape(x, shape)

@enhanced_dispatcher.register_op("aten::transpose")
def _transpose_impl(x, dim0, dim1):
    return torch.transpose(x, dim0, dim1)

@enhanced_dispatcher.register_op("aten::permute")
def _permute_impl(x, dims):
    return x.permute(dims)

@enhanced_dispatcher.register_op("aten::cat")
def _cat_impl(tensors, dim=0):
    return torch.cat(tensors, dim=dim)

@enhanced_dispatcher.register_op("aten::stack")
def _stack_impl(tensors, dim=0):
    return torch.stack(tensors, dim=dim)

@enhanced_dispatcher.register_op("aten::split")
def _split_impl(tensor, split_size_or_sections, dim=0):
    return torch.split(tensor, split_size_or_sections, dim=dim)

@enhanced_dispatcher.register_op("aten::squeeze")
def _squeeze_impl(x, dim=None):
    return torch.squeeze(x, dim=dim)

@enhanced_dispatcher.register_op("aten::unsqueeze")
def _unsqueeze_impl(x, dim):
    return torch.unsqueeze(x, dim)

# Reduction operations
@enhanced_dispatcher.register_op("aten::sum")
def _sum_impl(x, dim=None, keepdim=False, dtype=None):
    return torch.sum(x, dim=dim, keepdim=keepdim, dtype=dtype)

@enhanced_dispatcher.register_op("aten::mean")
def _mean_impl(x, dim=None, keepdim=False, dtype=None):
    return torch.mean(x, dim=dim, keepdim=keepdim, dtype=dtype)

@enhanced_dispatcher.register_op("aten::max")
def _max_impl(x, dim=None, keepdim=False):
    return torch.max(x, dim=dim, keepdim=keepdim)

@enhanced_dispatcher.register_op("aten::min")
def _min_impl(x, dim=None, keepdim=False):
    return torch.min(x, dim=dim, keepdim=keepdim)

# Comparison operations
@enhanced_dispatcher.register_op("aten::eq")
def _eq_impl(x, y):
    return torch.eq(x, y)

@enhanced_dispatcher.register_op("aten::ne")
def _ne_impl(x, y):
    return torch.ne(x, y)

@enhanced_dispatcher.register_op("aten::lt")
def _lt_impl(x, y):
    return torch.lt(x, y)

@enhanced_dispatcher.register_op("aten::gt")
def _gt_impl(x, y):
    return torch.gt(x, y)

@enhanced_dispatcher.register_op("aten::le")
def _le_impl(x, y):
    return torch.le(x, y)

@enhanced_dispatcher.register_op("aten::ge")
def _ge_impl(x, y):
    return torch.ge(x, y)


# Utility functions
def create_lazy_tensor_for_device_op(op_name: str, args: tuple, kwargs: dict):
    """Create LazyTensor for device operations."""
    # Use the enhanced dispatcher's method to ensure operation counting
    return enhanced_dispatcher._create_lazy_tensor(op_name, args, kwargs)


def set_enhanced_lazy_mode(enabled: bool) -> None:
    """Set lazy mode for enhanced dispatcher."""
    enhanced_dispatcher.set_lazy_mode(enabled)


def get_enhanced_stats() -> Dict[str, Any]:
    """Get enhanced dispatcher statistics."""
    return enhanced_dispatcher.get_stats()


# Log registration results
stats = enhanced_dispatcher.get_stats()
logger.info(f"Enhanced dispatcher loaded: {stats['successful_registrations']}/{stats['registered_ops']} operations registered successfully ({stats['coverage_percentage']:.1f}% coverage)")

if stats['failed_registrations'] > 0:
    logger.debug(f"Failed operations: {enhanced_dispatcher.get_failed_operations()}")
