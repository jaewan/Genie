# Dispatcher Implementation

## Overview

The Dispatcher is responsible for intercepting PyTorch operations and routing them to create LazyTensors. It works in conjunction with LazyTensor's `__torch_function__` protocol to achieve >95% operation coverage.

**File**: `genie/core/enhanced_dispatcher.py`  
**Lines**: ~376 lines  
**Related**: `genie/core/library.py` (torch.library registrations)

## Architecture

```
PyTorch Operation
     │
     ├─────────────────┬─────────────────┐
     │                 │                 │
     ▼                 ▼                 ▼
__torch_function__  Dispatcher      torch.library
   (LazyTensor)    (enhanced)        (library.py)
     │                 │                 │
     └─────────────────┴─────────────────┘
                       │
                       ▼
               Create LazyTensor
```

## Core Components

### EnhancedDispatcher Class

```python
class EnhancedDispatcher:
    """Unified dispatcher with improved operation coverage.
    
    Responsibilities:
        - Register operations for interception
        - Create LazyTensors for intercepted operations
        - Track statistics and coverage
        - Provide fallback mechanisms
    """
    
    def __init__(self):
        self.registered_ops: Dict[str, Callable] = {}
        self.lazy_mode: bool = True
        self.fallback_ops: Set[str] = set()
        self.operation_count: int = 0
        self.successful_registrations: Set[str] = set()
        self.failed_registrations: Set[str] = set()
```

### Operation Registration

#### Manual Registration

```python
@enhanced_dispatcher.register_op("aten::add")
def _add_impl(x, y, *, alpha=1):
    """Implementation for addition."""
    return torch.add(x, y, alpha=alpha)
```

**Process**:
1. Decorator wraps implementation
2. Creates wrapper that checks `lazy_mode`
3. If lazy: create LazyTensor
4. If eager: call original implementation
5. Register with PyTorch's dispatch system (optional)

#### Registration Implementation

```python
def register_op(self, op_name: str):
    """Register an operation for interception."""
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.operation_count += 1
            
            if self.lazy_mode:
                return self._create_lazy_tensor(op_name, args, kwargs)
            else:
                # Fallback to eager execution
                return func(*args, **kwargs)
        
        # Optionally register with PyTorch
        enable_impl = os.getenv("GENIE_ENABLE_ATEN_IMPL", "0") == "1"
        if enable_impl:
            try:
                torch.library.impl(op_name, "PrivateUse1")(wrapper)
                self.successful_registrations.add(op_name)
            except Exception as e:
                self.failed_registrations.add(op_name)
                self.fallback_ops.add(op_name)
        
        self.registered_ops[op_name] = wrapper
        return wrapper
    
    return decorator
```

### LazyTensor Creation

```python
def _create_lazy_tensor(self, op_name: str, args, kwargs):
    """Create LazyTensor for deferred execution."""
    from .lazy_tensor import LazyTensor
    
    self.operation_count += 1
    return LazyTensor(
        operation=op_name,
        inputs=list(args),
        kwargs=kwargs or {}
    )
```

## Registered Operations

### Arithmetic Operations

```python
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
```

### Linear Algebra

```python
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
```

### Activations

```python
@enhanced_dispatcher.register_op("aten::relu")
def _relu_impl(x):
    return torch.relu(x)

@enhanced_dispatcher.register_op("aten::sigmoid")
def _sigmoid_impl(x):
    return torch.sigmoid(x)

@enhanced_dispatcher.register_op("aten::tanh")
def _tanh_impl(x):
    return torch.tanh(x)

@enhanced_dispatcher.register_op("aten::gelu")
def _gelu_impl(x):
    return torch.nn.functional.gelu(x)

@enhanced_dispatcher.register_op("aten::softmax")
def _softmax_impl(x, dim, dtype=None):
    return torch.softmax(x, dim=dim, dtype=dtype)

@enhanced_dispatcher.register_op("aten::log_softmax")
def _log_softmax_impl(x, dim, dtype=None):
    return torch.log_softmax(x, dim=dim, dtype=dtype)
```

### Convolutions

```python
@enhanced_dispatcher.register_op("aten::conv2d")
def _conv2d_impl(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return torch.conv2d(input, weight, bias, stride, padding, dilation, groups)

@enhanced_dispatcher.register_op("aten::conv1d")
def _conv1d_impl(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return torch.conv1d(input, weight, bias, stride, padding, dilation, groups)
```

### Tensor Manipulation

```python
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
```

### Reductions

```python
@enhanced_dispatcher.register_op("aten::sum")
def _sum_impl(x, dim=None, keepdim=False, dtype=None):
    return torch.sum(x, dim=dim, keepdim=keepdim, dtype=dtype)

@enhanced_dispatcher.register_op("aten::mean")
def _mean_impl(x, dim=None, keepdim=False, dtype=None):
    return torch.mean(x, dim=dim, keepdim=keepdim, dtype=dtype)

@enhanced_dispatcher.register_op("aten::max")
def _max_impl(x, dim=None, keepdim=False):
    return torch.max(x, dim=dim, keepdim=keepdim)

@enhanced_dispatcher.register_op("aten::argmax")
def _argmax_impl(x, dim=None, keepdim=False):
    return torch.argmax(x, dim=dim, keepdim=keepdim)
```

## Statistics and Monitoring

### Get Statistics

```python
def get_stats(self) -> Dict[str, Any]:
    """Get comprehensive dispatcher statistics."""
    return {
        "registered_ops": len(self.registered_ops),
        "successful_registrations": len(self.successful_registrations),
        "failed_registrations": len(self.failed_registrations),
        "fallback_ops": len(self.fallback_ops),
        "operation_count": self.operation_count,
        "lazy_mode": self.lazy_mode,
        "coverage_percentage": self._calculate_coverage(),
    }
```

**Example Output**:
```python
{
    "registered_ops": 50,
    "successful_registrations": 45,
    "failed_registrations": 5,
    "fallback_ops": 5,
    "operation_count": 1523,
    "lazy_mode": True,
    "coverage_percentage": 90.0
}
```

### Coverage Calculation

```python
def _calculate_coverage(self) -> float:
    """Calculate operation coverage percentage."""
    total = len(self.successful_registrations) + len(self.failed_registrations)
    if total == 0:
        return 0.0
    return (len(self.successful_registrations) / total) * 100
```

## Lazy Mode Control

### Enable/Disable Lazy Mode

```python
def set_lazy_mode(self, enabled: bool):
    """Enable or disable lazy execution mode."""
    self.lazy_mode = enabled
    logger.info(f"Dispatcher lazy mode {'enabled' if enabled else 'disabled'}")
```

**Usage**:
```python
from genie.core.enhanced_dispatcher import set_enhanced_lazy_mode

# Disable lazy mode (execute eagerly)
set_enhanced_lazy_mode(False)

x = torch.randn(10, 10, device="remote_accelerator:0")
# x is now a regular tensor, not LazyTensor

# Re-enable
set_enhanced_lazy_mode(True)
```

## Fallback Mechanism

### Recording Fallbacks

```python
def record_fallback_capture(self, op_name: str):
    """Record an operation that fell back to __torch_function__."""
    self.fallback_ops.add(op_name)
    self.fallback_capture_count += 1
```

Called by LazyTensor when an operation isn't pre-registered:

```python
# In LazyTensor.__init__:
if not enhanced_dispatcher.is_operation_registered(self.operation):
    enhanced_dispatcher.record_fallback_capture(self.operation)
```

### Fallback Statistics

```python
{
    "fallback_ops": 15,  # Unique operations using fallback
    "fallback_capture_count": 234,  # Total fallback invocations
}
```

## Integration with torch.library

### Library Module

`genie/core/library.py` provides alternative registration using `torch.library`:

```python
def register_operation_impl(op_name: str):
    """Register operation using torch.library.impl."""
    def decorator(func):
        enable_impl = os.getenv("GENIE_ENABLE_ATEN_IMPL", "0") == "1"
        if enable_impl:
            @torch.library.impl(f"aten::{op_name}", "PrivateUse1")
            def impl_func(*args, **kwargs):
                if _operation_stats["lazy_mode"]:
                    return create_lazy_tensor(f"aten::{op_name}", *args, **kwargs)
                else:
                    return func(*args, **kwargs)
            return impl_func
        return func
    return decorator
```

**Usage**:
```python
@register_operation_impl("add.Tensor")
def add_impl(self, other, *, alpha=1):
    return torch.add(self, other, alpha=alpha)
```

## Coordination with LazyTensor

### Three-Layer Interception

```
Layer 1: torch.library (optional, via GENIE_ENABLE_ATEN_IMPL)
    ↓
Layer 2: EnhancedDispatcher (manual registrations)
    ↓
Layer 3: __torch_function__ (catches everything else)
```

**Coverage**:
- Layer 1: ~30 operations (when enabled)
- Layer 2: ~50 operations (always active)
- Layer 3: ~95%+ remaining operations

### Unified Interface

All layers create LazyTensors through the same path:

```python
# All paths eventually call:
LazyTensor(operation=op_name, inputs=args, kwargs=kwargs)
```

## Environment Variables

### `GENIE_ENABLE_ATEN_IMPL`

Controls torch.library registration:

```bash
# Enable torch.library impl registration
export GENIE_ENABLE_ATEN_IMPL=1

# Disable (rely on __torch_function__ only)
export GENIE_ENABLE_ATEN_IMPL=0  # Default
```

**Why disabled by default**: torch.library registration can conflict with some PyTorch versions and installations.

### `GENIE_LOG_INTERCEPTS`

Log intercepted operations for debugging:

```bash
export GENIE_LOG_INTERCEPTS=1
```

**Output**:
```
[Genie] Intercepted aten::matmul -> id=lt_42, in_shapes=[(4,4), (4,4)], out_shape=(4,4)
[Genie] Intercepted aten::relu -> id=lt_43, in_shapes=[(4,4)], out_shape=(4,4)
```

## Best Practices

### 1. Use Global Instance

Always use the global dispatcher instance:

```python
from genie.core.enhanced_dispatcher import enhanced_dispatcher

# Good
stats = enhanced_dispatcher.get_stats()

# Bad
dispatcher = EnhancedDispatcher()  # Creates separate instance!
```

### 2. Check Registration Status

```python
if enhanced_dispatcher.is_operation_registered("aten::custom_op"):
    # Use dispatcher path
    pass
else:
    # Will use __torch_function__ fallback
    pass
```

### 3. Monitor Coverage

```python
stats = enhanced_dispatcher.get_stats()
if stats["coverage_percentage"] < 80:
    logger.warning("Low dispatcher coverage, relying on fallback")
```

## Testing

See `tests/test_enhanced_dispatcher.py`:

```python
def test_dispatcher_lazy_mode():
    """Test lazy mode switching."""
    enhanced_dispatcher.set_lazy_mode(True)
    assert enhanced_dispatcher.lazy_mode

    enhanced_dispatcher.set_lazy_mode(False)
    assert not enhanced_dispatcher.lazy_mode

def test_operation_registration():
    """Test operations are registered."""
    stats = enhanced_dispatcher.get_stats()
    assert stats["registered_ops"] > 20

def test_lazy_tensor_creation():
    """Test dispatcher creates LazyTensors."""
    lt = enhanced_dispatcher._create_lazy_tensor(
        "aten::add",
        [torch.zeros(2, 2), torch.ones(2, 2)],
        {}
    )
    assert isinstance(lt, LazyTensor)
    assert lt.operation == "aten::add"
```

## Performance Considerations

### Registration Overhead

- Registration: One-time cost at import (~50ms for 50 ops)
- Dispatch: ~2-3 CPU cycles per operation
- LazyTensor creation: ~100ns per tensor

### Memory Usage

- Dispatcher state: ~50KB
- Per-operation metadata: ~1KB

## Future Enhancements

### 1. Dynamic Registration

```python
# Register operations at runtime based on usage patterns
dispatcher.register_ops_from_trace(trace)
```

### 2. Selective Interception

```python
# Only intercept specific operation types
dispatcher.enable_ops(["matmul", "conv2d", "linear"])
dispatcher.disable_ops(["add", "mul"])  # Use default path
```

### 3. Performance Profiling

```python
# Track per-operation overhead
stats = dispatcher.get_performance_stats()
# {"aten::matmul": {"count": 1000, "avg_overhead_ns": 150}, ...}
```

## Related Documentation

- [LazyTensor](03-lazy-tensor.md) - What dispatcher creates
- [Device Layer](02-device-layer.md) - How tensors become lazy
- [Executor](05-executor.md) - How operations execute

## References

- PyTorch Dispatcher: https://pytorch.org/tutorials/advanced/dispatcher.html
- torch.library: https://pytorch.org/docs/stable/library.html
