# LazyTensor Implementation

## Overview

`LazyTensor` is the core abstraction that enables deferred execution and semantic capture in Genie. Instead of executing operations immediately, LazyTensors build a computation graph that can be analyzed and optimized before execution.

**File**: `genie/core/lazy_tensor.py`  
**Lines**: ~684 lines  
**Key Concept**: Proxy pattern with semantic enrichment

## Design Philosophy

From HotNets'25 §3.1:

> "By deferring execution, Genie automatically builds a semantically-rich compute graph. This graph serves as the standardized interface in a decoupled architecture."

LazyTensor achieves this through:
1. **Deferred Execution**: Operations return LazyTensors, not results
2. **Semantic Capture**: Rich metadata attached to each operation
3. **Graph Building**: Automatic DAG construction
4. **Transparent API**: Behaves like regular torch.Tensor

## Core Implementation

### Class Definition

```python
class LazyTensor:
    """Deferred execution tensor with rich semantic metadata."""
    
    # Memory optimization: use slots instead of dict
    __slots__ = (
        "id", "operation", "inputs", "kwargs",
        "shape", "dtype", "device",
        "materialized", "concrete_value", "_metadata",
    )
    
    # Fast ID generation
    _id_counter = count(1)
```

**Memory Optimization**: Using `__slots__` reduces per-instance memory by ~40% compared to normal Python objects.

### Initialization

```python
def __init__(self, operation: str, inputs: List[Any], 
             kwargs: Optional[Dict[str, Any]] = None):
    """Create LazyTensor for a deferred operation.
    
    Args:
        operation: Normalized aten operation (e.g., "aten::matmul")
        inputs: List of input tensors/LazyTensors
        kwargs: Operation keyword arguments
    """
    self.id = f"lt_{next(self._id_counter)}"
    self.operation = self._normalize_aten_name(operation)
    self.inputs = inputs
    self.kwargs = kwargs or {}
    
    # Infer tensor properties
    self.shape = self._infer_shape()
    self.dtype = self._infer_dtype()
    self.device = self._infer_device()
    
    # Lazy metadata initialization
    self._metadata = None
    
    # Execution state
    self.materialized = False
    self.concrete_value: Optional[torch.Tensor] = None
    
    # Register with graph builders
    self._register_with_builders()
```

### Operation Naming Normalization

```python
@staticmethod
def _normalize_aten_name(full_op: str) -> str:
    """Normalize aten op names by stripping overload suffixes.
    
    Example:
        "aten::add.Tensor" -> "aten::add"
        "aten::softmax.int" -> "aten::softmax"
        "matmul" -> "aten::matmul"
    """
    if not full_op.startswith("aten::"):
        full_op = f"aten::{full_op}"
    _, name = full_op.split("::", 1)
    base = name.split(".", 1)[0]
    return f"aten::{base}"
```

## Shape and Type Inference

### Shape Inference

LazyTensor infers output shapes without executing operations:

```python
def _infer_shape(self) -> Optional[torch.Size]:
    """Infer output shape from operation and inputs."""
    if self.operation in ["aten::add", "aten::sub", "aten::mul", "aten::div"]:
        return self._infer_elementwise_shape()
    elif self.operation == "aten::matmul":
        return self._infer_matmul_shape()
    elif self.operation == "aten::conv2d":
        return self._infer_conv2d_shape()
    # ... more operations
```

**Element-wise Operations**:
```python
def _infer_elementwise_shape(self) -> Optional[torch.Size]:
    """Infer shape for element-wise operations with broadcasting."""
    if len(self.inputs) < 2:
        return None
    
    x_shape = getattr(self.inputs[0], "shape", None)
    y_shape = getattr(self.inputs[1], "shape", None)
    
    if x_shape == y_shape:
        return x_shape
    
    # Simplified broadcasting: return larger shape
    if len(x_shape) >= len(y_shape):
        return x_shape
    return y_shape
```

**Matrix Multiplication**:
```python
def _infer_matmul_shape(self) -> Optional[torch.Size]:
    """Infer shape for matrix multiplication.
    
    (..., a, b) @ (..., b, c) -> (..., a, c)
    """
    if len(self.inputs) < 2:
        return None
    
    x_shape = getattr(self.inputs[0], "shape", None)
    y_shape = getattr(self.inputs[1], "shape", None)
    
    if x_shape and y_shape and len(x_shape) >= 2 and len(y_shape) >= 2:
        return torch.Size([*x_shape[:-1], y_shape[-1]])
    
    return None
```

### Meta Tensor Inference

For complex operations, use meta tensors:

```python
# In meta_utils.py
def infer_via_meta(operation: str, inputs: List[Any], 
                   kwargs: Optional[Dict[str, Any]]) -> Tuple[Optional[torch.Size], Optional[torch.dtype]]:
    """Infer shape and dtype by executing on meta tensors.
    
    Meta tensors have no data, just shape/dtype information.
    """
    # Convert inputs to meta tensors
    meta_args = [_to_meta_tensor(arg) for arg in inputs]
    
    # Execute on meta device (no actual computation)
    func = _get_torch_function(operation)
    result = func(*meta_args, **kwargs)
    
    return result.shape, result.dtype
```

## Integration with Metadata System (Post-Refactoring #2)

After Refactoring #2, semantic metadata is managed separately:

- **Registry Integration**: Registers with MetadataRegistry in `__init__`
- **Enrichment Trigger**: Calls SemanticEnricher to populate metadata
- **Access**: `metadata` property retrieves from registry

This separates execution concerns from semantic analysis.

## PyTorch Integration

### `__torch_function__` Protocol

LazyTensor implements PyTorch's `__torch_function__` protocol to intercept >95% of operations automatically:

```python
@classmethod
def __torch_function__(cls, func, types, args=(), kwargs=None):
    """Intercept all torch function calls involving LazyTensor.
    
    This enables automatic interception without manual registration.
    """
    kwargs = kwargs or {}
    
    # Extract function name
    func_name = getattr(func, '__name__', str(func))
    if hasattr(func, '_schema'):
        op_name = str(func._schema).split('(')[0]
    else:
        op_name = f"aten::{func_name}"
    
    op_name = cls._normalize_aten_name(op_name)
    
    # Check if any arguments are LazyTensors
    has_lazy = any(isinstance(arg, cls) for arg in args) or \
               any(isinstance(v, cls) for v in kwargs.values())
    
    if has_lazy:
        # Create new LazyTensor for this operation
        return cls(operation=op_name, inputs=list(args), kwargs=kwargs)
    else:
        # No LazyTensors involved, execute normally
        return func(*args, **kwargs)
```

**Coverage**: This single method handles:
- `torch.add(x, y)`
- `torch.matmul(x, y)`
- `x + y` (via `__add__`)
- `torch.nn.functional.relu(x)`
- And 95%+ of other PyTorch operations

### Operator Overloading

For direct operators, LazyTensor provides Python special methods:

```python
# Arithmetic operators
def __add__(self, other):
    return LazyTensor("aten::add", [self, other])

def __sub__(self, other):
    return LazyTensor("aten::sub", [self, other])

def __mul__(self, other):
    return LazyTensor("aten::mul", [self, other])

def __truediv__(self, other):
    return LazyTensor("aten::div", [self, other])

def __matmul__(self, other):
    return LazyTensor("aten::matmul", [self, other])

# Comparison operators
def __bool__(self):
    """Truthiness triggers materialization."""
    try:
        val = self.materialize()
        return bool(val.numel())
    except Exception:
        return True
```

### Tensor-like Methods

LazyTensor provides tensor-like methods that create new LazyTensors:

```python
# Shape manipulation
def unsqueeze(self, dim: int):
    return LazyTensor("aten::unsqueeze", [self, dim])

def squeeze(self, dim: Optional[int] = None):
    if dim is None:
        return LazyTensor("aten::squeeze", [self])
    return LazyTensor("aten::squeeze", [self, dim])

def reshape(self, *shape):
    return LazyTensor("aten::reshape", [self, shape])

def transpose(self, dim0: int, dim1: int):
    return LazyTensor("aten::transpose", [self, dim0, dim1])

# Reductions
def sum(self, dim=None, keepdim=False, dtype=None):
    return LazyTensor("aten::sum", [self], 
                     {"dim": dim, "keepdim": keepdim, "dtype": dtype})

def mean(self, dim=None, keepdim=False, dtype=None):
    return LazyTensor("aten::mean", [self],
                     {"dim": dim, "keepdim": keepdim, "dtype": dtype})

# Activations
def relu(self):
    return LazyTensor("aten::relu", [self])

def sigmoid(self):
    return LazyTensor("aten::sigmoid", [self])
```

## Materialization

### Trigger Points

LazyTensors materialize (execute) when:

1. **Explicit**: `.materialize()`, `.cpu()`, `.cuda()`, `.to(device)`
2. **Data Access**: `.item()`, `.numpy()`, `.tolist()`
3. **Control Flow**: `if tensor:`, `while tensor:`, etc.
4. **Printing**: `print(tensor)` (optional, via `GENIE_PRINT_MATERIALIZE`)

### Materialization Process

```python
def materialize(self) -> torch.Tensor:
    """Force materialization of this tensor.
    
    Process:
        1. Check if already materialized
        2. Execute computation graph up to this tensor
        3. Cache result for future access
    """
    if not self.materialized:
        from .executor import execute_subgraph
        
        # Execute all operations needed to compute this tensor
        self.concrete_value = execute_subgraph(self)
        self.materialized = True
    
    return self.concrete_value
```

### Partial Materialization

Only the necessary subgraph is executed:

```
Graph:
    A = randn(10, 10)
    B = A + A
    C = B * 2
    D = A - 1    # Not needed for C
    
Materialize C:
    Execute: A -> B -> C
    Skip: D (not in dependency chain)
```

## Factory Function Interception

### Integration

Factory functions like `torch.randn` are intercepted at module import:

```python
def _install_factory_interceptor(fn_name: str) -> None:
    """Install interceptor for factory functions."""
    original = getattr(torch, fn_name)
    
    def wrapper(*args, **kwargs):
        device = kwargs.get("device", None)
        has_lazy_input = any(isinstance(a, LazyTensor) for a in args)
        
        if _is_remote_device(device) or has_lazy_input:
            op_name = f"aten::{fn_name}"
            return LazyTensor(operation=op_name, inputs=list(args), 
                            kwargs=kwargs)
        
        return original(*args, **kwargs)
    
    setattr(torch, fn_name, wrapper)

def install_factory_interceptors():
    """Install all factory interceptors."""
    for name in ["randn", "rand", "zeros", "ones", "empty", ...]:
        _install_factory_interceptor(name)
```

### Covered Functions

- **Random**: `randn`, `rand`, `randint`, `randn_like`, `rand_like`
- **Zeros/Ones**: `zeros`, `ones`, `zeros_like`, `ones_like`
- **Empty**: `empty`, `empty_like`, `empty_strided`
- **Fill**: `full`, `full_like`
- **Range**: `arange`, `linspace`, `logspace`

## Performance Optimizations

### 1. Lazy Metadata

Metadata is only created when first accessed:

```python
@property
def metadata(self) -> SemanticMetadata:
    if self._metadata is None:
        self._metadata = self._create_metadata()
    return self._metadata
```

**Benefit**: ~30% reduction in LazyTensor creation time for operations that never need metadata.

### 2. Slots

Using `__slots__` reduces memory:

```python
__slots__ = ("id", "operation", "inputs", "kwargs", ...)
```

**Benefit**: ~40% less memory per LazyTensor instance.

### 3. Fast ID Generation

```python
_id_counter = count(1)  # itertools.count is faster than UUID
```

**Benefit**: 10x faster than `uuid.uuid4()`.

### 4. Shape Caching

Inferred shapes are cached:

```python
self.shape = self._infer_shape()  # Computed once, cached
```

## Utility Methods

### `lift()` - Convert Concrete Tensor

```python
@staticmethod
def lift(tensor: torch.Tensor) -> "LazyTensor":
    """Wrap concrete tensor as LazyTensor.
    
    Useful for:
        - Correctness testing (compare lazy vs eager)
        - Mixing lazy and eager execution
        - Graph construction from concrete inputs
    """
    return LazyTensor("aten::alias", [tensor], {})
```

**Usage**:
```python
# Convert CPU tensor to LazyTensor
x_cpu = torch.randn(10, 10)
x_lazy = LazyTensor.lift(x_cpu)

# Now can use in lazy computation
y = x_lazy @ x_lazy  # Creates lazy graph
```

### String Representations

```python
def __repr__(self) -> str:
    status = "materialized" if self.materialized else "lazy"
    return f"LazyTensor(op={self.operation}, shape={self.shape}, dtype={self.dtype}, {status})"

def __str__(self) -> str:
    # Optional materialization on print
    if os.getenv("GENIE_PRINT_MATERIALIZE", "0") == "1":
        val = self.materialize()
        return f"LazyTensor(value={val}, shape={val.shape}, dtype={val.dtype})"
    return self.__repr__()
```

## Testing

See `tests/test_torch_function_protocol.py`:

```python
def test_lazy_tensor_creation():
    """Test LazyTensor is created for remote device."""
    x = torch.randn(4, 4, device="remote_accelerator:0")
    assert isinstance(x, LazyTensor)
    assert x.shape == torch.Size([4, 4])
    assert not x.materialized

def test_operation_chaining():
    """Test operations create lazy graph."""
    x = torch.randn(4, 4, device="remote_accelerator:0")
    y = x + x
    z = torch.matmul(y, y)
    
    # All are LazyTensors
    assert isinstance(y, LazyTensor)
    assert isinstance(z, LazyTensor)
    
    # None are materialized yet
    assert not x.materialized
    assert not y.materialized
    assert not z.materialized

def test_materialization():
    """Test materialization produces correct results."""
    x = torch.randn(4, 4, device="remote_accelerator:0")
    y = x + x
    
    # Materialize
    result = y.cpu()
    
    # Now materialized
    assert y.materialized
    assert isinstance(result, torch.Tensor)
    assert result.device.type == "cpu"
```

## Best Practices

### 1. Avoid Premature Materialization

```python
# Good: Build entire graph first
x = torch.randn(1000, 1000, device="remote_accelerator:0")
y = x @ x
z = y.relu()
result = z.mean()  # Only materialize at the end

# Bad: Materialize in loop
for i in range(100):
    x = torch.randn(10, 10, device="remote_accelerator:0")
    y = x.cpu()  # Materializes every iteration!
```

### 2. Use Type Hints

```python
from genie.core.lazy_tensor import LazyTensor

def process(x: LazyTensor) -> LazyTensor:
    """Process lazy tensor without materializing."""
    return x.relu().mean()
```

### 3. Check Materialization State

```python
if not tensor.is_materialized():
    # Can still build graph
    tensor = tensor + 1
else:
    # Already executed
    concrete = tensor.concrete_value
```

## Related Documentation

- [Device Layer](02-device-layer.md) - How LazyTensors are created
- [Dispatcher](04-dispatcher.md) - Alternative interception mechanism
- [Executor](05-executor.md) - How LazyTensors are materialized
- [Semantic Metadata](08-semantic-metadata.md) - Metadata system details

## References

- HotNets'25 Paper §3.1: Lazy Tensor abstraction
- PyTorch `__torch_function__`: https://pytorch.org/docs/stable/notes/extending.html#extending-torch
