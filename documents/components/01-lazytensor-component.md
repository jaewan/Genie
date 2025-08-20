# Component: LazyTensor Engine

## Purpose
Core abstraction that intercepts PyTorch operations and builds a semantically-rich computation graph through deferred execution.

## Context
- **Upstream**: PyTorch (`__torch_function__` protocol, factory functions)
- **Downstream**: Semantic Analyzer, Optimization Engine
- **Interactions**: Pattern Library, Materialization Tracker, Graph Builder

## Key Requirements
- Intercept >95% of PyTorch operations (via `__torch_function__`)
- <10µs overhead per operation
- <1% memory overhead for metadata
- Support autograd and control flow (Phase 2+)

## Core Implementation

### 1. PyTorch Device Registration
```python
class RemoteAcceleratorDevice(torch._C.Device):
    def __init__(self, device_id: int):
        self.device_type = "remote_accelerator"
        self.index = device_id
        self.dispatcher_key = torch._C.DispatchKey.PrivateUse1

# Register with PyTorch
torch._C._register_device("remote_accelerator", RemoteAcceleratorDevice)
```

### 2. Operation Interception Strategy
- **Primary**: `__torch_function__` on `LazyTensor` to intercept most tensor ops automatically
- **Factory hooks**: Patch `torch.randn/zeros/ones/empty/empty_strided` when `device==remote_accelerator` to return `LazyTensor`
- **Normalization**: Normalize op names to canonical `aten::op` form for graph consistency (strip overloads)
- **Dispatcher registration**: Optional for corner cases; not required for Phase 1

### 3. LazyTensor Data Structure
```python
class LazyTensor:
    def __init__(self, operation: str, inputs: List, kwargs: Dict):
        self.id = generate_uuid()
        self.operation = normalize_aten_name(operation)
        self.inputs = inputs
        self.kwargs = kwargs
        self.metadata = SemanticMetadata(
            operation_type=self.operation,
            tensor_shape=infer_shape(inputs, kwargs),
            dtype=infer_dtype(inputs, kwargs),
            device_hint="remote_accelerator:0",
        )
        self.materialized = False
        self.concrete_value = None

    # Materialization trigger examples
    def cpu(self):
        return self.materialize().cpu()

    def item(self):
        return self.materialize().item()
```

### 4. Graph Building
```python
class ComputationGraph:
    def add_operation(self, lazy_tensor: LazyTensor):
        node = ComputationNode(
            id=lazy_tensor.id,
            operation=lazy_tensor.operation,
            inputs=[self.get_or_create_node(inp) for inp in lazy_tensor.inputs],
            metadata=lazy_tensor.metadata,
        )
        self.nodes[node.id] = node
        self.update_edges(node)

    def get_execution_order(self) -> List[NodeId]:
        return topological_sort(self.nodes, self.edges)
```

### 5. Semantic Metadata Capture
```python
class SemanticMetadata:
    # Core metadata
    operation_type: str      # e.g., 'aten::matmul'
    tensor_shape: Tuple[int, ...]
    dtype: torch.dtype
    device_hint: str
    # Semantic enrichment (Phase 2+)
    module_path: Optional[str] = None
    execution_phase: Optional[str] = None
    workload_hints: Dict = field(default_factory=dict)
```

## Materialization Triggers

### Explicit Triggers
- `.item()` - Convert to Python scalar
- `.cpu()` - Move to CPU and materialize
- `.numpy()` - Convert to NumPy
- Print operations
- Explicit `.materialize()`

### Implicit Triggers
- Control flow dependencies
- Cross-device operations
- Non-intercepted operations
- Graph size limits

## Phase 1 Materialization Behavior
- Materialization coerces any `device` kwargs to CPU to avoid PrivateUse1 allocations during execution
- Fallback to eager PyTorch execution (CPU) in executor; inputs are materialized recursively
- Unsupported ops: log a warning and use eager execution path

## Implementation Checklist

### Phase 1: Basic Infrastructure
- [x] Device registration with PyTorch
- [x] `__torch_function__` interception + factory hooks
- [x] LazyTensor class with minimal metadata
- [x] Simple graph construction
- [x] Manual materialization (CPU fallback)

### Phase 2: Full Coverage
- [ ] 95% operation coverage with method/ufunc support
- [ ] Autograd support
- [ ] Control flow handling
- [ ] Semantic metadata collection (module_path, phase)
- [ ] Hook integration

### Phase 3: Optimization
- [ ] Memory-efficient metadata storage
- [ ] Graph compression
- [ ] Incremental materialization
- [ ] Performance profiling
- [ ] Cache integration

## Testing Requirements
```python
def test_lazy_tensor_creation():
    x = torch.randn(10, 10, device="remote_accelerator:0")
    assert isinstance(x, LazyTensor)
    assert x.metadata.operation_type == "aten::randn"
    assert tuple(x.metadata.tensor_shape) == (10, 10)

def test_operation_interception():
    x = torch.randn(10, 10, device="remote_accelerator:0")
    y = torch.randn(10, 10, device="remote_accelerator:0")
    z = torch.add(x, y)  # Should create new LazyTensor via __torch_function__
    assert isinstance(z, LazyTensor)
    assert z.operation == "aten::add"
    assert len(z.inputs) == 2

def test_materialization():
    x = torch.randn(10, 10, device="remote_accelerator:0")
    y = x.cpu()  # Should trigger materialization on CPU
    assert isinstance(y, torch.Tensor)
    assert y.device.type == "cpu"
```

## Performance Benchmarks
- Operation interception: <10µs per op
- Metadata storage: <100 bytes per tensor
- Graph construction: O(n) for n operations
- Materialization: <1ms for typical subgraphs (CPU fallback)

## Error Handling
```python
class LazyTensorError(Exception):
    pass

class MaterializationError(LazyTensorError):
    pass

class UnsupportedOperationError(LazyTensorError):
    pass

def handle_unsupported_op(op_name: str, *args, **kwargs):
    logger.warning(f"Unsupported operation {op_name}, falling back to eager")
    materialized_args = [materialize_if_lazy(arg) for arg in args]
    kwargs = coerce_device_to_cpu(kwargs)
    return getattr(torch.ops.aten, op_name)(*materialized_args, **kwargs)
```

## Integration Points
- **Semantic Analyzer**: Receives completed graph segments
- **Pattern Library**: Queries for pattern hints during construction
- **Optimization Engine**: Provides graph for optimization
- **Runtime**: Receives materialization requests (Phase 2+)

## Dependencies
- PyTorch 2.1.2 (`__torch_function__`, dispatcher)
- Python 3.10+
- NetworkX (graph algorithms)
- UUID (unique identifiers)
