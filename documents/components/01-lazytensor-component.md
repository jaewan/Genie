# Component: LazyTensor Engine (Fallback-First)

## Purpose
Core abstraction that captures PyTorch intent with minimal engineering by favoring a robust fallback path: build a semantically-rich graph lazily, and when execution is needed, materialize inputs and run eagerly on CPU. This ensures stability while we iterate on selective interception.

## Context
- **Upstream**: PyTorch (`__torch_function__` protocol, factory functions)
- **Downstream**: Semantic Analyzer, Optimization Engine
- **Interactions**: Pattern Library, Materialization Tracker, Graph Builder

## Key Requirements
- Prefer correctness and stability via eager CPU fallback
- Selectively intercept common ops (creation, arithmetic, matmul) for metadata richness
- <10µs overhead per intercepted op; near-zero overhead when bypassing
- <1% memory overhead for metadata
- Basic control-flow compatibility via implicit materialization triggers (Phase 1)

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

### 2. Operation Interception Strategy (Fallback-First)
- **Primary capture**: Factory hooks for `torch.randn/zeros/ones/empty/full/empty_strided` when `device == remote_accelerator` return `LazyTensor`
- **Lightweight wrappers**: Common Python operators on `LazyTensor` (e.g., `+`, `matmul`, `relu`) construct new `LazyTensor` nodes
- **Optional `__torch_function__`**: Retained for broad coverage when possible, but not relied upon for correctness
- **Normalization**: Normalize op names to canonical `aten::op` (strip overloads)
- **No hard dependency on dispatcher/library registrations**: Torch library/dispatcher integrations remain opt-in via env flags

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
- Coerce `device` kwargs to CPU for execution to avoid `PrivateUse1` allocations
- Eager CPU fallback via executor for non-intercepted ops; recursively materialize inputs
- Unsupported ops: log warning and return safe zeros-like fallback
- Env flags: `GENIE_LOG_INTERCEPTS=1` to log captured ops; `GENIE_ENABLE_ATEN_IMPL=1` to opt-in to dispatcher-style impls

## Implementation Checklist

### Phase 1: Fallback-First Core
- [x] Device bootstrap (non-strict)
- [x] Factory interception for creation ops on `remote_accelerator`
- [x] `LazyTensor` with minimal metadata
- [x] Graph construction and topological execution order
- [x] Executor with eager CPU fallback and device coercion
- [x] Logging flag `GENIE_LOG_INTERCEPTS`

### Phase 2: Selective Coverage and Semantics
- [ ] Expand selective wrappers (reductions, broadcasting edge cases)
- [ ] Control flow handling improvements (reduced eager triggers)
- [ ] Semantic enrichment (module_path, phase) via lightweight hooks
- [ ] Microbenchmarks for interception and materialization

### Phase 3: Optimization
- [ ] Memory-efficient metadata storage
- [ ] Incremental materialization
- [ ] Profiling and caching hooks

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
    z = x + y  # Uses lightweight operator wrapper
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
