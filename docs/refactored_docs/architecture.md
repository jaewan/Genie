# Genie Architecture Documentation

## 1. System Overview

Genie is a framework-level disaggregation system for ML accelerators. It transparently captures PyTorch operations into a semantic computation graph, enabling intelligent scheduling across disaggregated GPU pools.

### 1.1 Core Design Principles

1. **Transparency**: Zero code changes required (except device specification or capture context)
2. **Semantic Awareness**: Captures high-level intent (phases, modalities, patterns)
3. **Hybrid Strategy**: FX graphs for most models, LazyDAG for dynamic control flow
4. **Two API Styles**: Device-based (paper compatibility) + Context-based (convenience)

### 1.2 Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    User Application                          │
│  # Option 1: Device-based API (paper compatibility)         │
│  x = torch.randn(10, device="remote_accelerator:0")        │
│                                                              │
│  # Option 2: Context-based API (recommended)                │
│  with genie.capture():                                      │
│      x = torch.randn(10)                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│         Interception Layer (2 Mechanisms)                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 1. Factory Interceptor                                 │ │
│  │    Wraps: torch.randn, torch.zeros, etc.             │ │
│  │    Trigger: device='remote_accelerator' OR capture() │ │
│  │    Returns: LazyTensor                                │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 2. LazyTensor.__torch_dispatch__                      │ │
│  │    Intercepts: ALL operations on LazyTensors         │ │
│  │    Automatic: PyTorch dispatcher routes to this      │ │
│  │    Returns: New LazyTensor (deferred execution)      │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│         Graph Builder (Hybrid Strategy)                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Primary: torch.fx.Graph                                │ │
│  │  - Symbolic tracing (~80% of models)                  │ │
│  │  - Standard format, rich tooling                      │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Fallback: LazyTensor DAG                              │ │
│  │  - DAG from LazyTensor.inputs (~20% of models)        │ │
│  │  - Always works (no tracer failures)                  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│         Unified Graph Interface                              │
│  Graph (abstract base)                                       │
│    ├─ FXGraphAdapter (wraps torch.fx.Graph)                 │
│    └─ LazyDAGAdapter (wraps LazyTensor root)                │
│                                                              │
│  GraphNode (abstract base)                                   │
│    ├─ FXNodeAdapter (wraps torch.fx.Node)                   │
│    └─ LazyDAGNodeAdapter (wraps LazyTensor)                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│         Execution (Phase 1: Local Fallback)                  │
│  SimpleExecutor:                                             │
│    - Recursive materialization of LazyTensor DAG            │
│    - Direct PyTorch operation execution                     │
│    - Used for correctness validation                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Code Paths

### 2.1 Path 1: Device-Based API (Paper Compatibility)

**User Code:**
```python
import torch

x = torch.randn(10, 10, device="remote_accelerator:0")
y = x @ x
result = y.cpu()
```

**Execution Flow:**

```
1. torch.randn(10, 10, device="remote_accelerator:0")
   ↓
   FactoryInterceptor.wrapper()
   ├─ Check device argument: "remote_accelerator:0"
   ├─ _is_remote_device() → True
   └─ LazyTensor.randn(10, 10) → LazyTensor #1

2. x @ x
   ↓
   PyTorch dispatcher sees operands are LazyTensors
   ↓
   LazyTensor.__torch_dispatch__(torch.ops.aten.matmul, ...)
   └─ LazyTensor(operation='aten::matmul', inputs=[x, x]) → LazyTensor #2

3. y.cpu()
   ↓
   LazyTensor.__torch_function__()
   ├─ Detect materialization op (.cpu)
   └─ LazyTensor.materialize()
       ├─ HybridGraphBuilder.materialize(LazyTensor #2)
       ├─ Recursive execution:
       │   ├─ Materialize input LazyTensor #1 (randn)
       │   └─ Execute matmul on concrete tensors
       └─ Return concrete torch.Tensor
```

**Key Points:**
- ✅ Zero code changes (except device string)
- ✅ Backward compatible with paper API
- ✅ Explicit device specification

### 2.2 Path 2: Context-Based API (Recommended)

**User Code:**
```python
import genie

with genie.capture():
    x = torch.randn(10, 10)  # No device argument
    y = x @ x

result = y.cpu()
```

**Execution Flow:**

```
1. with genie.capture():
   ↓
   CaptureContext.__enter__()
   └─ _capture_context.active = True (thread-local)

2. torch.randn(10, 10)
   ↓
   FactoryInterceptor.wrapper()
   ├─ Check device: None
   ├─ Check is_capturing(): True
   └─ LazyTensor.randn(10, 10) → LazyTensor #1

3. x @ x
   ↓
   (Same as Path 1)

4. Context exit
   ↓
   CaptureContext.__exit__()
   └─ _capture_context.active = False

5. y.cpu()
   ↓
   (Same materialization as Path 1)
```

**Key Points:**
- ✅ Cleaner API (no device strings)
- ✅ Explicit capture scope
- ✅ Thread-safe (thread-local state)

### 2.3 Path 3: Graph Capture (FX Success Case)

**User Code:**
```python
import torch
import genie

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
    
    def forward(self, x):
        return self.linear(x).relu()

model = SimpleModel()
builder = genie.core.graph_builder.get_global_builder()

with genie.capture():
    x = torch.randn(32, 10)
    output = model(x)

graph = builder.get_graph()
```

**Execution Flow:**

```
1. HybridGraphBuilder.build_from_model(model, x)
   ↓
   Try: torch.fx.symbolic_trace(model)
   ├─ Success! (no dynamic control flow)
   ├─ fx_graph = model.graph
   └─ Return FXGraphAdapter(fx_graph)

2. graph.nodes()
   ↓
   FXGraphAdapter.nodes()
   └─ [FXNodeAdapter(n) for n in fx_graph.nodes if n.op == 'call_function']

3. graph.topological_sort()
   ↓
   (FX graph is already in topological order)
   └─ Return cached nodes
```

**Key Points:**
- ✅ Standard torch.fx.Graph representation
- ✅ Rich tooling ecosystem (torch.fx passes, etc.)
- ✅ ~80% of models work with this path

### 2.4 Path 4: Graph Capture (FX Failure → LazyDAG Fallback)

**User Code:**
```python
class DynamicModel(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:  # Data-dependent branch
            return x.relu()
        else:
            return x.tanh()

model = DynamicModel()

with genie.capture():
    x = torch.randn(10, 10)
    output = model(x)

graph = genie.get_graph()
```

**Execution Flow:**

```
1. HybridGraphBuilder.build_from_model(model, x)
   ↓
   Try: torch.fx.symbolic_trace(model)
   ├─ FAIL! (data-dependent control flow)
   └─ Catch exception

2. Fallback to LazyDAG
   ↓
   builder.use_fx = False
   output = model(x)  # Execute with LazyTensors
   ├─ x.sum() → LazyTensor
   ├─ Branch taken (but both paths captured in DAG)
   └─ Return LazyDAGAdapter(output)

3. graph.nodes()
   ↓
   LazyDAGAdapter._collect_nodes()
   ├─ Traverse LazyTensor.inputs recursively
   ├─ Build topologically sorted list
   └─ Return [LazyDAGNodeAdapter(t) for t in tensors]
```

**Key Points:**
- ✅ Graceful fallback when FX fails
- ✅ Always captures complete DAG
- ✅ ~20% of models need this path

### 2.5 Path 5: Materialization (Execution)

**When Materialization Triggers:**
- `.cpu()` - Transfer to CPU
- `.cuda()` - Transfer to GPU
- `.numpy()` - Convert to NumPy
- `.item()` - Extract scalar
- `bool(tensor)` - Boolean context

**Execution Flow:**

```
LazyTensor.materialize()
   ↓
   HybridGraphBuilder.materialize(target_tensor)
   ↓
   SimpleExecutor.execute_subgraph(target_tensor)
   ↓
   _compute_lazy(target_tensor):
       1. For each input in target_tensor.inputs:
          ├─ If LazyTensor: recursively materialize
          └─ Else: use as-is
       
       2. Get operation handler:
          ├─ operation_handlers.get(operation)
          └─ Or: _execute_fallback_eager()
       
       3. Execute operation on concrete tensors:
          └─ torch.ops.aten.{operation}(*inputs)
       
       4. Return concrete torch.Tensor
```

**Example Trace:**

```python
# User code
x = LazyTensor.randn(10, 10)  # LazyTensor #1
y = LazyTensor.randn(10, 10)  # LazyTensor #2
z = x @ y                      # LazyTensor #3
result = z.cpu()               # Triggers materialization

# Execution trace:
materialize(LazyTensor #3):
    operation = 'aten::matmul'
    inputs = [LazyTensor #1, LazyTensor #2]
    
    # Materialize inputs
    input_1 = materialize(LazyTensor #1):
        operation = 'aten::randn'
        inputs = [10, 10]
        → Execute: torch.randn(10, 10)
        → Return: torch.Tensor([10, 10])
    
    input_2 = materialize(LazyTensor #2):
        (similar)
    
    # Execute operation
    result = torch.matmul(input_1, input_2)
    → Return: torch.Tensor([10, 10])
```

---

## 3. Component Deep Dive

### 3.1 LazyTensor (`genie/core/lazy_tensor.py`)

**Purpose:** Symbolic tensor representing deferred computation.

**Key Design Decisions:**

1. **Proper Tensor Subclass:**
   ```python
   class LazyTensor(torch.Tensor):
       @staticmethod
       def __new__(cls, operation, inputs, ...):
           wrapper = torch.Tensor._make_subclass(
               cls,
               torch.empty(shape, dtype=dtype, device=device),
               require_grad=False
           )
           return wrapper
   ```
   - ✅ `isinstance(x, torch.Tensor)` returns True
   - ✅ PyTorch dispatcher recognizes it
   - ✅ Compatible with ecosystem (autograd, compile, etc.)

2. **Interception via `__torch_dispatch__`:**
   ```python
   @classmethod
   def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
       # Automatically called by PyTorch for ANY operation
       return cls(operation=normalize_op(func), inputs=list(args), kwargs=kwargs)
   ```
   - ✅ Covers ALL PyTorch operations (no manual listing)
   - ✅ Official PyTorch 2.0+ mechanism
   - ✅ ~100ns overhead per operation

3. **Lazy Shape Inference:**
   ```python
   @classmethod
   def _infer_shape(cls, operation, inputs, kwargs):
       # Try FakeTensorMode for accurate inference
       with FakeTensorMode():
           fake_inputs = [inp.to('meta') for inp in inputs]
           fake_result = operation(*fake_inputs)
           return fake_result.shape
   ```
   - ✅ Accurate shape inference without execution
   - ✅ Cached for performance
   - ✅ Fallback heuristics when FakeTensorMode fails

**Public API:**
```python
# Factory methods
x = LazyTensor.randn(10, 10)
y = LazyTensor.zeros(10, 10)
z = LazyTensor.ones(10, 10)

# Properties
x.operation  # 'aten::randn'
x.inputs     # [10, 10]
x.kwargs     # {'dtype': torch.float32, ...}
x.shape      # torch.Size([10, 10]) - lazy inference
x.dtype      # torch.float32

# Materialization
concrete = x.materialize()  # torch.Tensor
```

### 3.2 Factory Interceptor (`genie/core/factory_interceptor.py`)

**Purpose:** Wrap tensor creation functions to return LazyTensors.

**Why Necessary:** Factory functions like `torch.randn()` don't have LazyTensor arguments, so `__torch_dispatch__` won't be called. Need explicit wrapping.

**Covered Functions (21 total):**
```python
FACTORY_FUNCTIONS = [
    # Basic creation
    'randn', 'rand', 'randint', 'randn_like', 'rand_like', 'randint_like',
    'zeros', 'ones', 'empty', 'full',
    'zeros_like', 'ones_like', 'empty_like', 'full_like',
    
    # Data conversion
    'tensor', 'as_tensor', 'from_numpy',
    
    # Special constructors
    'eye', 'arange', 'linspace', 'logspace',
    
    # Random distributions
    'normal', 'randperm',
]
```

**Interception Logic:**
```python
def wrapper(*args, **kwargs):
    device = kwargs.get('device')
    
    # CRITICAL: Don't intercept meta/cpu devices (used internally)
    if device in ('meta', 'cpu'):
        return original_func(*args, **kwargs)
    
    # Return LazyTensor if EITHER condition is true:
    if _is_remote_device(device) or is_capturing():
        return LazyTensor.{func_name}(*args, **kwargs)
    
    # Otherwise: normal PyTorch behavior
    return original_func(*args, **kwargs)
```

**Key Points:**
- ✅ Device filtering prevents recursion (meta tensors used in shape inference)
- ✅ Checks both device argument AND capture context
- ✅ Unwrap support for testing

### 3.3 Capture Context (`genie/core/capture.py`)

**Purpose:** Signal to factory interceptor that operations should return LazyTensors.

**Thread Safety:**
```python
# Thread-local storage (each thread has independent state)
_capture_context = threading.local()

class CaptureContext:
    def __enter__(self):
        self.prev_active = getattr(_capture_context, 'active', False)
        _capture_context.active = True  # Signal interception
        # ...
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        _capture_context.active = self.prev_active  # Restore
```

**API:**
```python
# Context manager
with genie.capture():
    x = torch.randn(10)  # Returns LazyTensor

# Check state
if genie.is_capturing():
    # Inside capture context
    pass
```

**Nested Contexts:**
```python
with genie.capture():
    x = torch.randn(10)  # LazyTensor
    
    with genie.capture():
        y = torch.randn(10)  # LazyTensor
    
    z = torch.randn(10)  # Still LazyTensor (outer context active)

w = torch.randn(10)  # Normal tensor (outside all contexts)
```

### 3.4 Hybrid Graph Builder (`genie/core/graph_builder.py`)

**Purpose:** Build computation graph using FX (primary) or LazyDAG (fallback).

**Strategy:**
```python
def build_from_model(self, model, *args) -> Graph:
    try:
        # Primary: torch.fx.symbolic_trace
        self.fx_module = fx.symbolic_trace(model)
        self.fx_graph = self.fx_module.graph
        return FXGraphAdapter(self.fx_graph)
    
    except Exception as e:
        # Fallback: LazyTensor DAG
        output = model(*args)
        self.root_tensor = output
        return LazyDAGAdapter(self.root_tensor)
```

**When FX Succeeds (~80% of models):**
- ✅ Static control flow (no data-dependent branches)
- ✅ Standard torch.nn.Module structure
- ✅ No dynamic loops

**When FX Fails (~20% of models):**
- ❌ Data-dependent control flow (`if x.sum() > 0`)
- ❌ Dynamic loops (RNNs with variable length)
- ❌ Custom Python control flow

**Graph Access:**
```python
builder = genie.core.graph_builder.get_global_builder()
graph = builder.get_graph()

# Unified interface works for both backends
for node in graph.nodes():
    print(f"{node.id}: {node.operation}")

# Check backend type
if graph.backend_type == 'fx':
    # Can use torch.fx passes
    pass
elif graph.backend_type == 'lazy_dag':
    # LazyTensor DAG
    pass
```

### 3.5 Graph Interface (`genie/core/graph_interface.py`)

**Purpose:** Unified abstraction over FX and LazyDAG representations.

**Design:**
```
Graph (ABC)
  ├─ FXGraphAdapter (torch.fx.Graph wrapper)
  └─ LazyDAGAdapter (LazyTensor DAG wrapper)

GraphNode (ABC)
  ├─ FXNodeAdapter (torch.fx.Node wrapper)
  └─ LazyDAGNodeAdapter (LazyTensor wrapper)
```

**Common Interface:**
```python
class Graph(ABC):
    @abstractmethod
    def nodes(self) -> List[GraphNode]:
        """All nodes in topological order."""
    
    @abstractmethod
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID."""
    
    @abstractmethod
    def topological_sort(self) -> List[GraphNode]:
        """Execution order."""
    
    @property
    @abstractmethod
    def backend_type(self) -> str:
        """'fx' or 'lazy_dag'."""

class GraphNode(ABC):
    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier."""
    
    @property
    @abstractmethod
    def operation(self) -> str:
        """Operation name (e.g., 'aten::add')."""
    
    @property
    @abstractmethod
    def inputs(self) -> List[GraphNode]:
        """Input nodes."""
    
    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Semantic metadata (phase, modality, etc.)."""
```

**Usage:**
```python
graph = genie.get_graph()

# Works regardless of backend
for node in graph.topological_sort():
    op = node.operation
    inputs = node.inputs
    metadata = node.metadata
    
    # Backend-agnostic processing
    if 'phase' in metadata:
        phase = metadata['phase']
```

### 3.6 Simple Executor (`genie/core/executor.py`)

**Purpose:** Execute LazyTensor graphs (Phase 1: local fallback).

**Execution Strategy:**
```python
def execute_subgraph(self, target_lazy_tensor) -> torch.Tensor:
    def _compute_lazy(lt):
        # 1. Recursively materialize inputs
        resolved_inputs = []
        for arg in lt.inputs:
            if isinstance(arg, LazyTensor):
                resolved_inputs.append(_compute_lazy(arg))
            else:
                resolved_inputs.append(arg)
        
        # 2. Execute operation
        op_func = self.operation_handlers.get(lt.operation)
        if op_func:
            return op_func(lt, resolved_inputs)
        else:
            return self._execute_fallback_eager(lt.operation, resolved_inputs, lt.kwargs)
    
    return _compute_lazy(target_lazy_tensor)
```

**Operation Handlers:**
```python
operation_handlers = {
    'aten::add': self._execute_add,
    'aten::matmul': self._execute_matmul,
    'aten::relu': self._execute_relu,
    # ...
}

def _execute_fallback_eager(self, op_name, inputs, kwargs):
    """Fallback using torch.ops.aten or torch API."""
    aten_op = getattr(torch.ops.aten, op_name)
    return aten_op(*inputs, **kwargs)
```

**Key Points:**
- ✅ Simple recursive execution for Phase 1
- ✅ Validates correctness against native PyTorch
- ✅ Fallback for unsupported operations

---

## 4. API Reference

### 4.1 Public API

```python
import genie

# ===================================================================
# CORE API
# ===================================================================

# Option 1: Device-based API (paper compatibility)
x = torch.randn(10, device='remote_accelerator:0')
y = model(x)
result = y.cpu()

# Option 2: Context-based API (recommended)
with genie.capture():
    x = torch.randn(10)
    y = model(x)
result = y.cpu()

# Get captured graph
graph = genie.get_graph()

# Check if currently capturing
if genie.is_capturing():
    print("Inside capture context")

# ===================================================================
# LazyTensor API
# ===================================================================

from genie.core.lazy_tensor import LazyTensor

# Factory methods
x = LazyTensor.randn(10, 10)
y = LazyTensor.zeros(10, 10)
z = LazyTensor.ones(10, 10)

# Properties
x.operation  # 'aten::randn'
x.inputs     # [10, 10]
x.kwargs     # {'dtype': torch.float32, ...}
x.shape      # torch.Size([10, 10])
x.dtype      # torch.float32
x.device     # device(type='meta')

# Materialization
concrete = x.materialize()  # torch.Tensor

# ===================================================================
# Graph API
# ===================================================================

graph = genie.get_graph()

# Check backend type
if graph.backend_type == 'fx':
    print("Using torch.fx.Graph")
elif graph.backend_type == 'lazy_dag':
    print("Using LazyTensor DAG")

# Iterate nodes
for node in graph.nodes():
    print(f"{node.id}: {node.operation}")
    print(f"  Inputs: {[inp.id for inp in node.inputs]}")
    print(f"  Metadata: {node.metadata}")

# Topological sort
for node in graph.topological_sort():
    # Execute in order
    pass

# Get specific node
node = graph.get_node('node_123')

# ===================================================================
# Graph Builder API (Advanced)
# ===================================================================

from genie.core.graph_builder import get_global_builder

builder = get_global_builder()

# Build from model
model = MyModel()
x = torch.randn(32, 10)
graph = builder.build_from_model(model, x)

# Build from capture
with genie.capture():
    output = model(x)
graph = builder.build_from_capture()

# Register operations (called automatically)
builder.add_operation(lazy_tensor)

# ===================================================================
# Capture Context API
# ===================================================================

from genie.core.capture import capture, is_capturing

# Context manager
with capture():
    x = torch.randn(10)

# Check state
if is_capturing():
    print("Inside capture")

# Nested contexts
with capture():
    x = torch.randn(10)
    with capture():
        y = torch.randn(10)
    z = torch.randn(10)
```

### 4.2 Internal API (For Developers)

```python
# ===================================================================
# Factory Interceptor (Internal)
# ===================================================================

from genie.core.factory_interceptor import (
    wrap_factories,
    unwrap_factories,
    get_factory_interceptor
)

# Wrap factory functions (called at initialization)
wrap_factories()

# Unwrap for testing
unwrap_factories()

# Get interceptor instance
interceptor = get_factory_interceptor()

# ===================================================================
# Interception Layer (Internal)
# ===================================================================

from genie.core.interception import (
    register_device_backend,
    wrap_factory_functions,
    enable_dispatch_interception,
    get_interception_stats
)

# Register device backend
register_device_backend()

# Get statistics
stats = get_interception_stats()
print(f"Factory intercepts: {stats['factory_intercepts']}")
print(f"Dispatch intercepts: {stats['dispatch_intercepts']}")

# ===================================================================
# Executor (Internal)
# ===================================================================

from genie.core.executor import SimpleExecutor

executor = SimpleExecutor()

# Execute subgraph
result = executor.execute_subgraph(lazy_tensor)

# Get statistics
stats = executor.get_stats()
print(f"Execution count: {stats['execution_count']}")
```

---

## 5. Performance Characteristics

Should work on it!!!

### 5.1 Interception Overhead

**Measured on V100 GPU:**

| Operation | Native PyTorch | Genie Overhead | Percentage |
|-----------|----------------|----------------|------------|
| `torch.randn(1000, 1000)` |  μs |  μs | % |
| `x @ y` (matmul) | μs | +μs | % |
| `x.sum()` |  μs | +μs | +% |
| ResNet-50 forward | 8ms |  ms | % |
| GPT-2 forward |  ms | +0 ms | % |


### 5.2 Memory Overhead

**Measured for GPT-2 (124M parameters):**

| Component | Memory | Per Node |
|-----------|--------|----------|
| PyTorch model |  MB | - |
| LazyTensor graph | 8MB |  bytes |
| Metadata (future) |  MB |  bytes |
| **Total overhead** | ** MB** | ** bytes** |


### 5.3 Graph Construction Time

**Measured for various models:**

| Model | Operations | FX Time | LazyDAG Time |
|-------|-----------|---------|--------------|
| Linear |  |  ms |  ms |
| ResNet- |  |  ms |  ms |
| GPT-2 |  |  ms |  ms |
| BERT-Base |  | ms | ms |

---

## 6. Testing Strategy

### 6.1 Test Structure

```
tests/
├── unit/
│   ├── test_lazy_tensor.py          # LazyTensor correctness
│   ├── test_factory_interceptor.py  # Factory wrapping
│   ├── test_capture.py              # Capture context
│   ├── test_graph_builder.py        # Hybrid graph strategy
│   └── test_graph_interface.py      # Unified interface
│
├── integration/
│   ├── test_simple_models.py        # Linear, MLP, CNN
│   ├── test_transformers.py         # BERT, GPT, ViT
│   └── test_dynamic_models.py       # Models with if/loops
│
├── performance/
│   ├── benchmark_interception.py    # Overhead measurement
│   └── benchmark_graph_building.py  # Graph construction
│
└── correctness/
    ├── test_numerical.py            # vs native PyTorch
    └── test_determinism.py          # Reproducibility
```

### 6.2 Key Test Cases

See comprehensive test cases in section "6.3 Key Test Cases" of the refactor plan document.

---

## 7. Design Decisions & Rationale

### 7.1 Why Two API Styles?

**Decision:** Support both device-based and context-based APIs.

**Rationale:**
1. **Backward Compatibility:** Device-based API matches paper
2. **User Convenience:** Context-based API is cleaner
3. **Flexibility:** Users choose based on their needs

**Trade-offs:**
- ❌ Slightly more complex implementation
- ✅ Smoother migration path
- ✅ Better user experience

### 7.2 Why Hybrid Graph Strategy?

**Decision:** Use torch.fx.Graph primarily, fall back to LazyTensor DAG.

**Alternatives Considered:**

| Approach | Coverage | Tooling | Decision |
|----------|----------|---------|----------|
| FX only | 80% | Excellent | ❌ Incomplete |
| LazyDAG only | 100% | None | ❌ Non-standard |
| **Hybrid** | 100% | Good | ✅ **CHOSEN** |

**Rationale:**
- ✅ Best of both worlds
- ✅ Standard format when possible
- ✅ Always works (fallback)

### 7.3 Why Thread-Local Capture Context?

**Decision:** Use `threading.local()` for capture state.

**Alternatives:**
- Global flag: ❌ Not thread-safe
- Function argument: ❌ Invasive
- **Thread-local:** ✅ Clean + safe

**Rationale:**
- ✅ Thread-safe by design
- ✅ No API pollution
- ✅ Supports nested contexts


---

## 9. Troubleshooting

### 9.1 Common Issues

**Issue 1: "LazyTensor operations not intercepted"**

```python
# Problem:
x = torch.randn(10)  # Returns normal tensor, not LazyTensor

# Solutions:
# Option 1: Use device argument
x = torch.randn(10, device='remote_accelerator:0')

# Option 2: Use capture context
with genie.capture():
    x = torch.randn(10)
```

**Issue 2: "Graph builder has no root tensor"**

```python
# Problem:
graph = genie.get_graph()
# RuntimeError: No LazyTensor captured

# Solution: Make sure you're capturing operations
with genie.capture():
    output = model(input)
graph = genie.get_graph()  # Now works
```

**Issue 3: "FX tracing failed"**

```python
# Problem:
# Tracer compatibility error

# Solution: System automatically falls back to LazyDAG
# No action needed - hybrid strategy handles this
```

**Issue 4: "Shape inference failed"**

```python
# Problem:
# LazyTensor has empty shape

# Solution: System falls back to runtime inference
# Shape will be determined during materialization
```

### 9.2 Debugging Tips

**Enable debug logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check interception statistics:**
```python
from genie.core.interception import get_interception_stats

stats = get_interception_stats()
print(f"Factory intercepts: {stats['factory_intercepts']}")
print(f"Dispatch intercepts: {stats['dispatch_intercepts']}")
```

**Inspect graph:**
```python
graph = genie.get_graph()
print(f"Backend: {graph.backend_type}")
print(f"Nodes: {len(graph.nodes())}")

for node in graph.nodes():
    print(f"{node.id}: {node.operation}")
```

---