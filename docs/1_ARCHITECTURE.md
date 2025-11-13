# Djinn System Architecture

**Status**: Technical Reference Document  
**Version**: 1.0.0  
**Last Updated**: November 10, 2025  
**Target Audience**: System Architects, Core Developers, Platform Engineers

---

## Executive Summary

Djinn implements semantic-aware GPU disaggregation by intercepting PyTorch operations at the framework level. The system captures ML workload semantics (prefill/decode phases, KV cache patterns, compute intensity) to make intelligent scheduling decisions across distributed GPUs—all without requiring application code changes.

**Core Innovation**: Operating at the ML framework layer provides semantic visibility impossible at hardware/driver levels, enabling optimizations like KV cache co-location and intelligent operation routing.

**Architecture Philosophy**: 
- Transparency over performance (zero code changes)
- Semantic preservation through all layers
- Graceful degradation over hard failures
- Lazy evaluation for efficiency

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Abstractions](#2-core-abstractions)
3. [Four-Layer Architecture](#3-four-layer-architecture)
4. [Key Design Decisions](#4-key-design-decisions)
5. [Operation Routing & Classification](#5-operation-routing--classification)
6. [Data Flow & Execution Model](#6-data-flow--execution-model)
7. [Initialization & Startup](#7-initialization--startup)
8. [Network Protocol & Serialization](#8-network-protocol--serialization)
9. [Performance Optimizations](#9-performance-optimizations)
10. [Memory Management](#10-memory-management)
11. [Fault Tolerance](#11-fault-tolerance)
12. [Monitoring & Observability](#12-monitoring--observability)
13. [Extension Points](#13-extension-points)
14. [Future Architecture](#14-future-architecture)
15. [Limitations & Technical Debt](#15-limitations--technical-debt)

---

## 1. System Overview

### 1.1 Architectural Pattern

```
┌─────────────────────────────────────────────────────────┐
│                   User Application                       │
│                    (Unmodified PyTorch)                  │
└─────────────────────────────────────────────────────────┘
                              │
                    ┌─────────────────┐
                    │  Interception    │ ← Zero code changes
                    │  torch.Tensor    │
                    └─────────────────┘
                              │
┌─────────────────────────────────────────────────────────┐
│                    Four-Layer Stack                      │
│                                                          │
│  [Frontend]  → Capture & Enrich (LazyTensor + Semantic) │
│  [Scheduler] → Optimize Placement (Cost Model)           │
│  [Server]    → Coordinate Execution (Multi-tenant)       │
│  [Backend]   → Execute on GPUs (Runtime)                 │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Key Architectural Invariants

1. **Lazy Execution**: Djinn's version of Future/Promise. Operations build DAGs
2. **Semantic Flow**: Metadata preserved through all layers
3. **Fail-Safe Design**: Every remote operation has local fallback
4. **Thread Safety**: Thread-local state, immutable tensors
5. **Idempotency**: Operations can be safely retried


---

## 2. Core Abstractions

### 2.1 LazyTensor: The Foundation

LazyTensor is a `torch.Tensor` subclass that captures operations without executing them:

```python
class LazyTensor(torch.Tensor):
    """Symbolic tensor with deferred execution."""
    
    # Core operation DAG
    _operation: str              # "aten::matmul"
    _inputs: List[Any]           # Input tensors/scalars
    _kwargs: Dict[str, Any]      # Operation parameters
    
    # Lazy-computed properties
    _shape: Optional[torch.Size]  # None until accessed
    _dtype: Optional[torch.dtype] # Inferred or explicit
    _metadata: Optional[Dict]     # Semantic annotations
    
    # Device abstraction
    _logical_device: torch.device  # What PyTorch sees
    _physical_device: torch.device # Always 'meta' (zero memory)
```

**Key Properties:**
- **Zero Memory**: Uses meta device during capture
- **Thread-Safe**: Immutable after construction
- **Lazy Properties**: Shape/metadata computed only when needed
- **Dual Devices**: Logical (user-visible) vs physical (meta)

### 2.1.1 LazyTuple: Tuple Operations

For operations that return tuples (e.g., `split()`, `chunk()`, `topk()`), Djinn uses `LazyTuple` to preserve laziness:

```python
class LazyTuple(tuple):
    """Tuple of LazyTensors with deferred materialization."""
    
    # Contains LazyTensor elements (still lazy)
    # Materializes only when elements accessed
    # Behaves like Python tuple but preserves laziness
```

**Key Properties:**
- **Lazy Preservation**: Elements remain LazyTensors until accessed
- **Transparent**: Works like regular Python tuple
- **Efficient**: Only accessed chunks materialize, not entire tensor
- **Examples**: `split()`, `chunk()`, `unbind()`, `topk()`, `sort()`

**Usage:**
```python
chunks = x.split(300, dim=2)  # Returns LazyTuple (lazy)
a, b, c = chunks              # Unpacking works (still lazy)
result = a.cpu()              # Only chunk 0 materializes
```

### 2.2 Semantically Rich Graph (SRG)

The SRG annotates the LazyTensor DAG with ML semantics:

```python
@dataclass
class SemanticMetadata:
    # Execution characteristics
    execution_phase: ExecutionPhase      # prefill, decode, training
    data_residency: DataResidency        # ephemeral, persistent, stateful
    compute_intensity: float              # FLOPs per byte
    memory_pattern: MemoryPattern        # sequential, random
    
    # Optimization hints
    can_fuse: bool                       # Fusable with neighbors
    can_recompute: bool                  # Cheap to recompute
    must_colocate: Optional[NodeId]      # Pin to same device
    priority: int                        # Execution priority
```

**Semantic Dimensions:**

| Dimension | Values | Usage |
|-----------|--------|-------|
| **Execution Phase** | `llm_prefill`, `llm_decode`, `vision_encoding`, `training` | Memory allocation, device selection |
| **Data Residency** | `ephemeral_activation`, `persistent_weight`, `stateful_kv_cache` | Caching policy, eviction strategy |
| **Compute Intensity** | FLOPs/byte ratio | CPU vs GPU placement |
| **Memory Pattern** | `sequential`, `random`, `streaming` | Prefetch strategy |

### 2.3 ExecutionSchedule: The Plan

Output of the scheduler that bridges understanding and execution:

```python
@dataclass
class ExecutionSchedule:
    # Device placement
    device_assignments: Dict[NodeId, DeviceId]
    
    # Data movement plan
    transfers: List[Transfer]
    
    # Execution ordering
    execution_order: List[NodeId]  # Topologically sorted
    
    # Memory directives
    caching_policy: Dict[NodeId, CachePolicy]
    memory_budget: Dict[DeviceId, int]
    
    # Optimization flags
    enable_fusion: bool
    enable_compilation: bool
    enable_pipelining: bool
```

---

## 3. Four-Layer Architecture

### 3.1 Layer 1: Frontend (Intent Capture)

**Purpose**: Transparently intercept PyTorch operations and build semantically-enriched computation graphs.

#### 3.1.1 Component Structure

```
frontend/
├── core/                          # ~4,000 lines
│   ├── lazy_tensor.py             # 2,811 lines - Tensor subclass
│   ├── automatic_dispatch.py      # 350 lines - Meta tensor dispatch
│   ├── universal_dispatcher.py    # 400 lines - Operation routing
│   ├── factory_interceptor.py     # 246 lines - Creation functions
│   ├── shape_inference.py         # 350 lines - Shape computation
│   ├── operation_classifier.py    # 200 lines - Op categorization
│   └── interception_control.py    # 100 lines - Thread-local state
├── semantic/                      # ~1,000 lines
│   ├── analyzer.py                # Multi-tier analysis
│   ├── metadata_capture.py        # Lazy metadata
│   └── pattern_registry.py        # Pattern management
└── patterns/                      # ~500 lines
    ├── transformer_patterns.py    # Attention, layer norm
    └── cnn_patterns.py           # Convolution pipelines
```

#### 3.1.2 Interception Strategy

**Hybrid approach for maximum coverage:**

```python
# Coverage breakdown:
__torch_dispatch__: ~90%  # Primary: Most tensor operations
Factory wrapping: ~8%      # Secondary: tensor creation (randn, zeros)
__torch_function__: ~2%    # Fallback: Special cases
```

### 3.2 Layer 2: Scheduler (Optimization)

**Purpose**: Transform SRG into optimized execution plan using cost models and semantic understanding.

#### 3.2.1 Scheduling Algorithm

```python
def schedule(graph: SRG, strategy: SchedulingStrategy) -> ExecutionSchedule:
    # Phase 1: Semantic Analysis
    phases = detect_execution_phases(graph)
    patterns = recognize_patterns(graph)
    
    # Phase 2: Cost Estimation
    compute_costs = estimate_flops(graph.nodes)
    transfer_costs = estimate_bandwidth(graph.edges)
    
    # Phase 3: Optimization (strategy-dependent)
    if strategy == SchedulingStrategy.MINIMIZE_LATENCY:
        assignments = greedy_placement(graph, costs)
    elif strategy == SchedulingStrategy.MAXIMIZE_THROUGHPUT:
        assignments = bin_packing(graph, costs, memory_limits)
    
    # Phase 4: Execution Planning
    transfers = plan_minimal_transfers(assignments)
    order = topological_sort_with_priorities(graph)
    
    return ExecutionSchedule(assignments, transfers, order)
```

#### 3.2.2 Cost Model

```python
# Analytical cost model
total_cost = α * compute_cost + β * transfer_cost + γ * queuing_cost

where:
    compute_cost = Σ(FLOPs[op] / throughput[device])
    transfer_cost = Σ(bytes[edge] / bandwidth[link])
    queuing_cost = λ / (μ - λ)  # M/M/1 queue model
```

### 3.3 Layer 3: Server (Coordination)

**Purpose**: Manage distributed execution, multi-tenancy, and resource allocation.

#### 3.3.1 Execution Strategies

```python
class ExecutionStrategy(Enum):
    SUBGRAPH = "subgraph"      # Default: Single network RT
    COMPILED = "compiled"      # Large models: TorchScript/TensorRT
    STREAMING = "streaming"    # Future: Pipelined execution
    LOCAL = "local"           # Fallback: When remote unavailable
```

#### 3.3.2 Multi-Tenant Coordination

```python
class MultiTenantCoordinator:
    def schedule_request(self, request: Request) -> Response:
        # 1. Admission control
        if not self.check_quota(request.tenant_id):
            return reject_with_reason("quota_exceeded")
        
        # 2. Fair queuing (WFQ)
        priority = self.compute_priority(request)
        
        # 3. Device selection (load balancing)
        device = self.select_least_loaded_device(request.requirements)
        
        # 4. Resource isolation
        with ResourceLimits(memory=request.memory_limit):
            result = self.execute_isolated(request, device)
            
        return result
```

### 3.4 Layer 4: Backend (Execution)

**Purpose**: Low-level GPU management and network transport.

#### 3.4.1 Core Responsibilities

- GPU lifecycle (initialization, cleanup)
- Memory allocation primitives
- Network protocol implementation
- Tensor serialization/deserialization
- Error propagation

**Design Principle**: Backend is "dumb" - just executes what upper layers decide. No semantic understanding here.

---

## 4. Key Design Decisions

### 4.1 LazyTensor vs Alternative Approaches

| Approach | Pros | Cons | Why We Chose LazyTensor |
|----------|------|------|------------------------|
| **LazyTensor** | Works on 100% models, Fine-grained control | Per-op overhead | Universal compatibility critical |
| **torch.fx** | Native to PyTorch | Fails on 80% models | Dynamic control flow support needed |
| **torch.compile** | Fast execution | Requires decorators | Need semantic understanding |

### 4.2 Thread-Local vs Global State

**Decision**: Thread-local capture contexts

```python
# Each thread has independent state
_capture_context = threading.local()

# Benefits:
# - No lock contention (lock-free)
# - Natural isolation
# - Thread-safe by design
```

### 4.3 Lazy vs Eager Metadata

**Decision**: Lazy computation for all expensive metadata

```python
@property
def metadata(self):
    """Compute metadata only when accessed."""
    if self._metadata is None:
        self._metadata = get_metadata_capture().capture_metadata(...)
    return self._metadata
```

---

## 5. Operation Routing & Classification

### 5.1 Operation Classification System

Operations are classified into 5 categories:

```python
class OperationClass(Enum):
    MATERIALIZATION_TRIGGER = "materialization"  # Must execute now
    REDUCTION_OPERATION = "reduction"           # Prefer remote
    SHAPE_DEPENDENT = "shape_dependent"         # Need data
    TUPLE_RETURNING = "tuple_returning"         # Special handling
    COMPUTE_OPERATION = "compute"               # Standard deferred
```

**Classification Rules:**

| Category | Examples | Execution Strategy |
|----------|----------|-------------------|
| **MATERIALIZATION_TRIGGER** | `item()`, `all()`, `numpy()` | Immediate local execution |
| **REDUCTION_OPERATION** | `argmax()`, `sum(dim=)` | Remote execution preferred |
| **SHAPE_DEPENDENT** | `nonzero()`, `unique()` | Must materialize |
| **TUPLE_RETURNING** | `split()`, `chunk()`, `topk()`, `sort()` | Returns LazyTuple (lazy) |
| **COMPUTE_OPERATION** | `matmul()`, `add()` | Deferred execution |

### 5.2 Universal Dispatcher

Handles 95% of operations automatically:

```python
class UniversalDispatcher:
    def dispatch(self, func, args, kwargs, lazy_tensor_class):
        # 1. Materialize LazyTensor inputs
        # 2. Try PyTorch dispatch (covers 95%)
        # 3. Fall back to manual handlers (5%)
```

### 5.3 Materialization Triggers

Operations that require immediate execution:

```python
# Control flow triggers
if tensor.all():     # Must materialize to get bool
    do_something()

# Scalar extraction
value = tensor.item()  # Must materialize

# Data-dependent shapes
indices = tensor.nonzero()  # Shape depends on values

# Device transfers (materialize for transfer)
result = tensor.cpu()  # Materializes before transfer
```

**Note**: Tuple operations (`split()`, `chunk()`, etc.) return `LazyTuple` and remain lazy until elements are accessed. This enables efficient execution where only needed chunks are materialized.

---

## 6. Data Flow & Execution Model

### 6.1 Request Lifecycle

```
Phase                   Location        Key Operations
───────────────────────────────────────────────────────
1. Graph Capture       Client          LazyTensor creation
2. Subgraph Building   Client          Cache hit/miss
3. Serialization       Client          Numpy format
4. Network Transfer    Network         TCP
5. Deserialization     Server          Graph reconstruction
6. GPU Cache Lookup    Server          Hit/miss
7. Execution          Server/GPU       Model-dependent
8. Result Transfer    Network          Size-dependent
───────────────────────────────────────────────────────
```

### 6.2 Execution Flow

```python
def execute_request(lazy_tensor: LazyTensor) -> torch.Tensor:
    # 1. Trigger materialization
    if should_execute_locally(lazy_tensor):
        return execute_local_with_optimization(lazy_tensor)
    
    # 2. Build subgraph
    subgraph = SmartSubgraphBuilder().build(lazy_tensor)
    
    # 3. Check cache
    cache_key = compute_semantic_hash(subgraph)
    if cached := gpu_cache.get(cache_key):
        return cached
    
    # 4. Schedule
    schedule = scheduler.schedule(subgraph)
    
    # 5. Execute
    result = backend.execute(schedule)
    
    # 6. Cache result
    gpu_cache.put(cache_key, result)
    
    return result
```

---

## 7. Initialization & Startup

### 7.1 Lazy Initialization Strategy

Djinn initializes only when first used:

```python
# Trigger points for initialization
_initialization_triggers = [
    'torch.randn(..., device="remote_accelerator:0")',  
    'model.to("remote_accelerator:0")',
    'with djinn.capture(): ...',
    'first_lazytensor_operation',
]
```

### 7.2 Initialization Sequence

```python
def _ensure_async_init():
    """Lazy initialization on first Djinn operation."""
    if _runtime_state.initialized:
        return
        
    with _init_lock:
        if _runtime_state.initialized:
            return
            
        # Phase 1: GPU Discovery
        gpus = discover_gpus()
        
        # Phase 2: Server Connection
        servers = connect_to_servers(timeout=5)
        
        # Phase 3: Cache Warming (optional)
        if config.warm_cache_on_init:
            warm_common_operations()
        
        # Phase 4: Background Services
        start_memory_monitor()
        
        _runtime_state.initialized = True
```

### 7.3 Device Registration

```python
# Automatic device registration
RemoteAcceleratorSupport.initialize()

# Patches nn.Module.to() for seamless integration
```

---

## 8. Network Protocol & Serialization

### 8.1 TCP-Based Protocol

**Actual Implementation**: Simple struct-based protocol without magic bytes

```python
# Connection handling in server.py
async def _handle_connection(self, reader, writer):
    # Read transfer_id length
    transfer_id_len_bytes = await reader.readexactly(4)
    transfer_id_len = struct.unpack('>I', transfer_id_len_bytes)[0]
    
    # Read transfer_id
    transfer_id_bytes = await reader.readexactly(transfer_id_len)
    transfer_id = transfer_id_bytes.decode('utf-8')
    
    # Read metadata length
    metadata_len_bytes = await reader.readexactly(4)
    metadata_len = struct.unpack('>I', metadata_len_bytes)[0]
    
    # Read metadata
    metadata_bytes = await reader.readexactly(metadata_len)
    metadata = json.loads(metadata_bytes.decode('utf-8'))
    
    # Read tensor size
    size_bytes = await reader.readexactly(8)
    tensor_size = struct.unpack('>Q', size_bytes)[0]
    
    # Read tensor data
    tensor_bytes = await reader.readexactly(tensor_size)
```

**Protocol Structure**:
```
┌──────────────┬──────────────┬──────────┬─────────────┐
│ Transfer ID  │ Metadata     │ Size     │ Tensor Data │
│ (var length) │ (JSON)       │ (8 bytes)│ (variable)  │
└──────────────┴──────────────┴──────────┴─────────────┘
```

### 8.2 Serialization Format

```python
class TensorSerializer:
    @staticmethod
    def serialize(tensor: torch.Tensor) -> bytes:
        """Uses numpy format for efficiency"""
        buffer = io.BytesIO()
        np.save(buffer, tensor.detach().cpu().numpy())
        return buffer.getvalue()
    
    @staticmethod  
    def deserialize(data: bytes) -> torch.Tensor:
        """Auto-detect format for compatibility"""
        buffer = io.BytesIO(data)
        if data[:6] == b'\x93NUMPY':  # Numpy magic
            return torch.from_numpy(np.load(buffer))
        else:  # Assume torch format
            return torch.load(buffer)
```

**Serialization Performance**:
- Numpy format used for efficiency
- Backward compatibility with torch.save format
- Auto-detection of format type

### 8.3 Subgraph Protocol

```json
{
  "version": 1,
  "graph_id": "subgraph_abc123",
  "operations": [
    {
      "id": 0,
      "op": "aten::add",
      "inputs": [-1, -2],
      "kwargs": {"alpha": 1.0}
    }
  ],
  "input_tensors": {
    "-1": {"shape": [32, 784], "dtype": "float32"},
    "-2": {"shape": [32, 784], "dtype": "float32"}
  },
  "output_id": 0
}
```

---

## 9. Performance Optimizations

### 9.1 Critical Optimizations

#### 9.1.1 Reduction Operation Routing

**Strategy**: Smart routing based on reduction ratio:

```python
def should_execute_remotely(op: str, inputs: List) -> bool:
    if op not in REDUCTION_OPS:
        return False
        
    input_size = estimate_tensor_bytes(inputs[0])
    output_size = estimate_output_bytes(op, inputs)
    reduction_ratio = input_size / output_size
    
    # Execute remotely if significant reduction
    return reduction_ratio > 100 and input_size > 1_000_000
```

#### 9.1.2 Semantic Materialization Cache

**Cache by operation structure, not object identity:**

```python
class MaterializationCache:
    def compute_semantic_hash(self, lazy_tensor):
        """Hash based on operation semantics."""
        sig = f"{lazy_tensor.operation}"
        for inp in lazy_tensor.inputs:
            if isinstance(inp, LazyTensor):
                sig += f"|{self.compute_semantic_hash(inp)}"
            else:
                sig += f"|{type(inp).__name__}"
        return hashlib.sha256(sig.encode()).hexdigest()[:16]
```

#### 9.1.3 Materialization Optimizer

**Implementation**: `materialization_optimizer.py`

```python
class MaterializationOptimizer:
    """Optimizes LazyTensor materialization with:
    - Topological sort for batch execution
    - CUDA streams for pipelining
    - Pinned memory for transfers
    """
    
    def __init__(self, enable_pinned_memory=True, enable_streams=True):
        self.enable_pinned_memory = enable_pinned_memory
        self.enable_streams = enable_streams
        
    def execute_optimized(self, root_lazy_tensor, executor):
        # Build schedule
        schedule = self.build_schedule(root_lazy_tensor)
        
        # Execute with streams
        if self.enable_streams:
            return self._execute_with_streams(schedule, executor)
        else:
            return self._execute_sequential(schedule, executor)
```

**Optimizations**:
- Topological sort eliminates repeated graph traversal
- CUDA streams overlap compute and transfer
- Pinned memory enables faster CPU↔GPU transfers

### 9.2 Caching Hierarchy

| Cache Level | Hit Latency | Miss Penalty | Size Limit |
|------------|-------------|--------------|------------|
| Materialization Cache | <0.1ms | Recompute | 1000 entries |
| Graph Cache | ~1ms | Build time | 100 graphs |
| GPU Weight Cache | ~3ms | Transfer time | 4GB |

---

## 10. Memory Management

### 10.1 Memory Pressure Handling

**Implementation**: `memory_pressure_handler.py`

```python
class MemoryPressureHandler:
    THRESHOLDS = {
        'normal': 0.0,   # < 80% utilization
        'warning': 0.8,  # Start selective eviction  
        'critical': 0.95, # Aggressive eviction
    }
    
    def handle_pressure(self, utilization: float):
        if utilization > self.THRESHOLDS['critical']:
            # Emergency: Evict everything non-essential
            self.evict_all_ephemeral()
            self.disable_caching()
        elif utilization > self.THRESHOLDS['warning']:
            # Selective: Evict by priority
            self.evict_low_priority()
            self.reduce_cache_sizes()
```

**Features**:
- Proactive monitoring at configurable intervals
- Three-tier thresholds (normal/warning/critical)
- Semantic-aware eviction (protects KV cache during decode)
- Callback-based eviction system
- Memory statistics tracking

### 10.2 Memory Budgets

```python
# Phase-aware allocation
MEMORY_BUDGETS = {
    ExecutionPhase.LLM_PREFILL: {
        'activations': 0.6,  # 60% for activations
        'weights': 0.2,      # 20% for weights
        'kv_cache': 0.2      # 20% for KV cache
    },
    ExecutionPhase.LLM_DECODE: {
        'activations': 0.2,  # Less activations
        'weights': 0.2,      
        'kv_cache': 0.6      # More KV cache
    }
}
```

---

## 11. Fault Tolerance

### 11.1 Failure Modes & Recovery

| Failure Type | Detection | Recovery | Fallback |
|-------------|-----------|----------|----------|
| Network timeout | 30s timeout | Retry with backoff | Local execution |
| GPU OOM | CUDA error | Evict cache → Retry | Smaller batch |
| Shape inference fail | Exception/timeout | Skip inference | Materialization fallback |
| Remote node crash | Heartbeat loss | Failover to replica | Local execution |
| Serialization error | Exception | Skip optimization | Local execution |

### 11.2 Circuit Breaker Pattern

```python
class ShapeInferenceCircuitBreaker:
    def __init__(self, failure_threshold=10, timeout_ms=500):
        self.consecutive_failures = 0
        self.threshold = failure_threshold
        self.timeout = timeout_ms
        self.tripped = False
    
    def call(self, func, *args):
        if self.tripped:
            return None  # Fast fail
            
        try:
            with timeout(self.timeout):
                result = func(*args)
                self.consecutive_failures = 0
                return result
        except Exception:
            self.consecutive_failures += 1
            if self.consecutive_failures > self.threshold:
                self.tripped = True
            return None
```

---

## 12. Monitoring & Observability

### 12.1 Key Metrics

```python
# Capture metrics
djinn_capture_operations_total
djinn_capture_duration_seconds
djinn_shape_inference_failures_total

# Scheduling metrics
djinn_schedule_duration_seconds
djinn_schedule_strategy_used

# Execution metrics
djinn_execution_duration_seconds
djinn_execution_strategy_used
djinn_remote_execution_success_rate

# Cache metrics
djinn_cache_hits_total{cache="graph|gpu|materialization"}
djinn_cache_misses_total
djinn_cache_evictions_total

# Memory metrics
djinn_memory_usage_bytes{device="gpu0"}
djinn_memory_pressure_events_total{level="warning|critical"}
```

### 12.2 Debug Logging

```python
# Structured logging with levels
DJINN_LOG_LEVEL=DEBUG
DJINN_LOG_COMPONENT=interception,shape_inference,execution
```

---

## 13. Extension Points

### 13.1 Adding Custom Operations

```python
# Most operations work automatically via __torch_dispatch__

# For special operations, add to registry:
registry = OperationRegistry()
registry.register('custom_op', custom_implementation)

# Add shape rule if needed:
ShapeInference.SHAPE_RULES['custom_op'] = lambda x: compute_shape(x)
```

### 13.2 Custom Schedulers

```python
class LatencyOptimizedScheduler(SchedulerBase):
    """Custom scheduler for ultra-low latency."""
    
    def schedule(self, graph: SRG) -> ExecutionSchedule:
        critical_path = self.find_critical_path(graph)
        assignments = self.assign_to_fastest_gpu(critical_path)
        return ExecutionSchedule(assignments, ...)

# Register scheduler
scheduler_registry.register('low_latency', LatencyOptimizedScheduler)
```

---

## 14. Future Architecture

### 14.1 Global Scheduling Vision

```
Current (v1.0)                    Future (v2.0)
─────────────                     ─────────────
Client → Server → GPU             Global Scheduler
                                        ↓
                                 Fleet-wide view
                                        ↓
                                 Optimizations:
                                 • Cross-client batching
                                 • Dynamic GPU allocation
                                 • Preemption for SLA
                                 • Shared caching
```

### 14.2 Multi-Framework Support

```python
# Current: PyTorch only
frontend = PyTorchFrontend()

# Future: Framework abstraction
frontends = {
    'pytorch': PyTorchFrontend(),
    'jax': JAXFrontend(),
    'tensorflow': TensorFlowFrontend(),
}
```

### 14.3 Hardware Acceleration

```
Current                         Future
───────                         ──────
TCP transport                   RDMA/InfiniBand
Numpy serialization            Arrow/Flatbuffers
CPU scheduling                 GPU scheduling kernel
Software cache                 GPU L2 cache aware
```

---

## 15. Limitations & Technical Debt

### 15.1 Architectural Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| PyTorch coupling | Can't use with other frameworks | Framework abstraction layer (v2.0) |
| Thread-local state | Memory overhead, no sharing | Consider global state with fine locks |
| Shape inference brittleness | Circuit breaker trips | Invest in robust shape rules |
| Synchronous execution | No pipeline parallelism | Streaming execution (v2.0) |
| Static scheduling | Can't adapt at runtime | Runtime feedback loop |

### 15.2 Technical Debt

**High Priority:**
- Shape inference timeout (indicates fundamental issue)
- Circuit breaker pattern (band-aid for brittleness)
- Manual operation handlers (should be automatic)

**Medium Priority:**
- Serialization overhead (numpy not optimal for all cases)
- Cache invalidation strategy (currently manual)
- Memory fragmentation (no defragmentation)

**Low Priority:**
- Code duplication between executor paths
- Inconsistent error handling
- Missing comprehensive benchmarks

### 15.3 Known Issues

**Network Protocol**:
- Simplified protocol without magic bytes
- No version negotiation currently
- Limited error recovery

**Memory Management**:
- No defragmentation support
- Manual cache sizing
- Limited memory profiling

**Performance**:
- Shape inference can be slow (timeout guards in place)
- First execution has compilation overhead
- No predictive prefetching

---

## Summary

Djinn's architecture represents a pragmatic approach to GPU disaggregation that prioritizes:

1. **Transparency** - Zero application code changes through framework interception
2. **Semantic Awareness** - Understanding ML workload characteristics for optimization
3. **Fault Tolerance** - Graceful degradation over hard failures
4. **Production Readiness** - Real system with monitoring, caching, and error handling

The four-layer architecture cleanly separates concerns while maintaining semantic flow. Key components include:
- **LazyTensor**: Deferred execution with zero memory overhead
- **LazyTuple**: Lazy tuple operations for efficient chunked execution
- **Universal Dispatcher**: Automatic handling of 95% of operations
- **Memory Pressure Handler**: Proactive memory management with three-tier thresholds
- **Materialization Optimizer**: CUDA streams and pinned memory for efficiency

While limitations exist (framework coupling, shape inference brittleness, simplified network protocol), the architecture successfully demonstrates that framework-level interception can enable intelligent GPU sharing.

---

**For implementation details:**
- [Frontend Implementation](2_FRONTEND_IMPLEMENTATION.md) - Interception and capture
- [Scheduler Implementation](3_SCHEDULER_IMPLEMENTATION.md) - Optimization algorithms  
- [Backend Implementation](4_BACKEND_IMPLEMENTATION.md) - Execution runtime

**For deployment:**
- [Deployment Guide](deployment.md) - Production setup
- [Performance Tuning](performance.md) - Optimization guide
- [Troubleshooting](troubleshooting.md) - Common issues
```

---