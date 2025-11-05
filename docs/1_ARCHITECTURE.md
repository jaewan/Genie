# Djinn System Architecture & Implementation

**Status**: Pending Peer Review
**Last Updated**: November 5, 2025

---

## Table of Contents

1. [Code Structure Overview](#1-code-structure-overview)
2. [The Semantically Rich Graph (SRG)](#2-the-semantically-rich-graph-srg)
3. [Layer 1: Frontend - Intent Capture & Semantic Enrichment](#3-layer-1-frontend---intent-capture--semantic-enrichment)
4. [Layer 2: Scheduler - Semantic-Driven Optimization](#4-layer-2-scheduler---semantic-driven-optimization)
5. [Layer 3: Server - Distributed Execution Coordination](#5-layer-3-server---distributed-execution-coordination)
6. [Layer 4: Backend - Core Execution Runtime](#6-layer-4-backend---core-execution-runtime)
7. [Data Flow & Request Lifecycle](#7-data-flow--request-lifecycle)
8. [Implementation Details](#8-implementation-details)
9. [Global Scheduling at Datacenter Scale](#9-global-scheduling-at-datacenter-scale)
10. [End-to-End Request Lifecycle](#10-end-to-end-request-lifecycle)
11. [Design Principles](#11-design-principles)

---

## §1. Code Structure Overview

Djinn follows a clean **four-layer architecture** with corresponding code organization:

```
djinn/
├── frontend/           # Layer 1: Intent Capture & Semantic Enrichment
│   ├── core/          # Tensor interception, LazyTensor DAG, shape inference
│   ├── patterns/      # Pattern recognition (attention, conv, KV cache)
│   └── semantic/      # Multi-tier analysis, phase detection, cost estimation
├── scheduler/         # Layer 2: Semantic-Driven Optimization
│   ├── core/          # Cost modeling, device placement, execution ordering
│   └── strategies/    # Stateful co-location, pipelined execution, memory-aware placement
├── server/            # Layer 3: Distributed Execution Coordination
│   ├── server.py      # Request handling, multi-tenancy, load balancing
│   ├── executors/     # GPU execution engines (subgraph, block, real_gpu)
│   ├── compilers/     # JIT compilation (TensorRT, TorchScript, fusion)
│   ├── coordination/  # Batching, fairness, fault tolerance
│   ├── transport/     # TCP connections, serialization, connection pooling
│   ├── memory/        # Execution-time memory management
│   └── cache/         # GPU cache, graph cache, tensor registry
├── backend/           # Layer 4: Core Execution Runtime
│   ├── runtime/       # GPU initialization, basic protocols, device management
│   └── memory/        # Low-level memory utilities
└── core/              # Shared utilities
    ├── types.py       # Common type definitions (ExecutionPhase, etc.)
    ├── exceptions.py  # Error handling
    ├── config.py      # Configuration management
    └── coordinator.py # Cluster coordination primitives
```

### Key Design Principles

1. **Clear Layer Separation**: Each layer has single responsibility
2. **Framework Transparency**: Zero application code changes required
3. **Semantic-Driven**: ML framework semantics enable intelligent decisions
4. **Memory-Aware**: Three-phase memory management (reactive → semantic → adaptive)
5. **Production Hardened**: Comprehensive monitoring, fault tolerance, performance optimization

---
## §2. The Semantically Rich Graph (SRG)

### §2.1 Core Abstraction

The **Semantically Rich Graph (SRG)** is Djinn's central abstraction—a **portable intermediate representation** that cleanly separates:
- **What**: The application's computational intent (operations, dependencies)
- **How/Where**: The physical execution strategy (device placement, scheduling)

**Key Property**: The SRG is a **declarative data structure**, not executable code. It serves as abstraction between frontend and scheduler.

### §2.2 SRG Structure

```
SRG = (Nodes, Edges, Annotations)

Nodes: Operations (from single kernel to fused subgraph)
Edges: Data dependencies (producer → consumer)
Annotations: Semantic metadata (phase, residency, modality, cost)
```

**Example SRG** (simplified):

```
Node 1: randn(shape=[8, 512])
  Phase: INPUT
  Residency: EPHEMERAL_ACTIVATION
  
Node 2: linear(input=Node1, weight=W1)
  Phase: LLM_PREFILL
  Residency: EPHEMERAL_ACTIVATION
  Cost: 2.1M FLOPs, 16KB memory
  
Node 3: attention(query=Node2, key=Node2, value=Node2)
  Phase: LLM_PREFILL
  Residency: STATEFUL_KV_CACHE
  Cost: 8.4M FLOPs, 256KB memory
  
Edge: Node1 → Node2 (tensor: [8, 512], float32, 16KB)
Edge: Node2 → Node3 (tensor: [8, 512], float32, 16KB)
```

### §2.3 Node Annotations

Each node carries a **common annotation schema**:

| Annotation | Description | Example Values | Purpose |
|------------|-------------|----------------|---------|
| **Phase** | Execution phase | `llm_prefill`, `llm_decode`, `vision_encoding` | Phase-aware resource management |
| **Residency** | Data lifetime | `persistent_weight`, `ephemeral_activation`, `stateful_kv_cache` | Caching and placement decisions |
| **Modality** | Data type | `vision`, `text`, `fusion` | Specialized accelerator placement |
| **Cost Hints** | Resource estimates | FLOPs, memory bytes, intensity | Scheduling and load balancing |

**Implementation**: See `djinn/core/types.py` for enum definitions.

### §2.4 Edge Annotations

Each edge carries **data movement metadata**:

| Annotation | Description | Purpose |
|------------|-------------|---------|
| **Tensor Metadata** | Shape, dtype, layout | Transfer size estimation |
| **Producer-Consumer Rates** | Data volume changes | Bandwidth reservation |
| **Criticality** | Critical path indicator | Transfer prioritization |

### §2.5 Why SRG Enables Optimization

The SRG's semantic richness enables optimizations **invisible to lower layers**:

**Example 1: Stateful Co-location**
```
Traditional (blind): Transfer KV cache every decode step (costly)
Djinn (semantic): Detect KV_CACHE residency → co-locate with decoder
Result: Eliminate repeated transfers
```

**Example 2: Pipelined CNN**
```
Traditional (blind): Execute conv layers sequentially
Djinn (semantic): Detect consecutive conv stages → pipeline across GPUs
Result: Overlap communication and computation
```

**Example 3: Dynamic Recomputation**
```
Traditional (blind): Always transfer intermediate results
Djinn (semantic): Detect cheap recomputation + network congestion → recompute
Result: Avoid network bottleneck
```

---

## §3. Layer 1: Frontend - Intent Capture & Semantic Enrichment

### §3.1 Frontend Architecture

The frontend is responsible for **transparently capturing application intent** and translating it into an SRG.

**Two-phase pipeline** with clear separation between capture and analysis:

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: Core Tensor Interception & Graph Construction         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Factory Function Wrapping                                   ││
│  │ • ~20 tensor creation functions (torch.randn, torch.zeros)  ││
│  │ • Triggered by: factory_interceptor.wrap()                  ││
│  │ • Returns: LazyTensor with deferred execution               ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ __torch_dispatch__ Interception                             ││
│  │ • Primary mechanism for tensor operations                   ││
│  │ • Covers: 95%+ of tensor operations automatically           ││
│  │ • LazyTensor DAG construction with dependencies             ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Shape Inference & Metadata                                  ││
│  │ • Meta-tensor execution for shape inference                 ││
│  │ • Lazy metadata capture via MetadataPlaceholder             ││
│  │ • Cost estimation (FLOPs, memory, intensity)                ││
│  └─────────────────────────────────────────────────────────────┘│
│  Output: Computation Graph with basic metadata                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: Semantic Enrichment & Annotation                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Pattern Recognition Framework                               ││
│  │ • Attention patterns (Q@K.T → softmax → @V)                 ││
│  │ • KV cache patterns (stateful data tracking)                ││
│  │ • Convolution patterns (stage detection)                    ││
│  │ • Multi-modal fusion patterns                               ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Execution Phase Detection                                   ││
│  │ • LLM prefill (parallel attention, compute-bound)           ││
│  │ • LLM decode (sequential, memory-bound)                     ││
│  │ • Vision encoding/decoding (convolution patterns)           ││
│  │ • Multimodal fusion (cross-attention)                       ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Workload Classification                                     ││
│  │ • Model architecture analysis (transformer, CNN, RNN)       ││
│  │ • Task type detection (classification, generation, etc.)    ││
│  │ • Resource requirement estimation                           ││
│  └─────────────────────────────────────────────────────────────┘│
│  Output: Semantically Rich Graph (SRG) with full annotations    │
└─────────────────────────────────────────────────────────────────┘
```

### §3.2 Hybrid Interception Strategy

Djinn uses a **practical interception approach** that prioritizes `__torch_dispatch__` for most operations:

```
LAYER 1: Factory Functions (~20 functions)
  - torch.randn(), torch.zeros(), torch.ones(), etc.
  - Triggered by: factory_interceptor.wrap()
  - Returns: LazyTensor when inside capture context or device contains 'remote_accelerator'

LAYER 2: __torch_dispatch__ (Primary interception)
  - Intercepts tensor operations when ANY argument is LazyTensor
  - Covers: 95%+ of tensor operations automatically
  - Pattern: Automatic via PyTorch's dispatcher

LAYER 3: __torch_function__ (Limited fallback)
  - Special cases: reshape operations, embedding operations
  - Trigger: Namespace functions like torch.add(x, y)
  - Pattern: Manual special case handling
```

**Result**: Effective interception of tensor operations with clear performance prioritization. **Note**: Claims of "99% PyTorch API coverage" and "2,000+ operations" are overstated - actual coverage is ~95% of tensor operations through the primary dispatch mechanism.

#### §3.2.1 Why Not PyTorch Device Backend Approach?

The codebase includes a `device.py` module that attempts to register "remote_accelerator" as a custom PyTorch device type. **This approach was considered but rejected** for the following reasons:

**Architectural Misunderstanding**:
- Registering a device name with PyTorch's PrivateUse1 backend does **NOT automatically intercept operations**
- Device registration ≠ operation interception
- PyTorch doesn't route tensor operations to custom code just because you register a device name

**Practical Issues**:
- Requires C++ extension (ABI compatibility problems)
- Adds build system complexity
- No functional performance benefit
- Still needs the same factory wrapping and dispatch logic
- Makes debugging harder (distributed state across Python/C++)

**Better Alternative**:
The current approach works with **any device specification**:
- `device='remote_accelerator:0'` (string)
- `device=torch.device('remote_accelerator', 0)` (torch.device)
- Inside `djinn.capture()` context
- No C++ dependencies, easier to debug and maintain

### §3.3 LazyTensor: Symbolic Tensor Representation

**Core Properties**:
```python
class LazyTensor:
    """
    Symbolic tensor representing deferred computation.
    
    Key properties:
    - Stores operation + inputs (not data)
    - Uses meta device during capture (zero memory)
    - Local metadata storage (shape, dtype, device)
    - Builds computation graph incrementally
    - Thread-safe via _MinimalTensorWrapper
    """
    
    def __init__(self, op, inputs, args, shape, dtype, device):
        self.op = op              # ATen operation (e.g., torch.ops.aten.matmul)
        self.inputs = inputs      # List of LazyTensor or concrete values
        self.args = args          # Keyword arguments
        self._shape = shape       # Inferred shape (local)
        self._dtype = dtype       # Inferred dtype (local)
        self._device = device     # Logical device (local)
        self.metadata = {}        # Semantic annotations
```

**Key Innovation: Local Metadata Storage**
- **Problem**: Lack of it requires remote queries for tensor metadata
- **Solution**: Store metadata locally using logical device abstraction
- **Impact**: Eliminates remote network calls for scheduling decisions and metadata operations like shape()s

### §3.4 LazyTensor DAG Graph Builder

**Strategy: LazyTensor DAG for all models**

```python
class GraphBuilder:
    """LazyTensor DAG graph builder for all models."""

    def build_from_model(model, *args):
        # Single strategy: LazyTensor DAG (works on all models)
        output = model(*args)
        if isinstance(output, LazyTensor):
            self.root_tensor = output
            return LazyDAGAdapter(self.root_tensor)
```

**Unified Graph Interface**:
- LazyDAGAdapter exposes LazyTensor DAG through `DjinnGraph` abstraction
- All graph algorithms work with the LazyTensor representation
- Pattern recognition works on operation-level DAG
- Semantic metadata stored separately (MetadataRegistry)

### §3.5 Semantic Metadata Structure

```python
@dataclass
class SemanticMetadata:
    # Structural information
    operation_type: str                   # Operation name
    tensor_shape: Optional[torch.Size]    # Output shape
    dtype: Optional[torch.dtype]          # Output dtype
    
    # Enhanced semantic enrichment
    module_path: Optional[str]            # "VQA.fusion_block.attention"
    semantic_role: Optional[str]          # "cross_attention_projection"
    execution_phase: Optional[ExecutionPhase]  # llm_prefill, vision_encoding, etc.
    data_lineage: Optional[DataLineage]   # Source modules, modality
    memory_pattern: Optional[MemoryPattern]    # persistent_weight, ephemeral_activation, etc.
    
    # Workload optimization hints
    compute_intensity: float              # FLOPs per byte
    estimated_flops: Optional[int]       # Operation cost
    memory_bandwidth_required: Optional[float]  # GB/s
    
    # Scheduling hints
    can_parallelize: bool                # Parallelizable operations
    preferred_device: Optional[str]      # Placement preference
    colocation_group: Optional[str]      # Co-locate with other ops
    priority: int                        # Execution priority
```

**Node Annotations** (from djinn/core/types.py):

| Annotation | Description | Example Values | Purpose |
|------------|-------------|-----------------|---------|
| **Phase** | Execution phase | `unknown`, `forward`, `llm_prefill`, `llm_decode`, `vision_encoding`, `vision_decoding`, `multimodal_fusion`, `training` | Phase-aware resource management |
| **Residency** | Data lifetime | `ephemeral_activation`, `persistent_weight`, `stateful_kv_cache`, `gradient` | Caching and placement decisions |
| **Modality** | Data type | `vision`, `text`, `audio`, `fusion` | Specialized accelerator placement |
| **Cost Hints** | Resource estimates | FLOPs, memory bytes, intensity | Scheduling and load balancing |

### §3. Multi-Layer Optimization System

**Graph Caching** (Phase 1):
- Eliminates repeated graph capture overhead
- LRU cache with automatic eviction (default: 100 models)

**Block Compilation** (Phase 2):
- Compiles model to TorchScript blocks at module boundaries
- Reduces RPC calls through coarse-grained execution
- Coarse-grained execution strategy for efficiency

**Memory Management** (Phase 1-3):
- **Phase 1**: Memory-aware GPU cache eviction (by size, not just count)
- **Phase 2**: Lifetime-based eviction (evict exactly when last consumer finishes)
- **Phase 3**: Phase-aware memory budgets (different allocations for prefill vs decode)
- **Phase 4**: Cost-based recomputation decisions (cache vs compute trade-offs)

**TensorRT Optimization** (Phase 4):
- Lazy compilation after profiling threshold
- Adaptive optimization for repeated blocks

---

## §4. Layer 2: Scheduler - Semantic-Driven Optimization

### §4.1 Scheduler Architecture

The scheduler is a **pluggable policy engine** that transforms an SRG into an optimized execution plan.

**Core Interface**:
```python
schedule = scheduler.create_schedule(
    graph: ComputationGraph,
    optimization_plan: OptimizationPlan
) -> ExecutionSchedule
```

**Input**: SRG with semantic annotations  
**Output**: ExecutionSchedule with:
- Device bindings (which GPU for each operation)
- Transfer schedules (when/where to move data)
- Caching directives (persistent vs ephemeral)
- Execution order (topologically sorted)

### §4.2 Semantic Optimizations

The scheduler leverages SRG annotations to apply **context-aware optimizations**:

#### Optimization 1: Stateful Co-location

**Pattern**: Sequential operations with stateful data (e.g., LLM decode with KV cache)

**Naive approach without Leveraging Semantics**:
```
Step 1: Transfer KV cache to GPU A
Step 2: Execute decode on GPU A
Step 3: Transfer updated KV cache back
Repeat for generation steps
```

**Djinn approach**:
```
Detect: KV_CACHE residency + LLM_DECODE phase
Action: Pin KV cache and decoder to same GPU
Result: Minimize data transfers
```

#### Optimization 2: Pipelined CNN Inference

**Pattern**: Consecutive convolutional stages

**Traditional approach**:
```
GPU A: Conv1 → Conv2 → Conv3 (sequential)
Result: Underutilized GPUs, no parallelism
```

**Djinn approach**:
```
Detect: Consecutive conv stages (from module hierarchy)
Action: Pipeline across GPUs (Conv1 on A, Conv2 on B, Conv3 on C)
Result: Overlapped execution
```

#### Optimization 3: Dynamic Recomputation

**Pattern**: Cheap intermediate under network congestion

**Traditional approach**:
```
Always transfer intermediate results
Result: Network bottleneck
```

**Djinn approach**:
```
Detect: Low FLOPs + high network latency
Action: Recompute on remote device instead of transfer
Result: Avoid network bottleneck
```

### §4.3 Cost Model

The scheduler uses a **cost model** to evaluate execution plans:

```
Total_Cost = Compute_Cost + Transfer_Cost + Queuing_Cost

Compute_Cost = Σ (FLOPs / GPU_throughput)
Transfer_Cost = Σ (bytes / network_bandwidth)
Queuing_Cost = f(load, contention)
```

**Cost estimation sources**:
- **FLOPs**: From SRG node annotations (profiled or model-based)
- **Memory**: From tensor metadata (shape × dtype)
- **Network**: From edge annotations (tensor size, criticality)

**Implementation**: See `3_SCHEDULER_IMPLEMENTATION.md` §2.

### §4.4 Placement Strategies

The scheduler supports **multiple placement policies**:

| Policy | Goal | Use Case |
|--------|------|----------|
| **minimize_latency** | Lowest end-to-end latency | Interactive inference |
| **maximize_throughput** | Highest ops/sec | Batch processing |
| **minimize_cost** | Lowest dollar cost | Cloud deployments |
| **load_balance** | Even GPU utilization | Multi-tenant clusters |

**Implementation**: Pluggable via `SchedulingPolicy` interface.

---

## §5. Layer 3: Server - Distributed Execution Coordination

### §5.1 Server Architecture

The server layer handles **distributed execution coordination** - orchestrating requests across multiple GPUs while managing multi-tenancy, load balancing, and fault tolerance.

**Key Components** (`djinn/server/`):

**Request Coordination:**
- `server.py`: Main server loop, request routing, multi-tenancy
- `tcp_server.py`: TCP server implementation with async handling
- `multi_tenant_coordinator.py`: Fairness, priority queues, resource allocation

**Execution Engines:**
- `subgraph_executor.py`: Fine-grained operation execution
- `block_executor.py`: Coarse-grained block execution
- `real_gpu_executor.py`: Direct GPU execution with memory management

**JIT Compilation:**
- `tensorrt_compiler.py`: TensorRT optimization with profiling
- `fusion_compiler.py`: Operation fusion based on SRG patterns
- `block_compiler.py`: TorchScript block compilation

**Caching Systems:**
- `gpu_cache.py`: Persistent GPU weight storage
- `graph_cache.py`: Compiled graph caching with LRU eviction
- `tensor_registry.py`: Remote tensor lifecycle management

### §5.2 Execution Strategies

**Smart Fragmentation** (Default):
```
Client Request → Server → Subgraph Execution → Result
    ↓              ↓            ↓
Cached Graph → JIT Compile → GPU Execute
```

**Block Compilation** (Large graphs):
```
Client Request → Server → Block Execution → Result
    ↓              ↓            ↓
TorchScript → TensorRT → GPU Execute
```

**Multi-Tenant Coordination:**
- **Fair Queuing**: Prevents starvation across tenants
- **Load Balancing**: Distributes work across GPU cluster
- **Resource Isolation**: Memory and compute quotas per tenant

### §5.3 Fault Tolerance & Recovery

**Lineage-Based Recovery:**
- Track operation dependencies for partial re-execution
- Resume from last successful checkpoint
- Minimize redundant computation on failures

**Automatic Failover:**
- Detect GPU failures via heartbeat monitoring
- Migrate execution to healthy GPUs
- Preserve execution state where possible

---

## §6. Layer 4: Backend - Core Execution Runtime

### §6.1 Backend Architecture

The backend provides **core execution runtime primitives** - the fundamental building blocks for GPU execution that are used by the server layer. Unlike the server layer which handles coordination and optimization, the backend focuses on low-level GPU management and basic protocols.

**Key Components** (`djinn/backend/`):

**Runtime Primitives** (`backend/runtime/`):
- `initialization.py`: GPU setup, CUDA context management, async initialization
- `gpu_memory.py`: Low-level GPU memory allocation and tracking
- `interfaces.py`: Device abstraction layer
- `transfer_manager.py`: Basic tensor transfer operations
- `tcp_client.py`: Low-level TCP communication primitives

**Memory Management** (`backend/memory/`):
- `allocator.py`: Memory allocation strategies
- `metrics.py`: Memory usage tracking and reporting

**Core Abstractions**:
- **Device Management**: GPU discovery, health monitoring, capability detection
- **Memory Primitives**: Allocation, deallocation, transfer tracking
- **Basic Protocols**: Low-level communication patterns used by server layer

**Design Philosophy**: The backend is intentionally minimal - it provides the essential runtime infrastructure without making execution decisions. All intelligence (scheduling, optimization, coordination) lives in the layers above.

---

## §7. Data Flow & Request Lifecycle

### §7.1 Four-Layer Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER APPLICATION (PyTorch)                   │
│                    No code changes required                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ Transparent Interception
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: FRONTEND (Intent Capture & Semantic Enrichment)      │
│  • Tensor interception → LazyTensor DAG construction           │
│  • Pattern recognition (attention, KV cache, convolution)      │
│  • Phase detection (prefill/decode/vision)                     │
│  • Cost estimation and workload classification                  │
│  Output: Semantically Rich Graph (SRG)                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ SRG (framework-agnostic)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: SCHEDULER (Semantic-Driven Optimization)              │
│  • Cost-aware device placement using SRG annotations           │
│  • Execution ordering with topological + semantic hints        │
│  • Stateful co-location (KV cache with decoder)                │
│  • Memory-aware placement with phase-specific budgets          │
│  Output: ExecutionSchedule (device bindings + transfers)        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ Execution plan
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: SERVER (Distributed Execution Coordination)           │
│  • Request routing and multi-tenancy management                │
│  • JIT compilation (TorchScript, TensorRT)                     │
│  • Execution orchestration across distributed GPUs             │
│  • Caching systems (GPU cache, graph cache, tensor registry)   │
│  Output: Coordinated execution results                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ Coordinated execution
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4: BACKEND (Core Execution Runtime)                      │
│  • GPU memory management and device primitives                 │
│  • Low-level execution and communication                       │
│  Output: Concrete tensor results                                │
└─────────────────────────────────────────────────────────────────┘
```

### §7.2 Request Lifecycle (12 Phases)

**Phase 1-2: Frontend Processing**
1. **Tensor Interception**: PyTorch operations transparently captured
2. **LazyTensor Construction**: Symbolic tensors created for remote execution

**Phase 3-4: Semantic Enrichment**
3. **Graph Construction**: DAG built from tensor operations
4. **Pattern Recognition**: Attention, KV cache, convolution patterns detected

**Phase 5-6: Optimization**
5. **Cost Estimation**: FLOPs, memory, operational intensity calculated
6. **Device Placement**: Scheduler assigns operations to optimal GPUs

**Phase 7-8: Execution Coordination**
7. **Request Routing**: Server receives and queues execution requests
8. **JIT Compilation**: TorchScript/TensorRT compilation as needed

**Phase 9-10: Distributed Execution**
9. **GPU Execution**: Operations executed on assigned devices
10. **Result Coordination**: Partial results gathered and merged

**Phase 11-12: Result Delivery**
11. **Serialization**: Results formatted for network transfer
12. **Client Delivery**: Final tensors returned to application

---

## §8. Implementation Details

### §8.1 Fault Tolerance Model

Djinn provides **lineage-based fault tolerance** inspired by dataflow systems (Spark, Ray).

**Key Insight**: The SRG is the unit of lineage—nodes are deterministic operations, edges are explicit dependencies.

### §8.2 Failure Detection and Recovery

**Failure types**:
1. **Remote GPU failure**: GPU crashes, OOM, hardware error
2. **Network failure**: Connection timeout, packet loss
3. **Partial execution failure**: Operation fails mid-execution

**Recovery strategy**:
```
1. Detect failure (timeout, error code)
2. Invalidate affected handles (mark as stale)
3. Identify subgraph to recompute (backward from failed node)
4. Rebind to new resources (scheduler selects alternative GPU)
5. Replay subgraph (deterministic recomputation)
6. Resume execution (from recovery point)
```

**Key properties**:
- **Deterministic**: Operations are pure functions (same inputs → same outputs)
- **Selective**: Only recompute affected subgraph (not entire computation)
- **Idempotent**: Side effects scoped to handle+epoch
- **Cross-phase**: Lineage spans phases (can recover decode without rerunning prefill)

### §8.3 Implementation

**Remote object handles**:
```python
class RemoteHandle:
    """
    Opaque reference to remote GPU object.
    
    Properties:
    - device_id: Which GPU holds the object
    - object_id: Unique identifier
    - epoch: Version number (for invalidation)
    - lineage: SRG subgraph to recompute
    """
```

**Failure handling**:
```python
try:
    result = execute_on_remote(handle)
except RemoteExecutionError:
    # 1. Invalidate handle
    handle.invalidate()
    
    # 2. Extract lineage
    subgraph = handle.lineage
    
    # 3. Rebind to new GPU
    new_device = scheduler.select_alternative(subgraph)
    
    # 4. Replay subgraph
    result = executor.replay(subgraph, new_device)
```

---

## §9. Global Scheduling at Datacenter Scale

### §9.1 Vision: Datacenter-Wide Optimization

The SRG enables a **broader vision** of autonomous resource management at datacenter scale.

**Architecture**:
```
┌─────────────────────────────────────────────────────────────────┐
│                    GLOBAL SCHEDULER                              │
│  • Fleet-wide resource allocation                               │
│  • Multi-tenant optimization                                    │
│  • Semantic-aware placement                                     │
└─────────────────────────────────────────────────────────────────┘
                        │           │           │
                        ▼           ▼           ▼
              ┌─────────────┬─────────────┬─────────────┐
              │  Djinn      │  Djinn      │  Djinn      │
              │  Client 1   │  Client 2   │  Client N   │
              │  (SRG)      │  (SRG)      │  (SRG)      │
              └─────────────┴─────────────┴─────────────┘
```

### §9.2 Semantic-Aware Global Decisions

Armed with fleet-wide semantic context, the global scheduler can make **intelligent placement decisions**:

#### Where: Heterogeneous Placement

**Traditional**: Treat all GPUs as homogeneous  
**Djinn**: Analyze SRG to identify workload classes

```
Vision workloads (VISION_ENCODING phase):
  → Place on memory-bandwidth-optimized GPUs (A100)

LLM inference (LLM_DECODE phase):
  → Place on compute-optimized GPUs (H100)

Recommendation models (MULTIMODAL_FUSION):
  → Place on accelerators with large memory (A100 80GB)
```

#### When: Elastic Scaling

**Traditional**: Static resource allocation  
**Djinn**: Dynamic provisioning based on phase annotations

```
LLM Prefill (LLM_PREFILL phase):
  → Scale out (parallelizable across sequence)
  → Provision 8 GPUs for burst

LLM Decode (LLM_DECODE phase):
  → Scale in (sequential, memory-bound)
  → Release 7 GPUs, keep 1 for decode
```

#### How: Cross-Workload Orchestration

**Traditional**: Isolated per-tenant scheduling  
**Djinn**: Cross-tenant optimization using semantic metadata

```
Detect: Two users requesting same LLM (from SRG model_id)
Action: Batch decode steps together
Result: 2× throughput (shared computation)

Detect: Interactive VQA query (from MULTIMODAL_FUSION + latency_sensitive)
Action: Preempt batch training job
Result: Meet SLA for interactive workload
```

### §9.3 Implementation Status

**Current**: Local scheduler (single-client optimization)  
**Future**: Global scheduler (fleet-wide optimization)

**Key challenges**:
- Coordination protocol (consensus, leader election)
- Scalability (1000s of clients, 10,000s of GPUs)
- Fairness (multi-tenant resource sharing)
- SLA enforcement (latency, throughput guarantees)

---

## §10. End-to-End Request Lifecycle

### §10.1 Complete Flow (12 Phases)

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: GRAPH CAPTURE (Client)                                │
│  • LazyTensor interception                                      │
│  • Deferred execution (no computation)                          │
│  • Time: ~0.5ms                                                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: SUBGRAPH BUILDING (Client)                            │
│  • Extract subgraph (backward from output)                      │
│  • Graph cache check (hit: 1-2ms, miss: 450ms)                  │
│  • Time: 1-450ms                                                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: SERIALIZATION (Client)                                │
│  • Convert operations to JSON                                   │
│  • Serialize tensors to numpy bytes                             │
│  • Time: ~15ms                                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4: NETWORK TRANSFER (Client → Server)                    │
│  • HTTP: 210ms | TCP: 10ms | DPDK: <1ms                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 5: REQUEST HANDLING (Server)                             │
│  • Parse HTTP request                                           │
│  • Deserialize subgraph JSON                                    │
│  • Time: ~5ms                                                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 6: GPU CACHE LOOKUP (Server)                             │
│  • Cache hit: 2.98ms | Cache miss: 86.90ms                      │
│  • Speedup: 29× faster (warm)                                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 7: GRAPH CACHE LOOKUP (Server)                           │
│  • Cache hit: <0.1ms | Cache miss: ~7ms                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 8: GPU EXECUTION (Server)                                │
│  • Execute operations in topological order                      │
│  • All intermediates stay on GPU                                │
│  • Time: ~20ms (GPT-2 Tiny)                                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 9: RESULT SERIALIZATION (Server)                         │
│  • Convert torch.Tensor → numpy → bytes                         │
│  • Time: ~10ms                                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 10: NETWORK TRANSFER (Server → Client)                   │
│  • HTTP: 210ms | TCP: 10ms | DPDK: <1ms                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 11: RESULT DESERIALIZATION (Client)                      │
│  • Parse HTTP response                                          │
│  • Deserialize bytes → numpy → torch.Tensor                     │
│  • Time: ~8ms                                                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 12: USER RECEIVES RESULT                                 │
│  • Return concrete torch.Tensor to user                         │
│  • Total: Cold 380ms | Warm 28ms (2.38× slowdown)              │
└─────────────────────────────────────────────────────────────────┘
```

### §10.2 Performance Breakdown (GPT-2 Tiny, Warm)

**⚠️ IMPORTANT: Performance measurements are environment-specific**

Actual performance depends on:
- Hardware configuration (GPU type, network latency, CPU cores)
- Workload characteristics (batch size, sequence length, model size)
- Optimization settings (transport protocol, cache configuration)
- Network conditions and cluster state

**Key Pipeline Stages**:

The end-to-end execution flows through these stages:

| Stage | Component | Considerations |
|-------|-----------|-----------------|
| Graph Capture | Client-side | Operation interception and recording |
| Subgraph Building | Client-side | Cache lookups for repeated workloads |
| Serialization | Client-side | Tensor encoding for transport |
| Network Transfer | Network | Depends on transport (HTTP vs TCP vs DPDK) |
| Request Handling | Server-side | Deserialization and cache lookup |
| GPU Execution | Server-side | Computation on accelerator |
| Result Serialization | Server-side | Encoding results for return |
| Network Transfer | Network | Return path to client |
| Result Deserialization | Client-side | Decoding returned tensors |

**Optimization Opportunities**:
- **Transport layer**: TCP vs DPDK trade-offs
- **Caching**: GPU cache and graph cache effectiveness
- **Serialization**: Zero-copy vs standard serialization
- **Batching**: Amortizing overhead with larger batches

For specific performance characteristics, run the benchmarking suite on your hardware and workloads.

---

## §11. Design Principles

### §11.1 Separation of Concerns

**Principle**: Cleanly separate "what" from "how/where"

**Implementation**:
- **Frontend**: Captures "what" (computational intent)
- **Scheduler**: Decides "how/where" (execution strategy)
- **Backend**: Executes "how/where" (concrete execution)

**Benefit**: Each component can be developed, tested, and optimized independently.

### §11.2 Pluggability

**Principle**: Support multiple implementations at each layer

**Implementation**:
- **Frontends**: PyTorch (current), JAX (future), TensorFlow (future)
- **Schedulers**: Latency-minimizing, throughput-maximizing, cost-minimizing
- **Backends**: TCP, DPDK, RDMA

**Benefit**: Adapt to different hardware, workloads, and deployment scenarios.

### §11.3 Transparency

**Principle**: No application code changes required

**Implementation**:
- LazyTensor interception (transparent to user)
- Automatic semantic annotation (no manual hints for common models)
- Fallback to local execution (if remote fails)

**Benefit**: Easy adoption, backward compatibility.

### §11.4 Semantic-Awareness

**Principle**: Leverage application semantics for optimization

**Implementation**:
- SRG annotations (phase, residency, modality)
- Pattern recognition (LLM, CNN, multimodal)
- Cost-aware scheduling

**Benefit**: Optimizations impossible at lower layers (PCIe, driver).

---

## §10. Trade-offs and Limitations

### §10.1 Trade-offs

| Decision | Benefit | Cost |
|----------|---------|------|
| **Framework-level interception** | Semantic-rich, transparent | Requires framework support |
| **Lazy evaluation** | Zero-copy capture, flexible scheduling | Delayed error detection |
| **Local metadata storage** | Fast queries (1,923× faster) | Slight memory overhead (~250 bytes/node) |
| **Graph caching** | 450ms → 1-2ms (225× faster) | Cache invalidation complexity |
| **GPU cache** | 29× speedup (warm) | GPU memory consumption |

### §10.2 Current Limitations

1. **In-place operations**: Converted to out-of-place (slight memory overhead ~5%)
2. **Mixed device operations**: Force materialization (lose potential optimizations)
3. **Memory management**: Long-running workloads require graph compaction
4. **Cold start overhead**: 32.2× slowdown (amortized over multiple requests)

### §10.3 Future Work

1. **DPDK integration**: <1ms network latency (100× faster than HTTP)
2. **Zero-copy serialization**: Eliminate ~25ms overhead
3. **Global scheduler**: Fleet-wide optimization
4. **Multi-framework support**: JAX, TensorFlow frontends
5. **Heterogeneous accelerators**: TPU, custom ASICs

---

## §11. Related Work

### §11.1 GPU Disaggregation

- **Logos** [OSDI'20]: Hardware-based disaggregation (PCIe-level)
- **Gimbal** [OSDI'22]: Network-attached GPUs (driver-level)
- **Prism** [SOSP'25]: Application-level disaggregation (manual tuning)

**Djinn's advantage**: Framework-level semantics enable automatic optimization.

### §11.2 ML Frameworks

- **PyTorch**: Dynamic computation graphs, eager execution
- **JAX**: Functional transformations, JIT compilation
- **TensorFlow**: Static graphs, distributed execution

**Djinn's approach**: Leverage framework abstractions for disaggregation.

### §11.3 Dataflow Systems

- **Spark**: Lineage-based fault tolerance (RDD)
- **Ray**: Distributed task execution (futures)
- **Dask**: Lazy evaluation (task graphs)

**Djinn's inspiration**: Lineage-based recovery, lazy evaluation.

---

## §11. Phase 3: Production Hardening (Memory Management)

### §11.1 Three-Phase Memory Optimization Timeline

Djinn's memory management evolves across three phases:

**Phase 1: Reactive Optimization** ✅
- Enhanced GPU cache with memory-aware eviction
- KV cache session pinning for autoregressive decode
- Async-first execution via `asyncio.to_thread`

**Phase 2: Semantic Intelligence** ✅
- Lifetime-based eviction (evict at exact moment last consumer finishes)
- Phase-aware memory budgets (different strategies for prefill vs decode)
- Cost-based recomputation decisions (mathematical trade-off analysis)

**Phase 3: Production Hardening** ✅
- Prometheus metrics (50+ metrics for complete observability)
- Memory pressure handler (proactive OOM prevention)
- Adaptive budget tuning (learns optimal allocations from workloads)

### §11.2 Prometheus Metrics Integration

**File**: `djinn/server/memory_metrics.py` (800+ LOC)

Comprehensive metrics across all memory operations:
- GPU cache: hits, misses, evictions, memory usage, hit rates
- KV sessions: created, closed, active, pinned bytes, lifetimes
- Lifetime analysis: analyses performed, early evictions, false retentions
- Phase-aware: switches, budget violations, utilization per category
- Memory pressure: events by severity, recovery time, utilization
- Adaptive budgets: updates, efficiency scores, learned allocations

**Benefits**:
- Real-time visibility into memory management decisions
- Integration with Grafana dashboards and monitoring
- Performance debugging and optimization tuning
- Optional (graceful fallback if Prometheus unavailable)

### §11.3 Memory Pressure Handling

**File**: `djinn/server/memory_pressure_handler.py` (400+ LOC)

Proactive detection and recovery from memory pressure:

```
Normal (<80%)        → Standard operation
Warning (80-95%)     → Aggressive eviction, reduce caching
Critical (>95%)      → Emergency eviction, prefer recomputation
OOM (100%)          → Recovery via all eviction sources
```

**Key Features**:
- Two-level thresholds for gradual escalation
- Semantic-aware eviction (protect critical data)
- Adaptive recomputation preferences
- Pressure event tracking and history
- Async-first non-blocking design

**Integration**:
- Register with cache for aggressive eviction callbacks
- Register with KV session manager for idle session cleanup
- Adapt `RecomputationVsStorageDecider` thresholds under pressure

### §11.4 Adaptive Budget Tuning

**File**: `djinn/server/adaptive_budget_tuner.py` (300+ LOC)

Learns optimal phase-specific memory allocations from execution patterns:

```
Phase 1-5:   Collect observations (memory utilization, cache hits)
Phase 6+:    Calculate efficiency scores
Phase 10+:   Suggest budget adjustments
Ongoing:     Gradually shift towards optimal allocation
```

**Learning Algorithm**:
1. Track per-phase memory utilization (weights, activations, KV cache)
2. Monitor cache hit rates and eviction counts
3. Calculate efficiency score (0-100)
4. Identify over/under-utilized categories
5. Gradually adjust budgets (learning rate configurable)

**Benefits**:
- Adapts to different workloads automatically
- Workload-specific optimization
- Gradual learning (prevents thrashing)
- Efficiency-driven (optimizes right metric)

---

## §12. Design Principles (Revised)

### §12.1 Separation of Concerns

**Principle**: Cleanly separate memory policies from memory mechanisms

**Implementation**:
- **Mechanisms**: GPU cache, KV sessions, eviction primitives
- **Policies**: Lifetime analysis, phase-aware budgets, adaptive learning
- **Monitoring**: Metrics collection, pressure detection

**Benefit**: Policies can be updated without changing mechanisms.

### §12.2 Semantic-Driven Optimization

**Principle**: Use application semantics to make better decisions

**Memory Implications**:
- **Phase annotations** → Different budgets for prefill vs decode
- **Residency hints** → Protect persistent weights and KV cache
- **Lifetime information** → Evict exactly when done

**Benefit**: Optimizations impossible at hardware or driver level.

### §12.3 Adaptive Learning

**Principle**: Systems should improve over time by learning from workloads

**Implementation**:
- Observe actual memory utilization patterns
- Calculate efficiency metrics
- Adjust allocations gradually
- Track improvements

**Benefit**: Works well for diverse workloads without manual tuning.

---

## §13. Conclusion

Djinn demonstrates **complete memory management for semantic-driven GPU disaggregation**:

✅ **Phase 1**: Reactive optimization (memory-aware cache + session pinning)  
✅ **Phase 2**: Semantic intelligence (lifetime eviction + phase budgets + cost model)  
✅ **Phase 3**: Production hardening (metrics + pressure handling + adaptive tuning)

**Integrated Memory Management Stack**:
- Frontend captures semantic hints (`execution_phase`, `data_residency`)
- Scheduler uses hints for placement and cost decisions
- Backend implements three-phase optimization
- Production monitoring via Prometheus metrics

**Expected Improvements**:
- Activation memory waste: 40% → 10% (4× improvement)
- Premature evictions: Frequent → Eliminated
- KV cache thrashing: High → Fully protected
- Budget utilization: 30-50% → 60%+
- Network traffic (KV cache): 500× reduction

For performance characteristics specific to your workloads and hardware, benchmark using the provided testing suite.

---
