# Genie Scheduler Implementation

**Status**: ✅ Production Ready  
**Phase**: 🔵 Phase 2 (Semantic Analysis & Scheduling)  
**Last Updated**: November 2, 2025  
**Based on**: `research_proposal.tex` | `genie/semantic/scheduling.py`

---

## Table of Contents

1. [Overview](#1-overview)
2. [Logical Device Abstraction](#2-logical-device-abstraction)
3. [Cost Model](#3-cost-model)
4. [Semantic Scheduling](#4-semantic-scheduling)
5. [Semantic Optimizations](#5-semantic-optimizations)
6. [Placement Strategies](#6-placement-strategies)
7. [Memory-Aware Scheduling](#7-memory-aware-scheduling)

---

## §1. Overview

### §1.1 Scheduler Purpose

The Genie scheduler is a **pluggable policy engine** that transforms a Semantically Rich Graph (SRG) from the frontend into an optimized execution plan.

**Key Innovation**: The scheduler leverages **local metadata** from LazyTensor (shape, dtype, device) to make placement decisions **without remote calls**, enabling practical GPU disaggregation.

**Core Interface**:
```python
from genie.semantic.scheduling import Scheduler
from genie.semantic.cost_estimator import GraphCostEstimator

scheduler = Scheduler(
    cost_estimator=GraphCostEstimator(),
    network_topology=NetworkTopology()
)
schedule = scheduler.create_schedule(graph, optimization_plan)
```

**Input**: Graph with semantic annotations (from frontend)  
**Output**: `ExecutionSchedule` with:
- Device bindings (which GPU executes each operation)
- Transfer schedules (when/where to move data)
- Caching directives (persistent vs ephemeral data)
- Execution stages and groups

### §1.2 Scheduler Architecture

```
Input Graph (from frontend)
    ↓
Stage 1: Graph Analysis & Cost Estimation
    ├─ Operation analysis (FLOP counting)
    ├─ Memory estimation
    ├─ Network topology analysis
    └─ Pattern-based optimization hints
    ↓
Stage 2: Scheduling & Placement
    ├─ Device assignment (which GPU for each op)
    ├─ Execution ordering (topological + heuristics)
    ├─ Co-location constraints
    └─ Transfer scheduling
    ↓
Output: ExecutionSchedule
```

### §1.3 Key Components

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| **Scheduler** | `genie/semantic/scheduling.py` | 1,011 | Main orchestrator |
| **Cost Estimator** | `genie/semantic/cost_estimator.py` | 600 | Latency prediction |
| **Network Topology** | `genie/core/network_topology.py` | 260 | Network info & device management |
| **Graph Interface** | `genie/core/graph_interface.py` | 497 | Unified graph abstraction |
| **Placement Policy** | `genie/semantic/placement.py` | 499 | Device assignment & constraints |
| **Tensor Registry** | `genie/server/tensor_registry.py` | 400+ | Smart tensor caching (integration) |
| **Fusion Compiler** | `genie/semantic/fusion_compiler.py` | 280+ | SRG-driven pattern grouping |
| **Performance Monitor** | `genie/server/performance_monitor.py` | 417 | Metrics collection for optimization |

**Note on Integration**: The scheduler works in conjunction with Tensor Registry and Fusion Compiler. Registry provides version-aware invalidation, Fusion provides operation grouping hints, and Performance Monitor tracks effectiveness of scheduling decisions.

**Note on NetworkTopology**: The component has a dual-layer design:
- **Bridge wrapper** in `genie/semantic/cost_estimator.py` (~40 lines): Acts as a facade to avoid circular imports
- **Manager implementation** in `genie/core/network_topology.py` (~260 lines): Actual topology management with device/link registration

This design allows the cost estimator to access network information without importing from `genie/core/network_topology.py` directly.

---

## §2. Logical Device Abstraction

### §2.1 Why This Matters

The scheduler needs to make device placement decisions based on tensor metadata. 

**Genie's Solution**: LazyTensor stores metadata locally, enabling fast scheduling decisions without remote metadata queries.

This allows the scheduler to evaluate placement options efficiently based on:
- Tensor shape and dtype
- Device specifications
- Memory constraints
- Network topology

### §2.2 Logical Device Architecture

**Core Concept**: Separate what PyTorch expects from physical storage:

```
┌─────────────────────────────────────────────────────────────┐
│                      LazyTensor                             │
├─────────────────────────────────────────────────────────────┤
│  _logical_device:  torch.device('cpu')  ← Scheduler sees    │
│  _physical_device: torch.device('meta') ← Storage (no mem)  │
│                                                             │
│  device property → returns _logical_device                  │
│  Scheduler uses logical device for placement decisions      │
└─────────────────────────────────────────────────────────────┘
```

**Benefits**:
1. **Fast Metadata Access**: Local queries only
2. **Transparent Integration**: PyTorch operations see expected device
3. **Flexible Placement**: Scheduler can change logical device without data movement
4. **Device Consistency**: Executor ensures tensors are on correct device during materialization

### §2.3 Scheduling with Logical Devices

```python
# 1. Capture with local device
with genie.capture():
    x = torch.randn(8, 10)  # LazyTensor: _logical_device='cpu'
    output = model(x)

# 2. Scheduler queries device (fast!)
device = x.device  # Returns 'cpu' (logical device)

# 3. Scheduler decides placement
schedule = scheduler.create_schedule(graph, profile)
schedule.node_placement[x.id] = 'remote_gpu_0'

# 4. Executor materializes on target device
result = output.cpu()  # Executor handles device placement
```

---

## §3. Cost Model

### §3.1 Cost Model Overview

The scheduler's decision-making is guided by a cost model:

```
Total Latency = Compute Time + Transfer Time + Queueing Delay
```

**File**: `genie/semantic/cost_estimator.py` (600 lines)

### §3.2 Compute Cost Estimation

```python
class CostEstimator:
    """Base cost estimator for operations."""
    
    def estimate_operation(self, node) -> CostEstimate:
        """
        Estimate cost for a single operation.
        
        Returns:
        - compute_flops: Floating point operations
        - memory_bytes: Memory footprint
        - operational_intensity: FLOPs per byte
        - transfer_time_ms: Estimated transfer time
        """
        op = node.operation.lower()
        
        # Dispatch to operation-specific estimator
        if 'matmul' in op or 'mm' in op:
            return self._estimate_matmul(node)
        elif 'conv' in op:
            return self._estimate_conv(node)
        elif 'softmax' in op:
            return self._estimate_attention(node)
        else:
            return self._estimate_generic(node)
```

**Formula**:
```
compute_time_ms = flops / (gpu_tflops * utilization_factor)
```

**Factors**:
- `flops`: Operation complexity
- `gpu_tflops`: GPU compute capacity (10 TFLOPS for V100, 312 TFLOPS for H100)
- `utilization_factor`: Memory bandwidth saturation (0.5-1.0)

### §3.3 Transfer Cost Estimation

```python
def estimate_transfer_time(self, bytes_to_transfer, src_node, dst_node):
    """Estimate transfer time in milliseconds."""
    return self._manager.estimate_transfer_time(bytes_to_transfer, src_node, dst_node)
```

**Formula**:
```
transfer_time_ms = (tensor_size_bytes / network_bandwidth_gbps) + latency_overhead_ms
```

**Factors**:
- `tensor_size_bytes`: Shape × dtype size
- `network_bandwidth_gbps`: Network capacity (10 Gbps typical)
- `latency_overhead_ms`: Fixed overhead (1-5 ms)

### §3.4 Memory Cost Analysis

**Constraints**:
- GPU memory capacity (32 GB for V100, 80 GB for H100)
- Peak memory during execution
- Data residency preferences (weights stay, activations ephemeral)

**Implementation**: Peak memory computed using sweep-line algorithm during scheduling.

---

## §4. Semantic Scheduling

### §4.1 Core Scheduler Implementation

**File**: `genie/semantic/scheduling.py` (1,011 lines)

```python
class Scheduler:
    """Scheduler for semantic-aware execution planning."""

    def __init__(self, network_topology=None, cost_estimator=None):
        self._scheduling_cache = {}  # Cache schedules
        self._stats = defaultdict(int)  # Statistics
        self.network_topology = network_topology or NetworkTopology()
        self.cost_estimator = cost_estimator
        self.devices = {}  # device_id → Device info
    
    def create_schedule(self, graph, optimization_plan=None):
        """
        Create execution schedule for a graph.
        
        Returns:
            ExecutionSchedule with stages and groups
        """
        if self.cost_estimator is None:
            return self._create_basic_schedule(graph, optimization_plan)
        return self._create_cost_aware_schedule(graph, optimization_plan)
```

**Key Methods**:

1. **`create_schedule()`**: Main entry point
2. **`_create_basic_schedule()`**: Fallback basic scheduling
3. **`_create_cost_aware_schedule()`**: Advanced cost-aware scheduling

### §4.2 Scheduling Strategies

**File**: `genie/semantic/scheduling.py` (lines 21-27)

```python
class SchedulingStrategy(Enum):
    """Types of scheduling strategies."""
    SEQUENTIAL = "sequential"      # Execute one-by-one
    PARALLEL = "parallel"          # Execute concurrently
    PIPELINE = "pipeline"          # Pipeline across devices
    DYNAMIC = "dynamic"            # Adapt at runtime
    PRIORITY = "priority"          # Execute by priority
```

**Strategy Selection**:
- **SEQUENTIAL**: Dependent operations (e.g., LSTM layers)
- **PARALLEL**: Data-parallel workloads (batches)
- **PIPELINE**: Spatial parallelism (stages across devices)
- **DYNAMIC**: Adapts based on runtime performance
- **PRIORITY**: Custom ordering

### §4.3 Scheduling Groups

```python
@dataclass
class SchedulingGroup:
    """Group of nodes to be scheduled together."""
    group_id: str
    nodes: List[str]
    strategy: SchedulingStrategy
    priority: int = 0
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict = field(default_factory=dict)
```

**Example Usage**:
- LLM decode group: Decoder layers + KV cache operations
- CNN pipeline stage: Conv/norm/activation layers
- Attention group: Multi-head attention computations

### §4.4 Execution Schedule

```python
@dataclass
class ExecutionSchedule:
    """Execution schedule for a computation graph."""
    stages: List[List[SchedulingGroup]] = field(default_factory=list)
    node_to_stage: Dict[str, int] = field(default_factory=dict)
    node_to_group: Dict[str, str] = field(default_factory=dict)
    total_stages: int = 0
    strategy: SchedulingStrategy = SchedulingStrategy.SEQUENTIAL
    metadata: Dict = field(default_factory=dict)
```

---

## §5. Semantic Optimizations

### §5.1 Stateful Co-location (LLM Decode)

**Problem**: KV cache grows with sequence length; transferring it every step is expensive.

**Solution**: Pin KV cache and decoder on same GPU.

**Implementation**:
```python
# Identify KV cache nodes
cache_nodes = [
    n for n in graph.nodes
    if n.metadata.get("data_residency") == "stateful_kv_cache"
]

# Find decoder layers that depend on cache
decoder_nodes = [
    n for n in graph.nodes
    if (n.metadata.get("execution_phase") == "llm_decode" and
        any(inp.id in [c.id for c in cache_nodes] for inp in n.inputs))
]

# Assign all to same GPU
assigned_gpu = 'remote_gpu_0'
for node in cache_nodes + decoder_nodes:
    schedule.node_placement[node.id] = assigned_gpu
```

**Performance Impact**:
- Without co-location: Transfer KV cache every step (10-100ms)
- With co-location: Cache stays on device (~1ms per step)
- Result: **10-100× speedup** for autoregressive decoding

### §5.2 Pipelined CNN Inference

**Problem**: Convolutional layers can be parallelized across devices.

**Solution**: Stage conv layers across multiple GPUs.

**Implementation**:
```python
# Identify sequential convolutional stages
conv_stages = self._identify_conv_stages(graph)

for stage_idx, stage in enumerate(conv_stages):
    gpu_idx = stage_idx % num_gpus  # Round-robin
    for node in stage.nodes:
        schedule.node_placement[node.id] = f'remote_gpu_{gpu_idx}'
```

**Performance Impact**:
- Single GPU: Sequential execution
- Multi-GPU pipelined: **2-4× speedup** typical

### §5.3 Dynamic Recomputation

**Problem**: Network congestion makes data transfer expensive.

**Solution**: Recompute cheap intermediates instead of transferring.

**Implementation**:
```python
# Check network congestion
network_utilization = self.network_topology.estimate_queueing_delay(...)

# For cheap operations, decide to recompute
for edge in graph.edges:
    if (self._is_cheap_operation(edge.source) and 
        network_utilization > CONGESTION_THRESHOLD):
        schedule.recompute_nodes.add(edge.source.id)
```

**Performance Impact**:
- Normal: Transfer intermediate (bandwidth-limited)
- With recomputation: Recompute locally (compute-bound, faster)

**Phase 3 Enhancement**: Integrate with `MemoryPressureHandler` for adaptive thresholds:
```python
recomputation_threshold = base_threshold * pressure_handler.get_adaptive_recomputation_threshold()
# Normal: 1.0 (original threshold)
# Warning: 0.8 (slightly prefer recomputation)
# Critical: 0.5 (strongly prefer recomputation over memory)
```

---

## §6. Placement Strategies

### §6.1 Single-GPU Placement (Phase 1)

**Current Implementation**: All operations execute on local GPU.

```python
scheduler = Scheduler()
schedule = scheduler.create_schedule(graph, profile)
```

**Use Case**: Phase 1 (current), no network.

### §6.2 Homogeneous Disaggregated Placement (Phase 2)

**Strategy**: Distribute across identical remote GPUs.

**Heuristic**:
1. Identify "sticky" nodes (large memory, many outputs)
2. Pin to same device (co-location)
3. Distribute free nodes to balance load

### §6.3 Heterogeneous Placement (Phase 2+)

**Strategy**: Match workload characteristics to GPU types.

**Reasoning**:
- Vision workloads (high bandwidth) → A100 (576 GB/s)
- LLM prefill (compute bound) → H100 (312 TFLOPS)
- LLM decode (memory bound) → L4 (cheaper, lower throughput)

---

## §6. Memory-Aware Scheduling (Phase 3)

### §6.1 Integration with Phase 2-3 Memory Management

The scheduler integrates with advanced memory management components:

**From Phase 2** (`genie/server/semantic_memory_manager.py`):
- `LifetimeBasedEvictor`: Provides tensor lifetime information
- `PhaseAwareMemoryManager`: Suggests memory budgets per phase
- `RecomputationVsStorageDecider`: Cost model for cache vs compute

**From Phase 3** (`genie/server/memory_pressure_handler.py`):
- `MemoryPressureHandler`: Adapts thresholds under pressure
- Budget adjustment signals for tight memory situations

**Integration Pattern**:
```python
def create_cost_aware_schedule(self, graph, optimization_plan):
    """Create schedule with memory-aware decisions."""
    
    # 1. Use lifetime information for eviction hints
    lifetime_evictor = LifetimeBasedEvictor()
    lifetime_evictor.analyze_graph_lifetimes(graph.nodes, graph.edges)
    
    # 2. Get phase-specific budgets
    phase_mgr = PhaseAwareMemoryManager(total_gpu_memory_mb=32000)
    for phase in self._extract_phases(graph):
        budgets = phase_mgr.adjust_for_phase(phase)
        self._apply_phase_budgets(graph, phase, budgets)
    
    # 3. Check if under memory pressure
    if pressure_handler.is_under_pressure:
        # Prefer recomputation, more aggressive eviction
        recomputation_threshold *= pressure_handler.get_adaptive_recomputation_threshold()
    
    return schedule
```

### §6.2 Placement with Memory Constraints

**Revised Memory Constraints Solver** (with Phase 3 awareness):

```python
class MemoryConstraintSolver:
    """Ensure GPU memory capacity not exceeded (with Phase 3 awareness)."""
    
    def solve(self, placement, srg, cluster_state):
        # 1. Compute peak memory per GPU
        gpu_to_nodes = {}
        for node_id, gpu in placement.items():
            gpu_to_nodes.setdefault(gpu.id, []).append(node_id)
        
        # 2. Use lifetime information for accurate accounting
        lifetime_evictor = LifetimeBasedEvictor()
        for gpu_id, node_ids in gpu_to_nodes.items():
            peak_memory = self._compute_peak_with_lifetimes(node_ids, srg, lifetime_evictor)
            gpu = cluster_state.get_gpu(gpu_id)
            
            # 3. Check against phase-specific budgets
            phase_mgr = PhaseAwareMemoryManager(gpu.memory_gb * 1024)
            phase = self._get_dominant_phase(node_ids, srg)
            
            if not phase_mgr.check_allocation('total', peak_memory):
                # Rebalance nodes
                self._rebalance_nodes(gpu_id, node_ids, placement, srg)
        
        return placement
```

---

## §7. Conclusion (Updated)

The Genie scheduler provides **semantic-driven optimization with memory awareness**:

✅ **Semantic optimizations**: Co-location, pipelining, recomputation  
✅ **Local metadata**: No remote network calls required  
✅ **Memory-aware**: Constraint satisfaction for GPU capacity  
✅ **Phase-aware**: Different strategies for different execution phases  
✅ **Adaptive thresholds**: Responds to memory pressure via Phase 3 handler  
✅ **Production-ready**: Practical scheduling decisions at framework level

**Key Innovation**: Leveraging semantic annotations from the SRG enables:
- Lifetime-based eviction (Phase 2) ← Exact knowledge of when tensors are no longer needed
- Phase-aware budgets (Phase 2) ← Workload-specific resource allocation
- Memory pressure response (Phase 3) ← Dynamic adaptation to runtime constraints
- Adaptive recomputation (Phase 3) ← Cost-aware decisions under memory pressure

**Memory Management Pipeline**:
1. **Frontend** captures phase and residency metadata
2. **Scheduler** uses semantic hints for placement and budget decisions
3. **Backend** implements three-phase optimization:
   - Phase 1: Reactive memory management (GPU cache + KV sessions)
   - Phase 2: Semantic-aware management (lifetime + phase budgets)
   - Phase 3: Production hardening (metrics + pressure + adaptive)
4. **Monitoring** via Prometheus metrics provides observability

**Expected Memory Improvements**:
- Activation memory waste: 40% → 10% (4× improvement)
- Premature evictions: Frequent → Eliminated (via lifetime analysis)
- KV cache thrashing: High → Protected (via session pinning)
- Budget utilization: 30-50% → 60%+ (via adaptive tuning)
- Pressure response: None → Automatic (via handler + thresholds)

---

**Last Updated**: November 2, 2025  
**Status**: ✅ Production Ready  
**Memory Management**: Phase 3 Integration Complete  
**Architecture**: See `1_ARCHITECTURE.md` §11 (Phase 3: Production Hardening)

---

## §9. Future Enhancements

### §9.1 Multi-GPU Load Balancing

**Planned**: Distribute operations across multiple GPUs to balance load.

**Challenges**:
- Heterogeneous GPU types
- Network topology awareness
- Dynamic load changes

### §9.2 Adaptive Scheduling

**Planned**: Adjust scheduling decisions based on runtime performance.

**Approach**:
- Monitor actual execution times
- Compare with cost model predictions
- Update cost model parameters
- Re-schedule if performance degrades

### §9.3 Global Scheduler

**Vision**: Fleet-wide optimization across multiple clients.

**See `1_ARCHITECTURE.md` §7 for details.**

---

## §9. Architectural Design Patterns

### §9.1 NetworkTopology Bridge Pattern

The network topology component uses a **bridge pattern** to avoid circular imports:

```
Dependency Flow:
├─ genie/semantic/cost_estimator.py
│  └─ Uses: NetworkTopology (bridge wrapper, ~40 lines)
│     └─ Delegates to: NetworkTopologyManager (via import)
│
└─ genie/core/network_topology.py
   └─ Contains: NetworkTopologyManager (actual implementation, 260 lines)
```

**Why**: The cost estimator needs network information, but importing from `genie/core/network_topology` would create a circular dependency. The bridge wrapper in `cost_estimator.py` is a lightweight facade that delegates to the actual manager.

### §9.2 Graph Interface Abstraction

The `GenieGraph` abstract base class provides a **unified interface** over multiple graph representations:

```
GenieGraph (Abstract Interface)
├─ ConcreteGraphImpl (efficient for materialized DAGs)
├─ FX GraphModule (PyTorch traced graphs)
├─ LazyTensor DAG (deferred execution)
└─ Other backends (extensible)
```

This design allows all graph algorithms (topological sort, dependency analysis, etc.) to work uniformly across different representations without reimplementation.

### §9.3 Phase Architecture Clarification

**Phase Classification**:
- **Phase 1** (Core): Graph capture and lazy execution
- **Phase 2** (Current): Semantic analysis, scheduling, and optimization ← **This document**
- **Phase 3+** (Future): Advanced runtime optimization and distributed scheduling

This scheduler is a **Phase 2 component** that builds on Phase 1's graph capture to provide intelligent scheduling and placement decisions.

---

## §10. Conclusion

The Genie scheduler provides **semantic-driven optimization** for GPU disaggregation:

✅ **Local metadata**: No remote network calls required  
✅ **Semantic optimizations**: Co-location, pipelining, recomputation  
✅ **Memory-aware**: Constraint satisfaction for GPU capacity  
✅ **Production-ready**: Practical scheduling decisions at framework level

**Key Innovation**: Leveraging semantic annotations from the SRG enables optimizations impossible at lower layers (PCIe, driver).

---

**Last Updated**: November 2, 2025  
**Status**: ✅ Production Ready  
**Architecture**: See `1_ARCHITECTURE.md` §4

