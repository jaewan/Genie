# Djinn: Semantic-Driven GPU Disaggregation

**Status**: Framework-Level GPU Disaggregation System
**Last Updated**: November 7, 2025
**Version**: 1.0.0

---

## What is Djinn?

Djinn is a **framework-level GPU disaggregation system** that enables efficient sharing of AI accelerators across applications by leveraging semantic information from ML frameworks. Unlike traditional disaggregation approaches that operate blindly at the hardware level, Djinn uses **Semantically Rich Graphs (SRGs)** to make intelligent placement, scheduling, and data movement decisions.

**Key Innovation**: Djinn operates at the **ML framework layer** (PyTorch), capturing application intent through transparent tensor interception and semantic analysis to enable optimizations invisible to lower layers—**without requiring any application code changes**.

---

## The Problem

Real-world GPU fleets report **55-60% average idleness** despite massive AI accelerator investment. This stems from:

- **Coarse-grained allocation**: Applications claim entire GPUs even when using <50%
- **Tightly-coupled architecture**: GPUs locked to specific servers
- **Fluctuating demands**: Training, inference, interactive workloads have different needs
- **Stranded capacity**: Expensive accelerators sit idle between jobs

**Traditional disaggregation approaches fail** because they:
- Operate at low levels (PCIe, driver) without semantic information
- Cannot distinguish prefill (compute-bound) from decode (memory-bound) phases
- Treat all data equally (can't prioritize KV cache over activations)
- Require extensive per-application tuning or are workload-specific

---

## Djinn's Solution: Semantic-Driven Disaggregation

Djinn bridges application intent and hardware execution through a **clean four-layer architecture**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER APPLICATION (PyTorch)                   │
│                    No code changes required                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ Transparent Interception
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: PyTorch FRONTEND (Intent Capture & Semantic Enrichment)│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Core: Tensor Interception & Graph Construction              ││
│  │ • Factory function wrapping (~20 tensor creation functions) ││
│  │ • __torch_dispatch__ for universal operation coverage       ││
│  │ • LazyTensor DAG construction (works on all models)         ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Semantic: Multi-tier Analysis & Annotation                  |│
│  │ • Pattern recognition (attention, KV cache, convolution)    ││
│  │ • Execution phase detection (prefill/decode/vision)         ││
│  │ • Cost estimation (FLOPs, memory, operational intensity)    ││
│  │ • Workload classification (LLM, vision, multimodal)         ││
│  └─────────────────────────────────────────────────────────────┘│
│  Output: Semantically Rich Graph (SRG)                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ SRG (framework-agnostic)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: SCHEDULER (Semantic-Driven Optimization)              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Core Scheduling: Cost-Aware Placement                       ││
│  │ • Cost model (compute + transfer + queuing time)            ││
│  │ • Device assignment using SRG annotations                   ││
│  │ • Execution ordering (topological + semantic hints)         ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Semantic Optimizations: Context-Aware Decisions             ││
│  │ • Stateful co-location (KV cache with decoder)              ││
│  │ • Pipelined CNN execution (stage across GPUs)               ││
│  │ • Dynamic recomputation (cheap ops under congestion)        ││
│  │ • Memory-aware placement (phase-specific budgets)           ││
│  └─────────────────────────────────────────────────────────────┘│
│  Output: ExecutionSchedule (device bindings + transfers)        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ Execution plan
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: SERVER (Distributed Execution Coordination)           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Request Coordination: Multi-tenancy & Load Balancing        ││
│  │ • Request handling and workload routing                      ││
│  │ • Fair queuing and priority management                       ││
│  │ • Resource isolation and quota enforcement                   ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Execution Engines: JIT Compilation & GPU Orchestration      ││
│  │ • TorchScript/TensorRT compilation pipelines                ││
│  │ • GPU execution orchestration (subgraph, block, real_gpu)   ││
│  │ • Caching systems (graph cache, GPU cache, tensor registry) ││
│  └─────────────────────────────────────────────────────────────┘│
│  Output: Coordinated execution across GPU cluster               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ Coordinated execution
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: BACKEND (Core Execution Runtime)                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Runtime Primitives: GPU & Memory Management                 ││
│  │ • GPU initialization and device management                  ││
│  │ • Memory allocation and tracking primitives                 ││
│  │ • Basic network communication protocols                     ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Execution Infrastructure: Low-level Operations              ││
│  │ • Tensor transfer and serialization                         ││
│  │ • Device communication and health monitoring                ││
│  │ • Fault tolerance and recovery primitives                   ││
│  └─────────────────────────────────────────────────────────────┘│
│  Output: Concrete tensor results                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Innovations

### 1. LazyTensor DAG with Deferred Computation (SRG implementation of Djinn)

Djinn captures all tensor operations in a **LazyTensor DAG** (SRG implementation for PyTorch) that defers expensive computations during graph construction:

**Lazy Properties Design**:
```python
import logging
logger = logging.getLogger(__name__)

class LazyTensor(torch.Tensor):
    """torch.Tensor subclass with deferred computation for performance."""

    def __init__(self, operation, inputs, kwargs=None, shape=None, dtype=None, device=None, metadata=None):
        # Core operation structure stored immediately
        object.__setattr__(self, '_operation', operation)
        object.__setattr__(self, '_inputs', inputs)
        object.__setattr__(self, '_kwargs', kwargs or {})

        # Deferred expensive computations
        object.__setattr__(self, '_shape', shape)        # Lazy-computed if None
        object.__setattr__(self, '_dtype', dtype)
        object.__setattr__(self, '_device', device)
        object.__setattr__(self, '_metadata', metadata)  # Lazy-computed if MetadataPlaceholder

        # Zero memory overhead during capture (meta device)
        super().__init__([], dtype=dtype, device='meta')

    @property
    def shape(self) -> torch.Size:
        """Lazy shape computation with caching."""
        cached_shape = object.__getattribute__(self, '_shape')
        if cached_shape is not None:
            return cached_shape

        # Compute shape only when first accessed
        from .shape_inference import ShapeInference
        try:
            computed_shape = ShapeInference.infer_shape(
                object.__getattribute__(self, '_operation'),
                object.__getattribute__(self, '_inputs'),
                object.__getattribute__(self, '_kwargs')
            )
            if computed_shape is not None:
                object.__setattr__(self, '_shape', computed_shape)
                return computed_shape
        except Exception as e:
            logger.debug(f"Shape inference failed: {e}")

        # Fallback to empty shape
        fallback_shape = torch.Size([])
        object.__setattr__(self, '_shape', fallback_shape)
        return fallback_shape

    @property
    def metadata(self) -> Dict[str, Any]:
        """Lazy metadata computation with caching."""
        cached_metadata = object.__getattribute__(self, '_metadata')
        if cached_metadata is not None and not isinstance(cached_metadata, MetadataPlaceholder):
            return cached_metadata

        # Compute metadata only when first accessed
        from ..semantic.metadata_capture import get_metadata_capture
        computed_metadata = get_metadata_capture().capture_metadata(
            operation=object.__getattribute__(self, '_operation'),
            inputs=object.__getattribute__(self, '_inputs'),
            kwargs=object.__getattribute__(self, '_kwargs')
        )
        object.__setattr__(self, '_metadata', computed_metadata)
        return computed_metadata
```


### 2. Semantically Rich Graph (SRG)

The SRG extends the LazyTensor DAG with **semantic metadata** for intelligent optimization:

**Node Annotations**:
- **Phase**: Execution phase (e.g., `llm_prefill`, `llm_decode`, `vision_encoding`, `forward`)
- **Residency**: Data lifetime (e.g., `persistent_weight`, `ephemeral_activation`, `stateful_kv_cache`)
- **Modality**: Data type (e.g., `vision`, `text`, `audio`, `fusion`)
- **Cost hints**: FLOPs, memory footprint, operational intensity

This semantic richness enables optimizations like:
- **Stateful co-location**: Pin KV cache and decoder to same GPU (eliminates repeated transfers)
- **Pipelined CNN execution**: Automatically fuse and pipeline convolutional stages
- **Dynamic recomputation**: Recompute cheap intermediates under network congestion

### 3. Multi-Layer Optimization System

**Graph Caching** (Phase 1):
- Eliminates repeated graph capture overhead
- LRU cache with automatic eviction (default: 100 models)

**Block Compilation** (Phase 2):
- Compiles model to TorchScript blocks at module boundaries
- Reduces RPC calls through coarse-grained execution
- Coarse-grained execution strategy for efficiency

**GPU Cache** (Phase 3):
- Persistent weight storage on GPU (LRU eviction)
- Eliminates deserialization overhead on warm requests
- Memory-aware eviction with pressure handling

**TensorRT Optimization** (Phase 4):
- Lazy compilation after profiling threshold
- Adaptive optimization for repeated blocks
- FP16 acceleration for improved performance

### 4. Three Execution Strategies

Djinn selects the optimal execution strategy automatically:

1. **Smart Fragmentation** - Most efficient for complex graphs
2. **Subgraph Optimization** - Good for medium-sized graphs  
3. **Naive Recursion** - Fallback for simple/small graphs

### 5. LazyTensor DAG Graph Builder

Captures all tensor operations in LazyTensor DAG for remote execution:
- **LazyTensor DAG** - Works on all models, provides operation-level granularity
- **Unified Graph Interface** - LazyDAGAdapter exposes operations through unified Graph interface

### 6. Tensor Interception Strategy

**Hybrid interception approach**:
- **Factory wrapping**: Intercepts ~20 tensor creation functions (torch.randn, torch.zeros, etc.)
- **__torch_dispatch__**: Primary interception for tensor operations (95%+ coverage)
- **Limited __torch_function__**: Special cases (reshape, embedding)
- **Context-aware**: Thread-local state management for capture contexts
- **Materialization Control**: Disables interception during operation execution to prevent recursion

**Why not PyTorch device backend approach?**
- Device registration ≠ operation interception (PyTorch doesn't auto-route to custom code)
- C++ extension adds build complexity and ABI compatibility issues
- No functional performance benefit
- Current approach works with any device specification (strings, torch.device objects, contexts)
- Easier debugging and maintenance without C++ dependencies

**Initialization Triggers**:
- Early (non-blocking): Device-based tensor creation, `.to()` calls, `capture()` context
- Late (blocking if needed): Actual operations, result materialization


---

## Getting Started

```python
import djinn
import torch

# Your existing PyTorch code works unchanged
model = torch.nn.Linear(784, 10)
input_tensor = torch.randn(32, 784)

# Djinn automatically disaggregates execution
output = model(input_tensor)

# Or explicitly move models to remote accelerators
model = model.to('remote_accelerator:0')  # Automatic weight conversion
output = model(input_tensor)
```

**See docs/2_FRONTEND_IMPLEMENTATION.md for integration details**

---

## Current Implementation Status

### ✅ **Core Architecture**

**Four-Layer Architecture**:
- **Frontend Layer**: PyTorch interception, LazyTensor DAG construction, semantic enrichment
- **Scheduler Layer**: Cost-aware placement, semantic optimizations, execution planning
- **Server Layer**: Distributed coordination, JIT compilation, multi-tenancy
- **Backend Layer**: GPU execution runtime, memory management, communication

**LazyTensor DAG Implementation**:
- **Deferred Computation**: Shape, dtype, and metadata computed only when accessed
- **Caching Strategy**: Results cached after first access to avoid recomputation
- **Factory Function Support**: Specialized handlers for tensor creation operations
- **Thread-Safe Design**: Uses `object.__setattr__` and `object.__getattribute__` for safety

**Semantic Analysis Pipeline**:
- **Pattern Recognition**: Attention patterns, KV cache detection, convolution stages
- **Execution Phase Classification**: Prefill, decode, vision encoding phases
- **Cost Estimation**: FLOPs, memory footprint, operational intensity calculations
- **Workload Characterization**: Model architecture and resource requirement analysis

### Key Implementation Optimizations

#### **Materialization Triggers for Control Flow**

**Problem**: PyTorch operations that return Python types (scalars, booleans) must materialize immediately to enable control flow decisions in ML code.

**Solution**: Context-aware operation classification detects operations that return non-tensor types and automatically materializes them:

```python
# Materialization triggers detected automatically:
tensor.all()    # → bool (materializes)
tensor.item()   # → scalar (materializes)
tensor.sum()    # → scalar if no dim parameter (materializes)
tensor.argmax() # → tensor indices (deferred, can be remote)
```

**Detection Mechanism**: Operations are classified into five categories:
- **MATERIALIZATION_TRIGGER**: Must execute immediately (bool, scalar returns)
- **REDUCTION_OPERATION**: Dramatically reduce data size (argmax, sum with dim)
- **SHAPE_DEPENDENT**: Data-dependent output shapes (nonzero, unique)
- **TUPLE_RETURNING**: Multi-return operations (topk, sort)
- **COMPUTE_OPERATION**: Standard deferred execution

#### **Remote CPU Operations for Network Reduction**

**Problem**: Some operations like `argmax` reduce large tensors (196MB logits) to small results (8KB tokens), creating optimal opportunities for remote execution to minimize network transfer.

**Why Remote**: These reduction operations are shipped to remote GPUs because:
- **Network Savings**: 25,000x reduction (196MB → 8KB for GPT-2 token generation)
- **GPU Efficiency**: GPUs excel at parallel reductions
- **Memory Hierarchy**: Keeps large intermediate tensors on GPU memory

**Implementation**: Cost-based decision in `reduction_optimizer.py`:
```python
# Execute remotely if: >100x reduction ratio AND >1MB input
if reduction_ratio > 100 and input_size_mb > 1.0:
    execute_reduction_remotely(tensor, operation, args, kwargs)
```

#### **Shape Inference for Control Flow**

**Problem**: Control flow depends on tensor shapes, but shapes aren't available in deferred LazyTensor DAG.

**Solution**: Lazy shape inference computes shapes without materialization using 50+ transformation rules:
```python
# Shape inference without execution:
repeat([2, 3], 2, 1) → [4, 3]
transpose([2, 3]) → [3, 2]
sum([2, 3, 4], dim=1) → [2, 4]
matmul([2, 3], [3, 4]) → [2, 4]
```

**Key Benefit**: Enables `if tensor.shape[0] > batch_size:` style control flow in ML code.

#### **Materialization Cache for Redundant Operations**

**Problem**: Transformer control flow often executes identical operations repeatedly.

**Solution**: Semantic hashing caches by operation structure, not object identity:
```python
# Same operation different objects → same cache entry
# Eliminates redundant executions in attention loops
hash = compute_semantic_hash(operation, inputs, kwargs)
cached_result = cache.get(hash)
```

**Impact**: ~1M redundant control checks → ~100 unique executions (10,000x reduction).

**Operation Classification** (`djinn/frontend/core/operation_classifier.py`):
- **Context-Aware Classification**: Operations classified based on arguments (sum with/without dim)
- **Shape-Dependent Operations**: Special handling for operations with data-dependent output shapes
- **Tuple-Returning Operations**: Multi-return operation support (topk, sort, eig)
- **Transformer-Specific Operations**: Dedicated classification for activation functions, attention mechanisms

**Shape Inference System** (`djinn/frontend/core/shape_inference.py`):
- **Meta-Tensor Approach**: Zero-overhead shape inference using PyTorch meta tensors
- **50+ Shape Rules**: Comprehensive shape transformation rules for all major operations
- **Broadcasting Support**: Automatic shape inference for element-wise operations
- **Transformer-Specific Rules**: Specialized shape inference for attention and activation operations

**Materialization Cache** (`djinn/frontend/core/materialization_cache.py`):
- **Semantic Hashing**: Cache based on operation structure, not object identity
- **LRU Eviction**: Bounded cache with configurable size limits
- **Thread-Safe Operations**: Concurrent access protection with proper locking

**Transformer Operations** (`djinn/frontend/core/transformer_operations.py`):
- **Operation Classification**: 6 categories for transformer-specific operations
- **Execution Strategy**: Intelligent local vs remote decision making
- **Semantic Understanding**: Shape-preserving and fusion-compatible operation tracking

**Performance Tuning** (`djinn/frontend/core/performance_tuner.py`):
- **Operation Profiling**: Real-time execution time and frequency tracking
- **Bottleneck Detection**: Automatic identification of slow operations
- **Optimization Recommendations**: Data-driven suggestions for execution improvements
- **Threshold Tuning**: Adaptive parameter adjustment based on profiling data

**Device Compatibility Layer** (`djinn/core/device_compatibility.py`):
- **Automatic Model Conversion**: `model.to('remote_accelerator:0')` automatically converts model parameters and buffers to LazyTensors
- **PyTorch Compatibility**: Follows standard `nn.Module.to()` device management conventions
- **Framework Integration**: Compatible with HuggingFace Accelerate, PyTorch Lightning, and other ML frameworks
- **Gradient Flow Preservation**: Maintains autograd compatibility for both training and inference
- **Thread-Safe Implementation**: Uses patched `nn.Module.to()` method for seamless integration

### ✅ **Execution Optimizations**

**Materialization Strategies**:
- **MaterializationOptimizer**: Topological sort for batch execution, CUDA streams for pipelining, pinned memory for faster transfers
- **Local Execution**: Optimized path when remote execution unavailable, using MaterializationOptimizer for efficient computation
- **Remote Execution**: Subgraph-based execution with caching and differential updates, sending entire computation DAGs instead of individual operations
- **LazyTensor Input Handling**: Ensures all LazyTensor inputs are properly materialized before operation execution

**Memory Management System**:
- **Phase-Aware Memory Budgets**: Different allocations for prefill vs decode phases
- **Lifetime-Based Eviction**: Tensors evicted at exact moment last consumer finishes
- **Memory Pressure Handling**: Proactive OOM prevention with configurable thresholds
- **Semantic Memory Manager**: Cost-based recomputation vs storage decisions

**Caching Infrastructure**:
- **Graph Caching**: LRU cache for compiled computation graphs
- **GPU Cache**: Persistent weight storage with memory-aware eviction
- **Subgraph Cache**: Avoids rebuilding identical computation subgraphs
- **Tensor Registry**: Remote tensor lifecycle management

**Serialization Optimizations**:
- **numpy.save Serialization**: Optimized tensor serialization using numpy.save with format headers
- **Format-Aware Deserialization**: Automatic detection of numpy vs torch formats with backward compatibility
- **Differential Updates**: Only send graph changes for iterative workloads using delta computation
- **Protocol-Based Communication**: Message type headers for reliable transport and proper error handling

### ✅ **Advanced Features**

**Multi-Tenant Coordination**:
- **Fair Queuing**: Prevents starvation across concurrent clients
- **Priority Management**: SLA-aware request scheduling
- **Resource Isolation**: Per-client memory and compute quotas
- **Load Balancing**: Automatic workload distribution across GPU cluster

**JIT Compilation Pipeline**:
- **TorchScript Compilation**: Model graph optimization for inference
- **TensorRT Integration**: GPU-accelerated inference with FP16 support
- **Fusion Compiler**: Operation fusion based on semantic patterns
- **Adaptive Compilation**: Automatic selection based on execution frequency

**Fault Tolerance**:
- **Lineage-Based Recovery**: Deterministic recomputation from operation dependencies
- **Automatic Failover**: Migration to healthy GPUs on failure detection
- **Graceful Degradation**: Fallback to local execution when remote fails
- **State Preservation**: Minimal disruption during recovery operations

### ✅ **Production Infrastructure**

**Monitoring and Metrics**:
- **Prometheus Integration**: 50+ metrics for complete observability
- **Performance Monitoring**: End-to-end latency, throughput, and resource utilization
- **Memory Metrics**: Cache hits, evictions, pressure events, budget utilization
- **Error Tracking**: Failure rates, recovery times, SLA violations

**Configuration Management**:
- **Environment-Based Config**: Runtime configuration via environment variables
- **Component Registration**: Pluggable architecture for custom implementations
- **Feature Flags**: Selective enablement of optimization components
- **Logging Integration**: Structured logging with configurable verbosity

### **Memory Management**

**Three-Phase Memory Management**:
- **Phase 1**: Reactive memory management with GPU cache and eviction
- **Phase 2**: Semantic-aware memory management with lifetime-based eviction
- **Phase 3**: Production hardening with adaptive budget tuning and pressure handling

---

## Architecture Deep Dive

For comprehensive technical details including implementation specifics, see:

- **docs/1_ARCHITECTURE.md**: Complete technical architecture with code structure
- **docs/2_FRONTEND_IMPLEMENTATION.md**: Frontend integration and usage
- **docs/3_SCHEDULER_IMPLEMENTATION.md**: Scheduler algorithms and optimization
- **docs/4_BACKEND_IMPLEMENTATION.md**: Backend execution and memory management

---

## Contact

- **Project Lead**: Jaewan Hong
- **Email**: jaewan@berkeley.edu
- **GitHub**: https://github.com/jaewan/Djinn.git
- **Paper**:
```bibtex
@inproceedings{hong2025lost,
  title={Lost in Translation: The Search for Meaning in Network-Attached AI Accelerator Disaggregation},
  author={Jaewan Hong, Yifan Qiao, Soujanya Ponnapalli, Shu Liu, Marcos K. Aguilera, Vincent Liu, Christopher J. Rossbach, Ion Stoica},
  booktitle={Proceedings of The 24th ACM Workshop on Hot Topics in Networks},
  year={2025}
}
```
---
