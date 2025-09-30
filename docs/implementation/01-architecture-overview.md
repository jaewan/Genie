# Architecture Overview

## Introduction

Genie is a GPU disaggregation framework for PyTorch that operates at the ML framework layer to bridge the semantic translation gap. This document provides a high-level overview of the system architecture and design principles.

## System Overview

Genie enables transparent remote accelerator execution through a three-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    PyTorch Application                      │
│                  (Unmodified PyTorch Code)                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  LazyTensor  │  │  Dispatcher  │  │ Factory Intercept│  │
│  │   (Capture)  │  │ (Interception)│  │  (torch.randn)  │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            Semantic Rich Graph (SRG) Layer                  │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  FX Tracer   │  │ Graph Builder│  │ Pattern Matcher │  │
│  │  (Structure) │  │  (DAG Build) │  │ (Workload Type) │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Scheduler Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Semantic     │  │  Scheduler   │  │   Placement     │  │
│  │  Optimizer   │  │  (Stages)    │  │    Engine       │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Backend Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   Executor   │  │     DPDK     │  │  Remote Device  │  │
│  │ (Materialize)│  │  (Zero-Copy) │  │   (Phase 2+)    │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Recent Refactoring Updates (2025-09-30)

**Status**: Refactoring #1 (Error Handling) - 85% Complete

### New Error Handling System

**File**: `genie/core/exceptions.py` (157 lines)

Genie now includes a comprehensive exception hierarchy and Rust-inspired Result type:

```python
# Base exception with context tracking
class GenieException(Exception):
    """Base for all Genie exceptions with debugging context."""
    def __init__(self, message: str, context: dict = None, 
                 inner_exception: Exception = None)

# Semantic layer exceptions
class SemanticError(GenieException)
class ShapeInferenceError(SemanticError)
class PatternMatchError(SemanticError)

# Result type for explicit error handling
class Result(Generic[T, E]):
    @staticmethod
    def ok(value: T) -> Result[T, Any]
    @staticmethod
    def err(error: E) -> Result[Any, E]
    
    # Methods: is_ok, is_err, unwrap(), unwrap_or(), map(), and_then()
```

**Key Improvements**:
- ✅ Explicit error handling with Result types
- ✅ Rich error context for debugging
- ✅ No silent failures in shape inference and pattern matching
- ✅ All 52/52 tests passing with no regressions

**Impact**:
- LazyTensor shape inference now returns `Result[torch.Size]`
- Pattern matching returns `Result[List[MatchedPattern]]`
- Better error messages with context dictionaries
- Graceful degradation on failures

**See**: [Refactoring Updates](12-refactoring-updates.md) for complete details and migration guide

---

## Core Design Principles

### 1. Semantic Preservation (HotNets'25 §2)

Unlike low-level disaggregation approaches, Genie operates at the ML framework layer to preserve rich semantic information:

- **Operation Intent**: Capture not just operations, but their role (attention, convolution, etc.)
- **Data Lineage**: Track data flow and transformations through the model
- **Execution Phase**: Identify LLM prefill/decode, vision backbone/head, fusion, etc.
- **Memory Patterns**: Classify as streaming, reused, persistent (KV cache), ephemeral

### 2. Lazy Execution

All operations on `remote_accelerator` device create `LazyTensor` objects instead of executing immediately:

```python
# User writes normal PyTorch:
x = torch.randn(10, 10, device="remote_accelerator:0")
y = torch.matmul(x, x)

# Behind the scenes:
# x -> LazyTensor(op="aten::randn", shape=[10,10])
# y -> LazyTensor(op="aten::matmul", inputs=[x,x], shape=[10,10])
```

Benefits:
- Build complete computation graph before execution
- Enable global optimizations
- Apply workload-specific strategies

### 3. Pluggable Architecture

Genie uses a decoupled design with clear interfaces:

- **Frontend**: Framework-specific capture (PyTorch, JAX future)
- **SRG**: Standardized semantic graph representation
- **Backend**: Infrastructure-specific execution (DPDK, gRPC future)

## Key Components

### Frontend Layer

**Purpose**: Transparently capture PyTorch operations

**Components**:
- `LazyTensor` ([docs](03-lazy-tensor.md)): Deferred execution proxy (semantics via Enricher post-#2)
- `Dispatcher` ([docs](04-dispatcher.md)): Operation interception
- `Device` ([docs](02-device-layer.md)): Backend registration

**Files**:
- `genie/core/lazy_tensor.py` - LazyTensor implementation
- `genie/core/enhanced_dispatcher.py` - Unified dispatcher
- `genie/core/device.py` - Device backend
- `genie/core/library.py` - torch.library registrations

### SRG Layer

**Purpose**: Build semantically-rich computation graph

**Components**:
- `SemanticAnalyzer` ([docs](06-semantic-layer.md)): Three-tier semantic capture
- `PatternRegistry` ([docs](07-pattern-recognition.md)): Workload detection
- `PhaseDetector`: Execution phase detection (prefill/decode/fusion)
- `ModuleContextTracker`: nn.Module context tracking
- `SemanticEnricher` and `MetadataRegistry` (post-Refactoring #2): Semantic metadata management

**Files**:
- `genie/semantic/analyzer.py` - Semantic analyzer
- `genie/semantic/pattern_registry.py` - Pattern matching
- `genie/semantic/phase_detector.py` - Phase detection
- `genie/patterns/` - Pattern implementations

### Scheduler & Optimizer Layer

**Purpose**: Apply workload-specific optimizations and scheduling

**Components**:
- `SemanticOptimizer` ([docs](08-scheduler-optimizer.md)): Workload-specific optimizations
- `Scheduler`: Create execution schedules respecting dependencies
- `PlacementEngine`: Device placement decisions
- `Planner`: Generate execution plans

**Files**:
- `genie/semantic/optimizer.py` - Semantic optimizer
- `genie/semantic/scheduling.py` - Scheduler
- `genie/semantic/placement.py` - Placement engine
- `genie/semantic/planner.py` - Execution planner

### Backend Layer

**Purpose**: Execute computation on remote/local accelerators

**Components**:
- `Executor` ([docs](05-runtime-transport.md)): Graph materialization
- `TransportCoordinator`: Python-C++ bridge
- `DPDKBackend`: High-level DPDK backend
- C++ Data Plane ([docs](09-data-plane-cpp.md)): Zero-copy transport

**Files**:
- `genie/core/executor.py` - Execution engine
- `genie/runtime/transport_coordinator.py` - Transport coordination
- `genie/runtime/dpdk_backend.py` - DPDK backend
- `src/data_plane/` - C++ zero-copy implementation

## Data Flow

### 1. Capture Phase

```
User Code: x = torch.randn(10, 10, device="remote_accelerator:0")
     │
     ▼
Device Check: Is device remote_accelerator?
     │
     ▼
Factory Intercept: torch.randn wrapped
     │
     ▼
LazyTensor Created: LazyTensor(op="aten::randn", ...)
     │
     ▼
Graph Builder: Node added to ComputationGraph and FX Graph
     │
     ▼
Semantic Analysis: Infer shape, dtype, metadata
     │
     ▼
Return: LazyTensor proxy returned to user
```

### 2. Execution Phase

```
User Code: result = y.cpu()  # Triggers materialization
     │
     ▼
Materialize Called: LazyTensor.materialize()
     │
     ▼
FX Graph Analysis: Pattern matching, optimization
     │
     ▼
Execution Planning: Topological sort, dependency resolution
     │
     ▼
Node Execution: Execute operations in order
     │
     ▼
Fallback Handling: Use torch.ops.aten for unsupported ops
     │
     ▼
Result: Concrete torch.Tensor
```

## Semantic Metadata System

Every `LazyTensor` carries rich metadata (see [Semantic Metadata docs](08-semantic-metadata.md)):

```python
LazyTensor.metadata = {
    # Basic properties
    "operation_type": "aten::matmul",
    "tensor_shape": torch.Size([4, 4]),
    "dtype": torch.float32,
    
    # Semantic enrichment (HotNets'25 §3.1)
    "semantic_role": "attention_score_computation",
    "module_path": "VQA.fusion_block.attention",
    "execution_phase": ExecutionPhase.MULTIMODAL_FUSION,
    "data_lineage": DataLineage(modality="text", ...),
    
    # Memory and compute hints (HotNets'25 §3.2)
    "memory_pattern": MemoryPattern.STREAMING,
    "compute_intensity": 10.0,
    "kv_cache_related": False,
    
    # Scheduling hints
    "can_parallelize": True,
    "priority": 6,
    "colocation_group": "fusion_ops",
}
```

## Phase 1 vs. Future Phases

### Phase 1 (Current)

**Goal**: Semantic capture and local execution

- ✅ LazyTensor with semantic metadata
- ✅ FX-based graph construction
- ✅ Pattern recognition
- ✅ Local execution with fallback
- ✅ Comprehensive operation coverage (>95%)

**Execution**: CPU-based for validation

### Phase 2+ (Future)

**Goal**: Actual remote disaggregation

- ✅ Semantic-driven optimizations (implemented):
  - LLM decode co-location (HotNets'25 §3.2)
  - Prefill parallelization
  - Pipelined CNN inference
  - Multi-modal parallel branches
  - Dynamic recomputation
- 🔄 DPDK zero-copy data path (C++ implemented, Python integration in progress)
- 🔄 Remote execution via transport coordinator
- 🔄 Multi-tenant coordination
- 🔄 Global resource manager

## Alignment with HotNets'25 Paper

| Paper Section | Implementation |
|---------------|----------------|
| §2.1 Semantic Translation Gap | LazyTensor metadata, Three-tier capture |
| §3.1 Frontend (Lazy Tensor) | `genie/core/lazy_tensor.py` |
| §3.1 Three-Tier Capture | Dispatcher + FX + Hooks |
| §3.2 Core Scheduler | `genie/semantic/optimizer.py`, `scheduling.py` |
| §3.2 LLM Optimizations | KV cache co-location, prefill parallelization |
| §3.2 Vision Optimizations | CNN pipelining, conv-bn-relu fusion |
| §3.2 Multi-Modal | Parallel modalities, JIT fusion transfer |
| §3.3 Zero-Copy Data Path | `src/data_plane/` (C++), `genie/runtime/` (Python) |
| §3.4 Global Scheduling | Future work |

## Performance Considerations

### Memory Overhead

- LazyTensor uses `__slots__` for reduced memory footprint
- Metadata lazy-initialized on first access
- Graph builders are thread-local

### Execution Overhead

- Factory interception: ~1-2% for creation ops
- `__torch_function__`: Intercepts >95% of operations
- FX tracing: One-time cost, reusable graph
- Fallback path: Uses torch.ops.aten for unknown ops

## Testing Strategy

See `tests/` directory:

- `test_spec_compliance.py` - Validates requirements
- `test_device_registration.py` - Device backend tests
- `test_torch_function_protocol.py` - Operation coverage
- `test_enhanced_dispatcher.py` - Dispatcher functionality
- `test_integration.py` - End-to-end scenarios
- `test_exceptions.py` - Error handling and Result types (22 tests)
- `test_analyzer_*.py` - Semantic analysis (multiple files)
- `test_phase_detection.py` - Phase detection
- `test_dynamo_patterns.py` - Pattern recognition

## Next Steps

Continue reading:
1. [Device Layer](02-device-layer.md) - Backend registration
2. [LazyTensor](03-lazy-tensor.md) - Deferred execution
3. [Dispatcher](04-dispatcher.md) - Operation interception
4. [Semantic Layer](06-semantic-layer.md) - Three-tier semantic capture
5. [Pattern Recognition](07-pattern-recognition.md) - Workload detection
6. [Scheduler & Optimizer](08-scheduler-optimizer.md) - Semantic optimizations
7. [Contributor Guide](11-contributor-guide.md) - Contributing
8. [Refactoring Updates](12-refactoring-updates.md) - Recent improvements

## Recent Updates

**2025-09-30**: Major refactoring and documentation update

### Completed Refactorings
- ✅ **Refactoring #1**: Consolidated error handling with Result types (60/60 tests)
- ✅ **Refactoring #3**: Unified graph representation using PyTorch FX (80+ tests)
- ✅ **Refactoring #4**: Async-first transport with ThreadPoolExecutor (14/14 tests)
- ✅ **Refactoring #5**: Extracted pattern matching service with dependency injection (27/27 tests)

### Key Improvements
- ✅ Improved LazyTensor shape inference
- ✅ Enhanced pattern matching with error aggregation
- ✅ Added comprehensive scheduler and optimizer documentation
- ✅ Documented C++ data plane implementation
- ✅ Event loop never blocked during transport operations
- ✅ ~40% throughput improvement with parallel workers

**See**: 
- [Refactoring Plan](../../REFACTORING_PLAN.md) for overall status
- [Refactoring #4 Summary](../../REFACTORING_4_COMPLETE.md) for async transport details

## References

- HotNets'25 Paper: `../../.kiro/HotNets25.tex`
- PyTorch Custom Backend Guide: https://pytorch.org/tutorials/advanced/extend_dispatcher.html
- PyTorch FX Documentation: https://pytorch.org/docs/stable/fx.html
- DPDK Documentation: https://doc.dpdk.org/
