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
│  │ Cost Model   │  │  Optimizer   │  │   Placement     │  │
│  │  (Future)    │  │   (Future)   │  │    (Future)     │  │
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
- `LazyTensor` ([docs](03-lazy-tensor.md)): Deferred execution proxy
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
- `FXGraphBuilder` ([docs](06-fx-integration.md)): PyTorch FX integration
- `SemanticMetadata` ([docs](08-semantic-metadata.md)): Rich annotations
- `PatternMatcher` ([docs](07-pattern-recognition.md)): Workload detection

**Files**:
- `genie/core/fx_graph_builder.py` - FX graph construction
- `genie/core/semantic_metadata.py` - Metadata definitions
- `genie/patterns/` - Pattern recognition

### Backend Layer

**Purpose**: Execute computation on remote/local accelerators

**Components**:
- `Executor` ([docs](05-executor.md)): Graph materialization
- DPDK Runtime: Zero-copy data path (Phase 2)
- Remote Device: Actual disaggregation (Phase 2)

**Files**:
- `genie/core/executor.py` - Execution engine
- `genie/csrc/runtime.cpp` - DPDK bindings

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

- 🔄 DPDK zero-copy data path
- 🔄 Remote execution scheduler
- 🔄 Multi-tenant coordination
- 🔄 Global resource manager
- 🔄 Semantic-driven optimizations:
  - LLM decode co-location (HotNets'25 §3.2)
  - Pipelined CNN inference
  - Dynamic recomputation
  - Phase-aware allocation

## Alignment with HotNets'25 Paper

| Paper Section | Implementation |
|---------------|----------------|
| §2.1 Semantic Translation Gap | LazyTensor metadata, FX tracing |
| §3.1 Frontend (Lazy Tensor) | `genie/core/lazy_tensor.py` |
| §3.1 Hook-based Enhancement | Module context tracking |
| §3.2 Core Scheduler | Pattern-based optimization (Phase 2) |
| §3.3 Zero-Copy Data Path | DPDK runtime (`genie/csrc/runtime.cpp`) |
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

## Next Steps

Continue reading:
1. [Device Layer](02-device-layer.md) - Backend registration
2. [LazyTensor](03-lazy-tensor.md) - Deferred execution
3. [Dispatcher](04-dispatcher.md) - Operation interception
4. [Development Guide](09-development-guide.md) - Contributing

## References

- HotNets'25 Paper: `../HotNets25.tex`
- PyTorch Custom Backend Guide: https://pytorch.org/tutorials/advanced/extend_dispatcher.html
- PyTorch FX Documentation: https://pytorch.org/docs/stable/fx.html
