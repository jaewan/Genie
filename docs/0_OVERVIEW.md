# Djinn: Semantic-Driven GPU Disaggregation

**Status**: Performance Evaluation via Running Benchmarks
**Last Updated**: November 5, 2025
**Version**: 1.0.0

---

## What is Djinn?

Djinn is a **framework-level GPU disaggregation system** that enables efficient sharing of AI accelerators across applications by leveraging semantic information from ML frameworks. Unlike traditional disaggregation approaches that operate blindly at the hardware level, Djinn uses **Semantically Rich Graphs (SRGs)** to make intelligent placement, scheduling, and data movement decisions.

**Key Innovation**: Djinn operates at the **ML framework layer** (PyTorch), capturing application intent to enable optimizations that are invisible to lower layers—**without requiring any application code changes**.

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

### 1. Semantically Rich Graph (SRG)

The SRG is Djinn's central abstraction—a **portable intermediate representation** that captures both computational structure and semantic intent:

**Node Annotations**:
- **Phase**: Execution phase (e.g., `llm_prefill`, `llm_decode`, `vision_encoding`, `forward`)
- **Residency**: Data lifetime (e.g., `persistent_weight`, `ephemeral_activation`, `stateful_kv_cache`)
- **Modality**: Data type (e.g., `vision`, `text`, `audio`, `fusion`)
- **Cost hints**: FLOPs, memory footprint, operational intensity

This semantic richness enables optimizations like:
- **Stateful co-location**: Pin KV cache and decoder to same GPU (eliminates repeated transfers)
- **Pipelined CNN**: Automatically fuse and pipeline convolutional stages
- **Dynamic recomputation**: Recompute cheap intermediates under network congestion

### 2. Multi-Layer Optimization System

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
- FP16 acceleration for 2-3x speedup

### 3. Three Execution Strategies

Djinn selects the optimal execution strategy automatically:

1. **Smart Fragmentation** - Most efficient for complex graphs
2. **Subgraph Optimization** - Good for medium-sized graphs  
3. **Naive Recursion** - Fallback for simple/small graphs

### 4. LazyTensor DAG Graph Builder

Captures all tensor operations in LazyTensor DAG for remote execution:
- **LazyTensor DAG** - Works on all models, provides operation-level granularity
- **Unified Graph Interface** - LazyDAGAdapter exposes operations through unified Graph interface

### 5. Tensor Interception Strategy

**Hybrid interception approach**:
- **Factory wrapping**: Intercepts ~20 tensor creation functions (torch.randn, torch.zeros, etc.)
- **__torch_dispatch__**: Primary interception for tensor operations (95%+ coverage)
- **Limited __torch_function__**: Special cases (reshape, embedding)
- **Context-aware**: Thread-local state management for capture contexts

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
```

**See docs/2_FRONTEND_IMPLEMENTATION.md for integration details**

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
