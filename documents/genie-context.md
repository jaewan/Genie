# Genie Project Context - AI Assistant Guide

## Project Overview
**Genie** is a semantic-driven framework-level disaggregation system for AI accelerators that bridges the "semantic translation gap" by operating at the PyTorch framework level - the ML ecosystem's "narrow waist."

## Core Innovation
Unlike low-level approaches (driver/PCIe) that lose semantic context or high-level, application-specific approaches that lose generality, Genie captures rich semantic information at the framework level while maintaining generality across diverse AI workloads.

## Key Technical Concepts

### 1. The Semantic Translation Gap
- **Problem**: As computation descends the software stack, critical application knowledge (model structure, execution phases) is lost
- **Solution**: Operate at PyTorch framework level where semantics are preserved but generality remains

### 2. Lazy Tensor Abstraction
- Intercepts PyTorch intent on `remote_accelerator` with a fallback-first strategy
- Primary capture via factory hooks (creation ops) and lightweight wrappers; `__torch_function__` is optional
- Defers execution while building a semantically rich computation graph
- Captures operation type, tensor properties, module context, and execution phase

### 3. Pluggable Architecture
```
Frontend (PyTorch) → SRG (Semantic Graph) → Scheduler → Backend (Transport/Runtime)
```
- **Frontend**: Captures intent via LazyTensor
- **Scheduler**: Applies semantic optimizations, emits execution plans
- **Backend**: Executes plans locally or remotely; evolves to zero-copy transport

### 4. Execution Modes (Phased)
- `local` (Phase 1): Materialize on CPU via eager PyTorch fallback; force device to CPU during execution
- `local_remote` (Phase 2): In-process/subprocess C++ runtime using LibTorch; control-plane loopback; shared/pinned memory
- `remote` (Phase 3): Remote runtime over TCP pinned memory; same protocol artifacts
- `remote_zero_copy` (Phase 4): DPDK + GPUDev zero-copy transport (DEFAULT for production)
  - Uses standard NICs - no special hardware required
  - Custom reliable protocol optimized for GPU transfers
  - Works in all cloud environments (AWS, GCP, Azure)

### 5. Zero-Copy Data Path (DPDK + GPUDev Default)
- **PRIMARY**: DPDK + GPUDev for commodity cloud compatibility (no RNICs required)
- Works with standard NICs (Intel E810, Mellanox ConnectX) in AWS/GCP/Azure
- GPU-NIC direct DMA via GPUDev library (not RDMA - works without special hardware)
- Custom reliable protocol over UDP (software reliability for cloud environments)
- Pinned-memory/TCP fallback when DPDK unavailable
- Optional RDMA upgrade when RNICs present on both ends (not required)

### 6. CPU-Minimal Remote Nodes
- 2 CPU cores, 8GB RAM per node with 4–8 GPUs
- Thin runtime executes plans without full framework
- 8:1 GPU-to-CPU ratio (vs traditional 1:1)

## Workload Examples

### VQA Model (Running Example)
```python
# User code - unchanged
model = VQAModel().to("remote_accelerator")
answer = model(image, text_query)

# Genie observes:
# 1. Vision encoder ops (ViT)
# 2. Language encoder ops (BERT)
# 3. Cross-attention fusion
# → Places stages, overlaps transfers (Phase 3+), co-locates fused regions
```

### LLM Serving
- Identifies prefill (compute-bound) vs decode (memory-bound) phases
- Co-locates decode with KV cache to eliminate transfers (Phase 3+)
- Adaptive batching based on phase

### Vision Models
- Detects layer-wise parallelism opportunities
- Pipelines stages across accelerators
- Overlaps communication with computation

## Performance Targets
- **30% reduction** in network traffic (via semantic optimization)
- **15% reduction** in end-to-end latency
- **>90%** network bandwidth utilization (Phase 4)
- **<5%** total system overhead

## Development Principles
1. **Transparency**: No application code changes required
2. **Generality**: Support diverse workloads (LLM, Vision, RecSys, Multi-modal)
3. **Efficiency**: Minimize CPU usage at remote nodes
4. **Extensibility**: Plugin system for new patterns
5. **Correctness**: Numerical parity within 1e-5
6. **Pragmatism**: Phased execution from local to zero-copy remote

## Numerics Stability Policy (Executor & Capture)

Genie prioritizes correctness parity with PyTorch. Certain op chains are numerically sensitive (e.g., tanh → mul → softmax). Small deviations in dtype/broadcast semantics or evaluation order can be amplified. To guarantee parity while retaining performance, we adopt the following policy:

- FX-bypass whitelist for numerics-critical ops: `aten::softmax`, `aten::log_softmax`, `aten::logsumexp`, `aten::exp`, `aten::log`, `aten::tanh`. These ops (and their short chains) execute via eager fallback to avoid drift.
- Elementwise fast path (disabled by default): gated by `GENIE_ENABLE_ELEMENTWISE_FASTPATH=1`. When disabled (default), dtype promotion and broadcasting behavior mirror PyTorch via meta inference or eager fallback.
- Additional FX-bypass entries: `aten::{matmul,mm,bmm,mean,var,std,add,sub,mul,div,argmax,argmin}` are bypassed while the FX path matures for exact parity.
- Env flags:
  - `GENIE_ENABLE_ELEMENTWISE_FASTPATH=0|1` (default 0): micro-optimization for elementwise ops; enable only when parity is validated.
  - `GENIE_ANALYZER_DEBUG=0|1`: verbose pattern logs.
  - `GENIE_ENABLE_META_INFER=0|1` (default 1): meta-based shape/dtype inference without materialization.

This policy delivered 40/40 on the comprehensive correctness suite and is designed to remain stable as we evolve the FX executor.

## Critical Success Factors
1. **Semantic Capture**: Rich metadata without performance impact (overhead <10µs/op)
2. **Pattern Recognition**: >85% accuracy in workload classification
3. **Zero-Copy Path**: End-to-end DMA transfers (Phase 4)
4. **Graceful Degradation**: Fallback when patterns unknown or transport unavailable
5. **Production Stability**: >99% uptime, automatic recovery

## Technology Stack
- **Python**: 3.10.x (stable PyTorch support)
- **PyTorch**: 2.2.x–2.5.x (stable dispatcher; `torch.library` optional)
- **CUDA**: 12.1 (H100 support)
- **DPDK**: 23.11 LTS with GPUDev (PRIMARY network backend)
- **Network**: Standard Ethernet NICs (100-200Gbps)
  - Works with commodity cloud NICs (no RNICs required)
  - Optional RDMA/RoCE when available (not required)

## Architecture Layers
1. **PyTorch Integration**: Device registration, factory hooks, optional `__torch_function__`
2. **LazyTensor Engine**: Graph construction, metadata accumulation, materialization triggers
3. **Semantic Analyzer**: Pattern recognition, workload classification, plan generation
4. **Optimization Engine**: Placement decisions, scheduling
5. **Runtime & Transport**: Compatibility mode (TCP/pinned) → Zero-copy (DPDK/RDMA)
6. **Remote Execution**: Thin runtime, GPU orchestration

This context document provides the essential understanding needed for implementing any component of Genie. Refer to component-specific documents for detailed interfaces and implementation guidance.

## Fallback-First Strategy
- Correctness-first: Non-intercepted ops materialize inputs and execute eagerly on CPU.
- Selective interception: Common ops (creation, arithmetic, matmul, reductions) create `LazyTensor` nodes for semantic richness.
- Opt-in integrations: `torch.library` impls and extended dispatcher paths are gated by env flags.

### Environment Flags
- `GENIE_LOG_INTERCEPTS=1`: Log intercepted operations for debugging.
- `GENIE_ENABLE_ATEN_IMPL=1`: Enable torch.library impls for PrivateUse1 registration.
- `GENIE_ENABLE_META_INFER=1`: Allow meta-based shape/dtype inference in LazyTensor when needed.
