# Genie Project Context - AI Assistant Guide

## Project Overview
**Genie** is a semantic-driven framework-level disaggregation system for AI accelerators that bridges the "semantic translation gap" by operating at the PyTorch framework level - the ML ecosystem's "narrow waist."

## Core Innovation
Unlike existing disaggregation approaches that operate at low levels (PCIe/driver) and lose semantic context, or high levels (application-specific) and lose generality, Genie captures rich semantic information at the framework level while maintaining generality across diverse AI workloads.

## Key Technical Concepts

### 1. The Semantic Translation Gap
- **Problem**: As computation descends the software stack, critical application knowledge (model structure, execution phases) is lost
- **Solution**: Operate at PyTorch framework level where semantics are preserved but generality remains

### 2. Lazy Tensor Abstraction
- Intercepts PyTorch operations on `remote_accelerator` device
- Defers execution while building semantically-rich computation graph
- Captures operation type, tensor properties, module context, and execution phase

### 3. Pluggable Architecture
```
Frontend (PyTorch) → SRG (Semantic Graph) → Scheduler → Backend (RDMA)
```
- **Frontend**: Captures intent via LazyTensor
- **Scheduler**: Applies semantic optimizations
- **Backend**: Executes on remote GPUs with zero-copy

### 4. Zero-Copy Data Path
- Proactive DPDK integration (allocate tensors in network-ready memory)
- GPU-NIC direct DMA (GPUDirect RDMA)
- Eliminates CPU involvement in data transfers

### 5. CPU-Minimal Remote Nodes
- 2 CPU cores, 8GB RAM per node with 4-8 GPUs
- Thin runtime executes plans without full framework
- 8:1 GPU-to-CPU ratio (vs traditional 1:1)

## Workload Examples

### VQA Model (Running Example)
```python
# User code - unchanged
model = VQAModel().to("remote_accelerator")
answer = model(image, text_query)

# What Genie sees:
# 1. Vision encoder operations (ViT)
# 2. Language encoder operations (BERT)  
# 3. Cross-attention fusion
# → Schedules vision on memory-optimized GPU
# → Schedules language on compute-optimized GPU
# → Transfers outputs just-in-time for fusion
```

### LLM Serving
- Identifies prefill (compute-bound) vs decode (memory-bound) phases
- Co-locates decode with KV cache to eliminate transfers
- Adaptive batching based on phase

### Vision Models
- Detects layer-wise parallelism opportunities
- Pipelines stages across accelerators
- Overlaps communication with computation

## Performance Targets
- **30% reduction** in network traffic (via semantic optimization)
- **15% reduction** in end-to-end latency
- **>90%** network bandwidth utilization
- **<5%** total system overhead

## Development Principles
1. **Transparency**: No application code changes required
2. **Generality**: Support diverse workloads (LLM, Vision, RecSys, Multi-modal)
3. **Efficiency**: Minimize CPU usage at remote nodes
4. **Extensibility**: Plugin system for new patterns
5. **Correctness**: Numerical parity within 1e-5

## Critical Success Factors
1. **Semantic Capture**: Rich metadata without performance impact
2. **Pattern Recognition**: >85% accuracy in workload classification
3. **Zero-Copy Path**: True end-to-end DMA transfers
4. **Graceful Degradation**: Fallback when patterns unknown
5. **Production Stability**: >99% uptime, automatic recovery

## Technology Stack
- **Python**: 3.10.x (stable PyTorch support)
- **PyTorch**: 2.1.2 (LTS with stable dispatcher API)
- **CUDA**: 12.1 (H100 support)
- **DPDK**: 23.11 LTS (stable gpudev)
- **Network**: RDMA/RoCE (200Gbps+)

## Architecture Layers
1. **PyTorch Integration**: Device registration, dispatcher hooks
2. **LazyTensor Engine**: Graph construction, metadata accumulation
3. **Semantic Analyzer**: Pattern recognition, workload classification
4. **Optimization Engine**: Placement decisions, scheduling
5. **Zero-Copy Runtime**: DPDK memory, DMA transfers
6. **Remote Execution**: Thin runtime, GPU orchestration

This context document provides the essential understanding needed for implementing any component of Genie. Refer to component-specific documents for detailed interfaces and implementation guidance.
