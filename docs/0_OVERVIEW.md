**Status**: Beta (v1.0.0)  
**Last Updated**: November 10, 2025  
**Target Audience**: ML Engineers, Infrastructure Teams, Researchers

---

## TL;DR

Djinn enables efficient GPU sharing across ML workloads by intercepting PyTorch operations to understand their semantic intent (prefill vs decode, weights vs activations, compute vs memory-bound). This semantic awareness drives intelligent scheduling decisions across disaggregated GPUs—without requiring any application code changes.

**Key Value:** Transform 55-60% idle GPU capacity into usable compute through semantic-aware resource management.

---

## The Problem: Massive GPU Underutilization

Modern GPU clusters suffer from a fundamental mismatch between allocation granularity and actual usage:

- **Fact**: Production GPU fleets report 55-60% average idleness despite massive investment
- **Root Cause**: Applications claim entire GPUs even when using fraction of capacity
- **Result**: Expensive accelerators sit idle while other workloads queue

### Why Traditional Solutions Fail

**Hardware-level disaggregation** (PCIe, driver):
- ❌ No semantic understanding (can't distinguish prefill from decode)
- ❌ Treats all data equally (can't prioritize KV cache over activations)
- ❌ Requires hardware modifications

**Application-level solutions**:
- ❌ Require code changes for each model
- ❌ Break with framework updates
- ❌ Limited to specific workload types

---

## Djinn's Solution: Framework-Level Semantic Capture

Djinn operates at the sweet spot—the ML framework layer—where semantic information is available but applications remain unchanged.

### Core Innovation: Semantic Awareness

Instead of blindly moving data, Djinn understands what your model is doing:

| **What Djinn Sees** | **Traditional Systems See** | **Optimization Enabled** |
|---------------------|----------------------------|-------------------------|
| LLM prefill phase (parallel, compute-bound) | Generic matrix operations | Schedule on compute-optimized GPUs |
| LLM decode phase (sequential, memory-bound) | More matrix operations | Co-locate with KV cache, minimize transfers |
| KV cache (stateful, growing) | Generic tensors | Pin to GPU, avoid repeated transfers |
| Activation recomputation opportunity | Intermediate results | Recompute under network congestion |
| CNN pipeline stages | Independent operations | Pipeline across GPUs for overlap |

### How It Works: Transparent Interception

```python
# Your existing PyTorch code - unchanged
model = GPT2Model()
input_ids = torch.tensor([[1, 2, 3]])
output = model(input_ids)

# With Djinn - just change device
model = GPT2Model().to('remote_accelerator:0')  # Automatic interception
input_ids = torch.tensor([[1, 2, 3]], device='remote_accelerator:0')
output = model(input_ids)  # Operations captured, not executed
result = output.cpu()      # Triggers optimized remote execution
```

**What happens under the hood:**
1. **Capture**: Operations build a computation graph (no execution yet)
2. **Enrich**: Pattern recognition adds semantic metadata
3. **Model Registration** (first time): Model architecture and weights cached server-side
4. **Execute** (subsequent calls): Send only model fingerprint + inputs, server executes cached model directly
5. **Fallback**: Unregistered models fall back to graph execution (backward compatible)

---

## When to Use Djinn

### ✅ **Ideal Use Cases**

- **Multi-tenant GPU clusters**: Multiple models sharing resources
- **Mixed workloads**: Training + inference on same infrastructure
- **Phased computation**: Models with distinct prefill/decode phases
- **Standard architectures**: Transformers, CNNs, common patterns
- **Batch processing**: Where 10-50ms latency is acceptable

### ⚠️ **Use With Caution**

- **Ultra-low latency**: Requirements under 10ms
- **Custom operations**: Exotic ops may need manual handlers
- **Memory-critical**: Workloads needing precise memory control
- **Single-GPU dedicated**: No benefit without sharing

### ❌ **Not Recommended**

- **Real-time inference**: <5ms latency requirements
- **Debugging scenarios**: Adds complexity to error diagnosis
- **Prototype development**: Overhead not worth it for experimentation

---

## Performance Characteristics

### Architecture Performance

**Key Innovation**: Server-side model caching eliminates repeated graph transfer overhead.

| **Scenario** | **Latency** | **What Happens** |
|-------------|-------------|------------------|
| **First Request** (model registration) | 1-100s | Model architecture + weights transferred once, cached server-side |
| **Subsequent Requests** (cached execution) | 30-400ms | Only fingerprint + inputs sent, direct model execution |
| **GPU Execution** | 30-35ms | **12% faster than PyTorch baseline** (optimized memory management) |
| **Graph Fallback** | 500-4000ms | For unregistered models (backward compatible) |

### Performance Envelope

**Best Case** (small model, cached):
- **2-5ms** total latency (vs 700ms+ old system)
- **303x faster** than old graph-based execution
- **98% network reduction** (fingerprint vs full graph)

**Typical Case** (GPT-2-XL, cached):
- **400ms** total latency (vs 3.8s old system)
- **9.4x faster** than old system
- **89% network reduction**
- GPU execution: **30.82ms** (12% faster than PyTorch)

**Worst Case** (unregistered model, graph fallback):
- **500-4000ms** latency (old system behavior)
- Automatic fallback ensures compatibility
- First execution triggers registration for future speedup

*Note: All measurements hardware and workload dependent. Benchmark your specific use case.*

---

## Architecture Overview

Djinn uses a dual-path architecture optimized for production workloads:

```
Application (PyTorch) → No changes required
        ↓
[1] Frontend Layer → Intercepts operations, builds semantic graph
        ↓
[2] Model Manager → Detects registered models, routes to optimal path
        ↓
    ┌─────────────────┬──────────────────┐
    │                 │                  │
[3a] Model Cache     [3b] Graph Execution
    Path (Fast)      Path (Fallback)
    │                 │
    │                 │
    • Model ID only   • Full graph
    • Direct exec     • Scheduler
    • 9-300x faster   • Semantic-aware
    │                 │
    └────────┬────────┴──────────┐
             ↓                    ↓
[4] Server Layer → Coordinates execution, manages GPU cache
        ↓
[5] Backend Layer → Executes on GPUs with phase-aware memory
```

**Note**: The **Scheduler** (Layer 2) is still part of the architecture and actively used in the Graph Execution Path for semantic-aware device placement. The Model Cache Path bypasses the scheduler for speed, but semantic hints are still extracted and sent to the server.

**Key Design Principles:**
- **Transparency**: No application changes required
- **Semantic-Driven**: ML-aware optimization decisions
- **Model Caching**: Server-side caching eliminates repeated graph transfer
- **Dual Execution**: Fast path for registered models, fallback for compatibility
- **Phase-Aware**: Memory management adapts to execution phase (prefill/decode/vision)
- **Fault Tolerant**: Automatic fallback and recovery

**Architecture Evolution**: The redesigned system separates semantic understanding (client-side) from execution efficiency (server-side). Models are registered once, then executed directly without graph reconstruction overhead.

For detailed architecture, see [docs/1_ARCHITECTURE.md](1_ARCHITECTURE.md).

---

## Getting Started

### Installation

```bash
pip install djinn-gpu  # Coming soon
# OR
git clone https://github.com/jaewan/Djinn.git
cd Djinn && pip install -e .
```

### Basic Usage

```python
import djinn
import torch

# Option 1: Explicit device specification
model = YourModel().to('remote_accelerator:0')
input = torch.randn(32, 784, device='remote_accelerator:0')
output = model(input)       # Captured (no execution)
result = output.cpu()        # Executes remotely

# Option 2: Context-based capture  
with djinn.capture():
    model = YourModel()
    input = torch.randn(32, 784)
    output = model(input)    # Captured
result = output.cpu()        # Executes remotely
```

### What Just Happened?

**First Request (Model Registration)**:
1. **Graph Construction**: Operations recorded as LazyTensor DAG
2. **Model Fingerprinting**: Deterministic model identification
3. **Registration**: Model architecture + weights sent to server (one-time)
4. **Server Caching**: Model cached server-side for future requests

**Subsequent Requests (Cached Execution)**:
1. **Model Detection**: Client recognizes registered model
2. **Minimal Transfer**: Only fingerprint (16 bytes) + inputs sent
3. **Direct Execution**: Server executes cached model directly (no graph reconstruction)
4. **Result Return**: Output tensor transferred back

**Fallback (Unregistered Models)**:
- Falls back to graph execution (old system behavior)
- Ensures backward compatibility
- First execution triggers registration for future speedup

---

## Implementation Maturity

### ✅ Production Ready
- PyTorch operation interception (~95% of common ops)
- LazyTensor graph construction
- **Model cache system** (server-side caching)
- **Dual execution paths** (model cache + graph fallback)
- **Phase-aware memory management** (prefill/decode/vision)
- TCP-based remote execution
- Device compatibility layer
- **Optimized input preparation** (async transfer, pinned memory)

### ⚠️ Beta Quality
- Semantic pattern recognition (transformers, CNNs)
- Shape inference (has failure fallbacks)
- Multi-tenant fairness
- Memory pressure handling
- Architecture registry (model reconstruction)

### 🔬 Experimental
- TensorRT compilation (deferred - compatibility concerns)
- TorchScript compilation (deferred - compatibility concerns)
- Global cluster scheduling
- Cross-framework support

---

## Known Limitations

### Functional Limitations

1. **In-place operations** converted to out-of-place (5% memory overhead)
2. **Dynamic shapes** may force materialization (loses optimization)
3. **Mixed device operations** trigger early materialization
4. **Custom operations** require manual registration
5. **Debugging complexity** increases with remote execution
6. **Tuple operations** return `LazyTuple` (works like tuple but preserves laziness)

### Performance Limitations

1. **Model registration**: 1-100s one-time overhead (depends on model size)
2. **Network latency**: 30-400ms per request (unavoidable for remote GPU access)
3. **Cache eviction**: Large models may be evicted under memory pressure
4. **Graph fallback**: Unregistered models use slower graph execution path
5. **Shape inference**: May timeout after 500ms (circuit breaker, graph path only)

### When Things Go Wrong

**Shape Inference Failures:**
- Circuit breaker activates after 10 consecutive failures
- Falls back to materialization (eager execution)
- Check logs for problematic operations

**Network Issues:**
- Automatic fallback to local execution
- Lineage-based recovery for partial failures
- Timeout configurable (default 30s)

**Memory Pressure:**
- Proactive eviction at 80% utilization
- Emergency eviction at 95%
- Falls back to recomputation if needed

---

## Deployment Models

### Single Machine (Development)
```yaml
Configuration:
  - Djinn client: In Python process
  - Djinn server: Local daemon
  - Communication: TCP localhost
  - GPUs: Local CUDA devices
```

### Cluster (Production)
```yaml
Configuration:
  - Djinn clients: Across compute nodes
  - Djinn servers: GPU node pool
  - Communication: TCP/RDMA network
  - Coordinator: Central scheduler
```

See [deployment guide](docs/deployment.md) for detailed setup.

---

## Comparison with Alternatives

| **Aspect** | **Djinn** | **JAX** | **TensorFlow** | **torch.fx** | **torch.compile** |
|-----------|-----------|---------|----------------|--------------|-------------------|
| Semantic Awareness | ✅ Full | ❌ No | ❌ No | ⚠️ Limited | ❌ No |
| GPU Disaggregation | ✅ Native | ❌ No | ⚠️ tf.distribute | ❌ No | ❌ No |
| Zero Code Changes | ✅ Yes | ❌ No | ❌ No | ⚠️ Decorators | ⚠️ Decorators |
| Dynamic Control Flow | ✅ Yes | ⚠️ Limited | ⚠️ Limited | ❌ No | ✅ Yes |
| Production Maturity | ⚠️ Beta | ✅ Stable | ✅ Stable | ✅ Stable | ✅ Stable |

---

## Debugging

### Enable Debug Logging
```python
import djinn
djinn.set_debug_level('interception')  # See capture
djinn.set_debug_level('shape_inference')  # See shape calculations
djinn.set_debug_level('execution')  # See remote calls
```

### Common Issues & Solutions

| **Symptom** | **Likely Cause** | **Solution** |
|-------------|------------------|--------------|
| "Shape inference failed" | Unsupported operation | Check logs for op name, add to shape rules |
| "Circuit breaker tripped" | Repeated shape failures | Simplify model or disable shape inference |
| "Remote execution timeout" | Network/GPU overload | Increase timeout or use local execution |
| High first-run latency | Cold compilation | Warm up with dummy inputs |
| Memory errors | Cache overflow | Reduce cache size or enable pressure handler |

### Performance Profiling
```python
stats = djinn.get_profiler().get_stats()
print(f"Capture time: {stats['capture_ms']}ms")
print(f"Schedule time: {stats['schedule_ms']}ms")  
print(f"Execution time: {stats['execution_ms']}ms")
print(f"Cache hit rate: {stats['cache_hit_rate']}%")
```

---

## FAQ

**Q: Do I need to modify my model code?**  
A: No, just change the device to `remote_accelerator:0`.

**Q: What's the typical performance overhead?**  
A: **30-400ms** for cached model execution (vs 500-4000ms old system). First request includes one-time registration overhead (1-100s depending on model size). GPU execution is **12% faster than PyTorch** due to optimized memory management.

**Q: Does it work with HuggingFace Transformers?**  
A: Yes, the device compatibility layer handles standard frameworks. vmap operations (used for attention masking) are automatically supported.

**Q: Can I use torch.vmap with LazyTensors?**  
A: Yes, it works automatically. No code changes or configuration needed.

**Q: How do tuple operations like split() work?**  
A: Tuple operations return `LazyTuple`, which preserves laziness. Only accessed elements materialize, enabling efficient execution. For example, `chunks = x.split(300)` returns a LazyTuple, and only `chunks[0].cpu()` materializes the first chunk.

**Q: Can I mix Djinn with regular PyTorch tensors?**  
A: Yes, mixed operations automatically materialize as needed.

**Q: How do I know if Djinn is actually being used?**  
A: Enable debug logging to see interception and execution.

**Q: What happens if the remote GPU fails?**  
A: Automatic failover to another GPU or local execution.

---

## Next Steps

- **Quick Start**: Try the [example notebooks](examples/)
- **Integration**: See [docs/2_FRONTEND_IMPLEMENTATION.md](2_FRONTEND_IMPLEMENTATION.md)
- **Architecture**: Read [docs/1_ARCHITECTURE.md](1_ARCHITECTURE.md) for internals
- **Contributing**: Check [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Contact & Citation

**Project Lead**: Jaewan Hong (jaewan@berkeley.edu)  
**Repository**: https://github.com/jaewan/Djinn

```bibtex
@inproceedings{hong2025djinn,
  title={Lost in Translation: The Search for Meaning in 
         Network-Attached AI Accelerator Disaggregation},
  author={Hong, Jaewan and Qiao, Yifan and Ponnapalli, Soujanya and 
          Liu, Shu and Aguilera, Marcos K. and Liu, Vincent and 
          Rossbach, Christopher J. and Stoica, Ion},
  booktitle={Proceedings of HotNets '25},
  year={2025}
}
```

---