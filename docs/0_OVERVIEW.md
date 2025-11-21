**Status**: Production Ready (v2.3.10)
**Last Updated**: November 21, 2025
**Target Audience**: ML Engineers, Infrastructure Teams, Researchers

---

## TL;DR

Djinn is a **Distributed Tensor Operating System** that transforms GPU disaggregation from a hardware challenge into a transparent, high-performance framework-level solution. By operating at the ML framework layer, Djinn captures semantic intent (prefill vs decode phases, KV cache management) to enable intelligent GPU sharing—without requiring application code changes.

**Key features:**
- **Unified VMU**: Dual-lifecycle memory management with watermark-based allocation
- **Ghost Interception**: Zero-memory model loading with server-side weight management
- **Output Skeletonization**: Lazy tensor materialization with API transparency
- **Distributed GC**: Session-scoped memory management with heartbeat monitoring

**Key Value:** Transform 55-60% idle GPU capacity into usable compute through semantic-aware resource management. **10.7x faster** than graph-based execution with **99.7% network reduction**.

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

### How It Works: Distributed Tensor Operating System

```python
# Option A: Traditional device-based interception (default)
import djinn
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
model = model.to('remote_accelerator:0')  # Device-based interception
inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
output = model(**inputs)  # Remote execution with lazy outputs
result = output.logits.cpu()  # Triggers on-demand materialization

# Option B: Ghost interception (requires explicit configuration)
import djinn
djinn.config.intercept_huggingface = True  # Enable ghost interception

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2-xl")  # Automatic ghost interception
inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
output = model(**inputs)  # Remote execution with lazy outputs
result = output.logits.cpu()  # Triggers on-demand materialization
```

**What happens under the hood:**
1. **Ghost Interception**: HuggingFace loading creates zero-memory "ghost" model on meta device
2. **Server Registration**: Model weights downloaded server-side, cached in Unified VMU
3. **Lazy Execution**: Forward pass executed remotely with output skeletonization
4. **On-Demand Materialization**: Tensors pulled from server heap only when accessed
5. **Session GC**: Distributed garbage collection prevents memory leaks

**Key v2.3 Features:**
- **Unified VMU**: Watermark-based memory management prevents fragmentation
- **Output Skeletonization**: Full API transparency with lazy materialization
- **Capability Interlock**: Safe fallback logic prevents resource exhaustion
- **Distributed GC**: Session-scoped memory management with heartbeat monitoring

---

## v2.3 Core Innovations

### 1. Unified VMU: Zero-Fragmentation Memory Management

**Problem Solved**: LLM memory fragmentation during auto-regressive generation.

**Solution**: Dual-lifecycle memory management with persistent watermark:
- **Persistent Section**: KV cache and weights (never freed between requests)
- **Volatile Section**: Activations (reset after each request)
- **Watermark Pattern**: Moves boundary up for persistent allocations, resets volatile to watermark

**Benefits**:
- Zero fragmentation through byte-aligned allocations
- Efficient KV cache growth without reallocation
- **Prevents OOM** during prefill → decode transitions

### 2. Ghost Interception: Zero-Client Memory Loading

**Problem Solved**: Model weights unnecessarily downloaded to client memory.

**Solution**: Hooks HuggingFace `from_pretrained()` to create server-side only models:
- Client gets "ghost" model on meta device (zero memory)
- Weights downloaded directly to server VMU
- Execution delegates to server-side cached model

**Benefits**:
- **"Data never touches client until requested"**
- Eliminates client memory pressure
- Seamless HuggingFace integration

### 3. Output Skeletonization: Lazy Materialization

**Problem Solved**: Full tensor transfer even when only partial results needed.

**Solution**: Returns structured outputs with RemoteRefStubs:
- Preserves dict/tuple/list structure
- Tensors replaced with lightweight references
- DMA pull only when tensor accessed

**Benefits**:
- API transparency (works like regular PyTorch)
- Massive bandwidth savings for partial access
- On-demand materialization from server heap

### 4. Distributed GC: Session-Safe Memory Management

**Problem Solved**: Memory leaks when clients crash/disconnect.

**Solution**: Heartbeat-monitored session leases:
- All heap allocations tagged with session ID
- Automatic cleanup on heartbeat timeout
- Reference counting prevents use-after-free

**Benefits**:
- No memory leaks in production
- Safe concurrent multi-tenant operation
- Automatic resource reclamation

### 5. Capability Interlock: Safe Fallback Logic

**Problem Solved**: Resource exhaustion during emergency fallback.

**Solution**: Pre-execution resource auditing:
- Estimates model size and materialization overhead
- Checks available RAM against safety margins
- Gracefully fails rather than crashing system

**Benefits**:
- Prevents "crash-on-fallback" scenarios
- Safe degradation under resource pressure
- Clear error messages for capacity planning

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

**Key Innovations (v2.3):**
- **Unified VMU**: Dual-lifecycle memory management prevents fragmentation
- **Ghost Interception**: Zero-client-memory model loading
- **Output Skeletonization**: Lazy materialization with API transparency
- **Meta-Simulator**: Cached memory planning eliminates simulation overhead

**Network Optimization**: Binary protocol + lazy outputs achieve **99.7% network reduction**:

| **Scenario** | **Latency** | **What Happens** |
|-------------|-------------|------------------|
| **Ghost Loading** (HuggingFace) | <1s | Model weights downloaded server-side only |
| **Cached Execution** | 31-81ms | Model fingerprint + inputs sent, direct execution |
| **GPU Execution** | 30.82ms | **12% faster than PyTorch** (VMU optimization) |
| **Lazy Materialization** | 5-50ms | Tensors pulled on-demand from server heap |
| **Session Management** | <1ms | Distributed GC with heartbeat monitoring |

### Performance Envelope

**Best Case** (small model, cached, v2.3):
- **31ms** total latency (vs 868ms old graph-based)
- **28x faster** than old system
- **99.7% network reduction** (fingerprint + lazy outputs)
- **Zero client memory** for model loading

**Typical Case** (GPT-2-XL, cached, v2.3):
- **81ms** total latency (vs 3.8s old system)
- **47x faster** than old system
- **99.7% network reduction** (fingerprint + lazy outputs)
- GPU execution: **30.82ms** (12% faster than PyTorch)

**Memory Efficiency**:
- **Zero fragmentation**: VMU watermark prevents memory waste
- **Lazy materialization**: Outputs accessed on-demand only
- **Session GC**: Automatic cleanup prevents memory leaks
- **Capability interlock**: Prevents fallback-induced crashes

*Note: Benchmarks from real workloads. v2.3 delivers 10-50x improvement over v1.0 graph-based execution.*

---

## Architecture Overview

Djinn v2.3.10 implements a **Distributed Tensor Operating System** with four integrated layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT SIDE (Thin)                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. GHOST INTERCEPTION                               │   │
│  │     • Hooks HuggingFace from_pretrained()            │   │
│  │     • Creates zero-memory "ghost" models             │   │
│  │     • Server-side weight management                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                        │                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  2. CAPABILITY ENGINE                                │   │
│  │     • Resource auditing for safe fallback            │   │
│  │     • Prevents crash-on-fallback scenarios           │   │
│  │     • RAM availability checking                       │   │
│  └──────────────────────────────────────────────────────┘   │
│                        │                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  3. LAZY REFERENCE ENGINE                            │   │
│  │     • Receives skeletonized outputs                  │   │
│  │     • On-demand DMA pulls from server                │   │
│  │     • API transparency preservation                  │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┼─────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────────────┐
│                    SERVER SIDE (The Kernel)                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  4. SESSION MANAGER (Distributed GC)                │   │
│  │     • Heartbeat-monitored session leases              │   │
│  │     • Automatic cleanup on disconnect                 │   │
│  │     • Reference counting for safety                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                        │                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  5. UNIFIED VMU (Thread-Safe)                        │   │
│  │     • Watermark-based dual-lifecycle memory          │   │
│  │     • Persistent (KV/weights) + Volatile (activations)│   │
│  │     • Zero fragmentation through alignment            │   │
│  └──────────────────────────────────────────────────────┘   │
│                        │                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  6. META-SIMULATOR (Cached)                          │   │
│  │     • Memory planning via meta-device tracing        │   │
│  │     • LRU plan cache eliminates simulation overhead   │   │
│  │     • Input shape bucketing for cache efficiency     │   │
│  └──────────────────────────────────────────────────────┘   │
│                        │                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  7. HYBRID EXECUTOR                                  │   │
│  │     • Slab-based execution with stream locking       │   │
│  │     • Output skeletonization with lazy materialization│   │
│  │     • Two-stream pipelining (compute + transfer)     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Principles (v2.3):**
- **Distributed OS**: Treats remote GPUs as a kernel extension with automatic memory management
- **Zero Data Movement**: "The Data never touches the Client until requested"
- **Memory-First Architecture**: VMU solves fragmentation before it occurs
- **Session-Safe GC**: Prevents memory leaks in distributed environment
- **API Transparency**: Full PyTorch compatibility with lazy evaluation

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

**Device-Based Path (Default)**:
1. **Device Interception**: `.to('remote_accelerator:0')` creates LazyTensor parameters
2. **Model Registration**: EnhancedModelManager registers model with server
3. **Lazy Execution**: Forward pass creates LazyTensor outputs with semantic hints
4. **Server Execution**: Model cache executes directly with hint-guided optimization

**Ghost Interception Path (Optional)**:
1. **Hook Installation**: `from_pretrained()` automatically creates ghost model
2. **Server-Side Loading**: Weights downloaded directly to server VMU
3. **Zero Client Memory**: Client gets meta-device wrapper, no weight transfer
4. **Lazy Execution**: Same as device-based path but with "Data never touches client"

**Both Paths**:
- **Semantic Hints**: Scheduler analyzes SRG client-side, extracts hints
- **Model Cache**: Server executes via cached models (no graph reconstruction)
- **Lazy Outputs**: RemoteRefStubs provide API transparency with on-demand pulls
- **47x Performance**: 303x faster than v1.0 graph-based execution

---

## Implementation Maturity

### ✅ Production Ready (v2.3.10)
- **Unified VMU**: Dual-lifecycle memory management with zero fragmentation
- **Ghost Interception**: Zero-client-memory HuggingFace model loading
- **Output Skeletonization**: Lazy materialization with API transparency
- **Distributed GC**: Session-scoped memory management with heartbeat monitoring
- **Capability Interlock**: Safe fallback logic preventing resource exhaustion
- **Meta-Simulator**: Cached memory planning via meta-device tracing
- **Hybrid Executor**: Slab-based execution with stream locking
- **PyTorch operation interception** (~95% of common ops)
- **LazyTensor graph construction** with LazyTuple support
- **Model cache system** with fingerprinting and registration
- **TCP-based remote execution** with connection pooling
- **Device compatibility layer** for seamless framework integration

### ⚠️ Beta Quality
- Multi-tenant fairness algorithms
- Advanced semantic pattern recognition
- Cross-framework support (JAX, TensorFlow)
- Global cluster scheduling
- Real-time performance adaptation

### 🔬 Experimental
- TensorRT compilation integration
- TorchScript optimization paths
- Hardware-accelerated serialization
- Multi-GPU transaction coordination

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
4. **Model registration**: Explicit registration required before execution
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
A: **31-81ms** for cached model execution with lazy outputs. Ghost loading is <1s, GPU execution is **12% faster than PyTorch** due to VMU optimization. **47x faster** than v1.0 graph-based execution.

**Q: Does it work with HuggingFace Transformers?**
A: Yes, with **Ghost Interception** - models load with zero client memory. `from_pretrained()` automatically creates server-side cached models. Full API compatibility with lazy materialization.

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