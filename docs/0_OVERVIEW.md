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
3. **Schedule**: Cost model determines optimal GPU placement
4. **Execute**: Graph runs on selected GPUs with minimal data movement

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

### Expected Overhead

| **Scenario** | **Overhead** | **When It Occurs** |
|-------------|--------------|-------------------|
| First execution | 50-200ms | Graph compilation, shape inference |
| Warm execution | 10-20ms | Graph cached, remote dispatch only |
| Local fallback | 5-10ms | When remote unavailable |
| Shape inference failure | 100ms+ | Falls back to materialization |

### Performance Envelope

**Best Case** (warm cache, simple model):
- 5-10ms overhead
- 2-3x better GPU utilization
- 90% reduction in KV cache transfers

**Typical Case** (transformers, batch inference):
- 10-20ms overhead
- 30-40% better GPU utilization
- Amortized over batch processing

**Worst Case** (complex shapes, cold start):
- 100-200ms overhead
- May trigger local fallback
- Shape inference circuit breaker activated

*Note: All measurements hardware and workload dependent. Benchmark your specific use case.*

---

## Architecture Overview

Djinn uses a clean four-layer architecture:

```
Application (PyTorch) → No changes required
        ↓
[1] Frontend Layer → Intercepts operations, builds graph
        ↓
[2] Scheduler Layer → Analyzes semantics, plans execution
        ↓
[3] Server Layer → Coordinates distributed execution
        ↓
[4] Backend Layer → Executes on GPUs
```

**Key Design Principles:**
- **Transparency**: No application changes required
- **Semantic-Driven**: ML-aware optimization decisions
- **Fault Tolerant**: Automatic fallback and recovery
- **Pluggable**: Swappable schedulers and backends

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

1. **Graph Construction**: Operations recorded as LazyTensor DAG
2. **Semantic Analysis**: Patterns detected (attention, normalization, etc.)
3. **Scheduling**: Optimal GPU selected based on cost model
4. **Remote Execution**: Entire graph executed on GPU
5. **Result Return**: Only final tensor transferred back

---

## Implementation Maturity

### ✅ Production Ready
- PyTorch operation interception (~95% of common ops)
- LazyTensor graph construction
- TCP-based remote execution
- Device compatibility layer
- Basic caching (graph, GPU memory)

### ⚠️ Beta Quality
- Semantic pattern recognition (transformers, CNNs)
- Shape inference (has failure fallbacks)
- Multi-tenant fairness
- Memory pressure handling
- Differential graph updates

### 🔬 Experimental
- TensorRT compilation
- Global cluster scheduling
- Advanced memory management
- Cross-framework support

---

## Known Limitations

### Functional Limitations

1. **In-place operations** converted to out-of-place (5% memory overhead)
2. **Dynamic shapes** may force materialization (loses optimization)
3. **Mixed device operations** trigger early materialization
4. **Custom operations** require manual registration
5. **Debugging complexity** increases with remote execution

### Performance Limitations

1. **Cold start**: 50-200ms for first execution
2. **Shape inference**: May timeout after 500ms (circuit breaker)
3. **Network bandwidth**: Becomes bottleneck for large models
4. **Cache misses**: Cause compilation overhead

### When Things Go Wrong

**Shape Inference Failures:**
- Circuit breaker activates after 10 consecutive failures
- Falls back to eager materialization
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
A: 10-20ms for warm execution, 50-200ms for cold start. Varies by workload.

**Q: Does it work with HuggingFace Transformers?**  
A: Yes, the device compatibility layer handles standard frameworks.

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