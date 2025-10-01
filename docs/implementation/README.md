# Genie Implementation Documentation

Comprehensive implementation documentation for contributors to the Genie framework.

## Quick Navigation

### Essential Reading (Start Here)
1. **[INDEX](00-INDEX.md)** - Documentation overview and structure
2. **[Architecture Overview](01-architecture-overview.md)** - System design and principles
3. **[Contributor Guide](11-contributor-guide.md)** - How to contribute

### Core Components
- **[Device Layer](02-device-layer.md)** - Backend registration and device management
- **[LazyTensor](03-lazy-tensor.md)** - Deferred execution and semantic capture
- **[Dispatcher](04-dispatcher.md)** - Operation interception

### Runtime & Transport
- **[Runtime Transport](05-runtime-transport.md)** - Python transport coordination
- **[Semantic Layer](06-semantic-layer.md)** - Semantic analysis and profiling
- **[Pattern Recognition](07-pattern-recognition.md)** - Pattern matching algorithms
- **[Scheduler & Optimizer](08-scheduler-optimizer.md)** - Workload optimizations
- **[C++ Data Plane](09-data-plane-cpp.md)** - DPDK zero-copy implementation

### Cluster Management (New)
- **[Cluster Documentation](../../cluster/)** - Complete cluster management guide
  - **[Cluster Init Index](../../cluster/CLUSTER_INIT_INDEX.md)** - Navigation hub for all cluster docs
- **[Implementation Plan](../CLUSTER_INIT_IMPLEMENTATION_PLAN.md)** - Detailed implementation guide
- **[Quick Start](../CLUSTER_INIT_QUICK_START.md)** - Developer quick start guide

### Reference & Contributing
- **[Quick Reference](10-quick-reference.md)** - Common tasks and APIs
- **[Contributor Guide](11-contributor-guide.md)** - How to contribute
- **[Architecture FAQ](../ARCHITECTURE_FAQ.md)** - Common questions
- **[Refactoring Notes](../../REFACTORING_NOTES.md)** - Recent changes

## Documentation Status

| # | Document | Lines | Status |
|---|----------|-------|--------|
| 01 | Architecture Overview | ~400 | ✅ |
| 02 | Device Layer | ~450 | ✅ |
| 03 | LazyTensor | ~800 | ✅ |
| 04 | Dispatcher | ~500 | ✅ |
| 05 | Runtime Transport | ~1100 | ✅ |
| 06 | Semantic Layer | ~1300 | ✅ |
| 07 | Pattern Recognition | ~1150 | ✅ |
| 08 | Scheduler & Optimizer | ~1440 | ✅ |
| 09 | C++ Data Plane | ~2380 | ✅ |
| 10 | Quick Reference | ~370 | ✅ |
| 11 | Contributor Guide | ~200 | ✅ |

**Total**: 10,090 lines of implementation documentation

## Key Concepts

### Semantic Preservation
Genie operates at the ML framework layer to preserve rich application semantics that low-level approaches lose.

### Lazy Execution
Operations create LazyTensors instead of executing immediately, enabling global optimization.

### Pluggable Architecture
Clear separation between Frontend (capture), SRG (semantics), and Backend (execution).

## Code Examples

### Basic Usage

**With Cluster Initialization** (Recommended):
```python
import genie
import torch

# Initialize connection to GPU cluster (one-time setup)
await genie.init(master_addr='gpu-server.example.com')

# Use remote accelerators transparently
x = torch.randn(1000, 1000, device="remote_accelerator:0")
y = torch.matmul(x, x)  # Builds graph, doesn't execute
z = y.relu()            # Continues building graph

# Materialize when needed
result = z.cpu()  # Executes entire graph

# Clean shutdown
await genie.shutdown()
```

**Legacy Usage** (Manual setup):
```python
import genie
import torch

# Enable remote execution
genie.set_lazy_mode(True)

# Tensors on remote_accelerator become LazyTensors
x = torch.randn(1000, 1000, device="remote_accelerator:0")
y = torch.matmul(x, x)
z = y.relu()
result = z.cpu()
```

### Checking Status
```python
from genie import get_capture_stats

stats = get_capture_stats()
print(f"Operations registered: {stats['summary']['dispatcher_registered_ops']}")
print(f"Operations captured: {stats['dispatcher']['operation_count']}")
```

## Architecture at a Glance

```
User Code (PyTorch)
        ↓
   Device Layer (device.py)
        ↓
   LazyTensor (lazy_tensor.py) + Dispatcher (enhanced_dispatcher.py)
        ↓
   FX Graph Builder (fx_graph_builder.py)
        ↓
   Pattern Matcher (patterns/)
        ↓
   Executor (executor.py)
        ↓
   Result (torch.Tensor)
```

## Contributing

To contribute documentation:

1. Read existing docs to understand style and format
2. Use markdown with code examples
3. Include architecture diagrams (ASCII art)
4. Add cross-references to related docs
5. Update this README with your new docs

## File Organization

```
docs/implementation/
├── README.md (this file)
├── 00-INDEX.md
├── 01-architecture-overview.md
├── 02-device-layer.md
├── 03-lazy-tensor.md
├── 04-dispatcher.md
├── 05-executor.md
├── 06-fx-integration.md
├── 07-pattern-recognition.md
├── 08-semantic-metadata.md
├── 09-development-guide.md
└── 10-api-reference.md
```

## External References

- **HotNets'25 Paper**: `../../.kiro/HotNets25.tex`
- **Refactoring Docs**: `../../REFACTORING_COMPLETE.md`
- **Test Suite**: `../../tests/`
- **Source Code**: `../../genie/`

## Recent Updates

### October 2025: Cluster Initialization Feature
- 🆕 **Cluster Management**: Comprehensive cluster initialization system
  - Single `genie.init()` call for transparent cluster connection
  - Automatic network backend discovery (TCP/DPDK/RDMA)
  - Built-in GPU monitoring and health checks
  - Heartbeat-based failure detection
  - Complete implementation plan with step-by-step guide
  - See: [Cluster Init Documentation](../CLUSTER_INIT_INDEX.md)

### September 2025: Completed Refactorings
- ✅ **Refactoring #1**: Consolidated error handling (60/60 tests)
- ✅ **Refactoring #3**: Unified graph representation with PyTorch FX (80+ tests)
- ✅ **Refactoring #4**: Async-first transport with ThreadPoolExecutor (14/14 tests)
- ✅ **Refactoring #5**: Pattern matching service extraction (27/27 tests)

### Highlights
- **Async Transport**: All blocking ctypes calls now use ThreadPoolExecutor
  - Event loop never blocked during C++ operations
  - ~40% throughput improvement with parallel workers
  - 14 comprehensive async tests
- **FX Graph Migration**: Unified graph representation using PyTorch FX
  - `FXGraphAdapter` for seamless graph access
  - Backward compatible with `ComputationGraph`
  - Better integration with PyTorch ecosystem
- **Pattern Matching**: Dependency injection for pattern matchers
  - `IPatternMatcher` interface
  - Multiple implementations (NetworkX, Simplified, Composite)
  - Injectable into `SemanticAnalyzer`

**See**: [REFACTORING_PLAN.md](../../REFACTORING_PLAN.md) for details

## Getting Help

- Check tests in `tests/` for usage examples
- Read source code comments
- File issues on GitHub
- Join community discussions

## License

Documentation licensed under same terms as Genie source code.

---

**Last Updated**: 2025-09-30  
**Contributors**: Core Genie team  
**Status**: Active development