# Quick Reference Guide

## Overview

Quick reference for common tasks and APIs in Genie. For detailed documentation, see the individual component docs.

## Quick Start

### Basic Usage

```python
import genie
import torch

# Enable Genie
genie.set_lazy_mode(True)

# Create tensors on remote accelerator
x = torch.randn(1000, 1000, device="remote_accelerator:0")
y = torch.matmul(x, x)
z = y.relu()

# Materialize (executes on remote GPU)
result = z.cpu()
```

### Check Status

```python
from genie import get_capture_stats

stats = get_capture_stats()
print(f"Operations captured: {stats['dispatcher']['operation_count']}")
print(f"Graph nodes: {stats['graph']['total_nodes']}")
```

## Runtime Transport

### Initialize Transport

```python
from genie.runtime.transport_coordinator import initialize_transport

coordinator = await initialize_transport(
    node_id="node_a",
    config={
        'data_plane': {'local_ip': '192.168.1.100'},
        'control_plane': {'port': 5555}
    }
)
```

### Send Tensor

```python
tensor = torch.randn(1024, 1024, device='cuda')
transfer_id = await coordinator.send_tensor(tensor, target_node="node_b")
```

### DPDK Operations

```python
from genie.runtime.dpdk_bindings import get_dpdk

dpdk = get_dpdk()
dpdk.init_eal()
pool = dpdk.create_mempool("pool", n_mbufs=8192)
mbuf = dpdk.alloc_mbuf(pool)
```

## Semantic Analysis

### Analyze Graph

```python
from genie.semantic.analyzer import SemanticAnalyzer
from genie.semantic.graph_handoff import build_graph_handoff

analyzer = SemanticAnalyzer()
handoff = build_graph_handoff()
profile = analyzer.analyze_graph(handoff.graph)

print(f"Workload: {profile.workload_type.value}")
print(f"Patterns: {[p.pattern_name for p in profile.patterns]}")
```

### Phase Detection

```python
from genie.semantic.phase_detector import get_phase_detector

detector = get_phase_detector()
phase = detector.detect_phase(operation="matmul", inputs=[...])
print(f"Current phase: {phase.value}")

stats = detector.get_phase_statistics()
```

### Pattern Matching

```python
from genie.semantic.pattern_registry import PatternRegistry

registry = PatternRegistry()
matches = registry.match_patterns(graph)

for match in matches:
    print(f"{match.pattern_name}: {match.confidence:.2f}")
```

## Optimization

### Optimize Graph

```python
from genie.semantic.optimizer import SemanticOptimizer

optimizer = SemanticOptimizer()
opt_graph, plan = optimizer.optimize(fx_graph, profile)

print(f"Optimizations: {[o.value for o in plan.optimizations]}")
```

### Create Schedule

```python
from genie.semantic.scheduling import Scheduler

scheduler = Scheduler()
schedule = scheduler.create_schedule(opt_graph, plan)

print(f"Stages: {schedule.total_stages}")
```

### Place Nodes

```python
from genie.semantic.placement import PlacementEngine

engine = PlacementEngine()
placement = engine.create_placement_plan(opt_graph, plan)

print(f"Devices used: {placement.total_devices_used}")
```

## Common Patterns

### LLM Inference

```python
# Model setup
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Enable Genie
genie.set_lazy_mode(True)
model = model.to("remote_accelerator:0")

# Prefill phase
inputs = tokenizer("Hello, world!", return_tensors="pt")
inputs = {k: v.to("remote_accelerator:0") for k, v in inputs.items()}

# Decode phase (automatic KV cache co-location)
outputs = model.generate(**inputs, max_new_tokens=50)
```

### Vision Model

```python
from torchvision.models import resnet50

model = resnet50(pretrained=True)
model = model.to("remote_accelerator:0")

# Automatic CNN pipelining
image = torch.randn(1, 3, 224, 224, device="remote_accelerator:0")
features = model(image)
```

### Multi-Modal

```python
class VQAModel(nn.Module):
    def __init__(self):
        self.vision = vit_base_patch16_224()
        self.language = BertModel.from_pretrained("bert-base")
        self.fusion = CrossAttentionBlock()
    
    def forward(self, image, text):
        # Parallel modality processing (automatic)
        v = self.vision(image)
        t = self.language(text)
        # JIT fusion transfer (automatic)
        return self.fusion(v, t)

model = VQAModel().to("remote_accelerator:0")
```

## Configuration

### Environment Variables

```bash
# Analysis
export GENIE_ANALYZER_CACHE=1
export GENIE_ANALYZER_SLOW_MS=100
export GENIE_ANALYZER_DEBUG=1

# Transport
export GENIE_DISABLE_CPP_DATAPLANE=0
export GENIE_CONTROL_PORT=5555
export GENIE_DATA_PORT=5556

# Reliability
export GENIE_RELIABILITY_MODE=custom  # or "kcp"
```

### Programmatic Config

```python
# Optimizer
optimizer = SemanticOptimizer(enable_all=False)
optimizer.enabled_optimizations[OptimizationType.KV_CACHE_COLOCATION] = True

# Scheduler
scheduler = PipelineScheduler(num_stages=4)

# Placement
engine = PlacementEngine(devices=[...])
```

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Get Statistics

```python
# Analyzer stats
report = analyzer.get_performance_report()

# Pattern stats
pattern_stats = analyzer.get_pattern_statistics()

# Transport stats
stats = coordinator.get_statistics()

# Optimizer stats
opt_stats = optimizer.get_optimization_stats()
```

### Inspect Graph

```python
from genie.core.graph import GraphBuilder

builder = GraphBuilder.current()
graph = builder.get_graph()

print(f"Nodes: {len(graph.nodes)}")
print(f"Edges: {len(graph.edges)}")

for node_id, node in graph.nodes.items():
    print(f"{node_id}: {node.operation}")
```

## File Locations

### Runtime Transport
- **Coordinator**: `genie/runtime/transport_coordinator.py`
- **Control Server**: `genie/runtime/control_server.py`
- **DPDK Bindings**: `genie/runtime/dpdk_bindings.py`
- **Async Bridge**: `genie/runtime/async_zero_copy_bridge.py`
- **GPU Memory**: `genie/runtime/gpu_memory.py`

### Semantic Layer
- **Analyzer**: `genie/semantic/analyzer.py`
- **Enhanced Analyzer**: `genie/semantic/enhanced_analyzer.py`
- **Pattern Registry**: `genie/semantic/pattern_registry.py`
- **Phase Detector**: `genie/semantic/phase_detector.py`
- **Optimizer**: `genie/semantic/optimizer.py`
- **Scheduler**: `genie/semantic/scheduling.py`
- **Placement**: `genie/semantic/placement.py`

### Patterns
- **Base**: `genie/patterns/base.py`
- **Advanced**: `genie/patterns/advanced_patterns.py`
- **FX Patterns**: `genie/patterns/fx_patterns.py`
- **DSL**: `genie/patterns/pattern_dsl.py`

## Common Issues

### Import Errors

**Issue**: `ModuleNotFoundError: No module named 'genie.runtime.dpdk_eal'`

**Solution**: Use `from genie.runtime.dpdk_bindings import get_dpdk`

---

**Issue**: `cannot import name 'set_lazy_mode' from 'genie.core.enhanced_dispatcher'`

**Solution**: Use `from genie.core.library import set_lazy_mode`

### Circular Imports

**Issue**: `ImportError: cannot import name 'X' from partially initialized module`

**Solution**: Use delayed imports in functions, not at module level

### DPDK Not Available

**Issue**: `DPDK libraries not available`

**Solution**: Install DPDK or run in fallback mode (automatic)

## Testing

### Run Specific Tests

```bash
# DPDK functionality
pytest tests/test_dpdk_bindings.py -v

# Semantic analysis
pytest tests/test_analyzer_*.py -v

# Pattern recognition
pytest tests/test_dynamo_patterns.py -v

# Phase detection
pytest tests/test_phase_detection.py -v

# Control plane
pytest tests/test_control_server.py -v
```

### Run All Tests

```bash
# All tests (fast)
pytest tests/ -k "not slow and not stress" -v

# With coverage
pytest tests/ --cov=genie --cov-report=html
```

## See Also

- [Architecture Overview](01-architecture-overview.md)
- [Runtime Transport](05-runtime-transport.md)
- [Semantic Layer](06-semantic-layer.md)
- [Pattern Recognition](07-pattern-recognition.md)
- [Scheduler & Optimizer](08-scheduler-optimizer.md)
- [Refactoring Summary](../REFACTORING_2025-09-30.md)

---

**Last Updated**: 2025-09-30  
**Maintainers**: Genie Core Team
