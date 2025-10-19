# Genie Refactoring Plan: Production-Ready Open-Source Release

**Document Version:** 1.0  
**Date:** 2024  
**Author:** Senior Engineering Team  
**Status:** Proposed Architecture

---

## Executive Summary

This document outlines a comprehensive refactoring of Genie from research prototype to production-ready, open-source disaggregated execution framework. The refactoring addresses critical technical debt while preserving the core innovation: semantic-aware scheduling for GPU disaggregation.

**Key Objectives:**
1. **Clean Architecture:** Reduce from 3 interception mechanisms to 2 well-defined layers
2. **Production Quality:** <1μs per-operation overhead, robust fault handling
3. **Open-Source Ready:** Clean APIs, comprehensive docs, community-friendly design
4. **Incremental Migration:** Support existing code during transition

**Timeline:** 20 weeks (5 months) to v1.0 production release

**Risk Level:** Medium - Core architecture changes, but with clear fallback paths

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Target Architecture](#2-target-architecture)
3. [Deployment Environment Strategy](#3-deployment-environment-strategy)
4. [Detailed Refactoring Plan](#4-detailed-refactoring-plan)
5. [API Design](#5-api-design)
6. [Testing Strategy](#6-testing-strategy)
7. [Performance Targets](#7-performance-targets)
8. [Migration Guide](#8-migration-guide)
9. [Open-Source Considerations](#9-open-source-considerations)
10. [Risk Assessment](#10-risk-assessment)
11. [Timeline & Milestones](#11-timeline--milestones)

---

## 1. Current State Analysis

### 1.1 Critical Technical Debt

#### Issue 1: Triple Interception System

**Current State:**
```python
# Three competing mechanisms:
1. Factory wrapping (interception.py) - ~20 functions manually wrapped
2. __torch_dispatch__ (lazy_tensor.py) - Monkey-patched onto class
3. torch.library (library.py) - PrivateUse1 registrations (mostly disabled)
```

**Problems:**
- Race conditions when multiple mechanisms trigger
- Maintenance burden (changes need 3x updates)
- Confusion about which mechanism handles what
- library.py registrations are redundant

**Impact:** High - Core system complexity, hard to debug

#### Issue 2: Non-Standard LazyTensor

**Current State:**
```python
class LazyTensor:  # Does NOT inherit from torch.Tensor
    def __torch_dispatch__(cls, ...):  # Monkey-patched later
```

**Problems:**
- `isinstance(x, torch.Tensor)` returns False → breaks many libraries
- Won't integrate with autograd, torch.compile, etc.
- PyTorch dispatcher doesn't recognize LazyTensor as a tensor subclass
- Manual property implementations (shape, dtype, device)

**Note:** The existing `__torch_dispatch__` implementation is correct. The issue is that PyTorch's dispatcher won't call it until LazyTensor properly inherits from `torch.Tensor` using `_make_subclass`.

**Impact:** High - Limits ecosystem compatibility

#### Issue 3: Three Graph Representations

**Current State:**
```python
1. LazyTensor.inputs chain (implicit DAG)
2. FX Graph (optional, ~80% coverage)
3. ComputationGraph (debugging/legacy)
```

**Problems:**
- No single source of truth
- Synchronization bugs when graphs diverge
- 3x memory overhead
- Scheduler doesn't know which to use

**Impact:** Medium - Adds complexity, memory overhead

#### Issue 4: Incomplete Scheduler

**Current State:**
- Semantic analysis exists but is basic
- No real scheduling algorithm (just local execution)
- No cost model implementation
- No placement optimization

**Impact:** High - Core paper contribution not implemented

### 1.2 What's Working Well

✅ **Semantic enrichment concept** - Phase detection, modality tagging  
✅ **Factory interception** - Captures creation ops correctly  
✅ **Basic execution** - Local fallback works reliably  
✅ **DPDK integration design** - Zero-copy architecture is sound  
✅ **Paper validation** - Core ideas proven in HotNets

---

## 2. Target Architecture

### 2.1 Architecture Principles

1. **Simplicity Through Standards** - Use PyTorch's official extension APIs
2. **Single Source of Truth** - One graph representation with fallback
3. **Fail-Safe Design** - Graceful degradation at every level
4. **Open-Source First** - API designed for community adoption

### 2.2 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Application                          │
│  model = MyLLM()                                            │
│  with genie.capture():                                      │
│      output = model(input)  # Transparent interception     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│         Interception Layer (2 Mechanisms)                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 1. Factory Interceptor (torch.randn, etc.)            │ │
│  │    - Wraps ~20 creation functions                     │ │
│  │    - Returns LazyTensor for remote_accelerator device │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 2. LazyTensor (torch.Tensor subclass)                 │ │
│  │    - __torch_dispatch__ for operations                │ │
│  │    - Automatic coverage of ALL PyTorch ops            │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│         Graph Builder (Hybrid Strategy)                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Primary: torch.fx.Graph                                │ │
│  │  - Covers ~80% of models (static control flow)        │ │
│  │  - Standard format, rich tooling ecosystem            │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Fallback: LazyTensor DAG                              │ │
│  │  - Covers remaining 20% (dynamic control flow)        │ │
│  │  - Always available, no tracer errors                 │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│         Semantic Analysis (Core Innovation)                  │
│  - Phase Detection: prefill/decode/forward/backward         │
│  - Modality Tagging: vision/text/audio/fusion              │
│  - Pattern Matching: attention, convolution, KV cache       │
│  - Cost Estimation: FLOPs, memory, intensity                │
│  - Dependency Analysis: co-location requirements            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│         Scheduler (Multi-Objective Optimization)             │
│  - Cost Model: compute + transfer + queueing                │
│  - Placement: ILP solver / greedy with refinement           │
│  - Co-location: KV cache + decoder on same device          │
│  - Pipelining: overlap compute and communication           │
│  - Dynamic Recomputation: recompute vs transfer under load  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│         Execution Backends (Tiered Strategy)                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Tier 1: Local Fallback (CPU/GPU)                      │ │
│  │  - Always available, zero setup                       │ │
│  │  - Validates correctness                              │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Tier 2: Simple Remote (HTTP/gRPC)                     │ │
│  │  - Single-node disaggregation                         │ │
│  │  - Easy deployment, good for testing                  │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Tier 3: Production (RDMA/UCX + DPDK)                  │ │
│  │  - Multi-node clusters                                │ │
│  │  - Zero-copy datapath                                 │ │
│  │  - Full disaggregation benefits                       │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Why This Architecture?

**Comparison with Alternatives:**

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Pure FX Graph | Standard format, tooling | Fails on dynamic control flow | Use as primary |
| Pure LazyTensor DAG | Always works | Non-standard format | Use as fallback |
| Both (Hybrid) | Best of both worlds | Slightly more complex | ✅ **CHOSEN** |

**Rationale:** 
- 80% of models work with FX (standardization benefit)
- 20% need dynamic support (can't abandon them)
- Hybrid gives us both coverage and standards compliance

---

## 3. Deployment Environment Strategy

### 3.1 Target Deployment Tiers

For open-source adoption, support **three deployment tiers** with increasing complexity:

#### Tier 1: Single-Machine Development (Priority: Essential)

**Use Case:** Developers testing locally, CI/CD, small models

**Infrastructure:**
```yaml
Hardware:
  - Single machine with 1+ GPUs
  - Standard Ethernet (1-10 Gbps)
  - No special networking hardware

Software:
  - Python 3.9+
  - PyTorch 2.0+
  - Standard PyPI packages
```

**Execution Mode:**
- Local GPU execution with simulated disaggregation
- HTTP-based communication (fallback)
- Focus: Correctness validation, API testing

**Why This Matters:** Lowers barrier to entry, enables rapid iteration

#### Tier 2: Small Cluster (Priority: High)

**Use Case:** Research labs, small startups, paper reproduction

**Infrastructure:**
```yaml
Hardware:
  - 2-8 GPU nodes
  - 10-40 Gbps Ethernet (standard datacenter)
  - Optional: RDMA-capable NICs (RoCE)

Software:
  - Container orchestration (Docker/Kubernetes)
  - gRPC for control plane
  - Optional: UCX for RDMA
```

**Execution Mode:**
- Real disaggregation across nodes
- gRPC for scheduling and control
- TCP for data transfer (or RDMA if available)
- Focus: Validate disaggregation benefits

**Why This Matters:** Target environment for most academic users

#### Tier 3: Production Datacenter (Priority: Medium)

**Use Case:** Cloud providers, large ML infrastructure teams

**Infrastructure:**
```yaml
Hardware:
  - 10+ GPU nodes
  - 100+ Gbps InfiniBand/RoCE
  - GPUDirect RDMA support
  - Dedicated DPDK NICs

Software:
  - Kubernetes with GPU scheduling
  - UCX for RDMA
  - DPDK for zero-copy
  - Prometheus for monitoring
```

**Execution Mode:**
- Full disaggregation with zero-copy
- Global scheduler for multi-tenant optimization
- Fault tolerance with lineage-based recovery
- Focus: Production performance and reliability

**Why This Matters:** Demonstrates full paper vision

### 3.2 Recommended Phased Approach

```
Phase 1 (Weeks 1-4): ✅ Core Architecture - COMPLETED
Phase 2 (Weeks 5-8):  Tier 1 + Hybrid Graph - IN PROGRESS
Phase 3 (Weeks 9-16): Tier 2 - Multi-node with gRPC+TCP
Phase 4 (Weeks 17-24): Tier 3 - RDMA+DPDK optimization
```

**Justification:**
1. Tier 1 enables immediate community testing (wide adoption)
2. Tier 2 validates disaggregation benefits (research validation)
3. Tier 3 demonstrates production viability (long-term vision)

### 3.3 Deployment Strategy Document

**File: `docs/deployment.md`**

```markdown
# Genie Deployment Guide

## Quick Start (Tier 1 - Local Development)

```bash
pip install genie-pytorch

# Run locally (no cluster needed)
import genie
with genie.capture():
    output = model(input)
```

## Small Cluster Setup (Tier 2)

### Prerequisites
- 2+ machines with GPUs
- Docker or Kubernetes
- Network connectivity between nodes

### Setup
```bash
# On server nodes
docker run -d genie/server:latest --port 8888

# In your code
import genie
genie.init(cluster_config='cluster.yaml')
```

## Production Deployment (Tier 3)

### Prerequisites
- RDMA-capable network (InfiniBand or RoCE)
- GPUDirect RDMA support
- DPDK-compatible NICs

### Setup
See `docs/production-deployment.md` for detailed guide
```

---

## 4. Detailed Refactoring Plan

### Phase 2: Hybrid Graph Builder (Weeks 5-8)

#### 2.1 Unified Graph Interface

**File: `genie/core/graph_interface.py` (NEW)**

```python
"""
Unified graph interface supporting both FX and LazyTensor DAG.

Provides a common API that abstracts over the underlying representation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import torch.fx as fx


class GraphNode(ABC):
    """Abstract node in computation graph."""
    
    @property
    @abstractmethod
    def id(self) -> str:
        """Unique node identifier."""
        pass
    
    @property
    @abstractmethod
    def operation(self) -> str:
        """Operation name (e.g., 'aten::add')."""
        pass
    
    @property
    @abstractmethod
    def inputs(self) -> List['GraphNode']:
        """Input nodes."""
        pass
    
    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Semantic metadata."""
        pass


class Graph(ABC):
    """Abstract computation graph."""
    
    @abstractmethod
    def nodes(self) -> List[GraphNode]:
        """Get all nodes in topological order."""
        pass
    
    @abstractmethod
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID."""
        pass
    
    @abstractmethod
    def topological_sort(self) -> List[GraphNode]:
        """Get nodes in execution order."""
        pass
    
    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Backend type: 'fx' or 'lazy_dag'."""
        pass


class FXGraphAdapter(Graph):
    """Adapter for torch.fx.Graph."""
    
    def __init__(self, fx_graph: fx.Graph):
        self.fx_graph = fx_graph
        self._nodes_cache = None
    
    def nodes(self) -> List[GraphNode]:
        if self._nodes_cache is None:
            self._nodes_cache = [
                FXNodeAdapter(node) 
                for node in self.fx_graph.nodes
                if node.op == 'call_function'
            ]
        return self._nodes_cache
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        for node in self.nodes():
            if node.id == node_id:
                return node
        return None
    
    def topological_sort(self) -> List[GraphNode]:
        # FX graph is already in topological order
        return self.nodes()
    
    @property
    def backend_type(self) -> str:
        return 'fx'


class FXNodeAdapter(GraphNode):
    """Adapter for torch.fx.Node."""
    
    def __init__(self, fx_node: fx.Node):
        self.fx_node = fx_node
    
    @property
    def id(self) -> str:
        return self.fx_node.name
    
    @property
    def operation(self) -> str:
        return str(self.fx_node.target)
    
    @property
    def inputs(self) -> List[GraphNode]:
        return [
            FXNodeAdapter(arg)
            for arg in self.fx_node.args
            if isinstance(arg, fx.Node)
        ]
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self.fx_node.meta.get('semantic', {})


class LazyDAGAdapter(Graph):
    """Adapter for LazyTensor DAG."""
    
    def __init__(self, root_tensor):
        from .lazy_tensor import LazyTensor
        self.root = root_tensor
        self._nodes_cache = None
    
    def nodes(self) -> List[GraphNode]:
        if self._nodes_cache is None:
            self._nodes_cache = self._collect_nodes()
        return self._nodes_cache
    
    def _collect_nodes(self) -> List[GraphNode]:
        """Collect all nodes reachable from root."""
        from .lazy_tensor import LazyTensor
        
        visited = set()
        result = []
        
        def visit(tensor):
            if not isinstance(tensor, LazyTensor):
                return
            if id(tensor) in visited:
                return
            visited.add(id(tensor))
            
            # Visit inputs first (post-order)
            for inp in tensor.inputs:
                visit(inp)
            
            result.append(LazyDAGNodeAdapter(tensor))
        
        visit(self.root)
        return result
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        for node in self.nodes():
            if node.id == node_id:
                return node
        return None
    
    def topological_sort(self) -> List[GraphNode]:
        # Already collected in topological order
        return self.nodes()
    
    @property
    def backend_type(self) -> str:
        return 'lazy_dag'


class LazyDAGNodeAdapter(GraphNode):
    """Adapter for LazyTensor node."""
    
    def __init__(self, lazy_tensor):
        self.tensor = lazy_tensor
    
    @property
    def id(self) -> str:
        return str(self.tensor.tensor_id)
    
    @property
    def operation(self) -> str:
        return self.tensor.operation
    
    @property
    def inputs(self) -> List[GraphNode]:
        from .lazy_tensor import LazyTensor
        return [
            LazyDAGNodeAdapter(inp)
            for inp in self.tensor.inputs
            if isinstance(inp, LazyTensor)
        ]
    
    @property
    def metadata(self) -> Dict[str, Any]:
        # Metadata stored separately in registry
        try:
            from genie.semantic.metadata_registry import get_metadata_registry
            registry = get_metadata_registry()
            meta = registry.get_metadata(self.id)
            return meta.to_dict() if meta else {}
        except Exception:
            return {}
```

#### 2.2 Hybrid Graph Builder

**File: `genie/core/graph_builder.py` (NEW)**

```python
"""
Hybrid graph builder: Tries FX first, falls back to LazyTensor DAG.

This provides the "single source of truth" while maintaining compatibility
with models that have dynamic control flow.
"""

import torch
import torch.fx as fx
from typing import Optional, Any
import logging

from .graph_interface import Graph, FXGraphAdapter, LazyDAGAdapter
from .lazy_tensor import LazyTensor

logger = logging.getLogger(__name__)


class HybridGraphBuilder:
    """
    Builds computation graph using hybrid strategy.
    
    Strategy:
    1. Try torch.fx.symbolic_trace (covers ~80% of models)
    2. If that fails, use LazyTensor DAG (always works)
    
    Both representations are exposed through unified Graph interface.
    """
    
    def __init__(self):
        self.fx_graph: Optional[fx.Graph] = None
        self.fx_module: Optional[fx.GraphModule] = None
        self.use_fx = True
        
        # LazyTensor tracking
        self.root_tensor: Optional[LazyTensor] = None
        self.all_tensors = {}  # tensor_id -> LazyTensor
    
    def build_from_model(self, model: torch.nn.Module, *args) -> Graph:
        """
        Build graph from model using hybrid strategy.
        
        Args:
            model: PyTorch model to trace
            *args: Example inputs
        
        Returns:
            Unified Graph interface
        """
        # Try FX first
        try:
            logger.info("Attempting FX symbolic trace...")
            self.fx_module = fx.symbolic_trace(model)
            self.fx_graph = self.fx_module.graph
            self.use_fx = True
            logger.info(f"✓ FX trace successful ({len(list(self.fx_graph.nodes))} nodes)")
            return FXGraphAdapter(self.fx_graph)
        
        except Exception as e:
            # FX failed - fall back to LazyTensor DAG
            logger.info(f"FX trace failed: {e}")
            logger.info("Falling back to LazyTensor DAG capture...")
            
            self.use_fx = False
            
            # Capture using LazyTensor
            output = model(*args)
            
            if not isinstance(output, LazyTensor):
                raise RuntimeError(
                    "Model output is not a LazyTensor. "
                    "Make sure tensors are on remote_accelerator device."
                )
            
            self.root_tensor = output
            logger.info(f"✓ LazyTensor DAG built successfully")
            return LazyDAGAdapter(self.root_tensor)
    
    def build_from_capture(self) -> Graph:
        """
        Build graph from captured LazyTensors.
        
        Used with context manager:
            with genie.capture():
                output = model(input)
            graph = builder.build_from_capture()
        """
        if self.root_tensor is None:
            raise RuntimeError("No LazyTensor captured")
        
        return LazyDAGAdapter(self.root_tensor)
    
    def add_operation(self, tensor: LazyTensor):
        """
        Register LazyTensor operation (called from LazyTensor.__init__).
        
        This tracks all tensors as they're created, enabling DAG construction.
        """
        self.all_tensors[tensor.tensor_id] = tensor
        self.root_tensor = tensor  # Track most recent (used as output)
    
    def get_graph(self) -> Optional[Graph]:
        """Get the current graph (FX or LazyDAG)."""
        if self.use_fx and self.fx_graph is not None:
            return FXGraphAdapter(self.fx_graph)
        elif self.root_tensor is not None:
            return LazyDAGAdapter(self.root_tensor)
        else:
            return None


# Global graph builder
_global_builder: Optional[HybridGraphBuilder] = None


def initialize_global_builder():
    """Initialize global graph builder (called on import)."""
    global _global_builder
    _global_builder = HybridGraphBuilder()
    
    # Set as LazyTensor's graph builder
    LazyTensor._graph_builder = _global_builder


def get_global_builder() -> HybridGraphBuilder:
    """Get the global graph builder."""
    if _global_builder is None:
        raise RuntimeError("Graph builder not initialized")
    return _global_builder
```

#### 2.3 Capture Context Manager

**File: `genie/core/capture.py` (NEW)**

```python
"""
Capture context manager with thread-local state signaling.

This module provides the capture context manager that signals to the factory
interceptor when operations should return LazyTensors instead of concrete tensors.
Uses thread-local storage for thread safety.

Threading Behavior:
- Each thread has its own capture state
- Capture contexts don't interfere across threads
- Nested contexts work correctly within each thread
- State is properly isolated and restored
"""

import threading
from contextlib import contextmanager
from typing import Optional

from .graph_interface import Graph
from .graph_builder import get_global_builder

# Thread-local state for capture context
# This signals to factory interceptor that we're in capture mode
_capture_context = threading.local()


class CaptureContext:
    """Context for capturing operations into a graph."""

    def __init__(self):
        self.builder = get_global_builder()
        self.prev_root = None
        self.prev_active = False

    def __enter__(self):
        # Signal to factory interceptor that we're in capture mode
        self.prev_active = getattr(_capture_context, 'active', False)
        _capture_context.active = True

        # Save previous state
        self.prev_root = self.builder.root_tensor
        # Start fresh capture
        self.builder.root_tensor = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous state
        _capture_context.active = self.prev_active


@contextmanager
def capture():
    """
    Capture operations into a computation graph.

    Usage:
        with genie.capture():
            y = model(x)

        graph = genie.get_graph()

    Inside the capture context, factory functions like torch.randn() will
    return LazyTensors instead of concrete tensors, enabling graph capture.
    """
    ctx = CaptureContext()
    with ctx:
        yield ctx


def get_graph() -> Optional[Graph]:
    """Get the most recently captured graph."""
    builder = get_global_builder()
    return builder.get_graph()


def is_capturing() -> bool:
    """Check if currently inside a capture context."""
    return getattr(_capture_context, 'active', False)
```

**Deliverables (Phase 2):**
- ✅ Unified graph interface (FX + LazyDAG)
- ✅ Automatic fallback when FX fails
- ✅ Clean capture API (`with genie.capture()`)

---

### Phase 3: Semantic Analysis (Weeks 9-12)

*[Continue with detailed implementation of semantic analysis, scheduler, etc.]*

---

## 5. API Design

### 5.1 Public API (v1.0)

```python
# ===================================================================
# CORE API
# ===================================================================

import genie

# Initialize Genie (optional - auto-initialized on import)
genie.init()

# Option 1: Context-based API (recommended for new code)
with genie.capture():
    output = model(input)

# Option 2: Device-based API (legacy compatibility)
x = torch.randn(10, device='remote_accelerator:0')
output = model(x)

# Option 3: Hybrid API (flexible)
with genie.capture():
    x = torch.randn(10, device='remote_accelerator:0')  # Explicit device
    output = model(x)

# Get captured graph (works with all API styles)
graph = genie.get_graph()

# ===================================================================
# SEMANTIC ANALYSIS
# ===================================================================

# Annotate graph with semantic information
annotated_graph = genie.annotate(graph)

# Access semantic metadata
for node in annotated_graph.nodes():
    phase = node.metadata.get('phase')  # ExecutionPhase
    modality = node.metadata.get('modality')  # Modality
    compute = node.metadata.get('compute_flops')  # float

# ===================================================================
# SCHEDULING
# ===================================================================

# Define cluster
devices = [
    genie.Device(id='gpu0', type='GPU', memory_gb=16, compute_tflops=10, network_gbps=100),
    genie.Device(id='gpu1', type='GPU', memory_gb=16, compute_tflops=10, network_gbps=100),
]

# Create scheduler
scheduler = genie.Scheduler(
    devices=devices,
    policy='min_latency'  # or 'min_cost', 'max_throughput'
)

# Generate execution plan
plan = scheduler.schedule(annotated_graph)

# Inspect plan
print(f"Estimated latency: {plan.latency_ms:.2f}ms")
print(f"Device placement:")
for node, device in plan.placement.items():
    print(f"  {node.id} -> {device.id}")

# ===================================================================
# EXECUTION
# ===================================================================

# Execute locally (Tier 1)
result = genie.execute_local(plan, inputs)

# Execute on cluster (Tier 2)
genie.init_cluster(config='cluster.yaml')
result = genie.execute(plan, inputs)

# Execute with RDMA (Tier 3)
genie.init_cluster(config='production.yaml', rdma=True)
result = genie.execute(plan, inputs)

# ===================================================================
# HIGH-LEVEL API
# ===================================================================

# One-shot execution (capture + analyze + schedule + execute)
result = genie.run(
    model,
    input,
    devices=devices,
    policy='min_latency'
)

# ===================================================================
# CONFIGURATION
# ===================================================================

# Set backend preferences
genie.set_backend('rdma')  # 'local', 'grpc', 'rdma'

# Enable debug logging
genie.set_log_level('DEBUG')

# Get statistics
stats = genie.get_stats()
print(f"Operations captured: {stats['total_ops']}")
print(f"Graph size: {stats['graph_nodes']}")
print(f"Overhead: {stats['overhead_ns']}ns/op")
```

### 5.2 Configuration API

```python
# cluster.yaml (Tier 2)
cluster:
  devices:
    - id: gpu0
      address: 192.168.1.10:8888
      type: GPU
      memory_gb: 16
      compute_tflops: 10
      network_gbps: 100
    
    - id: gpu1
      address: 192.168.1.11:8888
      type: GPU
      memory_gb: 16
      compute_tflops: 10
      network_gbps: 100
  
  backend: grpc
  
  scheduling:
    policy: min_latency
    recomputation_threshold_ms: 5.0
    
  fault_tolerance:
    enabled: true
    checkpoint_interval: 100  # operations
```

### 5.3 Extension Points

```python
# Custom semantic patterns
class MyCustomPattern(genie.PatternMatcher):
    def match(self, graph):
        # Identify custom pattern
        pass
    
    def annotate(self, nodes):
        # Add semantic metadata
        pass

genie.register_pattern(MyCustomPattern())

# Custom cost model
class MyCustomCostModel(genie.CostModel):
    def estimate_compute(self, node):
        # Custom compute cost estimation
        pass
    
    def estimate_transfer(self, src, dst, bytes):
        # Custom transfer cost estimation
        pass

scheduler.set_cost_model(MyCustomCostModel())

# Custom placement strategy
class MyCustomPlacer(genie.PlacementStrategy):
    def place(self, graph, devices):
        # Custom placement algorithm
        pass

scheduler.set_placement_strategy(MyCustomPlacer())
```

### 5.4 Choosing Your API Style

#### When to Use Device-Based API

**Use when:**
- Reproducing paper results
- Porting existing disaggregation code
- Want explicit device placement
- Single forward pass

```python
# Paper's original API - no changes needed
model = MyLLM()
x = torch.randn(batch_size, seq_len, device="remote_accelerator:0")
output = model(x)  # All operations intercepted automatically
```

**Advantages:**
- ✅ Matches HotNets paper exactly
- ✅ Explicit device specification
- ✅ Simple for single-pass scenarios

**Disadvantages:**
- ❌ Less convenient for complex workflows
- ❌ Must remember device string

#### When to Use Context-Based API

**Use when:**
- Writing new code
- Want cleaner syntax
- Building complex graphs
- Multiple forward passes

```python
# New convenience API
with genie.capture():
    x = torch.randn(batch_size, seq_len)  # No device needed
    output = model(x)  # All operations intercepted

# Graph available for analysis/scheduling
graph = genie.get_graph()
```

**Advantages:**
- ✅ Cleaner, more readable code
- ✅ Explicit control over capture scope
- ✅ No hardcoded device strings

**Disadvantages:**
- ❌ Different from paper (but more convenient)
- ❌ Requires context manager

#### When to Use Hybrid API

**Use when:**
- Need fine-grained control
- Mixing captured and native code
- Debugging

```python
# Flexible hybrid approach
with genie.capture():
    # Some tensors captured
    x = torch.randn(10)

    # Some on specific devices
    y = torch.randn(10, device="remote_accelerator:1")

    # Some native (not captured)
    z = torch.randn(10)  # Normal if outside capture

    output = model(x, y, z)
```

**Advantages:**
- ✅ Maximum flexibility
- ✅ Fine-grained control
- ✅ Can mix different execution modes

**Disadvantages:**
- ❌ More complex to understand
- ❌ Requires careful scoping

#### API Compatibility Matrix

| Feature | Device-Based | Context-Based | Hybrid |
|---------|-------------|---------------|---------|
| Paper compatibility | ✅ Full | ❌ None | ✅ Full |
| New code convenience | ❌ Poor | ✅ Excellent | ✅ Good |
| Complex workflows | ❌ Limited | ✅ Good | ✅ Excellent |
| Debugging | ✅ Explicit | ✅ Clear | ⚠️ Complex |
| Multi-pass scenarios | ❌ Manual | ✅ Easy | ✅ Flexible |

---

## 6. Testing Strategy

### 6.1 Test Coverage Goals

- **Unit tests:** >90% code coverage
- **Integration tests:** All major workflows
- **Performance tests:** Validate overhead targets
- **Correctness tests:** Compare against native PyTorch

### 6.2 Test Structure

```
tests/
├── unit/
│   ├── test_lazy_tensor.py          # LazyTensor subclass
│   ├── test_factory_interceptor.py  # Factory wrapping
│   ├── test_graph_builder.py        # Graph construction
│   ├── test_semantic_analyzer.py    # Pattern matching
│   └── test_scheduler.py            # Placement optimization
│
├── integration/
│   ├── test_simple_models.py        # Linear, MLP, CNN
│   ├── test_transformers.py         # BERT, GPT, ViT
│   ├── test_multimodal.py           # CLIP, Flamingo
│   └── test_dynamic_control.py      # Models with if/loops
│
├── performance/
│   ├── benchmark_interception.py    # Overhead measurement
│   ├── benchmark_scheduling.py      # Scheduler performance
│   └── benchmark_e2e.py             # End-to-end latency
│
├── correctness/
│   ├── test_numerical.py            # Compare with native PyTorch
│   ├── test_gradient.py             # Autograd correctness
│   └── test_determinism.py          # Reproducibility
│
└── deployment/
    ├── test_tier1_local.py          # Single machine
    ├── test_tier2_cluster.py        # Multi-node gRPC
    └── test_tier3_rdma.py           # RDMA datapath
```

### 6.3 Key Test Cases

**Test: `test_lazy_tensor.py`**

```python
def test_lazy_tensor_is_proper_subclass():
    """Verify LazyTensor is recognized as torch.Tensor."""
    x = LazyTensor.randn(10, 10)

    assert isinstance(x, torch.Tensor)
    assert isinstance(x, LazyTensor)


def test_operations_intercepted():
    """Verify operations create new LazyTensors."""
    x = LazyTensor.randn(10, 10)
    y = LazyTensor.randn(10, 10)

    z = x + y  # Should return LazyTensor
    assert isinstance(z, LazyTensor)
    assert z.operation == 'aten::add'


def test_no_execution_until_materialization():
    """Verify operations are deferred."""
    x = LazyTensor.randn(10, 10)
    y = x @ x

    # No computation yet
    assert not hasattr(y, '_concrete_value')

    # Trigger materialization
    result = y.cpu()

    # Now we have concrete tensor
    assert isinstance(result, torch.Tensor)
    assert result.shape == (10, 10)


def test_correctness_vs_native():
    """Verify numerical correctness."""
    # Native PyTorch
    torch.manual_seed(42)
    x_native = torch.randn(32, 64)
    y_native = torch.randn(64, 128)
    z_native = (x_native @ y_native).relu()

    # Genie LazyTensor
    torch.manual_seed(42)
    x_lazy = LazyTensor.randn(32, 64)
    y_lazy = LazyTensor.randn(64, 128)
    z_lazy = (x_lazy @ y_lazy).relu()
    z_concrete = z_lazy.cpu()

    # Should be identical
    torch.testing.assert_close(z_concrete, z_native)


def test_dynamic_control_flow():
    """Verify fallback to LazyDAG for dynamic models."""
    class DynamicModel(torch.nn.Module):
        def forward(self, x):
            if x.sum() > 0:  # Data-dependent branch
                return x.relu()
            else:
                return x.tanh()

    model = DynamicModel()

    with genie.capture():
        x = torch.randn(10, 10)
        y = model(x)

    graph = genie.get_graph()
    assert graph.backend_type == 'lazy_dag'  # Should fall back from FX


def test_nested_capture():
    """Verify thread-local state works correctly."""
    with genie.capture():
        x = torch.randn(10)  # Should be LazyTensor

        # Nested context (should maintain state)
        with genie.capture():
            y = torch.randn(10)  # Also LazyTensor

        z = torch.randn(10)  # Still LazyTensor (outer context active)

    assert isinstance(x, LazyTensor)
    assert isinstance(y, LazyTensor)
    assert isinstance(z, LazyTensor)


def test_mixed_device_operations():
    """Verify behavior when mixing captured and native tensors."""
    with genie.capture():
        x_lazy = torch.randn(10, 10)

    x_cpu = torch.randn(10, 10)  # Native CPU tensor

    # What happens here?
    y = x_lazy + x_cpu  # Should materialize x_lazy and compute on CPU
    assert isinstance(y, torch.Tensor)
    assert not isinstance(y, LazyTensor)  # Should be materialized


def test_both_api_styles():
    """Verify both device-based and context-based APIs work."""
    # Device-based API (legacy)
    x_device = torch.randn(10, 10, device='remote_accelerator:0')
    assert isinstance(x_device, LazyTensor)

    # Context-based API (new)
    with genie.capture():
        x_context = torch.randn(10, 10)
        assert isinstance(x_context, LazyTensor)

    # Hybrid API
    with genie.capture():
        x_hybrid = torch.randn(10, 10, device='remote_accelerator:0')
        assert isinstance(x_hybrid, LazyTensor)


def test_device_api_works_without_capture():
    """Verify device-based API works outside capture context."""
    # This is the paper's API - must work!
    x = torch.randn(10, 10, device="remote_accelerator:0")
    assert isinstance(x, LazyTensor), "Device API broken!"

    y = x @ x
    assert isinstance(y, LazyTensor)

    # Materialize
    result = y.cpu()
    assert isinstance(result, torch.Tensor)
    assert result.shape == (10, 10)


def test_capture_api_works_without_device():
    """Verify context-based API works without device argument."""
    # This is the new convenience API
    with genie.capture():
        x = torch.randn(10, 10)  # No device argument
        assert isinstance(x, LazyTensor), "Capture API broken!"

        y = x @ x
        assert isinstance(y, LazyTensor)

    # Materialize outside context
    result = y.cpu()
    assert isinstance(result, torch.Tensor)
    assert result.shape == (10, 10)


def test_capture_context_is_thread_safe():
    """Verify capture contexts don't interfere across threads."""
    import threading

    results = {}

    def thread1_work():
        with genie.capture():
            x = torch.randn(10)
            results['thread1'] = isinstance(x, LazyTensor)

    def thread2_work():
        # NOT in capture context
        x = torch.randn(10)
        results['thread2'] = not isinstance(x, LazyTensor)

    t1 = threading.Thread(target=thread1_work)
    t2 = threading.Thread(target=thread2_work)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results['thread1'], "Thread 1 should have LazyTensor"
    assert results['thread2'], "Thread 2 should have normal tensor"


def test_nested_capture_contexts():
    """Verify nested capture contexts maintain correct state."""
    with genie.capture():
        x1 = torch.randn(10)
        assert isinstance(x1, LazyTensor)

        with genie.capture():
            x2 = torch.randn(10)
            assert isinstance(x2, LazyTensor)

        x3 = torch.randn(10)
        assert isinstance(x3, LazyTensor)  # Still in outer context

    x4 = torch.randn(10)
    assert not isinstance(x4, LazyTensor)  # Outside all contexts
```

**Test: `benchmark_interception.py`**

```python
def benchmark_interception_overhead():
    """Measure per-operation overhead."""
    import time

    # Warmup
    for _ in range(100):
        x = torch.randn(100, 100)
        y = x + 1

    # Benchmark native
    start = time.perf_counter()
    for _ in range(10000):
        x = torch.randn(100, 100)
        y = x + 1
    native_time = time.perf_counter() - start

    # Benchmark Genie with context-based API
    with genie.capture():
        start = time.perf_counter()
        for _ in range(10000):
            x = torch.randn(100, 100)  # Uses capture context
            y = x + 1
        genie_time = time.perf_counter() - start

    overhead_per_op = (genie_time - native_time) / 10000
    overhead_ns = overhead_per_op * 1e9

    print(f"Native: {native_time:.4f}s")
    print(f"Genie: {genie_time:.4f}s")
    print(f"Overhead: {overhead_ns:.1f}ns/op")

    # Target: <1000ns
    assert overhead_ns < 1000, f"Overhead too high: {overhead_ns}ns"


def benchmark_capture_context_overhead():
    """Measure overhead of capture context signaling."""
    import time

    n_ops = 10000

    # Without capture context
    start = time.perf_counter()
    for _ in range(n_ops):
        x = torch.randn(100, 100)
    no_capture_time = time.perf_counter() - start

    # With capture context (but no LazyTensor creation)
    start = time.perf_counter()
    with genie.capture():
        for _ in range(n_ops):
            x = torch.randn(100, 100)  # Creates LazyTensor
    capture_time = time.perf_counter() - start

    overhead_per_op = (capture_time - no_capture_time) / n_ops
    overhead_ns = overhead_per_op * 1e9

    print(f"No capture: {no_capture_time:.4f}s")
    print(f"With capture: {capture_time:.4f}s")
    print(f"Overhead: {overhead_ns:.1f}ns/op")

    # Should be minimal overhead
    assert overhead_ns < 100, f"Capture overhead too high: {overhead_ns}ns"
```

---

## 7. Performance Targets

### 7.1 Interception Overhead

| Metric | Target | Rationale |
|--------|--------|-----------|
| Per-operation overhead | <1μs | Negligible for model inference |
| Graph construction | <100ms for 1000 ops | One-time cost |
| Memory overhead | <5% of model size | Acceptable for graph metadata |

### 7.2 Scheduling Performance

| Metric | Target | Rationale |
|--------|--------|-----------|
| Placement optimization | <100ms for 1000 nodes | Acceptable for interactive |
| Schedule generation | <50ms | Needs to be fast |
| Cost model evaluation | <1ms per node | Called frequently |

### 7.3 End-to-End Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| Disaggregation overhead | <10% vs native | Latency increase |
| Network efficiency | >80% peak bandwidth | Transfer throughput |
| Memory efficiency | <2x native | Peak memory usage |

### 7.4 Performance Measurement

```python
# File: benchmarks/e2e_benchmark.py

def benchmark_e2e_disaggregation():
    """Measure end-to-end disaggregation overhead."""
    
    model = create_bert_base()
    input = torch.randn(32, 512)  # Batch=32, seq_len=512
    
    # Native PyTorch baseline
    native_time = benchmark_native(model, input, iterations=100)
    
    # Genie with local execution (overhead measurement)
    local_time = benchmark_genie_local(model, input, iterations=100)
    
    # Genie with remote execution (full system)
    remote_time = benchmark_genie_remote(model, input, iterations=100)
    
    overhead_local = (local_time - native_time) / native_time * 100
    overhead_remote = (remote_time - native_time) / native_time * 100
    
    print(f"Native: {native_time:.2f}ms")
    print(f"Genie (local): {local_time:.2f}ms ({overhead_local:.1f}% overhead)")
    print(f"Genie (remote): {remote_time:.2f}ms ({overhead_remote:.1f}% overhead)")
    
    # Targets
    assert overhead_local < 5, "Local overhead too high"
    assert overhead_remote < 15, "Remote overhead too high"
```

---

## 8. Migration Guide

### 8.1 Breaking Changes

**v0.1.x → v0.2.0:**

| Change | Old Code | New Code |
|--------|----------|----------|
| Graph access | `GraphBuilder.current().get_graph()` | `genie.get_graph()` |
| Metadata access | `tensor.metadata['phase']` | `node.metadata.get('phase')` |

**Note:** The device-based API (`device='remote_accelerator:0'`) continues to work unchanged, providing backward compatibility.

### 8.2 Migration Timeline

**Phase 1 (v0.2.0 - v0.3.0): Both APIs Supported**

Both APIs work without warnings (full backward compatibility):

```python
# Device-based API (legacy, continues to work)
x = torch.randn(10, device='remote_accelerator:0')

# Context-based API (recommended for new code)
with genie.capture():
    x = torch.randn(10)

# Hybrid API (flexible)
with genie.capture():
    x = torch.randn(10, device='remote_accelerator:0')
```

**Future versions:** Both APIs will continue to be supported for maximum compatibility.

### 8.3 Migration Script

```python
# File: scripts/migrate_to_v2.py

"""
Automated migration script for v0.1.x → v0.2.0
"""

import re
import sys

def migrate_code(source_code: str) -> str:
    """Migrate code from v0.1.x to v0.2.0."""

    # Pattern 1: GraphBuilder.current()
    source_code = re.sub(
        r"GraphBuilder\.current\(\)\.get_graph\(\)",
        "genie.get_graph()",
        source_code
    )

    # Pattern 2: tensor.metadata[...]
    source_code = re.sub(
        r"\.metadata\[(['\"])(\w+)\1\]",
        r".metadata.get('\2')",
        source_code
    )

    # Pattern 3: device='remote_accelerator:N' (no change needed)
    # This API continues to work unchanged for backward compatibility

    return source_code


if __name__ == '__main__':
    file_path = sys.argv[1]

    with open(file_path, 'r') as f:
        old_code = f.read()

    new_code = migrate_code(old_code)

    with open(file_path + '.migrated', 'w') as f:
        f.write(new_code)

    print(f"Migrated code written to {file_path}.migrated")
    print("Note: device='remote_accelerator:N' API continues to work unchanged")
```

---

## 9. Open-Source Considerations

### 9.1 Repository Structure

```
genie/
├── README.md                    # Quick start, badges, examples
├── CONTRIBUTING.md              # Contribution guidelines
├── LICENSE                      # Apache 2.0 (recommended)
├── CODE_OF_CONDUCT.md          # Community guidelines
├── .github/
│   ├── workflows/
│   │   ├── ci.yml              # CI/CD pipeline
│   │   ├── benchmarks.yml      # Performance regression tests
│   │   └── docs.yml            # Documentation build
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── performance_issue.md
│   └── PULL_REQUEST_TEMPLATE.md
├── docs/
│   ├── getting-started.md
│   ├── api-reference.md
│   ├── architecture.md
│   ├── deployment.md
│   ├── performance.md
│   └── paper/                   # HotNets paper reproduction
│       ├── reproduction.md
│       └── benchmarks/
├── genie/
│   ├── __init__.py
│   ├── core/                    # Core interception & graph
│   ├── semantic/                # Semantic analysis
│   ├── scheduler/               # Scheduling algorithms
│   ├── runtime/                 # Execution backends
│   └── utils/
├── tests/
├── benchmarks/
├── examples/
│   ├── 01_simple_model.py
│   ├── 02_transformer.py
│   ├── 03_multimodal.py
│   ├── 04_custom_scheduler.py
│   └── 05_production_deployment.py
├── setup.py
├── pyproject.toml
└── requirements.txt
```

### 9.2 Documentation Strategy

**Target Audiences:**

1. **End Users** (ML engineers deploying models)
   - Quick start guide
   - API reference
   - Deployment tutorials

2. **Contributors** (developers improving Genie)
   - Architecture documentation
   - Development guide
   - Code review standards

3. **Researchers** (reproducing paper results)
   - Paper reproduction guide
   - Benchmark scripts
   - Performance analysis

**Documentation Structure:**

```markdown
# docs/getting-started.md

# Getting Started with Genie

## Installation

```bash
pip install genie-pytorch
```

## Your First Disaggregated Model

```python
import torch
import genie

# Define model
model = torch.nn.Sequential(
    torch.nn.Linear(100, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 10)
)

# Capture operations
with genie.capture():
    x = torch.randn(32, 100)
    output = model(x)

# Analyze
graph = genie.get_graph()
annotated = genie.annotate(graph)

# Schedule
devices = [genie.Device('gpu0', 'GPU', 16, 10, 100)]
scheduler = genie.Scheduler(devices, policy='min_latency')
plan = scheduler.schedule(annotated)

# Execute
result = genie.execute_local(plan, [x])
```

## Next Steps

- [API Reference](api-reference.md)
- [Deployment Guide](deployment.md)
- [Performance Tuning](performance.md)
```

### 9.3 Community Building

**Launch Strategy:**

1. **Week 1-2: Soft Launch**
   - Announce on internal channels
   - Gather feedback from early adopters
   - Fix critical bugs

2. **Week 3-4: Public Announcement**
   - Blog post explaining vision
   - Reddit/HN announcement
   - Twitter thread with examples

3. **Week 5-8: Community Growth**
   - Respond to issues quickly (<24h)
   - Merge first external PRs
   - Publish tutorial videos

4. **Week 9-12: Conference Submission**
   - Submit to MLSys/OSDI
   - Demo at conferences
   - Workshop presentations

**Engagement Metrics:**

| Metric | 3 Months | 6 Months | 12 Months |
|--------|----------|----------|-----------|
| GitHub stars | 100 | 500 | 2000 |
| Contributors | 5 | 15 | 30 |
| PyPI downloads | 1K | 10K | 50K |
| Issues closed | 20 | 100 | 300 |

### 9.4 License & Attribution

**Recommended License:** Apache 2.0

**Rationale:**
- Permissive (allows commercial use)
- Patent grant (protects users)
- Compatible with PyTorch (BSD license)
- Industry standard for ML frameworks

**Attribution:**

```
Copyright 2024 Genie Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.

Based on research presented in:
"Genie: Semantic Disaggregated Execution"
HotNets 2024
```

---

## 10. Risk Assessment

### 10.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| FX fails on >20% models | Medium | Medium | Hybrid approach with LazyTensor fallback |
| Overhead >1μs/op | Low | High | Extensive profiling, optimization |
| Scheduler complexity | High | Medium | Start with greedy, iterate to optimal |
| RDMA integration issues | Medium | Low | Support TCP fallback |
| Autograd compatibility | Medium | High | Phase 2 feature, extensive testing |

### 10.2 Community Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Low adoption | Medium | High | Focus on easy Tier 1 deployment |
| Competing frameworks | Low | Medium | Differentiate with semantics |
| Maintenance burden | Medium | Medium | Build contributor community early |
| API churn | Low | High | Deprecation policy with long timeline |

### 10.3 Project Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Timeline slippage | Medium | Medium | Phased approach, MVP focus |
| Scope creep | High | Medium | Strict feature prioritization |
| Resource constraints | Low | High | Clear phase deliverables |
| Paper rejection | Low | Medium | Strong evaluation, reproduce paper |

---

## 11. Timeline & Milestones

### 11.1 Detailed Timeline

**Phase 1: Core Architecture (Weeks 1-4)** ✅ **COMPLETED**

**Phase 2: Hybrid Graph (Weeks 5-8)**

| Week | Task | Deliverable | Risk |
|------|------|-------------|------|
| 5 | FX adapter | FXGraphAdapter with unified interface | Low |
| 6 | LazyDAG adapter | LazyDAGAdapter with unified interface | Low |
| 7 | HybridGraphBuilder | Automatic fallback between FX and LazyDAG | Medium |
| 8 | Capture API | Context manager, get_graph() | Low |

**Milestone 2:** Hybrid graph working for 95% of models

**Phase 3: Semantic Analysis (Weeks 9-12)**

| Week | Task | Deliverable | Risk |
|------|------|-------------|------|
| 9 | Pattern matchers | Attention, convolution, KV cache detection | Medium |
| 10 | Phase detection | Prefill/decode classification | Medium |
| 11 | Cost estimation | FLOPs, memory, intensity | Low |
| 12 | Integration | Annotate API, metadata storage | Low |

**Milestone 3:** Semantic annotations on all nodes

**Phase 4: Scheduler (Weeks 13-16)**

| Week | Task | Deliverable | Risk |
|------|------|-------------|------|
| 13 | Cost model | Compute + transfer + queueing | Medium |
| 14 | Placement algorithm | Greedy placer with co-location | Medium |
| 15 | Schedule generation | Topological sort with pipelining hints | High |
| 16 | Optimization | Performance tuning, edge cases | High |

**Milestone 4:** Working scheduler with <10% overhead on local execution

**Phase 5: Tier 2 Execution (Weeks 17-20)**

| Week | Task | Deliverable | Risk |
|------|------|-------------|------|
| 17 | gRPC server | Remote execution server | Low |
| 18 | gRPC client | Client-side execution backend | Low |
| 19 | Multi-node | Cluster configuration, coordination | Medium |
| 20 | Integration | End-to-end Tier 2 deployment | Medium |

**Milestone 5:** Multi-node disaggregation working

**Phase 6: Documentation & Release (Weeks 21-24)**

| Week | Task | Deliverable | Risk |
|------|------|-------------|------|
| 21 | Documentation | Complete docs, tutorials, examples | Low |
| 22 | Benchmarks | Performance validation, paper reproduction | Medium |
| 23 | Polish | Bug fixes, API cleanup | Low |
| 24 | Release | v1.0 release, announcement | Low |

**Milestone 6:** v1.0 production release

### 11.2 Success Criteria

**Technical:**
- ✅ <1μs per-operation overhead (Phase 1.1 completed)
- ✅ >95% model compatibility (FX + LazyDAG)
- ✅ <10% end-to-end disaggregation overhead
- ✅ All paper results reproduced
- ✅ Support for both device-based and context-based APIs (Phase 1.1 completed)
- ✅ Transparent interception with thread-local capture context (Phase 1.1 completed)

**Community:**
- ✅ 100+ GitHub stars in first 3 months
- ✅ 5+ external contributors
- ✅ 1000+ PyPI downloads
- ✅ Zero critical bugs

**Research:**
- ✅ Paper accepted to top-tier venue (MLSys/OSDI/NSDI)
- ✅ 3+ citations in first year
- ✅ Industry adoption (1+ company)

---

## 12. Conclusion

This refactoring plan transforms Genie from a research prototype into a production-ready, open-source framework. Key improvements:

1. **Simplified Architecture:** 2 interception mechanisms instead of 3
2. **Production Quality:** <1μs overhead, robust error handling
3. **Wide Compatibility:** Hybrid graph (FX + LazyDAG) covers 95%+ models
4. **Open-Source Ready:** Clean APIs, comprehensive docs, community focus
5. **Incremental Deployment:** 3 tiers (local → cluster → RDMA)

**Recommended Next Steps:**

1. **Week 5:** Team alignment on Phase 2 (Hybrid Graph Builder)
2. **Week 5-6:** Implement unified graph interface and FX adapter
3. **Week 7-8:** Implement LazyDAG adapter and hybrid graph builder
4. **Week 8:** Review progress, adjust timeline for Phase 3

**Key Decision Points:**

1. ✅ **Factory wrapping is necessary** (confirmed)
2. ✅ **Target 3-tier deployment** (local → cluster → RDMA)
3. ✅ **Open-source with Apache 2.0 license**
4. ✅ **Support both API styles** (device-based + context-based) - COMPLETED
5. ✅ **Thread-local capture context signaling** (critical fix implemented) - COMPLETED
6. ✅ **Phase 1.1: Proper LazyTensor subclass** - COMPLETED
7. ⏳ **Timeline: 20 weeks to v1.0** (updated for Phase 1 completion)

This plan balances ambition with pragmatism, ensuring we deliver a system that is both research-innovative and production-viable.

---

## Appendix A: Alternative Approaches Considered

### A.1 Pure FX Graph (Rejected)

**Pros:** Standard format, rich tooling  
**Cons:** Fails on 20% of models with dynamic control flow  
**Decision:** Use as primary with fallback

### A.2 Pure LazyTensor DAG (Rejected)

**Pros:** Always works, no tracer failures  
**Cons:** Non-standard format, no tooling ecosystem  
**Decision:** Use as fallback only

### A.3 torch.library Only (Rejected)

**Pros:** Official PyTorch extension mechanism  
**Cons:** Doesn't intercept factory functions, redundant with __torch_dispatch__  
**Decision:** Remove this mechanism

### A.4 Four-Tier Deployment (Rejected)

**Pros:** More granular options
**Cons:** Too complex for users to choose
**Decision:** Three tiers is sufficient

### A.5 Single API Style (Rejected)

**Option 1: Device-based only**
**Pros:** Matches paper API exactly
**Cons:** Less flexible for users
**Decision:** Support both styles for maximum compatibility

**Option 2: Context-based only**
**Pros:** Cleaner, more explicit
**Cons:** Breaks existing paper code
**Decision:** Support both styles for smooth migration

---

## Appendix B: Performance Profiling Strategy

### B.1 Profiling Points

```python
# Where to measure overhead:
1. LazyTensor.__new__      # Object creation
2. __torch_dispatch__      # Operation interception
3. Shape inference         # Meta tensor overhead
4. Graph building          # Node insertion
5. Semantic analysis       # Pattern matching
6. Scheduling             # Placement optimization
```

### B.2 Profiling Tools

- **cProfile:** Python-level profiling
- **py-spy:** Low-overhead sampling profiler
- **torch.profiler:** PyTorch operation timing
- **NVIDIA Nsight:** GPU kernel profiling

### B.3 Optimization Targets

If overhead exceeds targets:

1. **LazyTensor creation:** Use `__slots__`, avoid dict
2. **Shape inference:** Cache results, fast paths for common ops
3. **Graph building:** Batch operations, lazy updates
4. **Semantic analysis:** Only compute when needed
5. **Scheduling:** Greedy algorithm, not ILP

---

**Document Status:** READY FOR REVIEW  
**Next Action:** Team review, approval, begin Phase 1 implementation