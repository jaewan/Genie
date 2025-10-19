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

**Timeline:** 24 weeks (6 months) to v1.0 production release

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
Phase 1 (Weeks 1-8):  Tier 1 - Single machine, local execution
Phase 2 (Weeks 9-16): Tier 2 - Multi-node with gRPC+TCP
Phase 3 (Weeks 17-24): Tier 3 - RDMA+DPDK optimization
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

### Phase 1: Core Architecture (Weeks 1-4)

#### 1.1 Proper LazyTensor Subclass

**File: `genie/core/lazy_tensor.py` (NEW VERSION)**

```python
"""
LazyTensor: Proper torch.Tensor subclass for deferred execution.

Key improvements:
1. Inherits from torch.Tensor (official API)
2. Uses _make_subclass for proper tensor semantics
3. __torch_dispatch__ called automatically by PyTorch
4. Compatible with all PyTorch APIs (autograd, compile, etc.)
"""

import torch
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LazyTensor(torch.Tensor):
    """
    Lazy tensor that captures operations without executing.
    
    This is a proper torch.Tensor subclass that integrates with PyTorch's
    dispatcher system. All operations on LazyTensors are automatically
    intercepted via __torch_dispatch__.
    
    Example:
        >>> x = LazyTensor.randn(10, 10)  # No computation yet
        >>> y = x @ x                      # Still no computation
        >>> z = y.cpu()                    # Triggers execution
    """
    
    # Class-level state
    _graph_builder: Optional['GraphBuilder'] = None
    _shape_cache: Dict[Tuple, Optional[torch.Size]] = {}

    # Fast path for simple operations (avoids FakeTensorMode overhead)
    _SIMPLE_OPS = {
        'aten::relu': lambda inputs: inputs[0].shape,
        'aten::sigmoid': lambda inputs: inputs[0].shape,
        'aten::tanh': lambda inputs: inputs[0].shape,
        'aten::abs': lambda inputs: inputs[0].shape,
        'aten::neg': lambda inputs: inputs[0].shape,
        'aten::exp': lambda inputs: inputs[0].shape,
        'aten::log': lambda inputs: inputs[0].shape,
        'aten::add': lambda inputs: inputs[0].shape,  # Simplified
        'aten::sub': lambda inputs: inputs[0].shape,  # Simplified
        'aten::mul': lambda inputs: inputs[0].shape,  # Simplified
        'aten::div': lambda inputs: inputs[0].shape,  # Simplified
    }
    
    # Track metadata without breaking tensor subclass protocol
    @staticmethod
    def __new__(
        cls,
        operation: str,
        inputs: List[Any],
        kwargs: Optional[Dict[str, Any]] = None,
        shape: Optional[torch.Size] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Create LazyTensor wrapper.
        
        CRITICAL: Must use _make_subclass for proper tensor subclass.
        This creates a tensor wrapper WITHOUT allocating actual storage.
        """
        # Infer shape/dtype if not provided
        if shape is None:
            shape = cls._infer_shape(operation, inputs, kwargs or {})
        if shape is None:
            shape = torch.Size([])  # Fallback for unknown shapes
        
        if dtype is None:
            dtype = cls._infer_dtype(inputs, kwargs or {})
        if dtype is None:
            dtype = torch.float32  # Fallback
        
        if device is None:
            device = torch.device('meta')  # Symbolic device (no storage)
        
        # Create tensor wrapper using official API
        # This is what makes LazyTensor a "real" tensor
        wrapper = torch.Tensor._make_subclass(
            cls,
            torch.empty(shape, dtype=dtype, device=device),
            require_grad=False  # Disable autograd for now (Phase 2 addition)
        )
        
        return wrapper
    
    def __init__(
        self,
        operation: str,
        inputs: List[Any],
        kwargs: Optional[Dict[str, Any]] = None,
        shape: Optional[torch.Size] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize LazyTensor metadata.
        
        Note: __new__ has already created the tensor wrapper.
        Here we attach operation metadata for graph building.
        """
        # Store operation info
        # These become attributes of the tensor instance
        object.__setattr__(self, '_operation', operation)
        object.__setattr__(self, '_inputs', inputs)
        object.__setattr__(self, '_kwargs', kwargs or {})
        object.__setattr__(self, '_tensor_id', id(self))
        
        # Register with graph builder
        if LazyTensor._graph_builder is not None:
            LazyTensor._graph_builder.add_operation(self)
    
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """
        Intercept ALL operations involving LazyTensor.
        
        This is automatically called by PyTorch's dispatcher for any
        operation that involves a LazyTensor operand.
        
        Args:
            func: ATen operation (e.g., torch.ops.aten.add.Tensor)
            types: Tuple of types involved (contains LazyTensor)
            args: Positional arguments
            kwargs: Keyword arguments
        
        Returns:
            LazyTensor representing the deferred operation
        """
        kwargs = kwargs or {}
        
        # Handle special operations that force materialization
        if func in cls._MATERIALIZATION_OPS:
            # These operations need concrete tensors
            materialized_args = tuple(
                arg.materialize() if isinstance(arg, LazyTensor) else arg
                for arg in args
            )
            return func(*materialized_args, **kwargs)
        
        # Normalize operation name
        op_name = cls._normalize_op_name(func)
        
        # Create new LazyTensor for the result
        result = cls(
            operation=op_name,
            inputs=list(args),
            kwargs=kwargs
        )
        
        return result
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Fallback for operations not going through __torch_dispatch__.
        
        This catches operations like:
        - .cpu(), .cuda() (device transfers)
        - .numpy(), .item() (conversion to non-tensor)
        - .tolist() (conversion to Python types)
        
        All of these trigger materialization.
        """
        # Operations that force execution
        if func in cls._MATERIALIZATION_OPS:
            lazy_tensor = args[0]
            materialized = lazy_tensor.materialize()
            return func(materialized, *args[1:], **(kwargs or {}))
        
        # Try dispatching
        return cls.__torch_dispatch__(func, types, args, kwargs)
    
    # ===================================================================
    # FACTORY METHODS
    # ===================================================================
    
    @classmethod
    def randn(cls, *size, dtype=None, device=None, requires_grad=False):
        """Create random normal LazyTensor."""
        return cls(
            operation='aten::randn',
            inputs=list(size),
            kwargs={'dtype': dtype, 'device': device, 'requires_grad': requires_grad},
            shape=torch.Size(size),
            dtype=dtype or torch.float32,
            device=device
        )

    @classmethod
    def tensor(cls, data, dtype=None, device=None, requires_grad=False):
        """Create LazyTensor from data."""
        return cls(
            operation='aten::tensor',
            inputs=[data],
            kwargs={'dtype': dtype, 'device': device, 'requires_grad': requires_grad},
            shape=torch.Size(data.shape) if hasattr(data, 'shape') else torch.Size([]),
            dtype=dtype or (data.dtype if hasattr(data, 'dtype') else torch.float32),
            device=device
        )

    @classmethod
    def as_tensor(cls, data, dtype=None, device=None):
        """Create LazyTensor from data (alias for tensor)."""
        return cls.tensor(data, dtype=dtype, device=device)

    @classmethod
    def from_numpy(cls, ndarray, dtype=None, device=None):
        """Create LazyTensor from numpy array."""
        return cls(
            operation='aten::from_numpy',
            inputs=[ndarray],
            kwargs={'dtype': dtype, 'device': device},
            shape=torch.Size(ndarray.shape),
            dtype=dtype or torch.from_numpy(ndarray).dtype,
            device=device
        )
    
    @classmethod
    def zeros(cls, *size, dtype=None, device=None, requires_grad=False):
        """Create zero-filled LazyTensor."""
        return cls(
            operation='aten::zeros',
            inputs=list(size),
            kwargs={'dtype': dtype, 'device': device, 'requires_grad': requires_grad},
            shape=torch.Size(size),
            dtype=dtype or torch.float32,
            device=device
        )
    
    @classmethod
    def ones(cls, *size, dtype=None, device=None, requires_grad=False):
        """Create one-filled LazyTensor."""
        return cls(
            operation='aten::ones',
            inputs=list(size),
            kwargs={'dtype': dtype, 'device': device, 'requires_grad': requires_grad},
            shape=torch.Size(size),
            dtype=dtype or torch.float32,
            device=device
        )
    
    # ===================================================================
    # MATERIALIZATION
    # ===================================================================
    
    def materialize(self) -> torch.Tensor:
        """
        Force execution of the computation graph.
        
        This traverses the DAG and executes operations to produce
        a concrete tensor.
        
        Returns:
            Concrete torch.Tensor with actual data
        """
        if LazyTensor._graph_builder is None:
            raise RuntimeError("No graph builder registered")
        
        return LazyTensor._graph_builder.materialize(self)
    
    # Operations that force materialization
    _MATERIALIZATION_OPS = {
        torch.Tensor.cpu,
        torch.Tensor.cuda,
        torch.Tensor.numpy,
        torch.Tensor.item,
        torch.Tensor.tolist,
        torch.Tensor.__bool__,
        torch.Tensor.__int__,
        torch.Tensor.__float__,
    }
    
    # ===================================================================
    # SHAPE INFERENCE
    # ===================================================================
    
    @staticmethod
    def _infer_shape(
        operation: str,
        inputs: List[Any],
        kwargs: Dict[str, Any]
    ) -> Optional[torch.Size]:
        """
        Infer output shape using PyTorch's meta tensor system.

        This executes the operation on "fake" tensors that only track
        shape/dtype without allocating storage.
        """
        # Fast path for simple operations (no FakeTensorMode overhead)
        if operation in cls._SIMPLE_OPS:
            try:
                return cls._SIMPLE_OPS[operation](inputs)
            except:
                pass  # Fall through to FakeTensorMode

        # Create cache key from operation and input shapes
        input_shapes = []
        for inp in inputs:
            if hasattr(inp, 'shape'):
                input_shapes.append(tuple(inp.shape))
            else:
                input_shapes.append(None)

        cache_key = (operation, tuple(input_shapes))

        # Check cache first
        if cache_key in LazyTensor._shape_cache:
            return LazyTensor._shape_cache[cache_key]

        try:
            # Use PyTorch's FakeTensor mode for accurate shape inference
            from torch._subclasses.fake_tensor import FakeTensorMode

            with FakeTensorMode():
                # Convert inputs to fake tensors
                fake_inputs = []
                for inp in inputs:
                    if isinstance(inp, LazyTensor):
                        # Use the LazyTensor's shape
                        fake_inputs.append(
                            torch.empty(inp.shape, dtype=inp.dtype, device='meta')
                        )
                    elif isinstance(inp, torch.Tensor):
                        # Convert to meta tensor
                        fake_inputs.append(inp.to('meta'))
                    else:
                        # Scalar or non-tensor (keep as-is)
                        fake_inputs.append(inp)

                # Get operation function
                op_func = LazyTensor._get_operation_function(operation)

                # Execute on fake tensors (shape inference only)
                fake_result = op_func(*fake_inputs, **kwargs)

                # Extract shape
                if isinstance(fake_result, torch.Tensor):
                    result_shape = fake_result.shape
                else:
                    result_shape = torch.Size([])

                # Cache result
                LazyTensor._shape_cache[cache_key] = result_shape
                return result_shape

        except Exception as e:
            logger.debug(f"Shape inference failed for {operation}: {e}")
            # Fallback: try simple heuristics
            result = LazyTensor._infer_shape_fallback(operation, inputs)
            # Cache fallback result too
            LazyTensor._shape_cache[cache_key] = result
            return result
    
    @staticmethod
    def _infer_shape_fallback(
        operation: str,
        inputs: List[Any]
    ) -> Optional[torch.Size]:
        """
        Fallback shape inference using simple heuristics.
        
        Used when FakeTensorMode fails (dynamic shapes, unsupported ops, etc.)
        """
        # Element-wise operations preserve shape
        if operation in ['aten::relu', 'aten::sigmoid', 'aten::tanh', 
                        'aten::abs', 'aten::neg', 'aten::exp', 'aten::log']:
            if inputs and hasattr(inputs[0], 'shape'):
                return inputs[0].shape
        
        # Matrix multiplication
        if operation in ['aten::matmul', 'aten::mm']:
            if len(inputs) >= 2:
                a_shape = getattr(inputs[0], 'shape', None)
                b_shape = getattr(inputs[1], 'shape', None)
                if a_shape and b_shape and len(a_shape) >= 2 and len(b_shape) >= 2:
                    # (..., M, K) @ (..., K, N) -> (..., M, N)
                    return torch.Size([*a_shape[:-1], b_shape[-1]])
        
        # Broadcasting operations
        if operation in ['aten::add', 'aten::sub', 'aten::mul', 'aten::div']:
            if len(inputs) >= 2:
                a_shape = getattr(inputs[0], 'shape', None)
                b_shape = getattr(inputs[1], 'shape', None)
                if a_shape and b_shape:
                    # Simple broadcasting (return larger shape)
                    if len(a_shape) >= len(b_shape):
                        return a_shape
                    else:
                        return b_shape
        
        # Unknown - return empty shape
        return None
    
    @staticmethod
    def _infer_dtype(inputs: List[Any], kwargs: Dict[str, Any]) -> Optional[torch.dtype]:
        """Infer output dtype from inputs or kwargs."""
        # Explicit dtype in kwargs
        if 'dtype' in kwargs and kwargs['dtype'] is not None:
            return kwargs['dtype']
        
        # Infer from first tensor input
        for inp in inputs:
            if isinstance(inp, (LazyTensor, torch.Tensor)):
                return inp.dtype
        
        # Default
        return None
    
    # ===================================================================
    # UTILITIES
    # ===================================================================
    
    @staticmethod
    def _normalize_op_name(func) -> str:
        """
        Normalize operation names to canonical form.
        
        Examples:
            torch.ops.aten.add.Tensor -> aten::add
            torch.add -> aten::add
            add -> aten::add
        """
        if hasattr(func, '__name__'):
            name = func.__name__
        elif hasattr(func, '_schema'):
            # ATen operation with schema
            schema_str = str(func._schema)
            name = schema_str.split('(')[0]
            if '::' in name:
                name = name.split('::')[-1]
        else:
            name = str(func)
        
        # Remove overload suffix (e.g., add.Tensor -> add)
        name = name.split('.')[0]
        
        # Ensure aten:: prefix
        if not name.startswith('aten::'):
            name = f'aten::{name}'
        
        return name
    
    @staticmethod
    def _get_operation_function(operation: str):
        """
        Get PyTorch function for an operation name.
        
        Maps "aten::add" -> torch.ops.aten.add
        """
        if operation.startswith('aten::'):
            op_name = operation[6:]  # Remove "aten::" prefix
            try:
                return getattr(torch.ops.aten, op_name)
            except AttributeError:
                # Fallback to torch namespace
                return getattr(torch, op_name)
        else:
            return getattr(torch, operation)
    
    # ===================================================================
    # PROPERTY ACCESSORS
    # ===================================================================
    
    @property
    def operation(self) -> str:
        """Get the operation that created this tensor."""
        return object.__getattribute__(self, '_operation')
    
    @property
    def inputs(self) -> List[Any]:
        """Get the input arguments to this operation."""
        return object.__getattribute__(self, '_inputs')
    
    @property
    def kwargs(self) -> Dict[str, Any]:
        """Get the keyword arguments to this operation."""
        return object.__getattribute__(self, '_kwargs')
    
    @property
    def tensor_id(self) -> int:
        """Get unique ID for this tensor."""
        return object.__getattribute__(self, '_tensor_id')
    
    # ===================================================================
    # STRING REPRESENTATION
    # ===================================================================
    
    def __repr__(self) -> str:
        return (f"LazyTensor(op={self.operation}, "
                f"shape={self.shape}, dtype={self.dtype})")
    
    def __str__(self) -> str:
        return self.__repr__()
```

**Key Improvements:**
1. ✅ Proper `torch.Tensor` subclass (uses `_make_subclass`)
2. ✅ `__torch_dispatch__` automatically called by PyTorch
3. ✅ FakeTensor for accurate shape inference
4. ✅ Materialization only when needed
5. ✅ Compatible with all PyTorch operations

#### 1.2 Clean Factory Interception

**File: `genie/core/factory_interceptor.py` (NEW)**

```python
"""
Factory function interceptor for tensor creation operations.

This module wraps torch.randn, torch.zeros, etc. to return LazyTensors
when device='remote_accelerator' is specified.

Why necessary: Factory functions don't have LazyTensor arguments, so
__torch_dispatch__ won't be called. We need explicit namespace wrapping.
"""

import torch
import functools
import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class FactoryInterceptor:
    """
    Intercepts PyTorch tensor creation functions.
    
    Wraps functions in the torch namespace to return LazyTensors when
    the device argument specifies 'remote_accelerator'.
    """
    
    # Functions to intercept (complete list)
    FACTORY_FUNCTIONS = [
        # Basic creation
        'randn', 'rand', 'randint', 'randn_like', 'rand_like', 'randint_like',
        'zeros', 'ones', 'empty', 'full',
        'zeros_like', 'ones_like', 'empty_like', 'full_like',

        # Data conversion
        'tensor', 'as_tensor', 'from_numpy',

        # Special constructors
        'eye', 'arange', 'linspace', 'logspace',

        # Random distributions
        'normal', 'randperm',
    ]
    
    def __init__(self):
        self._wrapped = False
        self._original_functions: Dict[str, Callable] = {}
    
    def wrap(self):
        """Wrap all factory functions in torch namespace."""
        if self._wrapped:
            logger.warning("Factory functions already wrapped")
            return
        
        for func_name in self.FACTORY_FUNCTIONS:
            if not hasattr(torch, func_name):
                logger.debug(f"torch.{func_name} not found, skipping")
                continue
            
            # Store original
            self._original_functions[func_name] = getattr(torch, func_name)
            
            # Create wrapper
            wrapper = self._create_wrapper(func_name)
            
            # Replace in torch namespace
            setattr(torch, func_name, wrapper)
        
        self._wrapped = True
        logger.info(f"Wrapped {len(self._original_functions)} factory functions")
    
    def unwrap(self):
        """Restore original factory functions (for testing)."""
        if not self._wrapped:
            return
        
        for func_name, original_func in self._original_functions.items():
            setattr(torch, func_name, original_func)
        
        self._wrapped = False
        self._original_functions.clear()
    
    def _create_wrapper(self, func_name: str) -> Callable:
        """Create wrapper function for a factory function."""
        original_func = self._original_functions[func_name]

        @functools.wraps(original_func)
        def wrapper(*args, **kwargs):
            # Check BOTH device argument AND capture context
            device = kwargs.get('device')

            # Import here to avoid circular dependency
            from .capture import is_capturing

            # Return LazyTensor if EITHER condition is true:
            # 1. Device explicitly set to remote_accelerator (paper API)
            # 2. Inside genie.capture() context (convenience API)
            if self._is_remote_device(device) or is_capturing():
                from .lazy_tensor import LazyTensor

                # Call LazyTensor factory method
                lazy_factory = getattr(LazyTensor, func_name, None)
                if lazy_factory is not None:
                    return lazy_factory(*args, **kwargs)
                else:
                    # Fallback: generic LazyTensor creation
                    return LazyTensor(
                        operation=f'aten::{func_name}',
                        inputs=list(args),
                        kwargs=kwargs
                    )

            # Only execute normally if BOTH are false:
            # - Device is NOT remote_accelerator
            # - NOT inside capture context
            return original_func(*args, **kwargs)

        return wrapper
    
    @staticmethod
    def _is_remote_device(device: Any) -> bool:
        """Check if device is remote_accelerator."""
        if device is None:
            return False
        
        device_str = str(device)
        return ('remote_accelerator' in device_str or 
                'privateuseone' in device_str)


# Global interceptor instance
_factory_interceptor = FactoryInterceptor()


def wrap_factories():
    """Wrap factory functions (call once at initialization)."""
    _factory_interceptor.wrap()


def unwrap_factories():
    """Unwrap factory functions (for testing)."""
    _factory_interceptor.unwrap()
```

**Key Points:**
- ✅ Explicit about why this is needed (factory functions)
- ✅ Complete list of ~20 functions
- ✅ Unwrap support for testing
- ✅ Clean separation from LazyTensor

#### 1.3 Remove library.py

**Action: Delete `genie/core/library.py`**

**Rationale:**
- `torch.library.impl` registrations are redundant
- Factory interceptor + `__torch_dispatch__` provide complete coverage
- Reduces complexity from 3 mechanisms to 2

**Migration:**
```python
# Old: library.py registrations (DELETE)
@register_operation_impl("add.Tensor")
def add_impl(...):
    ...

# New: Automatic via __torch_dispatch__ (NO CODE NEEDED)
# LazyTensor.__torch_dispatch__ handles add automatically
```

#### 1.4 Initialization Cleanup

**File: `genie/__init__.py`**

```python
"""
Genie: Semantic Disaggregated Execution Framework

Clean initialization:
1. Register device backend (optional, non-blocking)
2. Wrap factory functions
3. Set up LazyTensor graph builder
"""

import logging
logger = logging.getLogger(__name__)

# Version
__version__ = "0.2.0"

# Core interception setup
def _initialize():
    """Initialize Genie interception layer."""
    try:
        # Step 1: Try to register C++ backend (optional)
        try:
            from . import _C
            _C.register_remote_accelerator_device()
            logger.info("C++ backend registered")
        except ImportError:
            logger.info("C++ backend not available (Python-only mode)")
        
        # Step 2: Wrap factory functions (REQUIRED)
        from .core.factory_interceptor import wrap_factories
        wrap_factories()
        
        # Step 3: Initialize graph builder
        from .core.graph_builder import initialize_global_builder
        initialize_global_builder()
        
        logger.info("Genie initialized successfully")
        
    except Exception as e:
        logger.error(f"Genie initialization failed: {e}")
        raise

# Initialize on import
_initialize()

# Public API
from .core.lazy_tensor import LazyTensor
from .core.capture import capture, get_graph
from .semantic import annotate_graph

__all__ = [
    'LazyTensor',
    'capture',
    'get_graph',
    'annotate_graph',
]
```

---

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

*This section provides complete implementation of semantic analysis layer.*

#### 3.1 Pattern Matcher Framework

**File: `genie/semantic/patterns/base_matcher.py` (NEW)**

```python
"""
Base class for pattern matchers.

Pattern matchers identify high-level computational patterns in the computation
graph and annotate nodes with semantic metadata.

Each pattern matcher is responsible for:
1. Identifying a specific pattern (e.g., attention, convolution)
2. Locating all occurrences in the graph
3. Annotating matched nodes with semantic metadata
4. Providing hints for optimization
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Pattern:
    """Represents a detected pattern in the computation graph."""
    name: str
    nodes: List[Any]  # List of GraphNode
    metadata: Dict[str, Any]


class PatternMatcher(ABC):
    """
    Abstract base class for pattern matchers.
    
    Subclasses implement specific pattern detection logic.
    """
    
    @property
    @abstractmethod
    def pattern_name(self) -> str:
        """Name of the pattern being detected."""
        pass
    
    @abstractmethod
    def match(self, graph) -> List[Pattern]:
        """
        Detect pattern occurrences in graph.
        
        Args:
            graph: Unified Graph interface (FX or LazyDAG)
        
        Returns:
            List of detected Pattern instances
        """
        pass
    
    def is_match(self, node) -> bool:
        """
        Check if a single node matches this pattern.
        
        Override for simple node-level matching.
        """
        return False


class PatternRegistry:
    """
    Registry of all pattern matchers.
    
    Manages the lifecycle of pattern matchers and coordinates detection.
    """
    
    def __init__(self):
        self._matchers: List[PatternMatcher] = []
        self._patterns_cache = {}
    
    def register(self, matcher: PatternMatcher):
        """Register a new pattern matcher."""
        self._matchers.append(matcher)
    
    def match_all(self, graph) -> Dict[str, List[Pattern]]:
        """
        Run all pattern matchers on graph.
        
        Returns:
            Dict mapping pattern name → list of detected patterns
        """
        results = {}
        for matcher in self._matchers:
            try:
                patterns = matcher.match(graph)
                results[matcher.pattern_name] = patterns
            except Exception as e:
                logger.warning(f"Pattern matcher {matcher.pattern_name} failed: {e}")
        
        return results


# Global pattern registry
_global_pattern_registry = PatternRegistry()


def get_pattern_registry() -> PatternRegistry:
    """Get the global pattern registry."""
    return _global_pattern_registry
```

#### 3.2 Specific Pattern Matchers

**File: `genie/semantic/patterns/attention_matcher.py` (NEW)**

```python
"""
Matcher for attention mechanisms (self-attention, cross-attention).

Detects patterns like:
- Multi-head attention: Q@K.T softmax @ V
- Query-Key-Value projections
- Positional encodings
"""

import torch
import logging
from typing import List, Dict, Any
from .base_matcher import PatternMatcher, Pattern

logger = logging.getLogger(__name__)


class AttentionMatcher(PatternMatcher):
    """Detects attention patterns in computation graphs."""
    
    @property
    def pattern_name(self) -> str:
        return "attention"
    
    def match(self, graph) -> List[Pattern]:
        """
        Detect attention patterns in graph.
        
        Key indicators:
        1. Batch matrix multiplication (Q @ K.T)
        2. Softmax normalization
        3. Weighted sum (scores @ V)
        4. Linear projections (QKV)
        """
        patterns = []
        nodes = graph.nodes()
        
        # Look for softmax operations (characteristic of attention)
        for i, node in enumerate(nodes):
            if self._is_softmax_like(node):
                # Found potential attention - backtrack to find Q@K.T pattern
                matmul_nodes = self._find_matmul_chain(node, nodes, i)
                if matmul_nodes:
                    # Found attention pattern
                    pattern = self._extract_attention_pattern(matmul_nodes, node)
                    if pattern:
                        patterns.append(pattern)
        
        return patterns
    
    def _is_softmax_like(self, node) -> bool:
        """Check if node is softmax or related normalization."""
        op = node.operation.lower()
        return any(x in op for x in ['softmax', 'log_softmax', 'exp', 'normalize'])
    
    def _find_matmul_chain(self, softmax_node, nodes, idx) -> List[Any]:
        """Trace back from softmax to find Q@K.T @ V pattern."""
        # Implementation would traverse inputs looking for:
        # 1. Softmax input = Q @ K.T (matmul)
        # 2. Softmax output = scores @ V (matmul)
        return []  # Simplified for space
    
    def _extract_attention_pattern(self, matmul_nodes, softmax_node) -> Optional[Pattern]:
        """Extract attention pattern with metadata."""
        # Determine attention type
        attention_type = self._classify_attention(matmul_nodes)
        
        # All nodes involved in attention
        all_nodes = matmul_nodes + [softmax_node]
        
        return Pattern(
            name="attention",
            nodes=all_nodes,
            metadata={
                'semantic_role': 'attention',
                'attention_type': attention_type,  # 'self' or 'cross'
                'modality': self._infer_modality(matmul_nodes),
                'parallelizable': True,
                'memory_intensive': True,
                'compute_bound': True,
            }
        )
    
    def _classify_attention(self, matmul_nodes) -> str:
        """Determine if self-attention or cross-attention."""
        # Cross-attention if Q and K have different sources
        # This would check metadata on input nodes
        return "self_attention"  # Simplified
    
    def _infer_modality(self, nodes) -> str:
        """Infer if this is vision, text, or multimodal attention."""
        # Check metadata from preceding operations
        return "text"  # Simplified


class ConvolutionMatcher(PatternMatcher):
    """Detects convolutional patterns."""
    
    @property
    def pattern_name(self) -> str:
        return "convolution"
    
    def match(self, graph) -> List[Pattern]:
        """Detect convolutional patterns."""
        patterns = []
        
        for node in graph.nodes():
            if self._is_conv_like(node):
                pattern = self._extract_conv_pattern(node)
                patterns.append(pattern)
        
        return patterns
    
    def _is_conv_like(self, node) -> bool:
        """Check if node is convolution operation."""
        op = node.operation.lower()
        return any(x in op for x in ['conv', 'convolution', 'conv1d', 'conv2d', 'conv3d'])
    
    def _extract_conv_pattern(self, conv_node) -> Pattern:
        """Extract convolution pattern."""
        return Pattern(
            name="convolution",
            nodes=[conv_node],
            metadata={
                'semantic_role': 'convolution',
                'modality': 'vision',
                'can_fuse_bn': self._can_fuse_with_bn(conv_node),
                'can_pipeline': True,
                'memory_footprint': 'high',
                'optimization_hint': 'fusion_opportunity',
            }
        )
    
    def _can_fuse_with_bn(self, conv_node) -> bool:
        """Check if conv can be fused with BatchNorm."""
        # Check if next operation is BatchNorm
        return False  # Simplified


class KVCacheMatcher(PatternMatcher):
    """Detects KV cache accumulation patterns (LLM decode phase)."""
    
    @property
    def pattern_name(self) -> str:
        return "kv_cache"
    
    def match(self, graph) -> List[Pattern]:
        """
        Detect KV cache patterns.
        
        Signature:
            cache_t+1 = torch.cat([cache_t, new_kv], dim=seq_dim)
        
        Where cache feeds back into next iteration.
        """
        patterns = []
        
        for node in graph.nodes():
            if self._is_cat_like(node):
                if self._is_recurrent(node, graph):
                    pattern = self._extract_kv_cache_pattern(node)
                    patterns.append(pattern)
        
        return patterns
    
    def _is_cat_like(self, node) -> bool:
        """Check if node is concatenation."""
        op = node.operation.lower()
        return 'cat' in op or 'concat' in op
    
    def _is_recurrent(self, cat_node, graph) -> bool:
        """Check if output feeds back as input (recurrent pattern)."""
        # Would check graph structure for cycles
        return False  # Simplified
    
    def _extract_kv_cache_pattern(self, cat_node) -> Pattern:
        """Extract KV cache pattern."""
        return Pattern(
            name="kv_cache",
            nodes=[cat_node],
            metadata={
                'execution_phase': 'llm_decode',
                'residency': 'persistent_kv_cache',
                'requires_colocation': True,
                'sequential': True,
                'memory_intensive': True,
                'optimization_hint': 'colocate_with_decoder',
            }
        )
```

#### 3.3 Phase Detection

**File: `genie/semantic/phase_detector.py` (NEW)**

```python
"""
Phase detection for execution phases.

Identifies phases like:
- LLM prefill (parallel attention over sequence)
- LLM decode (sequential generation with KV cache)
- Forward propagation
- Backward propagation (future)
"""

from enum import Enum
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class ExecutionPhase(Enum):
    """Execution phase classification."""
    LLM_PREFILL = "llm_prefill"
    LLM_DECODE = "llm_decode"
    FORWARD = "forward"
    BACKWARD = "backward"
    INIT = "initialization"
    UNKNOWN = "unknown"


class PhaseDetector:
    """
    Detects execution phases in computation graphs.
    
    Uses pattern matches and heuristics to classify operations into phases.
    """
    
    def __init__(self, pattern_registry):
        self.pattern_registry = pattern_registry
    
    def detect_phases(self, graph) -> Dict[str, ExecutionPhase]:
        """
        Detect phases for all nodes in graph.
        
        Returns:
            Dict mapping node_id → ExecutionPhase
        """
        phases = {}
        patterns = self.pattern_registry.match_all(graph)
        
        # Phase 1: Detect KV cache (indicates decode)
        if "kv_cache" in patterns and patterns["kv_cache"]:
            decode_nodes = self._mark_decode_phase(patterns["kv_cache"], graph)
            phases.update(decode_nodes)
        
        # Phase 2: Detect parallel attention (indicates prefill)
        prefill_nodes = self._mark_prefill_phase(patterns.get("attention", []), graph)
        phases.update(prefill_nodes)
        
        # Phase 3: Mark remaining as forward
        for node in graph.nodes():
            if node.id not in phases:
                phases[node.id] = ExecutionPhase.FORWARD
        
        return phases
    
    def _mark_decode_phase(self, kv_cache_patterns, graph) -> Dict[str, ExecutionPhase]:
        """Mark nodes in decode phase."""
        phases = {}
        
        for pattern in kv_cache_patterns:
            for node in pattern.nodes:
                phases[node.id] = ExecutionPhase.LLM_DECODE
                
                # Also mark attention nodes connected to KV cache as decode
                # (they're part of the decode loop)
        
        return phases
    
    def _mark_prefill_phase(self, attention_patterns, graph) -> Dict[str, ExecutionPhase]:
        """Mark nodes in prefill phase."""
        phases = {}
        
        for pattern in attention_patterns:
            # Check if this is parallel attention (prefill)
            if self._is_parallel_attention(pattern):
                for node in pattern.nodes:
                    phases[node.id] = ExecutionPhase.LLM_PREFILL
        
        return phases
    
    def _is_parallel_attention(self, pattern) -> bool:
        """Check if attention processes multiple positions in parallel."""
        # Heuristic: If batch size in sequence dimension > 1
        return True  # Simplified


class PhaseAnnotator:
    """
    Annotates graph nodes with phase information.
    
    Stores phase annotations in node metadata.
    """
    
    def __init__(self, phase_detector):
        self.phase_detector = phase_detector
    
    def annotate(self, graph):
        """
        Annotate graph nodes with phase information.
        
        Modifies node metadata in-place.
        """
        phases = self.phase_detector.detect_phases(graph)
        
        for node in graph.nodes():
            phase = phases.get(node.id, ExecutionPhase.UNKNOWN)
            if 'phase' not in node.metadata:
                node.metadata['phase'] = phase
```

#### 3.4 Cost Estimation

**File: `genie/semantic/cost_estimator.py` (NEW)**

```python
"""
Cost estimation for operations.

Estimates:
- Compute cost (FLOPs)
- Memory footprint (bytes)
- Operational intensity (FLOPs/byte)
- Data movement cost
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class CostEstimate:
    """Cost estimate for an operation."""
    compute_flops: float  # Floating point operations
    memory_bytes: float   # Memory footprint
    operational_intensity: float  # FLOPs per byte
    data_movement_bytes: float  # Input/output data volume


class CostEstimator:
    """
    Estimates computational costs for operations.
    
    Uses shape information and operation type to estimate costs.
    """
    
    def estimate_operation(self, node) -> CostEstimate:
        """
        Estimate cost for a single operation.
        
        Args:
            node: GraphNode with shape and dtype information
        
        Returns:
            CostEstimate with compute, memory, and intensity metrics
        """
        op = node.operation.lower()
        
        # Dispatch to operation-specific estimator
        if 'matmul' in op or 'mm' in op:
            return self._estimate_matmul(node)
        elif 'conv' in op:
            return self._estimate_conv(node)
        elif 'attention' in node.metadata.get('semantic_role', ''):
            return self._estimate_attention(node)
        elif any(x in op for x in ['add', 'sub', 'mul', 'div']):
            return self._estimate_elementwise(node)
        elif 'softmax' in op or 'sigmoid' in op:
            return self._estimate_pointwise(node)
        else:
            return self._estimate_generic(node)
    
    def _estimate_matmul(self, node) -> CostEstimate:
        """Estimate cost for matrix multiplication."""
        # Get shapes from inputs
        inputs = node.inputs
        if len(inputs) < 2:
            return self._estimate_generic(node)
        
        # For A @ B: compute = 2 * M * N * K
        # (M, K) @ (K, N) -> (M, N)
        shape_a = getattr(inputs[0], 'shape', None)
        shape_b = getattr(inputs[1], 'shape', None)
        
        if shape_a and shape_b:
            m, k = shape_a[-2], shape_a[-1]
            n = shape_b[-1]
            compute_flops = 2 * m * n * k
            
            # Memory: inputs + output
            dtype_bytes = 4  # float32
            input_bytes = (m * k + k * n + m * n) * dtype_bytes
            
            return CostEstimate(
                compute_flops=float(compute_flops),
                memory_bytes=float(input_bytes),
                operational_intensity=compute_flops / input_bytes if input_bytes > 0 else 0,
                data_movement_bytes=float(input_bytes)
            )
        
        return self._estimate_generic(node)
    
    def _estimate_conv(self, node) -> CostEstimate:
        """Estimate cost for convolution."""
        # Conv2d: FLOPs = 2 * H_out * W_out * K_h * K_w * C_in * C_out
        # Simplified for now
        return CostEstimate(
            compute_flops=1e6,  # Placeholder
            memory_bytes=1e6,
            operational_intensity=1.0,
            data_movement_bytes=1e6
        )
    
    def _estimate_attention(self, node) -> CostEstimate:
        """Estimate cost for attention operation."""
        # Attention: Q @ K.T (O(n^2)) + softmax (O(n)) + @ V (O(n^2))
        # Very high memory and compute intensity
        return CostEstimate(
            compute_flops=1e9,  # High compute
            memory_bytes=1e7,
            operational_intensity=100.0,  # Very compute-bound
            data_movement_bytes=1e7
        )
    
    def _estimate_elementwise(self, node) -> CostEstimate:
        """Estimate cost for element-wise operations."""
        shape = getattr(node, 'shape', None)
        if shape:
            elements = 1
            for dim in shape:
                elements *= dim
            
            # Element-wise: 1 FLOP per element
            compute_flops = elements
            memory_bytes = elements * 4  # float32
            
            return CostEstimate(
                compute_flops=float(compute_flops),
                memory_bytes=float(memory_bytes),
                operational_intensity=1.0,  # Low intensity
                data_movement_bytes=float(memory_bytes)
            )
        
        return self._estimate_generic(node)
    
    def _estimate_pointwise(self, node) -> CostEstimate:
        """Estimate cost for pointwise operations (softmax, sigmoid, etc.)."""
        return self._estimate_elementwise(node)
    
    def _estimate_generic(self, node) -> CostEstimate:
        """Generic cost estimate (conservative)."""
        return CostEstimate(
            compute_flops=1000,  # Conservative estimate
            memory_bytes=1000,
            operational_intensity=1.0,
            data_movement_bytes=1000
        )


class GraphCostEstimator:
    """
    Estimates total cost for a computation graph.
    
    Aggregates costs of individual operations and estimates
    data movement between operations.
    """
    
    def __init__(self):
        self.operation_estimator = CostEstimator()
    
    def estimate_graph(self, graph) -> Dict[str, Any]:
        """
        Estimate cost for entire graph.
        
        Returns:
            Dict with total costs and per-node estimates
        """
        costs = {}
        total_compute = 0
        total_memory = 0
        total_data_movement = 0
        
        for node in graph.nodes():
            node_cost = self.operation_estimator.estimate_operation(node)
            costs[node.id] = node_cost
            
            total_compute += node_cost.compute_flops
            total_memory += node_cost.memory_bytes
            total_data_movement += node_cost.data_movement_bytes
        
        return {
            'per_node': costs,
            'total_compute_flops': total_compute,
            'total_memory_bytes': total_memory,
            'total_data_movement_bytes': total_data_movement,
            'mean_operational_intensity': (total_compute / total_memory 
                                          if total_memory > 0 else 0),
        }
```

#### 3.5 Metadata Storage & Annotation API

**File: `genie/semantic/metadata_registry.py` (NEW)**

```python
"""
Metadata registry for storing semantic annotations.

Provides thread-safe storage for all semantic metadata about graph nodes.
"""

import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class NodeMetadata:
    """Semantic metadata for a graph node."""
    node_id: str
    
    # Core metadata
    phase: Optional[str] = None  # ExecutionPhase
    modality: Optional[str] = None  # vision, text, audio, fusion
    semantic_role: Optional[str] = None  # attention, conv, linear, etc.
    
    # Cost estimates
    compute_flops: float = 0.0
    memory_bytes: float = 0.0
    operational_intensity: float = 0.0
    data_movement_bytes: float = 0.0
    
    # Optimization hints
    optimization_hints: Dict[str, Any] = field(default_factory=dict)
    
    # Dependencies
    co_location_required: bool = False
    co_location_targets: list = field(default_factory=list)
    
    # Additional metadata
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'node_id': self.node_id,
            'phase': self.phase,
            'modality': self.modality,
            'semantic_role': self.semantic_role,
            'compute_flops': self.compute_flops,
            'memory_bytes': self.memory_bytes,
            'operational_intensity': self.operational_intensity,
            'data_movement_bytes': self.data_movement_bytes,
            'optimization_hints': self.optimization_hints,
            'co_location_required': self.co_location_required,
            'co_location_targets': self.co_location_targets,
            'extra': self.extra,
        }


class MetadataRegistry:
    """
    Thread-safe registry of semantic metadata for all nodes.
    """
    
    def __init__(self):
        self._metadata: Dict[str, NodeMetadata] = {}
        self._lock = threading.RLock()
    
    def register_metadata(self, node_id: str, metadata: NodeMetadata):
        """Register metadata for a node."""
        with self._lock:
            self._metadata[node_id] = metadata
    
    def get_metadata(self, node_id: str) -> Optional[NodeMetadata]:
        """Get metadata for a node."""
        with self._lock:
            return self._metadata.get(node_id)
    
    def update_metadata(self, node_id: str, updates: Dict[str, Any]):
        """Update metadata for a node."""
        with self._lock:
            if node_id in self._metadata:
                metadata = self._metadata[node_id]
                for key, value in updates.items():
                    if hasattr(metadata, key):
                        setattr(metadata, key, value)
    
    def get_all_metadata(self) -> Dict[str, NodeMetadata]:
        """Get all registered metadata."""
        with self._lock:
            return dict(self._metadata)
    
    def clear(self):
        """Clear all metadata."""
        with self._lock:
            self._metadata.clear()


# Global metadata registry
_global_metadata_registry = MetadataRegistry()


def get_metadata_registry() -> MetadataRegistry:
    """Get the global metadata registry."""
    return _global_metadata_registry
```

**File: `genie/semantic/annotator.py` (NEW)**

```python
"""
Graph annotator: Combines all semantic analysis to produce annotated graph.

Main public API for semantic analysis.
"""

import logging
from typing import Optional
from .patterns.base_matcher import get_pattern_registry
from .phase_detector import PhaseDetector, PhaseAnnotator
from .cost_estimator import GraphCostEstimator
from .metadata_registry import (
    get_metadata_registry, NodeMetadata, MetadataRegistry
)

logger = logging.getLogger(__name__)


class SemanticAnnotator:
    """
    Annotates computation graphs with semantic information.
    
    Combines:
    1. Pattern detection (attention, convolution, KV cache)
    2. Phase detection (prefill, decode, forward)
    3. Cost estimation (FLOPs, memory, intensity)
    4. Metadata storage
    """
    
    def __init__(self):
        self.pattern_registry = get_pattern_registry()
        self.phase_detector = PhaseDetector(self.pattern_registry)
        self.phase_annotator = PhaseAnnotator(self.phase_detector)
        self.cost_estimator = GraphCostEstimator()
        self.metadata_registry = get_metadata_registry()
    
    def annotate(self, graph) -> 'AnnotatedGraph':
        """
        Fully annotate a computation graph.
        
        Args:
            graph: Unified Graph (FX or LazyDAG)
        
        Returns:
            AnnotatedGraph with semantic metadata
        """
        logger.info(f"Starting semantic analysis for {len(graph.nodes())} nodes")
        
        # Step 1: Detect patterns
        logger.info("Detecting patterns...")
        patterns = self.pattern_registry.match_all(graph)
        for pattern_name, pattern_list in patterns.items():
            logger.debug(f"  Found {len(pattern_list)} {pattern_name} patterns")
        
        # Step 2: Detect phases
        logger.info("Detecting execution phases...")
        phases = self.phase_detector.detect_phases(graph)
        logger.debug(f"  Phase distribution: {self._count_phases(phases)}")
        
        # Step 3: Estimate costs
        logger.info("Estimating operation costs...")
        costs = self.cost_estimator.estimate_graph(graph)
        logger.debug(f"  Total FLOPs: {costs['total_compute_flops']:.2e}")
        logger.debug(f"  Total Memory: {costs['total_memory_bytes']:.2e} bytes")
        
        # Step 4: Create metadata for each node
        logger.info("Creating node metadata...")
        for node in graph.nodes():
            metadata = NodeMetadata(
                node_id=node.id,
                phase=phases.get(node.id).value if node.id in phases else None,
                semantic_role=node.metadata.get('semantic_role'),
                modality=node.metadata.get('modality'),
            )
            
            # Add cost information
            if node.id in costs['per_node']:
                cost = costs['per_node'][node.id]
                metadata.compute_flops = cost.compute_flops
                metadata.memory_bytes = cost.memory_bytes
                metadata.operational_intensity = cost.operational_intensity
                metadata.data_movement_bytes = cost.data_movement_bytes
            
            # Add optimization hints
            metadata.optimization_hints = node.metadata.get('optimization_hints', {})
            
            # Register metadata
            self.metadata_registry.register_metadata(node.id, metadata)
        
        logger.info("Semantic analysis complete")
        
        return AnnotatedGraph(graph, patterns, phases, costs, self.metadata_registry)
    
    def _count_phases(self, phases) -> dict:
        """Count nodes by phase."""
        counts = {}
        for phase in phases.values():
            phase_name = phase.value if hasattr(phase, 'value') else str(phase)
            counts[phase_name] = counts.get(phase_name, 0) + 1
        return counts


class AnnotatedGraph:
    """
    Computation graph with semantic annotations.
    
    Wrapper around a Graph that includes semantic metadata.
    """
    
    def __init__(self, base_graph, patterns, phases, costs, metadata_registry):
        self.base_graph = base_graph
        self.patterns = patterns
        self.phases = phases
        self.costs = costs
        self.metadata_registry = metadata_registry
    
    def nodes(self):
        """Get all annotated nodes."""
        return self.base_graph.nodes()
    
    def get_node(self, node_id):
        """Get node by ID."""
        return self.base_graph.get_node(node_id)
    
    def get_metadata(self, node_id) -> Optional[NodeMetadata]:
        """Get metadata for a node."""
        return self.metadata_registry.get_metadata(node_id)
    
    @property
    def backend_type(self):
        """Backend type (FX or lazy_dag)."""
        return self.base_graph.backend_type
    
    def summary(self) -> str:
        """Get human-readable summary of annotations."""
        lines = [
            f"AnnotatedGraph ({self.backend_type} backend)",
            f"  Total nodes: {len(list(self.nodes()))}",
            f"  Total compute: {self.costs['total_compute_flops']:.2e} FLOPs",
            f"  Total memory: {self.costs['total_memory_bytes']:.2e} bytes",
            f"  Mean intensity: {self.costs['mean_operational_intensity']:.2f} FLOPs/byte",
            f"Patterns detected:",
        ]
        
        for pattern_name, pattern_list in self.patterns.items():
            lines.append(f"  - {pattern_name}: {len(pattern_list)} occurrences")
        
        return "\n".join(lines)


# Main public API
def annotate_graph(graph) -> AnnotatedGraph:
    """
    Annotate a computation graph with semantic information.
    
    Convenience function - creates annotator and runs full analysis.
    
    Args:
        graph: Unified Graph (from genie.get_graph())
    
    Returns:
        AnnotatedGraph with full semantic metadata
    """
    annotator = SemanticAnnotator()
    return annotator.annotate(graph)
```

---

#### 3.6 Phase 3 Integration

**Update File: `genie/__init__.py`**

```python
# ... existing imports ...

# Public API for semantic analysis
from .semantic.annotator import annotate_graph, AnnotatedGraph
from .semantic.patterns.base_matcher import get_pattern_registry

__all__ = [
    # ... existing exports ...
    'annotate_graph',
    'AnnotatedGraph',
    'get_pattern_registry',
]
```

---

**Milestone 3 Deliverables:**
- ✅ Pattern detection framework (base class + matchers)
- ✅ Phase detection (prefill/decode classification)
- ✅ Cost estimation (FLOPs, memory, intensity)
- ✅ Metadata registry (thread-safe storage)
- ✅ Annotation API (main public interface)
- ✅ Semantic annotations on all graph nodes

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

**Phase 1: Core Architecture (Weeks 1-4)**

| Week | Task | Deliverable | Risk |
|------|------|-------------|------|
| 1 | LazyTensor subclass | Proper torch.Tensor subclass with __torch_dispatch__ | Low |
| 2 | Factory interceptor | Clean wrapping of ~25 factory functions (including tensor, as_tensor, from_numpy) | Low |
| 2 | Remove library.py | Delete redundant mechanism | Low |
| 2 | Capture context manager | Thread-local state signaling for transparent interception | Low |
| 3 | Graph interface | Unified Graph/Node abstractions | Medium |
| 4 | Testing & docs | Unit tests, API docs, performance benchmarks | Low |

**Milestone 1:** Clean interception layer with 2 mechanisms supporting both API styles, >90% test coverage

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
- ✅ <1μs per-operation overhead
- ✅ >95% model compatibility (FX + LazyDAG)
- ✅ <10% end-to-end disaggregation overhead
- ✅ All paper results reproduced
- ✅ Support for both device-based and context-based APIs
- ✅ Transparent interception with thread-local capture context

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

1. **Week 1:** Team alignment on this plan (incorporating peer review fixes)
2. **Week 1-2:** Set up repository structure, CI/CD
3. **Week 2-4:** Implement Phase 1 (core architecture with capture context fixes)
4. **Week 4:** Review progress, adjust timeline

**Key Decision Points:**

1. ✅ **Factory wrapping is necessary** (confirmed)
2. ✅ **Target 3-tier deployment** (local → cluster → RDMA)
3. ✅ **Open-source with Apache 2.0 license**
4. ✅ **Support both API styles** (device-based + context-based)
5. ✅ **Thread-local capture context signaling** (critical fix implemented)
6. ⏳ **Timeline: 24 weeks to v1.0** (aggressive but achievable)

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