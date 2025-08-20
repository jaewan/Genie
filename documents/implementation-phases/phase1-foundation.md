# Phase 1: Foundation Implementation Guide

## Overview
Phase 1 establishes the core infrastructure for Genie, focusing on PyTorch integration and basic LazyTensor functionality. This phase proves feasibility and creates the foundation for semantic capture.

## Timeline: Weeks 1-8

## Success Criteria
- [ ] PyTorch device registration working
- [ ] Basic LazyTensor with 10+ operations
- [ ] Simple graph construction
- [ ] <10μs operation overhead
- [ ] Basic pattern recognition (2 types)
- [ ] Single GPU remote execution demo

## Week 1-2: Project Setup

### Task 1.1: Development Environment
```bash
# Setup commands
git init genie
cd genie
python3.10 -m venv venv
source venv/bin/activate
pip install torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install pytest black mypy pre-commit

# Project structure
mkdir -p genie/{core,patterns,runtime,tests}
touch genie/__init__.py
```

### Task 1.2: Build System
```python
# setup.py
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='genie',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            'genie._C',
            ['genie/csrc/device.cpp'],
            extra_compile_args=['-std=c++17']
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=[
        'torch>=2.1.0,<2.2.0',
        'numpy>=1.24.0',
        'networkx>=3.0',
    ]
)
```

## Week 3-4: PyTorch Device Registration

### Task 2.1: Custom Device Implementation
```python
# genie/core/device.py
import torch
from typing import Optional

class RemoteAcceleratorDevice:
    """Custom PyTorch device for disaggregated execution"""
    
    _devices = {}  # Registry of device instances
    
    def __init__(self, index: int = 0):
        self.type = "remote_accelerator"
        self.index = index
        self._register_with_pytorch()
        
    def _register_with_pytorch(self):
        # Register device type with PyTorch
        if not hasattr(torch._C, '_remote_accelerator_registered'):
            torch._C._register_remote_accelerator_device()
            torch._C._remote_accelerator_registered = True
        
        # Store in registry
        self._devices[self.index] = self
    
    @classmethod
    def get_device(cls, index: int = 0) -> 'RemoteAcceleratorDevice':
        if index not in cls._devices:
            cls._devices[index] = cls(index)
        return cls._devices[index]
    
    def __repr__(self):
        return f"remote_accelerator:{self.index}"

# C++ extension for device registration
# genie/csrc/device.cpp
#include <torch/extension.h>

void register_remote_accelerator_device() {
    // Register with PyTorch's device registry
    c10::DeviceType device_type = c10::DeviceType::PrivateUse1;
    // Set device name
    c10::SetDeviceTypeName(device_type, "remote_accelerator");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("register_remote_accelerator_device", 
          &register_remote_accelerator_device);
}
```

### Task 2.2: Dispatcher Hooks
```python
# genie/core/dispatcher.py
import torch
import functools
from typing import Any, Callable

class DispatcherIntegration:
    """Integrates with PyTorch dispatcher for operation interception"""
    
    def __init__(self):
        self.registered_ops = {}
        self.lazy_mode = True
        
    def register_op(self, op_name: str):
        """Decorator to register operation handler"""
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if self.lazy_mode:
                    # Create LazyTensor instead of executing
                    return self._create_lazy_tensor(op_name, args, kwargs)
                else:
                    # Fallback to eager execution
                    return func(*args, **kwargs)
            
            # Register with PyTorch
            torch.library.impl(op_name, "remote_accelerator")(wrapper)
            self.registered_ops[op_name] = wrapper
            return wrapper
        return decorator
    
    def _create_lazy_tensor(self, op_name: str, args, kwargs):
        from genie.core.lazy_tensor import LazyTensor
        return LazyTensor(
            operation=op_name,
            inputs=args,
            kwargs=kwargs
        )

# Usage
dispatcher = DispatcherIntegration()

@dispatcher.register_op("aten::add")
def add_impl(x, y, *, alpha=1):
    # This won't execute in lazy mode
    return torch.add(x, y, alpha=alpha)

@dispatcher.register_op("aten::matmul")
def matmul_impl(x, y):
    return torch.matmul(x, y)
```

## Week 5-6: LazyTensor Implementation

### Task 3.1: LazyTensor Core
```python
# genie/core/lazy_tensor.py
import torch
from dataclasses import dataclass
from typing import Any, List, Optional, Dict
from uuid import uuid4

@dataclass
class LazyTensor:
    """Deferred execution tensor with semantic metadata"""
    
    # Core fields
    id: str
    operation: str
    inputs: List[Any]
    kwargs: Dict[str, Any]
    
    # Metadata
    shape: Optional[torch.Size] = None
    dtype: Optional[torch.dtype] = None
    device: str = "remote_accelerator:0"
    
    # Execution state
    materialized: bool = False
    concrete_value: Optional[torch.Tensor] = None
    
    def __init__(self, operation: str, inputs: List, kwargs: Dict = None):
        self.id = str(uuid4())
        self.operation = operation
        self.inputs = inputs
        self.kwargs = kwargs or {}
        
        # Infer properties
        self.shape = self._infer_shape()
        self.dtype = self._infer_dtype()
        
        # Register with graph
        from genie.core.graph import GraphBuilder
        GraphBuilder.current().add_tensor(self)
    
    def _infer_shape(self) -> Optional[torch.Size]:
        """Infer output shape from operation and inputs"""
        # Simplified shape inference
        if self.operation == "aten::add":
            if hasattr(self.inputs[0], 'shape'):
                return self.inputs[0].shape
        elif self.operation == "aten::matmul":
            if len(self.inputs) == 2:
                x_shape = getattr(self.inputs[0], 'shape', None)
                y_shape = getattr(self.inputs[1], 'shape', None)
                if x_shape and y_shape:
                    return torch.Size([x_shape[0], y_shape[-1]])
        return None
    
    def _infer_dtype(self) -> Optional[torch.dtype]:
        """Infer output dtype from inputs"""
        for inp in self.inputs:
            if hasattr(inp, 'dtype'):
                return inp.dtype
        return None
    
    def materialize(self) -> torch.Tensor:
        """Force materialization of this tensor"""
        if not self.materialized:
            from genie.core.executor import execute_subgraph
            self.concrete_value = execute_subgraph(self)
            self.materialized = True
        return self.concrete_value
    
    # Tensor-like interface
    def __repr__(self):
        return f"LazyTensor(op={self.operation}, shape={self.shape}, dtype={self.dtype})"
    
    def __add__(self, other):
        return LazyTensor("aten::add", [self, other])
    
    def __matmul__(self, other):
        return LazyTensor("aten::matmul", [self, other])
    
    def cpu(self):
        """Trigger materialization when moving to CPU"""
        return self.materialize().cpu()
    
    def item(self):
        """Trigger materialization for scalar conversion"""
        return self.materialize().item()
```

### Task 3.2: Graph Builder
```python
# genie/core/graph.py
from typing import Dict, List, Set
from dataclasses import dataclass
import threading

@dataclass
class ComputationNode:
    id: str
    operation: str
    inputs: List[str]  # IDs of input tensors
    outputs: List[str]  # IDs of output tensors
    metadata: Dict

@dataclass  
class ComputationGraph:
    nodes: Dict[str, ComputationNode]
    edges: List[tuple]  # (source_id, target_id)
    entry_points: Set[str]
    
    def topological_sort(self) -> List[str]:
        """Return nodes in execution order"""
        visited = set()
        stack = []
        
        def visit(node_id):
            if node_id in visited:
                return
            visited.add(node_id)
            node = self.nodes[node_id]
            for inp in node.inputs:
                if inp in self.nodes:
                    visit(inp)
            stack.append(node_id)
        
        for node_id in self.nodes:
            visit(node_id)
        
        return stack

class GraphBuilder:
    """Thread-local graph builder"""
    
    _thread_local = threading.local()
    
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.tensor_to_node = {}
    
    @classmethod
    def current(cls) -> 'GraphBuilder':
        """Get current thread's graph builder"""
        if not hasattr(cls._thread_local, 'builder'):
            cls._thread_local.builder = cls()
        return cls._thread_local.builder
    
    def add_tensor(self, lazy_tensor):
        """Add LazyTensor to graph"""
        # Create node for this operation
        node = ComputationNode(
            id=lazy_tensor.id,
            operation=lazy_tensor.operation,
            inputs=[self._get_tensor_id(inp) for inp in lazy_tensor.inputs],
            outputs=[lazy_tensor.id],
            metadata={
                'shape': lazy_tensor.shape,
                'dtype': lazy_tensor.dtype
            }
        )
        
        self.nodes[node.id] = node
        self.tensor_to_node[lazy_tensor.id] = node
        
        # Add edges from inputs
        for inp in lazy_tensor.inputs:
            if hasattr(inp, 'id'):
                self.edges.append((inp.id, lazy_tensor.id))
    
    def _get_tensor_id(self, tensor) -> str:
        """Get ID for tensor or create placeholder"""
        if hasattr(tensor, 'id'):
            return tensor.id
        # Create placeholder for concrete tensors
        return f"concrete_{id(tensor)}"
    
    def get_graph(self) -> ComputationGraph:
        """Build and return computation graph"""
        return ComputationGraph(
            nodes=self.nodes.copy(),
            edges=self.edges.copy(),
            entry_points={n for n in self.nodes if not any(e[1] == n for e in self.edges)}
        )
```

## Week 7-8: Basic Patterns and Testing

### Task 4.1: Simple Pattern Recognition
```python
# genie/patterns/base.py
from abc import ABC, abstractmethod
from typing import Optional

class PatternPlugin(ABC):
    """Base class for pattern recognition plugins"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def match(self, graph: ComputationGraph) -> Optional['PatternMatch']:
        pass

class PatternMatch:
    def __init__(self, pattern_name: str, confidence: float, nodes: List[str]):
        self.pattern_name = pattern_name
        self.confidence = confidence
        self.matched_nodes = nodes

# genie/patterns/matmul_pattern.py
class MatMulPattern(PatternPlugin):
    """Detect matrix multiplication patterns"""
    
    @property
    def name(self) -> str:
        return "matmul_chain"
    
    def match(self, graph: ComputationGraph) -> Optional[PatternMatch]:
        matmul_nodes = []
        
        for node_id, node in graph.nodes.items():
            if node.operation == "aten::matmul":
                matmul_nodes.append(node_id)
        
        if len(matmul_nodes) >= 2:
            # Check if they form a chain
            return PatternMatch(
                pattern_name=self.name,
                confidence=0.9,
                nodes=matmul_nodes
            )
        
        return None
```

### Task 4.2: Integration Tests
```python
# tests/test_integration.py
import torch
import pytest
from genie.core.device import RemoteAcceleratorDevice
from genie.core.lazy_tensor import LazyTensor

def test_device_registration():
    """Test custom device registration"""
    device = RemoteAcceleratorDevice.get_device(0)
    assert str(device) == "remote_accelerator:0"
    
    # Create tensor on custom device
    x = torch.randn(10, 10, device="remote_accelerator:0")
    assert isinstance(x, LazyTensor)

def test_lazy_execution():
    """Test lazy tensor creation and execution"""
    x = torch.randn(10, 10, device="remote_accelerator:0")
    y = torch.randn(10, 10, device="remote_accelerator:0")
    
    # Operations should create LazyTensors
    z = x + y
    assert isinstance(z, LazyTensor)
    assert not z.materialized
    
    # Moving to CPU should trigger materialization
    z_cpu = z.cpu()
    assert isinstance(z_cpu, torch.Tensor)
    assert z.materialized

def test_graph_construction():
    """Test computation graph building"""
    from genie.core.graph import GraphBuilder
    
    x = torch.randn(10, 10, device="remote_accelerator:0")
    y = torch.randn(10, 10, device="remote_accelerator:0")
    z = x @ y  # Matrix multiplication
    w = z + x  # Addition
    
    graph = GraphBuilder.current().get_graph()
    
    # Should have nodes for matmul and add
    assert len(graph.nodes) >= 2
    
    # Check topological order
    order = graph.topological_sort()
    assert len(order) > 0

@pytest.mark.benchmark
def test_overhead(benchmark):
    """Benchmark operation overhead"""
    def create_lazy_tensor():
        x = torch.randn(100, 100, device="remote_accelerator:0")
        y = torch.randn(100, 100, device="remote_accelerator:0")
        return x + y
    
    result = benchmark(create_lazy_tensor)
    assert isinstance(result, LazyTensor)
    
    # Check overhead is < 10μs
    assert benchmark.stats['mean'] < 10e-6
```

## Deliverables Checklist

### Code Deliverables
- [ ] PyTorch device registration (C++ and Python)
- [ ] Dispatcher integration with 10+ operations
- [ ] LazyTensor class with metadata
- [ ] Graph builder and representation
- [ ] 2 basic pattern recognizers
- [ ] Simple executor for materialization

### Test Deliverables  
- [ ] Unit tests for each component
- [ ] Integration tests for device registration
- [ ] Performance benchmarks
- [ ] Graph construction tests

### Documentation Deliverables
- [ ] API documentation for LazyTensor
- [ ] Device registration guide
- [ ] Pattern plugin tutorial
- [ ] Architecture overview

### Demo Deliverables
- [ ] Simple CNN (ResNet-18) running with LazyTensor
- [ ] Graph visualization of execution
- [ ] Performance comparison vs eager mode

## Success Metrics
- Operation interception overhead: <10μs (measure with pytest-benchmark)
- Memory overhead: <1% (measure tensor vs LazyTensor size)
- Test coverage: >85% (measure with pytest-cov)
- Pattern recognition: 2 patterns working

## Common Issues and Solutions

### Issue 1: Device Registration Fails
```python
# Solution: Ensure C++ extension is built
python setup.py build_ext --inplace
```

### Issue 2: Operation Not Intercepted
```python
# Solution: Register operation explicitly
@dispatcher.register_op("aten::your_op")
def your_op_impl(*args, **kwargs):
    return LazyTensor("aten::your_op", args, kwargs)
```

### Issue 3: Shape Inference Fails
```python
# Solution: Add operation-specific shape inference
def infer_shape_for_op(op_name, inputs):
    if op_name == "aten::conv2d":
        # Custom shape inference for conv2d
        return calculate_conv_output_shape(inputs)
```

## Next Phase Preview
Phase 2 will build on this foundation to add:
- Complete operation coverage (95%)
- FX integration for static analysis
- Advanced pattern recognition
- Semantic metadata collection
- Hook-based enrichment
