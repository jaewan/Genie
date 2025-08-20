# Phase 1: Foundation Implementation Guide

## Overview
Phase 1 establishes the core infrastructure for Genie, focusing on PyTorch integration and basic LazyTensor functionality. This phase proves feasibility and creates the foundation for semantic capture.

## Timeline: Weeks 1-8

## Success Criteria
- [x] PyTorch device registration working
- [x] LazyTensor with `__torch_function__` + factory interception (>95% capture)
- [x] Simple graph construction
- [x] <10µs operation overhead
- [x] Basic correctness tests against eager mode
- [ ] Basic pattern recognition (2 types)
- [ ] Local-remote (subprocess) executor prototype (Phase 2)

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
# setup.py (C++ extension optional in Phase 1)
from setuptools import setup, find_packages

setup(
    name='genie',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.1.0,<2.2.0',
        'numpy>=1.24.0',
        'networkx>=3.0',
    ]
)
```

## Week 3-4: PyTorch Device & Capture

### Task 2.1: Custom Device Implementation
- Register PrivateUse1 backend as `remote_accelerator`
- Provide `RemoteAcceleratorDevice.get_device(index)` helper

### Task 2.2: Interception
- Implement `LazyTensor.__torch_function__`
- Add factory interception for `randn/zeros/ones/empty/empty_strided`
- Normalize op names (`aten::op` without overload suffix)

## Week 5-6: LazyTensor & Graph

### Task 3.1: LazyTensor Core
- Metadata: op type, shape (best-effort), dtype, device hint
- Materialization triggers: `.cpu()`, `.item()`, print, `.numpy()`
- Executor fallback: eager PyTorch on CPU; coerce `device` kwargs to CPU

### Task 3.2: Graph Builder
- Add nodes/edges on LazyTensor creation
- Provide `get_graph()` and topological order

## Week 7-8: Tests and Demos

### Task 4.1: Correctness & Performance
- Add correctness examples comparing lazy vs eager (simple to complex chains)
- Add performance microbenchmarks for creation/materialization

### Task 4.2: Demos
- `example/perf_compare.py` (CPU)
- `example/resnet18_demo.py` (conv2d→relu chain; FX structure print)

## Deliverables Checklist

### Code Deliverables
- [x] Device registration (Python, C++ optional)
- [x] `__torch_function__` interception + factory hooks
- [x] LazyTensor class with metadata
- [x] Graph builder and representation
- [x] Simple executor (CPU materialization)
- [ ] 2 basic pattern recognizers
- [ ] Local-remote executor prototype (Phase 2)

### Test Deliverables  
- [x] Unit tests for device and interception
- [x] Integration tests for materialization triggers
- [x] Correctness suite (lazy vs eager)
- [x] Performance microbenchmarks

### Documentation Deliverables
- [x] Updated architecture and component docs
- [x] Device registration guide
- [ ] Pattern plugin tutorial (Phase 2)

### Demo Deliverables
- [x] Simple CNN (ResNet-18) running with LazyTensor
- [x] Graph visualization of structure via FX
- [x] Performance comparison vs eager mode

## Success Metrics
- Operation interception overhead: <10µs (pytest-benchmark)
- Memory overhead: <1% (tensor vs LazyTensor size)
- Test coverage: >85% (pytest-cov)
- Correctness parity: all supported ops match eager within 1e-5

## Known Constraints (Phase 1)
- PrivateUse1 allocations not supported during execution; device coerced to CPU in executor
- Dispatcher registrations optional; `__torch_function__` is primary
- Autograd/control flow support deferred to Phase 2+

## Next Phase Preview (Phase 2)
- Local-remote subprocess C++ runtime using LibTorch
- Control-plane over loopback TCP; shared/pinned memory transport
- Plan fragments, placement hints, and basic scheduling
- Begin DPDK allocator scaffolding for later RDMA integration
