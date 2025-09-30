# Contributor Guide

## Welcome to Genie!

This guide helps new contributors understand the Genie codebase and start contributing effectively.

**Target Audience**: Open-source contributors  
**Prerequisites**: Python 3.8+, basic C++ knowledge, understanding of PyTorch  
**Time to Read**: 30 minutes

---

## Quick Start

### 1. Understand the Architecture (15 minutes)

**Start with these docs**:
1. [Architecture Overview](01-architecture-overview.md) - System design
2. This guide - Contributor workflow

**Choose your area**:
- **Frontend/Semantic** (Python) → [LazyTensor](03-lazy-tensor.md), [Semantic Layer](06-semantic-layer.md)
- **Backend/Transport** (C++) → [Data Plane](09-data-plane-cpp.md), [Runtime Transport](05-runtime-transport.md)

### 2. Set Up Development Environment (30 minutes)

```bash
# Clone repository
git clone https://github.com/yourorg/genie.git
cd genie

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -e ".[dev]"

# (Optional) Build C++ data plane
cd src/data_plane
mkdir build && cd build
cmake ..
make -j$(nproc)
cd ../../..
```

### 3. Run Tests (10 minutes)

```bash
# Run Python tests (fast, no special setup)
pytest tests/test_lazy_tensor.py -v
pytest tests/test_device_registration.py -v

# Run semantic tests
pytest tests/test_analyzer_*.py -v
pytest tests/test_phase_detection.py -v

# All tests (excluding slow ones)
pytest tests/ -k "not slow and not stress" -v
```

### 4. Make Your First Contribution (1-2 hours)

**Good first issues**:
- Fix typos in documentation
- Add code examples to docstrings
- Improve error messages
- Add unit tests for existing functionality

---

## Project Structure

```
Genie/
├── genie/                    # Python package (user-facing)
│   ├── core/                # Core abstractions
│   │   ├── lazy_tensor.py  # Lazy execution
│   │   ├── dispatcher.py   # Operation interception
│   │   ├── device.py       # Device backend
│   │   └── executor.py     # Graph execution
│   ├── semantic/            # Semantic analysis
│   ├── patterns/            # Pattern recognition
│   ├── runtime/             # Transport layer
│   └── csrc/                # Small C++ extensions
│
├── src/                     # C++ performance-critical code
│   └── data_plane/         # DPDK zero-copy transport
│
├── tests/                   # Test suite
├── docs/                    # Documentation
└── examples/                # Usage examples
```

---

## Development Workflow

### Creating a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### Making Changes

**Python changes**:
```bash
# Edit code
vim genie/core/lazy_tensor.py

# Run tests
pytest tests/test_lazy_tensor.py -v

# Check code style
flake8 genie/core/lazy_tensor.py
```

**C++ changes**:
```bash
# Edit code
vim src/data_plane/genie_data_plane.cpp

# Rebuild
cd src/data_plane/build
make -j$(nproc)

# Run C++ tests
sudo ./test_data_plane  # May require root for DPDK
```

### Running Tests

```bash
# Fast tests only
pytest tests/ -k "not slow" -v

# With coverage
pytest tests/ --cov=genie --cov-report=html

# Specific component
pytest tests/test_semantic_*.py -v
```

### Code Style

**Python** (PEP 8):
```python
# Use 4 spaces for indentation
# Maximum line length: 100 characters
# Use type hints
def transfer_tensor(tensor: torch.Tensor, target: str) -> str:
    """Transfer tensor to remote accelerator."""
    pass
```

**C++** (Google Style):
```cpp
// Use 2 spaces for indentation
// Maximum line length: 80 characters
// CamelCase for classes, snake_case for functions
class GenieDataPlane {
  void send_tensor(...);
private:
  uint32_t max_retries_;
};
```

### Commit Messages

```
# Format
<type>(<scope>): <subject>

<body>

# Examples
feat(semantic): Add multi-modal pattern detection

Implement AdvancedMultiModalPattern that detects vision + language
models like VQA. Includes cross-attention fusion point detection.

Closes #123

fix(data-plane): Handle packet loss in fragmentation

Add retransmission logic for lost fragments. Improves reliability
over lossy networks.

docs(transport): Add zero-copy path explanation

Clarify how GPU Direct RDMA enables true zero-copy.
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

---

## Contributing Areas

### Frontend/Semantic Layer (Python)

**Good for**: Python developers, ML researchers

**Skills needed**: Python, PyTorch, graph algorithms

**Components**:
- LazyTensor (`genie/core/lazy_tensor.py`)
- Dispatcher (`genie/core/enhanced_dispatcher.py`)
- Semantic Analyzer (`genie/semantic/analyzer.py`)
- Pattern Matching (`genie/patterns/`)

**Example contributions**:
- Add new operation support
- Improve shape inference
- Create custom patterns
- Enhance phase detection

**Getting started**:
1. Read [LazyTensor docs](03-lazy-tensor.md)
2. Read [Semantic Layer docs](06-semantic-layer.md)
3. Check `tests/test_lazy_tensor.py` for examples
4. Make small improvements to existing code

### Backend/Data Plane (C++)

**Good for**: Systems programmers, networking experts

**Skills needed**: C++17, DPDK, networking, CUDA (optional)

**Components**:
- Main transport (`src/data_plane/genie_data_plane.cpp`)
- Zero-copy (`src/data_plane/genie_zero_copy_transport.cpp`)
- Threading (`src/data_plane/genie_dpdk_thread_model.cpp`)

**Example contributions**:
- Optimize packet processing
- Add new reliability protocols
- Improve GPU memory management
- Add hardware offloads

**Getting started**:
1. Read [Data Plane docs](09-data-plane-cpp.md)
2. Read [Runtime Transport docs](05-runtime-transport.md)
3. Build the C++ code
4. Run `test_data_plane` to understand flow

### Testing

**Good for**: QA engineers, anyone

**Skills needed**: pytest, basic Python

**Components**:
- Unit tests (`tests/test_*.py`)
- Integration tests (`tests/test_*_integration.py`)
- Performance tests (`tests/performance/`)

**Example contributions**:
- Add test coverage for untested code
- Create integration test scenarios
- Add performance benchmarks
- Improve test documentation

**Getting started**:
1. Run existing tests
2. Pick a component with low coverage
3. Write tests following existing patterns
4. Submit PR

---

## Code Review Process

### Submitting a Pull Request

1. **Create PR** with clear description
2. **Link issues** being addressed
3. **Add tests** for new functionality
4. **Update docs** if API changed
5. **Run all tests** and ensure they pass

### Review Checklist

**For reviewers**:
- [ ] Code follows style guide
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] No performance regressions
- [ ] API changes are backward-compatible
- [ ] Error handling is appropriate

---

## Getting Help

- **Documentation**: Check `docs/implementation/` first
- **Issues**: File bugs on GitHub
- **Discussions**: Join community chat
- **Tests**: Look at `tests/` for usage examples

---

## Related Documentation

- [Architecture Overview](01-architecture-overview.md)
- [Data Plane Implementation](09-data-plane-cpp.md)
- [Semantic Layer](06-semantic-layer.md)
- [Pattern Recognition](07-pattern-recognition.md)

---

**Last Updated**: 2025-09-30  
**Status**: Ready for open-source contributors
