# Djinn Frontend: Open Source Strategy

**Status**: Planning Phase
**Last Updated**: November 5, 2025

---

## Executive Summary

### Current Challenge
Djinn's frontend currently requires **PyTorch 2.8.0+** (December 2024), limiting adoption to ~20% of PyTorch users and blocking enterprise environments with LTS version constraints.

### Target Outcome
Support **PyTorch 1.5.0+** (April 2020) via progressive feature detection, expanding market coverage to ~80% of users while maintaining full functionality for modern PyTorch versions.

### Business Impact
- **4x increase** in potential user base
- **Enterprise compatibility** with Ubuntu/CentOS LTS PyTorch packages
- **Academic adoption** across diverse research environments

---

## PyTorch Version Compatibility Analysis

### Version-Dependent Features Matrix

| Feature | PyTorch Version | Impact | Fallback Strategy |
|---------|----------------|---------|-------------------|
| `__torch_dispatch__` | 2.0.0+ (stable) | ✅ Core interception | Manual operation wrapping |
| `__torch_function__` | 1.5.0+ (stable) | ✅ Edge case handling | Context-aware wrapping |
| `torch.device('meta')` | 1.9.0+ | ⚠️ Shape inference | Zero-tensor symbolic execution |
| Factory functions | Varies by function | ⚠️ Missing in older versions | Dynamic feature detection |
| ATen operation signatures | Varies by version | ⚠️ Compatibility issues | Fallback operation mapping |

### Compatibility Matrix

| PyTorch Version | Release Date | Support Level | Key Limitations | Target Support |
|----------------|--------------|---------------|----------------|----------------|
| 2.8.0+ | Dec 2024 | ✅ Full | None | ✅ Full |
| 2.0.0 - 2.7.x | Mar 2023 - Sep 2024 | ⚠️ Partial | No meta tensors | ✅ Graceful degradation |
| 1.9.0 - 1.13.x | Sep 2021 - Mar 2023 | ⚠️ Partial | Limited factory functions | ✅ Core functionality |
| 1.5.0 - 1.8.x | Apr 2020 - Mar 2022 | ❌ Minimal | Unstable mechanisms | ✅ Basic operations |

---

## Recommended Implementation Strategy

### Option A: Progressive Feature Detection (Recommended)

#### Architecture Overview

```python
class VersionAwareInterceptor:
    """Interception layer that adapts to PyTorch version capabilities."""

    def __init__(self):
        self.pytorch_version = self._get_version()
        self.capabilities = self._detect_capabilities()
        self.fallback_strategies = self._initialize_fallbacks()

    def _detect_capabilities(self):
        """Runtime feature detection for PyTorch version compatibility."""
        return {
            'torch_dispatch': self.pytorch_version >= (2, 0),
            'torch_function': self.pytorch_version >= (1, 5),
            'meta_device': self.pytorch_version >= (1, 9),
            'factory_functions': self._detect_factory_functions(),
            'aten_ops': self._detect_aten_operations(),
        }

    def _initialize_fallbacks(self):
        """Initialize fallback strategies based on detected capabilities."""
        fallbacks = {}

        if not self.capabilities['meta_device']:
            fallbacks['shape_inference'] = ZeroTensorShapeInference()

        if not self.capabilities['torch_dispatch']:
            fallbacks['operation_dispatch'] = ManualOperationWrapping()

        return fallbacks
```

#### Fallback Strategy Implementations

##### 1. Shape Inference (Meta Tensor Alternative)

```python
class ZeroTensorShapeInference:
    """Shape inference using zero tensors instead of meta tensors."""

    def infer_shape(self, operation, inputs, kwargs):
        """Execute operation with zero tensors to determine output shape."""
        try:
            # Create zero tensors matching input shapes
            zero_inputs = []
            for inp in inputs:
                if isinstance(inp, LazyTensor):
                    shape = inp.shape
                    dtype = inp.dtype
                    zero_inputs.append(torch.zeros(shape, dtype=dtype))
                else:
                    zero_inputs.append(inp)

            # Execute with zero tensors (minimal memory overhead)
            with torch.no_grad():
                result = operation(*zero_inputs, **kwargs)
                return result.shape

        except Exception as e:
            # Fallback to manual shape rules
            return self._manual_shape_rules(operation, inputs, kwargs)
```

##### 2. Operation Dispatch (Pre-2.0 Compatibility)

```python
class ManualOperationWrapping:
    """Manual operation wrapping for PyTorch versions < 2.0."""

    def __init__(self):
        self.operation_registry = self._build_operation_registry()

    def _build_operation_registry(self):
        """Build operation registry for manual dispatch."""
        return {
            'aten::add': lambda ins, kw: torch.add(ins[0], ins[1], **kw),
            'aten::mul': lambda ins, kw: torch.mul(ins[0], ins[1], **kw),
            'aten::matmul': lambda ins, kw: torch.matmul(ins[0], ins[1]),
            # ... comprehensive operation mapping
        }

    def dispatch_operation(self, operation_name, inputs, kwargs):
        """Manually dispatch operation based on name."""
        handler = self.operation_registry.get(operation_name)
        if handler:
            return handler(inputs, kwargs)
        else:
            raise UnsupportedOperationError(f"Operation {operation_name} not supported")
```

#### Benefits
- **80% user coverage** (PyTorch 1.5.0+)
- **Single codebase** with runtime adaptation
- **Future-proof** automatic support for new PyTorch versions
- **Graceful degradation** based on available features

### Option B: Version-Specific Packages (Alternative)

#### Package Structure
```
djinn-pytorch2/          # Full-featured (PyTorch 2.0+)
├── djinn/
│   ├── core/
│   ├── frontend/
│   └── ...
└── setup.py

djinn-pytorch1/          # Compatibility version (PyTorch 1.5+)
├── djinn/
│   ├── core/           # Shared logic
│   ├── frontend_legacy/
│   └── ...
└── setup.py

djinn-core/              # Shared components
├── djinn/
│   ├── shared/
│   └── utils/
└── setup.py
```

#### Build System Integration
```python
# setup.py for version-specific packages
def get_pytorch_requirement():
    """Determine appropriate PyTorch requirement based on package."""
    package_name = get_package_name()

    if 'pytorch2' in package_name:
        return 'torch>=2.0.0'
    elif 'pytorch1' in package_name:
        return 'torch>=1.5.0,<2.0.0'
    else:
        return 'torch>=1.5.0'  # Universal package
```

#### Pros & Cons

**Advantages:**
- **Optimal performance** per PyTorch version
- **Clear compatibility boundaries**
- **Easier maintenance** and testing
- **No runtime overhead** from feature detection

**Disadvantages:**
- **Package management complexity**
- **Code duplication** risk
- **User confusion** about which package to install
- **Maintenance burden** for multiple packages

---

## Implementation Roadmap

### Phase 1: Compatibility Foundation (Q1 2026)

#### Week 1-2: Core Infrastructure
- [ ] Implement `VersionAwareInterceptor` base class
- [ ] Add capability detection logic
- [ ] Create fallback strategy interfaces
- [ ] Set up multi-version testing infrastructure

#### Week 3-4: Shape Inference Fallback
- [ ] Implement `ZeroTensorShapeInference`
- [ ] Test shape inference accuracy across PyTorch versions
- [ ] Performance benchmarking vs meta tensors
- [ ] Integration with existing shape inference pipeline

#### Week 5-6: Operation Dispatch Fallback
- [ ] Build comprehensive operation registry for PyTorch 1.5+
- [ ] Implement manual operation wrapping
- [ ] Test operation compatibility across versions
- [ ] Performance and correctness validation

#### Week 7-8: Factory Function Detection
- [ ] Dynamic factory function detection
- [ ] Conditional wrapping based on availability
- [ ] Fallback strategies for missing functions
- [ ] Integration testing

### Phase 2: Ecosystem Integration (Q2 2026)

#### Continuous Integration Setup
```yaml
# .github/workflows/compatibility.yml
name: PyTorch Compatibility Testing
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        pytorch-version: ['1.5.1', '1.9.1', '2.0.1', '2.4.1', '2.8.0']
        python-version: ['3.8', '3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install PyTorch ${{ matrix.pytorch-version }}
      run: |
        pip install torch==${{ matrix.pytorch-version }} torchvision torchaudio

    - name: Install Djinn
      run: pip install -e .

    - name: Run compatibility tests
      run: python -m pytest tests/compatibility/ -v
```

#### Automated Compatibility Testing
- [ ] PyTorch version matrix CI pipeline
- [ ] Feature capability detection tests
- [ ] Fallback strategy validation
- [ ] Performance regression monitoring
- [ ] Memory usage tracking across versions

### Phase 3: Advanced Compatibility (Q3-Q4 2026)

#### Multi-Framework Evaluation
- [ ] JAX operation mapping assessment
- [ ] TensorFlow 2.x compatibility analysis
- [ ] Common abstraction layer design
- [ ] Proof-of-concept implementations

#### Enterprise LTS Support
- [ ] Ubuntu 20.04 LTS PyTorch compatibility
- [ ] CentOS/RHEL LTS package testing
- [ ] Corporate environment validation
- [ ] Documentation for enterprise deployment

#### Performance Optimization
- [ ] Version-specific performance tuning
- [ ] Memory optimization for older PyTorch versions
- [ ] Caching strategy optimization
- [ ] Benchmarking across PyTorch versions

---

## Risk Assessment & Mitigation

### Technical Risks

#### Runtime Feature Detection Overhead
**Risk**: Capability detection adds startup overhead
**Impact**: Increased cold start time
**Mitigation**:
- Cache detection results
- Lazy evaluation of expensive checks
- Compile-time feature detection where possible

#### Fallback Strategy Complexity
**Risk**: Multiple code paths increase maintenance burden
**Impact**: Bug introduction, inconsistent behavior
**Mitigation**:
- Comprehensive testing of all fallback paths
- Clear abstraction boundaries
- Automated testing for path consistency

#### Operation Signature Changes
**Risk**: ATen operation signatures evolve between versions
**Impact**: Silent failures or incorrect results
**Mitigation**:
- Version-specific operation registries
- Runtime signature validation
- Comprehensive operation testing

### Business Risks

#### Market Fragmentation
**Risk**: Supporting many versions dilutes focus on modern PyTorch
**Impact**: Slower innovation, resource competition
**Mitigation**:
- Clear deprecation policies for old versions
- Phased support reduction for very old versions
- Feature flags for experimental functionality

#### Support Complexity
**Risk**: Users report issues across many version combinations
**Impact**: Support team overload, inconsistent user experience
**Mitigation**:
- Automated issue triaging based on PyTorch version
- Clear compatibility documentation
- Community-driven support for older versions

---

## Success Metrics

### Technical Metrics
- **Compatibility Coverage**: >95% of PyTorch operations work across supported versions
- **Performance Parity**: <10% performance difference between PyTorch versions
- **Memory Overhead**: <5% additional memory usage for compatibility layer
- **Startup Time**: <50ms added overhead for feature detection

### Business Metrics
- **User Adoption**: Track PyTorch version distribution in user base
- **Enterprise Deployment**: Number of enterprise LTS deployments
- **Issue Resolution**: Time to resolve version-specific issues
- **Community Growth**: GitHub stars, contributors, and ecosystem projects

### Quality Metrics
- **Test Coverage**: >90% code coverage across all PyTorch versions
- **CI Reliability**: >99% CI pipeline success rate
- **Bug Reports**: <5% of issues related to version compatibility
- **Documentation**: User-reported clarity score >4.5/5

---

## Conclusion

Progressive feature detection provides the optimal balance between broad compatibility and implementation complexity. This strategy enables Djinn to reach 80% of PyTorch users while maintaining a single, maintainable codebase.

**Key Success Factors:**
- Runtime capability detection eliminates version-specific code paths
- Comprehensive fallback strategies ensure graceful degradation
- Automated testing prevents regressions across PyTorch versions
- Clear documentation manages user expectations

**Next Steps:**
1. Begin Phase 1 implementation with core infrastructure
2. Set up multi-version CI pipeline
3. Create comprehensive compatibility test suite
4. Update documentation and user communication
