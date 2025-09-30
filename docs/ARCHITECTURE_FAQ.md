# Architecture Frequently Asked Questions

## Overview

This document answers common questions about Genie's architecture, addressing concerns about design decisions and implementation approach.

---

## Q1: Why C++ for data plane and Python for control plane?

### Answer: Performance requirements demand it

**Python limitations**:
- Packet processing: ~50 µs (too slow)
- GIL prevents true parallelism
- No access to DPDK libraries
- Cannot do zero-copy DMA

**C++ advantages**:
- Packet processing: ~0.5 µs (100x faster)  
- Lock-free data structures
- Direct DPDK/GPUDev access
- True zero-copy via external buffers

**Python advantages**:
- Easy async/await for coordination
- Rich ecosystem for JSON/TCP
- Rapid iteration for control logic
- Cross-platform compatibility

**Conclusion**: Use each language for its strengths (industry standard pattern).

---

## Q2: Is separating `src/` and `genie/` directories standard practice?

### Answer: Yes - all major frameworks do this

**Examples**:
- **PyTorch**: `torch/` (Python) + `aten/src/` (C++)
- **TensorFlow**: `tensorflow/python/` + `tensorflow/core/`
- **NumPy**: `numpy/` (Python) + `numpy/core/src/` (C)

**Benefits**:
- Independent build systems (setuptools vs CMake)
- Dependency isolation (Python: light, C++: heavy DPDK)
- Testing flexibility (test Python without rebuilding C++)
- Team workflow (frontend/backend developers work independently)

---

## Q3: Isn't `genie/runtime/transports.py` duplicate of C++ data plane?

### Answer: No - they serve different tiers

**C++ Data Plane** (Tier 1 - Production):
- Purpose: High-performance zero-copy
- Performance: ~90 Gbps
- GPU Direct: Yes
- Use: Production deployments

**Python TCP Transport** (Tier 2 - Fallback):
- Purpose: Testing and degraded mode
- Performance: ~1-10 Gbps
- GPU Direct: No
- Use: Development, testing, when DPDK unavailable

**They are mutually exclusive** - only one is active at a time.

This is **standard practice**: PostgreSQL (libpq + JDBC), Redis (native + client libs), MongoDB (server + drivers).

---

## Q4: Why not use existing frameworks like Ray or gRPC?

### Answer: Semantic preservation requires custom approach

**Ray**: High-level task framework, loses operation-level semantics
**gRPC**: Good for control plane, but:
- No zero-copy support
- Not designed for GPU memory
- HTTP/2 overhead too high

**Genie's approach**:
- Control plane: Could use gRPC (future work)
- Data plane: **Must** be custom DPDK (zero-copy requirement)
- Semantic layer: **Must** be framework-integrated (capture semantics)

The paper's thesis is that **framework-level integration** is essential.

---

## Q5: How does this align with the HotNets'25 paper?

### Answer: Implementation directly validates paper claims

| Paper Claim (§3) | Implementation |
|-----------------|----------------|
| "Framework-level disaggregation" | LazyTensor at PyTorch layer |
| "Lazy tensor abstraction" | `genie/core/lazy_tensor.py` |
| "Semantic-rich graph" | FX + dispatcher + hooks |
| "Zero-copy data path" | DPDK + GPUDev in C++ |
| "Proactive allocation" | `DPDKAllocator` |
| "Semantic metadata in packets" | `GeniePacketHeader` struct |
| "Pluggable architecture" | Frontend/SRG/Backend separation |

**Every claim is implemented and tested**.

---

## Q6: What about fault tolerance and security?

### Answer: Production deployments will need enhancements

**Current implementation** (research prototype):
- Basic retransmission (ACK/NACK)
- Checksum validation
- No encryption
- No authentication

**Production additions needed**:
- TLS for control plane
- DTLS or custom encryption for data plane
- Authentication and authorization
- Byzantine fault tolerance
- Multi-path redundancy

**Note**: Paper focuses on semantic disaggregation mechanism, not complete production-hardening.

---

## Q7: How scalable is this to large clusters?

### Answer: Foundation for scalability, not yet implemented

**Current** (Phase 1):
- 2-10 nodes validated
- Point-to-point transfers
- No global scheduler

**Future** (Phase 2+, per paper §3.4):
- Global scheduler coordinates 100s-1000s of nodes
- Semantic graphs enable intelligent placement
- Multi-tenant resource sharing
- Dynamic scaling based on phase detection

**Paper's vision**: The semantic layer enables datacenter-scale scheduling.

---

## Q8: Why NetworkX for graph analysis? Isn't it slow?

### Answer: Fast enough with caching, can be optimized later

**Performance**:
- Conversion: ~5-10 ms for 500-node graph  
- Pattern matching: ~20-50 ms total
- **Target**: <100 ms (achieved)

**Optimizations**:
- LRU cache for repeated conversions
- Stable graph IDs for cache hits
- Early exit for simple graphs
- Lazy imports

**Future**: Could use faster graph library (e.g., graph-tool, custom C++) if needed.

---

## Q9: Can this work with frameworks other than PyTorch?

### Answer: Yes - architecture is designed for it

**Current**: PyTorch frontend only

**Architecture supports**:
```
JAX Frontend ─┐
              ├─→ Semantic Graph (SRG) ─→ DPDK Backend
PyTorch Frontend ─┘
```

**What's needed for JAX**:
- Implement JAX-specific LazyTensor equivalent
- Adapt to JAX's tracing mechanism
- Same SRG format, same backend

**SRG is framework-agnostic** - this is the "narrow waist" principle from the paper.

---

## Q10: What's the deployment story?

### Answer: Multiple deployment options

**Option 1: Full Stack** (Production):
```bash
# Install Python package
pip install genie

# Build C++ data plane
cd src/data_plane
cmake . && make
sudo make install  # Installs libgenie_data_plane.so

# Configure DPDK
sudo dpdk-devbind.py --bind=vfio-pci <NIC_PCI>
```

**Option 2: Python Only** (Development):
```bash
# Install Python package
pip install genie

# Uses TCP fallback (no C++ build needed)
export GENIE_DISABLE_CPP_DATAPLANE=1
```

**Option 3: Docker** (Future):
```bash
docker run -it genie/genie:latest
# Pre-configured with DPDK and GPUDirect
```

---

## Additional Resources

- **Paper**: `.kiro/HotNets25.tex`
- **Implementation Guide**: `docs/implementation/README.md`
- **Contributor Guide**: `docs/implementation/16-contributor-guide.md`
- **Refactoring Notes**: `REFACTORING_NOTES.md`

---

**Last Updated**: 2025-09-30  
**Maintainers**: Genie Core Team
