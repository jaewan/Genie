## Genie

Semantic-driven, framework-level disaggregation for AI accelerators. Phase 1 provides a custom device, LazyTensor engine, FX integration, basic patterns, and a runnable example.

### Requirements

- Python 3.10+ recommended (3.12 tested)
- PyTorch 2.8.0+ for RTX 50-series GPU support, or PyTorch 2.1.2+ for other GPUs
- Optional (GPU): CUDA 12.8+ (RTX 50-series) or CUDA 12.1+ (other GPUs), NVIDIA driver 535+, cuDNN
- **RTX 5060 Ti / RTX 5080**: Supported via PyTorch 2.8.0+cu128 (automatic detection)

### Quickstart (one command)

```bash
# Interactive setup (recommended)
./setup.sh

# CPU-only development
./setup.sh --mode basic
# RTX 50-series GPU development (PyTorch 2.8 + CUDA 12.8)
./setup.sh --mode basic --pytorch-cuda cu128
# Other GPU development (PyTorch 2.2–2.5 + CUDA 12.1)
./setup.sh --mode basic --pytorch-cuda cu121
# Full DPDK production setup
./setup.sh --mode dpdk
```

#### RTX 5060 Ti / RTX 5080 GPU Support

These next-generation GPUs use CUDA Compute Capability 12.0 (sm_120) and are now **fully supported** via PyTorch 2.8.0+cu128.

**Current Status**: ✅ **Full GPU acceleration available**

The system automatically detects and utilizes RTX 50-series GPUs for:
- Matrix operations and tensor computations
- Deep learning model training and inference
- Zero-copy memory transfers (when available)

**Performance**: RTX 5060 Ti provides excellent performance with 15.4 GB memory and fast matrix operations.

### DPDK Zero-Copy Setup (Advanced)

For high-performance zero-copy tensor transfers using DPDK and GPUDev:

#### Setup Instructions

1. **Automated Setup** (Recommended):
```bash
# Full DPDK setup with GPU-dev support
./setup.sh --mode dpdk

# Or with custom configuration
cp genie.conf my_config.conf
# Edit my_config.conf with your settings
./setup.sh --mode dpdk --config my_config.conf
```

2. **Verify Installation**:
```bash
# Run comprehensive test
python3 test_dpdk_complete.py

# Or use built-in diagnostics
./setup.sh --mode fix --diagnose
```

The setup script handles:
- System dependencies and kernel configuration
- DPDK 23.11 LTS installation with GPUDev support
- CUDA Toolkit 12.8 installation (or 12.1 when targeting PyTorch 2.2–2.5)
- IOMMU configuration and enablement
- Hugepage allocation and VFIO setup
- NIC auto-detection and binding to DPDK drivers
- GPUDirect RDMA configuration
- Comprehensive verification and testing

**Current Status:**
- ✅ DPDK 23.11 installed with MLX5 and GPUDev support
- ✅ CUDA Toolkit 12.8 with GPU Direct enabled (or 12.1)
- ✅ Mellanox ConnectX-5 bound to DPDK (vfio-pci)
- ✅ Hugepages allocated (2GB)
- ✅ IOMMU enabled and configured
- ✅ Full production-ready setup

See `README_SETUP.md` for detailed setup instructions and troubleshooting.

Expected example output (CPU-only):

```
Output shape: (1, 64, 112, 112)
Materialization time: XX.XX ms
Graph nodes: 2 | edges: 1
FX nodes: 7X
handoff.valid=True
```

Notes:
- requirements-dev.txt intentionally excludes torch. PyTorch is installed by the setup script with auto-detection of CUDA capabilities or manual specification using `--pytorch-cuda` (auto|cpu|cu121|cu118).
- The setup script builds C++ extensions in-place, so no editable install is required. If you prefer, you can still run `python setup.py build_ext --inplace` manually.

Note: Phase 1 executes locally (CPU materialization) even when CUDA is available. GPU/remote execution is targeted in later phases.

### Build C++ extension manually (optional)

The `genie/csrc/device.cpp` extension is built automatically by `pip install -e .`. To build in-place:

```bash
python setup.py build_ext --inplace
```

### Project layout

- `genie/core`: device, dispatcher, LazyTensor, FX utilities
- `genie/semantic`: Semantic Analyzer scaffold, pattern registry, handoff contracts
- `genie/patterns`: basic pattern plugins and FX patterns
- `example/resnet18_demo.py`: minimal demo using ResNet-18 conv1 weights
- `tests/`: unit/integration tests for Phase 1

### Troubleshooting

- Editable install fails complaining about `torch` not found:
  - Ensure you install `torch` first, then run `pip install -e .`.
- `ModuleNotFoundError: genie` when running the example:
  - Prefix with `PYTHONPATH=$(pwd)` or install with `pip install -e .`.
- Missing `torchvision` for the demo:
  - `pip install torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu` (or `+cu121`).
- DPDK not found warning:
  - Phase 1 is CPU-only; DPDK is optional. The warning can be ignored for now.

### License

See `LICENSE`.
