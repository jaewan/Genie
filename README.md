## Genie

Semantic-driven, framework-level disaggregation for AI accelerators. Phase 1 provides a custom device, LazyTensor engine, FX integration, basic patterns, and a runnable example.

### Requirements

- Python 3.10.x recommended (3.8+ may work)
- PyTorch 2.1.2
- Optional (GPU): CUDA 12.1, NVIDIA driver 535+, cuDNN

### Quickstart (one command)

```bash
# CPU-only
bash setup_gpu_server.sh
# CUDA 12.1 wheels
bash setup_gpu_server.sh --pytorch-cuda cu121
# CUDA 11.8 wheels
bash setup_gpu_server.sh --pytorch-cuda cu118
```

### DPDK Zero-Copy Setup (Advanced)

For high-performance zero-copy tensor transfers using DPDK and GPUDev:

#### 🚨 TEMPORARY: No-IOMMU Development Mode

**Currently running in no-IOMMU mode for development purposes.**

⚠️ **IMPORTANT**: This setup provides DPDK functionality but without memory protection. It's suitable for development and testing but **NOT for production use**.

**To enable full IOMMU support:**
1. Ask your server administrator to follow instructions in `BIOS_IOMMU_SETUP.md`
2. After BIOS changes and reboot, run: `sudo ./enable_iommu_mode.sh`
3. Remove this section from README.md

#### Setup Instructions

1. **Automated Setup** (Recommended):
```bash
# Run the automated setup script
sudo ./setup_dpdk_server.sh

# Or with custom configuration
cp dpdk_config.template.env dpdk_config.env
# Edit dpdk_config.env with your settings
sudo ./setup_dpdk_server.sh --config dpdk_config.env
```

2. **Enable No-IOMMU Mode** (Current temporary setup):
```bash
# Already configured - NIC bound to DPDK in no-IOMMU mode
# Mellanox ConnectX-5 (0000:18:00.0) → vfio-pci driver
```

3. **Verify Installation**:
```bash
./verify_dpdk_setup.sh
```

The setup script handles:
- System dependencies and kernel configuration
- DPDK 23.11 LTS installation with GPUDev support
- Hugepage allocation and VFIO setup
- NIC binding to DPDK drivers (no-IOMMU mode)
- GPUDirect RDMA configuration
- Python bindings creation
- Systemd service for persistence

**Current Status:**
- ✅ DPDK 23.11 installed with MLX5 and GPUDev support
- ✅ Mellanox ConnectX-5 bound to DPDK (vfio-pci)
- ✅ Hugepages allocated (2GB)
- ⚠️ Running in no-IOMMU mode (temporary)
- ⚠️ IOMMU disabled (requires BIOS configuration)

See `documents/todos/01-dpdk-setup-tasks.md` for manual setup instructions.

Expected example output (CPU-only):

```
Output shape: (1, 64, 112, 112)
Materialization time: XX.XX ms
Graph nodes: 2 | edges: 1
FX nodes: 7X
handoff.valid=True
```

Notes:
- requirements-dev.txt intentionally excludes torch. Torch is installed by the setup script according to `--pytorch-cuda` (cpu|cu121|cu118) using 2.2.2 wheels.
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
