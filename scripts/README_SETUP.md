# Genie Setup Guide

Genie provides a unified setup system for AI accelerator disaggregation with DPDK and GPU support.

## Quick Start

### Basic Setup (Development)
```bash
./setup.sh --mode basic
```
This installs Python environment, PyTorch, and basic dependencies.

### Full DPDK Setup (Production)
```bash
./setup.sh --mode dpdk
```
This installs everything including DPDK, GPU-dev support, and high-performance networking.

### Fix Issues
```bash
./setup.sh --mode fix
```
Diagnoses and fixes common installation problems.

## Setup Modes

### 1. Basic Mode
- ✅ Python virtual environment
- ✅ PyTorch (auto-detects CUDA version)
- ✅ Genie C++ extensions
- ✅ Basic dependencies
- 🎯 **Use for**: Development, testing, CPU-only workloads

### 2. DPDK Mode  
- ✅ Everything from Basic mode
- ✅ DPDK 23.11 with GPU-dev support
- ✅ CUDA Toolkit 12.8 (PyTorch 2.8), or 12.1 (PyTorch 2.2–2.5)
- ✅ IOMMU configuration
- ✅ Hugepages setup
- ✅ NIC binding to DPDK
- ✅ GPU Direct configuration
- 🎯 **Use for**: Production, high-performance networking

### 3. Fix Mode
- 🔧 Diagnoses system issues
- 🔧 Fixes DPDK problems
- 🔧 Repairs driver issues
- 🔧 Interactive or automatic fixing

## Advanced Usage

### Custom Configuration
```bash
# Copy and edit configuration
cp genie.conf my_config.conf
# Edit my_config.conf with your settings
./setup.sh --mode dpdk --config my_config.conf
```

### Specific Options
```bash
# Force specific CUDA version for PyTorch wheels
#   cu128 -> PyTorch 2.8 (CUDA 12.8)
#   cu121 -> PyTorch 2.2–2.5 (CUDA 12.1)
./setup.sh --mode basic --pytorch-cuda cu128

# Skip driver installation
./setup.sh --mode dpdk --skip-driver

# Non-interactive mode
./setup.sh --mode dpdk --non-interactive
```

## System Requirements

### Basic Mode
- Ubuntu 20.04+ (22.04+ recommended)
- Python 3.8+
- 4GB RAM minimum
- Build tools (installed automatically)

### DPDK Mode
- Ubuntu 22.04+ (24.04 recommended)
- Kernel 5.15+ 
- 8GB RAM minimum
- IOMMU-capable CPU (Intel VT-d / AMD-Vi)
- Compatible NIC (Mellanox ConnectX-5+, Intel E810+)
- NVIDIA GPU (optional, for GPU Direct)

## Hardware Compatibility

### Tested NICs
- ✅ Mellanox ConnectX-5, ConnectX-6, ConnectX-7
- ✅ Intel E810, X710, XXV710
- ⚠️ Other NICs may work but are untested

### Tested GPUs
- ✅ NVIDIA RTX 40xx/50xx series (CUDA 12.1 / 12.8)
- ✅ NVIDIA RTX 30xx series (CUDA 11.8)
- ✅ NVIDIA Tesla/Quadro series
- ⚠️ AMD GPUs not supported for GPU Direct

## Troubleshooting

### Common Issues

**"IOMMU not enabled"**
```bash
# Check BIOS settings for VT-d/AMD-Vi
# Run fix mode to update GRUB
./setup.sh --mode fix
# Reboot required
```

**"No hugepages allocated"**
```bash
# Fix hugepages configuration
sudo ./scripts/fix_installation.sh --auto
```

**"NIC not bound to DPDK"**
```bash
# Check available NICs
lspci | grep -E "Ethernet|Network"
# Manually specify NIC in config
echo 'NIC_PCI_ADDR="0000:18:00.0"' >> genie.conf
```

**"CUDA not found"**
```bash
# Install CUDA toolkit
./setup.sh --mode fix
# Or run DPDK mode which includes CUDA
./setup.sh --mode dpdk
```

### Getting Help

1. **Run diagnostics**:
   ```bash
   ./setup.sh --mode fix --diagnose
   ```

2. **Check logs**:
   ```bash
   # DPDK setup logs
   sudo tail -f /var/log/genie_dpdk_setup.log
   ```

3. **Verify installation**:
   ```bash
   # Test DPDK + GPU
   python3 test_dpdk_complete.py
   ```

## Migration from Old Scripts

If you were using the old setup scripts:

- `setup_gpu_server.sh` → `./setup.sh --mode basic`
- `setup_dpdk_server.sh` → `./setup.sh --mode dpdk` 
- `setup_environment.sh` → `./setup.sh --mode dpdk`
- `fix_dpdk_setup.sh` → `./setup.sh --mode fix`

The old scripts are still available but deprecated.

## Environment Activation

After setup, activate the environment:

```bash
# Activate Python environment
source .venv/bin/activate

# Load DPDK environment (DPDK mode only)
source /etc/profile.d/dpdk.sh

# Load CUDA environment (DPDK mode only)  
source /etc/profile.d/cuda.sh
```

## Performance Tuning

For optimal performance with DPDK mode:

1. **CPU isolation**: Add `isolcpus=2-7` to kernel parameters
2. **NUMA awareness**: Use `numactl` to bind processes to NUMA nodes
3. **Interrupt affinity**: Configure NIC interrupts to specific cores
4. **Hugepage sizing**: Consider 1GB hugepages for large workloads

See `genie.conf` for detailed configuration options.
