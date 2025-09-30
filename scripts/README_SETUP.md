# Genie GPUdirect RDMA Setup Guide

This guide covers setting up GPUdirect RDMA for high-performance GPU networking with DPDK gpu-dev support.

## Quick Start

### GPUdirect RDMA Setup
```bash
# Run the comprehensive setup script (requires sudo)
sudo /home/jaewan/Genie/scripts/setup_gpudirect_rdma_ultimate.sh
```

This script handles everything needed for GPUdirect RDMA:
- ✅ ConnectX-5 RNIC driver binding (vfio-pci → mlx5_core)
- ✅ nvidia_peermem kernel module loading
- ✅ RDMA device verification
- ✅ GPU information reporting

### Test GPU Dev Functionality
```bash
# Test the complete GPUdirect RDMA setup
sudo /home/jaewan/Genie/scripts/run_gpu_dev_test.sh
```

## What This Setup Provides

### Hardware Support
- ✅ **RTX 5060 Ti** (Blackwell architecture) - GPUdirect RDMA capable
- ✅ **ConnectX-5** RNIC - High-performance RDMA networking
- ✅ **Kernel 6.8.0-79** - Compatible with GPUdirect RDMA

### Software Stack
- ✅ **nvidia_peermem** - Kernel module for GPU-RDMA memory registration
- ✅ **mlx5_core/mlx5_ib** - ConnectX-5 RDMA drivers
- ✅ **CUDA 12.6** - GPU compute runtime
- ✅ **RDMA devices** - `rocep24s0` detected and functional

### DPDK gpu-dev Capabilities
- ✅ **Direct GPU-to-RNIC transfers** (zero-copy networking)
- ✅ **High-performance GPU I/O** (GPU-accelerated networking)
- ✅ **Low-latency GPU networking** (GPU-direct communication)
- ✅ **AI/ML inference optimization** (GPU-direct data paths)

## Setup Verification

### Check Status
```bash
# Verify nvidia_peermem is loaded
lsmod | grep nvidia_peermem

# Check RDMA devices
ibv_devices

# Verify GPU information
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

### Expected Results
```
nvidia_peermem         12288  0
    device                 node GUID
    ------              ----------------
    rocep24s0           08c0eb03007e9038
name, driver_version, memory.total [MiB]
NVIDIA GeForce RTX 5060 Ti, 575.64.03, 16311 MiB
```

## Usage Examples

### Basic GPUdirect RDMA Test
```bash
# Run the comprehensive test
/home/jaewan/Genie/test_gpudirect_rdma_comprehensive
```

### GPU Dev Test with Logging
```bash
# Run with detailed logging
sudo /home/jaewan/Genie/scripts/run_gpu_dev_test.sh
```

## Troubleshooting

### If nvidia_peermem Won't Load
```bash
# Check for corrupted DKMS modules
ls -la /lib/modules/$(uname -r)/updates/dkms/nvidia-peermem.ko

# Remove corrupted module if too small (< 10KB)
sudo rm /lib/modules/$(uname -r)/updates/dkms/nvidia-peermem.ko

# Try loading again
sudo modprobe nvidia_peermem
```

### If ConnectX-5 Not Detected
```bash
# Check driver binding
lspci -k -s 18:00.0

# Should show: Kernel driver in use: mlx5_core
# If showing vfio-pci, run the setup script again
```

### If RDMA Devices Not Found
```bash
# Verify ConnectX-5 is in InfiniBand mode
# Check firmware and driver configuration
ibv_devices  # Should show ConnectX-5 interfaces
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DPDK gpu-dev Application                 │
├─────────────────────────────────────────────────────────────┤
│  CUDA Runtime + RDMA APIs  │  nvidia_peermem Kernel Module  │
├─────────────────────────────────────────────────────────────┤
│  mlx5_core/mlx5_ib Drivers │  ConnectX-5 RNIC Hardware     │
├─────────────────────────────────────────────────────────────┤
│              RTX 5060 Ti GPU (GPUdirect RDMA)              │
└─────────────────────────────────────────────────────────────┘
```

## Performance Characteristics

- **Zero-copy transfers**: GPU memory ↔ RNIC (no CPU involvement)
- **Low latency**: Direct PCIe communication paths
- **High bandwidth**: Full PCIe 4.0/5.0 speeds
- **GPU acceleration**: Hardware-accelerated networking

## Next Steps

1. **Deploy applications** using DPDK gpu-dev for GPU-accelerated networking
2. **Benchmark performance** with your specific workloads
3. **Optimize configuration** based on your use case
4. **Scale deployment** across multiple nodes with GPUdirect RDMA

## Support

For issues or questions about GPUdirect RDMA setup:
1. Check the comprehensive setup script handles most common issues
2. Verify hardware compatibility (RTX 5060 Ti + ConnectX-5)
3. Ensure proper driver binding and module loading
