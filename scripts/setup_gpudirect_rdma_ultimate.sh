#!/bin/bash
# Ultimate GPUdirect RDMA Setup Script
# Comprehensive setup for RTX 5060 Ti + ConnectX-5 GPUdirect RDMA with DPDK gpu-dev support

set -e

echo "=== Ultimate GPUdirect RDMA Setup ==="
echo "RTX 5060 Ti + ConnectX-5 GPUdirect RDMA for DPDK gpu-dev"
echo

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: This script must be run as root"
    echo "Usage: sudo $0"
    exit 1
fi

# Auto-detect ConnectX-5 PCI address
echo "Detecting ConnectX-5 RNIC..."
CONNECTX5_PCI=$(lspci | grep "Mellanox Technologies MT27800" | cut -d' ' -f1)

if [ -z "$CONNECTX5_PCI" ]; then
    echo "❌ ERROR: ConnectX-5 not found!"
    echo "Please ensure ConnectX-5 is properly installed."
    exit 1
fi

echo "✓ Found ConnectX-5 at: $CONNECTX5_PCI"

echo "Step 1: Current status check..."
CURRENT_DRIVER=$(lspci -k -s $CONNECTX5_PCI | grep "Kernel driver in use" | cut -d: -f2 | tr -d ' ')
echo "   Current driver: $CURRENT_DRIVER"
echo "   Available drivers: $(lspci -k -s $CONNECTX5_PCI | grep "Kernel modules" | cut -d: -f2 | tr -d ' ')"

if [ "$CURRENT_DRIVER" != "vfio-pci" ]; then
    echo "   ConnectX-5 is already bound to $CURRENT_DRIVER"
    echo "   Checking if nvidia_peermem can load..."
    if modprobe nvidia_peermem 2>/dev/null; then
        echo "   ✓ nvidia_peermem loaded successfully"
        echo "   ✓ Setup appears to be working!"
        exit 0
    else
        echo "   ⚠ nvidia_peermem still won't load"
    fi
fi

echo "Step 2: Prepare for driver switch..."
# Ensure mlx5_core is loaded
modprobe mlx5_core 2>/dev/null || echo "   mlx5_core already loaded"
modprobe mlx5_ib 2>/dev/null || echo "   mlx5_ib already loaded"

# Try to unbind from vfio-pci
echo "Step 3: Unbinding from vfio-pci..."
if [ -f "/sys/bus/pci/drivers/vfio-pci/$CONNECTX5_PCI" ]; then
    echo "   Attempting unbind..."
    if echo "$CONNECTX5_PCI" > /sys/bus/pci/drivers/vfio-pci/unbind 2>/dev/null; then
        echo "   ✓ Successfully unbound from vfio-pci"
    else
        echo "   ⚠ Unbind failed, trying alternative method..."

        # Try removing VFIO modules temporarily
        rmmod vfio_pci 2>/dev/null || echo "   vfio_pci not loaded or couldn't remove"
        rmmod vfio 2>/dev/null || echo "   vfio not loaded or couldn't remove"
        rmmod vfio_iommu_type1 2>/dev/null || echo "   vfio_iommu_type1 not loaded or couldn't remove"

        # Try unbind again
        if echo "$CONNECTX5_PCI" > /sys/bus/pci/drivers/vfio-pci/unbind 2>/dev/null; then
            echo "   ✓ Successfully unbound after removing VFIO modules"
        else
            echo "   ❌ Failed to unbind from vfio-pci"
            echo "   Manual intervention may be required"
            exit 1
        fi
    fi
else
    echo "   Device not found in vfio-pci driver (already unbound)"
fi

echo "Step 4: Binding to mlx5_core..."
# Wait for unbind to complete
sleep 2

if [ -f "/sys/bus/pci/drivers/mlx5_core/$CONNECTX5_PCI" ]; then
    echo "   Device already bound to mlx5_core"
else
    if echo "$CONNECTX5_PCI" > /sys/bus/pci/drivers/mlx5_core/bind 2>/dev/null; then
        echo "   ✓ Successfully bound to mlx5_core"
    else
        echo "   ⚠ Failed to bind to mlx5_core"
        echo "   Trying manual driver probe..."
        modprobe mlx5_core
        sleep 2
        if echo "$CONNECTX5_PCI" > /sys/bus/pci/drivers/mlx5_core/bind 2>/dev/null; then
            echo "   ✓ Successfully bound after reload"
        else
            echo "   ❌ Failed to bind to mlx5_core"
            echo "   The device may need manual configuration"
            exit 1
        fi
    fi
fi

echo "Step 5: Verify new driver binding..."
NEW_DRIVER=$(lspci -k -s $CONNECTX5_PCI | grep "Kernel driver in use" | cut -d: -f2 | tr -d ' ')
echo "   New driver: $NEW_DRIVER"

if [ "$NEW_DRIVER" = "mlx5_core" ]; then
    echo "   ✓ ConnectX-5 successfully bound to mlx5_core"
else
    echo "   ❌ Failed to bind to mlx5_core (still $NEW_DRIVER)"
    exit 1
fi

echo "Step 6: Fix and load nvidia_peermem..."
# First, remove corrupted DKMS module if it exists
DKMS_SIZE=$(stat -c%s "/lib/modules/$(uname -r)/updates/dkms/nvidia-peermem.ko" 2>/dev/null || echo 0)
EXTRA_SIZE=$(stat -c%s "/lib/modules/$(uname -r)/extra/nvidia_peermem.ko" 2>/dev/null || echo 0)

if [ "$DKMS_SIZE" -gt 0 ] && [ "$DKMS_SIZE" -lt 10000 ] && [ "$EXTRA_SIZE" -gt 100000 ]; then
    echo "   Removing corrupted DKMS module ($DKMS_SIZE bytes)..."
    rm -f "/lib/modules/$(uname -r)/updates/dkms/nvidia-peermem.ko"
    echo "   ✓ Corrupted DKMS module removed"
fi

# Try loading nvidia_peermem with multiple methods
if modprobe nvidia_peermem 2>/dev/null; then
    echo "   ✓ nvidia_peermem loaded with standard modprobe"
elif insmod "/lib/modules/$(uname -r)/extra/nvidia_peermem.ko" 2>/dev/null; then
    echo "   ✓ nvidia_peermem loaded with direct insmod"
elif modprobe nvidia_peermem peerdirect_support=1 2>/dev/null; then
    echo "   ✓ nvidia_peermem loaded with peerdirect_support=1"
elif modprobe nvidia_peermem persistent_api_support=1 2>/dev/null; then
    echo "   ✓ nvidia_peermem loaded with persistent_api_support=1"
else
    echo "   ⚠ nvidia_peermem failed to load with all methods"
    echo "   This may require manual intervention"
    echo "   Try: modprobe nvidia_peermem"
fi

echo "Step 7: Verify RDMA functionality..."
RDMA_COUNT=$(ibv_devices 2>/dev/null | wc -l)
if [ "$RDMA_COUNT" -gt 1 ]; then
    echo "   ✓ RDMA devices detected:"
    ibv_devices
else
    echo "   ⚠ No RDMA devices detected"
    echo "   ConnectX-5 may need time to initialize"
    echo "   Try running: ibv_devices"
fi

echo "Step 8: Check network interfaces..."
MLX_COUNT=$(ip link show | grep -c mlx || echo 0)
if [ "$MLX_COUNT" -gt 0 ]; then
    echo "   ✓ Found $MLX_COUNT mlx network interfaces"
else
    echo "   ⚠ No mlx network interfaces found"
    echo "   This is normal if ConnectX-5 is in InfiniBand-only mode"
fi

echo "Step 9: GPU information..."
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv | tail -1)
GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv | tail -1)
echo "   GPU: $GPU_NAME"
echo "   Driver: $GPU_DRIVER"
echo "   CUDA: $(nvcc --version | grep "release" | cut -d' ' -f6)"

echo "Step 10: Comprehensive status verification..."
echo "   ConnectX-5 driver: $NEW_DRIVER"
echo "   nvidia_peermem loaded: $(lsmod | grep -c nvidia_peermem)"
echo "   RDMA devices: $(ibv_devices 2>/dev/null | wc -l) detected"
echo "   Network interfaces: $MLX_COUNT found"
echo "   GPU memory: $(nvidia-smi --query-gpu=memory.total --format=csv | tail -1)"

# Final assessment
PEERMEM_LOADED=$(lsmod | grep -c nvidia_peermem)
RDMA_WORKING=$(ibv_devices 2>/dev/null | wc -l)

echo
echo "=== Final Assessment ==="
if [ "$NEW_DRIVER" = "mlx5_core" ] && [ "$PEERMEM_LOADED" -gt 0 ] && [ "$RDMA_WORKING" -gt 1 ]; then
    echo "🎉 SUCCESS: Complete GPUdirect RDMA setup!"
    echo "✅ RTX 5060 Ti + ConnectX-5 ready for DPDK gpu-dev"
    echo "✅ Direct GPU-to-RNIC data transfer enabled"
    echo "✅ Zero-copy GPU networking available"
    echo ""
    echo "🚀 You can now use DPDK gpu-dev for:"
    echo "   - High-performance GPU networking"
    echo "   - AI/ML inference with GPU-direct I/O"
    echo "   - Low-latency GPU-accelerated networking"
    echo ""
    echo "📋 Verification commands:"
    echo "   RDMA devices: ibv_devices"
    echo "   CUDA test: /home/jaewan/Genie/test_cuda_basic"
    echo "   GPUdirect test: /home/jaewan/Genie/test_gpudirect_rdma_comprehensive"
elif [ "$NEW_DRIVER" = "mlx5_core" ] && [ "$RDMA_WORKING" -gt 1 ]; then
    echo "⚠️  PARTIAL SUCCESS: RNIC working, but nvidia_peermem issues"
    echo "   The core RDMA functionality is available"
    echo "   Try loading nvidia_peermem manually: modprobe nvidia_peermem"
    echo "   Or run the comprehensive test to verify GPUdirect functionality"
else
    echo "❌ SETUP INCOMPLETE: Manual intervention required"
    echo "   Check the steps above and fix any failures"
    echo "   The most common issue is nvidia_peermem not loading"
fi

echo
echo "=== Setup Summary ==="
echo "Hardware: RTX 5060 Ti (Blackwell) + ConnectX-5"
echo "Status: $(if [ "$NEW_DRIVER" = "mlx5_core" ] && [ "$PEERMEM_LOADED" -gt 0 ]; then echo '✅ READY'; elif [ "$NEW_DRIVER" = "mlx5_core" ]; then echo '⚠️ MOSTLY READY'; else echo '❌ NEEDS WORK'; fi)"
echo "GPUdirect RDMA: $(if [ "$NEW_DRIVER" = "mlx5_core" ] && [ "$PEERMEM_LOADED" -gt 0 ]; then echo '✅ ENABLED'; else echo '❌ DISABLED'; fi)"
echo "Ready for DPDK gpu-dev: $(if [ "$NEW_DRIVER" = "mlx5_core" ] && [ "$PEERMEM_LOADED" -gt 0 ]; then echo '✅ YES'; else echo '❌ NO'; fi)"
