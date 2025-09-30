#!/usr/bin/env bash

set -euo pipefail

LOG_DIR="/home/jaewan/Genie"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/gpu_dev_test_${TS}.log"

echo "=== Genie GPUDev Zero-Copy Test (sudo) ===" | tee "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "User: $(whoami)" | tee -a "$LOG_FILE"
echo "Kernel: $(uname -r)" | tee -a "$LOG_FILE"

echo "\n[1/6] Loading kernel modules (nvidia_uvm, nvidia_peermem/peermem, vfio-pci)" | tee -a "$LOG_FILE"

# Load nvidia_uvm first
modprobe nvidia_uvm || true

# Handle nvidia_peermem loading with fallback methods
echo "   Loading nvidia_peermem..." | tee -a "$LOG_FILE"
if modprobe nvidia_peermem 2>/dev/null; then
    echo "   ✓ nvidia_peermem loaded with modprobe" | tee -a "$LOG_FILE"
elif insmod "/lib/modules/$(uname -r)/extra/nvidia_peermem.ko" 2>/dev/null; then
    echo "   ✓ nvidia_peermem loaded with insmod (bypassed corrupted DKMS)" | tee -a "$LOG_FILE"
elif modprobe peermem 2>/dev/null; then
    echo "   ✓ peermem loaded as fallback" | tee -a "$LOG_FILE"
else
    echo "   ⚠ nvidia_peermem/peermem not available" | tee -a "$LOG_FILE"
fi

modprobe vfio-pci || true

echo "Loaded modules:" | tee -a "$LOG_FILE"
LOADED_PEERMEM=$(lsmod | grep -c "^nvidia_peermem ")
lsmod | egrep "^(nvidia_peermem|peermem|nvidia_uvm|vfio_pci)" || echo "(some modules may not be loaded)" | tee -a "$LOG_FILE"

if [ "$LOADED_PEERMEM" -eq 0 ]; then
  echo "WARNING: peer memory module not loaded (nvidia_peermem/peermem). GPU Direct will be disabled." | tee -a "$LOG_FILE"
else
  echo "✓ nvidia_peermem loaded successfully ($LOADED_PEERMEM modules)" | tee -a "$LOG_FILE"
fi

echo "\n[2/6] Hugepages setup" | tee -a "$LOG_FILE"
HP_CUR=$(cat /proc/sys/vm/nr_hugepages || echo 0)
echo "Current hugepages: $HP_CUR" | tee -a "$LOG_FILE"
if [ "$HP_CUR" -lt 1024 ]; then
  echo 1024 > /proc/sys/vm/nr_hugepages || true
fi
mkdir -p /dev/hugepages || true
mount | grep -q "/dev/hugepages" || mount -t hugetlbfs nodev /dev/hugepages || true
echo "Hugepages after setup: $(cat /proc/sys/vm/nr_hugepages)" | tee -a "$LOG_FILE"

echo "\n[3/6] IOMMU status" | tee -a "$LOG_FILE"
dmesg | grep -i -E "iommu|DMAR" | tail -n 20 | tee -a "$LOG_FILE" || true

echo "\n[4/6] DPDK environment" | tee -a "$LOG_FILE"
export PKG_CONFIG_PATH="/opt/dpdk/dpdk-23.11/install/lib/x86_64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="/opt/dpdk/dpdk-23.11/install/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
echo "PKG_CONFIG_PATH=$PKG_CONFIG_PATH" | tee -a "$LOG_FILE"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" | tee -a "$LOG_FILE"

echo "\n[5/6] EAL arguments (IOVA=PA)" | tee -a "$LOG_FILE"
export GENIE_EAL_ARGS="-l 0-3 -n 4 -m 2048 --proc-type=auto --iova=pa --file-prefix=genie --huge-dir=/dev/hugepages"
echo "GENIE_EAL_ARGS=\"$GENIE_EAL_ARGS\"" | tee -a "$LOG_FILE"

echo "\n[6/6] Running GPUdirect RDMA test (timeout 120s)" | tee -a "$LOG_FILE"
cd /home/jaewan/Genie
export PYTHONPATH="/home/jaewan/Genie:${PYTHONPATH:-}"

# Use our comprehensive GPUdirect RDMA test instead of missing Python test
if [ -f "/home/jaewan/Genie/test_gpudirect_rdma_comprehensive" ]; then
    CMD=(timeout 120s /home/jaewan/Genie/test_gpudirect_rdma_comprehensive)
    echo "> ${CMD[*]}" | tee -a "$LOG_FILE"
    set +e
    "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
    RC=${PIPESTATUS[0]}
    set -e
else
    echo "Comprehensive test not found, running basic CUDA test..." | tee -a "$LOG_FILE"
    if [ -f "/home/jaewan/Genie/test_cuda_basic" ]; then
        CMD=(timeout 60s /home/jaewan/Genie/test_cuda_basic)
        echo "> ${CMD[*]}" | tee -a "$LOG_FILE"
        set +e
        "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
        RC=${PIPESTATUS[0]}
        set -e
    else
        echo "No test binaries found!" | tee -a "$LOG_FILE"
        RC=1
    fi
fi

echo "\nExit code: $RC" | tee -a "$LOG_FILE"
echo "Logs saved to: $LOG_FILE" | tee -a "$LOG_FILE"
exit $RC


