#!/bin/bash

# Genie DPDK Setup Verification Script
# Checks that DPDK environment is properly configured
# Usage: ./verify_dpdk_setup.sh

# Don't exit on errors - we want to check everything
# set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Counters
PASSED=0
FAILED=0
WARNINGS=0

echo "========================================="
echo "   Genie DPDK Setup Verification"
echo "========================================="
echo ""

# Function to check a condition
check() {
    local description="$1"
    local command="$2"
    local required="${3:-yes}"
    
    echo -n "Checking $description... "
    
    # Use a subshell to prevent set -e from affecting the main script
    if (eval "$command") &>/dev/null; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((PASSED++))
        return 0
    else
        if [[ "$required" == "yes" ]]; then
            echo -e "${RED}✗ FAILED${NC}"
            ((FAILED++))
        else
            echo -e "${YELLOW}⚠ WARNING${NC}"
            ((WARNINGS++))
        fi
        return 0  # Don't exit, continue checking
    fi
}

# System checks
echo "=== System Configuration ==="

check "Kernel version >= 5.15" \
    "[[ $(uname -r | cut -d. -f1) -ge 5 ]] || [[ $(uname -r | cut -d. -f1) -eq 5 && $(uname -r | cut -d. -f2) -ge 15 ]]"

check "IOMMU enabled" \
    "dmesg | grep -q -E 'IOMMU.*enabled|DMAR.*enabled'"

check "Hugepages allocated" \
    "[[ $(cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages) -gt 0 ]]"

check "Hugepages mounted" \
    "mount | grep -q '/mnt/huge'"

check "VFIO driver loaded" \
    "lsmod | grep -q vfio_pci"

echo ""
echo "=== DPDK Installation ==="

check "DPDK libraries installed" \
    "[[ -f /opt/dpdk/dpdk-23.11/install/lib/x86_64-linux-gnu/librte_eal.so ]] || [[ -f /usr/local/lib/librte_eal.so ]]"

check "dpdk-devbind tool available" \
    "which dpdk-devbind.py"

check "DPDK environment variables set" \
    "[[ -n \$RTE_SDK ]] || [[ -f /etc/profile.d/dpdk.sh ]]" "no"

echo ""
echo "=== NIC Configuration ==="

# Check if any NIC is bound to DPDK
if dpdk-devbind.py --status 2>/dev/null | grep -q "drv=vfio-pci\|drv=igb_uio"; then
    echo -e "NIC bound to DPDK... ${GREEN}✓ FOUND${NC}"
    ((PASSED++))
    dpdk-devbind.py --status | grep "drv=vfio-pci\|drv=igb_uio"
else
    echo -e "NIC bound to DPDK... ${YELLOW}⚠ NOT FOUND${NC} (may be intentional for testing)"
    ((WARNINGS++))
fi

echo ""
echo "=== GPU Configuration ==="

check "NVIDIA driver loaded" \
    "lsmod | grep -q nvidia" "no"

check "CUDA available" \
    "which nvcc" "no"

if command -v nvidia-smi &>/dev/null; then
    echo -n "GPUDirect support... "
    if nvidia-smi -q | grep -q "GPUDirect" || [[ -f /etc/modprobe.d/nvidia.conf ]] && grep -q "NVreg_EnableGpuDirect=1" /etc/modprobe.d/nvidia.conf; then
        echo -e "${GREEN}✓ ENABLED${NC}"
        ((PASSED++))
    else
        echo -e "${YELLOW}⚠ NOT ENABLED${NC}"
        ((WARNINGS++))
    fi
    
    echo -n "GPU persistence mode... "
    if nvidia-smi -q | grep -q "Persistence Mode.*Enabled"; then
        echo -e "${GREEN}✓ ENABLED${NC}"
        ((PASSED++))
    else
        echo -e "${YELLOW}⚠ NOT ENABLED${NC}"
        ((WARNINGS++))
    fi
fi

check "GPUDev library available" \
    "[[ -f /opt/dpdk/dpdk-23.11/install/lib/x86_64-linux-gnu/librte_gpudev.so ]] || [[ -f /usr/local/lib/librte_gpudev.so ]]" "no"

echo ""
echo "=== Python Integration ==="

check "Python 3 available" \
    "which python3"

check "Python DPDK bindings exist" \
    "[[ -f /opt/genie/dpdk_bindings/test_dpdk_gpu.py ]]"

# Test Python bindings
echo -n "Testing Python DPDK bindings... "
if python3 /opt/genie/dpdk_bindings/test_dpdk_gpu.py &>/dev/null; then
    echo -e "${GREEN}✓ WORKING${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ NOT WORKING${NC} (may need environment setup)"
    ((WARNINGS++))
fi

echo ""
echo "=== Services and Configuration ==="

check "Genie DPDK systemd service exists" \
    "[[ -f /etc/systemd/system/genie-dpdk.service ]]" "no"

check "Genie configuration file exists" \
    "[[ -f /etc/genie/dpdk.conf ]]" "no"

echo ""
echo "=== Performance Settings ==="

# Check CPU frequency governor
echo -n "CPU frequency governor... "
GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
if [[ "$GOV" == "performance" ]]; then
    echo -e "${GREEN}✓ performance${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ $GOV (recommend 'performance')${NC}"
    ((WARNINGS++))
fi

# Check for isolated CPUs
echo -n "Isolated CPUs for DPDK... "
if grep -q "isolcpus" /proc/cmdline; then
    ISOLATED=$(grep -o "isolcpus=[^ ]*" /proc/cmdline)
    echo -e "${GREEN}✓ $ISOLATED${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ None (optional but recommended)${NC}"
    ((WARNINGS++))
fi

echo ""
echo "========================================="
echo "            Verification Summary"
echo "========================================="
echo ""
echo -e "  Passed:   ${GREEN}$PASSED${NC}"
echo -e "  Failed:   ${RED}$FAILED${NC}"
echo -e "  Warnings: ${YELLOW}$WARNINGS${NC}"
echo ""

if [[ $FAILED -eq 0 ]]; then
    if [[ $WARNINGS -eq 0 ]]; then
        echo -e "${GREEN}✅ All checks passed! System is ready for Genie DPDK.${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠️  System is functional but has $WARNINGS warnings.${NC}"
        echo "   Review warnings above for optimal performance."
        exit 0
    fi
else
    echo -e "${RED}❌ System has $FAILED critical failures.${NC}"
    echo "   Please fix the issues above before proceeding."
    exit 1
fi
