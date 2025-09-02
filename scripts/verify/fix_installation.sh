#!/usr/bin/env bash
#
# Genie Installation Fix Script
# Diagnoses and fixes common installation issues
# Based on fix_dpdk_setup.sh but more comprehensive
#

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_fix() { echo -e "${BLUE}[FIX]${NC} $1"; }

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Diagnose system issues
diagnose_system() {
    log_info "🔍 Diagnosing system configuration..."
    
    local issues=0
    
    # Check IOMMU
    if ! dmesg | grep -q -E 'IOMMU.*enabled|DMAR.*enabled'; then
        log_warn "IOMMU not enabled"
        ((issues++))
    else
        log_info "✓ IOMMU is enabled"
    fi
    
    # Check hugepages
    local hugepages=$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages 2>/dev/null || echo 0)
    if [[ $hugepages -eq 0 ]]; then
        log_warn "No hugepages allocated"
        ((issues++))
    else
        log_info "✓ Hugepages allocated: $hugepages"
    fi
    
    # Check VFIO module
    if ! lsmod | grep -q vfio_pci; then
        log_warn "VFIO-PCI module not loaded"
        ((issues++))
    else
        log_info "✓ VFIO-PCI module loaded"
    fi
    
    # Check NVIDIA driver
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log_warn "NVIDIA driver not available"
        ((issues++))
    else
        log_info "✓ NVIDIA driver available"
    fi
    
    # Check CUDA
    if ! command -v nvcc >/dev/null 2>&1; then
        log_warn "CUDA toolkit not available"
        ((issues++))
    else
        log_info "✓ CUDA toolkit available"
    fi
    
    # Check DPDK
    if [[ ! -f /opt/dpdk/dpdk-23.11/install/lib/x86_64-linux-gnu/librte_eal.so ]]; then
        log_warn "DPDK not installed or incomplete"
        ((issues++))
    else
        log_info "✓ DPDK libraries found"
    fi
    
    return $issues
}

# Fix IOMMU issues
fix_iommu() {
    log_fix "Fixing IOMMU configuration..."
    
    local cpu_vendor=$(lscpu | grep "Vendor ID" | awk -F: '{print $2}' | xargs)
    local iommu_param
    
    if [[ "$cpu_vendor" == "GenuineIntel" ]]; then
        iommu_param="intel_iommu=on"
    elif [[ "$cpu_vendor" == "AuthenticAMD" ]]; then
        iommu_param="amd_iommu=on"
    else
        iommu_param="intel_iommu=on"
    fi
    
    if ! grep -q "$iommu_param" /etc/default/grub; then
        log_info "Adding IOMMU parameters to GRUB..."
        sed -i "s/GRUB_CMDLINE_LINUX=\"/GRUB_CMDLINE_LINUX=\"$iommu_param iommu=pt /" /etc/default/grub
        update-grub
        log_warn "GRUB updated - reboot required"
    fi
}

# Fix hugepages
fix_hugepages() {
    log_fix "Fixing hugepages configuration..."
    
    local hugepage_count=1024
    local hugepage_size=2048
    
    # Allocate hugepages
    echo $hugepage_count > /sys/kernel/mm/hugepages/hugepages-${hugepage_size}kB/nr_hugepages
    
    # Make persistent
    echo "vm.nr_hugepages = $hugepage_count" > /etc/sysctl.d/80-hugepages.conf
    sysctl -p /etc/sysctl.d/80-hugepages.conf
    
    # Mount hugepages
    mkdir -p /mnt/huge
    if ! mount | grep -q "/mnt/huge"; then
        mount -t hugetlbfs nodev /mnt/huge
        if ! grep -q "/mnt/huge" /etc/fstab; then
            echo "nodev /mnt/huge hugetlbfs defaults 0 0" >> /etc/fstab
        fi
    fi
    
    local allocated=$(cat /sys/kernel/mm/hugepages/hugepages-${hugepage_size}kB/nr_hugepages)
    log_info "Hugepages allocated: $allocated"
}

# Fix VFIO module
fix_vfio() {
    log_fix "Loading VFIO module..."
    
    modprobe vfio-pci 2>/dev/null || true
    echo "vfio-pci" > /etc/modules-load.d/vfio.conf
    
    if lsmod | grep -q vfio_pci; then
        log_info "✓ VFIO-PCI module loaded"
    else
        log_error "Failed to load VFIO-PCI module"
    fi
}

# Fix NVIDIA driver
fix_nvidia_driver() {
    log_fix "Installing NVIDIA driver..."
    
    apt-get update
    ubuntu-drivers autoinstall || {
        log_warn "Autoinstall failed, trying fallback..."
        apt-get install -y nvidia-driver-535 || true
    }
    
    log_warn "NVIDIA driver installed - reboot may be required"
}

# Fix CUDA toolkit
fix_cuda() {
    log_fix "Installing CUDA toolkit..."
    
    # Add NVIDIA repository if not present
    if [[ ! -f /etc/apt/sources.list.d/cuda.list ]]; then
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        dpkg -i cuda-keyring_1.1-1_all.deb
        apt-get update
    fi
    
    # Install CUDA toolkit
    apt-get install -y cuda-toolkit-12-6
    
    # Set up environment
    cat > /etc/profile.d/cuda.sh << 'EOF'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF
    
    log_info "CUDA toolkit installed"
}

# Fix DPDK installation
fix_dpdk() {
    log_fix "Fixing DPDK installation..."
    
    # Install missing dependencies
    apt-get update
    apt-get install -y \
        build-essential \
        python3-pyelftools \
        libnuma-dev \
        libpcap-dev \
        meson \
        ninja-build \
        libelf-dev \
        zlib1g-dev \
        libssl-dev \
        libjansson-dev \
        libibverbs-dev \
        librdmacm-dev \
        rdma-core
    
    # Check if DPDK needs to be rebuilt
    local dpdk_dir="/opt/dpdk/dpdk-23.11"
    if [[ -d "$dpdk_dir" ]] && [[ ! -f "$dpdk_dir/install/lib/x86_64-linux-gnu/librte_eal.so" ]]; then
        log_info "Rebuilding DPDK..."
        cd "$dpdk_dir"
        
        rm -rf build
        meson setup build \
            --prefix="$dpdk_dir/install" \
            -Dmax_numa_nodes=2 \
            -Dmax_lcores=128 \
            -Dmachine=native \
            -Dtests=false
        
        ninja -C build
        ninja -C build install
        
        # Update library cache
        echo "$dpdk_dir/install/lib/x86_64-linux-gnu" > /etc/ld.so.conf.d/dpdk.conf
        ldconfig
    fi
    
    # Set up DPDK environment
    cat > /etc/profile.d/dpdk.sh << EOF
export RTE_SDK=$dpdk_dir
export RTE_TARGET=build
export PATH=\$RTE_SDK/install/bin:\$PATH
export LD_LIBRARY_PATH=\$RTE_SDK/install/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=\$RTE_SDK/install/lib/x86_64-linux-gnu/pkgconfig:\$PKG_CONFIG_PATH
EOF
}

# Fix NIC binding
fix_nic_binding() {
    log_fix "Fixing NIC binding..."
    
    # Ensure dpdk-devbind is available
    local devbind_script="/opt/dpdk/dpdk-23.11/usertools/dpdk-devbind.py"
    if [[ -f "$devbind_script" ]]; then
        cp "$devbind_script" /usr/local/bin/
        chmod +x /usr/local/bin/dpdk-devbind.py
    fi
    
    # Auto-detect and bind Mellanox NIC
    local mellanox_nics=$(lspci -D | grep "Mellanox" | awk '{print $1}')
    for nic in $mellanox_nics; do
        local interface=$(ls /sys/bus/pci/devices/$nic/net/ 2>/dev/null | head -1)
        local default_interface=$(ip route | grep default | awk '{print $5}' | head -1)
        
        # Skip management interface
        if [[ -n "$interface" && "$interface" == "$default_interface" ]]; then
            log_info "Skipping management interface: $interface ($nic)"
            continue
        fi
        
        log_info "Binding Mellanox NIC $nic to DPDK..."
        dpdk-devbind.py --unbind "$nic" 2>/dev/null || true
        dpdk-devbind.py --bind=vfio-pci "$nic"
        
        if dpdk-devbind.py --status | grep -q "$nic.*drv=vfio-pci"; then
            log_info "✓ NIC $nic bound to DPDK"
            break
        fi
    done
}

# Fix GPU Direct
fix_gpu_direct() {
    log_fix "Configuring GPU Direct..."
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        # Enable GPU Direct
        if [[ ! -f /etc/modprobe.d/nvidia.conf ]] || ! grep -q "NVreg_EnableGpuDirect=1" /etc/modprobe.d/nvidia.conf; then
            echo "options nvidia NVreg_EnableGpuDirect=1" >> /etc/modprobe.d/nvidia.conf
        fi
        
        # Set persistence mode
        nvidia-smi -pm 1 2>/dev/null || true
        
        log_info "✓ GPU Direct configured"
    else
        log_warn "No NVIDIA GPU available for GPU Direct"
    fi
}

# Interactive fix menu
interactive_fix() {
    while true; do
        echo ""
        echo "🔧 Genie Installation Fix Menu"
        echo "============================="
        echo "1) Diagnose all issues"
        echo "2) Fix IOMMU configuration"
        echo "3) Fix hugepages"
        echo "4) Fix VFIO module"
        echo "5) Fix NVIDIA driver"
        echo "6) Fix CUDA toolkit"
        echo "7) Fix DPDK installation"
        echo "8) Fix NIC binding"
        echo "9) Fix GPU Direct"
        echo "10) Fix all issues automatically"
        echo "0) Exit"
        echo ""
        
        read -p "Select option [0-10]: " choice
        
        case $choice in
            1) diagnose_system ;;
            2) fix_iommu ;;
            3) fix_hugepages ;;
            4) fix_vfio ;;
            5) fix_nvidia_driver ;;
            6) fix_cuda ;;
            7) fix_dpdk ;;
            8) fix_nic_binding ;;
            9) fix_gpu_direct ;;
            10) fix_all ;;
            0) break ;;
            *) log_error "Invalid option" ;;
        esac
    done
}

# Fix all issues automatically
fix_all() {
    log_info "🔧 Fixing all detected issues..."
    
    local issues
    issues=$(diagnose_system)
    
    if [[ $issues -eq 0 ]]; then
        log_info "✅ No issues detected!"
        return 0
    fi
    
    log_info "Found $issues issues, fixing..."
    
    # Fix in order of dependency
    fix_iommu
    fix_hugepages
    fix_vfio
    fix_nvidia_driver
    fix_cuda
    fix_dpdk
    fix_nic_binding
    fix_gpu_direct
    
    log_info "✅ All fixes applied!"
    
    # Re-diagnose
    echo ""
    log_info "Re-checking system..."
    issues=$(diagnose_system)
    
    if [[ $issues -eq 0 ]]; then
        log_info "🎉 All issues resolved!"
    else
        log_warn "$issues issues remain - manual intervention may be required"
    fi
}

# Main function
main() {
    cd "$PROJECT_ROOT"
    
    log_info "🔧 Genie Installation Fix Tool"
    echo ""
    
    check_root
    
    # Check if we should run automatically or interactively
    if [[ $# -eq 0 ]]; then
        interactive_fix
    else
        case "$1" in
            --auto|--fix-all)
                fix_all
                ;;
            --diagnose)
                diagnose_system
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Usage: $0 [--auto|--diagnose]"
                exit 1
                ;;
        esac
    fi
    
    echo ""
    log_info "Fix session complete!"
}

main "$@"
