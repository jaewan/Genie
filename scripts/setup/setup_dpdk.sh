#!/usr/bin/env bash
#
# Genie DPDK Setup - Full DPDK Server with GPU-dev Support
# Combines and optimizes setup_dpdk_server.sh and setup_environment.sh
#

set -euo pipefail

# Configuration
PYTORCH_CUDA_FLAVOR="auto"
SKIP_DRIVER_INSTALL="false"
CONFIG_FILE=""
# Repository root (two levels up from scripts/setup)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# DPDK Configuration
DPDK_VERSION="23.11"
DPDK_DIR="/opt/dpdk"
HUGEPAGE_SIZE="2048"
HUGEPAGE_COUNT="1024"
NIC_PCI_ADDR="auto"
GPU_DIRECT="1"

# Colors and logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pytorch-cuda)
            PYTORCH_CUDA_FLAVOR="$2"
            shift 2
            ;;
        --skip-driver)
            SKIP_DRIVER_INSTALL="true"
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Load config file if specified
if [[ -n "$CONFIG_FILE" && -f "$CONFIG_FILE" ]]; then
    log_info "Loading configuration from $CONFIG_FILE"
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
fi

# Check root privileges
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root (use sudo)"
    exit 1
fi

cd "$PROJECT_ROOT"

log_info "🚀 Setting up DPDK server with GPU-dev support..."

# Step 1: Basic setup first
log_step "[1/8] Running basic Python/PyTorch setup..."
if [[ -n "${SUDO_USER:-}" ]]; then
    sudo -u "$SUDO_USER" bash "$PROJECT_ROOT/scripts/setup/setup_basic.sh" \
        --pytorch-cuda "$PYTORCH_CUDA_FLAVOR" \
        $([ "$SKIP_DRIVER_INSTALL" == "true" ] && echo "--skip-driver")
else
    bash "$PROJECT_ROOT/scripts/setup/setup_basic.sh" \
        --pytorch-cuda "$PYTORCH_CUDA_FLAVOR" \
        $([ "$SKIP_DRIVER_INSTALL" == "true" ] && echo "--skip-driver")
fi

# Step 2: Install CUDA Toolkit (required for GPU-dev)
log_step "[2/8] Installing CUDA Toolkit..."
if ! command -v nvcc >/dev/null 2>&1; then
    log_info "Installing CUDA Toolkit (12.8 recommended for PyTorch 2.8; 12.1 for PyTorch 2.2–2.5)..."
    
    # Add NVIDIA repository
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update
    
    # Install CUDA toolkit
    apt-get install -y cuda-toolkit-12-6
    
    # Set up environment
    cat > /etc/profile.d/cuda.sh << 'EOF'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF
    
    log_info "CUDA Toolkit installed"
else
    log_info "CUDA Toolkit already installed"
fi

# Step 3: Install system dependencies
log_step "[3/8] Installing DPDK dependencies..."
apt-get update
apt-get install -y \
    build-essential \
    python3-pip \
    python3-pyelftools \
    python3-dev \
    libnuma-dev \
    libpcap-dev \
    pkg-config \
    meson \
    ninja-build \
    cmake \
    autoconf automake libtool \
    linux-headers-"$(uname -r)" \
    pciutils \
    iproute2 \
    net-tools \
    wget \
    git \
    libelf-dev \
    zlib1g-dev \
    libssl-dev \
    libjansson-dev \
    libibverbs-dev \
    librdmacm-dev \
    rdma-core

# Step 4: Setup IOMMU
log_step "[4/8] Configuring IOMMU..."
CPU_VENDOR=$(lscpu | grep "Vendor ID" | awk -F: '{print $2}' | xargs)

if [[ "$CPU_VENDOR" == "GenuineIntel" ]]; then
    IOMMU_PARAM="intel_iommu=on"
elif [[ "$CPU_VENDOR" == "AuthenticAMD" ]]; then
    IOMMU_PARAM="amd_iommu=on"
else
    IOMMU_PARAM="intel_iommu=on"
    log_warn "Unknown CPU vendor, defaulting to Intel IOMMU"
fi

if ! grep -q "$IOMMU_PARAM" /etc/default/grub; then
    log_info "Enabling IOMMU in GRUB..."
    sed -i "s/GRUB_CMDLINE_LINUX=\"/GRUB_CMDLINE_LINUX=\"$IOMMU_PARAM iommu=pt /" /etc/default/grub
    update-grub
    log_warn "IOMMU enabled - reboot required for changes to take effect"
else
    log_info "IOMMU already enabled"
fi

# Load VFIO driver
modprobe vfio-pci 2>/dev/null || true
echo "vfio-pci" > /etc/modules-load.d/vfio.conf

# Step 5: Setup hugepages
log_step "[5/8] Configuring hugepages..."
log_info "Allocating ${HUGEPAGE_COUNT} hugepages of ${HUGEPAGE_SIZE}KB"

echo "$HUGEPAGE_COUNT" > /sys/kernel/mm/hugepages/hugepages-${HUGEPAGE_SIZE}kB/nr_hugepages

# Make persistent
echo "vm.nr_hugepages = $HUGEPAGE_COUNT" > /etc/sysctl.d/80-hugepages.conf
sysctl -p /etc/sysctl.d/80-hugepages.conf

# Mount hugepages
mkdir -p /mnt/huge
if ! mount | grep -q "/mnt/huge"; then
    mount -t hugetlbfs nodev /mnt/huge
    echo "nodev /mnt/huge hugetlbfs defaults 0 0" >> /etc/fstab
fi

ALLOCATED=$(cat /sys/kernel/mm/hugepages/hugepages-${HUGEPAGE_SIZE}kB/nr_hugepages)
log_info "Hugepages allocated: $ALLOCATED"

# Step 6: Install DPDK
log_step "[6/8] Installing DPDK ${DPDK_VERSION} with GPU-dev support..."
INSTALL_PREFIX="$DPDK_DIR/dpdk-${DPDK_VERSION}/install"

if [[ -f "$INSTALL_PREFIX/lib/x86_64-linux-gnu/librte_eal.so" ]] && [[ -f "$INSTALL_PREFIX/lib/x86_64-linux-gnu/librte_gpudev.so" ]]; then
    log_info "DPDK with GPU-dev already installed"
else
    mkdir -p "$DPDK_DIR"
    cd "$DPDK_DIR"
    
    # Download DPDK
    if [[ ! -f "dpdk-${DPDK_VERSION}.tar.xz" ]]; then
        wget "https://fast.dpdk.org/rel/dpdk-${DPDK_VERSION}.tar.xz"
    fi
    
    if [[ ! -d "dpdk-${DPDK_VERSION}" ]]; then
        tar xf "dpdk-${DPDK_VERSION}.tar.xz"
    fi
    
    cd "dpdk-${DPDK_VERSION}"
    
    # Clean previous build
    rm -rf build
    
    # Configure build
    meson setup build \
        --prefix="$INSTALL_PREFIX" \
        -Dmax_numa_nodes=2 \
        -Dmax_lcores=128 \
        -Dmachine=native \
        -Dtests=false \
        -Dexamples=l2fwd,l3fwd
    
    # Build and install
    ninja -C build
    ninja -C build install
    
    # Set up library paths
    echo "$INSTALL_PREFIX/lib/x86_64-linux-gnu" > /etc/ld.so.conf.d/dpdk.conf
    ldconfig
    
    log_info "DPDK installation completed"
fi
# Optional: install KCP (ikcp)
log_step "[6b/8] Checking KCP (ikcp) support..."
if ! pkg-config --exists ikcp && [[ ! -f /usr/local/include/ikcp.h ]]; then
    log_info "Installing KCP (ikcp) from source..."
    TMPDIR=$(mktemp -d)
    pushd "$TMPDIR" >/dev/null
    git clone --depth 1 https://github.com/skywind3000/kcp.git
    cd kcp
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j"$(nproc)"
    make install
    ldconfig
    popd >/dev/null
    rm -rf "$TMPDIR"
else
    log_info "KCP (ikcp) already available"
fi


# Set up environment
cat > /etc/profile.d/dpdk.sh << EOF
export RTE_SDK=$DPDK_DIR/dpdk-${DPDK_VERSION}
export RTE_TARGET=build
export PATH=\$RTE_SDK/install/bin:\$PATH
export LD_LIBRARY_PATH=\$RTE_SDK/install/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=\$RTE_SDK/install/lib/x86_64-linux-gnu/pkgconfig:\$PKG_CONFIG_PATH
EOF

# Step 7: Setup DPDK drivers and bind NIC
log_step "[7/8] Setting up DPDK drivers and NIC binding..."

# Copy dpdk-devbind script
cp "$DPDK_DIR/dpdk-${DPDK_VERSION}/usertools/dpdk-devbind.py" /usr/local/bin/
chmod +x /usr/local/bin/dpdk-devbind.py

# Auto-detect NIC if needed
if [[ "$NIC_PCI_ADDR" == "auto" ]]; then
    log_info "Auto-detecting suitable NIC..."
    
    # Look for Mellanox or Intel NICs
    CANDIDATE_NICS=$(lspci -D | grep -E "Mellanox|Intel.*Ethernet" | awk '{print $1}')
    
    for nic in $CANDIDATE_NICS; do
        INTERFACE=$(ls /sys/bus/pci/devices/$nic/net/ 2>/dev/null | head -1)
        
        if [[ -n "$INTERFACE" ]]; then
            DEFAULT_INTERFACE=$(ip route | grep default | awk '{print $5}' | head -1)
            
            if [[ "$INTERFACE" != "$DEFAULT_INTERFACE" ]]; then
                NIC_PCI_ADDR=$nic
                log_info "Selected NIC: $NIC_PCI_ADDR (interface: $INTERFACE)"
                break
            fi
        else
            # For Mellanox cards without active interface
            if lspci -s "$nic" | grep -q "Mellanox"; then
                NIC_PCI_ADDR=$nic
                log_info "Selected Mellanox NIC: $NIC_PCI_ADDR"
                break
            fi
        fi
    done
    
    if [[ "$NIC_PCI_ADDR" == "auto" ]]; then
        log_warn "Could not auto-detect suitable NIC"
        NIC_PCI_ADDR=""
    fi
fi

# Bind NIC to DPDK if found
if [[ -n "$NIC_PCI_ADDR" ]]; then
    log_info "Binding NIC $NIC_PCI_ADDR to DPDK..."
    
    # Unbind from current driver
    dpdk-devbind.py --unbind "$NIC_PCI_ADDR" 2>/dev/null || true
    
    # Bind to vfio-pci
    dpdk-devbind.py --bind=vfio-pci "$NIC_PCI_ADDR"
    
    if dpdk-devbind.py --status | grep -q "$NIC_PCI_ADDR.*drv=vfio-pci"; then
        log_info "NIC successfully bound to DPDK"
    else
        log_error "Failed to bind NIC to DPDK"
    fi
else
    log_warn "No NIC specified or detected for DPDK binding"
fi

# Step 8: Setup GPU Direct
log_step "[8/8] Configuring GPU Direct..."
if [[ "$GPU_DIRECT" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    # Enable GPU Direct in NVIDIA driver
    if [[ ! -f /etc/modprobe.d/nvidia.conf ]] || ! grep -q "NVreg_EnableGpuDirect=1" /etc/modprobe.d/nvidia.conf; then
        echo "options nvidia NVreg_EnableGpuDirect=1" >> /etc/modprobe.d/nvidia.conf
    fi
    
    # Set GPU to persistence mode
    nvidia-smi -pm 1 2>/dev/null || true
    
    log_info "GPU Direct configured"
else
    log_info "GPU Direct disabled or no GPU available"
fi

# Create configuration file
mkdir -p /etc/genie
cat > /etc/genie/dpdk.conf << EOF
# Genie DPDK Configuration
DPDK_VERSION=$DPDK_VERSION
DPDK_DIR=$DPDK_DIR
HUGEPAGE_SIZE=$HUGEPAGE_SIZE
HUGEPAGE_COUNT=$HUGEPAGE_COUNT
NIC_PCI_ADDR=$NIC_PCI_ADDR
GPU_DIRECT=$GPU_DIRECT
EOF

# Final verification
log_info "🔍 Verifying DPDK installation..."
cd "$PROJECT_ROOT"

# Run comprehensive test
if [[ -f "test_dpdk_complete.py" ]]; then
    sudo -u "$SUDO_USER" bash -c "source /etc/profile.d/dpdk.sh && source /etc/profile.d/cuda.sh && python3 test_dpdk_complete.py"
else
    log_warn "Comprehensive test script not found"
fi

echo ""
log_info "✅ DPDK setup complete!"
echo ""
echo "📋 Summary:"
echo "  DPDK Version: $DPDK_VERSION"
echo "  Hugepages: $HUGEPAGE_COUNT x ${HUGEPAGE_SIZE}KB"
echo "  NIC: ${NIC_PCI_ADDR:-Not configured}"
echo "  GPU Direct: $GPU_DIRECT"
echo ""
echo "🎯 Next steps:"
echo "  1. Source environment: source /etc/profile.d/dpdk.sh"
echo "  2. Activate Python env: source .venv/bin/activate"
echo "  3. Test your application"
echo ""

if grep -q "GRUB_CMDLINE_LINUX.*iommu" /etc/default/grub && ! dmesg | grep -q "IOMMU.*enabled"; then
    log_warn "⚠️  Reboot required for IOMMU changes to take effect"
fi
