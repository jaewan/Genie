#!/usr/bin/env bash

set -euo pipefail

echo "=== Install Mellanox OFED for GPUDirect Support (sudo) ==="
echo "Started: $(date)"

# Require root
if [[ $EUID -ne 0 ]]; then
  echo "This script must be run with sudo/root" 1>&2
  exit 1
fi

KREL="$(uname -r)"
# Detect Ubuntu version
if [[ -f /etc/os-release ]]; then
    source /etc/os-release
    case "$VERSION_ID" in
        "22.04") DISTRO="ubuntu22.04" ;;
        "24.04") DISTRO="ubuntu24.04" ;;
        "20.04") DISTRO="ubuntu20.04" ;;
        *) 
            echo "WARN: Unsupported Ubuntu version $VERSION_ID, trying ubuntu22.04"
            DISTRO="ubuntu22.04"
            ;;
    esac
else
    DISTRO="ubuntu22.04"  # fallback
fi
ARCH="x86_64"

echo "Detected distribution: $DISTRO"

# MLNX OFED version - using LTS version
OFED_VER="24.04-0.6.6.0"
OFED_DIR="MLNX_OFED_LINUX-${OFED_VER}-${DISTRO}-${ARCH}"
OFED_TGZ="${OFED_DIR}.tgz"
OFED_URL="https://www.mellanox.com/downloads/ofed/MLNX_OFED-${OFED_VER}/${OFED_TGZ}"

echo "Checking if MLNX OFED is already installed..."
if command -v ofed_info >/dev/null 2>&1; then
    echo "MLNX OFED already installed: $(ofed_info -s)"
    # Check if peer-memory is available
    if modinfo nv_peer_mem >/dev/null 2>&1 || modinfo ib_peer_mem >/dev/null 2>&1; then
        echo "Peer-memory module already available."
        modprobe nv_peer_mem 2>/dev/null || modprobe ib_peer_mem 2>/dev/null || true
        if lsmod | grep -E "(nv_peer_mem|ib_peer_mem)" >/dev/null; then
            echo "Peer-memory module loaded successfully."
            exit 0
        fi
    fi
fi

echo "Installing prerequisites..."
apt-get update -y
apt-get install -y wget build-essential dkms linux-headers-"$KREL" \
    python3 python3-pip libnl-3-dev libnl-route-3-dev \
    flex bison libelf-dev bc

echo "Downloading MLNX OFED ${OFED_VER}..."
cd /tmp
if [[ ! -f "$OFED_TGZ" ]]; then
    wget "$OFED_URL" -O "$OFED_TGZ" || {
        echo "ERROR: Failed to download OFED. Trying alternative sources..."
        # Try NVIDIA networking (Mellanox acquired by NVIDIA)
        ALT_URL="https://content.mellanox.com/ofed/MLNX_OFED-${OFED_VER}/${OFED_TGZ}"
        wget "$ALT_URL" -O "$OFED_TGZ" || {
            echo "ERROR: Could not download MLNX OFED from any source."
            echo "Please download manually from: https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/"
            exit 1
        }
    }
fi

echo "Extracting MLNX OFED..."
tar -xzf "$OFED_TGZ"
cd "$OFED_DIR"

echo "Installing MLNX OFED (this may take several minutes)..."
# Set compiler to match kernel build environment
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
# Install with GPU Direct support, skip firmware update, non-interactive
./mlnxofedinstall --user-space-only --without-fw-update --force \
    --enable-gds --with-nvmf --with-nv-peer-mem --dpdk \
    --add-kernel-support --skip-repo || {
    echo "WARN: Full install failed, trying basic install..."
    export CC=/usr/bin/gcc-13
    export CXX=/usr/bin/g++-13
    ./mlnxofedinstall --basic --without-fw-update --force \
        --with-nv-peer-mem --add-kernel-support --skip-repo
}

echo "Starting OFED services..."
/etc/init.d/openibd restart || systemctl restart openibd || true

echo "Loading peer-memory module..."
modprobe nv_peer_mem 2>/dev/null || modprobe ib_peer_mem 2>/dev/null || {
    echo "WARN: Could not load peer-memory module directly."
    echo "Checking if module was built..."
    find /lib/modules/"$KREL" -name "*peer*" -type f 2>/dev/null || true
}

# Verify installation
if lsmod | grep -E "(nv_peer_mem|ib_peer_mem)" >/dev/null; then
    echo "SUCCESS: Peer-memory module loaded."
    lsmod | grep -E "(nv_peer_mem|ib_peer_mem)"
    exit 0
else
    echo "WARN: Peer-memory module not loaded, but OFED installed."
    echo "You may need to reboot or manually load the module."
    exit 1
fi

echo "=== MLNX OFED Installation Complete ==="
