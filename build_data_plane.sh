#!/bin/bash
# Build script for Genie DPDK data plane

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Genie DPDK Data Plane${NC}"
echo "=================================="

# Try to source DPDK environment if present
if [ -f "/etc/profile.d/dpdk.sh" ]; then
    # shellcheck disable=SC1091
    . /etc/profile.d/dpdk.sh
fi

# Check for DPDK
if ! pkg-config --exists libdpdk; then
    echo -e "${RED}Error: DPDK not found. Please install DPDK first.${NC}"
    echo "Run: sudo bash scripts/setup/setup_dpdk.sh"
    exit 1
fi

# Get DPDK version
DPDK_VERSION=$(pkg-config --modversion libdpdk)
echo -e "${GREEN}Found DPDK version: ${DPDK_VERSION}${NC}"

# Check for CUDA (optional)
if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}CUDA found, GPU support enabled${NC}"
    CUDA_SUPPORT="-DCUDA_FOUND=ON"
else
    echo -e "${YELLOW}CUDA not found, using CPU staging fallback${NC}"
    CUDA_SUPPORT="-DCUDA_FOUND=OFF"
fi

# Create build directory
BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# Configure with CMake
echo -e "${GREEN}Configuring with CMake...${NC}"
cmake ../src/data_plane \
    -DCMAKE_BUILD_TYPE=Release \
    $CUDA_SUPPORT \
    -DGENIE_WITH_KCP=ON \
    -DBUILD_TESTS=ON

# Build
echo -e "${GREEN}Building...${NC}"
make -j$(nproc)

# Check if build succeeded
if [ -f "libgenie_data_plane.so" ]; then
    echo -e "${GREEN}Build successful!${NC}"
    
    # Install library to system paths
    echo -e "${GREEN}Installing library...${NC}"
    sudo make install
    sudo ldconfig
    
    # Create symlink for Python to find
    PYTHON_LIB_PATH="/usr/local/lib/libgenie_data_plane.so"
    if [ -f "$PYTHON_LIB_PATH" ]; then
        echo -e "${GREEN}Library installed to: $PYTHON_LIB_PATH${NC}"
    fi
    
    # Only attempt to link/run test executable if it exists
    if [ -f "test_data_plane" ]; then
        echo -e "${GREEN}Built test_data_plane successfully${NC}"
    fi
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Build complete!${NC}"
echo "=================================="
echo "Library installed to: /usr/local/lib/libgenie_data_plane.so"
echo ""
echo "To use in Python:"
echo "  1. Ensure LD_LIBRARY_PATH includes /usr/local/lib"
echo "  2. Run Python with appropriate permissions for DPDK"
echo ""
echo "Example:"
echo "  sudo LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH python3 your_script.py"