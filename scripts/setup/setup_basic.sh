#!/usr/bin/env bash
#
# Genie Basic Setup - Python Environment + PyTorch
# Extracted and optimized from setup_gpu_server.sh
#

set -euo pipefail

# Configuration
PYTORCH_CUDA_FLAVOR="auto"
SKIP_DRIVER_INSTALL="false"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

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
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

log_info "🐍 Setting up Python environment and PyTorch..."

# Step 1: System packages
log_info "[1/6] Installing system packages..."
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    python-is-python3 \
    build-essential \
    software-properties-common \
    ca-certificates \
    curl \
    git \
    ubuntu-drivers-common

# Ensure python command works
if ! command -v python >/dev/null 2>&1; then
    log_warn "Setting up python command alias..."
    if command -v python3 >/dev/null 2>&1; then
        sudo update-alternatives --install /usr/bin/python python "$(command -v python3)" 1 || true
    fi
fi

# Step 2: NVIDIA Driver (if needed)
log_info "[2/6] Checking NVIDIA driver..."
if [[ "$SKIP_DRIVER_INSTALL" == "false" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log_info "Installing NVIDIA driver..."
        sudo ubuntu-drivers autoinstall || {
            log_warn "Autoinstall failed, trying fallback driver..."
            sudo apt-get install -y nvidia-driver-535 || true
        }
        log_warn "Driver installed. Reboot may be required."
    else
        log_info "NVIDIA driver already installed"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
    fi
else
    log_info "Skipping NVIDIA driver installation"
fi

# Step 3: Python virtual environment
log_info "[3/6] Creating Python virtual environment..."
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Step 4: Project dependencies
log_info "[4/6] Installing project dependencies..."
if [[ -f requirements-dev.txt ]]; then
    python -m pip install -r requirements-dev.txt
else
    log_warn "requirements-dev.txt not found, installing basic dependencies..."
    python -m pip install numpy networkx
fi

# Ensure runtime metrics and test plugins are present
python -m pip install --upgrade prometheus_client pytest pytest-asyncio anyio || true

# Step 5: PyTorch
log_info "[5/6] Installing PyTorch..."

# Auto-detect CUDA version if needed
if [[ "$PYTORCH_CUDA_FLAVOR" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        # Check CUDA capability
        gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "")
        if [[ "$gpu_info" =~ RTX.*[4-5][0-9] ]]; then
            PYTORCH_CUDA_FLAVOR="cu121"  # RTX 40xx/50xx series
            log_info "Detected modern GPU, using CUDA 12.1"
        else
            PYTORCH_CUDA_FLAVOR="cu118"  # Older GPUs
            log_info "Detected older GPU, using CUDA 11.8"
        fi
    else
        PYTORCH_CUDA_FLAVOR="cpu"
        log_info "No GPU detected, using CPU-only PyTorch"
    fi
fi

# Uninstall existing PyTorch
python -m pip uninstall -y torch torchvision torchaudio || true

# Install PyTorch variant
case "$PYTORCH_CUDA_FLAVOR" in
    cpu)
        python -m pip install --index-url https://download.pytorch.org/whl/cpu \
            torch==2.2.2+cpu torchvision==0.17.2+cpu torchaudio==2.2.2+cpu
        ;;
    cu128)
        # PyTorch 2.8 with CUDA 12.8 wheels
        python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
            torch==2.8.0+cu128
        # Install matching torchvision/torchaudio without hard pin to allow resolver to pick compatible wheels
        python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
            torchvision torchaudio
        ;;
    cu121)
        python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
            torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121
        ;;
    cu118)
        python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
            torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118
        ;;
    *)
        log_error "Unknown PyTorch CUDA variant: $PYTORCH_CUDA_FLAVOR"
        exit 1
        ;;
esac

# Step 6: Build C++ extensions
log_info "[6/6] Building Genie C++ extensions..."
if [[ -f setup.py ]]; then
    python setup.py build_ext -j "$(nproc)" --inplace
else
    log_warn "setup.py not found, skipping C++ extensions"
fi

# Verification
log_info "🔍 Verifying installation..."
echo "Python: $(python --version)"
echo "Location: $(which python)"
python -c 'import torch; print(f"PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")'

# Test Genie imports
python -c "
try:
    import genie
    print('✓ Genie package imported successfully')
    try:
        from genie import _C, _runtime
        print('✓ C++ extensions loaded')
    except ImportError as e:
        print(f'⚠ C++ extensions not available: {e}')
except ImportError as e:
    print(f'⚠ Genie package not found: {e}')
"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
fi

echo ""
log_info "✅ Basic setup complete!"
log_info "To activate the environment: source .venv/bin/activate"

if [[ "$SKIP_DRIVER_INSTALL" == "false" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
    log_warn "⚠️  Reboot may be required for NVIDIA driver to work"
fi
