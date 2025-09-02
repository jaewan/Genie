#!/usr/bin/env bash
set -euo pipefail

# Ubuntu 24.x GPU setup script for this project
# - Makes `python` point to python3
# - Creates `.venv` and installs from requirements-dev.txt
# - Installs NVIDIA driver if `nvidia-smi` is absent
# - Installs PyTorch (cpu|cu121|cu118) and builds C++ extensions in-place
#
# Usage examples:
#   bash setup_gpu_server.sh                      # CPU-only PyTorch
#   bash setup_gpu_server.sh --pytorch-cuda cu121 # Install PyTorch CUDA 12.1 wheels
#   bash setup_gpu_server.sh --pytorch-cuda cu118 # Install PyTorch CUDA 11.8 wheels
#   bash setup_gpu_server.sh --skip-driver        # Skip NVIDIA driver installation
#
# Notes:
# - This script uses sudo for apt operations. You may be prompted for your password.
# - After installing NVIDIA drivers, a reboot is typically required.
# - If Secure Boot is enabled, you may need to disable it or enroll MOK for driver modules.

PYTORCH_CUDA_FLAVOR="cpu"   # one of: cpu|cu121|cu118
SKIP_DRIVER_INSTALL="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pytorch-cuda)
      shift
      PYTORCH_CUDA_FLAVOR="${1:-cpu}"
      ;;
    --skip-driver)
      SKIP_DRIVER_INSTALL="true"
      ;;
    -h|--help)
      grep '^# ' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
  shift || true
done

# Ensure we are in the repository root (directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Detect OS
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  if [[ "${ID:-}" != "ubuntu" ]]; then
    echo "This script is intended for Ubuntu. Detected: ${ID:-unknown}" >&2
    exit 1
  fi
else
  echo "/etc/os-release not found; cannot verify OS." >&2
fi

echo "[1/6] Updating apt and installing base packages..."
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
  lsb-release \
  ubuntu-drivers-common

# Make sure `python` points to python3 (python-is-python3 provides the symlink)
if ! command -v python >/dev/null 2>&1; then
  echo "python command not found after installation; attempting to set alternatives..."
  if command -v python3 >/dev/null 2>&1; then
    sudo update-alternatives --install /usr/bin/python python "$(command -v python3)" 1 || true
  fi
fi

echo "[2/6] Checking NVIDIA driver and nvidia-smi..."
NEED_DRIVER_INSTALL="false"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  NEED_DRIVER_INSTALL="true"
fi

if [[ "$SKIP_DRIVER_INSTALL" == "true" ]]; then
  echo "Skipping NVIDIA driver installation as requested (--skip-driver)."
else
  if [[ "$NEED_DRIVER_INSTALL" == "true" ]]; then
    echo "nvidia-smi not found. Installing recommended NVIDIA driver via ubuntu-drivers..."
    # ubuntu-drivers will select the recommended proprietary driver
    sudo ubuntu-drivers autoinstall || {
      echo "ubuntu-drivers autoinstall failed; attempting fallback installation of a recent driver..."
      # Fallback to commonly available driver series (adjust if needed)
      sudo apt-get install -y nvidia-driver-535 || true
    }
    echo "Driver installation attempted. A reboot may be required for nvidia-smi to become available."
  else
    echo "nvidia-smi found; skipping driver installation."
  fi
fi

# Show GPU info if available now
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Current NVIDIA devices:"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
else
  echo "nvidia-smi still not available. You may need to reboot and re-run the verification steps."
fi

echo "[3/6] Creating Python virtual environment at .venv ..."
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "[4/6] Installing project dependencies from requirements-dev.txt (excluding torch) ..."
if [[ -f requirements-dev.txt ]]; then
  # requirements-dev.txt intentionally does not pin torch; install other deps first
  python -m pip install -r requirements-dev.txt
else
  echo "requirements-dev.txt not found in $(pwd)." >&2
  exit 1
fi

echo "[5/6] Installing PyTorch variant: $PYTORCH_CUDA_FLAVOR ..."
python -m pip uninstall -y torch torchvision torchaudio || true
case "$PYTORCH_CUDA_FLAVOR" in
  cpu)
    python -m pip install --index-url https://download.pytorch.org/whl/cpu \
      torch==2.2.2+cpu torchvision==0.17.2+cpu torchaudio==2.2.2+cpu
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
    echo "Unknown --pytorch-cuda value: $PYTORCH_CUDA_FLAVOR (use: cpu|cu121|cu118)" >&2
    exit 1
    ;;
esac

# Build C++ extensions in-place so imports work without editable install
echo "[6/6] Building Genie C++ extensions in-place ..."
python setup.py build_ext -j "$(nproc)" --inplace

echo "Verifying setup..."
python --version
which python
python -c 'import sys; print("python executable:", sys.executable)'
python -c 'import torch; print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())'
python - << 'PY'
import importlib, sys
print("Importing genie...")
m = importlib.import_module("genie")
print("genie imported from:", m.__file__)
try:
    from genie import _C, _runtime
    print("C++ extensions loaded:", hasattr(_C, "device_count"), hasattr(_runtime, "__doc__"))
except Exception as e:
    print("Warning: failed to load C++ extensions:", e, file=sys.stderr)
PY
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not available yet (driver may need a reboot)."
fi

echo "\nSetup complete."
if [[ "$NEED_DRIVER_INSTALL" == "true" && "$SKIP_DRIVER_INSTALL" != "true" ]]; then
  echo "A system reboot is recommended if drivers were installed."
fi
