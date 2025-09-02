#!/usr/bin/env bash

set -euo pipefail

echo "=== Install and Load GPU Peer-Memory Module (sudo) ==="
echo "Started: $(date)"

# Require root
if [[ $EUID -ne 0 ]]; then
  echo "This script must be run with sudo/root" 1>&2
  exit 1
fi

KREL="$(uname -r)"

have_mod() {
  /sbin/modinfo "$1" >/dev/null 2>&1
}

load_mod() {
  modprobe "$1" 2>/dev/null || true
  lsmod | egrep -q "^$1\b"
}

echo "Checking for nvidia_uvm..."
load_mod nvidia_uvm || echo "WARN: nvidia_uvm not loaded (CUDA may be limited)"

echo "Checking for nvidia_peermem..."
if load_mod nvidia_peermem; then
  echo "nvidia_peermem loaded."
  exit 0
fi

echo "Checking for upstream peermem/ib_peer_mem..."
if load_mod peermem; then
  echo "peermem loaded."
  exit 0
fi
if load_mod ib_peer_mem; then
  echo "ib_peer_mem loaded."
  exit 0
fi

echo "Attempting to install upstream peermem from linux-modules-extra-$KREL..."
if apt-get update -y && apt-get install -y "linux-modules-extra-$KREL"; then
  if load_mod peermem || load_mod ib_peer_mem; then
    echo "peermem loaded via linux-modules-extra-$KREL."
    exit 0
  fi
fi

echo "Trying to install DKMS and prerequisites..."
apt-get install -y dkms build-essential linux-headers-"$KREL" git || true

echo "Attempting to install nvidia-peermem via DKMS package (if available)..."
if apt-get install -y nvidia-peermem-dkms 2>/dev/null; then
  if load_mod nvidia_peermem; then
    echo "nvidia_peermem loaded via DKMS package."
    exit 0
  fi
fi

echo "Attempting to build nvidia-peermem from source..."
TMPDIR="/tmp/nvidia-peermem.$$"
mkdir -p "$TMPDIR"
set +e
git clone --depth 1 https://github.com/NVIDIA/nvidia-peermem.git "$TMPDIR/repo"
GIT_RC=$?
set -e
if [[ $GIT_RC -eq 0 ]]; then
  pushd "$TMPDIR/repo" >/dev/null
  if [[ -f dkms.conf ]]; then
    MODVER="1.0"
    DEST="/usr/src/nvidia-peermem-$MODVER"
    rm -rf "$DEST"
    mkdir -p "$DEST"
    cp -a . "$DEST"/
    dkms add -m nvidia-peermem -v "$MODVER" || true
    dkms build -m nvidia-peermem -v "$MODVER"
    dkms install -m nvidia-peermem -v "$MODVER" || true
  else
    make || true
    make install || true
  fi
  popd >/dev/null
  rm -rf "$TMPDIR"
  if load_mod nvidia_peermem || load_mod peermem || load_mod ib_peer_mem; then
    echo "Peer-memory module loaded from source."
    exit 0
  fi
else
  echo "WARN: Could not clone NVIDIA nvidia-peermem repo. Trying Mellanox nv_peer_memory..."
  # Try Mellanox nv_peer_memory (legacy) as a fallback
  set +e
  git clone --depth 1 https://github.com/Mellanox/nv_peer_memory.git "$TMPDIR/repo_nv" 2>/dev/null
  GIT_RC2=$?
  set -e
  if [[ $GIT_RC2 -eq 0 ]]; then
    pushd "$TMPDIR/repo_nv" >/dev/null
    # Build with DKMS if spec present
    if [[ -f dkms.conf || -d sources ]]; then
      MODVER="1.0"
      DEST="/usr/src/nv_peer_memory-$MODVER"
      rm -rf "$DEST"
      mkdir -p "$DEST"
      cp -a . "$DEST"/
      dkms add -m nv_peer_memory -v "$MODVER" || true
      dkms build -m nv_peer_memory -v "$MODVER" || true
      dkms install -m nv_peer_memory -v "$MODVER" || true
    else
      make || true
      make install || true
    fi
    popd >/dev/null
    rm -rf "$TMPDIR"
    # Try loading either module name used by nv_peer_memory
    if load_mod nv_peer_mem || load_mod nv_peer_memory || load_mod nvidia_peermem || load_mod peermem || load_mod ib_peer_mem; then
      echo "Peer-memory module loaded via nv_peer_memory."
      exit 0
    fi
  fi
fi

echo "ERROR: Could not install/load peer-memory module (nvidia_peermem/peermem)."
echo "Please install MLNX OFED with NVIDIA peer-memory support or provide a DKMS package matching your driver."
exit 2


