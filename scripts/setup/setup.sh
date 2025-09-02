#!/usr/bin/env bash
#
# Genie Unified Setup Script
# Comprehensive setup for AI accelerator disaggregation with DPDK + GPU support
#
# Usage:
#   ./setup.sh                                    # Interactive setup
#   ./setup.sh --mode basic                       # Basic Python/PyTorch only
#   ./setup.sh --mode dpdk                        # Full DPDK + GPU setup
#   ./setup.sh --mode fix                         # Fix existing installation
#   ./setup.sh --pytorch-cuda cu121               # Specify CUDA version
#   ./setup.sh --skip-driver --skip-reboot        # Skip driver install/reboot
#
# Modes:
#   basic  - Python environment + PyTorch + basic dependencies
#   dpdk   - Full DPDK server setup with GPU-dev support
#   fix    - Fix common DPDK/GPU issues
#

set -euo pipefail

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Default configuration
SETUP_MODE=""
PYTORCH_CUDA_FLAVOR="auto"  # auto|cpu|cu128|cu121|cu118
SKIP_DRIVER_INSTALL="false"
SKIP_REBOOT_CHECK="false"
INTERACTIVE="true"
CONFIG_FILE=""

# DPDK Configuration
DPDK_VERSION="23.11"
HUGEPAGE_COUNT="1024"
NIC_PCI_ADDR="auto"
GPU_DIRECT="auto"

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

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                SETUP_MODE="$2"
                INTERACTIVE="false"
                shift 2
                ;;
            --pytorch-cuda)
                PYTORCH_CUDA_FLAVOR="$2"
                shift 2
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --skip-driver)
                SKIP_DRIVER_INSTALL="true"
                shift
                ;;
            --skip-reboot)
                SKIP_REBOOT_CHECK="true"
                shift
                ;;
            --non-interactive)
                INTERACTIVE="false"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    cat << EOF
Genie Unified Setup Script

USAGE:
    ./setup.sh [OPTIONS]

MODES:
    basic    Basic Python/PyTorch environment setup
    dpdk     Full DPDK server setup with GPU-dev support  
    fix      Fix common installation issues

OPTIONS:
    --mode MODE              Setup mode (basic|dpdk|fix)
    --pytorch-cuda VARIANT   PyTorch CUDA variant (auto|cpu|cu128|cu121|cu118)
    --config FILE           Load configuration from file
    --skip-driver           Skip NVIDIA driver installation
    --skip-reboot           Skip reboot requirement checks
    --non-interactive       Run without prompts
    -h, --help              Show this help

EXAMPLES:
    ./setup.sh                                # Interactive mode
    ./setup.sh --mode basic                   # Basic setup only
    ./setup.sh --mode dpdk --pytorch-cuda cu121  # Full DPDK with CUDA 12.1
    ./setup.sh --mode fix                     # Fix existing issues

EOF
}

# Interactive mode selection
interactive_setup() {
    if [[ "$INTERACTIVE" != "true" ]]; then
        return
    fi
    
    echo "🚀 Welcome to Genie Setup!"
    echo "=========================="
    echo ""
    echo "This script will set up your system for AI accelerator disaggregation."
    echo ""
    echo "Setup modes:"
    echo "  1) Basic    - Python environment + PyTorch (recommended for development)"
    echo "  2) DPDK     - Full DPDK server with GPU-dev support (for production)"
    echo "  3) Fix      - Fix existing installation issues"
    echo ""
    
    while true; do
        read -p "Select setup mode [1-3]: " choice
        case $choice in
            1) SETUP_MODE="basic"; break ;;
            2) SETUP_MODE="dpdk"; break ;;
            3) SETUP_MODE="fix"; break ;;
            *) echo "Please select 1, 2, or 3." ;;
        esac
    done
    
    # Auto-detect CUDA if not specified
    if [[ "$PYTORCH_CUDA_FLAVOR" == "auto" ]]; then
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo ""
            echo "NVIDIA GPU detected. Select PyTorch variant:"
            echo "  1) CUDA 12.8 (PyTorch 2.8 wheels)"
            echo "  2) CUDA 12.1 (stable with PyTorch 2.2–2.5)"
            echo "  3) CUDA 11.8 (for older GPUs)"
            echo "  4) CPU only"
            echo ""
            
            while true; do
                read -p "Select PyTorch variant [1-4]: " cuda_choice
                case $cuda_choice in
                    1) PYTORCH_CUDA_FLAVOR="cu128"; break ;;
                    2) PYTORCH_CUDA_FLAVOR="cu121"; break ;;
                    3) PYTORCH_CUDA_FLAVOR="cu118"; break ;;
                    4) PYTORCH_CUDA_FLAVOR="cpu"; break ;;
                    *) echo "Please select 1, 2, 3, or 4." ;;
                esac
            done
        else
            PYTORCH_CUDA_FLAVOR="cpu"
            log_info "No NVIDIA GPU detected, using CPU-only PyTorch"
        fi
    fi
}

# Load configuration file if specified
load_config() {
    if [[ -n "$CONFIG_FILE" && -f "$CONFIG_FILE" ]]; then
        log_info "Loading configuration from $CONFIG_FILE"
        # shellcheck disable=SC1090
        source "$CONFIG_FILE"
    fi
}

# Check system requirements
check_system() {
    log_step "Checking system requirements..."
    
    # Check OS
    if [[ ! -f /etc/os-release ]]; then
        log_error "Cannot detect OS version"
        exit 1
    fi
    
    # shellcheck disable=SC1091
    source /etc/os-release
    if [[ "$ID" != "ubuntu" ]]; then
        log_error "This script is designed for Ubuntu. Detected: $ID"
        exit 1
    fi
    
    log_info "Detected: $PRETTY_NAME"
    
    # Check kernel version for DPDK
    if [[ "$SETUP_MODE" == "dpdk" ]]; then
        kernel_version=$(uname -r | cut -d. -f1,2)
        kernel_major=$(echo "$kernel_version" | cut -d. -f1)
        kernel_minor=$(echo "$kernel_version" | cut -d. -f2)
        
        if [[ $kernel_major -lt 5 ]] || [[ $kernel_major -eq 5 && $kernel_minor -lt 15 ]]; then
            log_error "DPDK requires kernel >= 5.15 (current: $(uname -r))"
            exit 1
        fi
        
        log_info "Kernel version $(uname -r) is compatible with DPDK"
    fi
}

# Main setup orchestrator
main() {
    echo "🧬 Genie Setup - AI Accelerator Disaggregation"
    echo "=============================================="
    echo ""
    
    parse_args "$@"
    load_config
    interactive_setup
    check_system
    
    case "$SETUP_MODE" in
        "basic")
            log_info "Starting basic setup..."
            exec "$SCRIPT_DIR/setup_basic.sh" \
                --pytorch-cuda "$PYTORCH_CUDA_FLAVOR" \
                $([ "$SKIP_DRIVER_INSTALL" == "true" ] && echo "--skip-driver")
            ;;
        "dpdk")
            log_info "Starting DPDK server setup..."
            exec "$SCRIPT_DIR/setup_dpdk.sh" \
                --pytorch-cuda "$PYTORCH_CUDA_FLAVOR" \
                $([ "$SKIP_DRIVER_INSTALL" == "true" ] && echo "--skip-driver") \
                $([ -n "$CONFIG_FILE" ] && echo "--config $CONFIG_FILE")
            ;;
        "fix")
            log_info "Starting fix mode..."
            exec "$SCRIPT_DIR/../verify/fix_installation.sh"
            ;;
        "")
            log_error "No setup mode specified. Use --mode or run interactively."
            show_help
            exit 1
            ;;
        *)
            log_error "Unknown setup mode: $SETUP_MODE"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
