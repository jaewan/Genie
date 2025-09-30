# Genie Scripts Directory

This directory contains the essential scripts for GPUdirect RDMA setup and testing.

## Directory Structure

```
scripts/
├── setup/              # Installation and setup scripts
│   ├── setup.sh       # Main setup orchestrator
│   ├── setup_basic.sh # Basic Python/PyTorch setup
│   ├── setup_dpdk.sh  # DPDK and networking setup
│   └── setup_gpu_server.sh # GPU server configuration
│
├── build/             # Build scripts
│   ├── build_extensions.py # Build PyTorch C++ extensions
│   └── setup.py      # Package installation (pip)
│
├── verify/            # Verification and testing scripts
│   ├── verify_dpdk_setup.sh # Verify DPDK installation
│   └── fix_installation.sh  # Fix common issues
│
└── utils/             # Utility scripts
    └── (future utilities)
```

## Usage

### Initial Setup
```bash
# For development (CPU or basic GPU)
./setup/setup.sh --mode basic

# For production with DPDK
./setup/setup.sh --mode dpdk

# For GPU server with full stack
./setup/setup_gpu_server.sh
```

### Building Components
```bash
# Build C++ extensions
python scripts/build/build_extensions.py

# Install package in development mode
pip install -e scripts/build/
```

### Verification
```bash
# Verify DPDK setup
./scripts/verify/verify_dpdk_setup.sh

# Fix installation issues
./scripts/verify/fix_installation.sh
```

## Script Descriptions

### Setup Scripts
- **setup.sh**: Main orchestrator that calls appropriate setup scripts based on mode
- **setup_basic.sh**: Sets up Python environment, PyTorch, and basic dependencies
- **setup_dpdk.sh**: Installs and configures DPDK with GPU support
- **setup_gpu_server.sh**: Complete GPU server setup including drivers and CUDA

### Build Scripts
- **build_extensions.py**: Builds Genie's PyTorch C++ extensions for device registration
- **setup.py**: Standard Python package setup for pip installation

### Verification Scripts
- **verify_dpdk_setup.sh**: Checks DPDK installation and configuration
- **fix_installation.sh**: Diagnoses and fixes common installation problems

## Best Practices

1. **Always use virtual environments**: Scripts automatically create/activate venvs
2. **Check prerequisites**: Scripts verify system requirements before proceeding
3. **Idempotent execution**: Scripts can be run multiple times safely
4. **Clear error messages**: Scripts provide actionable error messages
5. **Logging**: Important operations are logged for debugging

## Troubleshooting

If you encounter issues:
1. Run `./scripts/verify/fix_installation.sh` first
2. Check logs in `/tmp/genie_setup.log`
3. Ensure you have appropriate permissions (some scripts need sudo)
4. For GPU issues, ensure CUDA drivers and CUDA toolkit match PyTorch wheels:
   - PyTorch 2.8 → cu128 (CUDA 12.8)
   - PyTorch 2.2–2.5 → cu121 (CUDA 12.1)

## Contributing

When adding new scripts:
1. Place in appropriate subdirectory
2. Add clear documentation in script header
3. Update this README with description
4. Follow existing naming conventions
5. Make scripts idempotent when possible
