## Genie

Semantic-driven, framework-level disaggregation for AI accelerators. Phase 1 provides a custom device, LazyTensor engine, FX integration, basic patterns, and a runnable example.

### Requirements

- Python 3.10.x recommended (3.8+ may work)
- PyTorch 2.1.2
- Optional (GPU): CUDA 12.1, NVIDIA driver 535+, cuDNN

### Quickstart (CPU)

```bash
# Create and activate virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU) and torchvision matching 2.1.2
pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cpu

# Dev dependencies (pytest, etc.)
pip install -r requirements-dev.txt

# Build & install genie (editable)
pip install -e .

# Validate environment
python -c "from genie import validate_environment; print('env:', validate_environment(strict=False))"

# Run tests
pytest -q

# Run example (ResNet-18 based minimal demo)
PYTHONPATH=$(pwd) python example/resnet18_demo.py
```

Expected example output (CPU-only):

```
Output shape: (1, 64, 112, 112)
Materialization time: XX.XX ms
Graph nodes: 2 | edges: 1
FX nodes: 7X
handoff.valid=True
```

### Quickstart (CUDA 12.1)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install PyTorch + CUDA 12.1 binaries
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements-dev.txt
pip install -e .

python -c "from genie import validate_environment; print('env:', validate_environment(strict=False))"

pytest -q
PYTHONPATH=$(pwd) python example/resnet18_demo.py
```

Note: Phase 1 executes locally (CPU materialization) even when CUDA is available. GPU/remote execution is targeted in later phases.

### Build C++ extension manually (optional)

The `genie/csrc/device.cpp` extension is built automatically by `pip install -e .`. To build in-place:

```bash
python setup.py build_ext --inplace
```

### Project layout

- `genie/core`: device, dispatcher, LazyTensor, FX utilities
- `genie/semantic`: Semantic Analyzer scaffold, pattern registry, handoff contracts
- `genie/patterns`: basic pattern plugins and FX patterns
- `example/resnet18_demo.py`: minimal demo using ResNet-18 conv1 weights
- `tests/`: unit/integration tests for Phase 1

### Troubleshooting

- Editable install fails complaining about `torch` not found:
  - Ensure you install `torch` first, then run `pip install -e .`.
- `ModuleNotFoundError: genie` when running the example:
  - Prefix with `PYTHONPATH=$(pwd)` or install with `pip install -e .`.
- Missing `torchvision` for the demo:
  - `pip install torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu` (or `+cu121`).
- DPDK not found warning:
  - Phase 1 is CPU-only; DPDK is optional. The warning can be ignored for now.

### License

See `LICENSE`.
