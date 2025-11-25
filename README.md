# Djinn: Semantic-Aware AI Accelerator Disaggregation

**Framework-level disaggregation for AI accelerators with semantic awareness.**


### Requirements



## Quick Start

### Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd Genie

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install PyTorch (choose based on your hardware)
# For CUDA 12.1+ GPUs:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.8+ (RTX 50-series):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# For CPU-only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install Genie
pip install -e .
```

### Basic Usage

```python
# example/simple_remote_demo.py
import asyncio
import torch
from djinn.core.coordinator import DjinnCoordinator, CoordinatorConfig

async def demo():
    # Start client coordinator
    config = CoordinatorConfig(
        node_id='client',
        tcp_fallback=True
    )
    coordinator = GenieCoordinator(config)
    await coordinator.start()

    # Start server (in another terminal)
    # python3 -m djinn.backend.server.server --node-id server --control-port 5555 --data-port 5556

    # Execute operations remotely
    x = torch.randn(100, 100)
    y = torch.randn(100, 100)

    # Matrix multiplication
    result = await coordinator.execute_remote_operation(
        'aten::matmul', [x, y], 'localhost:5556'
    )
    print(f"Matrix multiply: {x.shape} @ {y.shape} = {result.shape}")

    # Add bias
    bias = torch.randn(100, 100)
    result2 = await coordinator.execute_remote_operation(
        'aten::add', [result, bias], 'localhost:5556'
    )
    print(f"Add bias: {result2.shape}")

    # Activation
    result3 = await coordinator.execute_remote_operation(
        'aten::relu', [result2], 'localhost:5556'
    )
    print(f"ReLU: {result3.shape}")

    # Verify correctness
    expected = torch.relu(torch.matmul(x, y) + bias)
    assert torch.allclose(result3, expected)
    print("✅ All operations correct!")

    await coordinator.stop()

if __name__ == '__main__':
    asyncio.run(demo())
```

### Running Tests

```bash
# Basic functionality tests
python3 -m pytest tests/integration/test_phase1_remote_execution.py -v

# Connection pooling performance
python3 -m pytest tests/integration/test_connection_pooling.py -v

# Error handling edge cases
python3 -m pytest tests/integration/test_error_handling.py -v

# Full integration (hero test)
python3 -m pytest tests/integration/test_hero_integration.py::TestHeroIntegration::test_full_workflow_integration -v

# All tests
python3 -m pytest tests/ -x --tb=short
```

## Documentation

Genie documentation is organized into 7 core documents for different audiences:

| Document | Purpose | Audience |
|----------|---------|----------|
| [0_OVERVIEW.md](docs/0_OVERVIEW.md) | System introduction, quick start | Everyone |
| [1_ARCHITECTURE.md](docs/1_ARCHITECTURE.md) | System architecture, design principles | Researchers, architects |
| [2_FRONTEND_IMPLEMENTATION.md](docs/2_FRONTEND_IMPLEMENTATION.md) | Frontend implementation details | Frontend engineers |
| [3_SCHEDULER_IMPLEMENTATION.md](docs/3_SCHEDULER_IMPLEMENTATION.md) | Scheduler implementation details | Scheduler engineers |
| [4_BACKEND_IMPLEMENTATION.md](docs/4_BACKEND_IMPLEMENTATION.md) | Backend & network transport | Backend engineers |
| [5_PERFORMANCE_VALIDATION.md](docs/5_PERFORMANCE_VALIDATION.md) | Benchmarks, profiling, validation | Performance engineers |
| [6_DEPLOYMENT_GUIDE.md](docs/6_DEPLOYMENT_GUIDE.md) | Production deployment guide | DevOps, SREs |

**Reading Paths**:
- **For Researchers**: [0_OVERVIEW](docs/0_OVERVIEW.md) → [1_ARCHITECTURE](docs/1_ARCHITECTURE.md) → [5_PERFORMANCE](docs/5_PERFORMANCE_VALIDATION.md)
- **For Engineers (New)**: [0_OVERVIEW](docs/0_OVERVIEW.md) → [1_ARCHITECTURE](docs/1_ARCHITECTURE.md) → Implementation docs (2/3/4)
- **For Engineers (Existing)**: Jump to relevant implementation doc (2/3/4)
- **For Operators**: [0_OVERVIEW](docs/0_OVERVIEW.md) → [6_DEPLOYMENT](docs/6_DEPLOYMENT_GUIDE.md)


## Development

### Running Tests

```bash
# Unit tests
python3 -m pytest tests/unit/ -v

# Integration tests
python3 -m pytest tests/integration/ -v

# Performance tests
python3 -m pytest tests/performance/ -v

# All tests with coverage
python3 -m pytest tests/ --cov=djinn
```

### Development Setup

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python3 -m pytest tests/unit/test_lazy_tensor.py -v

# Run examples
python3 examples/simple_remote_demo.py
```

## Contact & Citation

**Project Lead**: Jaewan Hong (jaewan@berkeley.edu)  

```bibtex
@inproceedings{hong2025lost,
  title={Lost in Translation: The Search for Meaning in Network-Attached AI Accelerator Disaggregation},
  author={Hong, Jaewan and Qiao, Yifan and Ponnapalli, Soujanya and Liu, Shu and Aguilera, Marcos K and Liu, Vincent and Rossbach, Christopher J and Stoica, Ion},
  booktitle={Proceedings of the 24th ACM Workshop on Hot Topics in Networks},
  pages={131--138},
  year={2025}
}
```