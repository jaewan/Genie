"""
Pytest configuration for Genie testing.

This module provides shared fixtures and configuration for all tests.

GPU Testing Strategy:
- Skip GPU tests if no GPU available (CI/local development)
- Mark tests with GPU requirements explicitly
- Provide GPU fixtures for common setups
"""

import sys
from pathlib import Path
import pytest
import os
import torch
import subprocess
import time
import requests


# Ensure repository root is on sys.path so `import genie` works without editable install
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


# GPU availability checks
def has_gpu():
    """Check if CUDA GPU is available."""
    return torch.cuda.is_available()


def has_multiple_gpus():
    """Check if multiple GPUs available."""
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2


def get_gpu_memory_gb(device=0):
    """Get GPU memory in GB."""
    if not has_gpu():
        return 0
    props = torch.cuda.get_device_properties(device)
    return props.total_memory / (1024**3)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
    config.addinivalue_line("markers", "multigpu: marks tests as requiring multiple GPUs")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu_memory: marks tests as requiring minimum GPU memory")


# Fixtures
@pytest.fixture
def device():
    """Default device for tests."""
    return torch.device("cpu")


@pytest.fixture
def gpu_device():
    """Provide GPU device if available, skip test otherwise."""
    if not has_gpu():
        pytest.skip("GPU not available")

    device = torch.device('cuda:0')

    # Reset GPU state before test
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    yield device

    # Cleanup after test
    torch.cuda.empty_cache()


@pytest.fixture
def multi_gpu_devices():
    """Provide multiple GPU devices if available."""
    if not has_multiple_gpus():
        pytest.skip("Multiple GPUs not available")

    num_gpus = torch.cuda.device_count()
    devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]

    # Reset all GPUs
    for i in range(num_gpus):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    yield devices

    # Cleanup
    for i in range(num_gpus):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()


@pytest.fixture
def gpu_server(gpu_device):
    """Start GPU execution server for remote tests."""
    # Start server in background
    proc = subprocess.Popen(
        ['python', '-m', 'genie.runtime.simple_server',
         '--port', '8888',
         '--device', 'cuda:0'],  # ← Explicitly use GPU
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait for server to start
    time.sleep(2)

    # Check if server is responding
    try:
        response = requests.get('http://localhost:8888/health')
        assert response.status_code == 200
        server_info = response.json()
        assert server_info.get('device') == 'cuda:0', "Server not using GPU!"
    except Exception as e:
        proc.terminate()
        pytest.skip(f"Server startup failed: {e}")

    yield proc

    # Cleanup
    proc.terminate()
    proc.wait()


@pytest.fixture
def sample_tensors():
    """Provide diverse test tensors for correctness testing."""
    return [
        # Normal cases
        torch.randn(10, 10),
        torch.randn(1, 100, 50),
        torch.randn(4, 3, 224, 224),  # Image-like

        # Edge cases
        torch.randn(1),  # Scalar-like
        torch.randn(1, 1, 1),  # All-ones shape
        torch.randn(100, 1),  # Broadcasting candidate
        torch.randn(0, 10),  # Empty tensor

        # Different dtypes
        torch.randn(10, 10, dtype=torch.float16),
        torch.randn(10, 10, dtype=torch.float64),
        torch.randint(0, 100, (10, 10)),  # Integer

        # Different devices (for mixed-device testing)
        torch.randn(10, 10, device='cpu'),
    ]


@pytest.fixture
def correctness_test_suite():
    """Provide correctness test suite."""
    class CorrectnessTestSuite:
        """Test that LazyTensor operations produce identical results to PyTorch."""

        def test_operation(self, op_name, native_op, lazy_op, test_inputs):
            """Test a single operation for correctness."""
            # Run native
            native_result = native_op(*test_inputs)

            # Run lazy
            with genie.capture():
                lazy_inputs = []
                for inp in test_inputs:
                    if isinstance(inp, torch.Tensor):
                        lazy_inputs.append(torch.tensor(inp.numpy()))
                    else:
                        # Scalar value
                        lazy_inputs.append(inp)
                lazy_result = lazy_op(*lazy_inputs)

            # Materialize
            lazy_concrete = lazy_result.cpu()

            # Compare
            try:
                torch.testing.assert_close(
                    lazy_concrete, native_result,
                    rtol=1e-4, atol=1e-6,
                    msg=f"{op_name} results differ"
                )
            except AssertionError as e:
                # Detailed error report
                print(f"FAILED: {op_name}")
                print(f"Native: {native_result}")
                print(f"Lazy: {lazy_concrete}")
                print(f"Max diff: {(lazy_concrete - native_result).abs().max()}")
                raise

    return CorrectnessTestSuite()


@pytest.fixture(autouse=True)
def reset_graph_state():
    """Reset thread-local FX builder and LazyTensor IDs before each test.
    Prevents 'Cannot add operations to finalized graph' across tests.
    """
    try:
        from genie.core.fx_graph_builder import FXGraphBuilder
        FXGraphBuilder.reset()
    except Exception:
        pass


@pytest.fixture(autouse=True, scope="session")
def default_disable_cpp_dataplane():
    """Disable C++ data plane by default for test stability.
    Can be overridden per-test by setting GENIE_DISABLE_CPP_DATAPLANE explicitly.
    """
    if os.environ.get("GENIE_DISABLE_CPP_DATAPLANE") is None:
        os.environ["GENIE_DISABLE_CPP_DATAPLANE"] = "1"

    try:
        from genie.core.lazy_tensor import LazyTensor
        if hasattr(LazyTensor, "reset_id_counter"):
            LazyTensor.reset_id_counter()
    except Exception:
        pass


