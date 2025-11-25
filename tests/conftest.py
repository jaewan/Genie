"""
Pytest configuration and shared fixtures.

Provides:
- Event loop for async tests
- Result tracking
- Common utilities
- Test environment setup
"""

import pytest
import json
import torch
from datetime import datetime
from pathlib import Path
import sys
import logging

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def pytest_configure(config):
    """Configure pytest and initialize metadata."""

    # Add markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "remote: marks tests that require remote server"
    )
    config.addinivalue_line(
        "markers", "robustness: marks robustness tests (thread safety, memory leaks)"
    )

    # Initialize test metadata
    config.test_metadata = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    print("\n" + "="*60)
    print("TEST ENVIRONMENT")
    print("="*60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {'Available' if torch.cuda.is_available() else 'Not available'}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPUs: {torch.cuda.device_count()}")
    print("="*60 + "\n")


def pytest_sessionfinish(session, exitstatus):
    """Save test metadata after session."""

    results_dir = Path('test_results')
    results_dir.mkdir(exist_ok=True)

    # Collect test results (simplified for compatibility)
    test_results = {
        'metadata': session.config.test_metadata,
        'exit_status': exitstatus,
        'test_count': {
            'total': session.testscollected,
            'passed': 0,  # Will be updated by terminal reporter
            'failed': 0,  # Will be updated by terminal reporter
            'skipped': 0, # Will be updated by terminal reporter
        }
    }

    metadata_file = results_dir / 'test_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\n✅ Test metadata saved to {metadata_file}")


@pytest.fixture
def assert_numerical_correctness():
    """Fixture for numerical correctness assertions."""

    def _assert(result, expected, rtol=1e-5, atol=1e-8, name=""):
        """Assert numerical correctness with clear error message."""
        if not torch.allclose(result, expected, rtol=rtol, atol=atol):
            max_error = torch.abs(result - expected).max().item()
            rel_error = max_error / torch.abs(expected).max().item()
            raise AssertionError(
                f"Numerical mismatch{' in ' + name if name else ''}:\n"
                f"  Max absolute error: {max_error:.2e}\n"
                f"  Max relative error: {rel_error:.2e}\n"
                f"  Tolerance: rtol={rtol}, atol={atol}"
            )

    return _assert


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture(autouse=True)
def reset_genie_state():
    """Reset genie state before each test to prevent state accumulation."""
    # Clear any cached state
    import djinn
    from djinn.frontend.core import lazy_tensor
    
    # Reset shape inference cache
    if hasattr(lazy_tensor, '_shape_inference_cache'):
        lazy_tensor._shape_inference_cache.clear()
    
    # Reset shape cache statistics
    if hasattr(lazy_tensor, '_shape_cache_hits'):
        lazy_tensor._shape_cache_hits = 0
    if hasattr(lazy_tensor, '_shape_cache_misses'):
        lazy_tensor._shape_cache_misses = 0
    
    # Reset circuit breaker
    if hasattr(lazy_tensor, '_shape_inference_circuit_breaker'):
        lazy_tensor._shape_inference_circuit_breaker['failures'] = 0
    
    yield
    
    # Cleanup after test
    if hasattr(lazy_tensor, '_shape_inference_cache'):
        lazy_tensor._shape_inference_cache.clear()


def get_free_port():
    """Get free port for testing.

    Returns:
        int: Available port number
    """
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port
