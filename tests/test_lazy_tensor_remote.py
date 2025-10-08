"""
Test LazyTensor with remote execution.
File: tests/test_lazy_tensor_remote.py
"""

import pytest
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_lazy_tensor_device():
    """Test LazyTensor device is set correctly."""
    from genie.core.lazy_tensor import LazyTensor

    # Create LazyTensor directly with device in kwargs
    x = LazyTensor("aten::randn", [(10, 10)], {"device": "remote_accelerator:0"})

    # Check type
    assert isinstance(x, LazyTensor), f"Expected LazyTensor, got {type(x)}"

    # Check device
    assert x.device == "remote_accelerator:0", f"Device is {x.device}"

    logger.info(f"✅ LazyTensor device set correctly: {x.device}")


def test_lazy_tensor_stays_lazy():
    """Test operations stay lazy."""
    from genie.core.lazy_tensor import LazyTensor

    # Create LazyTensors directly
    x = LazyTensor("aten::randn", [(10, 10)], {"device": "remote_accelerator:0"})
    y = LazyTensor("aten::relu", [x], {})

    # Check both are LazyTensors
    assert isinstance(x, LazyTensor)
    assert isinstance(y, LazyTensor)

    # Check neither is materialized
    assert not x.materialized, "Input should not be materialized"
    assert not y.materialized, "Output should not be materialized"

    logger.info("✅ Operations stay lazy")


def test_remote_execution():
    """Test actual remote execution."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("🧪 Testing Remote Execution")
    logger.info("=" * 60)
    logger.info("")
    logger.info("IMPORTANT: Make sure server is running!")
    logger.info("  python -m genie.runtime.simple_server")
    logger.info("")

    # Create LazyTensor on remote device
    from genie.core.lazy_tensor import LazyTensor
    x = LazyTensor("aten::randn", [(10, 10)], {"device": "remote_accelerator:0"})
    logger.info(f"1. Created LazyTensor: {x.device}")

    # Apply operation (stays lazy)
    y = LazyTensor("aten::relu", [x], {})
    logger.info(f"2. Applied relu (still lazy): materialized={y.materialized}")

    # Materialize (triggers remote execution)
    logger.info("3. Materializing (will execute remotely)...")
    result = y.materialize()

    # Verify result
    assert isinstance(result, torch.Tensor), "Result should be torch.Tensor"
    assert result.shape == (10, 10), f"Shape mismatch: {result.shape}"
    assert (result >= 0).all(), "ReLU should produce non-negative values"

    logger.info(f"4. ✅ Remote execution successful!")
    logger.info(f"   Result shape: {result.shape}")
    logger.info(f"   Result device: {result.device}")
    logger.info(f"   Result dtype: {result.dtype}")
    logger.info("")


def test_remote_execution_correctness():
    """Test remote execution matches local execution."""
    # Create identical tensors
    x_cpu = torch.randn(10, 10)
    x_remote = x_cpu.clone()

    # Local execution
    y_local = torch.relu(x_cpu)

    # Remote execution
    # Move to remote device
    from genie.core.lazy_tensor import LazyTensor
    x_lazy = LazyTensor("aten::alias", [x_remote], {})
    x_lazy.device = "remote_accelerator:0"
    y_lazy = LazyTensor("aten::relu", [x_lazy], {})
    y_remote = y_lazy.materialize()

    # Compare
    assert torch.allclose(y_local, y_remote, atol=1e-5), "Results don't match!"

    logger.info("✅ Remote execution matches local execution")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("LazyTensor Remote Execution Tests")
    logger.info("=" * 60)
    logger.info("")

    # Run tests
    test_lazy_tensor_device()
    test_lazy_tensor_stays_lazy()
    test_remote_execution()
    test_remote_execution_correctness()

    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ All tests passed!")
    logger.info("=" * 60)
