"""
Test simple client.
File: tests/test_simple_client.py
"""

import pytest
import torch
from genie.runtime.simple_client import RemoteExecutionClient, get_client
import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_health_check():
    """Test health check."""
    client = RemoteExecutionClient(server_url="http://localhost:8888")

    try:
        health = client.health_check()

        assert health['status'] == 'healthy'
        assert 'device' in health

        logger.info(f"✅ Health check passed: {health}")

    except Exception as e:
        pytest.fail(f"Health check failed: {e}")


def test_execute_relu():
    """Test executing relu."""
    client = RemoteExecutionClient(server_url="http://localhost:8888")

    # Create test tensor
    x = torch.randn(10, 10)

    # Execute remotely
    result = client.execute("relu", x)

    # Verify
    expected = torch.relu(x)
    assert result.shape == expected.shape
    assert torch.allclose(result, expected, atol=1e-5)

    logger.info(f"✅ ReLU execution passed")


def test_execute_multiple_operations():
    """Test multiple operations."""
    client = RemoteExecutionClient(server_url="http://localhost:8888")

    x = torch.randn(5, 5)

    operations = ['relu', 'sigmoid', 'tanh', 'abs']

    for op in operations:
        result = client.execute(op, x)

        # Verify shape
        assert result.shape == x.shape

        # Verify against local execution
        expected = getattr(torch, op)(x)
        assert torch.allclose(result, expected, atol=1e-5)

        logger.info(f"✅ {op} passed")


def test_client_statistics():
    """Test client statistics."""
    client = RemoteExecutionClient(server_url="http://localhost:8888")

    # Execute some operations
    x = torch.randn(10, 10)
    client.execute("relu", x)
    client.execute("sigmoid", x)

    # Get stats
    stats = client.get_stats()

    assert stats['requests_total'] == 2
    assert stats['requests_success'] == 2
    assert stats['requests_failed'] == 0
    assert stats['success_rate'] == 1.0
    assert stats['avg_time_seconds'] > 0

    logger.info(f"✅ Statistics: {stats}")


if __name__ == "__main__":
    # Run tests
    logger.info("=" * 60)
    logger.info("Testing Simple Client")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Make sure server is running:")
    logger.info("  python -m genie.runtime.simple_server")
    logger.info("")

    test_health_check()
    test_execute_relu()
    test_execute_multiple_operations()
    test_client_statistics()

    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ All client tests passed!")
    logger.info("=" * 60)
