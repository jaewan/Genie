"""
Test that co-location optimization works.
File: tests/test_colocation.py
"""

import torch
import logging
from genie.core.lazy_tensor import LazyTensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_colocation_metadata():
    """Test that we can set co-location metadata."""
    # Create LazyTensor
    x = torch.randn(10, 10, device="remote_accelerator:0")

    # Set co-location metadata (simulating what optimizer does)
    if x.metadata:
        x.metadata.colocation_group = 'kv_cache'
        x.metadata.priority = 10

    # Verify
    assert x.metadata.colocation_group == 'kv_cache'
    assert x.metadata.priority == 10

    logger.info("✅ Co-location metadata can be set")


def test_colocation_device_assignment():
    """Test that co-located operations use same device."""
    from genie.core.executor import _get_device_for_node

    # Create two tensors with same colocation group
    x = torch.randn(10, 10, device="remote_accelerator:0")
    y = torch.randn(10, 10, device="remote_accelerator:0")

    # Set metadata for co-location
    if x.metadata and y.metadata:
        x.metadata.colocation_group = 'test_group'
        y.metadata.colocation_group = 'test_group'

    # Get device assignments
    device_x = _get_device_for_node(x)
    device_y = _get_device_for_node(y)

    # Should be same device!
    assert device_x == device_y, f"Devices don't match: {device_x} vs {device_y}"

    logger.info(f"✅ Co-located operations assigned to same device: {device_x}")


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Testing Co-Location Implementation")
    logger.info("=" * 70)
    logger.info("")

    test_colocation_metadata()
    test_colocation_device_assignment()

    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ All co-location tests passed!")
    logger.info("=" * 70)
