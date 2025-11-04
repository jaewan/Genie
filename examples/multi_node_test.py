#!/usr/bin/env python3
"""
Multi-node testing script for Genie Week 1 implementation.

Usage:
    # On SERVER machine (with GPU):
    python -m genie.runtime.simple_server --host 0.0.0.0 --port 8888

    # On CLIENT machine:
    python examples/multi_node_test.py SERVER_IP
"""

import sys
import os
import torch
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_connectivity(server_url: str):
    """Test basic HTTP connectivity to server."""
    import requests

    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        response.raise_for_status()
        health = response.json()

        logger.info("✅ Server connectivity: OK"        logger.info(f"   Status: {health['status']}")
        logger.info(f"   Device: {health['device']}")
        logger.info(f"   CUDA available: {health['cuda_available']}")
        return True

    except Exception as e:
        logger.error(f"❌ Server connectivity failed: {e}")
        return False

def test_tensor_transfer(server_url: str):
    """Test tensor transfer via HTTP."""
    import requests
    import io

    try:
        # Create test tensor
        test_tensor = torch.randn(100, 100)

        # Serialize
        tensor_bytes = io.BytesIO()
        torch.save(test_tensor, tensor_bytes)
        tensor_bytes.seek(0)

        # Send to server
        response = requests.post(
            f"{server_url}/execute",
            files={'tensor_file': ('tensor.pt', tensor_bytes, 'application/octet-stream')},
            data={'operation': 'relu'},
            timeout=30
        )
        response.raise_for_status()

        # Deserialize result
        result = torch.load(io.BytesIO(response.content))

        # Verify correctness
        expected = torch.relu(test_tensor)
        if torch.allclose(result, expected, atol=1e-5):
            logger.info(f"✅ Tensor transfer: OK (shape: {result.shape})")
            return True
        else:
            logger.error("❌ Tensor transfer: Results don't match")
            return False

    except Exception as e:
        logger.error(f"❌ Tensor transfer failed: {e}")
        return False

def test_lazy_tensor_remote(server_url: str):
    """Test LazyTensor remote execution."""
    try:
        # Set server URL for this test
        os.environ['GENIE_SERVER_URL'] = server_url

        # Create tensor on remote device
        x = torch.randn(100, 100, device='remote_accelerator:0')
        logger.info(f"Created LazyTensor: device={x.device}, type={type(x)}")

        # Chain operations (should stay lazy)
        y = x.relu().sigmoid()
        logger.info(f"Chained operations: materialized={y.materialized}")

        # Execute remotely
        start_time = time.time()
        result = y.materialize()
        elapsed = time.time() - start_time

        # Verify result
        if isinstance(result, torch.Tensor) and result.shape == (100, 100):
            logger.info(f"✅ Remote execution: OK (time: {elapsed:.".3f")")
            logger.info(f"   Result device: {result.device}")
            logger.info(f"   Result dtype: {result.dtype}")
            return True
        else:
            logger.error(f"❌ Remote execution: Invalid result (type: {type(result)})")
            return False

    except Exception as e:
        logger.error(f"❌ Remote execution failed: {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python multi_node_test.py SERVER_IP")
        print("Example: python multi_node_test.py 192.168.1.100")
        sys.exit(1)

    server_ip = sys.argv[1]
    server_url = f"http://{server_ip}:8888"

    logger.info("=" * 60)
    logger.info("🧪 Genie Multi-Node Testing")
    logger.info("=" * 60)
    logger.info(f"Server URL: {server_url}")
    logger.info("")

    # Test 1: Basic connectivity
    logger.info("Test 1: Server Connectivity")
    if not test_basic_connectivity(server_url):
        logger.error("❌ Cannot proceed - server not accessible")
        sys.exit(1)
    logger.info("")

    # Test 2: Tensor transfer
    logger.info("Test 2: Tensor Transfer")
    if not test_tensor_transfer(server_url):
        logger.error("❌ Tensor transfer failed")
        sys.exit(1)
    logger.info("")

    # Test 3: LazyTensor remote execution
    logger.info("Test 3: LazyTensor Remote Execution")
    if not test_lazy_tensor_remote(server_url):
        logger.error("❌ LazyTensor remote execution failed")
        sys.exit(1)
    logger.info("")

    logger.info("=" * 60)
    logger.info("🎉 All tests passed! Multi-node setup working correctly.")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
