"""
Simple demo of remote execution with Genie.
File: examples/simple_remote_demo.py

Prerequisites:
1. Start server: python -m genie.runtime.simple_server
2. Run this demo: python examples/simple_remote_demo.py
"""

import torch
import logging
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("🎯 Genie Remote Execution Demo")
    logger.info("=" * 70)
    logger.info("")

    # Step 1: Create tensor on remote device
    logger.info("Step 1: Creating tensor on remote_accelerator...")
    from genie.core.lazy_tensor import LazyTensor
    x = LazyTensor("aten::randn", [(100, 100)], {"device": "remote_accelerator:0"})
    logger.info(f"  ✅ Created tensor: shape={x.shape}, device={x.device}")
    logger.info("")

    # Step 2: Chain operations (all stay lazy)
    logger.info("Step 2: Chaining operations...")
    y = x.relu()
    logger.info(f"  ✅ Applied relu (lazy)")

    z = y.sigmoid()
    logger.info(f"  ✅ Applied sigmoid (lazy)")
    logger.info("")

    # Step 3: Materialize (triggers remote execution)
    logger.info("Step 3: Materializing (executing remotely)...")
    start_time = time.time()
    result = z.cpu()
    elapsed = time.time() - start_time
    logger.info(f"  ✅ Execution completed in {elapsed:.3f}s")
    logger.info("")

    # Step 4: Verify result
    logger.info("Step 4: Verifying result...")
    logger.info(f"  Result shape: {result.shape}")
    logger.info(f"  Result device: {result.device}")
    logger.info(f"  Result dtype: {result.dtype}")
    logger.info(f"  Value range: [{result.min():.4f}, {result.max():.4f}]")
    logger.info("")

    # Step 5: Compare with local execution
    logger.info("Step 5: Comparing with local execution...")
    x_local = torch.randn(100, 100)

    start_local = time.time()
    result_local = torch.sigmoid(torch.relu(x_local))
    elapsed_local = time.time() - start_local

    logger.info(f"  Local execution: {elapsed_local:.3f}s")
    logger.info(f"  Remote execution: {elapsed:.3f}s")
    logger.info(f"  Overhead: {(elapsed - elapsed_local) * 1000:.1f}ms")
    logger.info("")

    logger.info("=" * 70)
    logger.info("✅ Demo completed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
