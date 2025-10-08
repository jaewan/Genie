"""
Measure baseline LLM performance WITHOUT co-location.
File: benchmarks/measure_baseline_llm.py

This simulates semantic-blind placement:
- KV cache on one device
- Decoder on another device
- Must transfer cache every step
"""

import sys
import os
sys.path.append('../examples')

import torch
import time
import logging
from simple_llm import SimpleLLM, estimate_transfer_size

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def measure_baseline_no_colocation(model: SimpleLLM, num_steps: int = 10) -> dict:
    """
    Measure baseline WITHOUT co-location.

    Simulates:
    - KV cache on server A
    - Decoder on server B
    - Transfer cache every step
    """
    logger.info("🔍 Measuring BASELINE (no co-location)...")
    logger.info("   Simulating: KV cache and decoder on DIFFERENT servers")
    logger.info("")

    latencies = []

    initial_token = torch.randn(1, model.hidden_size)
    current_token = initial_token

    for step in range(num_steps):
        logger.info(f"  Step {step + 1}/{num_steps}")

        start = time.time()

        # Simulate transfer overhead
        # In reality: would transfer KV cache over network
        # Here: add artificial delay (15ms per MB transferred)
        sizes = estimate_transfer_size(model)
        transfer_mb = sizes['total_per_step_without_colocation']
        transfer_time = transfer_mb * 0.015  # 15ms per MB (typical for 100G network)

        logger.debug(f"    Simulated transfer: {transfer_mb:.2f} MB → {transfer_time*1000:.2f}ms")
        time.sleep(transfer_time)

        # Execute decode step (CPU)
        output = model.decode_step(current_token, device="cpu")

        elapsed = (time.time() - start) * 1000  # ms
        latencies.append(elapsed)

        logger.debug(f"    Total latency: {elapsed:.2f}ms")

        current_token = output

    avg_latency = sum(latencies) / len(latencies)

    logger.info("")
    logger.info(f"✅ Baseline measurement complete:")
    logger.info(f"   Steps: {num_steps}")
    logger.info(f"   Average latency: {avg_latency:.2f}ms per step")
    logger.info(f"   Total time: {sum(latencies):.2f}ms")
    logger.info("")

    return {
        'num_steps': num_steps,
        'latencies_ms': latencies,
        'avg_latency_ms': avg_latency,
        'total_ms': sum(latencies),
        'strategy': 'no_colocation'
    }


def main():
    logger.info("=" * 70)
    logger.info("📊 Baseline LLM Measurement (NO Co-location)")
    logger.info("=" * 70)
    logger.info("")

    # Create model
    model = SimpleLLM(hidden_size=768, cache_seq_len=128, batch_size=1)
    logger.info("")

    # Measure baseline
    result = measure_baseline_no_colocation(model, num_steps=10)

    # Save result
    import json
    with open('baseline_no_colocation.json', 'w') as f:
        json.dump(result, f, indent=2)

    logger.info("💾 Results saved to: baseline_no_colocation.json")
    logger.info("")

    logger.info("=" * 70)
    logger.info("✅ Baseline measurement complete!")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Next step: Implement co-location and measure improvement")


if __name__ == "__main__":
    main()
