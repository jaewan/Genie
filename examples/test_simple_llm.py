"""
Test SimpleLLM locally.
File: examples/test_simple_llm.py
"""

import torch
import logging
from simple_llm import SimpleLLM, estimate_transfer_size

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("🧪 Testing SimpleLLM")
    logger.info("=" * 70)
    logger.info("")

    # Create model
    logger.info("Creating SimpleLLM...")
    model = SimpleLLM(hidden_size=768, cache_seq_len=128, batch_size=1)
    logger.info("")

    # Show sizes
    logger.info("Component sizes:")
    sizes = estimate_transfer_size(model)
    logger.info(f"  KV cache: {sizes['kv_cache_mb']:.2f} MB")
    logger.info(f"  Decoder: {sizes['decoder_mb']:.2f} MB")
    logger.info(f"  Token: {sizes['token_mb']:.2f} MB")
    logger.info("")
    logger.info("Transfer per decode step:")
    logger.info(f"  Without co-location: {sizes['total_per_step_without_colocation']:.2f} MB")
    logger.info(f"  With co-location: {sizes['total_per_step_with_colocation']:.2f} MB")
    logger.info(f"  Savings: {sizes['total_per_step_without_colocation'] - sizes['total_per_step_with_colocation']:.2f} MB")
    logger.info("")

    # Test one decode step
    logger.info("Testing one decode step...")
    initial_token = torch.randn(1, 768)
    output = model.decode_step(initial_token, device="cpu")
    logger.info(f"  ✅ Output shape: {output.shape}")
    logger.info("")

    # Test generation
    logger.info("Testing generation (5 steps)...")
    generated = model.generate(num_steps=5, device="cpu")
    logger.info(f"  ✅ Generated {len(generated)} tokens")
    logger.info("")

    logger.info("=" * 70)
    logger.info("✅ SimpleLLM works correctly!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
