"""
Simple LLM-like workload for testing co-location optimization.
File: examples/simple_llm.py

Simulates:
- Large KV cache (persistent)
- Small decoder (per-token)
- Sequential decode steps
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class SimpleLLM(nn.Module):
    """
    Simplified LLM for testing co-location optimization.

    Components:
    - KV cache: Large, persistent tensor (simulates attention cache)
    - Decoder: Small network (simulates token generation)
    """

    def __init__(self,
                 hidden_size: int = 768,
                 cache_seq_len: int = 128,
                 batch_size: int = 1):
        """
        Initialize SimpleLLM.

        Args:
            hidden_size: Hidden dimension size
            cache_seq_len: Sequence length for KV cache
            batch_size: Batch size
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.cache_seq_len = cache_seq_len
        self.batch_size = batch_size

        # KV cache (large, persistent)
        # Shape: (batch, seq_len, hidden_size)
        self.kv_cache = torch.randn(batch_size, cache_seq_len, hidden_size)
        logger.info(f"KV cache size: {self.kv_cache.numel() * 4 / 1024 / 1024:.2f} MB")

        # Decoder (small network)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        logger.info(f"Decoder size: {decoder_params * 4 / 1024 / 1024:.2f} MB")

    def decode_step(self, token_embedding: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        """
        Perform one decode step.

        This simulates:
        1. Accessing KV cache (large)
        2. Running decoder (small)

        Args:
            token_embedding: Current token embedding (batch, hidden_size)
            device: Where to execute ("cpu" or "remote_accelerator:0")

        Returns:
            Next token prediction (batch, hidden_size)
        """
        # Move token to device
        if device != "cpu":
            token_embedding = token_embedding.to(device)

        # Access KV cache (simulates attention)
        # In real LLM: Q @ K.T → softmax → @ V
        # Here: Simple matmul for demonstration
        kv_on_device = self.kv_cache.to(device)
        attention = torch.matmul(
            token_embedding.unsqueeze(1),  # (batch, 1, hidden)
            kv_on_device.transpose(-1, -2)  # (batch, hidden, seq_len)
        )  # Result: (batch, 1, seq_len)

        # Apply softmax (simulates attention weights)
        attention_weights = torch.softmax(attention, dim=-1)

        # Weighted sum (simulates attention output)
        context = torch.matmul(
            attention_weights,  # (batch, 1, seq_len)
            kv_on_device  # (batch, seq_len, hidden)
        )  # Result: (batch, 1, hidden)

        context = context.squeeze(1)  # (batch, hidden)

        # Run decoder
        decoder_on_device = self.decoder.to(device)
        output = decoder_on_device(context)

        return output

    def generate(self,
                 num_steps: int = 10,
                 device: str = "cpu",
                 initial_token: torch.Tensor = None) -> list:
        """
        Generate multiple tokens.

        Args:
            num_steps: Number of decode steps
            device: Where to execute
            initial_token: Initial token embedding

        Returns:
            List of generated token embeddings
        """
        if initial_token is None:
            initial_token = torch.randn(self.batch_size, self.hidden_size)

        generated = []
        current_token = initial_token

        for step in range(num_steps):
            logger.debug(f"Decode step {step + 1}/{num_steps}")

            # Decode one step
            next_token = self.decode_step(current_token, device=device)
            generated.append(next_token)

            # Use output as next input
            current_token = next_token

        return generated


def estimate_transfer_size(model: SimpleLLM) -> dict:
    """Estimate transfer sizes for co-location analysis."""
    kv_size_mb = model.kv_cache.numel() * 4 / 1024 / 1024
    decoder_size_mb = sum(p.numel() for p in model.decoder.parameters()) * 4 / 1024 / 1024
    token_size_mb = model.hidden_size * 4 / 1024 / 1024

    return {
        'kv_cache_mb': kv_size_mb,
        'decoder_mb': decoder_size_mb,
        'token_mb': token_size_mb,
        'total_per_step_without_colocation': kv_size_mb + token_size_mb,
        'total_per_step_with_colocation': token_size_mb
    }
