"""
Model utilities for benchmark workloads.

Provides common model loading, management, and utility functions
used across different benchmark types.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    GPT2LMHeadModel, GPT2Tokenizer,
    BertModel, BertTokenizer,
    LlamaForCausalLM, LlamaTokenizer
)
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading and caching for benchmarks."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self.loaded_models = {}

    def load_gpt2(self, model_name: str = "gpt2-medium", device: str = "auto") -> Tuple[GPT2LMHeadModel, GPT2Tokenizer]:
        """Load GPT-2 model and tokenizer."""
        cache_key = f"gpt2_{model_name}"

        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]

        logger.info(f"Loading GPT-2 model: {model_name}")

        tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
        tokenizer.pad_token = tokenizer.eos_token

        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        result = (model, tokenizer)
        self.loaded_models[cache_key] = result
        return result

    def load_bert(self, model_name: str = "bert-base-uncased", device: str = "auto") -> Tuple[BertModel, BertTokenizer]:
        """Load BERT model and tokenizer."""
        cache_key = f"bert_{model_name}"

        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]

        logger.info(f"Loading BERT model: {model_name}")

        tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
        model = BertModel.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        result = (model, tokenizer)
        self.loaded_models[cache_key] = result
        return result

    def load_llama(self, model_name: str = "meta-llama/Llama-2-7b-hf", device: str = "auto") -> Tuple[LlamaForCausalLM, LlamaTokenizer]:
        """Load Llama model and tokenizer."""
        cache_key = f"llama_{model_name.replace('/', '_')}"

        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]

        logger.info(f"Loading Llama model: {model_name}")

        tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if device == "auto" else None
        )

        if device != "auto":
            model = model.to(device)

        result = (model, tokenizer)
        self.loaded_models[cache_key] = result
        return result

    def get_model_info(self, model, tokenizer) -> Dict[str, Any]:
        """Get basic model information."""
        return {
            'model_type': type(model).__name__,
            'tokenizer_type': type(tokenizer).__name__,
            'vocab_size': getattr(tokenizer, 'vocab_size', None),
            'max_position_embeddings': getattr(model.config, 'max_position_embeddings', None),
            'hidden_size': getattr(model.config, 'hidden_size', None),
            'num_layers': getattr(model.config, 'num_hidden_layers', None),
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'device': next(model.parameters()).device,
            'dtype': next(model.parameters()).dtype,
        }

    def estimate_model_memory(self, model) -> float:
        """Estimate model memory usage in GB."""
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())

        # Estimate activation memory (rough heuristic)
        activation_memory = param_memory * 0.5  # Conservative estimate

        total_memory = param_memory + buffer_memory + activation_memory
        return total_memory / (1024**3)  # Convert to GB


# Global model manager instance
model_manager = ModelManager()


def get_model_memory_usage(model) -> Dict[str, float]:
    """Get detailed memory usage breakdown for a model."""
    param_count = sum(p.numel() for p in model.parameters())
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())

    return {
        'parameters': param_count,
        'parameter_memory_gb': param_memory / (1024**3),
        'buffer_memory_gb': buffer_memory / (1024**3),
        'total_memory_gb': (param_memory + buffer_memory) / (1024**3),
    }


def generate_sample_input(tokenizer, model_name: str, batch_size: int = 1,
                         seq_length: int = 128) -> torch.Tensor:
    """Generate sample input for testing."""
    if "gpt2" in model_name.lower():
        # GPT-2 style input
        input_text = "The quick brown fox jumps over the lazy dog. " * (seq_length // 10)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True,
                          max_length=seq_length, padding="max_length")
    elif "bert" in model_name.lower():
        # BERT style input
        input_text = "The quick brown fox jumps over the lazy dog."
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True,
                          max_length=seq_length, padding="max_length")
    else:
        # Generic input
        input_text = "Hello world! " * (seq_length // 5)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True,
                          max_length=seq_length, padding="max_length")

    # Expand to batch size
    if batch_size > 1:
        for k, v in inputs.items():
            inputs[k] = v.repeat(batch_size, 1)

    return inputs['input_ids']


def benchmark_model_forward(model, input_tensor: torch.Tensor, num_runs: int = 10) -> Dict[str, float]:
    """Benchmark model forward pass performance."""
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_tensor)

    # Benchmark
    import time
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(input_tensor)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            latencies.append((time.time() - start) * 1000)  # Convert to ms

    return {
        'avg_latency_ms': sum(latencies) / len(latencies),
        'min_latency_ms': min(latencies),
        'max_latency_ms': max(latencies),
        'throughput_samples_per_sec': (len(latencies) * input_tensor.shape[0]) / (sum(latencies) / 1000),
    }
