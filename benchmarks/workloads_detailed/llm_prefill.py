"""
LLM Prefill Workload - Shows phase-aware parallelization.

Semantic optimization:
- Detect prefill phase (parallel, large batch)
- Parallelize across sequence dimension
- Use multiple GPUs for throughput

Expected result:
- No Semantics: 200ms (single GPU)
- Full Genie: 120ms (parallelized across 2 GPUs)
- Speedup: 1.66x
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional
import time


class LLMPrefillWorkload:
    """
    LLM Prefill - Shows phase-aware parallelization.

    Semantic optimization:
    - Detect prefill phase (parallel, large batch)
    - Parallelize across sequence dimension
    - Use multiple GPUs for throughput
    """

    def __init__(self, model_name: str = "gpt2", seq_length: int = 512):
        self.model_name = model_name
        self.seq_length = seq_length
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model from HuggingFace."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading {self.model_name} for prefill...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.eval()

            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"✓ Loaded {self.model_name} for prefill")
            return True

        except ImportError:
            print("❌ transformers not available - using mock model")
            self.model = self._create_mock_model()
            self.tokenizer = self._create_mock_tokenizer()
            return False

    def _create_mock_model(self):
        """Create a simple mock model for testing."""
        class MockGPT2(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = nn.Embedding(50257, 768)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(768, 12, batch_first=True),
                    num_layers=12
                )
                self.lm_head = nn.Linear(768, 50257)

            def forward(self, input_ids, **kwargs):
                # Mock prefill implementation
                embeddings = self.embeddings(input_ids)
                hidden = self.transformer(embeddings)
                logits = self.lm_head(hidden)
                return type('Output', (), {'logits': logits})()

        return MockGPT2()

    def _create_mock_tokenizer(self):
        """Create a simple mock tokenizer."""
        class MockTokenizer:
            def __init__(self):
                self.pad_token = "[PAD]"
                self.eos_token = "[EOS]"

            def encode(self, text, return_tensors='pt', **kwargs):
                # Create longer sequence for prefill
                tokens = list(range(100, 100 + self.seq_length))
                return torch.tensor([tokens])

            def decode(self, tokens):
                return "Mock prefill decoded text"

        return MockTokenizer()

    def get_sample_inputs(self) -> List[torch.Tensor]:
        """Get sample inputs for the workload."""
        if self.tokenizer is None:
            self.load_model()

        # Create long prompt for prefill phase
        prompt = "Once upon a time " * (self.seq_length // 4)  # ~512 tokens
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        return [input_ids]

    def run(self, baseline_config: Dict) -> Dict[str, Any]:
        """
        Run prefill workload.

        Key metrics:
        - Total latency for long sequence
        - Throughput (tokens/second)
        - Memory usage
        """
        if self.model is None:
            self.load_model()

        # Create long input for prefill
        prompt = "Once upon a time " * (self.seq_length // 4)
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        print(f"Prefill input: {input_ids.shape} (seq_len={input_ids.shape[1]})")

        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(input_ids)

        end_time = time.perf_counter()

        return {
            'input_length': input_ids.shape[1],
            'output_length': outputs.logits.shape[1],
            'total_latency_ms': (end_time - start_time) * 1000,
            'throughput_tokens_per_sec': input_ids.shape[1] / (end_time - start_time),
            'baseline': baseline_config.get('baseline', 'unknown'),
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get workload metadata."""
        return {
            'workload': 'llm_prefill',
            'model': self.model_name,
            'seq_length': self.seq_length,
            'description': 'LLM context processing (prefill phase)',
            'key_optimization': 'parallelization',
            'expected_speedup': '1.3x vs no semantics',
            'parallel_dimension': 'sequence_length'
        }
