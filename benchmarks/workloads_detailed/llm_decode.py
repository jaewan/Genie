"""
LLM Decode Workload - Shows co-location is critical.

Semantic optimization:
- Detect decode phase (sequential, small batch)
- Co-locate KV cache + decoder on same GPU
- Avoid transferring GB-sized cache every step

Expected result:
- No Semantics: 500ms (transfers cache every step)
- Full Genie: 100ms (co-location, only transfer query)
- Speedup: 5x
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
import time


class LLMDecodeWorkload:
    """GPT-2 autoregressive generation."""

    def __init__(self, model_name: str = "gpt2", max_new_tokens: int = 100):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model from HuggingFace."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.eval()

            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"✓ Loaded {self.model_name}")
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

            def forward(self, input_ids, past_key_values=None, use_cache=False, **kwargs):
                # Simple mock implementation
                embeddings = self.embeddings(input_ids)
                hidden = self.transformer(embeddings)
                logits = self.lm_head(hidden)

                if use_cache:
                    # Mock past_key_values (simplified)
                    mock_kv = []
                    for _ in range(12):  # 12 layers
                        mock_kv.append((
                            torch.randn(1, 12, 1, 64),  # key
                            torch.randn(1, 12, 1, 64)   # value
                        ))
                    return type('Output', (), {'logits': logits, 'past_key_values': mock_kv})()

                return type('Output', (), {'logits': logits})()

        return MockGPT2()

    def _create_mock_tokenizer(self):
        """Create a simple mock tokenizer."""
        class MockTokenizer:
            def __init__(self):
                self.pad_token = "[PAD]"
                self.eos_token = "[EOS]"

            def encode(self, text, return_tensors='pt'):
                # Simple mock encoding
                tokens = [101, 202, 303, 404]  # Mock token IDs
                return torch.tensor([tokens])

            def decode(self, tokens):
                return "Mock decoded text"

        return MockTokenizer()

    def get_sample_inputs(self) -> List[torch.Tensor]:
        """Get sample inputs for the workload."""
        if self.tokenizer is None:
            self.load_model()

        prompt = "Once upon a time"
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        return [input_ids]

    def run_reference(self) -> torch.Tensor:
        """
        Reference implementation - GROUND TRUTH.
        
        ALL baselines must produce identical output to this reference.
        This ensures we're measuring the same workload.
        """
        if self.model is None:
            self.load_model()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        prompt = "Once upon a time"
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            # Generate exactly max_new_tokens
            outputs = self.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + self.max_new_tokens,
                num_return_sequences=1,
                do_sample=False,  # Deterministic
                pad_token_id=self.tokenizer.eos_token_id
            )

        return outputs.cpu()

    def expected_operation_count(self) -> int:
        """
        Expected number of forward passes.
        
        For autoregressive generation with KV cache:
        - Prefill: 1 forward pass on full input
        - Decode: max_new_tokens forward passes (1 per generated token)
        - Total: 1 + max_new_tokens
        """
        return 1 + self.max_new_tokens

    def expected_network_bytes(self) -> int:
        """
        Expected network traffic for remote execution.
        
        WITHOUT semantic optimization (no co-location):
        - KV cache size: ~4 bytes per element × seq_len × num_layers × hidden_dim
        - For GPT-2: ~4MB per decode step
        - Total for 100 steps: ~400 MB
        
        WITH semantic optimization (co-location):
        - Only transfer query vector: ~3KB per step
        - Total for 100 steps: ~300 KB
        
        Returns bytes for co-located execution.
        """
        # Query vector size (approximately)
        query_size_bytes = 3072  # 3KB
        return query_size_bytes * self.max_new_tokens

    def run(self, baseline_config: Dict) -> Dict[str, Any]:
        """
        Run autoregressive decode.

        CRITICAL: Must generate exactly max_new_tokens for consistency.

        Key metrics:
        - Latency per token
        - Network bytes transferred
        - GPU utilization
        """
        if self.model is None:
            self.load_model()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        prompt = "Once upon a time"
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)

        # Generate tokens
        generated_tokens = []
        kv_cache = None

        start_time = time.perf_counter()

        for i in range(self.max_new_tokens):
            with torch.no_grad():
                if kv_cache is None:
                    # Initial forward pass (prefill)
                    outputs = self.model(input_ids)
                    if hasattr(outputs, 'past_key_values'):
                        kv_cache = outputs.past_key_values
                else:
                    # Decode step (use cached keys/values)
                    # This is where co-location matters!
                    outputs = self.model(
                        input_ids[:, -1:],  # Only last token
                        past_key_values=kv_cache,  # Reuse cache
                        use_cache=True
                    )
                    if hasattr(outputs, 'past_key_values'):
                        kv_cache = outputs.past_key_values

                # Sample next token
                next_token = outputs.logits[:, -1, :].argmax(dim=-1)
                generated_tokens.append(next_token.item())

                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

        end_time = time.perf_counter()

        return {
            'tokens_generated': len(generated_tokens),
            'total_latency_ms': (end_time - start_time) * 1000,
            'latency_per_token_ms': (end_time - start_time) * 1000 / max(len(generated_tokens), 1),
            'kv_cache_size_mb': self._estimate_kv_cache_size(kv_cache),
            'baseline': baseline_config.get('baseline', 'unknown'),
        }

    def _estimate_kv_cache_size(self, kv_cache) -> float:
        """Estimate KV cache size in MB."""
        if kv_cache is None:
            return 0.0

        total_bytes = 0

        try:
            # Handle real GPT-2 kv_cache format
            if hasattr(kv_cache, '__iter__') and len(kv_cache) > 0:
                for layer_cache in kv_cache:
                    if isinstance(layer_cache, (list, tuple)) and len(layer_cache) >= 2:
                        for tensor in layer_cache:
                            if hasattr(tensor, 'numel'):
                                total_bytes += tensor.numel() * tensor.element_size()
        except Exception:
            # Fallback for mock model
            total_bytes = 1024 * 1024 * 4  # 4MB mock cache

        return total_bytes / (1024 * 1024)

    def get_metadata(self) -> Dict[str, Any]:
        """Get workload metadata."""
        return {
            'workload': 'llm_decode',
            'model': self.model_name,
            'max_new_tokens': self.max_new_tokens,
            'description': 'LLM autoregressive generation (decode phase)',
            'key_optimization': 'kv_cache_colocation',
            'expected_speedup': '3-5x vs no semantics',
            'expected_network_reduction': '95%+'
        }
