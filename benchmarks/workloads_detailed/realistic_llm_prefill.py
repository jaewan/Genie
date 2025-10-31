"""
Realistic LLM Prefill Workload - Production Scale Batch Processing.

This workload tests semantic optimization for parallel batch processing.

Configuration:
- Model: BERT-large-uncased (336M parameters)
- Task: Batch document embedding/classification
- Batch size: 32 documents
- Document length: 2048 tokens each
- Expected runtime: 3-5 seconds (not 70ms)
- Semantic benefit: 1.5-2x from parallel batch processing
- Use case: Semantic search, document classification

Key insight:
- Prefill phase is compute-bound and parallelizable
- Semantic awareness enables better resource utilization
- Batch processing amortizes overhead
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
import time


class RealisticLLMPrefillWorkload:
    """Production-scale LLM prefill workload for batch document processing."""

    def __init__(
        self,
        model_name: str = "bert-large-uncased",
        batch_size: int = 32,
        max_length: int = 2048,  # Long documents
        use_real_model: bool = False
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_real_model = use_real_model
        # Load model and tokenizer immediately
        self.load_model()

    def load_model(self) -> bool:
        """Load BERT model from HuggingFace."""
        if self.use_real_model:
            try:
                from transformers import AutoModel, AutoTokenizer

                print(f"Loading {self.model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                # CRITICAL FIX: Use explicit device placement, not device_map="auto"
                # device_map="auto" can place on cuda:1 or other devices
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map=None              # No auto-mapping - explicit control
                )
                self.model.eval()

                # CRITICAL FIX: Ensure model is on cuda:0
                if torch.cuda.is_available():
                    device = torch.device('cuda:0')
                    self.model.to(device)

                print(f"✓ Loaded {self.model_name}")
                return True

            except ImportError as e:
                print(f"❌ transformers not available: {e}")
                print("Falling back to mock model...")

        # Fallback to mock model (keep on CPU for validation)
        self.model = self._create_mock_model()
        self.tokenizer = self._create_mock_tokenizer()
        
        # Keep mock model on CPU to avoid GPU memory issues during validation
        # Real benchmarks will use real models on GPU
        
        return False

    def _create_mock_model(self) -> nn.Module:
        """Create a realistic mock BERT model."""
        class MockBERT(nn.Module):
            def __init__(self, hidden_size: int = 1024, num_layers: int = 24):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                # BERT-large architecture
                self.embeddings = nn.Embedding(30522, hidden_size)
                self.token_embeddings = nn.Embedding(30522, hidden_size)
                self.position_embeddings = nn.Embedding(2048, hidden_size)
                self.token_type_embeddings = nn.Embedding(2, hidden_size)
                
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=16,  # BERT-large uses 16 heads
                        dim_feedforward=hidden_size * 4,
                        batch_first=True,
                        dropout=0.0
                    ),
                    num_layers=num_layers
                )
                
                self.ln = nn.LayerNorm(hidden_size)
                
                # Classification head (for downstream task)
                self.classifier = nn.Linear(hidden_size, 768)

            def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict:
                """Forward pass."""
                batch_size, seq_len = input_ids.shape
                
                # Embeddings
                token_embeds = self.token_embeddings(input_ids)
                
                # Position embeddings
                pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
                pos_embeds = self.position_embeddings(pos_ids)
                
                # Token type embeddings (all zeros for single sequence)
                token_type_ids = torch.zeros_like(input_ids)
                token_type_embeds = self.token_type_embeddings(token_type_ids)
                
                # Combine embeddings
                hidden_states = token_embeds + pos_embeds + token_type_embeds
                hidden_states = self.ln(hidden_states)
                
                # Transformer
                if attention_mask is not None:
                    # Convert attention mask to transformer format
                    attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
                    attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0
                
                hidden_states = self.transformer(hidden_states, src_key_padding_mask=None)
                
                # Use [CLS] token representation (first token)
                cls_output = hidden_states[:, 0, :]

                # For BERT, return the pooler_output (CLS token representation)
                # This is what most downstream tasks use
                return cls_output

        return MockBERT()

    def _create_mock_tokenizer(self):
        """Create a mock BERT tokenizer."""
        class MockBERTTokenizer:
            def __init__(self):
                self.vocab_size = 30522
                self.cls_token = "[CLS]"
                self.sep_token = "[SEP]"
                self.pad_token = "[PAD]"

            def encode(self, text: str, return_tensors: str = 'pt', **kwargs) -> torch.Tensor:
                # Simulate token encoding
                tokens = [101] + [103] * min(len(text.split()), 100) + [102]
                return torch.tensor([tokens])

            def __call__(
                self,
                texts: List[str],
                return_tensors: str = 'pt',
                padding: bool = True,
                truncation: bool = False,
                max_length: Optional[int] = None,
                **kwargs
            ):
                # CRITICAL FIX: Use the provided max_length, don't default to 512
                max_len = max_length if max_length is not None else 512
                batch = []
                attention_masks = []
                
                for text in texts:
                    # [CLS] + tokens + [SEP]
                    tokens = [101] + [103] * min(len(text.split()), max_len - 2) + [102]
                    # Truncate to max_len if needed
                    tokens = tokens[:max_len]
                    batch.append(tokens)
                    # Create attention mask (1 for real tokens, 0 for padding)
                    attention_masks.append([1] * len(tokens))
                
                # Pad to same length
                if padding and batch:
                    max_batch_len = max(len(b) for b in batch)
                    batch = [b + [0] * (max_batch_len - len(b)) for b in batch]
                    attention_masks = [
                        m + [0] * (max_batch_len - len(m))
                        for m in attention_masks
                    ]

                # Return a simple dictionary that supports subscript access
                return {
                    'input_ids': torch.tensor(batch),
                    'attention_mask': torch.tensor(attention_masks),
                    'token_type_ids': torch.zeros_like(torch.tensor(batch))
                }

        return MockBERTTokenizer()

    def _get_sample_documents(self) -> List[str]:
        """Get sample documents for processing."""
        # Create synthetic long documents
        documents = []
        document_templates = [
            "Machine learning is a subset of artificial intelligence that focuses on the ability of computers to learn from data.",
            "Deep learning uses artificial neural networks with multiple layers (hence 'deep') to process inputs and generate outputs.",
            "Natural language processing aims to enable computers to understand, interpret, and generate human language.",
            "Computer vision enables machines to interpret and make decisions based on visual input from images and videos.",
            "Reinforcement learning is a type of machine learning where agents learn to make decisions through trial and error.",
            "Transfer learning leverages pre-trained models to solve new problems with limited training data.",
            "Generative models create new data samples that resemble the training data distribution.",
            "Convolutional neural networks are particularly effective for image processing and computer vision tasks.",
            "Recurrent neural networks are designed to process sequential data and capture temporal dependencies.",
            "Transformers have revolutionized natural language processing and achieved state-of-the-art results.",
        ]
        
        # Create batch_size documents by repeating/extending templates
        for i in range(self.batch_size):
            # Repeat template to create long document (simulating realistic length)
            template = document_templates[i % len(document_templates)]
            doc = " ".join([template] * (self.max_length // len(template.split()) + 1))
            documents.append(doc[:self.max_length * 10])  # Rough character limit
        
        return documents

    def get_sample_inputs(self) -> List[torch.Tensor]:
        """Get batch of sample documents - returns just the input_ids tensor."""
        if self.tokenizer is None:
            self.load_model()

        documents = self._get_sample_documents()

        # Take first batch_size documents
        documents = documents[:self.batch_size]

        batch_encodings = self.tokenizer(
            documents,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        # CRITICAL FIX: Ensure input is on the correct device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        input_ids = batch_encodings['input_ids'].to(device)
        
        # Return input_ids as a tensor (dict['input_ids'] extracts the tensor)
        if isinstance(batch_encodings, dict):
            return [input_ids]
        # If tokenizer returns a proper object with input_ids attribute
        return [input_ids]

    def run_reference(self) -> torch.Tensor:
        """Reference implementation - GROUND TRUTH."""
        if self.model is None:
            self.load_model()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        documents = self._get_sample_documents()

        batch_encodings = self.tokenizer(
            documents,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(device)

        with torch.no_grad():
            outputs = self.model(
                batch_encodings['input_ids'],
                attention_mask=batch_encodings.get('attention_mask')
            )

        return outputs.pooler_output.cpu()

    def run(self, baseline_config: Dict) -> Dict[str, Any]:
        """
        Run batch document processing workload.
        
        Measures:
        - Total latency for batch
        - Throughput (documents per second)
        - Peak memory usage
        """
        if self.model is None:
            self.load_model()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        documents = self._get_sample_documents()

        batch_encodings = self.tokenizer(
            documents,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(device)

        # Warm-up
        with torch.no_grad():
            _ = self.model(
                batch_encodings['input_ids'],
                attention_mask=batch_encodings.get('attention_mask')
            )

        # Measurement
        torch.cuda.synchronize(device) if torch.cuda.is_available() else None
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(
                batch_encodings['input_ids'],
                attention_mask=batch_encodings.get('attention_mask')
            )

        torch.cuda.synchronize(device) if torch.cuda.is_available() else None
        end_time = time.perf_counter()

        latency_sec = end_time - start_time
        throughput_docs_per_sec = self.batch_size / max(latency_sec, 0.001)

        return {
            'workload': 'realistic_llm_prefill',
            'latency_sec': latency_sec,
            'latency_ms': latency_sec * 1000,
            'throughput_docs_per_sec': throughput_docs_per_sec,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'total_tokens_processed': self.batch_size * self.max_length,
            'baseline': baseline_config.get('baseline', 'unknown'),
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get workload metadata."""
        return {
            'workload': 'realistic_llm_prefill',
            'model': self.model_name,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'expected_runtime_sec': 4.0,
            'description': 'Batch document processing with BERT-large',
            'use_case': 'semantic search, document classification',
            'key_optimization': 'parallel_batch_processing',
            'expected_speedup_vs_no_semantics': '1.5-2x',
            'total_tokens_processed': self.batch_size * self.max_length,
        }
