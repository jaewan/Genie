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
        batch_size: int = 8,  # Reduced from 32 for GPU memory
        max_length: int = 512,  # Reduced from 2048 for GPU memory
        use_real_model: bool = True   # ALWAYS use real models
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = None  # Will be set during load_model
        self.load_model()

    def load_model(self) -> bool:
        """Load BERT model from HuggingFace - ONLY REAL MODELS."""
        try:
            from transformers import AutoModel, AutoTokenizer

            print(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load with explicit device placement on GPU
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=None  # No auto-mapping - explicit control
            )
            self.model.eval()

            # CRITICAL: Ensure model is on GPU (cuda:0 by default)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available - GPU required for benchmarking")
            
            self.device = torch.device('cuda:0')
            self.model.to(self.device)

            print(f"✓ Loaded {self.model_name} on {self.device}")
            return True

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

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
        input_ids = batch_encodings['input_ids'].to(self.device)
        
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
            'workload': 'llm_prefill',
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
            'workload': 'llm_prefill',
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
