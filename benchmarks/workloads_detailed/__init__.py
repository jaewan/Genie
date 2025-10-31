"""
Production-scale workload implementations for OSDI evaluation.

Each workload demonstrates specific semantic optimizations using
real model implementations from HuggingFace.

All workloads use real models (GPT-2-XL, ResNet-50) for publication-quality results.
"""

# Production-scale realistic workloads with real models
from .realistic_llm_decode import RealisticLLMDecodeWorkload
from .realistic_llm_prefill import RealisticLLMPrefillWorkload
from .realistic_vision_cnn import RealisticVisionCNNWorkload

# Multimodal and microbenchmark workloads
from .multimodal_vqa import MultimodalVQAWorkload
from .microbenchmark import MicrobenchmarkWorkload

__all__ = [
    # Real model workloads (primary)
    'RealisticLLMDecodeWorkload',
    'RealisticLLMPrefillWorkload',
    'RealisticVisionCNNWorkload',
    
    # Supporting workloads
    'MultimodalVQAWorkload',
    'MicrobenchmarkWorkload',
]
