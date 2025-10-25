"""
Detailed workload implementations for comprehensive OSDI evaluation.

Each workload demonstrates specific semantic optimizations and includes
real model implementations from HuggingFace.

Includes both microbenchmark workloads (for fast iteration) and realistic
production-scale workloads (for SOSP submission).
"""

from .llm_decode import LLMDecodeWorkload
from .llm_prefill import LLMPrefillWorkload
from .vision_cnn import VisionCNNWorkload
from .multimodal_vqa import MultimodalVQAWorkload
from .microbenchmark import MicrobenchmarkWorkload

# New realistic production-scale workloads (Phase 1: Week 1)
from .realistic_llm_decode import RealisticLLMDecodeWorkload
from .realistic_llm_prefill import RealisticLLMPrefillWorkload
from .realistic_vision_cnn import RealisticVisionCNNWorkload

__all__ = [
    # Microbenchmarks (legacy)
    'LLMDecodeWorkload',
    'LLMPrefillWorkload',
    'VisionCNNWorkload',
    'MultimodalVQAWorkload',
    'MicrobenchmarkWorkload',
    
    # Production-scale realistic workloads (new)
    'RealisticLLMDecodeWorkload',
    'RealisticLLMPrefillWorkload',
    'RealisticVisionCNNWorkload',
]
