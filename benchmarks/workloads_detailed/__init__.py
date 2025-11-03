"""
Production-scale workload implementations for OSDI evaluation.

Each workload demonstrates specific semantic optimizations using
real model implementations from HuggingFace.

All workloads use real models for publication-quality results.

Workload Coverage:
- Language: GPT-2-XL (LLM Decode, LLM Prefill)
- Vision: ResNet-152 (Vision CNN)
- Multimodal: CLIP (VQA)
- Speech: Whisper-small (Speech Recognition) - NEW
- Recommendation: DLRM (CTR Prediction) - NEW

This diversity proves framework-level abstraction works broadly across
fundamentally different ML workload patterns.
"""

# Production-scale realistic workloads with real models
from .realistic_llm_decode import RealisticLLMDecodeWorkload
from .realistic_llm_prefill import RealisticLLMPrefillWorkload
from .realistic_vision_cnn import RealisticVisionCNNWorkload

# Multimodal and microbenchmark workloads
from .multimodal_vqa import MultimodalVQAWorkload
from .microbenchmark import MicrobenchmarkWorkload

# NEW: Speech and Recommendation workloads for broader coverage
from .speech_recognition import RealisticSpeechRecognitionWorkload
from .recommendation_dlrm import RealisticRecommendationWorkload

__all__ = [
    # Real model workloads (primary)
    'RealisticLLMDecodeWorkload',
    'RealisticLLMPrefillWorkload',
    'RealisticVisionCNNWorkload',
    
    # Supporting workloads
    'MultimodalVQAWorkload',
    'MicrobenchmarkWorkload',
    
    # NEW: Broader framework coverage
    'RealisticSpeechRecognitionWorkload',
    'RealisticRecommendationWorkload',
]
