"""
Workload implementations for OSDI evaluation.

This module implements the active workload configurations used in the
4 main OSDI benchmarks.
"""

from .llm_decode import RealisticLLMDecodeWorkload
from .llm_prefill import RealisticLLMPrefillWorkload

__all__ = [
    'RealisticLLMDecodeWorkload',
    'RealisticLLMPrefillWorkload',
]
