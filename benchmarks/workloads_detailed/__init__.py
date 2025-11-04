"""
Workload implementations for OSDI evaluation.

This module implements the active workload configurations used in the
4 main OSDI benchmarks.
"""

from .realistic_llm_decode import RealisticLLMDecodeWorkload
from .realistic_llm_prefill import RealisticLLMPrefillWorkload

__all__ = [
    'RealisticLLMDecodeWorkload',
    'RealisticLLMPrefillWorkload',
]
