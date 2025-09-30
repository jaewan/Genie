"""
Pattern recognition module for Genie.

This module provides semantic pattern recognition for identifying
common workload patterns (LLM, Vision, RecSys, MultiModal) in computation graphs.
"""

from .base import PatternPlugin, PatternMatch  # noqa: F401
from .advanced_patterns import (  # noqa: F401
    AdvancedLLMPattern,
    AdvancedVisionPattern,
    RecSysPattern,
    MultiModalPattern,
    ResidualBlockPattern,
)
from .fx_patterns import (  # noqa: F401
    find_matmul_chain,
    find_attention_pattern_fx,
    find_conv_activation_pattern_fx,
    find_mlp_pattern_fx,
)

__all__ = [
    "PatternPlugin",
    "PatternMatch",
    "AdvancedLLMPattern",
    "AdvancedVisionPattern",
    "RecSysPattern",
    "MultiModalPattern",
    "ResidualBlockPattern",
    "find_matmul_chain",
    "find_attention_pattern_fx",
    "find_conv_activation_pattern_fx",
    "find_mlp_pattern_fx",
]