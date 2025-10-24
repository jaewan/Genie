"""
Shared types and enums for the Genie framework.

This module contains types that are used across multiple modules to avoid circular imports.
"""

from enum import Enum


class ExecutionPhase(Enum):
    """Execution phases for workload-specific optimizations."""
    # LLM phases
    PREFILL = "prefill"  # LLM initial token processing
    LLM_PREFILL = "llm_prefill"  # LLM initial token processing (alias)
    DECODE = "decode"    # LLM autoregressive generation
    LLM_DECODE = "llm_decode"    # LLM autoregressive generation (alias)

    # General phases
    FORWARD = "forward"  # Forward propagation
    BACKWARD = "backward"  # Backward propagation
    INIT = "initialization"  # Initialization

    # Vision phases
    VISION_BACKBONE = "vision_backbone"  # Vision model feature extraction
    VISION_HEAD = "vision_head"  # Vision model classification/detection

    # Multi-modal phases
    MULTIMODAL_FUSION = "multimodal_fusion"  # Cross-modal attention

    # Other phases
    EMBEDDING = "embedding"  # Embedding lookup
    UNKNOWN = "unknown"


class MatchingMode(Enum):
    """Pattern matching modes."""
    EXHAUSTIVE = "exhaustive"  # Match all patterns (default, safe)
    FAST = "fast"              # Early termination (opt-in)
    REQUIRED_ONLY = "required" # Only match explicitly required patterns


class MemoryPattern(Enum):
    """Memory access patterns for optimization."""
    STREAMING = "streaming"  # One-time use, no reuse
    REUSED = "reused"       # Multiple accesses, cache-friendly
    EPHEMERAL = "ephemeral" # Short-lived intermediate
    PERSISTENT = "persistent"  # Long-lived (e.g., KV cache)
    RANDOM = "random"       # Random access pattern


class TransportType(Enum):
    """Transport types for data movement."""
    TCP = "tcp"
    DPDK = "dpdk"
    RDMA = "rdma"


class Modality(Enum):
    """Data modalities."""
    VISION = "vision"
    TEXT = "text"
    AUDIO = "audio"
    FUSION = "fusion"
    UNKNOWN = "unknown"
