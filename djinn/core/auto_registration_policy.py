"""
Smart Auto-Registration Policy for Model Caching.

Provides intelligent, memory-safe automatic model registration based on usage patterns,
model size, and memory availability.
"""

import time
import logging
from typing import Dict, Optional, Set
from dataclasses import dataclass, field
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelUsageTracker:
    """Tracks model usage for smart auto-registration."""
    
    def __init__(self):
        self.usage_counts: Dict[str, int] = {}  # fingerprint -> usage count
        self.last_used: Dict[str, float] = {}  # fingerprint -> timestamp
        self.first_used: Dict[str, float] = {}  # fingerprint -> first use timestamp
    
    def record(self, fingerprint: str) -> int:
        """Record model usage and return current count."""
        current_time = time.time()
        
        if fingerprint not in self.usage_counts:
            self.first_used[fingerprint] = current_time
        
        self.usage_counts[fingerprint] = self.usage_counts.get(fingerprint, 0) + 1
        self.last_used[fingerprint] = current_time
        
        return self.usage_counts[fingerprint]
    
    def get_count(self, fingerprint: str) -> int:
        """Get usage count for a model."""
        return self.usage_counts.get(fingerprint, 0)
    
    def get_age_seconds(self, fingerprint: str) -> float:
        """Get age of model (time since first use)."""
        if fingerprint not in self.first_used:
            return 0.0
        return time.time() - self.first_used[fingerprint]
    
    def reset(self, fingerprint: str):
        """Reset tracking for a model (e.g., after explicit registration)."""
        if fingerprint in self.usage_counts:
            del self.usage_counts[fingerprint]
        if fingerprint in self.last_used:
            del self.last_used[fingerprint]
        if fingerprint in self.first_used:
            del self.first_used[fingerprint]


@dataclass
class AutoRegistrationPolicy:
    """
    Policy for automatic model registration.
    
    Smart auto-registration only registers models that meet ALL criteria:
    - Usage threshold: Used at least N times
    - Size limit: Model size under threshold (if set)
    - Memory check: Sufficient GPU memory available (if enabled)
    - Not explicitly opted out
    """
    
    # Usage-based: Auto-register after N uses
    usage_threshold: int = 3  # Register after 3 uses
    
    # Size-based: Only auto-register models under size limit (MB)
    # None = no size limit
    max_size_mb: Optional[int] = 500  # Only auto-register <500MB models
    
    # Memory-aware: Check memory availability before auto-registering
    require_memory_check: bool = True
    memory_safety_margin: float = 1.5  # Require 1.5x model size as free memory
    
    # Explicit opt-out: Models that should never be auto-registered
    explicit_opt_out: Set[str] = field(default_factory=set)
    
    # Enable/disable auto-registration entirely
    enabled: bool = True
    
    def should_auto_register(
        self, 
        fingerprint: str, 
        model: nn.Module,
        usage_count: int,
        usage_tracker: Optional[ModelUsageTracker] = None
    ) -> bool:
        """
        Determine if model should be auto-registered.
        
        Args:
            fingerprint: Model fingerprint
            model: PyTorch model
            usage_count: Current usage count
            usage_tracker: Optional usage tracker for additional checks
        
        Returns:
            True if model should be auto-registered
        """
        # Check if auto-registration is enabled
        if not self.enabled:
            return False
        
        # Check explicit opt-out
        if fingerprint in self.explicit_opt_out:
            logger.debug(f"Model {fingerprint[:16]}... explicitly opted out of auto-registration")
            return False
        
        # Check usage threshold
        if usage_count < self.usage_threshold:
            logger.debug(
                f"Model {fingerprint[:16]}... usage count {usage_count} < threshold {self.usage_threshold}"
            )
            return False
        
        # Check size limit
        if self.max_size_mb is not None:
            model_size_mb = self._estimate_model_size(model)
            if model_size_mb > self.max_size_mb:
                logger.debug(
                    f"Model {fingerprint[:16]}... size {model_size_mb:.1f}MB > limit {self.max_size_mb}MB"
                )
                return False
        
        # Check memory availability
        if self.require_memory_check:
            if not self._check_memory_availability(model):
                logger.debug(f"Model {fingerprint[:16]}... insufficient GPU memory available")
                return False
        
        # All checks passed
        logger.info(
            f"✅ Auto-registering model {fingerprint[:16]}... "
            f"(usage={usage_count}, threshold={self.usage_threshold})"
        )
        return True
    
    def _estimate_model_size(self, model: nn.Module) -> float:
        """Estimate model size in MB."""
        total_params = sum(p.numel() for p in model.parameters())
        # Assume float32 (4 bytes per parameter)
        size_bytes = total_params * 4
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    
    def _check_memory_availability(self, model: nn.Module) -> bool:
        """Check if sufficient GPU memory is available."""
        if not torch.cuda.is_available():
            return True  # CPU mode, assume memory available
        
        try:
            # Estimate model size
            model_size_mb = self._estimate_model_size(model)
            required_mb = model_size_mb * self.memory_safety_margin
            
            # Get free GPU memory
            free_memory_bytes, _ = torch.cuda.mem_get_info(0)
            free_mb = free_memory_bytes / (1024 * 1024)
            
            if free_mb < required_mb:
                logger.debug(
                    f"Insufficient memory: need {required_mb:.1f}MB, have {free_mb:.1f}MB free"
                )
                return False
            
            return True
        except Exception as e:
            logger.warning(f"Memory check failed: {e}, allowing auto-registration")
            return True  # On error, allow (conservative)


# Default policy configurations
DEFAULT_POLICY = AutoRegistrationPolicy(
    usage_threshold=3,
    max_size_mb=500,
    require_memory_check=True,
    enabled=True
)

PRODUCTION_POLICY = AutoRegistrationPolicy(
    usage_threshold=3,
    max_size_mb=500,
    require_memory_check=True,
    enabled=True
)

DEVELOPMENT_POLICY = AutoRegistrationPolicy(
    usage_threshold=2,
    max_size_mb=1000,
    require_memory_check=True,
    enabled=True
)

DISABLED_POLICY = AutoRegistrationPolicy(
    enabled=False
)

