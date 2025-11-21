"""
Model Cache for v2.3 - Server-side model storage in VMU.

Stores registered models in GPU memory (VMU slab) for fast execution.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Optional
import threading

logger = logging.getLogger(__name__)


class ModelCacheV23:
    """
    Server-side model cache using VMU for storage.
    
    Models are loaded into GPU memory once and reused across requests.
    """
    
    def __init__(self, vmu=None):
        """
        Initialize model cache.
        
        Args:
            vmu: VMU instance for memory management
        """
        from ..backend.runtime.unified_vmu import get_vmu
        self.vmu = vmu or get_vmu()
        
        # Cache: fingerprint -> model
        self.models: Dict[str, nn.Module] = {}
        self.lock = threading.Lock()
        
        logger.info("✅ ModelCacheV23 initialized")
    
    def register_model(self, fingerprint: str, model: nn.Module, model_id: Optional[str] = None):
        """
        Register model in cache with VMU-backed memory.
        
        Args:
            fingerprint: Model fingerprint
            model: PyTorch model
            model_id: Optional model identifier
        """
        with self.lock:
            if fingerprint in self.models:
                logger.debug(f"Model {fingerprint[:8]} already in cache")
                return
            
            # Move model to GPU if not already there
            if torch.cuda.is_available():
                model_device = next(model.parameters()).device if list(model.parameters()) else None
                if model_device is None or model_device.type != 'cuda':
                    logger.info(f"Moving model {fingerprint[:8]} to GPU with VMU backing...")
                    
                    # First move to GPU using default allocator
                    model = model.to(self.vmu.device)
                    logger.info(f"✅ Model on GPU: {next(model.parameters()).device}")
                    
                    # Now move parameters to VMU slab
                    from djinn.backend.runtime.vmu_allocator import move_model_to_vmu
                    vmu_stats = move_model_to_vmu(model, self.vmu, verbose=False)
                    
                    logger.info(
                        f"✅ Model {fingerprint[:8]} backed by VMU: "
                        f"{vmu_stats['total_bytes'] / 1024**2:.2f}MB in slab"
                    )
            
            # Set to eval mode
            model.eval()
            
            # Store in cache
            self.models[fingerprint] = model
            
            logger.info(f"✅ Model {fingerprint[:8]} cached on GPU with VMU backing")
    
    def get_model(self, fingerprint: str) -> Optional[nn.Module]:
        """
        Get model from cache.
        
        Args:
            fingerprint: Model fingerprint
            
        Returns:
            Model if found, None otherwise
        """
        with self.lock:
            return self.models.get(fingerprint)
    
    def clear(self):
        """Clear all cached models."""
        with self.lock:
            self.models.clear()
            logger.info("Model cache cleared")


# Global instance
_global_cache: Optional[ModelCacheV23] = None


def get_model_cache() -> ModelCacheV23:
    """Get or create global model cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ModelCacheV23()
    return _global_cache

