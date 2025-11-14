"""
Model Tracker: Tracks models for automatic model cache integration.

When models are moved to remote_accelerator:0, we track them so we can
use model cache execution instead of graph execution.
"""

import threading
import logging
from typing import Dict, Optional, Set
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelTracker:
    """
    Tracks models that have been moved to remote device.
    
    This enables automatic detection of model forward passes so we can
    use the model cache system instead of graph execution.
    """
    
    _instance: Optional['ModelTracker'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance
    
    def _init(self):
        """Initialize tracker state."""
        # Map model object id → fingerprint
        self._model_fingerprints: Dict[int, str] = {}
        
        # Map parameter tensor id → model fingerprint
        self._param_to_model: Dict[int, str] = {}
        
        # Track registered models (fingerprint → True)
        self._registered_models: Set[str] = set()
        
        self._tracker_lock = threading.RLock()
    
    def track_model(self, model: nn.Module, fingerprint: str):
        """
        Track a model that's been moved to remote device.
        
        Args:
            model: PyTorch model
            fingerprint: Model fingerprint from ModelFingerprint.compute()
        """
        with self._tracker_lock:
            model_id = id(model)
            self._model_fingerprints[model_id] = fingerprint
            
            # Track all parameters
            for param in model.parameters():
                param_id = id(param)
                self._param_to_model[param_id] = fingerprint
                # Also track param.data
                self._param_to_model[id(param.data)] = fingerprint
            
            logger.debug(f"Tracked model {fingerprint} (id={model_id})")
    
    def get_model_fingerprint(self, tensor: torch.Tensor) -> Optional[str]:
        """
        Check if a tensor belongs to a tracked model.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Model fingerprint if tensor belongs to a tracked model, None otherwise
        """
        with self._tracker_lock:
            tensor_id = id(tensor)
            return self._param_to_model.get(tensor_id)
    
    def mark_registered(self, fingerprint: str):
        """Mark a model as registered with the server."""
        with self._tracker_lock:
            self._registered_models.add(fingerprint)
            logger.debug(f"Marked model {fingerprint} as registered")
    
    def is_registered(self, fingerprint: str) -> bool:
        """Check if a model is registered."""
        with self._tracker_lock:
            return fingerprint in self._registered_models
    
    def clear(self):
        """Clear all tracked models."""
        with self._tracker_lock:
            self._model_fingerprints.clear()
            self._param_to_model.clear()
            self._registered_models.clear()
            logger.info("Model tracker cleared")


# Global singleton accessor
_tracker: Optional[ModelTracker] = None
_tracker_lock = threading.Lock()


def get_model_tracker() -> ModelTracker:
    """Get the global model tracker."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = ModelTracker()
    return _tracker

