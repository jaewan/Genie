"""
Model Security Validator: Security validation for model registration and execution.

Prevents:
- Fingerprint collisions
- Dangerous code execution
- Resource exhaustion attacks
- Malicious model architectures

This is part of the redesign plan (Week 1).
"""

import hashlib
import logging
from typing import Dict, Set

import torch

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Security validation failed."""
    pass


class ModelSecurityValidator:
    """
    Security validation for model registration and execution.
    
    Prevents:
    - Fingerprint collisions (security issue)
    - Dangerous code execution
    - Resource exhaustion attacks
    - Malicious model architectures
    """
    
    def __init__(self):
        # Whitelist of allowed modules
        self.allowed_modules: Set[str] = {
            'torch.nn',
            'torch.nn.functional',
            'transformers.models',
            'torchvision.models',
        }
        
        # Blacklist of dangerous operations
        self.blacklisted_ops: Set[str] = {
            'exec', 'eval', '__import__',
            'open', 'file', 'compile',
            'os.system', 'subprocess',
            'pickle.loads', 'marshal.loads'
        }
        
        # Fingerprint collision detection
        self.registered_fingerprints: Dict[str, str] = {}  # fp -> structure_hash
        
        # Size limits (configurable)
        self.MAX_PARAM_SIZE = 10 * 1024**3  # 10GB max per parameter
        self.MAX_TOTAL_SIZE = 100 * 1024**3  # 100GB max total
        self.MAX_INPUT_SIZE = 1 * 1024**3  # 1GB max input
    
    def validate_model_registration(self, 
                                   fingerprint: str,
                                   architecture_data: bytes,
                                   state_dict: Dict) -> bool:
        """
        Validate model registration for security.
        
        Args:
            fingerprint: Model fingerprint
            architecture_data: Serialized architecture bytes
            state_dict: Model state dict (parameter tensors)
        
        Returns:
            True if validation passes
        
        Raises:
            SecurityError: If validation fails
        """
        
        # 1. Check fingerprint collision
        self._check_fingerprint_collision(fingerprint, state_dict)
        
        # 2. Validate architecture data
        self._validate_architecture(architecture_data)
        
        # 3. Validate state dict
        self._validate_state_dict(state_dict)
        
        logger.debug(f"Security validation passed for {fingerprint}")
        return True
    
    def _check_fingerprint_collision(self, fingerprint: str, state_dict: Dict):
        """
        Check for fingerprint collisions (security issue).
        
        A fingerprint collision occurs when two different models have the same
        fingerprint. This could be exploited to load malicious models.
        """
        
        # Compute hash of state dict structure
        structure = sorted(state_dict.keys())
        structure_hash = hashlib.sha256(str(structure).encode()).hexdigest()
        
        if fingerprint in self.registered_fingerprints:
            existing_hash = self.registered_fingerprints[fingerprint]
            if existing_hash != structure_hash:
                raise SecurityError(
                    f"Fingerprint collision detected for {fingerprint}. "
                    "Different model structure with same fingerprint. "
                    "This could be a security issue."
                )
        else:
            self.registered_fingerprints[fingerprint] = structure_hash
            logger.debug(f"Registered fingerprint: {fingerprint}")
    
    def _validate_architecture(self, architecture_data: bytes):
        """
        Validate architecture data for dangerous code.
        
        Checks for blacklisted operations in serialized architecture data.
        """
        
        if not architecture_data:
            return
        
        # Check for dangerous strings in pickled data
        # Note: This is a simple heuristic - in production, you might want
        # more sophisticated analysis
        data_str = str(architecture_data)
        for dangerous in self.blacklisted_ops:
            if dangerous in data_str:
                raise SecurityError(
                    f"Architecture contains potentially dangerous operation: {dangerous}"
                )
    
    def _validate_state_dict(self, state_dict: Dict):
        """
        Validate state dict structure and sizes.
        
        Ensures:
        - All values are tensors
        - Parameter sizes are within limits
        - Total model size is within limits
        """
        
        total_size = 0
        
        for name, param in state_dict.items():
            if not isinstance(param, torch.Tensor):
                raise SecurityError(
                    f"State dict contains non-tensor: {name} ({type(param)})"
                )
            
            param_size = param.numel() * param.element_size()
            
            if param_size > self.MAX_PARAM_SIZE:
                raise SecurityError(
                    f"Parameter {name} too large: {param_size/1024**3:.1f}GB "
                    f"(max: {self.MAX_PARAM_SIZE/1024**3:.1f}GB)"
                )
            
            total_size += param_size
        
        if total_size > self.MAX_TOTAL_SIZE:
            raise SecurityError(
                f"Model too large: {total_size/1024**3:.1f}GB "
                f"(max: {self.MAX_TOTAL_SIZE/1024**3:.1f}GB)"
            )
        
        logger.debug(f"State dict validation passed: {len(state_dict)} parameters, {total_size/1024**3:.2f}GB")
    
    def validate_inputs(self, inputs: Dict) -> bool:
        """
        Validate execution inputs.
        
        Prevents resource exhaustion attacks via oversized inputs.
        
        Args:
            inputs: Input dictionary for model execution
        
        Returns:
            True if validation passes
        
        Raises:
            SecurityError: If inputs are too large
        """
        
        total_size = 0
        for name, value in inputs.items():
            if isinstance(value, torch.Tensor):
                total_size += value.numel() * value.element_size()
            elif isinstance(value, (list, dict)):
                # Rough estimate for non-tensor inputs
                total_size += len(str(value))
        
        if total_size > self.MAX_INPUT_SIZE:
            raise SecurityError(
                f"Input too large: {total_size/1024**3:.1f}GB "
                f"(max: {self.MAX_INPUT_SIZE/1024**3:.1f}GB)"
            )
        
        return True
    
    def clear_registry(self):
        """Clear fingerprint registry (for testing)."""
        self.registered_fingerprints.clear()
        logger.debug("Security validator registry cleared")
    
    def get_stats(self) -> Dict:
        """Get security validator statistics."""
        return {
            'registered_fingerprints': len(self.registered_fingerprints),
            'max_param_size_gb': self.MAX_PARAM_SIZE / 1024**3,
            'max_total_size_gb': self.MAX_TOTAL_SIZE / 1024**3,
            'max_input_size_gb': self.MAX_INPUT_SIZE / 1024**3
        }

