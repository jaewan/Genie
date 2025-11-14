"""
Memory-Aware Model Cache: Production-grade model caching with intelligent memory management.

Key Features:
- Size-aware eviction (not count-based)
- Value-based eviction (access frequency / age / size)
- OOM protection with automatic recovery
- Integration with Phase 1 weight cache
- Integration with Phase 2 FastSerializer

This is part of the redesign plan (Week 2).
"""

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ModelMemoryProfile:
    """Memory profile for cached model."""
    fingerprint: str
    param_bytes: int
    activation_bytes: int  # Estimated peak activation memory
    total_bytes: int
    last_access_time: float
    access_count: int
    execution_time_ms: float = 0.0  # Track execution time for value calculation


class MemoryAwareModelCache:
    """
    Production-grade model cache with intelligent memory management.
    
    Key Features:
    - Size-aware eviction (not count-based)
    - Value-based eviction (access frequency / age / size)
    - OOM protection with automatic recovery
    - Integration with Phase 1 weight cache
    """
    
    def __init__(self, 
                 max_memory_gb: Optional[float] = None,
                 target_utilization: float = 0.8,
                 device: str = 'cuda:0'):
        
        self.device = torch.device(device)
        
        # Auto-detect available memory if not specified
        if max_memory_gb is None:
            if torch.cuda.is_available():
                gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
                max_memory_gb = (gpu_memory_bytes / 1024**3) * target_utilization
            else:
                max_memory_gb = 8.0  # Default 8GB for CPU
        
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        
        # Model storage
        self.models: OrderedDict[str, nn.Module] = OrderedDict()
        self.profiles: Dict[str, ModelMemoryProfile] = {}
        
        # Architecture registry
        from .architecture_registry import HybridArchitectureRegistry
        self.architecture_registry = HybridArchitectureRegistry()
        
        # Reuse Phase 1 weight cache
        from .gpu_cache import get_global_cache
        self.weight_cache = get_global_cache()
        
        # Memory tracking
        self.current_memory_bytes = 0
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'evictions': 0,
            'oom_events': 0,
            'total_executions': 0,
            'total_latency_ms': 0.0
        }
        
        logger.info(
            f"MemoryAwareModelCache initialized: max_memory={max_memory_gb:.1f}GB, "
            f"device={device}"
        )
    
    def register_model(self, 
                       fingerprint: str,
                       descriptor: Dict,
                       weight_ids: Dict[str, str],
                       uncached_weights: Dict[str, torch.Tensor],
                       architecture_data: Optional[bytes] = None):
        """
        Register model with memory-aware eviction.
        
        Integrates with Phase 1 weight cache for efficient weight handling.
        
        Args:
            fingerprint: Model fingerprint
            descriptor: Architecture descriptor dict
            weight_ids: Dict mapping parameter names to weight identifiers
            uncached_weights: Dict of uncached weight tensors
            architecture_data: Optional serialized architecture bytes
        
        Raises:
            ValueError: If architecture cannot be reconstructed
            MemoryError: If model is too large for available memory
        """
        
        if fingerprint in self.models:
            logger.info(f"Model {fingerprint} already registered")
            return
        
        logger.info(f"Registering model {fingerprint}")
        
        # Load architecture
        try:
            model = self.architecture_registry.load_architecture(
                fingerprint, architecture_data, descriptor
            )
        except ValueError as e:
            # Cannot reconstruct - will use graph execution fallback
            raise ValueError(f"Cannot reconstruct model architecture: {e}")
        
        # Calculate memory requirement
        param_bytes = sum(
            p.numel() * p.element_size() 
            for p in uncached_weights.values()
            if isinstance(p, torch.Tensor)
        )
        
        # Add cached weights size (from Phase 1 cache)
        for weight_id in weight_ids.values():
            if weight_id in self.weight_cache.cache_new:
                cached_tensor = self.weight_cache.cache_new[weight_id]
                param_bytes += cached_tensor.numel() * cached_tensor.element_size()
        
        # Estimate activation memory (heuristic: ~25% of params for peak activations)
        activation_bytes = param_bytes // 4
        total_bytes = param_bytes + activation_bytes
        
        # Evict models if needed (memory-aware!)
        while self.current_memory_bytes + total_bytes > self.max_memory_bytes:
            if not self.models:
                raise MemoryError(
                    f"Model requires {total_bytes/1024**3:.1f}GB but limit is "
                    f"{self.max_memory_bytes/1024**3:.1f}GB. Cannot fit even one model."
                )
            self._evict_least_valuable_model()
        
        # Load state dict (leveraging Phase 1 cache)
        state_dict = {}
        for param_name, weight_id in weight_ids.items():
            if param_name in uncached_weights:
                # New weight - add to Phase 1 cache
                weight_tensor = uncached_weights[param_name]
                if isinstance(weight_tensor, dict):
                    weight_tensor = self._deserialize_tensor(weight_tensor)
                
                gpu_tensor = weight_tensor.to(self.device, non_blocking=True)
                # Add to Phase 1 cache
                self.weight_cache.cache_new[weight_id] = gpu_tensor
                state_dict[param_name] = gpu_tensor
            else:
                # Get from Phase 1 cache
                if weight_id in self.weight_cache.cache_new:
                    state_dict[param_name] = self.weight_cache.cache_new[weight_id]
                else:
                    logger.warning(f"Weight {weight_id} not found in cache")
        
        # Load state dict into model
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        model.eval()
        
        # Store model and profile
        self.models[fingerprint] = model
        self.profiles[fingerprint] = ModelMemoryProfile(
            fingerprint=fingerprint,
            param_bytes=param_bytes,
            activation_bytes=activation_bytes,
            total_bytes=total_bytes,
            last_access_time=time.time(),
            access_count=0
        )
        
        self.current_memory_bytes += total_bytes
        
        logger.info(
            f"Model {fingerprint} registered: {total_bytes/1024**3:.2f}GB "
            f"(total cached: {self.current_memory_bytes/1024**3:.2f}GB)"
        )
    
    def _evict_least_valuable_model(self):
        """
        Evict model using value score.
        
        Value = (access_count * execution_speed) / (age * size)
        
        Higher value = worth keeping
        Lower value = safe to evict
        """
        
        if not self.models:
            return
        
        current_time = time.time()
        
        # Calculate value scores
        scores = {}
        for fp, profile in self.profiles.items():
            age_seconds = max(1.0, current_time - profile.last_access_time)
            size_gb = profile.total_bytes / 1024**3
            access_count = profile.access_count + 1  # +1 to avoid division by zero
            execution_speed = 1.0 / max(profile.execution_time_ms, 1.0)  # Faster = higher value
            
            # Value formula: favor frequently accessed, fast-executing, small models
            # Penalize old, large, slow models
            value = (access_count * execution_speed) / (age_seconds * size_gb)
            scores[fp] = value
        
        # Evict lowest value model
        victim = min(scores, key=scores.get)
        victim_profile = self.profiles[victim]
        
        # Remove from cache
        del self.models[victim]
        self.current_memory_bytes -= victim_profile.total_bytes
        del self.profiles[victim]
        
        # Clear from Phase 1 cache (optional - may want to keep weights)
        # For now, keep weights in Phase 1 cache (they're shared)
        
        self.stats['evictions'] += 1
        
        logger.info(
            f"Evicted model {victim}: {victim_profile.total_bytes/1024**3:.2f}GB "
            f"(value={scores[victim]:.4f}, freed {self.current_memory_bytes/1024**3:.2f}GB)"
        )
    
    def execute(self, fingerprint: str, inputs: Dict[str, Any], 
                hints: Optional[Dict] = None) -> torch.Tensor:
        """
        Execute model with OOM protection.
        
        Args:
            fingerprint: Model fingerprint
            inputs: Input tensors dict
            hints: Optional scheduling hints
        
        Returns:
            Output tensor (on CPU)
        
        Raises:
            ValueError: If model not found
            RuntimeError: If execution fails
        """
        
        start_time = time.perf_counter()
        
        self.stats['total_executions'] += 1
        
        if fingerprint not in self.models:
            self.stats['cache_misses'] += 1
            raise ValueError(f"Model {fingerprint} not found. Please register first.")
        
        self.stats['cache_hits'] += 1
        
        # Update access tracking (LRU)
        self.models.move_to_end(fingerprint)
        profile = self.profiles[fingerprint]
        profile.last_access_time = time.time()
        profile.access_count += 1
        
        model = self.models[fingerprint]
        
        # Prepare inputs - handle both dict format and direct tensors
        gpu_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, dict) and 'data' in value:
                # Deserialize from dict format
                tensor = self._deserialize_tensor(value)
                gpu_inputs[key] = tensor.to(self.device, non_blocking=True)
            elif isinstance(value, torch.Tensor):
                # Direct tensor
                gpu_inputs[key] = value.to(self.device, non_blocking=True)
            else:
                # Other types (e.g., lists, scalars)
                gpu_inputs[key] = value
        
        # Execute model with OOM protection
        # Handle nn.Sequential models - they expect positional args, not keyword args
        try:
            # Use new autocast API if available, fallback to old API
            autocast_context = None
            if hasattr(torch.amp, 'autocast'):
                autocast_context = torch.amp.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu', enabled=True)
            else:
                autocast_context = torch.cuda.amp.autocast(enabled=True)
            
            with autocast_context:
                with torch.no_grad():
                    # Handle nn.Sequential models - they expect positional args
                    if isinstance(model, nn.Sequential) and len(gpu_inputs) == 1:
                        # Sequential models expect: model(input_tensor)
                        input_tensor = list(gpu_inputs.values())[0]
                        output = model(input_tensor)
                    else:
                        # Other models expect: model(**inputs) or model(input_tensor)
                        # Try keyword args first, fallback to positional
                        try:
                            output = model(**gpu_inputs)
                        except TypeError:
                            # Fallback to positional args (for Sequential or models that don't accept kwargs)
                            input_values = list(gpu_inputs.values())
                            if len(input_values) == 1:
                                output = model(input_values[0])
                            else:
                                output = model(*input_values)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                self.stats['oom_events'] += 1
                logger.warning(f"OOM during execution of {fingerprint}, clearing cache")
                
                # Emergency eviction
                self._handle_oom()
                
                # Clear CUDA cache
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Retry once with same logic
                if hasattr(torch.amp, 'autocast'):
                    autocast_context = torch.amp.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu', enabled=True)
                else:
                    autocast_context = torch.cuda.amp.autocast(enabled=True)
                
                with autocast_context:
                    with torch.no_grad():
                        if isinstance(model, nn.Sequential) and len(gpu_inputs) == 1:
                            input_tensor = list(gpu_inputs.values())[0]
                            output = model(input_tensor)
                        else:
                            try:
                                output = model(**gpu_inputs)
                            except TypeError:
                                input_values = list(gpu_inputs.values())
                                if len(input_values) == 1:
                                    output = model(input_values[0])
                                else:
                                    output = model(*input_values)
            else:
                raise
        
        # Handle output - transformers models return ModelOutput objects
        if hasattr(output, 'logits'):
            # Transformers model output (e.g., GPT2LMHeadModel)
            output = output.logits
        elif isinstance(output, tuple):
            # Tuple output (e.g., (logits, past_key_values))
            output = output[0]
        
        result = output.cpu()
        
        # Record execution time
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        profile.execution_time_ms = execution_time_ms
        self.stats['total_latency_ms'] += execution_time_ms
        
        return result
    
    def _handle_oom(self):
        """Handle out-of-memory by aggressive eviction."""
        # Evict 25% of models
        evict_count = max(1, len(self.models) // 4)
        for _ in range(evict_count):
            if self.models:
                self._evict_least_valuable_model()
    
    def _deserialize_tensor(self, tensor_dict: Dict) -> torch.Tensor:
        """Deserialize tensor from dict."""
        import numpy as np
        
        data = tensor_dict['data']
        shape = tensor_dict['shape']
        dtype_str = tensor_dict.get('dtype', 'float32')
        
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'int64': torch.int64,
            'int32': torch.int32,
            'bool': torch.bool,
        }
        dtype = dtype_map.get(dtype_str, torch.float32)
        
        if isinstance(data, list):
            tensor = torch.tensor(data, dtype=dtype)
        else:
            tensor = torch.from_numpy(np.array(data, dtype=dtype.numpy()))
        
        if list(tensor.shape) != shape:
            tensor = tensor.reshape(shape)
        
        return tensor
    
    def get_memory_status(self) -> Dict:
        """Get current memory status."""
        return {
            'used_gb': self.current_memory_bytes / 1024**3,
            'max_gb': self.max_memory_bytes / 1024**3,
            'utilization': self.current_memory_bytes / self.max_memory_bytes if self.max_memory_bytes > 0 else 0.0,
            'num_models': len(self.models),
            'stats': self.stats.copy()
        }
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_execs = self.stats['total_executions']
        if total_execs > 0:
            hit_rate = self.stats['cache_hits'] / total_execs
            avg_latency = self.stats['total_latency_ms'] / total_execs
        else:
            hit_rate = 0
            avg_latency = 0
        
        return {
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': hit_rate,
            'avg_latency_ms': avg_latency,
            'total_executions': total_execs,
            'cached_models': len(self.models),
            'evictions': self.stats['evictions'],
            'oom_events': self.stats['oom_events'],
            **self.get_memory_status()
        }
    
    def clear(self):
        """Clear all cached models."""
        self.models.clear()
        self.profiles.clear()
        self.current_memory_bytes = 0
        logger.info("Model cache cleared")

