"""
Optimized shape inference for LazyTensor operations.

Features:
- Lazy shape computation (only when needed)
- Efficient caching with bounded size
- Meta tensor inference (zero-copy shape propagation)
- Graceful fallback for complex operations
"""

from __future__ import annotations
import logging
from typing import Optional, Dict, Any, Tuple, Callable
import torch
from functools import lru_cache

logger = logging.getLogger(__name__)


# ============================================================================
# SHAPE INFERENCE OPTIMIZATION
# ============================================================================

class ShapeInferenceCache:
    """
    Efficient shape inference cache with lazy evaluation.
    
    Provides:
    - O(1) lookup for previously inferred shapes
    - Automatic cache eviction when full
    - Thread-safe operations (using threading.RLock)
    """
    
    MAX_CACHE_SIZE = 2048
    
    def __init__(self):
        import threading
        self._cache: Dict[str, torch.Size] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, cache_key: str) -> Optional[torch.Size]:
        """Get cached shape (O(1) lookup)."""
        with self._lock:
            if cache_key in self._cache:
                self._hits += 1
                return self._cache[cache_key]
            self._misses += 1
            return None
    
    def put(self, cache_key: str, shape: torch.Size) -> None:
        """Store shape in cache."""
        with self._lock:
            if len(self._cache) >= self.MAX_CACHE_SIZE:
                # Simple eviction: remove oldest entries
                keys_to_remove = list(self._cache.keys())[:int(self.MAX_CACHE_SIZE * 0.2)]
                for key in keys_to_remove:
                    del self._cache[key]
            
            self._cache[cache_key] = shape
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                'cache_size': len(self._cache),
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': f"{hit_rate:.1f}%",
            }
    
    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


class LazyShapeInference:
    """
    Lazy shape inference engine for LazyTensor operations.
    
    Strategy:
    1. Cache frequently accessed shapes
    2. Use meta tensors for zero-copy inference
    3. Gracefully fallback for complex operations
    4. Lazy compute (only when accessed)
    """
    
    def __init__(self):
        self._cache = ShapeInferenceCache()
        self._meta_mode_enabled = self._check_meta_support()
    
    def _check_meta_support(self) -> bool:
        """Check if PyTorch meta device is available."""
        try:
            torch.empty(1, device='meta')
            return True
        except Exception:
            return False
    
    def infer_shape(self, operation: str, input_shapes: list,
                   operation_args: Dict[str, Any]) -> Optional[torch.Size]:
        """
        Infer output shape of operation.
        
        Args:
            operation: Operation name (e.g., 'aten::matmul')
            input_shapes: List of input tensor shapes
            operation_args: Additional arguments (e.g., kernel size for conv)
        
        Returns:
            Inferred shape or None if inference fails
        """
        # Generate cache key
        cache_key = self._make_cache_key(operation, input_shapes)
        
        # Check cache
        cached_shape = self._cache.get(cache_key)
        if cached_shape is not None:
            return cached_shape
        
        # ✅ CYCLE DETECTION: Prevent infinite recursion
        if not hasattr(self, '_inference_stack'):
            self._inference_stack = set()
        
        if cache_key in self._inference_stack:
            logger.warning(f"Cycle detected in shape inference for {operation}")
            return None  # Graceful fallback
        
        self._inference_stack.add(cache_key)
        
        try:
            # Infer shape
            shape = self._infer_shape_unsafe(operation, input_shapes, operation_args)
            
            if shape is not None:
                self._cache.put(cache_key, shape)
            
            return shape
        
        finally:
            # Always clean up the stack to prevent false positives
            self._inference_stack.discard(cache_key)
    
    def _infer_shape_unsafe(self, operation: str, input_shapes: list,
                           operation_args: Dict[str, Any]) -> Optional[torch.Size]:
        """Internal shape inference without cycle detection."""
        try:
            if self._meta_mode_enabled:
                shape = self._infer_with_meta_tensors(
                    operation, input_shapes, operation_args
                )
            else:
                shape = self._infer_with_heuristics(
                    operation, input_shapes, operation_args
                )
            
            return shape
        
        except Exception as e:
            logger.debug(f"Shape inference failed for {operation}: {e}")
            return None
    
    def _infer_with_meta_tensors(self, operation: str, input_shapes: list,
                                operation_args: Dict[str, Any]) -> Optional[torch.Size]:
        """
        Infer shape using meta tensors (zero-copy).
        
        This is the most efficient method: creates fake tensors with correct shape
        but no data, executes operation symbolically, and extracts output shape.
        """
        try:
            # Create meta tensors (no memory allocated)
            meta_inputs = []
            for shape in input_shapes:
                meta_tensor = torch.empty(shape, device='meta')
                meta_inputs.append(meta_tensor)
            
            # Execute operation on meta tensors
            with torch.no_grad():
                # Parse operation name to get actual function
                op_fn = self._get_op_function(operation)
                if op_fn is None:
                    return None
                
                meta_output = op_fn(*meta_inputs, **operation_args)
            
            # Extract shape from result
            if isinstance(meta_output, torch.Tensor):
                return meta_output.shape
            
            return None
        
        except Exception as e:
            logger.debug(f"Meta tensor inference failed: {e}")
            return None
    
    def _infer_with_heuristics(self, operation: str, input_shapes: list,
                              operation_args: Dict[str, Any]) -> Optional[torch.Size]:
        """
        Infer shape using operation-specific heuristics.
        
        Fallback when meta tensors unavailable.
        """
        if not input_shapes:
            return None
        
        op_lower = operation.lower()
        
        # Matrix multiply: output shape = (N, K) for (N, M) @ (M, K)
        if 'matmul' in op_lower or 'gemm' in op_lower:
            if len(input_shapes[0]) >= 2 and len(input_shapes[1]) >= 2:
                return torch.Size(list(input_shapes[0][:-1]) + [input_shapes[1][-1]])
        
        # Element-wise operations: output shape = input shape (broadcasted)
        elif any(op in op_lower for op in ['add', 'sub', 'mul', 'div', 'relu']):
            return input_shapes[0]  # Simplified broadcasting
        
        # Convolution: output shape = (N, C_out, H_out, W_out)
        elif 'conv' in op_lower:
            N, C_in, H_in, W_in = input_shapes[0]
            kernel_size = operation_args.get('kernel_size', 3)
            padding = operation_args.get('padding', 0)
            stride = operation_args.get('stride', 1)
            out_channels = operation_args.get('out_channels', C_in)
            
            H_out = (H_in + 2 * padding - kernel_size) // stride + 1
            W_out = (W_in + 2 * padding - kernel_size) // stride + 1
            
            return torch.Size([N, out_channels, H_out, W_out])
        
        # Flatten: (N, ...) -> (N, -1)
        elif 'flatten' in op_lower or 'reshape' in op_lower:
            shape = operation_args.get('shape', [-1])
            return torch.Size(shape) if shape else input_shapes[0]
        
        # Pooling: reduces spatial dimensions
        elif any(op in op_lower for op in ['pool', 'avg_pool', 'max_pool']):
            if len(input_shapes[0]) == 4:
                N, C, H, W = input_shapes[0]
                kernel_size = operation_args.get('kernel_size', 2)
                stride = operation_args.get('stride', kernel_size)
                H_out = (H - kernel_size) // stride + 1
                W_out = (W - kernel_size) // stride + 1
                return torch.Size([N, C, H_out, W_out])
        
        # Transpose: swap dimensions
        elif 'transpose' in op_lower or 't' in op_lower:
            dims = operation_args.get('dims', (0, 1))
            shape = list(input_shapes[0])
            shape[dims[0]], shape[dims[1]] = shape[dims[1]], shape[dims[0]]
            return torch.Size(shape)
        
        # Default: return input shape (conservative)
        return input_shapes[0]
    
    def _make_cache_key(self, operation: str, input_shapes: list) -> str:
        """Create deterministic cache key."""
        shapes_str = ",".join(str(tuple(s)) for s in input_shapes)
        return f"{operation}:{shapes_str}"
    
    def _get_op_function(self, operation: str) -> Optional[Callable]:
        """Get actual operation function from operation name."""
        import torch.nn.functional as F
        
        op_map = {
            'aten::matmul': torch.matmul,
            'aten::add': torch.add,
            'aten::mul': torch.mul,
            'aten::conv2d': F.conv2d,
            'aten::relu': F.relu,
            'aten::flatten': torch.flatten,
        }
        
        # Try exact match
        if operation in op_map:
            return op_map[operation]
        
        # Try partial match
        for key, func in op_map.items():
            if key in operation or operation in key:
                return func
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return self._cache.stats()


# ============================================================================
# GLOBAL SHAPE INFERENCE ENGINE
# ============================================================================

_global_shape_inferencer: Optional[LazyShapeInference] = None


def get_shape_inferencer() -> LazyShapeInference:
    """Get or create global shape inference engine."""
    global _global_shape_inferencer
    if _global_shape_inferencer is None:
        _global_shape_inferencer = LazyShapeInference()
    return _global_shape_inferencer


def infer_shape(operation: str, input_shapes: list,
               operation_args: Dict[str, Any] = None) -> Optional[torch.Size]:
    """
    Infer output shape (convenience function).
    
    Usage:
        shape = infer_shape('aten::matmul', [torch.Size([10, 20]), torch.Size([20, 30])])
        # Returns: torch.Size([10, 30])
    """
    operation_args = operation_args or {}
    inferencer = get_shape_inferencer()
    return inferencer.infer_shape(operation, input_shapes, operation_args)


__all__ = [
    'LazyShapeInference',
    'ShapeInferenceCache',
    'get_shape_inferencer',
    'infer_shape',
]
