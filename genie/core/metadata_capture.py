"""
Metadata capture during LazyTensor construction.

Key insight: We can extract semantic info from:
1. Call stack (module path)
2. Operation sequence (pattern hints)
3. Tensor properties (shape, dtype)
"""

import inspect
import torch.nn as nn
from typing import Dict, Optional


class MetadataCapture:
    """Captures semantic metadata during graph construction."""

    def __init__(self):
        self._module_stack = []  # Track nn.Module hierarchy
        self._operation_sequence = []  # Track recent operations

    def capture_metadata(self, operation: str, inputs: list, kwargs: dict) -> Dict:
        """
        Capture metadata for a LazyTensor operation.

        Called from LazyTensor.__torch_dispatch__ and __torch_function__.
        """
        metadata = {}

        # 1. Capture module context (if inside nn.Module)
        module_path = self._get_module_context()
        if module_path:
            metadata['module_path'] = module_path
            metadata['module_type'] = self._get_module_type(module_path)

        # 2. Infer semantic role from operation
        semantic_role = self._infer_semantic_role(operation, inputs)
        if semantic_role:
            metadata['semantic_role'] = semantic_role

        # 3. Track operation sequence for pattern hints
        self._operation_sequence.append(operation)
        if len(self._operation_sequence) > 10:
            self._operation_sequence.pop(0)

        # 4. Check for common patterns in recent operations
        pattern_hints = self._detect_pattern_hints()
        if pattern_hints:
            metadata['pattern_hints'] = pattern_hints

        # 5. Infer modality from tensor properties
        modality = self._infer_modality(inputs, kwargs)
        if modality:
            metadata['modality'] = modality

        return metadata

    def _get_module_context(self) -> Optional[str]:
        """Extract current nn.Module path from call stack."""
        stack = inspect.stack()

        for frame_info in stack:
            frame_locals = frame_info.frame.f_locals

            # Look for 'self' that's an nn.Module
            if 'self' in frame_locals:
                obj = frame_locals['self']
                if isinstance(obj, nn.Module):
                    # Found the module - return its path
                    return self._get_module_path(obj)

        return None

    def _get_module_path(self, module: nn.Module) -> str:
        """Get hierarchical path of module (e.g., 'encoder.layer.0.attention')."""
        # Try to get from module's _modules dict if it's a submodule
        if hasattr(module, '_forward_hooks'):
            # This is a registered submodule
            for name, mod in module.named_modules():
                if mod is module:
                    return name

        # Fallback: use class name
        return module.__class__.__name__

    def _get_module_type(self, module_path: str) -> str:
        """Get module type from path."""
        if 'attention' in module_path.lower():
            return 'MultiHeadAttention'
        elif 'conv' in module_path.lower():
            return 'Convolution'
        elif 'linear' in module_path.lower():
            return 'Linear'
        return 'Unknown'

    def _infer_semantic_role(self, operation: str, inputs: list) -> Optional[str]:
        """Infer semantic role from operation."""
        op_lower = operation.lower()

        if 'matmul' in op_lower or 'bmm' in op_lower:
            # Could be attention or linear layer
            if len(inputs) >= 2:
                # Check shapes for attention pattern
                # Q@K.T has matching inner dims
                return 'matmul'  # Will be refined by pattern detection

        elif 'softmax' in op_lower:
            return 'attention_softmax'

        elif 'conv' in op_lower:
            return 'convolution'

        elif 'relu' in op_lower or 'gelu' in op_lower:
            return 'activation'

        return None

    def _detect_pattern_hints(self) -> Optional[Dict]:
        """Detect pattern hints from recent operation sequence."""
        recent_ops = self._operation_sequence[-5:]  # Last 5 ops

        # Attention pattern: matmul → softmax → matmul
        if len(recent_ops) >= 3:
            if ('matmul' in recent_ops[-3] and
                'softmax' in recent_ops[-2] and
                'matmul' in recent_ops[-1]):
                return {'likely_pattern': 'attention'}

        # Conv-BN-ReLU pattern
        if len(recent_ops) >= 3:
            if ('conv' in recent_ops[-3] and
                'batch_norm' in recent_ops[-2] and
                'relu' in recent_ops[-1]):
                return {'likely_pattern': 'conv_bn_relu'}

        return None

    def _infer_modality(self, inputs: list, kwargs: dict) -> Optional[str]:
        """Infer modality from tensor properties."""
        for inp in inputs:
            if hasattr(inp, 'shape') and len(inp.shape) >= 4:
                # 4D tensor likely image: [N, C, H, W]
                return 'vision'
            elif hasattr(inp, 'shape') and len(inp.shape) == 2:
                # 2D tensor likely text embeddings: [batch, seq_len]
                return 'text'

        return None


# Global metadata capture instance
_metadata_capture = MetadataCapture()


def get_metadata_capture() -> MetadataCapture:
    """Get global metadata capture instance."""
    return _metadata_capture
