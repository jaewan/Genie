"""
Shape Inference System for LazyTensor

This module provides shape, dtype, and device inference for operations
during graph capture. This enables local metadata queries without remote calls.

Design Principles:
1. Infer shapes locally during graph construction (Stage 1)
2. No remote calls - all inference is local
3. Conservative fallbacks when exact inference is complex
4. Validate against PyTorch behavior in tests
"""

import torch
from typing import Any, Dict, List, Optional, Tuple, Union
import math


class ShapeInferenceError(Exception):
    """Raised when shape cannot be inferred."""
    pass


class ShapeInference:
    """
    Infer output shapes, dtypes, and devices from operations and inputs.
    
    This runs LOCALLY during graph capture to populate LazyTensor metadata
    without requiring remote calls or actual computation.
    """
    
    @staticmethod
    def infer_shape(operation: str, inputs: List[Any], kwargs: Dict[str, Any]) -> torch.Size:
        """
        Infer output shape from operation and inputs.
        
        Args:
            operation: Operation name (e.g., 'aten::randn', 'aten::matmul')
            inputs: List of input tensors/values
            kwargs: Keyword arguments to the operation
            
        Returns:
            Inferred output shape
            
        Raises:
            ShapeInferenceError: If shape cannot be inferred
        """
        # Factory functions (create new tensors)
        if operation in ('aten::randn', 'aten::rand', 'aten::zeros', 
                        'aten::ones', 'aten::empty', 'aten::full', 'aten::randint'):
            return ShapeInference._infer_factory_shape(operation, inputs, kwargs)
        
        # Element-wise operations (preserve shape with broadcasting)
        # Include in-place variants (add_, sub_, mul_, div_, etc.)
        elif operation in ('aten::add', 'aten::add_', 'aten::sub', 'aten::sub_', 
                          'aten::mul', 'aten::mul_', 'aten::div', 'aten::div_',
                          'aten::pow', 'aten::neg', 'aten::abs', 'aten::sqrt',
                          'aten::exp', 'aten::log', 'aten::sin', 'aten::cos',
                          'aten::relu', 'aten::sigmoid', 'aten::tanh',
                          'aten::gelu', 'aten::silu'):
            return ShapeInference._infer_elementwise_shape(operation, inputs, kwargs)
        
        # Matrix operations
        elif operation in ('aten::matmul', 'aten::mm', 'aten::bmm'):
            return ShapeInference._infer_matmul_shape(operation, inputs, kwargs)
        
        # Reduction operations
        elif operation in ('aten::sum', 'aten::mean', 'aten::max', 'aten::min',
                          'aten::prod', 'aten::std', 'aten::var'):
            return ShapeInference._infer_reduction_shape(operation, inputs, kwargs)
        
        # Shape manipulation
        elif operation in ('aten::reshape', 'aten::view', 'aten::flatten'):
            return ShapeInference._infer_reshape_shape(operation, inputs, kwargs)
        
        elif operation in ('aten::transpose', 'aten::permute'):
            return ShapeInference._infer_transpose_shape(operation, inputs, kwargs)
        
        elif operation in ('aten::squeeze', 'aten::unsqueeze'):
            return ShapeInference._infer_squeeze_shape(operation, inputs, kwargs)
        
        # Concatenation and stacking
        elif operation in ('aten::cat', 'aten::concat', 'aten::concatenate'):
            return ShapeInference._infer_cat_shape(operation, inputs, kwargs)
        
        elif operation == 'aten::stack':
            return ShapeInference._infer_stack_shape(operation, inputs, kwargs)
        
        # Indexing and slicing
        elif operation in ('aten::select', 'aten::index_select', 'aten::gather'):
            return ShapeInference._infer_indexing_shape(operation, inputs, kwargs)
        
        # Convolution operations
        elif operation in ('aten::conv2d', 'aten::conv1d', 'aten::conv3d'):
            return ShapeInference._infer_conv_shape(operation, inputs, kwargs)
        
        # Pooling operations
        elif operation in ('aten::max_pool2d', 'aten::avg_pool2d', 'aten::adaptive_avg_pool2d'):
            return ShapeInference._infer_pool_shape(operation, inputs, kwargs)
        
        # Normalization
        elif operation in ('aten::batch_norm', 'aten::layer_norm', 'aten::group_norm'):
            return ShapeInference._infer_norm_shape(operation, inputs, kwargs)
        
        # Dropout (preserves shape)
        elif operation in ('aten::dropout', 'aten::dropout_'):
            return ShapeInference._infer_norm_shape(operation, inputs, kwargs)
        
        # Embedding
        elif operation == 'aten::embedding':
            return ShapeInference._infer_embedding_shape(operation, inputs, kwargs)
        
        # Linear
        elif operation in ('aten::linear', 'aten::addmm'):
            return ShapeInference._infer_linear_shape(operation, inputs, kwargs)
        
        # Fallback: try to infer from first input
        else:
            return ShapeInference._infer_generic(operation, inputs, kwargs)
    
    @staticmethod
    def _infer_factory_shape(operation: str, inputs: List[Any], kwargs: Dict) -> torch.Size:
        """Infer shape for factory functions (randn, zeros, ones, etc.)."""
        # Special case for randint: torch.randint(low, high, size)
        if operation == 'aten::randint':
            # Size is typically the 3rd argument or in kwargs
            if len(inputs) >= 3:
                size = inputs[2]
                if isinstance(size, (list, tuple)):
                    return torch.Size(size)
                elif isinstance(size, int):
                    return torch.Size([size])
                elif isinstance(size, torch.Size):
                    return size
            if 'size' in kwargs:
                size = kwargs['size']
                if isinstance(size, (list, tuple)):
                    return torch.Size(size)
                elif isinstance(size, int):
                    return torch.Size([size])
                elif isinstance(size, torch.Size):
                    return size
            # Fallback: scalar
            return torch.Size([])
        
        # Try to get size from inputs first
        if inputs:
            size = inputs[0] if len(inputs) == 1 else inputs
            if isinstance(size, (list, tuple)):
                return torch.Size(size)
            elif isinstance(size, int):
                return torch.Size([size])
            elif isinstance(size, torch.Size):
                return size
        
        # Try kwargs
        if 'size' in kwargs:
            size = kwargs['size']
            if isinstance(size, (list, tuple)):
                return torch.Size(size)
            elif isinstance(size, int):
                return torch.Size([size])
            elif isinstance(size, torch.Size):
                return size
        
        # Default: scalar
        return torch.Size([])
    
    @staticmethod
    def _infer_elementwise_shape(operation: str, inputs: List[Any], kwargs: Dict) -> torch.Size:
        """Infer shape for element-wise operations (with broadcasting)."""
        if not inputs:
            raise ShapeInferenceError(f"No inputs for {operation}")
        
        # Unary operations
        if operation in ('aten::neg', 'aten::abs', 'aten::sqrt', 'aten::exp', 
                        'aten::log', 'aten::sin', 'aten::cos', 'aten::relu',
                        'aten::sigmoid', 'aten::tanh'):
            return ShapeInference._get_shape(inputs[0])
        
        # Binary operations with broadcasting
        if len(inputs) >= 2:
            shape_a = ShapeInference._get_shape(inputs[0])
            shape_b = ShapeInference._get_shape(inputs[1])
            return torch.broadcast_shapes(shape_a, shape_b)
        
        # Single input
        return ShapeInference._get_shape(inputs[0])
    
    @staticmethod
    def _infer_matmul_shape(operation: str, inputs: List[Any], kwargs: Dict) -> torch.Size:
        """Infer shape for matrix multiplication operations."""
        if len(inputs) < 2:
            raise ShapeInferenceError(f"Need 2 inputs for {operation}")
        
        shape_a = ShapeInference._get_shape(inputs[0])
        shape_b = ShapeInference._get_shape(inputs[1])
        
        # Handle different matmul cases
        if operation == 'aten::mm':
            # Matrix-matrix: [M, K] @ [K, N] -> [M, N]
            if len(shape_a) != 2 or len(shape_b) != 2:
                raise ShapeInferenceError(f"mm requires 2D tensors")
            return torch.Size([shape_a[0], shape_b[1]])
        
        elif operation == 'aten::bmm':
            # Batch matrix-matrix: [B, M, K] @ [B, K, N] -> [B, M, N]
            if len(shape_a) != 3 or len(shape_b) != 3:
                raise ShapeInferenceError(f"bmm requires 3D tensors")
            return torch.Size([shape_a[0], shape_a[1], shape_b[2]])
        
        else:  # aten::matmul (general case)
            # Handle various combinations
            if len(shape_a) == 1 and len(shape_b) == 1:
                # Vector dot product: [] (scalar)
                return torch.Size([])
            elif len(shape_a) == 2 and len(shape_b) == 1:
                # Matrix-vector: [M, K] @ [K] -> [M]
                return torch.Size([shape_a[0]])
            elif len(shape_a) == 1 and len(shape_b) == 2:
                # Vector-matrix: [K] @ [K, N] -> [N]
                return torch.Size([shape_b[1]])
            elif len(shape_a) == 2 and len(shape_b) == 2:
                # Matrix-matrix: [M, K] @ [K, N] -> [M, N]
                return torch.Size([shape_a[0], shape_b[1]])
            else:
                # Batched matmul: broadcast batch dimensions
                # [..., M, K] @ [..., K, N] -> [..., M, N]
                batch_shape = torch.broadcast_shapes(shape_a[:-2], shape_b[:-2])
                return torch.Size(list(batch_shape) + [shape_a[-2], shape_b[-1]])
    
    @staticmethod
    def _infer_reduction_shape(operation: str, inputs: List[Any], kwargs: Dict) -> torch.Size:
        """Infer shape for reduction operations (sum, mean, etc.)."""
        if not inputs:
            raise ShapeInferenceError(f"No inputs for {operation}")
        
        input_shape = ShapeInference._get_shape(inputs[0])
        dim = kwargs.get('dim', None)
        keepdim = kwargs.get('keepdim', False)
        
        if dim is None:
            # Reduce all dimensions
            return torch.Size([]) if not keepdim else torch.Size([1] * len(input_shape))
        
        # Reduce specific dimension(s)
        if isinstance(dim, int):
            dim = [dim]
        
        # Normalize negative dimensions
        dim = [d if d >= 0 else len(input_shape) + d for d in dim]
        
        if keepdim:
            # Keep reduced dimensions as size 1
            new_shape = list(input_shape)
            for d in dim:
                new_shape[d] = 1
            return torch.Size(new_shape)
        else:
            # Remove reduced dimensions
            new_shape = [s for i, s in enumerate(input_shape) if i not in dim]
            return torch.Size(new_shape)
    
    @staticmethod
    def _infer_reshape_shape(operation: str, inputs: List[Any], kwargs: Dict) -> torch.Size:
        """Infer shape for reshape/view/flatten operations."""
        if not inputs:
            raise ShapeInferenceError(f"No inputs for {operation}")
        
        input_shape = ShapeInference._get_shape(inputs[0])
        
        if operation == 'aten::flatten':
            start_dim = kwargs.get('start_dim', 0) if len(inputs) < 2 else inputs[1]
            end_dim = kwargs.get('end_dim', -1) if len(inputs) < 3 else inputs[2]
            
            # Normalize negative dimensions
            start_dim = start_dim if start_dim >= 0 else len(input_shape) + start_dim
            end_dim = end_dim if end_dim >= 0 else len(input_shape) + end_dim
            
            # Flatten dimensions from start_dim to end_dim
            flattened_size = math.prod(input_shape[start_dim:end_dim+1])
            new_shape = list(input_shape[:start_dim]) + [flattened_size] + list(input_shape[end_dim+1:])
            return torch.Size(new_shape)
        
        else:  # reshape/view
            # Get new shape from inputs or kwargs
            new_shape = inputs[1] if len(inputs) > 1 else kwargs.get('shape', kwargs.get('size'))
            
            if isinstance(new_shape, (list, tuple)):
                # Handle -1 (infer dimension)
                new_shape = list(new_shape)
                if -1 in new_shape:
                    known_size = math.prod([s for s in new_shape if s != -1])
                    total_size = math.prod(input_shape)
                    inferred_size = total_size // known_size
                    new_shape = [inferred_size if s == -1 else s for s in new_shape]
                
                return torch.Size(new_shape)
            
            raise ShapeInferenceError(f"Cannot infer shape for {operation}")
    
    @staticmethod
    def _infer_transpose_shape(operation: str, inputs: List[Any], kwargs: Dict) -> torch.Size:
        """Infer shape for transpose/permute operations."""
        if not inputs:
            raise ShapeInferenceError(f"No inputs for {operation}")
        
        input_shape = ShapeInference._get_shape(inputs[0])
        
        if operation == 'aten::transpose':
            dim0 = inputs[1] if len(inputs) > 1 else kwargs.get('dim0', 0)
            dim1 = inputs[2] if len(inputs) > 2 else kwargs.get('dim1', 1)
            
            # Normalize negative dimensions
            dim0 = dim0 if dim0 >= 0 else len(input_shape) + dim0
            dim1 = dim1 if dim1 >= 0 else len(input_shape) + dim1
            
            # Swap dimensions
            new_shape = list(input_shape)
            new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
            return torch.Size(new_shape)
        
        else:  # permute
            dims = inputs[1] if len(inputs) > 1 else kwargs.get('dims')
            if dims is None:
                raise ShapeInferenceError(f"No dims for permute")
            
            # Permute dimensions
            new_shape = [input_shape[d] for d in dims]
            return torch.Size(new_shape)
    
    @staticmethod
    def _infer_squeeze_shape(operation: str, inputs: List[Any], kwargs: Dict) -> torch.Size:
        """Infer shape for squeeze/unsqueeze operations."""
        if not inputs:
            raise ShapeInferenceError(f"No inputs for {operation}")
        
        input_shape = ShapeInference._get_shape(inputs[0])
        
        if operation == 'aten::unsqueeze':
            dim = inputs[1] if len(inputs) > 1 else kwargs.get('dim', 0)
            # Normalize negative dimension
            dim = dim if dim >= 0 else len(input_shape) + dim + 1
            
            # Insert dimension of size 1
            new_shape = list(input_shape)
            new_shape.insert(dim, 1)
            return torch.Size(new_shape)
        
        else:  # squeeze
            dim = inputs[1] if len(inputs) > 1 else kwargs.get('dim', None)
            
            if dim is None:
                # Remove all dimensions of size 1
                new_shape = [s for s in input_shape if s != 1]
                return torch.Size(new_shape)
            else:
                # Remove specific dimension if size 1
                dim = dim if dim >= 0 else len(input_shape) + dim
                new_shape = list(input_shape)
                if input_shape[dim] == 1:
                    new_shape.pop(dim)
                return torch.Size(new_shape)
    
    @staticmethod
    def _infer_cat_shape(operation: str, inputs: List[Any], kwargs: Dict) -> torch.Size:
        """Infer shape for concatenation operations."""
        tensors = inputs[0] if inputs and isinstance(inputs[0], (list, tuple)) else inputs
        dim = kwargs.get('dim', 0)
        
        if not tensors:
            raise ShapeInferenceError(f"No tensors for {operation}")
        
        # Get first tensor shape
        base_shape = ShapeInference._get_shape(tensors[0])
        dim = dim if dim >= 0 else len(base_shape) + dim
        
        # Sum sizes along concatenation dimension
        concat_size = sum(ShapeInference._get_shape(t)[dim] for t in tensors)
        
        # Build output shape
        new_shape = list(base_shape)
        new_shape[dim] = concat_size
        return torch.Size(new_shape)
    
    @staticmethod
    def _infer_stack_shape(operation: str, inputs: List[Any], kwargs: Dict) -> torch.Size:
        """Infer shape for stack operation."""
        tensors = inputs[0] if inputs and isinstance(inputs[0], (list, tuple)) else inputs
        dim = kwargs.get('dim', 0)
        
        if not tensors:
            raise ShapeInferenceError(f"No tensors for {operation}")
        
        # Get first tensor shape
        base_shape = ShapeInference._get_shape(tensors[0])
        dim = dim if dim >= 0 else len(base_shape) + dim + 1
        
        # Insert new dimension
        new_shape = list(base_shape)
        new_shape.insert(dim, len(tensors))
        return torch.Size(new_shape)
    
    @staticmethod
    def _infer_indexing_shape(operation: str, inputs: List[Any], kwargs: Dict) -> torch.Size:
        """Infer shape for indexing operations (conservative fallback)."""
        # This is complex - use conservative fallback
        if inputs:
            return ShapeInference._get_shape(inputs[0])
        raise ShapeInferenceError(f"Cannot infer shape for {operation}")
    
    @staticmethod
    def _infer_conv_shape(operation: str, inputs: List[Any], kwargs: Dict) -> torch.Size:
        """Infer shape for convolution operations."""
        # Simplified - assumes default parameters
        # Full implementation would need stride, padding, dilation, etc.
        if len(inputs) < 2:
            raise ShapeInferenceError(f"Need input and weight for {operation}")
        
        input_shape = ShapeInference._get_shape(inputs[0])
        weight_shape = ShapeInference._get_shape(inputs[1])
        
        # Conv output: [batch, out_channels, ...]
        # Spatial dimensions depend on stride, padding, etc.
        # Conservative: keep spatial dimensions same (assumes padding='same')
        batch_size = input_shape[0]
        out_channels = weight_shape[0]
        spatial_dims = input_shape[2:]  # Conservative: keep same
        
        return torch.Size([batch_size, out_channels] + list(spatial_dims))
    
    @staticmethod
    def _infer_pool_shape(operation: str, inputs: List[Any], kwargs: Dict) -> torch.Size:
        """Infer shape for pooling operations."""
        if not inputs:
            raise ShapeInferenceError(f"No inputs for {operation}")
        
        input_shape = ShapeInference._get_shape(inputs[0])
        
        if 'adaptive' in operation:
            # Adaptive pooling: output size specified
            output_size = kwargs.get('output_size', inputs[1] if len(inputs) > 1 else None)
            if output_size is None:
                raise ShapeInferenceError(f"No output_size for {operation}")
            
            if isinstance(output_size, int):
                output_size = [output_size] * (len(input_shape) - 2)
            
            return torch.Size([input_shape[0], input_shape[1]] + list(output_size))
        
        # Regular pooling: keep batch and channels, reduce spatial
        # Conservative: keep spatial dimensions same
        return input_shape
    
    @staticmethod
    def _infer_norm_shape(operation: str, inputs: List[Any], kwargs: Dict) -> torch.Size:
        """Infer shape for normalization operations."""
        # Normalization preserves shape
        if inputs:
            return ShapeInference._get_shape(inputs[0])
        raise ShapeInferenceError(f"No inputs for {operation}")
    
    @staticmethod
    def _infer_embedding_shape(operation: str, inputs: List[Any], kwargs: Dict) -> torch.Size:
        """Infer shape for embedding operation.
        
        Note: aten::embedding signature is embedding(weight, indices, ...)
        where weight is [vocab_size, embedding_dim] and indices is [batch, seq_len]
        """
        if len(inputs) < 2:
            raise ShapeInferenceError(f"Need weight and indices for {operation}")
        
        weight_shape = ShapeInference._get_shape(inputs[0])  # [vocab_size, embedding_dim]
        indices_shape = ShapeInference._get_shape(inputs[1])  # [batch, seq_len]
        
        embedding_dim = weight_shape[1] if len(weight_shape) > 1 else weight_shape[0]
        
        # Output: [...indices_shape, embedding_dim]
        return torch.Size(list(indices_shape) + [embedding_dim])
    
    @staticmethod
    def _infer_linear_shape(operation: str, inputs: List[Any], kwargs: Dict) -> torch.Size:
        """Infer shape for linear/addmm operations."""
        if len(inputs) < 2:
            raise ShapeInferenceError(f"Need input and weight for {operation}")
        
        input_shape = ShapeInference._get_shape(inputs[0])
        weight_shape = ShapeInference._get_shape(inputs[1])
        
        # Linear: [..., in_features] @ [out_features, in_features].T -> [..., out_features]
        out_features = weight_shape[0]
        
        return torch.Size(list(input_shape[:-1]) + [out_features])
    
    @staticmethod
    def _infer_generic(operation: str, inputs: List[Any], kwargs: Dict) -> torch.Size:
        """Generic fallback: try to infer from first input."""
        if inputs:
            try:
                return ShapeInference._get_shape(inputs[0])
            except:
                pass
        
        # Last resort: return empty shape (scalar)
        return torch.Size([])
    
    @staticmethod
    def _get_shape(obj: Any) -> torch.Size:
        """Extract shape from an object (tensor, LazyTensor, or value)."""
        if hasattr(obj, 'shape'):
            shape = obj.shape
            if isinstance(shape, torch.Size):
                return shape
            elif isinstance(shape, (list, tuple)):
                return torch.Size(shape)
        
        if hasattr(obj, '_shape'):
            return obj._shape
        
        if isinstance(obj, (int, float)):
            return torch.Size([])  # Scalar
        
        if isinstance(obj, (list, tuple)):
            return torch.Size([len(obj)])
        
        raise ShapeInferenceError(f"Cannot extract shape from {type(obj)}")
    
    @staticmethod
    def infer_dtype(operation: str, inputs: List[Any], kwargs: Dict[str, Any]) -> torch.dtype:
        """
        Infer output dtype from operation and inputs.
        
        Args:
            operation: Operation name
            inputs: List of input tensors/values
            kwargs: Keyword arguments
            
        Returns:
            Inferred output dtype
        """
        # Check kwargs first
        if 'dtype' in kwargs and kwargs['dtype'] is not None:
            return kwargs['dtype']
        
        # Factory functions: use specified dtype or default
        if operation in ('aten::randn', 'aten::rand', 'aten::zeros', 
                        'aten::ones', 'aten::empty'):
            return kwargs.get('dtype', torch.float32)
        
        # randint returns int64 by default
        if operation == 'aten::randint':
            return kwargs.get('dtype', torch.int64)
        
        # Get dtype from first input
        if inputs:
            for inp in inputs:
                if hasattr(inp, 'dtype'):
                    return inp.dtype
                elif hasattr(inp, '_dtype'):
                    return inp._dtype
        
        # Default: float32
        return torch.float32
    
    @staticmethod
    def infer_device(operation: str, inputs: List[Any], kwargs: Dict[str, Any]) -> torch.device:
        """
        Infer output device from operation and inputs.
        
        Args:
            operation: Operation name
            inputs: List of input tensors/values
            kwargs: Keyword arguments
            
        Returns:
            Inferred output device
        """
        # Check kwargs first
        if 'device' in kwargs and kwargs['device'] is not None:
            device = kwargs['device']
            if isinstance(device, torch.device):
                return device
            return torch.device(device)
        
        # Get device from first input
        if inputs:
            for inp in inputs:
                if hasattr(inp, 'device'):
                    return inp.device
                elif hasattr(inp, '_device'):
                    return inp._device
        
        # Default: CPU
        return torch.device('cpu')
