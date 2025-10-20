"""
Remote execution handler for Genie server.

Receives tensors and operations, executes them, returns results.
"""

import torch
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class RemoteExecutor:
    """Executes operations on received tensors."""

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')

    async def execute(
        self,
        operation: str,
        tensor: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Execute operation on tensor.

        Args:
            operation: Operation name (e.g., 'relu', 'matmul')
            tensor: Input tensor
            **kwargs: Operation-specific arguments

        Returns:
            Result tensor
        """
        # Move tensor to GPU
        gpu_tensor = tensor.to(self.device)

        # Execute operation based on ATen operation name
        if operation == 'aten::relu':
            result = torch.relu(gpu_tensor)
        elif operation == 'aten::matmul':
            other = kwargs.get('other')
            if other is None:
                raise ValueError("matmul requires 'other' tensor")
            other_gpu = other.to(self.device)
            result = torch.matmul(gpu_tensor, other_gpu)
        elif operation == 'aten::add':
            other = kwargs.get('other')
            if other is None:
                raise ValueError("add requires 'other' tensor or scalar")
            if isinstance(other, torch.Tensor):
                other_gpu = other.to(self.device)
                result = torch.add(gpu_tensor, other_gpu)
            else:
                result = torch.add(gpu_tensor, other)
        elif operation == 'aten::sum':
            dim = kwargs.get('dim')
            keepdim = kwargs.get('keepdim', False)
            result = torch.sum(gpu_tensor, dim=dim, keepdim=keepdim)
        elif operation == 'aten::mean':
            dim = kwargs.get('dim')
            keepdim = kwargs.get('keepdim', False)
            result = torch.mean(gpu_tensor, dim=dim, keepdim=keepdim)
        elif operation == 'aten::reshape':
            shape = kwargs.get('shape')
            if shape is None:
                raise ValueError("reshape requires 'shape'")
            result = torch.reshape(gpu_tensor, shape)
        elif operation == 'aten::transpose':
            dim0 = kwargs.get('dim0', 0)
            dim1 = kwargs.get('dim1', 1)
            result = torch.transpose(gpu_tensor, dim0, dim1)
        elif operation == 'aten::cat':
            tensors = kwargs.get('tensors', [])
            dim = kwargs.get('dim', 0)
            # Move all tensors to GPU
            gpu_tensors = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in tensors]
            result = torch.cat([gpu_tensor] + gpu_tensors, dim=dim)
        elif operation == 'aten::conv2d':
            weight = kwargs.get('weight')
            bias = kwargs.get('bias')
            stride = kwargs.get('stride', 1)
            padding = kwargs.get('padding', 0)
            dilation = kwargs.get('dilation', 1)
            groups = kwargs.get('groups', 1)

            if weight is None:
                raise ValueError("conv2d requires 'weight'")

            weight_gpu = weight.to(self.device)
            if bias is not None:
                bias_gpu = bias.to(self.device)

            result = torch.nn.functional.conv2d(
                gpu_tensor, weight_gpu, bias_gpu, stride, padding, dilation, groups
            )
        elif operation == 'aten::linear':
            weight = kwargs.get('weight')
            bias = kwargs.get('bias')

            if weight is None:
                raise ValueError("linear requires 'weight'")

            weight_gpu = weight.to(self.device)
            if bias is not None:
                bias_gpu = bias.to(self.device)

            result = torch.nn.functional.linear(gpu_tensor, weight_gpu, bias_gpu)
        else:
            # Fallback: try to execute as native PyTorch operation
            try:
                # This is a simplified fallback - in practice, you'd want a more complete mapping
                result = getattr(torch, operation.replace('aten::', ''))(gpu_tensor, **kwargs)
            except (AttributeError, TypeError) as e:
                raise NotImplementedError(f"Operation {operation} not supported. Error: {e}")

        # Keep result on GPU for potential subsequent operations
        return result

    def get_supported_operations(self) -> list:
        """Return list of supported operations."""
        return [
            'aten::relu', 'aten::matmul', 'aten::add', 'aten::sum', 'aten::mean',
            'aten::reshape', 'aten::transpose', 'aten::cat', 'aten::conv2d', 'aten::linear'
        ]
