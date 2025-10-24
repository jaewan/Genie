"""
Remote execution handler for Genie server.

Receives tensors and operations, executes them, returns results.
"""

import torch
import logging
from typing import Dict, Any, Optional, List
from ..core.operation_registry import get_operation_registry

logger = logging.getLogger(__name__)


class RemoteExecutor:
    """Executes operations on received tensors."""

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.operation_registry = get_operation_registry()

    async def execute(
        self,
        operation: str,
        *tensors
    ) -> torch.Tensor:
        """
        Execute operation on one or more input tensors.

        Args:
            operation: Operation name (e.g., 'aten::add', 'aten::matmul')
            *tensors: Input tensors (1 for unary, 2+ for binary/n-ary operations)

        Returns:
            Result tensor
        """
        if not tensors:
            raise ValueError("At least one tensor required")

        # Move all tensors to GPU
        gpu_tensors = [t.to(self.device) for t in tensors]
        
        # Get primary tensor
        primary = gpu_tensors[0]

        # Execute operation using shared registry
        result = self.operation_registry.execute(operation, gpu_tensors)

        # Move result to CPU for transmission to client
        return result.cpu()

    def get_supported_operations(self) -> list:
        """Return list of supported operations."""
        return self.operation_registry.get_supported_operations()
