"""
Remote execution handler for Genie server.

⚠️  DEPRECATED: Use OptimizationExecutor instead
Receives tensors and operations, executes them, returns results.

This module is kept for backward compatibility only.
All new code should use OptimizationExecutor from optimization_executor.py
which includes:
- TensorRegistry for weight caching
- SRGFusionCompiler for operation grouping
- PerformanceMonitor for metrics collection
"""

import logging
import warnings

logger = logging.getLogger(__name__)

# Issue deprecation warning when this module is imported
warnings.warn(
    "RemoteExecutor is deprecated. Use OptimizationExecutor instead. "
    "OptimizationExecutor includes TensorRegistry, SRGFusionCompiler, and "
    "PerformanceMonitor for better performance and observability.",
    DeprecationWarning,
    stacklevel=2
)


import torch
from typing import Dict, Any, Optional, List
from ..core.operation_registry import get_operation_registry


class RemoteExecutor:
    """
    ⚠️  DEPRECATED: Use OptimizationExecutor instead
    
    Executes operations on received tensors.
    
    This executor is outdated and kept only for backward compatibility.
    Use OptimizationExecutor for better performance:
    - Caches model weights (skip redundant transfers)
    - Groups and fuses operations (reduce kernel overhead)
    - Collects comprehensive metrics
    """

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.operation_registry = get_operation_registry()
        logger.warning(
            "RemoteExecutor is deprecated. Please use OptimizationExecutor instead. "
            "See genie/server/optimization_executor.py for details."
        )

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
