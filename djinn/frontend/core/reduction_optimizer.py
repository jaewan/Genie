"""
Reduction Operation Optimizer for Smart Remote Execution

This module implements intelligent placement decisions for reduction operations
to minimize network transfer while maintaining execution correctness.

Key Insight: Reduction operations (argmax, sum, mean, etc.) dramatically
reduce data size (often 1000x+), making remote execution highly beneficial
to avoid transferring large intermediate tensors back to CPU.

Example:
    # Without optimization: Transfer 200MB of logits
    logits = model(input)  # [1, 1024, 50257] on GPU
    tokens = logits.argmax(-1).cpu()  # Transfer 200MB back
    
    # With optimization: Transfer only 4KB of tokens
    # argmax stays remote, only 4KB result transferred
"""

import math
from typing import List, Optional


class RemoteReductionOptimizer:
    """
    Optimizes reduction operations to minimize network transfer.
    
    Uses a cost model to decide whether reduction should execute remotely
    (where data lives) or locally (on CPU).
    """
    
    # Reduction operations that dramatically reduce data size
    REDUCTION_OPERATIONS = {
        'argmax', 'argmin', 'argsort',
        'sum', 'mean', 'std', 'var', 'prod',
        'max', 'min',
        'topk', 'kthvalue', 'mode', 'median',
    }
    
    def __init__(self, network_bandwidth_gbps: float = 10.0):
        """
        Initialize optimizer with network characteristics.
        
        Args:
            network_bandwidth_gbps: Network bandwidth in Gbps (default 10Gbps for 1GbE)
        """
        # Convert Gbps to bytes/sec for calculations
        self.network_bandwidth_bytes_per_sec = network_bandwidth_gbps * 1e9 / 8
    
    def should_execute_remotely(self, operation: str, inputs: List) -> bool:
        """
        Decide whether to execute reduction remotely based on cost model.
        
        Core decision logic:
        - If output is 100x+ smaller than input, execute remotely
        - This amortizes the cost of computation against transfer savings
        - Also requires input to be large enough to matter (>1MB)
        
        Args:
            operation: Operation name (e.g., 'argmax', 'sum')
            inputs: List of input tensors/LazyTensors
        
        Returns:
            True if remote execution is beneficial
        """
        # Only optimize known reduction operations
        if operation not in self.REDUCTION_OPERATIONS:
            return False
        
        if not inputs:
            return False
        
        try:
            # Calculate data sizes
            input_size = self._estimate_input_size(inputs)
            output_size = self._estimate_output_size(operation, inputs)
            
            if input_size == 0 or output_size == 0:
                return False
            
            # Calculate reduction factor
            reduction_factor = input_size / output_size
            
            # Remote execution beneficial if:
            # - Data is reduced by at least 100x (conservative threshold)
            # - Input is large enough to matter (>1MB)
            # - Output is small enough to transfer quickly
            should_remote = (
                reduction_factor > 100.0 and 
                input_size > 1_000_000 and  # At least 1MB input
                output_size < 10_000_000  # Less than 10MB output
            )
            
            return should_remote
        
        except Exception:
            # Conservative: don't optimize if we can't calculate
            return False
    
    def _estimate_input_size(self, inputs: List) -> int:
        """Estimate total input size in bytes."""
        total = 0
        for inp in inputs:
            try:
                if hasattr(inp, 'numel') and hasattr(inp, 'element_size'):
                    # LazyTensor or torch.Tensor
                    total += inp.numel() * inp.element_size()
                elif hasattr(inp, '__len__'):
                    # Sequence - rough estimate
                    total += len(inp) * 8
            except Exception:
                pass
        return total
    
    def _estimate_output_size(self, operation: str, inputs: List) -> int:
        """Estimate output size in bytes based on operation type."""
        if not inputs or not hasattr(inputs[0], 'shape'):
            return 0
        
        input_shape = inputs[0].shape
        element_size = inputs[0].element_size() if hasattr(inputs[0], 'element_size') else 4
        
        try:
            if operation in ['argmax', 'argmin']:
                # Reduces last dimension to indices
                output_shape = list(input_shape)[:-1] if len(input_shape) > 1 else [1]
                return math.prod(output_shape) * 8  # int64 indices
            
            elif operation in ['sum', 'mean', 'std', 'var', 'prod']:
                # Can reduce to scalar or along dimension
                # Conservative: assume scalar reduction
                return element_size
            
            elif operation in ['max', 'min']:
                # Similar to sum - assume scalar
                return element_size
            
            elif operation == 'topk':
                # Returns k elements along last dimension
                k = 10  # Default k value
                output_shape = list(input_shape)[:-1] + [k]
                return math.prod(output_shape) * element_size
            
            elif operation in ['mode', 'median']:
                # Returns single value
                return element_size
            
            # Conservative: assume same size as input
            return math.prod(input_shape) * element_size
        
        except Exception:
            # If calculation fails, be conservative
            return math.prod(input_shape) * element_size if input_shape else 0


# Global instance for use across the system
_reduction_optimizer = RemoteReductionOptimizer()


def should_execute_reduction_remotely(operation: str, inputs: List) -> bool:
    """
    Determine if a reduction operation should execute remotely.
    
    Public API for checking reduction optimization.
    """
    return _reduction_optimizer.should_execute_remotely(operation, inputs)

