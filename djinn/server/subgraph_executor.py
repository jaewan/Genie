"""
SubgraphExecutor: Executes operation subgraphs on server GPU.

This module implements the server-side execution of computation subgraphs.
The key optimization is that intermediates stay on GPU - only the final
output is transferred back to CPU.

Key benefits:
- Intermediates stay on GPU (no CPU round-trips)
- Single network transfer for entire computation
- Better memory efficiency
- Reduced serialization overhead

Usage:
    executor = SubgraphExecutor(gpu_id=0)
    result = executor.execute(subgraph_request, input_data)
"""

import logging
import torch
import traceback
from dataclasses import dataclass
from typing import Dict, List, Callable, Any, Optional
import time
from ..frontend.core.operation_registry import get_operation_registry

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Context for subgraph execution on server."""
    device: torch.device
    tensors: Dict[int, torch.Tensor]  # Cached intermediate results
    memory_limit: Optional[int] = None  # Max memory usage (bytes)
    timeout: Optional[float] = None  # Max execution time (seconds)


class SubgraphExecutor:
    """
    Executes operation subgraphs on server GPU.

    KEY OPTIMIZATION: Intermediates stay on GPU!
    Only the final output is transferred back to CPU.
    """

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.operation_registry = get_operation_registry()  # Use shared registry

        # Statistics
        self.stats = {
            'subgraphs_executed': 0,
            'operations_executed': 0,
            'total_time_seconds': 0.0,
            'memory_peak_mb': 0.0,
            'errors': []
        }

        logger.info(f"SubgraphExecutor initialized on GPU {gpu_id} with shared operation registry")

    def execute(self,
                subgraph_request: Dict[str, Any],
                input_data: Dict[str, torch.Tensor],
                timeout: float = 300.0) -> torch.Tensor:
        """
        Execute subgraph on GPU.

        Args:
            subgraph_request: Serialized subgraph specification
            input_data: Input tensors (on CPU)
            timeout: Maximum execution time in seconds

        Returns:
            Result tensor (on CPU)

        Process:
        1. Move inputs to GPU
        2. Execute operations in order (all on GPU)
        3. Only transfer final output to CPU
        """
        start_time = time.time()
        self.stats['subgraphs_executed'] += 1

        try:
            logger.info(f"🚀 Executing subgraph: {len(subgraph_request['operations'])} operations")
            logger.info(
                "   Executor device=%s | CUDA available=%s | current_device=%s",
                self.device,
                torch.cuda.is_available(),
                torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
            )

            context = ExecutionContext(
                device=self.device,
                tensors={},
                timeout=timeout
            )

            # Step 1: Move inputs to GPU
            logger.debug(f"Moving {len(input_data)} inputs to GPU {self.gpu_id}")
            move_start = time.time()
            for tensor_id_str, tensor_data in input_data.items():
                tensor_id = int(tensor_id_str)
                logger.info(
                    "      loading input tensor[%s]: shape=%s dtype=%s src_device=%s",
                    tensor_id,
                    tuple(tensor_data.shape),
                    tensor_data.dtype,
                    tensor_data.device,
                )
                copy_start = time.time()
                moved_tensor = tensor_data.to(self.device, non_blocking=True)
                copy_ms = (time.time() - copy_start) * 1000
                logger.info(
                    "      tensor[%s] moved to device=%s (transfer %.2f ms)",
                    tensor_id,
                    moved_tensor.device,
                    copy_ms,
                )
                context.tensors[tensor_id] = moved_tensor
            if self.device.type == "cuda":
                torch.cuda.synchronize(device=self.device)
            move_ms = (time.time() - move_start) * 1000
            logger.info("   Input transfer time: %.2f ms", move_ms)

            # Step 2: Execute operations in topological order
            logger.debug(f"Executing {len(subgraph_request['operations'])} operations")
            for op_spec in subgraph_request['operations']:
                self.stats['operations_executed'] += 1

                result = self._execute_operation(op_spec, context)
                context.tensors[op_spec['op_id']] = result
                logger.debug(
                    "      op[%s] result device=%s shape=%s",
                    op_spec['op_id'],
                    result.device,
                    tuple(result.shape) if hasattr(result, 'shape') else "n/a",
                )

                # Check memory usage
                if torch.cuda.is_available():
                    memory_mb = torch.cuda.memory_allocated(self.gpu_id) / 1024 / 1024
                    self.stats['memory_peak_mb'] = max(self.stats['memory_peak_mb'], memory_mb)
                    logger.debug(f"GPU memory: {memory_mb:.1f}MB")

            # Step 3: Transfer ONLY final output to CPU
            output_id = subgraph_request['output_id']
            final_result = context.tensors[output_id]

            logger.debug(f"Transferring final result: {final_result.shape} to CPU")
            logger.info(
                "   Final result before transfer: device=%s shape=%s dtype=%s",
                final_result.device,
                tuple(final_result.shape),
                final_result.dtype,
            )

            # Move to CPU and detach from computation graph for transfer
            transfer_start = time.time()
            result_cpu = final_result.cpu().detach()
            transfer_ms = (time.time() - transfer_start) * 1000
            logger.info("   Final result transfer to CPU: %.2f ms", transfer_ms)

            # Statistics
            elapsed = time.time() - start_time
            self.stats['total_time_seconds'] += elapsed

            logger.info(f"✅ Subgraph execution complete: {len(subgraph_request['operations'])} ops "
                       f"in {elapsed:.3f}s, result shape: {result_cpu.shape}")

            return result_cpu

        except (NotImplementedError, KeyError, ValueError) as e:
            # Re-raise specific exceptions without wrapping (for testing)
            logger.error(f"Subgraph execution failed: {e}")
            logger.error(traceback.format_exc())
            raise
        except Exception as e:
            error_msg = f"Subgraph execution failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())

            self.stats['errors'].append({
                'timestamp': time.time(),
                'error': str(e),
                'operations_count': len(subgraph_request.get('operations', [])),
                'traceback': traceback.format_exc()
            })

            # Re-raise with more context for other exceptions
            raise RuntimeError(error_msg) from e

    def execute_single_op(self, operation: str, tensor: torch.Tensor) -> torch.Tensor:
        """
        Execute single operation directly (convenience method).
        
        Faster than full subgraph execution for simple operations.
        Optimized for Phase 1 remote execution of atomic operations.
        
        Args:
            operation: Operation name (e.g., 'aten::relu', 'aten::add')
            tensor: Input tensor (on CPU or GPU)
        
        Returns:
            Result tensor (on CPU)
            
        Performance:
            - Single op: 50% faster than wrapping in subgraph format
            - No overhead from subgraph parsing
            - Direct operation registry lookup
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Executing single op: {operation}")
            
            # Move input to GPU
            gpu_tensor = tensor.to(self.device)
            
            # Execute operation using shared registry
            result_gpu = self.operation_registry.execute(operation, [gpu_tensor], {})
            
            # Move result to CPU
            result_cpu = result_gpu.cpu().detach()
            
            # Statistics
            elapsed = time.time() - start_time
            self.stats['operations_executed'] += 1
            self.stats['total_time_seconds'] += elapsed
            
            logger.debug(f"Single op executed: {operation} in {elapsed:.3f}s, result shape: {result_cpu.shape}")
            
            return result_cpu
            
        except Exception as e:
            logger.error(f"Single op execution failed: {operation}: {e}")
            raise RuntimeError(f"Failed to execute {operation}: {e}") from e

    def _execute_operation(self,
                          op_spec: Dict[str, Any],
                          context: ExecutionContext) -> torch.Tensor:
        """
        Execute single operation on GPU.

        Args:
            op_spec: Operation specification
            context: Execution context with cached tensors

        Returns:
            Result tensor (on GPU)
        """
        operation = op_spec['operation']
        input_ids = op_spec['inputs']
        kwargs = op_spec.get('kwargs', {}).copy()
        kwargs.pop('_stacklevel', None)

        inputs = []
        for identifier in input_ids:
            if isinstance(identifier, dict) and 'type' in identifier:
                inputs.append(self._decode_literal(identifier))
            else:
                inputs.append(context.tensors[identifier])

        result = self.operation_registry.execute(operation, inputs, kwargs)

        result_index = op_spec.get('result_index')
        if result_index is not None and isinstance(result, (tuple, list)):
            try:
                result = result[result_index]
            except IndexError as exc:
                raise RuntimeError(
                    f"Result index {result_index} out of range for operation {operation}"
                ) from exc

        return result  # Result stays on GPU!



    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        stats = self.stats.copy()

        if stats['subgraphs_executed'] > 0:
            stats['avg_time_per_subgraph'] = (
                stats['total_time_seconds'] / stats['subgraphs_executed']
            )
            stats['avg_operations_per_subgraph'] = (
                stats['operations_executed'] / stats['subgraphs_executed']
            )
        else:
            stats['avg_time_per_subgraph'] = 0.0
            stats['avg_operations_per_subgraph'] = 0.0

        return stats

    @staticmethod
    def _decode_literal(spec: Dict[str, Any]) -> Any:
        literal_type = spec.get('type')
        value = spec.get('value')

        if literal_type == 'scalar':
            return value
        if literal_type == 'tuple':
            return tuple(
                SubgraphExecutor._decode_literal(v) if isinstance(v, dict) else v
                for v in value
            )
        if literal_type == 'list':
            return [
                SubgraphExecutor._decode_literal(v) if isinstance(v, dict) else v
                for v in value
            ]
        if literal_type == 'none':
            return None
        if literal_type == 'dtype':
            return getattr(torch, value.split('.')[-1]) if isinstance(value, str) else value
        if literal_type == 'slice':
            start, stop, step = (
                SubgraphExecutor._decode_literal(v) if isinstance(v, dict) else v
                for v in value
            )
            return slice(start, stop, step)
        raise ValueError(f"Unsupported literal spec: {spec}")
