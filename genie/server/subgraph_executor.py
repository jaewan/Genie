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
        self.operation_registry = self._build_operation_registry()

        # Statistics
        self.stats = {
            'subgraphs_executed': 0,
            'operations_executed': 0,
            'total_time_seconds': 0.0,
            'memory_peak_mb': 0.0,
            'errors': []
        }

        logger.info(f"SubgraphExecutor initialized on GPU {gpu_id}")

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

            context = ExecutionContext(
                device=self.device,
                tensors={},
                timeout=timeout
            )

            # Step 1: Move inputs to GPU
            logger.debug(f"Moving {len(input_data)} inputs to GPU {self.gpu_id}")
            for tensor_id_str, tensor_data in input_data.items():
                tensor_id = int(tensor_id_str)
                context.tensors[tensor_id] = tensor_data.to(self.device)

            # Step 2: Execute operations in topological order
            logger.debug(f"Executing {len(subgraph_request['operations'])} operations")
            for op_spec in subgraph_request['operations']:
                self.stats['operations_executed'] += 1

                result = self._execute_operation(op_spec, context)
                context.tensors[op_spec['op_id']] = result

                # Check memory usage
                if torch.cuda.is_available():
                    memory_mb = torch.cuda.memory_allocated(self.gpu_id) / 1024 / 1024
                    self.stats['memory_peak_mb'] = max(self.stats['memory_peak_mb'], memory_mb)
                    logger.debug(f"GPU memory: {memory_mb:.1f}MB")

            # Step 3: Transfer ONLY final output to CPU
            output_id = subgraph_request['output_id']
            final_result = context.tensors[output_id]

            logger.debug(f"Transferring final result: {final_result.shape} to CPU")

            # Move to CPU and detach from computation graph for transfer
            result_cpu = final_result.cpu().detach()

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
        kwargs = op_spec.get('kwargs', {})

        # Get input tensors from context (already on GPU)
        inputs = [context.tensors[inp_id] for inp_id in input_ids]

        # Execute operation
        op_func = self.operation_registry.get(operation)
        if op_func:
            result = op_func(inputs, kwargs)
        else:
            # Fallback to dynamic dispatch
            result = self._execute_dynamic(operation, inputs, kwargs)

        return result  # Result stays on GPU!

    def _build_operation_registry(self) -> Dict[str, Callable]:
        """Build operation registry for fast dispatch."""
        return {
            # Arithmetic operations
            'aten::add': lambda inputs, kwargs: torch.add(inputs[0], inputs[1], **kwargs),
            'aten::sub': lambda inputs, kwargs: torch.sub(inputs[0], inputs[1], **kwargs),
            'aten::mul': lambda inputs, kwargs: torch.mul(inputs[0], inputs[1], **kwargs),
            'aten::div': lambda inputs, kwargs: torch.div(inputs[0], inputs[1], **kwargs),

            # Linear algebra
            'aten::matmul': lambda inputs, kwargs: torch.matmul(inputs[0], inputs[1]),
            'aten::mm': lambda inputs, kwargs: torch.mm(inputs[0], inputs[1]),
            'aten::bmm': lambda inputs, kwargs: torch.bmm(inputs[0], inputs[1]),

            # Transpose operations
            'aten::t': lambda inputs, kwargs: torch.t(inputs[0]),
            'aten::transpose': lambda inputs, kwargs: torch.transpose(inputs[0], **kwargs),

            # Activations
            'aten::relu': lambda inputs, kwargs: torch.relu(inputs[0]),
            'aten::sigmoid': lambda inputs, kwargs: torch.sigmoid(inputs[0]),
            'aten::tanh': lambda inputs, kwargs: torch.tanh(inputs[0]),
            'aten::gelu': lambda inputs, kwargs: torch.nn.functional.gelu(inputs[0]),
            'aten::leaky_relu': lambda inputs, kwargs: torch.nn.functional.leaky_relu(inputs[0], **kwargs),
            'aten::elu': lambda inputs, kwargs: torch.nn.functional.elu(inputs[0], **kwargs),

            # Element-wise operations
            'aten::abs': lambda inputs, kwargs: torch.abs(inputs[0]),
            'aten::neg': lambda inputs, kwargs: torch.neg(inputs[0]),
            'aten::exp': lambda inputs, kwargs: torch.exp(inputs[0]),
            'aten::log': lambda inputs, kwargs: torch.log(inputs[0]),
            'aten::sqrt': lambda inputs, kwargs: torch.sqrt(inputs[0]),
            'aten::square': lambda inputs, kwargs: torch.square(inputs[0]),

            # Reduction operations
            'aten::sum': lambda inputs, kwargs: torch.sum(inputs[0], **kwargs),
            'aten::mean': lambda inputs, kwargs: torch.mean(inputs[0], **kwargs),
            'aten::softmax': lambda inputs, kwargs: torch.softmax(inputs[0], **kwargs),
            'aten::log_softmax': lambda inputs, kwargs: torch.log_softmax(inputs[0], **kwargs),

            # Convolution
            'aten::conv2d': lambda inputs, kwargs: torch.ops.aten.conv2d(*inputs, **kwargs),

            # Pooling
            'aten::max_pool2d': lambda inputs, kwargs: torch.ops.aten.max_pool2d(inputs[0], **kwargs),
            'aten::avg_pool2d': lambda inputs, kwargs: torch.ops.aten.avg_pool2d(inputs[0], **kwargs),

            # Adaptive pooling
            'aten::adaptive_avg_pool2d': lambda inputs, kwargs: torch.nn.functional.adaptive_avg_pool2d(inputs[0], **kwargs),
            'aten::adaptive_max_pool2d': lambda inputs, kwargs: torch.nn.functional.adaptive_max_pool2d(inputs[0], **kwargs),

            # Normalization
            'aten::batch_norm': lambda inputs, kwargs: torch.ops.aten.batch_norm(*inputs, **kwargs),
            'aten::layer_norm': lambda inputs, kwargs: torch.nn.functional.layer_norm(inputs[0], **kwargs),

            # Dropout
            'aten::dropout': lambda inputs, kwargs: torch.nn.functional.dropout(inputs[0], **kwargs),

            # Interpolation
            'aten::interpolate': lambda inputs, kwargs: torch.nn.functional.interpolate(inputs[0], **kwargs),

            # Tensor manipulation
            'aten::split': lambda inputs, kwargs: torch.split(*inputs, **kwargs),

            # Special operations
            'aten::alias': lambda inputs, kwargs: inputs[0],  # Pass-through
            'aten::clone': lambda inputs, kwargs: inputs[0].clone(),
            'aten::detach': lambda inputs, kwargs: inputs[0].detach(),

            # Device operations (should not appear in subgraphs, but handle gracefully)
            'aten::cpu': lambda inputs, kwargs: inputs[0].cpu(),
            'aten::cuda': lambda inputs, kwargs: inputs[0].cuda(**kwargs),
        }

    def _execute_dynamic(self, operation: str, inputs: List, kwargs: Dict) -> torch.Tensor:
        """Fallback: dynamic operation dispatch."""
        # Normalize operation name
        op_name = operation.replace('aten::', '')

        # Try torch namespace first
        try:
            torch_func = getattr(torch, op_name, None)
            if torch_func:
                return torch_func(*inputs, **kwargs)
        except Exception as e:
            logger.debug(f"torch.{op_name} failed: {e}")

        # Try torch.nn.functional
        try:
            import torch.nn.functional as F
            func = getattr(F, op_name, None)
            if func:
                return func(*inputs, **kwargs)
        except Exception as e:
            logger.debug(f"torch.nn.functional.{op_name} failed: {e}")

        # Try torch.ops.aten
        try:
            aten_func = getattr(torch.ops.aten, op_name, None)
            if aten_func:
                return aten_func(*inputs, **kwargs)
        except Exception as e:
            logger.debug(f"torch.ops.aten.{op_name} failed: {e}")

        # Last resort: raise error
        raise NotImplementedError(
            f"Operation '{operation}' not supported in subgraph execution. "
            f"Supported operations: {list(self.operation_registry.keys())}"
        )

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
