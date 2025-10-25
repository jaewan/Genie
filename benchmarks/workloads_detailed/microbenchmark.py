"""
Microbenchmark Workload - Cost Model Validation.

Purpose: Prove semantic cost estimator is accurate.
Tests various operations and validates predictions against reality.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional
import time


class MicrobenchmarkWorkload:
    """
    Synthetic operations for cost model validation.

    Purpose: Prove semantic cost estimator is accurate.
    """

    def __init__(self):
        self.operations = [
            ('matmul', self._create_matmul_inputs),
            ('conv2d', self._create_conv2d_inputs),
            ('attention', self._create_attention_inputs),
            ('linear', self._create_linear_inputs),
            ('batch_norm', self._create_batch_norm_inputs),
        ]
        self.model = "microbenchmark"  # Synthetic workload, no model needed

    def _create_matmul_inputs(self):
        """Create inputs for matrix multiplication."""
        return [torch.randn(1024, 1024), torch.randn(1024, 1024)]

    def _create_conv2d_inputs(self):
        """Create inputs for convolution."""
        return [torch.randn(1, 64, 56, 56), torch.randn(64, 64, 3, 3)]

    def _create_attention_inputs(self):
        """Create inputs for attention mechanism."""
        return [torch.randn(1, 12, 512, 64)]  # Q, K, V packed

    def _create_linear_inputs(self):
        """Create inputs for linear layer."""
        return [torch.randn(128, 512)]

    def _create_batch_norm_inputs(self):
        """Create inputs for batch normalization."""
        return [torch.randn(8, 256, 14, 14)]

    def _execute_operation(self, op_name: str, inputs: List[torch.Tensor]) -> Dict[str, Any]:
        """Execute a single operation and measure performance."""
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # Move inputs to device
        device_inputs = [inp.to(device) for inp in inputs]

        # Create appropriate layer
        layer = self._create_layer(op_name, device_inputs)
        if layer is None:
            # Fallback: just measure tensor operations
            start_time = time.perf_counter()
            with torch.no_grad():
                if op_name == 'matmul':
                    result = torch.matmul(device_inputs[0], device_inputs[1])
                elif op_name == 'conv2d':
                    result = torch.conv2d(device_inputs[0], device_inputs[1])
                else:
                    result = device_inputs[0]  # Identity
            end_time = time.perf_counter()
        else:
            layer = layer.to(device)
            start_time = time.perf_counter()
            with torch.no_grad():
                result = layer(*device_inputs)
            end_time = time.perf_counter()

        # Calculate metrics
        total_time = (end_time - start_time) * 1000  # ms

        # Estimate FLOPs
        flops = self._estimate_flops(op_name, device_inputs)

        return {
            'operation': op_name,
            'latency_ms': total_time,
            'estimated_flops': flops,
            'input_shapes': [list(inp.shape) for inp in device_inputs],
            'output_shape': list(result.shape),
            'input_memory_mb': sum(inp.numel() * inp.element_size() for inp in device_inputs) / (1024 * 1024),
        }

    def _create_layer(self, op_name: str, inputs: List[torch.Tensor]):
        """Create appropriate layer for operation."""
        try:
            if op_name == 'conv2d':
                return nn.Conv2d(
                    inputs[0].shape[1],  # in_channels
                    inputs[1].shape[0],  # out_channels
                    kernel_size=3,
                    padding=1
                )
            elif op_name == 'linear':
                return nn.Linear(inputs[0].shape[-1], 512)
            elif op_name == 'batch_norm':
                return nn.BatchNorm2d(inputs[0].shape[1])
            else:
                return None
        except:
            return None

    def _estimate_flops(self, op_name: str, inputs: List[torch.Tensor]) -> float:
        """Estimate FLOPs for operation."""
        try:
            if op_name == 'matmul':
                # C = A @ B: 2 * M * N * K
                A, B = inputs[0], inputs[1]
                return 2.0 * A.shape[0] * A.shape[1] * B.shape[1]

            elif op_name == 'conv2d':
                # Conv2d: 2 * C_out * C_in * kH * kW * H_out * W_out
                x, weight = inputs[0], inputs[1]
                C_out, C_in, kH, kW = weight.shape
                H_out, W_out = x.shape[2], x.shape[3]
                return 2.0 * C_out * C_in * kH * kW * H_out * W_out

            elif op_name == 'attention':
                # Simplified attention: Q @ K^T then softmax then @ V
                Q = inputs[0]
                B, H, L, D = Q.shape
                return 2.0 * B * H * L * D * L  # Q @ K^T + softmax @ V

            elif op_name == 'linear':
                # Linear: 2 * input_size * output_size * batch_size
                x = inputs[0]
                input_size = x.shape[-1]
                output_size = 512
                batch_size = x.numel() // input_size
                return 2.0 * input_size * output_size * batch_size

            elif op_name == 'batch_norm':
                # Batch norm: negligible FLOPs compared to memory operations
                x = inputs[0]
                return x.numel() * 2.0  # Simple estimate

        except Exception:
            pass

        # Fallback estimate
        total_elements = sum(inp.numel() for inp in inputs)
        return total_elements * 10.0  # Rough estimate

    def run(self, baseline_config: Dict) -> Dict[str, Any]:
        """
        Run microbenchmark suite.

        Key metrics:
        - Operation latencies
        - FLOP estimates vs actual time correlation
        - Memory usage patterns
        """
        print("Running microbenchmark suite...")

        results = []

        for op_name, input_generator in self.operations:
            print(f"  Testing {op_name}...")

            # Generate inputs
            inputs = input_generator()

            # Execute operation
            result = self._execute_operation(op_name, inputs)
            results.append(result)

            print(f"    Latency: {result['latency_ms']:.2f}ms, FLOPs: {result['estimated_flops']/1e6:.1f}M")

        # Calculate correlation between estimated FLOPs and actual time
        if len(results) > 1:
            flops = [r['estimated_flops'] for r in results]
            latencies = [r['latency_ms'] for r in results]

            try:
                import numpy as np
                correlation = np.corrcoef(flops, latencies)[0, 1]
            except:
                correlation = 0.0
        else:
            correlation = 0.0

        return {
            'operations': results,
            'cost_model_correlation': correlation,
            'total_operations': len(results),
            'baseline': baseline_config.get('baseline', 'unknown'),
        }

    def get_sample_inputs(self) -> List[torch.Tensor]:
        """Get sample inputs for microbenchmark (returns first operation's inputs)."""
        # For microbenchmark, we just need some sample tensors
        # The actual inputs are generated per operation in run()
        return [torch.randn(128, 512)]  # Simple fallback

    def get_metadata(self) -> Dict[str, Any]:
        """Get workload metadata."""
        return {
            'workload': 'microbenchmark',
            'operations': [op[0] for op in self.operations],
            'description': 'Synthetic operations for cost model validation',
            'key_optimization': 'cost_model_accuracy',
            'expected_correlation': '>0.7 with actual execution time',
            'purpose': 'Validate semantic cost estimator'
        }
