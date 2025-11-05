"""
Unit tests for subgraph execution optimization.

Tests the core optimization described in the network enhancement plan:
extracting computation subgraphs and executing them remotely in a single request
instead of per-operation execution.
"""

import pytest
import torch
import os
import json
import tempfile
from unittest.mock import Mock, patch

import djinn
from djinn.core.subgraph_builder import SubgraphBuilder, RemoteSubgraph
from djinn.frontend.core.lazy_tensor import LazyTensor
from djinn.server.subgraph_executor import SubgraphExecutor, ExecutionContext


class TestSubgraphBuilder:
    """Test SubgraphBuilder functionality."""

    def setup_method(self):
        """Set up test environment."""
        # Reset feature flag
        os.environ.pop('GENIE_SUBGRAPH_OPT', None)

    def test_simple_chain_subgraph(self):
        """Test subgraph extraction for simple operation chain."""
        with genie.capture():
            x = torch.randn(10, 10, device='remote_accelerator:0')
            y = x @ x
            z = torch.relu(y)

        builder = SubgraphBuilder()
        subgraph = builder.build_from_device_chain(z)

        assert subgraph is not None
        assert len(subgraph.operations) == 2  # matmul, relu
        assert len(subgraph.input_tensors) == 1  # x (factory op)
        assert subgraph.output_tensor.operation == 'aten::relu'

    def test_topological_order(self):
        """Verify operations are in correct execution order."""
        with genie.capture():
            x = torch.randn(10, 10, device='remote_accelerator:0')
            y = x + 1
            z = x * 2
            result = y + z

        builder = SubgraphBuilder()
        subgraph = builder.build_from_device_chain(result)

        assert subgraph is not None

        # Verify dependencies are satisfied
        # Include external inputs in seen_ops since they're already available
        seen_ops = set(id(tensor) for tensor in subgraph.input_tensors.values())
        for op in subgraph.operations:
            for inp in op.inputs:
                if isinstance(inp, LazyTensor):
                    # Check if it's an external input or already processed operation
                    if id(inp) not in seen_ops:
                        assert id(inp) in seen_ops, f"Dependency not satisfied! {inp.operation} not seen before {op.operation}"
            seen_ops.add(id(op))

    def test_serialization(self):
        """Test subgraph serialization/deserialization."""
        with genie.capture():
            x = torch.randn(10, 10, device='remote_accelerator:0')
            y = x @ x

        builder = SubgraphBuilder()
        subgraph = builder.build_from_device_chain(y)

        # Serialize
        serialized = subgraph.serialize()

        # Verify structure
        assert 'operations' in serialized
        assert 'input_tensors' in serialized
        assert 'output_id' in serialized
        assert len(serialized['operations']) == 1  # matmul

        # Verify operation structure
        op = serialized['operations'][0]
        assert 'op_id' in op
        assert 'operation' in op
        assert 'inputs' in op
        assert 'kwargs' in op
        assert 'shape' in op
        assert 'dtype' in op

    def test_external_input_detection(self):
        """Test detection of external inputs (factory operations)."""
        with genie.capture():
            x = torch.randn(10, 10, device='remote_accelerator:0')  # Factory op
            y = torch.zeros(10, 10, device='remote_accelerator:0')  # Factory op
            z = x + y

        builder = SubgraphBuilder()
        subgraph = builder.build_from_device_chain(z)

        assert subgraph is not None
        assert len(subgraph.input_tensors) == 2  # x and y are both factory ops
        assert len(subgraph.operations) == 1  # add operation

    def test_non_remote_chain_ignored(self):
        """Test that non-remote chains are ignored."""
        with genie.capture():
            x = torch.randn(10, 10)  # Local tensor
            y = x @ x
            z = torch.relu(y)

        builder = SubgraphBuilder()
        subgraph = builder.build_from_device_chain(z)

        assert subgraph is None  # Should not build subgraph for local tensors

    def test_complex_chain(self):
        """Test complex operation chain."""
        with genie.capture():
            x = torch.randn(32, 64, device='remote_accelerator:0')
            y = x @ x.t()
            z = torch.softmax(y, dim=-1)
            w = torch.relu(z)
            result = w.sum(dim=-1)

        builder = SubgraphBuilder()
        subgraph = builder.build_from_device_chain(result)

        assert subgraph is not None
        assert len(subgraph.operations) == 5  # transpose, matmul, softmax, relu, sum
        assert len(subgraph.input_tensors) == 1  # x (factory op)

        # Verify topological order
        operations = [op.operation for op in subgraph.operations]
        # transpose should come before matmul, matmul before softmax
        transpose_idx = operations.index('aten::t')
        matmul_idx = operations.index('aten::matmul')
        softmax_idx = operations.index('aten::softmax')
        assert transpose_idx < matmul_idx < softmax_idx


class TestSubgraphExecutor:
    """Test SubgraphExecutor functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.executor = SubgraphExecutor(gpu_id=0)

    def test_executor_initialization(self):
        """Test executor initialization."""
        assert self.executor.gpu_id == 0
        assert self.executor.device == torch.device('cuda:0')
        assert len(self.executor.operation_registry) > 0
        assert 'aten::matmul' in self.executor.operation_registry
        assert 'aten::relu' in self.executor.operation_registry

    def test_simple_execution(self):
        """Test simple subgraph execution."""
        # Create test subgraph request
        subgraph_request = {
            'operations': [
                {
                    'op_id': 1,
                    'operation': 'aten::relu',
                    'inputs': [0],
                    'kwargs': {},
                    'shape': [10, 10],
                    'dtype': 'torch.float32'
                }
            ],
            'input_tensors': {
                '0': {
                    'shape': [10, 10],
                    'dtype': 'torch.float32'
                }
            },
            'output_id': 1
        }

        # Create input data
        input_data = {
            '0': torch.randn(10, 10)
        }

        # Execute
        result = self.executor.execute(subgraph_request, input_data)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 10)
        assert result.device.type == 'cpu'

    def test_matmul_execution(self):
        """Test matrix multiplication execution."""
        subgraph_request = {
            'operations': [
                {
                    'op_id': 1,
                    'operation': 'aten::matmul',
                    'inputs': [0, 1],
                    'kwargs': {},
                    'shape': [10, 5],
                    'dtype': 'torch.float32'
                }
            ],
            'input_tensors': {
                '0': {
                    'shape': [10, 10],
                    'dtype': 'torch.float32'
                },
                '1': {
                    'shape': [10, 5],
                    'dtype': 'torch.float32'
                }
            },
            'output_id': 1
        }

        input_data = {
            '0': torch.randn(10, 10),
            '1': torch.randn(10, 5)
        }

        result = self.executor.execute(subgraph_request, input_data)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 5)

    def test_operation_registry_coverage(self):
        """Test that operation registry covers key operations."""
        # Test that all important operations are registered
        important_ops = [
            'aten::add', 'aten::sub', 'aten::mul', 'aten::div',
            'aten::matmul', 'aten::relu', 'aten::sigmoid', 'aten::tanh',
            'aten::softmax', 'aten::sum', 'aten::mean'
        ]

        for op in important_ops:
            assert op in self.executor.operation_registry, f"Missing operation: {op}"

    def test_statistics_tracking(self):
        """Test statistics tracking."""
        initial_stats = self.executor.get_stats()

        # Execute a subgraph
        subgraph_request = {
            'operations': [
                {
                    'op_id': 1,
                    'operation': 'aten::relu',
                    'inputs': [0],
                    'kwargs': {},
                    'shape': [5, 5],
                    'dtype': 'torch.float32'
                }
            ],
            'input_tensors': {
                '0': {
                    'shape': [5, 5],
                    'dtype': 'torch.float32'
                }
            },
            'output_id': 1
        }

        input_data = {'0': torch.randn(5, 5)}
        result = self.executor.execute(subgraph_request, input_data)

        # Check statistics
        final_stats = self.executor.get_stats()
        assert final_stats['subgraphs_executed'] == initial_stats['subgraphs_executed'] + 1
        assert final_stats['operations_executed'] == initial_stats['operations_executed'] + 1
        assert final_stats['total_time_seconds'] > initial_stats['total_time_seconds']


class TestSubgraphIntegration:
    """Test integration between components."""

    def setup_method(self):
        """Set up test environment."""
        # Disable subgraph optimization for these tests
        os.environ['GENIE_SUBGRAPH_OPT'] = '0'

    def test_end_to_end_simple_chain(self):
        """Test end-to-end subgraph execution for simple chain."""
        # Enable subgraph optimization
        os.environ['GENIE_SUBGRAPH_OPT'] = '1'

        with genie.capture():
            x = torch.randn(10, 10, device='remote_accelerator:0')
            y = x @ x
            z = torch.relu(y)
            result = z.cpu()

        # Should work without errors
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 10)

    def test_fallback_on_subgraph_failure(self):
        """Test fallback to recursive execution when subgraph fails."""
        # Enable subgraph optimization but force failure
        os.environ['GENIE_SUBGRAPH_OPT'] = '1'

        # Mock the client to raise an exception
        with patch('genie.runtime.simple_client.get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.execute_subgraph.side_effect = RuntimeError("Network error")
            mock_get_client.return_value = mock_client

            with genie.capture():
                x = torch.randn(10, 10, device='remote_accelerator:0')
                y = x @ x
                result = y.cpu()

            # Should still work via fallback
            assert isinstance(result, torch.Tensor)
            assert result.shape == (10, 10)

    def test_subgraph_enabled_by_default(self):
        """Test that subgraph optimization is enabled by default."""
        # Don't set the environment variable (should use default: enabled)

        with genie.capture():
            x = torch.randn(10, 10, device='remote_accelerator:0')
            y = x @ x
            result = y.cpu()

        # Should work with subgraph optimization enabled by default
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 10)

    def test_subgraph_serialization_compatibility(self):
        """Test that subgraph serialization produces valid JSON."""
        with genie.capture():
            x = torch.randn(5, 5, device='remote_accelerator:0')
            y = x + x
            z = torch.relu(y)

        builder = SubgraphBuilder()
        subgraph = builder.build_from_device_chain(z)

        # Serialize and verify it's valid JSON
        serialized = subgraph.serialize()
        json_str = json.dumps(serialized)
        deserialized = json.loads(json_str)

        # Verify structure is preserved
        assert len(deserialized['operations']) == len(serialized['operations'])
        assert len(deserialized['input_tensors']) == len(serialized['input_tensors'])
        assert deserialized['output_id'] == serialized['output_id']


class TestSubgraphExecutorErrors:
    """Test error handling in subgraph execution."""

    def setup_method(self):
        """Set up test environment."""
        self.executor = SubgraphExecutor(gpu_id=0)

    def test_unsupported_operation_error(self):
        """Test handling of unsupported operations."""
        subgraph_request = {
            'operations': [
                {
                    'op_id': 1,
                    'operation': 'aten::unsupported_op',
                    'inputs': [0],
                    'kwargs': {},
                    'shape': [10, 10],
                    'dtype': 'torch.float32'
                }
            ],
            'input_tensors': {
                '0': {
                    'shape': [10, 10],
                    'dtype': 'torch.float32'
                }
            },
            'output_id': 1
        }

        input_data = {'0': torch.randn(10, 10)}

        # Should raise NotImplementedError for unsupported operation
        with pytest.raises(NotImplementedError, match="Operation 'aten::unsupported_op' not supported"):
            self.executor.execute(subgraph_request, input_data)

    def test_missing_input_error(self):
        """Test handling of missing input tensors."""
        subgraph_request = {
            'operations': [
                {
                    'op_id': 1,
                    'operation': 'aten::relu',
                    'inputs': [999],  # Non-existent input
                    'kwargs': {},
                    'shape': [10, 10],
                    'dtype': 'torch.float32'
                }
            ],
            'input_tensors': {},
            'output_id': 1
        }

        input_data = {}

        # Should raise KeyError for missing input
        with pytest.raises(KeyError):
            self.executor.execute(subgraph_request, input_data)
