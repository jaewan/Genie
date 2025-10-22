"""
Unit tests for smart fragmentation (Phase 2).

Tests the cost-based graph fragmentation and optimization features
that extend the basic subgraph execution with intelligent decisions
about when and how to fragment computation graphs.
"""

import pytest
import torch
import os
from unittest.mock import Mock, patch

import genie
from genie.core.smart_subgraph_builder import (
    SmartSubgraphBuilder,
    FragmentationConfig,
    CostEstimate,
    SubgraphFragment,
    MemoryEstimator,
    CostCalculator
)
from genie.core.lazy_tensor import LazyTensor


class TestMemoryEstimator:
    """Test memory estimation functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.estimator = MemoryEstimator()

    def test_tensor_memory_estimation(self):
        """Test memory estimation for different tensor shapes and dtypes."""
        # Test float32 tensor
        shape = torch.Size([100, 100])
        dtype = torch.float32
        memory_gb = self.estimator._estimate_tensor_memory_from_shape(shape, dtype)

        expected_elements = 100 * 100  # 10,000
        expected_bytes = expected_elements * 4  # 4 bytes per float32
        expected_gb = expected_bytes / (1024 ** 3)

        assert abs(memory_gb - expected_gb) < 1e-6

    def test_different_dtypes(self):
        """Test memory estimation for different data types."""
        shape = torch.Size([10, 10])

        # float32
        mem_f32 = self.estimator._estimate_tensor_memory_from_shape(shape, torch.float32)

        # float64 (should be 2x)
        mem_f64 = self.estimator._estimate_tensor_memory_from_shape(shape, torch.float64)

        assert mem_f64 == 2 * mem_f32

        # int64 (should be 2x float32 since int64 = 8 bytes)
        mem_i64 = self.estimator._estimate_tensor_memory_from_shape(shape, torch.int64)
        assert mem_i64 == 2 * mem_f32  # int64 = 8 bytes, float32 = 4 bytes

    def test_lazy_tensor_memory(self):
        """Test memory estimation for LazyTensor objects."""
        with genie.capture():
            x = torch.randn(50, 50, device='remote_accelerator:0')

        memory_gb = self.estimator._estimate_tensor_memory(x)
        assert memory_gb > 0
        assert isinstance(memory_gb, float)


class TestCostCalculator:
    """Test cost calculation functionality."""

    def setup_method(self):
        """Set up test environment."""
        config = FragmentationConfig(
            memory_limit_gb=8.0,
            network_gbps=100.0,
            compute_tflops=10.0
        )
        self.calculator = CostCalculator(config)

    def test_matmul_cost_estimation(self):
        """Test cost estimation for matrix multiplication."""
        with genie.capture():
            x = torch.randn(100, 100, device='remote_accelerator:0')
            y = x @ x  # Should be 100x100 @ 100x100 -> 100x100

        # Create cost estimate for the matmul operation
        cost = self.calculator.estimate_operations_cost([y])

        assert cost.compute_cost_ms > 0
        assert cost.memory_usage_gb > 0
        assert cost.total_cost_ms > 0
        assert cost.operations_count == 1
        assert cost.efficiency_score > 0

    def test_transfer_cost_estimation(self):
        """Test network transfer cost estimation."""
        from genie.core.subgraph_builder import RemoteSubgraph

        # Create subgraph with known input/output sizes
        with genie.capture():
            x = torch.randn(100, 100, device='remote_accelerator:0')
            y = x @ x

        subgraph = RemoteSubgraph(
            operations=[y],
            input_tensors={id(x): x},
            output_tensor=y
        )

        transfer_cost = self.calculator._estimate_transfer_cost(subgraph)

        # Should be positive (network transfer takes time)
        assert transfer_cost > 0
        assert isinstance(transfer_cost, float)

    def test_complex_chain_cost(self):
        """Test cost estimation for complex operation chains."""
        with genie.capture():
            x = torch.randn(32, 64, device='remote_accelerator:0')
            y = x @ x.t()  # 32x64 @ 64x32 -> 32x32
            z = torch.relu(y)
            w = z.sum(dim=-1)  # 32x32 -> 32

        operations = [y, z, w]
        cost = self.calculator.estimate_operations_cost(operations)

        # Should have 3 operations
        assert cost.operations_count == 3

        # Total cost should be sum of compute + transfer
        assert cost.total_cost_ms == cost.compute_cost_ms + cost.transfer_cost_ms

        # Memory should be reasonable for this size
        assert cost.memory_usage_gb > 0


class TestSmartSubgraphBuilder:
    """Test smart fragmentation functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.config = FragmentationConfig(
            memory_limit_gb=4.0,  # Low limit to trigger fragmentation
            network_gbps=100.0,
            compute_tflops=10.0,
            fragmentation_threshold=0.8,
            prefer_local_compute=True
        )
        self.builder = SmartSubgraphBuilder(self.config)

    def test_small_subgraph_no_fragmentation(self):
        """Test that small subgraphs don't get fragmented."""
        with genie.capture():
            x = torch.randn(10, 10, device='remote_accelerator:0')
            y = x @ x
            z = torch.relu(y)

        fragments = self.builder.build_with_fragmentation(z)

        # Should not fragment small graphs
        assert len(fragments) == 1
        assert fragments[0].execution_mode in ['local', 'remote']

    def test_fragmentation_decision(self):
        """Test fragmentation decision making."""
        # Create a larger computation that should trigger fragmentation
        with genie.capture():
            # Create a chain that might exceed memory limits
            x = torch.randn(100, 100, device='remote_accelerator:0')
            y = x @ x  # 100x100 -> 100x100
            z = torch.relu(y)
            w = z + z  # Another operation

        fragments = self.builder.build_with_fragmentation(w)

        # Should create multiple fragments or at least analyze the cost
        assert len(fragments) >= 1

        # Each fragment should have proper cost estimates
        for fragment in fragments:
            assert isinstance(fragment.cost_estimate, CostEstimate)
            assert fragment.cost_estimate.compute_cost_ms >= 0
            assert fragment.cost_estimate.memory_usage_gb >= 0

    def test_execution_mode_selection(self):
        """Test execution mode selection (local vs remote)."""
        with genie.capture():
            x = torch.randn(50, 50, device='remote_accelerator:0')
            y = x @ x

        fragments = self.builder.build_with_fragmentation(y)

        for fragment in fragments:
            assert fragment.execution_mode in ['local', 'remote']

            # If memory usage is very high, should prefer remote
            if fragment.cost_estimate.memory_usage_gb > self.config.memory_limit_gb * 0.9:
                assert fragment.execution_mode == 'remote'

    def test_fragment_dependencies(self):
        """Test that fragment dependencies are computed correctly."""
        with genie.capture():
            x = torch.randn(20, 20, device='remote_accelerator:0')
            y = x @ x
            z = torch.relu(y)
            w = z.sum()

        fragments = self.builder.build_with_fragmentation(w)

        # Check that dependencies are properly computed
        for fragment in fragments:
            for dep in fragment.dependencies:
                assert isinstance(dep, SubgraphFragment)

    def test_cost_optimization(self):
        """Test cost-based optimization decisions."""
        # Test with different configurations
        local_config = FragmentationConfig(
            memory_limit_gb=8.0,
            network_gbps=10.0,  # Slow network - prefer local
            compute_tflops=10.0,
            prefer_local_compute=True
        )

        local_builder = SmartSubgraphBuilder(local_config)

        with genie.capture():
            x = torch.randn(50, 50, device='remote_accelerator:0')
            y = x @ x

        fragments = local_builder.build_with_fragmentation(y)

        # With slow network, should prefer local execution
        for fragment in fragments:
            # High transfer cost should lead to local execution
            transfer_ratio = fragment.cost_estimate.transfer_cost_ms / max(fragment.cost_estimate.compute_cost_ms, 1.0)
            if transfer_ratio > 2.0 and local_config.prefer_local_compute:
                assert fragment.execution_mode == 'local'


class TestSmartFragmentationIntegration:
    """Test integration of smart fragmentation with executor."""

    def setup_method(self):
        """Set up test environment."""
        # Disable fragmentation for baseline tests
        os.environ.pop('GENIE_SMART_FRAGMENTATION', None)

    def test_feature_flag_integration(self):
        """Test integration with feature flag system."""
        # Test enabled by default
        with genie.capture():
            x = torch.randn(10, 10, device='remote_accelerator:0')
            y = x @ x
            result = y.cpu()

        assert isinstance(result, torch.Tensor)

        # Test disabled with flag
        os.environ['GENIE_SMART_FRAGMENTATION'] = '0'

        with genie.capture():
            x = torch.randn(10, 10, device='remote_accelerator:0')
            y = x @ x
            result = y.cpu()

        assert isinstance(result, torch.Tensor)

        # Reset environment
        os.environ.pop('GENIE_SMART_FRAGMENTATION', None)

    def test_fallback_behavior(self):
        """Test fallback when smart fragmentation fails."""
        os.environ['GENIE_SMART_FRAGMENTATION'] = '1'

        # Mock fragmentation to raise an exception
        with patch.object(SmartSubgraphBuilder, 'build_with_fragmentation') as mock_fragment:
            mock_fragment.side_effect = RuntimeError("Fragmentation failed")

            with genie.capture():
                x = torch.randn(10, 10, device='remote_accelerator:0')
                y = x @ x
                result = y.cpu()

            # Should still work via fallback
            assert isinstance(result, torch.Tensor)

    def test_configuration_via_environment(self):
        """Test configuration via environment variables."""
        # Set environment variables
        os.environ['GENIE_SMART_FRAGMENTATION'] = '1'
        os.environ['GENIE_MEMORY_LIMIT_GB'] = '2.0'
        os.environ['GENIE_NETWORK_Gbps'] = '50.0'
        os.environ['GENIE_COMPUTE_TFLOPS'] = '5.0'

        with genie.capture():
            x = torch.randn(10, 10, device='remote_accelerator:0')
            y = x @ x
            result = y.cpu()

        assert isinstance(result, torch.Tensor)

        # Clean up
        os.environ.pop('GENIE_SMART_FRAGMENTATION', None)
        os.environ.pop('GENIE_MEMORY_LIMIT_GB', None)
        os.environ.pop('GENIE_NETWORK_Gbps', None)
        os.environ.pop('GENIE_COMPUTE_TFLOPS', None)


class TestFragmentationEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test environment."""
        self.config = FragmentationConfig(memory_limit_gb=1.0)  # Very low limit
        self.builder = SmartSubgraphBuilder(self.config)

    def test_empty_subgraph(self):
        """Test behavior with empty or minimal subgraphs."""
        # This would be an edge case - empty subgraph
        fragments = self.builder.build_with_fragmentation(None)

        # Should handle gracefully
        assert isinstance(fragments, list)

    def test_very_small_operations(self):
        """Test fragmentation with very small operations."""
        with genie.capture():
            x = torch.randn(1, 1, device='remote_accelerator:0')  # Tiny tensor
            y = x + x  # Simple operation

        fragments = self.builder.build_with_fragmentation(y)

        # Should not over-fragment small operations
        assert len(fragments) <= 2

    def test_high_memory_operations(self):
        """Test handling of operations with high memory requirements."""
        with genie.capture():
            # Create operations that might exceed memory limits
            x = torch.randn(200, 200, device='remote_accelerator:0')  # Large tensor
            y = x @ x  # Even larger intermediate

        fragments = self.builder.build_with_fragmentation(y)

        # Should handle large operations gracefully
        assert isinstance(fragments, list)
        assert len(fragments) >= 1

        # High memory operations should prefer remote execution
        for fragment in fragments:
            if fragment.cost_estimate.memory_usage_gb > self.config.memory_limit_gb * 0.8:
                assert fragment.execution_mode == 'remote'


class TestCostEstimationAccuracy:
    """Test accuracy of cost estimation algorithms."""

    def setup_method(self):
        """Set up test environment."""
        self.config = FragmentationConfig(
            memory_limit_gb=8.0,
            network_gbps=100.0,
            compute_tflops=10.0
        )
        self.calculator = CostCalculator(self.config)

    def test_flop_estimation_matmul(self):
        """Test FLOP estimation for matrix multiplication."""
        # Create matmul operation: (M, K) @ (K, N) -> (M, N)
        M, K, N = 100, 200, 50

        with genie.capture():
            x = torch.randn(M, K, device='remote_accelerator:0')
            w = torch.randn(K, N, device='remote_accelerator:0')
            y = x @ w

        flops = self.calculator._estimate_operation_flops(y)

        # Expected FLOPs: 2 * M * N * K
        expected_flops = 2.0 * M * N * K

        # Should be reasonably close (within 10% due to implementation details)
        assert abs(flops - expected_flops) / expected_flops < 0.1

    def test_element_wise_cost(self):
        """Test cost estimation for element-wise operations."""
        with genie.capture():
            x = torch.randn(100, 100, device='remote_accelerator:0')
            y = torch.relu(x)

        flops = self.calculator._estimate_operation_flops(y)

        # Element-wise operations should have FLOPs = number of elements
        expected_elements = 100 * 100
        expected_flops = float(expected_elements)

        # Should be exact for element-wise operations
        assert flops == expected_flops
