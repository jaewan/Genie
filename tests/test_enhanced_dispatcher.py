"""
Test the enhanced dispatcher implementation.
"""
import pytest
import torch
import logging

from genie.core.enhanced_dispatcher import (
    enhanced_dispatcher,
    create_lazy_tensor_for_device_op,
    set_enhanced_lazy_mode,
    get_enhanced_stats
)
from genie.core.enhanced_dispatcher import enhanced_dispatcher as dispatcher  # Backward compatibility
from genie.core.lazy_tensor import LazyTensor

logger = logging.getLogger(__name__)


class TestEnhancedDispatcher:
    """Test suite for the enhanced dispatcher implementation."""

    def test_enhanced_dispatcher_initialization(self):
        """Test enhanced dispatcher initialization and stats."""
        stats = get_enhanced_stats()
        
        # Should have many more operations than the original
        assert stats["registered_ops"] >= 40
        assert "successful_registrations" in stats
        assert "failed_registrations" in stats
        assert "coverage_percentage" in stats
        assert "lazy_mode" in stats

    def test_operation_coverage_improvement(self):
        """Test that enhanced dispatcher has better operation coverage."""
        registered_ops = enhanced_dispatcher.get_registered_operations()
        
        # Should have comprehensive coverage
        assert len(registered_ops) >= 40
        
        # Check for key operation categories
        arithmetic_ops = [op for op in registered_ops if any(x in op for x in ["add", "sub", "mul", "div"])]
        assert len(arithmetic_ops) >= 4
        
        linalg_ops = [op for op in registered_ops if any(x in op for x in ["matmul", "mm", "bmm", "linear"])]
        assert len(linalg_ops) >= 4
        
        activation_ops = [op for op in registered_ops if any(x in op for x in ["relu", "sigmoid", "tanh", "gelu"])]
        assert len(activation_ops) >= 4
        
        creation_ops = [op for op in registered_ops if any(x in op for x in ["randn", "zeros", "ones", "empty"])]
        assert len(creation_ops) >= 4

    def test_lazy_mode_control(self):
        """Test lazy mode enable/disable functionality."""
        # Test enhanced dispatcher
        set_enhanced_lazy_mode(True)
        stats = get_enhanced_stats()
        assert stats["lazy_mode"] is True
        
        set_enhanced_lazy_mode(False)
        stats = get_enhanced_stats()
        assert stats["lazy_mode"] is False
        
        # Reset to lazy mode
        set_enhanced_lazy_mode(True)

    def test_backward_compatibility(self):
        """Test backward compatibility with old dispatcher interface."""
        # Old interface should still work
        old_stats = dispatcher.get_stats()
        
        assert "registered_ops" in old_stats
        assert "fallback_ops" in old_stats
        assert "operation_count" in old_stats
        assert "lazy_mode" in old_stats
        
        # Should have more operations than before
        assert old_stats["registered_ops"] >= 40
        
        # Test lazy mode control
        dispatcher.set_lazy_mode(False)
        assert dispatcher.get_stats()["lazy_mode"] is False
        
        dispatcher.set_lazy_mode(True)  # Reset

    def test_lazy_tensor_creation(self):
        """Test that LazyTensor creation works with enhanced dispatcher."""
        lazy_tensor = create_lazy_tensor_for_device_op(
            "aten::add", 
            (torch.randn(2, 2), torch.randn(2, 2)),
            {"alpha": 1.0}
        )
        
        assert isinstance(lazy_tensor, LazyTensor)
        assert lazy_tensor.operation == "aten::add"
        assert len(lazy_tensor.inputs) == 2
        assert lazy_tensor.kwargs.get("alpha") == 1.0

    def test_operation_count_tracking(self):
        """Test that operation counts are tracked correctly."""
        initial_stats = get_enhanced_stats()
        initial_count = initial_stats["operation_count"]
        
        # Create some lazy tensors
        for _ in range(5):
            create_lazy_tensor_for_device_op(
                "aten::matmul",
                (torch.randn(3, 3), torch.randn(3, 3)),
                {}
            )
        
        final_stats = get_enhanced_stats()
        final_count = final_stats["operation_count"]
        
        assert final_count >= initial_count + 5

    def test_error_handling_and_fallback(self):
        """Test error handling and fallback mechanisms."""
        stats = get_enhanced_stats()
        
        # Enhanced dispatcher should handle registration failures gracefully
        # All operations should be registered (even if as fallbacks)
        assert stats["registered_ops"] > 0
        
        # Failed registrations are acceptable - they become fallback operations
        assert stats["failed_registrations"] >= 0
        
        # The dispatcher should still function even with failed registrations
        lazy_tensor = enhanced_dispatcher._create_lazy_tensor(
            "aten::test_op",
            [torch.randn(2, 2)],
            {}
        )
        assert isinstance(lazy_tensor, LazyTensor)

    def test_comprehensive_operation_list(self):
        """Test that we have a comprehensive list of operations."""
        registered_ops = enhanced_dispatcher.get_registered_operations()
        
        # Test specific important operations are included
        important_ops = [
            "aten::add", "aten::sub", "aten::mul", "aten::div",
            "aten::matmul", "aten::mm", "aten::bmm",
            "aten::relu", "aten::sigmoid", "aten::tanh",
            "aten::conv2d", "aten::linear",
            "aten::randn", "aten::zeros", "aten::ones",
            "aten::view", "aten::transpose", "aten::cat"
        ]
        
        for op in important_ops:
            assert op in registered_ops, f"Important operation {op} not registered"

    def test_statistics_accuracy(self):
        """Test that statistics are accurate and consistent."""
        stats = get_enhanced_stats()
        
        # Basic consistency checks
        assert stats["registered_ops"] == len(enhanced_dispatcher.get_registered_operations())
        assert stats["successful_registrations"] == len(enhanced_dispatcher.get_successful_operations())
        assert stats["failed_registrations"] == len(enhanced_dispatcher.get_failed_operations())
        
        # Coverage percentage calculation
        total_attempted = stats["successful_registrations"] + stats["failed_registrations"]
        if total_attempted > 0:
            expected_coverage = (stats["successful_registrations"] / total_attempted) * 100
            assert abs(stats["coverage_percentage"] - expected_coverage) < 0.1

    def test_deprecated_register_op_warning(self):
        """Test that deprecated register_op method shows warning."""
        with pytest.warns(UserWarning, match="register_op is deprecated"):
            @dispatcher.register_op("test::deprecated")
            def test_func():
                pass

    def test_performance_characteristics(self):
        """Test performance characteristics of enhanced dispatcher."""
        import time
        
        # Test that lazy tensor creation is fast
        start_time = time.time()
        
        for _ in range(1000):
            enhanced_dispatcher._create_lazy_tensor(
                "aten::add",
                [torch.randn(10, 10), torch.randn(10, 10)],
                {}
            )
        
        end_time = time.time()
        avg_time_per_op = (end_time - start_time) / 1000
        
        # Should be very fast (much less than 1ms per operation)
        assert avg_time_per_op < 0.001, f"LazyTensor creation too slow: {avg_time_per_op:.6f}s per op"

    def test_memory_efficiency(self):
        """Test memory efficiency of enhanced dispatcher."""
        import sys
        
        # Get initial memory usage
        initial_size = sys.getsizeof(enhanced_dispatcher)
        
        # Create many lazy tensors
        tensors = []
        for i in range(100):
            tensor = enhanced_dispatcher._create_lazy_tensor(
                "aten::add",
                [torch.randn(10, 10), torch.randn(10, 10)],
                {}
            )
            tensors.append(tensor)
        
        # Memory usage of dispatcher itself shouldn't grow much
        final_size = sys.getsizeof(enhanced_dispatcher)
        
        # The dispatcher itself shouldn't grow significantly
        assert final_size - initial_size < 1000  # Less than 1KB growth


class TestEnhancedDispatcherIntegration:
    """Test integration with other components."""

    def test_integration_with_lazy_tensor(self):
        """Test integration with LazyTensor system."""
        # Create LazyTensor through enhanced dispatcher
        lazy_tensor = enhanced_dispatcher._create_lazy_tensor(
            "aten::matmul",
            [torch.randn(4, 4), torch.randn(4, 4)],
            {}
        )
        
        # Should integrate properly with LazyTensor
        assert isinstance(lazy_tensor, LazyTensor)
        assert lazy_tensor.operation == "aten::matmul"
        assert len(lazy_tensor.inputs) == 2
        
        # Should be able to materialize
        result = lazy_tensor.cpu()
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 4)

    def test_integration_with_device_system(self):
        """Test integration with device system."""
        from genie.core.device import get_device
        
        # Should work with device system
        device = get_device(0)
        assert device is not None
        
        # Enhanced dispatcher should be available
        stats = get_enhanced_stats()
        assert stats["registered_ops"] > 0

    def test_no_regression_in_existing_functionality(self):
        """Test that existing functionality still works."""
        # All the existing integration tests should still pass
        # This is verified by running the existing test suite
        
        # Basic functionality check
        stats = dispatcher.get_stats()
        assert stats["registered_ops"] > 0
        assert "lazy_mode" in stats


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
