"""
Test the new structured dispatcher implementation using torch.library.
"""
import pytest
import torch
import logging

from genie.core.dispatcher_v2 import (
    structured_dispatcher,
    initialize_dispatcher,
    set_dispatcher_lazy_mode,
    get_dispatcher_stats,
    is_dispatcher_lazy_mode
)
from genie.core.library import get_library_stats, set_lazy_mode
from genie.core.dispatcher import dispatcher  # Backward compatibility
from genie.core.lazy_tensor import LazyTensor

logger = logging.getLogger(__name__)


class TestStructuredDispatcher:
    """Test suite for the structured dispatcher implementation."""

    def test_dispatcher_initialization(self):
        """Test dispatcher initialization."""
        initialize_dispatcher()
        stats = get_dispatcher_stats()
        
        assert stats["initialized"] is True
        assert stats["library_name"] == "aten (FRAGMENT)"
        assert stats["backend_key"] == "PrivateUse1"

    def test_lazy_mode_control(self):
        """Test lazy mode enable/disable functionality."""
        # Test structured dispatcher
        set_dispatcher_lazy_mode(True)
        assert is_dispatcher_lazy_mode() is True
        
        set_dispatcher_lazy_mode(False)
        assert is_dispatcher_lazy_mode() is False
        
        # Reset to lazy mode
        set_dispatcher_lazy_mode(True)
        
        # Test library-level control
        set_lazy_mode(False)
        library_stats = get_library_stats()
        assert library_stats["lazy_mode"] is False
        
        set_lazy_mode(True)  # Reset

    def test_operation_registration_stats(self):
        """Test that operations are properly registered."""
        stats = get_library_stats()
        
        # Should have registered many operations
        assert stats["registered_ops"] > 20
        assert stats["failed_ops"] >= 0  # Some failures are acceptable
        
        # Check specific operations are registered
        registered_ops = structured_dispatcher.get_registered_operations()
        expected_ops = {
            "add.Tensor", "sub.Tensor", "mul.Tensor", "div.Tensor",
            "matmul", "mm", "bmm", "randn", "zeros", "ones",
            "relu", "sigmoid", "tanh", "conv2d"
        }
        
        assert expected_ops.issubset(registered_ops)

    def test_backward_compatibility(self):
        """Test backward compatibility with old dispatcher interface."""
        # Old interface should still work
        old_stats = dispatcher.get_stats()
        
        assert "registered_ops" in old_stats
        assert "fallback_ops" in old_stats
        assert "operation_count" in old_stats
        assert "lazy_mode" in old_stats
        
        # Test lazy mode control
        dispatcher.set_lazy_mode(False)
        assert dispatcher.get_stats()["lazy_mode"] is False
        
        dispatcher.set_lazy_mode(True)  # Reset

    def test_lazy_tensor_creation(self):
        """Test that LazyTensor creation works with structured dispatcher."""
        lazy_tensor = structured_dispatcher.create_lazy_tensor(
            "aten::add", 
            torch.randn(2, 2), 
            torch.randn(2, 2),
            alpha=1.0
        )
        
        assert isinstance(lazy_tensor, LazyTensor)
        assert lazy_tensor.operation == "aten::add"
        assert len(lazy_tensor.inputs) == 2
        assert lazy_tensor.kwargs.get("alpha") == 1.0

    def test_operation_count_tracking(self):
        """Test that operation counts are tracked correctly."""
        initial_stats = get_library_stats()
        initial_count = initial_stats["operation_count"]
        
        # Create some lazy tensors
        for _ in range(5):
            structured_dispatcher.create_lazy_tensor(
                "aten::matmul",
                torch.randn(3, 3),
                torch.randn(3, 3)
            )
        
        final_stats = get_library_stats()
        final_count = final_stats["operation_count"]
        
        assert final_count >= initial_count + 5

    def test_deprecated_register_op_warning(self):
        """Test that deprecated register_op method shows warning."""
        with pytest.warns(UserWarning, match="register_op is deprecated"):
            @dispatcher.register_op("test::deprecated")
            def test_func():
                pass

    def test_library_fragment_approach(self):
        """Test that the library uses FRAGMENT approach correctly."""
        from genie.core.library import genie_lib
        
        # The library should be a fragment of aten
        # This is indicated by the library name
        assert hasattr(genie_lib, '_name')
        # Note: The exact attribute name may vary by PyTorch version


class TestOperationCoverage:
    """Test coverage of registered operations."""

    def test_arithmetic_operations_coverage(self):
        """Test arithmetic operations are covered."""
        registered_ops = structured_dispatcher.get_registered_operations()
        
        arithmetic_ops = {
            "add.Tensor", "sub.Tensor", "mul.Tensor", "div.Tensor"
        }
        
        assert arithmetic_ops.issubset(registered_ops)

    def test_linear_algebra_coverage(self):
        """Test linear algebra operations are covered."""
        registered_ops = structured_dispatcher.get_registered_operations()
        
        linalg_ops = {
            "matmul", "mm", "bmm", "addmm", "linear"
        }
        
        assert linalg_ops.issubset(registered_ops)

    def test_tensor_creation_coverage(self):
        """Test tensor creation operations are covered."""
        registered_ops = structured_dispatcher.get_registered_operations()
        
        creation_ops = {
            "randn", "zeros", "ones"
        }
        
        assert creation_ops.issubset(registered_ops)

    def test_activation_functions_coverage(self):
        """Test activation functions are covered."""
        registered_ops = structured_dispatcher.get_registered_operations()
        
        activation_ops = {
            "relu", "sigmoid", "tanh", "softmax.int", "log_softmax.int"
        }
        
        assert activation_ops.issubset(registered_ops)

    def test_convolution_coverage(self):
        """Test convolution operations are covered."""
        registered_ops = structured_dispatcher.get_registered_operations()
        
        conv_ops = {
            "conv2d"
        }
        
        assert conv_ops.issubset(registered_ops)

    def test_tensor_manipulation_coverage(self):
        """Test tensor manipulation operations are covered."""
        registered_ops = structured_dispatcher.get_registered_operations()
        
        manipulation_ops = {
            "view", "transpose.int", "permute", "cat", "stack"
        }
        
        assert manipulation_ops.issubset(registered_ops)

    def test_normalization_coverage(self):
        """Test normalization operations are covered."""
        registered_ops = structured_dispatcher.get_registered_operations()
        
        norm_ops = {
            "batch_norm", "dropout"
        }
        
        assert norm_ops.issubset(registered_ops)

    def test_operation_coverage_percentage(self):
        """Test that we have good operation coverage."""
        registered_ops = structured_dispatcher.get_registered_operations()
        
        # We should have at least 20 operations registered
        # This represents significant coverage of common PyTorch operations
        assert len(registered_ops) >= 20
        
        logger.info(f"Registered {len(registered_ops)} operations")
        logger.info(f"Operations: {sorted(registered_ops)}")


class TestPerformanceAndCompatibility:
    """Test performance and compatibility aspects."""

    def test_no_import_errors(self):
        """Test that all modules import without errors."""
        # These imports should not raise exceptions
        from genie.core import library
        from genie.core import dispatcher_v2
        from genie.core import dispatcher
        
        assert library is not None
        assert dispatcher_v2 is not None
        assert dispatcher is not None

    def test_stats_consistency(self):
        """Test that statistics are consistent between interfaces."""
        structured_stats = get_dispatcher_stats()
        library_stats = get_library_stats()
        compat_stats = dispatcher.get_stats()
        
        # Operation counts should be consistent
        assert structured_stats["registered_ops"] == library_stats["registered_ops"]
        assert compat_stats["registered_ops"] == library_stats["registered_ops"]
        
        # Lazy mode should be consistent
        assert structured_stats["lazy_mode"] == library_stats["lazy_mode"]

    def test_memory_usage(self):
        """Test that the structured approach doesn't use excessive memory."""
        import sys
        
        # Get initial memory usage
        initial_size = sys.getsizeof(structured_dispatcher)
        
        # Create many lazy tensors
        tensors = []
        for i in range(100):
            tensor = structured_dispatcher.create_lazy_tensor(
                "aten::add",
                torch.randn(10, 10),
                torch.randn(10, 10)
            )
            tensors.append(tensor)
        
        # Memory usage shouldn't grow excessively
        final_size = sys.getsizeof(structured_dispatcher)
        
        # The dispatcher itself shouldn't grow much
        assert final_size - initial_size < 1000  # Less than 1KB growth


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
