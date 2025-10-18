"""
Unit tests for factory function interception.

Tests the FactoryInterceptor class including:
- Factory function wrapping and unwrapping
- Meta and CPU device filtering (recursion prevention)
- Remote accelerator device handling
- Capture context integration
- Factory method delegation
"""

import pytest
import torch
import numpy as np

from genie.core.factory_interceptor import FactoryInterceptor
from genie.core.lazy_tensor import LazyTensor
import genie


class TestFactoryInterceptor:
    """Test FactoryInterceptor functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.interceptor = FactoryInterceptor()

    def teardown_method(self):
        """Clean up after tests."""
        # Restore original functions
        self.interceptor.unwrap()

    def test_factory_wrapping(self):
        """Test that factory functions are properly wrapped."""
        # Initially not wrapped
        assert not self.interceptor._wrapped

        # Wrap factory functions
        self.interceptor.wrap()
        assert self.interceptor._wrapped

        # Verify original functions are stored
        assert len(self.interceptor._original_functions) > 0
        assert 'randn' in self.interceptor._original_functions
        assert 'zeros' in self.interceptor._original_functions

    def test_unwrap_restores_original(self):
        """Test that unwrap() restores original functions."""
        # Wrap functions
        self.interceptor.wrap()

        # Get original randn function
        original_randn = self.interceptor._original_functions['randn']

        # Unwrap
        self.interceptor.unwrap()

        # Should be back to original
        assert torch.randn is original_randn
        assert not self.interceptor._wrapped
        assert len(self.interceptor._original_functions) == 0

    def test_device_api_without_capture(self):
        """Verify device-based API works outside capture context."""
        # This is the paper's API - must work!
        x = torch.randn(10, 10, device="remote_accelerator:0")
        assert isinstance(x, LazyTensor), "Device API broken!"

        y = x @ x
        assert isinstance(y, LazyTensor)

        # Materialize
        result = y.cpu()
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 10)

    def test_capture_api_without_device(self):
        """Verify context-based API works without device argument."""
        # This is the new convenience API
        with genie.capture():
            x = torch.randn(10, 10)  # No device argument
            assert isinstance(x, LazyTensor), "Capture API broken!"

            y = x @ x
            assert isinstance(y, LazyTensor)

        # Materialize outside context
        result = y.cpu()
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 10)

    def test_meta_device_not_intercepted(self):
        """Test that meta devices are not intercepted (critical for recursion prevention)."""
        # Create LazyTensor internally uses meta device for shape inference
        x = LazyTensor.randn(5, 5)

        # This should work without infinite recursion
        y = x + 1
        result = y.cpu()

        assert isinstance(result, torch.Tensor)
        assert result.shape == (5, 5)

    def test_cpu_device_not_intercepted(self):
        """Test that CPU devices are not intercepted."""
        # CPU device should not be intercepted
        x = torch.randn(3, 3, device='cpu')
        assert isinstance(x, torch.Tensor)
        assert not isinstance(x, LazyTensor)

    def test_factory_functions_return_lazy_tensors_in_capture(self):
        """Test that factory functions return LazyTensors when in capture context."""
        self.interceptor.wrap()

        with genie.capture():
            # These should return LazyTensors
            x = torch.randn(10, 10)
            y = torch.zeros(5, 5)
            z = torch.ones(3, 3)
            w = torch.empty(2, 2)
            v = torch.full(1.5, 4, 4)

            assert isinstance(x, LazyTensor)
            assert isinstance(y, LazyTensor)
            assert isinstance(z, LazyTensor)
            assert isinstance(w, LazyTensor)
            assert isinstance(v, LazyTensor)

            # Check operations
            assert x.operation == 'aten::randn'
            assert y.operation == 'aten::zeros'
            assert z.operation == 'aten::ones'
            assert w.operation == 'aten::empty'
            assert v.operation == 'aten::full'

    def test_factory_functions_return_lazy_tensors_with_remote_device(self):
        """Test that factory functions return LazyTensors with remote device."""
        self.interceptor.wrap()

        # Remote accelerator device should return LazyTensors
        x = torch.randn(10, 10, device='remote_accelerator:0')
        y = torch.zeros(5, 5, device='remote_accelerator:0')

        assert isinstance(x, LazyTensor)
        assert isinstance(y, LazyTensor)

    def test_factory_functions_return_normal_tensors_outside_capture(self):
        """Test that factory functions return normal tensors outside capture context."""
        self.interceptor.wrap()

        # Outside capture context, should return normal tensors
        x = torch.randn(10, 10)
        y = torch.zeros(5, 5)

        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert not isinstance(x, LazyTensor)
        assert not isinstance(y, LazyTensor)

    def test_data_conversion_functions(self):
        """Test tensor, as_tensor, and from_numpy functions."""
        self.interceptor.wrap()

        with genie.capture():
            # Test tensor function
            data = torch.randn(3, 4)
            x = torch.tensor(data)
            assert isinstance(x, LazyTensor)
            assert x.operation == 'aten::tensor'

            # Test as_tensor function
            y = torch.as_tensor(data)
            assert isinstance(y, LazyTensor)
            assert y.operation == 'aten::as_tensor'

            # Test from_numpy function
            numpy_data = np.random.randn(2, 3).astype(np.float32)
            z = torch.from_numpy(numpy_data)
            assert isinstance(z, LazyTensor)
            assert z.operation == 'aten::from_numpy'

    def test_special_constructor_functions(self):
        """Test eye, arange, linspace functions."""
        self.interceptor.wrap()

        with genie.capture():
            # Test eye function
            x = torch.eye(3)
            assert isinstance(x, LazyTensor)
            assert x.operation == 'aten::eye'

            # Test arange function
            y = torch.arange(5)
            assert isinstance(y, LazyTensor)
            assert y.operation == 'aten::arange'

    def test_materialization_with_factory_functions(self):
        """Test that LazyTensors created from factory functions can be materialized."""
        self.interceptor.wrap()

        with genie.capture():
            x = torch.randn(10, 10)
            y = torch.zeros(10, 10)
            z = x + y

        # Materialize the result
        result = z.cpu()
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 10)

    def test_correctness_of_materialized_factory_functions(self):
        """Test that materialized factory functions produce correct results."""
        self.interceptor.wrap()

        # Set seed for reproducibility
        torch.manual_seed(42)

        with genie.capture():
            x = torch.randn(5, 5)
            y = torch.zeros(5, 5)
            z = x + y

        result = z.cpu()

        # Create equivalent computation with native PyTorch
        torch.manual_seed(42)
        x_native = torch.randn(5, 5)
        y_native = torch.zeros(5, 5)
        z_native = x_native + y_native

        # Results should be identical
        torch.testing.assert_close(result, z_native)

    def test_multiple_factory_interceptor_instances(self):
        """Test that multiple interceptor instances don't interfere."""
        interceptor1 = FactoryInterceptor()
        interceptor2 = FactoryInterceptor()

        # Wrap both
        interceptor1.wrap()
        interceptor2.wrap()

        with genie.capture():
            x = torch.randn(5, 5)
            assert isinstance(x, LazyTensor)

        # Unwrap one
        interceptor1.unwrap()

        with genie.capture():
            y = torch.randn(5, 5)
            # Should still work because interceptor2 is still active
            assert isinstance(y, LazyTensor)

        # Unwrap the other
        interceptor2.unwrap()

        # Now should return normal tensors
        z = torch.randn(5, 5)
        assert not isinstance(z, LazyTensor)


class TestIntegration:
    """Integration tests for factory interceptor."""

    def test_both_api_styles_simultaneously(self):
        """Test that both device-based and context-based APIs work together."""
        interceptor = FactoryInterceptor()
        interceptor.wrap()

        try:
            # Device-based API
            x_device = torch.randn(10, 10, device='remote_accelerator:0')
            assert isinstance(x_device, LazyTensor)

            # Context-based API
            with genie.capture():
                x_context = torch.randn(10, 10)
                assert isinstance(x_context, LazyTensor)

                # Hybrid: device + context
                x_hybrid = torch.randn(10, 10, device='remote_accelerator:0')
                assert isinstance(x_hybrid, LazyTensor)

                # Combine them
                result = x_device + x_context + x_hybrid
                assert isinstance(result, LazyTensor)

            # Outside context, device API should still work
            x_outside = torch.randn(10, 10, device='remote_accelerator:0')
            assert isinstance(x_outside, LazyTensor)

        finally:
            interceptor.unwrap()

    def test_nested_capture_contexts_with_factory_interceptor(self):
        """Test nested capture contexts work correctly with factory interceptor."""
        interceptor = FactoryInterceptor()
        interceptor.wrap()

        try:
            with genie.capture():
                x1 = torch.randn(10)
                assert isinstance(x1, LazyTensor)

                with genie.capture():
                    x2 = torch.randn(10)
                    assert isinstance(x2, LazyTensor)

                x3 = torch.randn(10)
                assert isinstance(x3, LazyTensor)  # Still in outer context

            x4 = torch.randn(10)
            assert not isinstance(x4, LazyTensor)  # Outside all contexts

        finally:
            interceptor.unwrap()


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
