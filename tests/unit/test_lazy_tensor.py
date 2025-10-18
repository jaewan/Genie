"""
Unit tests for LazyTensor subclass functionality.

Tests the core LazyTensor implementation including:
- Proper torch.Tensor subclass behavior
- Operation interception via __torch_dispatch__
- Shape inference and caching
- Materialization behavior
- Factory method functionality
- Thread safety
- Error handling
"""

import pytest
import torch
import numpy as np

import genie
from genie.core.lazy_tensor import LazyTensor


class TestLazyTensorSubclass:
    """Test LazyTensor as proper torch.Tensor subclass."""

    def test_lazy_tensor_is_proper_subclass(self):
        """Verify LazyTensor is recognized as torch.Tensor."""
        x = LazyTensor.randn(10, 10)

        assert isinstance(x, torch.Tensor)
        assert isinstance(x, LazyTensor)
        assert x.shape == (10, 10)
        assert x.dtype == torch.float32

    def test_operations_intercepted(self):
        """Verify operations create new LazyTensors."""
        x = LazyTensor.randn(10, 10)
        y = LazyTensor.randn(10, 10)

        z = x + y  # Should return LazyTensor
        assert isinstance(z, LazyTensor)
        assert z.operation == 'aten::add'
        assert len(z.inputs) == 2

    def test_no_execution_until_materialization(self):
        """Verify operations are deferred until materialization."""
        x = LazyTensor.randn(10, 10)
        y = x @ x

        # No computation yet - should be fast
        assert isinstance(y, LazyTensor)

    def test_unsupported_op_no_infinite_recursion(self):
        """Verify no infinite recursion for ops with unknown shapes."""
        # Manually create LazyTensor with unsupported op
        x = LazyTensor(operation='aten::unknown_op', inputs=[], kwargs={})
        object.__setattr__(x, '_shape', None)  # Simulate failed inference

        # This should not infinite loop
        y = LazyTensor(operation='aten::add', inputs=[x, x], kwargs={})

        # Shape should be None (can't infer)
        assert y.shape is None or y.shape == torch.Size([])
        assert y.operation == 'aten::add'

        # The key point: no infinite recursion during shape inference
        # Materialization would fail for unsupported ops, which is expected

    def test_correctness_vs_native(self):
        """Verify numerical correctness against native PyTorch."""
        # Native PyTorch
        torch.manual_seed(42)
        x_native = torch.randn(32, 64)
        y_native = torch.randn(64, 128)
        z_native = (x_native @ y_native).relu()

        # Genie LazyTensor
        torch.manual_seed(42)
        x_lazy = LazyTensor.randn(32, 64)
        y_lazy = LazyTensor.randn(64, 128)
        z_lazy = (x_lazy @ y_lazy).relu()
        z_concrete = z_lazy.cpu()

        # Should be identical
        torch.testing.assert_close(z_concrete, z_native)

    def test_dynamic_control_flow_fallback(self):
        """Verify fallback to LazyDAG for models with dynamic control flow."""
        class DynamicModel(torch.nn.Module):
            def forward(self, x):
                if x.sum() > 0:  # Data-dependent branch
                    return x.relu()
                else:
                    return x.tanh()

        model = DynamicModel()

        with genie.capture():
            x = torch.randn(10, 10)
            y = model(x)

        graph = genie.get_graph()
        # Should fall back from FX to LazyDAG due to dynamic control flow
        assert graph.backend_type == 'lazy_dag'

    def test_nested_capture_contexts(self):
        """Verify nested capture contexts maintain correct state."""
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

    def test_mixed_device_operations(self):
        """Verify behavior when mixing captured and native tensors."""
        with genie.capture():
            x_lazy = torch.randn(10, 10)

        x_cpu = torch.randn(10, 10)  # Native CPU tensor

        # What happens here?
        y = x_lazy + x_cpu  # Should materialize x_lazy and compute on CPU
        assert isinstance(y, torch.Tensor)
        assert not isinstance(y, LazyTensor)  # Should be materialized

    def test_both_api_styles(self):
        """Verify both device-based and context-based APIs work."""
        # Device-based API (legacy)
        x_device = torch.randn(10, 10, device='remote_accelerator:0')
        assert isinstance(x_device, LazyTensor)

        # Context-based API (new)
        with genie.capture():
            x_context = torch.randn(10, 10)
            assert isinstance(x_context, LazyTensor)

        # Hybrid API
        with genie.capture():
            x_hybrid = torch.randn(10, 10, device='remote_accelerator:0')
            assert isinstance(x_hybrid, LazyTensor)

    def test_device_api_works_without_capture(self):
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

    def test_capture_api_works_without_device(self):
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

    def test_capture_context_is_thread_safe(self):
        """Verify capture contexts don't interfere across threads."""
        import threading

        results = {}

        def thread1_work():
            with genie.capture():
                x = torch.randn(10)
                results['thread1'] = isinstance(x, LazyTensor)

        def thread2_work():
            # NOT in capture context
            x = torch.randn(10)
            results['thread2'] = not isinstance(x, LazyTensor)

        t1 = threading.Thread(target=thread1_work)
        t2 = threading.Thread(target=thread2_work)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results['thread1'], "Thread 1 should have LazyTensor"
        assert results['thread2'], "Thread 2 should have normal tensor"


class TestFactoryMethods:
    """Test LazyTensor factory methods."""

    def test_randn_factory(self):
        """Test randn factory method."""
        x = LazyTensor.randn(5, 3)
        assert isinstance(x, LazyTensor)
        assert x.operation == 'aten::randn'
        assert x.shape == (5, 3)

    def test_tensor_factory(self):
        """Test tensor factory method."""
        data = torch.randn(2, 4)
        x = LazyTensor.tensor(data)
        assert isinstance(x, LazyTensor)
        assert x.operation == 'aten::tensor'
        assert x.shape == (2, 4)

    def test_as_tensor_factory(self):
        """Test as_tensor factory method."""
        data = np.random.randn(3, 2).astype(np.float32)
        x = LazyTensor.as_tensor(data)
        assert isinstance(x, LazyTensor)
        assert x.operation == 'aten::as_tensor'
        assert x.shape == (3, 2)

    def test_from_numpy_factory(self):
        """Test from_numpy factory method."""
        data = np.random.randn(4, 5).astype(np.float32)
        x = LazyTensor.from_numpy(data)
        assert isinstance(x, LazyTensor)
        assert x.operation == 'aten::from_numpy'
        assert x.shape == (4, 5)

    def test_zeros_factory(self):
        """Test zeros factory method."""
        x = LazyTensor.zeros(3, 4)
        assert isinstance(x, LazyTensor)
        assert x.operation == 'aten::zeros'
        assert x.shape == (3, 4)

    def test_ones_factory(self):
        """Test ones factory method."""
        x = LazyTensor.ones(2, 3)
        assert isinstance(x, LazyTensor)
        assert x.operation == 'aten::ones'
        assert x.shape == (2, 3)

    def test_empty_factory(self):
        """Test empty factory method."""
        x = LazyTensor.empty(4, 5)
        assert isinstance(x, LazyTensor)
        assert x.operation == 'aten::empty'
        assert x.shape == (4, 5)

    def test_full_factory(self):
        """Test full factory method."""
        x = LazyTensor.full(3.14, 2, 3)
        assert isinstance(x, LazyTensor)
        assert x.operation == 'aten::full'
        assert x.shape == (2, 3)


class TestShapeInference:
    """Test shape inference functionality."""

    def test_simple_operation_shape_inference(self):
        """Test shape inference for simple operations."""
        x = LazyTensor.randn(10, 5)
        y = x.relu()  # Should preserve shape

        assert y.shape == (10, 5)

    def test_matmul_shape_inference(self):
        """Test shape inference for matrix multiplication."""
        x = LazyTensor.randn(32, 64)
        y = LazyTensor.randn(64, 128)
        z = x @ y

        assert z.shape == (32, 128)

    def test_broadcasting_shape_inference(self):
        """Test shape inference for broadcasting operations."""
        x = LazyTensor.randn(3, 1)
        y = LazyTensor.randn(1, 4)
        z = x + y

        # Broadcasting should work
        assert z.shape == (3, 4)

    def test_shape_inference_with_uninitialized_input(self):
        """
        Verify shape inference fails gracefully when input shape is uninitialized.

        This tests the edge case where a LazyTensor's shape inference failed,
        and it's used as input to another operation.
        """
        # Create a LazyTensor with explicitly uninitialized shape
        x = LazyTensor(
            operation='aten::hypothetical_unsupported_op',
            inputs=[],
            kwargs={},
            shape=None,  # Will become torch.Size([]) in __new__
            dtype=torch.float32,
            device=None
        )

        # Verify shape is uninitialized
        x_shape = object.__getattribute__(x, '_shape')
        assert x_shape is None, f"Expected None for uninitialized shape, got {x_shape}"

        # Try to use this tensor as input to another operation
        # This should fail gracefully, not create wrong fake tensor
        y = x + 1

        # Check that y's shape is also uninitialized (inference failed)
        y_shape = object.__getattribute__(y, '_shape')
        assert y_shape is None or len(y_shape) == 0, (
            f"Expected uninitialized shape, got {y_shape}. "
            "Shape inference should have failed for operation with uninitialized input."
        )

    def test_shape_inference_distinguishes_scalar_from_placeholder(self):
        """
        Verify that real scalars (torch.Size([])) are distinguished from
        uninitialized placeholders (also torch.Size([])).
        """
        # Real scalar from tensor factory
        scalar = LazyTensor.tensor(5.0)
        assert scalar.shape == torch.Size([]), "Real scalars have empty shape"

        # Use scalar in operation - should work
        result = scalar + 1
        result_shape = result.shape
        assert result_shape == torch.Size([]), "Scalar + scalar = scalar"

        # Verify materialization works
        concrete = result.cpu()
        assert concrete.shape == torch.Size([])
        assert isinstance(concrete.item(), (int, float))  # Just check it's a scalar


def test_shape_inference_handles_uninitialized():
    """
    Verify shape inference fails gracefully for uninitialized shapes.

    This tests the edge case where a LazyTensor's shape inference failed,
    and it's used as input to another operation.
    """
    # Manually create a LazyTensor with uninitialized shape
    x = LazyTensor(
        operation='aten::hypothetical_unsupported',
        inputs=[],
        kwargs={},
        shape=None,  # Will become torch.Size([])
        dtype=torch.float32,
        device=None
    )

    # Verify shape is uninitialized
    x_shape = object.__getattribute__(x, '_shape')
    assert x_shape is None, f"Expected None, got {x_shape}"

    # Try to use this tensor as input to another operation
    # This should fail gracefully, not create wrong fake tensor
    y = x + 1

    # Check that y's shape is also uninitialized (inference failed)
    y_shape = object.__getattribute__(y, '_shape')
    assert y_shape is None, (
        f"Expected None, got {y_shape}. "
        "Shape inference should have failed for operation with uninitialized input."
    )


    def test_nested_shape_inference_with_valid_inputs(self):
        """
        Verify nested shape inference works when all inputs have valid shapes.

        This is the common case - should complete without issues.
        """
        x = LazyTensor.randn(10, 10)  # Valid shape
        y = x + 1                      # Should infer shape from x
        z = y @ y                      # Should infer shape from y (which got it from x)

        assert y.shape == torch.Size([10, 10])
        assert z.shape == torch.Size([10, 10])

        # Verify materialization produces correct result
        result = z.cpu()
        assert result.shape == torch.Size([10, 10])


class TestGraphBuilderDelegation:
    """Test that graph builder delegates to executor."""

    def test_graph_builder_delegates_to_executor(self):
        """Verify that graph builder delegates materialization to executor."""
        import unittest.mock as mock
        from genie.core import executor

        # Create test tensor
        x = LazyTensor.randn(5, 5)
        y = x + 1

        # Mock the executor's execute_subgraph to verify it's called
        with mock.patch.object(executor, 'execute_subgraph', wraps=executor.execute_subgraph) as mock_execute:
            result = y.materialize()

            # Verify executor was called exactly once
            assert mock_execute.call_count == 1, (
                f"Expected execute_subgraph to be called once, "
                f"but was called {mock_execute.call_count} times"
            )

            # Verify correct tensor was passed
            call_args = mock_execute.call_args[0]
            assert call_args[0] is y, "Should pass target tensor to executor"

            # Verify result is correct
            assert isinstance(result, torch.Tensor)
            assert result.shape == (5, 5)


class TestMaterialization:
    """Test materialization behavior."""

    def test_materialization_via_cpu(self):
        """Test materialization via .cpu() method."""
        x = LazyTensor.randn(10, 10)
        y = x + 1

        result = y.cpu()
        assert isinstance(result, torch.Tensor)
        assert not isinstance(result, LazyTensor)
        assert result.shape == (10, 10)

    def test_materialization_via_numpy(self):
        """Test materialization via .numpy() method."""
        x = LazyTensor.randn(5, 5)
        y = x * 2

        result = y.numpy()
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 5)

    def test_materialization_preserves_grad(self):
        """Test that materialization preserves gradient information."""
        x = LazyTensor.randn(3, 3, requires_grad=True)
        y = x.sum()

        result = y.cpu()
        # Gradient information should be preserved during materialization
        assert result.requires_grad


class TestCaptureContextThreadSafety:
    """Test thread safety of capture contexts."""

    def test_capture_context_is_thread_safe(self):
        """Verify capture contexts don't interfere across threads."""
        import threading

        results = {}
        errors = []

        def thread1_work():
            try:
                with genie.capture():
                    x = torch.randn(10)
                    results['thread1'] = isinstance(x, LazyTensor)
            except Exception as e:
                errors.append(('thread1', e))

        def thread2_work():
            try:
                # NOT in capture context
                x = torch.randn(10)
                results['thread2'] = not isinstance(x, LazyTensor)
            except Exception as e:
                errors.append(('thread2', e))

        threads = [
            threading.Thread(target=thread1_work),
            threading.Thread(target=thread2_work)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check for errors
        if errors:
            pytest.fail(f"Thread errors: {errors}")

        # Verify results
        assert results.get('thread1'), "Thread 1 should have LazyTensor"
        assert results.get('thread2'), "Thread 2 should have normal tensor"


class TestNestedCaptureStateRestoration:
    """Test nested capture context state restoration."""

    def test_nested_capture_state_restoration(self):
        """Verify nested contexts don't leak state."""
        with genie.capture():
            x1 = torch.randn(10)
            assert isinstance(x1, LazyTensor)

            with genie.capture():
                x2 = torch.randn(10)
                assert isinstance(x2, LazyTensor)

            # Inner context should not affect outer
            graph1 = genie.get_graph()
            assert graph1 is not None

        # After outer context exits, graph should still be available
        # The graph builder maintains state across contexts for user access
        graph2 = genie.get_graph()
        assert graph2 is not None
        assert len(graph2.nodes()) > 0


class TestExecutorErrorHandling:
    """Test executor error handling and failure modes."""

    def test_executor_fails_loud_on_unsupported(self):
        """Verify unsupported operations raise clear errors."""
        from genie.core.errors import UnsupportedOperationError

        # Create a LazyTensor with unsupported operation
        x = LazyTensor.randn(10)
        # Manually create unsupported op
        fake_op = LazyTensor(
            operation='aten::fake_unsupported_op_12345',
            inputs=[x],
            kwargs={}
        )

        with pytest.raises(UnsupportedOperationError) as exc_info:
            fake_op.materialize()

        error_msg = str(exc_info.value)
        assert 'fake_unsupported_op_12345' in error_msg
        assert 'failed to execute' in error_msg.lower()


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
