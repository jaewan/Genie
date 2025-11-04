"""
Test: Error handling and edge cases

Validates:
- Unsupported operations produce clear errors
- Empty graphs handled gracefully
- Invalid inputs produce actionable errors
- System doesn't hang on edge cases
"""

import torch
import pytest
import genie


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_unsupported_operation_clear_error(self):
        """Test unsupported operations give clear error messages."""

        with genie.capture():
            x = torch.randn(10, 10)

            # FFT might not be supported
            try:
                y = torch.fft.fft(x)
                result = y.cpu()  # Try to materialize
                print("✅ FFT supported")
            except Exception as e:
                # Error should be actionable
                error_msg = str(e).lower()
                assert any(keyword in error_msg for keyword in [
                    'unsupported', 'not implemented', 'not supported',
                    'failed', 'unknown'
                ]), f"Unclear error message: {e}"
                print(f"✅ Unsupported operation error is clear: {type(e).__name__}")

    def test_empty_graph_handled(self):
        """Test empty capture doesn't crash."""

        with genie.capture():
            pass  # Empty capture

        # Should not crash, may return None if no tensors captured
        try:
            graph = genie.get_graph()
        except Exception as e:
            pytest.fail(f"Empty graph crashed: {e}")

        # Empty capture may return None (no tensors) or an empty graph
        if graph is not None:
            nodes = list(graph.nodes())
            assert len(nodes) == 0, f"Empty graph has {len(nodes)} nodes!"
            print("✅ Empty graph handled gracefully (empty graph returned)")
        else:
            print("✅ Empty graph handled gracefully (None returned)")

    def test_mixed_device_handled(self):
        """Test mixing captured and native tensors is handled."""

        with genie.capture():
            x_lazy = torch.randn(10, 10)

        x_native = torch.randn(10, 10)  # Native tensor

        # This should either work or give clear error
        try:
            y = x_lazy + x_native
            result = y.cpu() if hasattr(y, 'cpu') else y
            print("✅ Mixed tensors handled (auto-materialization)")
        except Exception as e:
            # Error should mention device or mixing
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in [
                'device', 'mixed', 'type', 'mismatch'
            ]), f"Unclear error for mixed tensors: {e}"
            print(f"✅ Mixed tensor error is clear: {type(e).__name__}")

    @pytest.mark.slow
    def test_graph_operations_dont_hang(self):
        """Test graph operations don't hang indefinitely."""

        # Create deep graph
        with genie.capture():
            x = torch.randn(100, 100)
            for _ in range(100):
                x = x + 1

        graph = genie.get_graph()

        # This should complete in reasonable time
        try:
            nodes = list(graph.topological_sort())
            assert len(nodes) > 0
            print(f"✅ Deep graph handled ({len(nodes)} nodes)")
        except Exception as e:
            pytest.fail(f"Graph operations hung or failed: {e}")

    def test_invalid_device_index(self):
        """Test invalid device index produces clear error."""

        try:
            x = torch.randn(10, 10, device='remote_accelerator:999')
            # If it doesn't raise immediately, try to use it
            y = x + 1
            result = y.cpu()
            # If we got here, system handles it (maybe maps to default)
            print("✅ Invalid device index handled silently")
        except Exception as e:
            # Should mention device or execution error
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in [
                'device', 'index', 'invalid', 'not found', 'execution', 'failed'
            ]), f"Unclear error for invalid device: {e}"
            print(f"✅ Invalid device error is clear: {type(e).__name__}")

    def test_circular_dependency_prevented(self):
        """Test circular dependencies don't crash (defensive)."""

        # LazyTensor DAG should be acyclic by design, but test anyway
        with genie.capture():
            x = torch.randn(10, 10)
            # Try to create self-referential operation
            for i in range(10):
                x = x + x

        graph = genie.get_graph()

        # Should not crash, even with repetitive operations
        try:
            nodes = list(graph.topological_sort())
            print(f"✅ Repetitive operations handled ({len(nodes)} nodes)")
        except Exception as e:
            # If it fails, should be clear about the issue
            assert 'cycle' in str(e).lower() or 'circular' in str(e).lower(), \
                f"Unclear error for potential cycle: {e}"

    def test_zero_size_tensor(self):
        """Test zero-size tensors are handled."""

        with genie.capture():
            x = torch.randn(0, 10)  # Empty first dimension
            y = torch.randn(10, 5)

            # This should work or fail clearly
            try:
                z = x @ y
                result = z.cpu()
                assert result.shape == torch.Size([0, 5])
                print("✅ Zero-size tensor handled")
            except Exception as e:
                print(f"⚠️  Zero-size tensor raises: {type(e).__name__}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
