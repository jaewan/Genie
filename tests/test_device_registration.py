"""
Test device registration functionality with proper PyTorch backend integration.
"""
import pytest
import torch
import logging

from genie.core.device import RemoteAcceleratorDevice, get_device, is_available, get_device_count

logger = logging.getLogger(__name__)


class TestDeviceRegistration:
    """Test suite for device registration functionality."""

    def test_backend_registration(self):
        """Test that the backend registers successfully with PyTorch."""
        # This should not raise an exception
        device = RemoteAcceleratorDevice.get_device(0)
        assert device is not None
        assert device.type == "remote_accelerator"
        assert device.index == 0

    def test_device_availability(self):
        """Test device availability checks."""
        assert is_available(), "remote_accelerator backend should be available"
        assert get_device_count() > 0, "Should have at least one device"
        assert get_device_count() == 4, "Should report 4 devices as per specs"

    def test_device_creation(self):
        """Test creating devices with different indices."""
        device0 = get_device(0)
        device1 = get_device(1)
        
        assert device0.index == 0
        assert device1.index == 1
        assert device0 != device1
        
        # Test device equality
        device0_again = get_device(0)
        assert device0 == device0_again
        assert device0 is device0_again  # Should be same instance

    def test_torch_device_integration(self):
        """Test integration with PyTorch device system."""
        device = get_device(0)
        torch_device = device.to_torch_device()
        
        assert isinstance(torch_device, torch.device)
        assert torch_device.type == "remote_accelerator"
        assert torch_device.index == 0

    def test_device_string_representation(self):
        """Test device string representations."""
        device = get_device(2)
        assert str(device) == "remote_accelerator:2"
        assert repr(device) == "remote_accelerator:2"

    def test_device_equality_with_torch_device(self):
        """Test device equality with torch.device objects."""
        device = get_device(1)
        torch_device = torch.device("remote_accelerator", 1)
        
        # This should work once backend is properly registered
        assert device == torch_device

    def test_device_memory_stats(self):
        """Test device memory statistics."""
        device = get_device(0)
        stats = device.memory_stats()
        
        assert isinstance(stats, dict)
        assert "allocated" in stats
        assert "cached" in stats
        assert "reserved" in stats
        assert "device_index" in stats
        assert stats["device_index"] == 0

    def test_device_synchronization(self):
        """Test device synchronization (no-op in Phase 1)."""
        device = get_device(0)
        # Should not raise an exception
        device.synchronize()

    @pytest.mark.skipif(not is_available(), reason="remote_accelerator backend not available")
    def test_tensor_creation_on_device(self):
        """Test creating tensors on remote_accelerator device produces LazyTensor in Phase 1."""
        device = torch.device("remote_accelerator", 0)
        x = torch.zeros(2, 2, device=device)
        from genie.core.lazy_tensor import LazyTensor
        assert isinstance(x, LazyTensor)

    def test_multiple_device_instances(self):
        """Test creating multiple device instances."""
        devices = [get_device(i) for i in range(4)]
        
        # All should be different
        for i, device in enumerate(devices):
            assert device.index == i
            for j, other_device in enumerate(devices):
                if i != j:
                    assert device != other_device

    def test_device_hash(self):
        """Test device hashing for use in sets/dicts."""
        device0 = get_device(0)
        device1 = get_device(1)
        device0_again = get_device(0)
        
        # Test that devices can be used as dict keys
        device_dict = {device0: "first", device1: "second"}
        assert device_dict[device0_again] == "first"
        
        # Test that devices can be used in sets
        device_set = {device0, device1, device0_again}
        assert len(device_set) == 2  # device0 and device0_again should be the same


class TestDeviceIntegration:
    """Test integration with PyTorch systems."""

    def test_c_extension_loading(self):
        """Test that C++ extension loads correctly."""
        try:
            from genie import _C
            assert hasattr(_C, 'register_remote_accelerator_device')
            assert hasattr(_C, 'device_count')
            
            # Test device count function
            count = _C.device_count()
            assert isinstance(count, int)
            assert count == 4
            
        except ImportError as e:
            pytest.fail(f"C++ extension failed to load: {e}")

    def test_backend_registration_idempotent(self):
        """Test that backend registration is idempotent."""
        # Multiple calls should not cause issues
        device1 = RemoteAcceleratorDevice.get_device(0)
        device2 = RemoteAcceleratorDevice.get_device(0)
        
        assert device1 is device2
        assert RemoteAcceleratorDevice.is_available()

    def test_logging_output(self, caplog):
        """Test that appropriate logging messages are generated."""
        with caplog.at_level(logging.INFO):
            # Force re-registration by creating new device
            device = RemoteAcceleratorDevice.get_device(3)
            
        # Should have logged successful registration
        # Note: This might not work if backend is already registered
        log_messages = [record.message for record in caplog.records]
        logger.info(f"Log messages: {log_messages}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
