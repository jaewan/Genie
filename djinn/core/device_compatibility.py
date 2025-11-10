"""
PyTorch Device Compatibility Layer for Djinn.

Provides PyTorch-compatible device semantics by patching nn.Module.to()
to automatically convert model weights to LazyTensors when 'remote_accelerator'
device is requested. This bridges PyTorch's device model with Djinn's
LazyTensor-based computation graph capture.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def _register_device_python_fallback():
    """
    Python fallback for device registration when C++ backend is unavailable.

    Instead of registering with PyTorch, we work around device validation.
    """
    # We can't register devices with PyTorch, but we can catch the validation errors
    # and handle them gracefully. The actual device handling is done in LazyTensor.
    logger.info("✅ Python device fallback initialized (no registration needed)")


class RemoteAcceleratorSupport:
    """Enables remote_accelerator device for PyTorch models."""

    _initialized = False
    _original_module_to = None

    @classmethod
    def initialize(cls):
        """Initialize remote_accelerator device support."""
        if cls._initialized:
            return

        # Try C++ backend first, fall back to Python
        try:
            from .. import _C
            _C.register_remote_accelerator_device()
            logger.info("✅ C++ device backend registered")
        except (ImportError, AttributeError):
            logger.info("ℹ️  C++ backend unavailable, using Python fallback")
            _register_device_python_fallback()

        # Store original method
        cls._original_module_to = nn.Module.to

        # Patch Module.to()
        def patched_to(self, *args, **kwargs):
            # Check if this is a remote_accelerator request before parsing
            device_arg = None
            if args and isinstance(args[0], (str, torch.device)):
                device_arg = args[0]
            elif 'device' in kwargs:
                device_arg = kwargs['device']

            if device_arg is not None:
                device_str = str(device_arg)

                # Check if remote_accelerator is requested
                if 'remote_accelerator' in device_str:
                    logger.info(f"Converting model {self.__class__.__name__} to remote_accelerator")

                    # Import here to avoid circular dependency
                    from djinn.frontend.core.lazy_tensor import LazyTensor

                    # Convert all parameters and buffers to LazyTensors
                    lazy_params = {}
                    lazy_buffers = {}

                    for name, param in self.named_parameters():
                        lazy_params[name] = LazyTensor.tensor(
                            param.detach(),
                            device=param.device  # Preserve original device
                        )

                    for name, buffer in self.named_buffers():
                        lazy_buffers[name] = LazyTensor.tensor(
                            buffer.detach(),
                            device=buffer.device  # Preserve original device
                        )

                    # Replace parameters and buffers
                    for name, lazy_param in lazy_params.items():
                        # Get the parameter object and replace its data
                        param = dict(self.named_parameters())[name]
                        param.data = lazy_param

                    for name, lazy_buffer in lazy_buffers.items():
                        buffer = dict(self.named_buffers())[name]
                        buffer.data = lazy_buffer

                    # Track conversion
                    param_count = len(lazy_params)
                    buffer_count = len(lazy_buffers)
                    logger.info(f"Converted {param_count} parameters and {buffer_count} buffers to LazyTensors")

                    # Return self (device conversion complete)
                    return self

            # For all other devices, use original implementation
            # Handle the case where PyTorch might reject our custom device string
            try:
                return cls._original_module_to(self, *args, **kwargs)
            except RuntimeError as e:
                if 'remote_accelerator' in str(e):
                    # This is our custom device, handle it
                    logger.warning(f"PyTorch rejected remote_accelerator device: {e}")
                    # Return self unchanged - the device conversion is handled at tensor level
                    return self
                else:
                    # Re-raise for actual errors
                    raise

        # Apply the patch
        nn.Module.to = patched_to
        cls._initialized = True

        logger.info("✅ Remote accelerator device support initialized")


def setup():
    """Setup device compatibility layer."""
    RemoteAcceleratorSupport.initialize()