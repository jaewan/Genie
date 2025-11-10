"""
Minimal device support for remote_accelerator.

Enables model.to('remote_accelerator:0') following PyTorch conventions.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class RemoteAcceleratorSupport:
    """Enables remote_accelerator device for PyTorch models."""

    _initialized = False
    _original_module_to = None

    @classmethod
    def initialize(cls):
        """Initialize remote_accelerator device support."""
        if cls._initialized:
            return

        # Store original method
        cls._original_module_to = nn.Module.to

        # Patch Module.to()
        def patched_to(self, *args, **kwargs):
            # Check if this is a remote_accelerator request before parsing
            # We need to handle this specially since PyTorch's parser doesn't recognize it
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

                    # Convert parameters - we need to traverse the module tree
                    param_count = 0
                    def convert_module_params(module, prefix=''):
                        nonlocal param_count
                        for name, param in module.named_parameters(recurse=False):
                            if not isinstance(param.data, LazyTensor):
                                # Create new LazyTensor
                                lazy_data = LazyTensor.tensor(
                                    data=param.data,
                                    device=device_str,
                                    dtype=param.dtype,
                                    requires_grad=param.requires_grad
                                )
                                # Replace the entire parameter
                                new_param = torch.nn.Parameter(lazy_data, requires_grad=param.requires_grad)
                                setattr(module, name, new_param)
                                param_count += 1

                        # Recurse into child modules
                        for child_name, child_module in module.named_children():
                            convert_module_params(child_module, f"{prefix}.{child_name}" if prefix else child_name)

                    convert_module_params(self)

                    # Convert buffers - traverse module tree
                    buffer_count = 0
                    def convert_module_buffers(module, prefix=''):
                        nonlocal buffer_count
                        for name, buffer in module.named_buffers(recurse=False):
                            if not isinstance(buffer, LazyTensor):
                                # Create new LazyTensor for buffer
                                lazy_buffer = LazyTensor.tensor(
                                    data=buffer,
                                    device=device_str,
                                    dtype=buffer.dtype,
                                    requires_grad=False  # Buffers don't require gradients
                                )
                                setattr(module, name, lazy_buffer)
                                buffer_count += 1

                        # Recurse into child modules
                        for child_name, child_module in module.named_children():
                            convert_module_buffers(child_module, f"{prefix}.{child_name}" if prefix else child_name)

                    convert_module_buffers(self)

                    logger.info(f"Converted {param_count} parameters and {buffer_count} buffers to LazyTensors")
                    return self

            # Fall back to original for other devices
            return cls._original_module_to(self, *args, **kwargs)

        # Replace method
        nn.Module.to = patched_to

        # Add convenience method
        nn.Module.remote = lambda self: self.to('remote_accelerator:0')

        cls._initialized = True
        logger.info("Remote accelerator support initialized")


# Initialize on module import
def setup():
    """Setup function to be called from djinn.__init__"""
    RemoteAcceleratorSupport.initialize()
