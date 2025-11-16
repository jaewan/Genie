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
    
    Uses PyTorch's PrivateUse1 backend system to register remote_accelerator
    as a valid device type. This allows PyTorch functions to accept
    device='remote_accelerator:0' without validation errors.
    """
    try:
        # Register remote_accelerator using PyTorch's PrivateUse1 backend
        # This makes PyTorch recognize 'remote_accelerator' as a valid device
        if hasattr(torch.utils, 'rename_privateuse1_backend'):
            torch.utils.rename_privateuse1_backend("remote_accelerator")
            logger.info("✅ Registered remote_accelerator device via PrivateUse1 backend")
        else:
            # PyTorch version doesn't support rename_privateuse1_backend
            logger.warning("⚠️  PyTorch version doesn't support PrivateUse1 backend renaming")
            logger.info("✅ Python device fallback initialized (no registration needed)")
        
        # ✅ FIX: Device string mapping happens in LazyTensor constructor
        # PyTorch's device validation is handled by PrivateUse1 backend registration above
        # The ModuleNotFoundError for torch.remote_accelerator is expected and harmless
        # since we use string-based device handling in LazyTensor, not module imports
        
        return True
    except Exception as e:
        logger.warning(f"⚠️  Failed to register device backend: {e}")
        logger.info("✅ Python device fallback initialized (will work around validation)")
        return False


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

                    # ✅ DESIGN FIX: Move tensors to CPU during LazyTensor conversion
                    # For GPU disaggregation, models should stay on CPU (not GPU)
                    # This eliminates unnecessary GPU→CPU copy during registration
                    has_gpu_params = any(p.device.type == 'cuda' for p in self.parameters())
                    if has_gpu_params:
                        logger.info(
                            "Model is on GPU - moving to CPU for GPU disaggregation. "
                            "For best performance, load models on CPU and move directly to 'remote_accelerator:0'"
                        )
                    
                    for name, param in self.named_parameters():
                        # ✅ DESIGN FIX: Move to CPU before converting to LazyTensor
                        # This ensures models stay on CPU (correct for GPU disaggregation)
                        # and eliminates GPU→CPU copy during registration
                        if param.device.type == 'cuda':
                            param_cpu = param.cpu()
                            logger.debug(f"Moving parameter {name} from GPU to CPU (required for GPU disaggregation)")
                        else:
                            param_cpu = param
                        
                        # ✅ ELEGANT FIX: Explicitly preserve dtype during conversion
                        # This ensures model parameters maintain their dtype (e.g., float16 for GPT2-XL)
                        lazy_params[name] = LazyTensor.tensor(
                            param_cpu.detach(),
                            dtype=param.dtype,  # Preserve dtype from original param
                            device='cpu'  # ✅ FIX: Explicitly CPU (correct for GPU disaggregation)
                        )

                    for name, buffer in self.named_buffers():
                        # ✅ DESIGN FIX: Move buffers to CPU too
                        if buffer.device.type == 'cuda':
                            buffer_cpu = buffer.cpu()
                            logger.debug(f"Moving buffer {name} from GPU to CPU (required for GPU disaggregation)")
                        else:
                            buffer_cpu = buffer
                        
                        # ✅ ELEGANT FIX: Explicitly preserve dtype for buffers too
                        lazy_buffers[name] = LazyTensor.tensor(
                            buffer_cpu.detach(),
                            dtype=buffer.dtype,  # Preserve dtype from original buffer
                            device='cpu'  # ✅ FIX: Explicitly CPU (correct for GPU disaggregation)
                        )

                    # Replace parameters and buffers
                    for name, lazy_param in lazy_params.items():
                        # ✅ FIX: Create new Parameter with LazyTensor data
                        # PyTorch's Parameter.set_data() doesn't accept LazyTensor directly
                        # So we need to create a new Parameter object
                        param = dict(self.named_parameters())[name]
                        new_param = nn.Parameter(lazy_param, requires_grad=param.requires_grad)
                        # Use setattr to replace the parameter in the module
                        parts = name.split('.')
                        if len(parts) == 1:
                            setattr(self, name, new_param)
                        else:
                            # Navigate to the parent module
                            parent = self
                            for part in parts[:-1]:
                                parent = getattr(parent, part)
                            setattr(parent, parts[-1], new_param)

                    for name, lazy_buffer in lazy_buffers.items():
                        # ✅ FIX: Buffers can be set directly, but let's be safe
                        buffer = dict(self.named_buffers())[name]
                        parts = name.split('.')
                        if len(parts) == 1:
                            setattr(self, name, lazy_buffers[name])
                        else:
                            # Navigate to the parent module
                            parent = self
                            for part in parts[:-1]:
                                parent = getattr(parent, part)
                            setattr(parent, parts[-1], lazy_buffers[name])

                    # Track conversion
                    param_count = len(lazy_params)
                    buffer_count = len(lazy_buffers)
                    logger.info(f"Converted {param_count} parameters and {buffer_count} buffers to LazyTensors")

                    # ✅ NEW: Auto-register model on device transfer (smart registration)
                    # We have the model object here, so we know the boundary and user intent
                    try:
                        self._auto_register_on_device_transfer()
                    except Exception as e:
                        # Don't fail device transfer if auto-registration fails
                        logger.debug(f"Auto-registration on device transfer failed: {e}")

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

        # Add auto-registration method to nn.Module
        def _auto_register_on_device_transfer(self):
            """Auto-register model when moved to remote device (if safe)."""
            try:
                import asyncio
                import threading
                from djinn.core.model_fingerprint import ModelFingerprint
                from djinn.core.enhanced_model_manager import EnhancedModelManager
                from djinn.core.auto_registration_policy import AutoRegistrationPolicy
                
                # Compute fingerprint
                fingerprint = ModelFingerprint.compute(self)
                
                # Check if should auto-register (size/memory checks, but no usage threshold)
                # Device transfer = explicit user intent, so we can register immediately
                policy = AutoRegistrationPolicy(
                    usage_threshold=1,  # No usage requirement (device transfer = intent)
                    max_size_mb=500,    # Size limit still applies
                    require_memory_check=True,
                    enabled=True
                )
                
                if policy.should_auto_register(fingerprint, self, usage_count=1):
                    logger.info(
                        f"🚀 Auto-registering model {fingerprint[:16]}... "
                        f"on device transfer (size/memory checks passed)"
                    )
                    
                    # Start background registration (non-blocking)
                    # Device transfer is synchronous, so we need to handle async registration
                    def register_in_background():
                        """Register model in background thread with its own event loop."""
                        try:
                            # Create new event loop for this thread
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            
                            try:
                                # Create manager and register
                                manager = EnhancedModelManager()
                                new_loop.run_until_complete(
                                    manager._register_model_async(self, fingerprint)
                                )
                            finally:
                                new_loop.close()
                        except Exception as e:
                            logger.debug(f"Background registration failed: {e}")
                    
                    # Start registration in background thread (non-blocking)
                    thread = threading.Thread(target=register_in_background, daemon=True)
                    thread.start()
            except Exception as e:
                logger.debug(f"Auto-registration on device transfer failed: {e}")
                # Don't raise - device transfer should succeed even if registration fails
        
        # Attach method to nn.Module
        nn.Module._auto_register_on_device_transfer = _auto_register_on_device_transfer
        
        # Apply the patch
        nn.Module.to = patched_to
        cls._initialized = True

        logger.info("✅ Remote accelerator device support initialized")


def setup():
    """Setup device compatibility layer."""
    RemoteAcceleratorSupport.initialize()