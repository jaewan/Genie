import torch
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)


class RemoteAcceleratorDevice:
	"""Custom PyTorch device for disaggregated execution.
	
	This device integrates with PyTorch's PrivateUse1 backend system to
	intercept tensor operations and create LazyTensors instead of executing
	eagerly, enabling semantic capture and remote execution.
	"""

	_devices = {}
	_registered = False
	_backend_registered = False

	def __init__(self, index: int = 0):
		self.type = "remote_accelerator"
		self.index = index
		self._register_backend()
		
		# Try to create torch device after backend registration
		try:
			self._torch_device = torch.device("remote_accelerator", index)
		except RuntimeError:
			# Fallback to privateuseone if remote_accelerator not recognized
			logger.warning("remote_accelerator not recognized, using privateuseone")
			self._torch_device = torch.device("privateuseone", index)
		
		self._devices[self.index] = self

	def _register_backend(self):
		"""Register the remote_accelerator backend with PyTorch."""
		if not RemoteAcceleratorDevice._backend_registered:
			try:
				from genie import _C  # type: ignore
				
				# Register the C++ backend
				_C.register_remote_accelerator_device()
				
				# Register Python-side hooks for LazyTensor creation
				self._register_python_hooks()
				
				RemoteAcceleratorDevice._backend_registered = True
				logger.info("Successfully registered remote_accelerator backend")
				
			except Exception as e:
				logger.error(f"Failed to register remote_accelerator backend: {e}")
				raise RuntimeError(f"Cannot initialize remote_accelerator device: {e}")

	def _register_python_hooks(self):
		"""Register Python hooks to intercept operations and create LazyTensors."""
		# For Phase 1, torch.library registrations handle interception.
		# We keep a lightweight placeholder to maintain API compatibility
		# without importing the enhanced dispatcher (avoids double registration).
		def _noop_hook(*_args, **_kwargs):  # noqa: D401
			"""No-op hook; interception handled via torch.library impls."""
			return None
		self._tensor_creation_hook = _noop_hook

	@classmethod
	def get_device(cls, index: int = 0) -> "RemoteAcceleratorDevice":
		"""Get or create a remote accelerator device."""
		if index not in cls._devices:
			cls._devices[index] = cls(index)
		return cls._devices[index]

	@classmethod
	def device_count(cls) -> int:
		"""Return number of available remote accelerator devices."""
		try:
			from genie import _C  # type: ignore
			return _C.device_count()
		except Exception:
			return 4  # Default from specs

	@classmethod
	def is_available(cls) -> bool:
		"""Check if remote_accelerator backend is available."""
		try:
			from genie import _C  # type: ignore
			return _C.is_backend_registered()
		except Exception:
			return cls._backend_registered

	def to_torch_device(self) -> torch.device:
		"""Convert to PyTorch device object."""
		return self._torch_device

	def synchronize(self) -> None:
		"""Synchronize device operations (no-op for Phase 1)."""
		# In Phase 1, this is a no-op since we don't have actual remote execution
		pass

	def memory_stats(self) -> dict:
		"""Get memory statistics for this device."""
		# Placeholder for Phase 1
		return {
			"allocated": 0,
			"cached": 0,
			"reserved": 0,
			"device_index": self.index
		}

	def __repr__(self) -> str:
		return f"remote_accelerator:{self.index}"
	
	def __str__(self) -> str:
		return f"remote_accelerator:{self.index}"

	def __eq__(self, other) -> bool:
		if isinstance(other, RemoteAcceleratorDevice):
			return self.index == other.index
		elif isinstance(other, torch.device):
			return (other.type == "remote_accelerator" and 
					other.index == self.index)
		return False

	def __hash__(self) -> int:
		return hash((self.type, self.index))


# Utility functions for device management
def get_device_count() -> int:
	"""Get the number of available remote_accelerator devices."""
	return RemoteAcceleratorDevice.device_count()


def is_available() -> bool:
	"""Check if remote_accelerator backend is available."""
	return RemoteAcceleratorDevice.is_available()


def get_device(index: int = 0) -> RemoteAcceleratorDevice:
	"""Get a remote_accelerator device by index."""
	return RemoteAcceleratorDevice.get_device(index)


def synchronize(device: Optional[Union[int, RemoteAcceleratorDevice]] = None) -> None:
	"""Synchronize operations on the specified device."""
	if device is None:
		device = 0
	
	if isinstance(device, int):
		device = get_device(device)
	
	device.synchronize()


# Initialize default device on module import
try:
	_default_device = RemoteAcceleratorDevice.get_device(0)
	logger.info("Initialized default remote_accelerator device")
except Exception as e:
	logger.warning(f"Could not initialize default remote_accelerator device: {e}")
	_default_device = None


