from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
from itertools import count
import os

import torch

logger = logging.getLogger(__name__)


from .meta_utils import infer_via_meta
from .exceptions import Result, ShapeInferenceError

# Note: SemanticMetadata now comes from metadata_registry (Refactoring #2)
# ExecutionPhase, MemoryPattern, DataLineage no longer used in LazyTensor


class LazyTensor:
	"""Deferred execution tensor with rich semantic metadata.
	
	LazyTensor captures PyTorch operations and builds a computation graph
	instead of executing immediately, enabling semantic analysis and 
	optimization before remote execution.
	"""

	# Micro-optimization: reduce per-instance dict by using slots
	__slots__ = (
		"id", "operation", "inputs", "kwargs",
		"shape", "dtype", "device",
		"materialized", "concrete_value", "_metadata",
	)

	# Faster lightweight ids
	_id_counter = count(1)
	
	@classmethod
	def reset_id_counter(cls):
		"""Reset ID counter for testing."""
		cls._id_counter = count(1)

	def __init__(self, operation: str, inputs: List[Any], kwargs: Optional[Dict[str, Any]] = None) -> None:
		self.id = f"lt_{next(self._id_counter)}"
		self.operation = self._normalize_aten_name(operation)
		self.inputs = inputs
		self.kwargs = kwargs or {}

		# Fast path for common element-wise ops to minimize per-op overhead
		self.shape = None
		self.dtype = None
		self.device = "remote_accelerator:0"
		_is_elementwise_fastpath = False
		_enable_fastpath = os.getenv("GENIE_ENABLE_ELEMENTWISE_FASTPATH", "0") == "1"
		if _enable_fastpath and self.operation in ("aten::add", "aten::sub", "aten::mul", "aten::div") and inputs:
			first = inputs[0]
			# Avoid function calls; copy from first input when available
			self.shape = getattr(first, "shape", None)
			self.dtype = getattr(first, "dtype", None)
			self.device = getattr(first, "device", self.device)
			_is_elementwise_fastpath = True
		else:
			# Infer tensor properties (still lightweight for non-elementwise ops)
			shape_result = self._infer_shape()
			if shape_result.is_ok:
				self.shape = shape_result.unwrap()
			else:
				# Log error but don't fail construction
				logger.debug(f"Shape inference failed for {self.operation}: {shape_result.error}")
				self.shape = None
			
			self.dtype = self._infer_dtype()
			self.device = self._infer_device()
			
			# Optional meta inference if unknown
			if (self.shape is None or self.dtype is None) and os.getenv("GENIE_ENABLE_META_INFER", "1") == "1":
				meta_shape, meta_dtype = infer_via_meta(operation, inputs, self.kwargs)
				if self.shape is None:
					self.shape = meta_shape
				if self.dtype is None:
					self.dtype = meta_dtype
		
		# No longer store metadata inline - use MetadataRegistry (Refactoring #2)
		# Metadata is managed separately for clean separation of concerns
		self._metadata = None

		# Execution state
		self.materialized = False
		self.concrete_value: Optional[torch.Tensor] = None
		
		# Register with metadata registry for semantic analysis (Refactoring #2)
		if not _is_elementwise_fastpath:
			self._register_for_metadata()
		
		# Optional debug logging for intercepted ops
		if os.getenv("GENIE_LOG_INTERCEPTS", "0") == "1":
			try:
				in_shapes = []
				for arg in inputs:
					shape = getattr(arg, "shape", None)
					in_shapes.append(tuple(shape) if shape is not None else None)
				logger.info(f"[Genie] Intercepted {self.operation} -> id={self.id}, in_shapes={in_shapes}, out_shape={self.shape}")
			except Exception:
				logger.info(f"[Genie] Intercepted {self.operation} -> id={self.id}")

		# Notify enhanced dispatcher only for non-fastpath ops
		if not _is_elementwise_fastpath:
			try:
				from .enhanced_dispatcher import enhanced_dispatcher
				if not enhanced_dispatcher.is_operation_registered(self.operation):
					enhanced_dispatcher.record_fallback_capture(self.operation)
			except Exception:
				pass

		# Register with graph builders.
		# For elementwise fastpath, skip registration entirely to minimize overhead.
		if not _is_elementwise_fastpath:
			try:
				from .fx_graph_builder import FXGraphBuilder
				FXGraphBuilder.current().add_lazy_tensor(self)
			except ImportError:
				pass
			# Always keep ComputationGraph in sync for pattern analyzers
			try:
				from .graph import GraphBuilder
				GraphBuilder.current().add_tensor(self)
			except Exception:
				pass

	def _infer_shape(self) -> Result[torch.Size]:
		"""
		Infer output shape from operation and inputs.
		
		Returns:
			Result[torch.Size]: Shape if successful, error otherwise
		"""
		try:
			shape = None
			
			if self.operation in ["aten::add", "aten::sub", "aten::mul", "aten::div"]:
				shape = self._infer_elementwise_shape()
			elif self.operation == "aten::matmul":
				shape = self._infer_matmul_shape()
			elif self.operation == "aten::mm":
				shape = self._infer_mm_shape()
			elif self.operation == "aten::conv2d":
				shape = self._infer_conv2d_shape()
			elif self.operation in ["aten::randn", "aten::zeros", "aten::ones"]:
				shape = self._infer_creation_shape()
			elif self.operation in ["aten::relu", "aten::sigmoid", "aten::tanh"]:
				# Activation functions preserve shape
				first = self.inputs[0] if self.inputs else None
				shape = getattr(first, "shape", None)
			
			if shape is None:
				return Result.err(ShapeInferenceError(
					f"Could not infer shape for {self.operation}",
					context={'operation': self.operation, 'inputs': len(self.inputs)}
				))
			
			return Result.ok(shape)
			
		except Exception as e:
			return Result.err(ShapeInferenceError(
				f"Shape inference raised exception: {e}",
				context={'operation': self.operation, 'error': str(e)}
			))

	def _infer_elementwise_shape(self) -> Optional[torch.Size]:
		"""Infer shape for element-wise operations with broadcasting."""
		if len(self.inputs) < 2:
			return None
		
		x_shape = getattr(self.inputs[0], "shape", None)
		y_shape = getattr(self.inputs[1], "shape", None)
		
		if x_shape is None or y_shape is None:
			return x_shape or y_shape
		
		# Simple broadcasting logic
		if x_shape == y_shape:
			return x_shape
		
		# Return the larger shape (simplified)
		if len(x_shape) >= len(y_shape):
			return x_shape
		return y_shape

	def _infer_matmul_shape(self) -> Optional[torch.Size]:
		"""Infer shape for matrix multiplication."""
		if len(self.inputs) < 2:
			return None
			
		x_shape = getattr(self.inputs[0], "shape", None)
		y_shape = getattr(self.inputs[1], "shape", None)
		
		if not (x_shape and y_shape):
			return None
		
		if len(x_shape) >= 2 and len(y_shape) >= 2:
			# Basic matmul: (..., a, b) @ (..., b, c) -> (..., a, c)
			return torch.Size([*x_shape[:-1], y_shape[-1]])
		
		return None

	def _infer_mm_shape(self) -> Optional[torch.Size]:
		"""Infer shape for 2D matrix multiplication."""
		if len(self.inputs) < 2:
			return None
			
		x_shape = getattr(self.inputs[0], "shape", None)
		y_shape = getattr(self.inputs[1], "shape", None)
		
		if x_shape and y_shape and len(x_shape) == 2 and len(y_shape) == 2:
			return torch.Size([x_shape[0], y_shape[1]])
		
		return None

	def _infer_conv2d_shape(self) -> Optional[torch.Size]:
		"""Infer shape for 2D convolution (simplified)."""
		if len(self.inputs) < 2:
			return None
		
		input_shape = getattr(self.inputs[0], "shape", None)
		weight_shape = getattr(self.inputs[1], "shape", None)
		
		if input_shape and weight_shape and len(input_shape) == 4 and len(weight_shape) == 4:
			# Simplified: assume same spatial dimensions (would need stride/padding for exact)
			N, C_in, H, W = input_shape
			C_out, C_in_w, K_h, K_w = weight_shape
			return torch.Size([N, C_out, H, W])  # Simplified
		
		return None

	def _infer_creation_shape(self) -> Optional[torch.Size]:
		"""Infer shape for tensor creation operations."""
		# For creation ops, shape is typically in the first arguments
		if self.inputs:
			# Handle both tuple and individual args
			if isinstance(self.inputs[0], (tuple, list)):
				return torch.Size(self.inputs[0])
			else:
				# Individual size arguments
				size_args = []
				for arg in self.inputs:
					if isinstance(arg, int):
						size_args.append(arg)
					else:
						break
				if size_args:
					return torch.Size(size_args)
		return None

	def _infer_dtype(self) -> Optional[torch.dtype]:
		"""Infer output dtype from inputs."""
		# Check kwargs first (explicit dtype)
		if "dtype" in self.kwargs and self.kwargs["dtype"] is not None:
			return self.kwargs["dtype"]
		
		# Infer from inputs
		for inp in self.inputs:
			if hasattr(inp, "dtype"):
				return inp.dtype  # type: ignore[return-value]
		
		# Default for creation operations
		if self.operation in ["aten::randn", "aten::zeros", "aten::ones"]:
			return torch.float32
		
		return None

	def _infer_device(self) -> str:
		"""Infer device from inputs or kwargs."""
		# Check kwargs first
		if "device" in self.kwargs and self.kwargs["device"] is not None:
			device = self.kwargs["device"]
			if isinstance(device, str):
				# Handle custom device types like "remote_accelerator:0"
				if "remote_accelerator" in device:
					return device
				# Try to create torch.device to validate
				try:
					torch.device(device)
					return device
				except RuntimeError:
					# Custom device type not recognized by PyTorch
					return device
			return str(device)

		# Infer from inputs
		for inp in self.inputs:
			if hasattr(inp, "device"):
				return str(inp.device)
		
		return "remote_accelerator:0"

	def materialize(self) -> torch.Tensor:
		"""Force materialization of this tensor."""
		if not self.materialized:
			from .executor import execute_subgraph

			self.concrete_value = execute_subgraph(self)
			self.materialized = True
		return self.concrete_value  # type: ignore[return-value]

	def add_metadata(self, key: str, value: Any, source: str = "user") -> None:
		"""Add custom metadata to this tensor."""
		md = self.metadata
		if not hasattr(md, 'custom'):
			md.custom = {}
		md.custom[key] = {"value": value, "source": source}

	def get_memory_footprint(self) -> int:
		"""Estimate memory footprint in bytes."""
		if self.shape and self.dtype:
			element_size = torch.tensor([], dtype=self.dtype).element_size()
			try:
				from math import prod
				return int(prod(self.shape)) * element_size
			except Exception:
				return int(torch.prod(torch.tensor(self.shape)).item()) * element_size
		return 0

	def is_materialized(self) -> bool:
		"""Check if tensor has been materialized."""
		return self.materialized

	# Tensor-like interface methods
	def cpu(self) -> torch.Tensor:
		"""Move tensor to CPU (triggers materialization)."""
		return self.materialize().cpu()

	def cuda(self, device: Optional[Union[int, str]] = None) -> torch.Tensor:
		"""Move tensor to CUDA (triggers materialization)."""
		return self.materialize().cuda(device)

	def to(self, *args, **kwargs) -> torch.Tensor:
		"""Convert tensor type/device (triggers materialization)."""
		# Special handling for remote_accelerator device to prevent allocation errors
		if args and isinstance(args[0], (str, torch.device)):
			device = args[0]
			if (isinstance(device, str) and device.startswith("remote_accelerator")) or \
			   (isinstance(device, torch.device) and device.type == "remote_accelerator"):
				# For remote_accelerator, just return self (no actual device change)
				# The actual device change will happen during materialization
				return self
		
		# For other devices, proceed normally
		return self.materialize().to(*args, **kwargs)

	def item(self):  # noqa: ANN201
		"""Get scalar value (triggers materialization)."""
		return self.materialize().item()

	def numpy(self):  # noqa: ANN201
		"""Convert to numpy (triggers materialization)."""
		return self.materialize().numpy()

	# Printing should not force a full tensor dump; keep representation light
	def __format__(self, format_spec: str) -> str:  # noqa: D401
		return self.__repr__()

	def size(self, dim: Optional[int] = None):  # noqa: ANN201
		"""Get tensor size."""
		if self.shape is not None:
			if dim is not None:
				return self.shape[dim]
			return self.shape
		# Fallback to materialization
		return self.materialize().size(dim)

	def dim(self) -> int:
		"""Get number of dimensions."""
		if self.shape is not None:
			return len(self.shape)
		return self.materialize().dim()

	def numel(self) -> int:
		"""Get total number of elements."""
		if self.shape is not None:
			try:
				from math import prod
				return int(prod(self.shape))
			except Exception:
				return int(torch.prod(torch.tensor(self.shape)).item())
		return self.materialize().numel()

	# Arithmetic operators
	def __add__(self, other):  # noqa: ANN001, ANN201
		return LazyTensor("aten::add", [self, other])

	def __sub__(self, other):  # noqa: ANN001, ANN201
		return LazyTensor("aten::sub", [self, other])

	def __mul__(self, other):  # noqa: ANN001, ANN201
		return LazyTensor("aten::mul", [self, other])

	def __truediv__(self, other):  # noqa: ANN001, ANN201
		return LazyTensor("aten::div", [self, other])

	def __matmul__(self, other):  # noqa: ANN001, ANN201
		return LazyTensor("aten::matmul", [self, other])

	# Activation functions
	def relu(self):  # noqa: ANN201
		return LazyTensor("aten::relu", [self])

	def sigmoid(self):  # noqa: ANN201
		return LazyTensor("aten::sigmoid", [self])

	def tanh(self):  # noqa: ANN201
		return LazyTensor("aten::tanh", [self])

	# Common tensor methods that should create new LazyTensors
	def unsqueeze(self, dim: int):  # noqa: ANN201
		return LazyTensor("aten::unsqueeze", [self, dim])
	
	def squeeze(self, dim: Optional[int] = None):  # noqa: ANN201
		if dim is None:
			return LazyTensor("aten::squeeze", [self])
		return LazyTensor("aten::squeeze", [self, dim])
	
	def reshape(self, *shape):  # noqa: ANN201
		return LazyTensor("aten::reshape", [self, shape])
	
	def view(self, *shape):  # noqa: ANN201
		return LazyTensor("aten::view", [self, shape])
	
	def transpose(self, dim0: int, dim1: int):  # noqa: ANN201
		return LazyTensor("aten::transpose", [self, dim0, dim1])
	
	def permute(self, *dims):  # noqa: ANN201
		return LazyTensor("aten::permute", [self, dims])
	
	def sum(self, dim=None, keepdim=False, dtype=None):  # noqa: ANN201
		return LazyTensor("aten::sum", [self], {"dim": dim, "keepdim": keepdim, "dtype": dtype})
	
	def mean(self, dim=None, keepdim=False, dtype=None):  # noqa: ANN201
		return LazyTensor("aten::mean", [self], {"dim": dim, "keepdim": keepdim, "dtype": dtype})
	
	def max(self, dim=None, keepdim=False):  # noqa: ANN201
		return LazyTensor("aten::max", [self], {"dim": dim, "keepdim": keepdim})
	
	def prod(self, dim=None, keepdim=False, dtype=None):  # noqa: ANN201
		return LazyTensor("aten::prod", [self], {"dim": dim, "keepdim": keepdim, "dtype": dtype})

	def var(self, dim=None, keepdim=False, unbiased=True):  # noqa: ANN201
		return LazyTensor("aten::var", [self], {"dim": dim, "keepdim": keepdim, "unbiased": unbiased})

	def std(self, dim=None, keepdim=False, unbiased=True):  # noqa: ANN201
		return LazyTensor("aten::std", [self], {"dim": dim, "keepdim": keepdim, "unbiased": unbiased})

	def all(self, dim=None, keepdim=False):  # noqa: ANN201
		return LazyTensor("aten::all", [self], {"dim": dim, "keepdim": keepdim})

	def any(self, dim=None, keepdim=False):  # noqa: ANN201
		return LazyTensor("aten::any", [self], {"dim": dim, "keepdim": keepdim})

	def argmax(self, dim=None, keepdim=False):  # noqa: ANN201
		return LazyTensor("aten::argmax", [self], {"dim": dim, "keepdim": keepdim})

	def argmin(self, dim=None, keepdim=False):  # noqa: ANN201
		return LazyTensor("aten::argmin", [self], {"dim": dim, "keepdim": keepdim})

	def softmax(self, dim, dtype=None):  # noqa: ANN201
		return LazyTensor("aten::softmax", [self], {"dim": dim, "dtype": dtype})
	
	def abs(self):  # noqa: ANN201
		return LazyTensor("aten::abs", [self])
	
	def exp(self):  # noqa: ANN201
		return LazyTensor("aten::exp", [self])
	
	def log(self):  # noqa: ANN201
		return LazyTensor("aten::log", [self])
	
	def sin(self):  # noqa: ANN201
		return LazyTensor("aten::sin", [self])
	
	def cos(self):  # noqa: ANN201
		return LazyTensor("aten::cos", [self])
	
	def sqrt(self):  # noqa: ANN201
		return LazyTensor("aten::sqrt", [self])
	
	def pow(self, exponent):  # noqa: ANN201
		return LazyTensor("aten::pow", [self, exponent])
	
	def clamp(self, min=None, max=None):  # noqa: ANN201
		return LazyTensor("aten::clamp", [self], {"min": min, "max": max})

	# String representation
	def __repr__(self) -> str:
		status = "materialized" if self.materialized else "lazy"
		return f"LazyTensor(op={self.operation}, shape={self.shape}, dtype={self.dtype}, {status})"

	def __str__(self) -> str:
		# Optional: print-based materialization (explicit trigger)
		if os.getenv("GENIE_PRINT_MATERIALIZE", "0") == "1":
			try:
				val = self.materialize()
				return f"LazyTensor(value={val}, shape={tuple(val.shape)}, dtype={val.dtype})"
			except Exception:
				pass
		return self.__repr__()

	# Control-flow triggers (implicit)
	def __bool__(self) -> bool:  # noqa: D401
		"""Truthiness triggers materialization to support control flow."""
		try:
			val = self.materialize()
			return bool(val.numel())
		except Exception:
			return True

	def __len__(self) -> int:
		if self.shape is not None and len(self.shape) > 0:
			return self.shape[0]
		return int(self.materialize().size(0))

	def _register_for_metadata(self):
		"""Register this tensor with metadata registry for semantic analysis (Refactoring #2)."""
		try:
			from genie.semantic.metadata_registry import get_metadata_registry
			from genie.semantic.enricher import get_semantic_enricher
			
			registry = get_metadata_registry()
			registry.register(self.id, self)
			
			# Trigger semantic enrichment (async/lazy)
			enricher = get_semantic_enricher()
			metadata_result = enricher.enrich(
				operation=self.operation,
				inputs=self.inputs,
				kwargs=self.kwargs,
				shape=self.shape,
				dtype=self.dtype
			)
			
			if metadata_result.is_ok:
				registry.set_metadata(self.id, metadata_result.unwrap())
			else:
				logger.debug(f"Metadata enrichment failed for {self.id}: {metadata_result.error}")
		
		except Exception as e:
			logger.debug(f"Could not register metadata for {self.id}: {e}")
	
	@property
	def metadata(self) -> Optional[SemanticMetadata]:  # type: ignore[override]
		"""
		Get semantic metadata for this tensor (Refactoring #2).
		
		Metadata is stored separately in MetadataRegistry and retrieved on demand.
		This coordinates with Refactoring #3: FX Graph stores tensor_id in meta dict.
		"""
		try:
			from genie.semantic.metadata_registry import get_metadata_registry
			registry = get_metadata_registry()
			return registry.get_metadata(self.id)
		except Exception:
			return None
	
	def __del__(self):
		"""Cleanup metadata when tensor is garbage collected (Refactoring #2)."""
		try:
			from genie.semantic.metadata_registry import get_metadata_registry
			registry = get_metadata_registry()
			registry.remove(self.id)
		except Exception:
			pass  # Ignore errors during cleanup
	
	def prepare_for_transfer(self) -> Dict:
		"""Prepare tensor metadata for DPDK backend transfer."""
		return {
			"tensor_id": self.id,
			"shape": list(self.shape) if self.shape else [],
			"dtype": str(self.dtype) if self.dtype else None,
			"device": self.device,
			"semantic_metadata": self.metadata.to_dict() if self.metadata else {}
		}

	# PyTorch Function Protocol for comprehensive operation interception
	@classmethod
	def __torch_function__(cls, func, types, args=(), kwargs=None):
		"""Intercept all torch function calls involving LazyTensor.
		
		This implements PyTorch's __torch_function__ protocol to automatically
		capture >95% of operations without manual registration, as per the spec.
		"""
		kwargs = kwargs or {}
		
		# Extract function name for operation tracking
		func_name = getattr(func, '__name__', str(func))
		if hasattr(func, '_schema'):
			# Use schema name if available (more precise)
			op_name = str(func._schema).split('(')[0]
		else:
			# Fallback to function name
			op_name = f"aten::{func_name}" if not func_name.startswith("aten::") else func_name
		# Normalize to canonical aten::op
		op_name = cls._normalize_aten_name(op_name)
		
		# Check if any arguments are LazyTensors
		has_lazy = any(isinstance(arg, cls) for arg in args) or any(isinstance(v, cls) for v in kwargs.values())
		
		if has_lazy:
			# Create new LazyTensor for this operation
			return cls(operation=op_name, inputs=list(args), kwargs=kwargs)
		else:
			# No LazyTensors involved, execute normally
			return func(*args, **kwargs)

	@staticmethod
	def _normalize_aten_name(full_op: str) -> str:
		"""Normalize aten op names by stripping overload suffixes.

		Example: "aten::add.Tensor" -> "aten::add"
		         "aten::softmax.int" -> "aten::softmax"
		         "matmul" -> "aten::matmul"
		"""
		try:
			if not full_op:
				return full_op
			if not full_op.startswith("aten::"):
				full_op = f"aten::{full_op}"
			_, name = full_op.split("::", 1)
			base = name.split(".", 1)[0]
			return f"aten::{base}"
		except Exception:
			return full_op

	@staticmethod
	def lift(tensor: torch.Tensor) -> "LazyTensor":
		"""Create a LazyTensor that aliases a concrete CPU tensor without allocating on remote.

		This wraps the tensor using a no-op alias op, enabling graphs that start from
		concrete tensors (for correctness comparisons) while still flowing through the
		lazy pipeline.
		"""
		return LazyTensor("aten::alias", [tensor], {})


# Factory function interception for torch.randn, torch.zeros, etc.
def _is_remote_device(dev: Any) -> bool:
	"""Check if device is remote_accelerator."""
	if dev is None:
		return False
	if isinstance(dev, str):
		return dev.startswith("remote_accelerator") or dev.startswith("privateuseone")
	if isinstance(dev, torch.device):
		return dev.type in ("privateuseone", "remote_accelerator")
	return False


def _install_factory_interceptor(fn_name: str) -> None:
	"""Install interceptor for factory functions like torch.randn."""
	original: Callable[..., Any] = getattr(torch, fn_name)

	def wrapper(*args: Any, **kwargs: Any):
		device = kwargs.get("device", None)
		# Interception: remote device or LazyTensor inputs
		has_lazy_input = any(isinstance(a, LazyTensor) for a in args) or any(isinstance(v, LazyTensor) for v in kwargs.values())
		if _is_remote_device(device) or has_lazy_input:
			op_name = f"aten::{fn_name}"
			lt = LazyTensor(operation=op_name, inputs=list(args), kwargs=kwargs)
			import os
			if os.getenv("GENIE_LOG_INTERCEPTS", "0") == "1":
				try:
					shapes = [getattr(a, "shape", None) for a in args]
					logger.info(f"[Genie] Factory intercepted torch.{fn_name} device={device}, args_shapes={shapes} -> {lt}")
				except Exception:
					logger.info(f"[Genie] Factory intercepted torch.{fn_name} device={device}")
			return lt
		return original(*args, **kwargs)

	setattr(torch, fn_name, wrapper)
	logger.debug(f"Installed factory interceptor for torch.{fn_name}")


def install_factory_interceptors() -> None:
	"""Install interceptors for common factory functions."""
	for name in (
		"randn", "rand", "randint", "zeros", "ones", "empty", "full", "empty_strided",
		"rand_like", "randint_like", "zeros_like", "ones_like", "empty_like", "full_like",
		"arange", "linspace", "logspace",
	):
		try:
			_install_factory_interceptor(name)
		except Exception:
			logger.debug(f"Could not install interceptor for torch.{name}")


