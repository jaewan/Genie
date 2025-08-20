from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
from itertools import count
import os

import torch

logger = logging.getLogger(__name__)


from .semantic_metadata import SemanticMetadata
from .meta_utils import infer_via_meta


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

	def __init__(self, operation: str, inputs: List[Any], kwargs: Optional[Dict[str, Any]] = None) -> None:
		self.id = f"lt_{next(self._id_counter)}"
		self.operation = operation
		self.inputs = inputs
		self.kwargs = kwargs or {}

		# Infer tensor properties cheaply first
		self.shape = self._infer_shape()
		self.dtype = self._infer_dtype()
		self.device = self._infer_device()
		# Optional meta inference if unknown
		if (self.shape is None or self.dtype is None) and os.getenv("GENIE_ENABLE_META_INFER", "1") == "1":
			meta_shape, meta_dtype = infer_via_meta(operation, inputs, self.kwargs)
			if self.shape is None:
				self.shape = meta_shape
			if self.dtype is None:
				self.dtype = meta_dtype
		
		# Lazy init metadata to avoid upfront object allocation
		self._metadata = None

		# Execution state
		self.materialized = False
		self.concrete_value: Optional[torch.Tensor] = None
		
		# Register with graph builder
		from .graph import GraphBuilder
		GraphBuilder.current().add_tensor(self)

	def _infer_shape(self) -> Optional[torch.Size]:
		"""Infer output shape from operation and inputs."""
		try:
			if self.operation in ["aten::add", "aten::sub", "aten::mul", "aten::div"]:
				return self._infer_elementwise_shape()
			elif self.operation == "aten::matmul":
				return self._infer_matmul_shape()
			elif self.operation == "aten::mm":
				return self._infer_mm_shape()
			elif self.operation == "aten::conv2d":
				return self._infer_conv2d_shape()
			elif self.operation in ["aten::randn", "aten::zeros", "aten::ones"]:
				return self._infer_creation_shape()
			elif self.operation in ["aten::relu", "aten::sigmoid", "aten::tanh"]:
				# Activation functions preserve shape
				first = self.inputs[0] if self.inputs else None
				return getattr(first, "shape", None)
		except Exception as e:
			logger.debug(f"Shape inference failed for {self.operation}: {e}")
		return None

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
		return self.materialize().to(*args, **kwargs)

	def item(self):  # noqa: ANN201
		"""Get scalar value (triggers materialization)."""
		return self.materialize().item()

	def numpy(self):  # noqa: ANN201
		"""Convert to numpy (triggers materialization)."""
		return self.materialize().numpy()

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

	# String representation
	def __repr__(self) -> str:
		status = "materialized" if self.materialized else "lazy"
		return f"LazyTensor(op={self.operation}, shape={self.shape}, dtype={self.dtype}, {status})"

	def __str__(self) -> str:
		return self.__repr__()

	@property
	def metadata(self) -> SemanticMetadata:  # type: ignore[override]
		if self._metadata is None:
			self._metadata = SemanticMetadata(
				operation_type=self.operation,
				tensor_shape=self.shape,
				dtype=self.dtype,
				device_hint=self.device
			)
		return self._metadata


