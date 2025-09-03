from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
from itertools import count
import os

import torch

logger = logging.getLogger(__name__)


from .semantic_metadata import SemanticMetadata, ExecutionPhase, MemoryPattern, DataLineage
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

	@property
	def metadata(self) -> SemanticMetadata:  # type: ignore[override]
		if self._metadata is None:
			# Create rich semantic metadata as per HotNets'25 paper
			self._metadata = SemanticMetadata(
				operation_type=self.operation,
				tensor_shape=self.shape,
				dtype=self.dtype,
				device_hint=self.device,
				# Enhanced semantic enrichment
				semantic_role=self._infer_semantic_role(),
				model_module=self._get_module_context(),
				execution_phase=self._detect_phase(),
				data_lineage=self._track_lineage(),
				memory_pattern=self._analyze_memory_pattern(),
				# Additional context
				layer_depth=self._get_layer_depth(),
				is_activation=self._is_activation(),
				kv_cache_related=self._is_kv_cache_operation(),
				# Performance hints
				compute_intensity=self._estimate_compute_intensity(),
				can_parallelize=self._can_parallelize(),
				priority=self._calculate_priority()
			)
		return self._metadata
	
	def _infer_semantic_role(self) -> Optional[str]:
		"""Infer the semantic role of this operation in the model."""
		try:
			from genie.semantic.module_context import get_module_context_tracker
			tracker = get_module_context_tracker()
			context = tracker.get_current_context()
			
			if context:
				return tracker.infer_semantic_role(self.operation, context)
			
			# Fallback heuristics based on operation type
			op_name = self.operation.split("::")[-1]
			
			# Attention patterns
			if "matmul" in op_name and len(self.inputs) >= 2:
				# Check if this looks like attention
				if hasattr(self.inputs[0], "shape") and hasattr(self.inputs[1], "shape"):
					shape1 = self.inputs[0].shape
					shape2 = self.inputs[1].shape
					if len(shape1) >= 3 and len(shape2) >= 3:
						if shape1[-1] == shape2[-2]:  # Compatible for matmul
							return "attention_score_computation"
			
			# Activation functions
			if op_name in ["relu", "gelu", "sigmoid", "tanh", "softmax"]:
				return f"{op_name}_activation"
			
			# Normalization
			if "norm" in op_name or "layer_norm" in op_name:
				return "normalization"
			
			# Projection operations
			if op_name == "linear" or (op_name == "matmul" and len(self.inputs) == 2):
				return "linear_projection"
			
			return None
		except Exception as e:
			logger.debug(f"Failed to infer semantic role: {e}")
			return None
	
	def _get_module_context(self) -> Optional[str]:
		"""Get the module context where this operation occurs."""
		try:
			from genie.semantic.module_context import get_module_context_tracker
			tracker = get_module_context_tracker()
			context = tracker.get_current_context()
			
			if context:
				return context.module_path
			
			return None
		except Exception:
			return None
	
	def _detect_phase(self) -> Optional[ExecutionPhase]:
		"""Detect the execution phase (prefill/decode/fusion etc)."""
		try:
			# Try the new phase detector first
			from genie.semantic.phase_detector import get_phase_detector
			detector = get_phase_detector()
			
			# Pass operation and inputs to detector
			phase = detector.detect_phase(
				self.operation,
				self.inputs,
				metadata={
					'tensor_shape': self.shape,
					'dtype': self.dtype,
					'module_path': self._get_module_context()
				}
			)
			
			if isinstance(phase, ExecutionPhase):
				return phase
			
			# Fallback to module context tracker
			from genie.semantic.module_context import get_module_context_tracker
			tracker = get_module_context_tracker()
			phase_str = tracker.detect_execution_phase()
			
			# Map string to ExecutionPhase enum
			phase_map = {
				"prefill": ExecutionPhase.PREFILL,
				"decode": ExecutionPhase.DECODE,
				"vision_backbone": ExecutionPhase.VISION_BACKBONE,
				"vision_head": ExecutionPhase.VISION_HEAD,
				"multimodal_fusion": ExecutionPhase.MULTIMODAL_FUSION,
				"embedding": ExecutionPhase.EMBEDDING
			}
			
			return phase_map.get(phase_str, ExecutionPhase.UNKNOWN)
		except Exception:
			return ExecutionPhase.UNKNOWN
	
	def _track_lineage(self) -> Optional[DataLineage]:
		"""Track data lineage through the computation."""
		try:
			lineage = DataLineage()
			
			# Track source tensors and propagate transformations
			for inp in self.inputs:
				if isinstance(inp, LazyTensor):
					lineage.source_tensors.append(inp.id)
					# Propagate lineage from inputs
					if inp.metadata and inp.metadata.data_lineage:
						lineage.source_modules.extend(inp.metadata.data_lineage.source_modules)
						if inp.metadata.data_lineage.modality:
							lineage.modality = inp.metadata.data_lineage.modality
						# Propagate transformation chain from inputs
						lineage.transformation_chain.extend(inp.metadata.data_lineage.transformation_chain)
			
			# Add current module to lineage
			module_context = self._get_module_context()
			if module_context:
				lineage.source_modules.append(module_context)
			
			# Add current transformation
			lineage.transformation_chain.append(self.operation)
			
			# Infer modality from context
			if not lineage.modality:
				if module_context:
					if "vision" in module_context.lower() or "visual" in module_context.lower():
						lineage.modality = "vision"
					elif "text" in module_context.lower() or "language" in module_context.lower():
						lineage.modality = "text"
					elif "audio" in module_context.lower():
						lineage.modality = "audio"
			
			return lineage
		except Exception:
			return None
	
	def _analyze_memory_pattern(self) -> Optional[MemoryPattern]:
		"""Analyze the memory access pattern of this operation."""
		try:
			op_name = self.operation.split("::")[-1]
			
			# KV cache operations are persistent
			if self._is_kv_cache_operation():
				return MemoryPattern.PERSISTENT
			
			# Weights are typically reused
			if self._is_weight_tensor():
				return MemoryPattern.REUSED
			
			# Convolution and matmul have predictable streaming patterns
			if op_name in ["conv2d", "conv1d", "matmul", "mm", "bmm"]:
				return MemoryPattern.STREAMING
			
			# Random operations like dropout
			if "dropout" in op_name or "random" in op_name:
				return MemoryPattern.RANDOM
			
			# Intermediate activations (after checking specific ops)
			if self._is_activation():
				# Check if this feeds into multiple operations
				# (Would need graph analysis for accurate detection)
				return MemoryPattern.EPHEMERAL
			
			return MemoryPattern.STREAMING  # Default
		except Exception:
			return MemoryPattern.STREAMING
	
	def _get_layer_depth(self) -> Optional[int]:
		"""Get the layer depth in the model."""
		try:
			from genie.semantic.module_context import get_module_context_tracker
			tracker = get_module_context_tracker()
			context = tracker.get_current_context()
			
			if context:
				return context.layer_depth
			
			return None
		except Exception:
			return None
	
	def _is_activation(self) -> bool:
		"""Check if this is an activation tensor vs weight."""
		# Activations are typically outputs of operations on inputs
		# Weights are parameters that don't change during forward pass
		
		# Check if any input is a parameter (would be a weight operation)
		for inp in self.inputs:
			if isinstance(inp, torch.nn.Parameter):
				return False
		
		# Activation functions produce activations
		op_name = self.operation.split("::")[-1]
		if op_name in ["relu", "gelu", "sigmoid", "tanh", "softmax", "dropout"]:
			return True
		
		# Default to activation for dynamic operations
		return True
	
	def _is_weight_tensor(self) -> bool:
		"""Check if this operates on weight tensors."""
		for inp in self.inputs:
			if isinstance(inp, torch.nn.Parameter):
				return True
		return False
	
	def _is_kv_cache_operation(self) -> bool:
		"""Check if this is related to KV cache in LLMs."""
		module_context = self._get_module_context()
		if module_context:
			context_lower = module_context.lower()
			return any(kv in context_lower for kv in ["kv_cache", "past_key", "past_value", "key_cache", "value_cache"])
		
		# Check operation name
		op_lower = self.operation.lower()
		return any(kv in op_lower for kv in ["cache", "past"])
	
	def _estimate_compute_intensity(self) -> float:
		"""Estimate compute intensity (FLOPs/byte ratio)."""
		try:
			op_name = self.operation.split("::")[-1]
			
			# High compute intensity operations
			if op_name in ["matmul", "mm", "bmm", "conv2d", "conv1d"]:
				return 10.0  # High
			
			# Medium compute intensity
			if op_name in ["softmax", "layer_norm", "batch_norm"]:
				return 5.0
			
			# Low compute intensity (memory bound)
			if op_name in ["add", "mul", "concat", "transpose", "reshape"]:
				return 1.0
			
			return 2.0  # Default medium
		except Exception:
			return 1.0
	
	def _can_parallelize(self) -> bool:
		"""Check if this operation can be parallelized."""
		op_name = self.operation.split("::")[-1]
		
		# Operations that typically can't be parallelized
		sequential_ops = ["cumsum", "cumprod", "rnn", "lstm", "gru"]
		if op_name in sequential_ops:
			return False
		
		# Most operations can be parallelized
		return True
	
	def _calculate_priority(self) -> int:
		"""Calculate execution priority (higher = more urgent)."""
		priority = 0
		
		# Critical path operations get higher priority
		if self._is_kv_cache_operation():
			priority += 10  # KV cache is critical for LLM decode
		
		# High compute operations
		if self._estimate_compute_intensity() > 5.0:
			# Ensure high-compute ops outrank phase-only bumps (e.g., fusion=+6)
			priority += 7
		
		# Operations in certain phases
		phase = self._detect_phase()
		if phase == ExecutionPhase.DECODE:
			priority += 8  # Decode is latency sensitive
		elif phase == ExecutionPhase.MULTIMODAL_FUSION:
			priority += 6  # Fusion is on critical path
		
		return priority
	
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


