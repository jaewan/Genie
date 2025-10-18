from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional

import torch

from .errors import MaterializationError, UnsupportedOperationError

# Global executor instance (singleton for performance)
_executor: Optional['SimpleExecutor'] = None
_executor_lock = threading.Lock()  # Protects executor creation and access

logger = logging.getLogger(__name__)

# Thread-local state to track when we're inside executor materialization
# This tells the factory interceptor not to intercept tensor creation
_in_executor = threading.local()


class SimpleExecutor:
	"""Simple local executor for Phase 1.
	
	Executes LazyTensor graphs eagerly on CPU for validation and testing.
	In later phases, this will be replaced with remote execution.
	"""

	def __init__(self):
		self.execution_count = 0
		self._recursion_depth = 0  # Track recursion depth
		self.operation_handlers = self._build_operation_handlers()
		self._execution_lock = threading.Lock()  # Protects execution state
		self._executing_threads = set()  # Track which threads are executing

		# ✅ ADD: Statistics tracking
		self.stats = {
			'total_executions': 0,
			'ops_executed': {},  # op_name -> count
			'failures': [],  # List of (op_name, error_msg)
		}

	def _build_operation_handlers(self) -> Dict[str, callable]:
		"""Build mapping of operations to handler functions."""
		return {
			# Arithmetic operations
			"aten::add": self._execute_add,
			"aten::sub": self._execute_sub,
			"aten::mul": self._execute_mul,
			"aten::div": self._execute_div,

			# No-op alias (used by lift)
			"aten::alias": self._execute_alias,
			
			# Linear algebra
			"aten::matmul": self._execute_matmul,
			# NOTE: Intentionally omit explicit handlers for mm/bmm to exercise unified fallback
			
			# Tensor creation
			"aten::randn": self._execute_randn,
			"aten::zeros": self._execute_zeros,
			"aten::ones": self._execute_ones,
			
			# Activations
			"aten::relu": self._execute_relu,
			"aten::sigmoid": self._execute_sigmoid,
			"aten::tanh": self._execute_tanh,
			
			# Convolution
			"aten::conv2d": self._execute_conv2d,
			
			# Reductions
			"aten::sum": self._execute_sum,
			"aten::mean": self._execute_mean,
			"aten::var": self._execute_var,
			"aten::std": self._execute_std,
			"aten::argmax": self._execute_argmax,
			"aten::argmin": self._execute_argmin,
			"aten::softmax": self._execute_softmax,
		}

	def _ensure_concrete(self, value: Any) -> Any:  # noqa: ANN001
		"""Convert LazyTensor to concrete tensor if needed."""
		if type(value).__name__ == 'LazyTensor':
			# Call materialize() directly to avoid recursion through __torch_function__
			return value.materialize()
		elif type(value).__name__ == 'Tensor':
			return value.cpu() if value.device.type != 'cpu' else value
		else:
			return value

	def execute_subgraph(self, target_lazy_tensor) -> torch.Tensor:  # noqa: ANN001
		"""Execute computation graph up to target tensor.

		For Phase 1, uses simple direct recursive evaluation on LazyTensor chain.
		Remote execution and FX-based execution are Phase 2+ features.
		"""
		current_thread = threading.get_ident()

		with self._execution_lock:
			if current_thread in self._executing_threads:
				raise RuntimeError("Recursive execution detected")

			self.execution_count += 1
			self.stats['total_executions'] += 1
			self._executing_threads.add(current_thread)

		_in_executor.active = True
		try:
			# Phase 1: Simple direct recursive evaluation
			from .lazy_tensor import LazyTensor as _LT

			# Use function parameter for depth tracking (thread-safe, correct for nested calls)
			def _compute_lazy(lt: _LT, depth: int = 0) -> torch.Tensor:
				"""Recursively compute LazyTensor with depth tracking."""
				# Guard against infinite recursion
				if depth > 1000:
					raise RuntimeError(
						f"Computation graph too deep (>1000 levels): {lt.operation}\n"
						"This may indicate a cycle or extremely deep graph."
					)

				# Materialize all inputs first
				resolved_inputs = []
				for arg in lt.inputs:
					if isinstance(arg, _LT):
						resolved_inputs.append(_compute_lazy(arg, depth + 1))  # ← Pass depth
					else:
						resolved_inputs.append(arg)

				# Execute the operation using the resolved inputs
				try:
					op_func = self.operation_handlers.get(lt.operation)
					if op_func:
						return op_func(lt, resolved_inputs, lt.kwargs)
					else:
						# Fallback for unhandled operations using executor's fallback mechanism
						return self._execute_fallback_eager(lt.operation, resolved_inputs, lt.kwargs)
				except Exception as e:
					logger.error(f"Failed to execute operation {lt.operation}: {e}")
					# Let UnsupportedOperationError propagate (it's actionable)
					if isinstance(e, UnsupportedOperationError):
						raise
					# Wrap other errors as MaterializationError
					raise MaterializationError(f"Execution failed for {lt.operation}") from e

			return _compute_lazy(target_lazy_tensor, depth=0)
		finally:
			with self._execution_lock:
				self._executing_threads.discard(current_thread)
			_in_executor.active = False # Reset flag after execution

	def get_stats(self) -> Dict[str, Any]:
		"""Get executor statistics."""
		return {
			'execution_count': self.execution_count,
			'supported_operations': len(self.operation_handlers),
			'total_executions': self.stats['total_executions'],
			'ops_executed': self.stats['ops_executed'],
			'failure_count': len(self.stats['failures']),
			'recent_failures': self.stats['failures'][-10:],  # Last 10
		}

	def _track_operation(self, op_name: str):
		"""Track operation execution for statistics."""
		if op_name not in self.stats['ops_executed']:
			self.stats['ops_executed'][op_name] = 0
		self.stats['ops_executed'][op_name] += 1

	def _execute_fallback_eager(self, op_name: str, inputs, kwargs) -> torch.Tensor:
		"""Fallback to eager execution using torch.ops.aten or torch API.

		This implements the graceful degradation described in the spec: if an
		operation isn't intercepted, materialize inputs and run it eagerly.
		"""
		# Track operation
		if op_name not in self.stats['ops_executed']:
			self.stats['ops_executed'][op_name] = 0
		self.stats['ops_executed'][op_name] += 1
		# Ensure all inputs are concrete tensors; avoid nested materialization recursion
		concrete_inputs = []
		for inp in inputs:
			if isinstance(inp, torch.Tensor):
				concrete_inputs.append(inp)
			else:
				# Try to use pre-materialized value
				val = getattr(inp, "concrete_value", None)
				if val is not None and isinstance(val, torch.Tensor):
					concrete_inputs.append(val)
				else:
					concrete_inputs.append(inp)

		# Only pass device for creation ops; strip for others to avoid invalid kwargs
		kwargs = kwargs.copy() if kwargs else {}
		creation_ops = {
			"randn", "rand", "randint",
			"zeros", "ones", "empty", "full", "empty_strided",
			"arange", "linspace", "logspace",
		}
		aten_prefix = "aten::"
		base_name = op_name[len(aten_prefix):] if op_name.startswith(aten_prefix) else op_name
		if base_name in creation_ops:
			kwargs["device"] = "cpu"
		else:
			kwargs.pop("device", None)
		
		# Also ensure any input tensors are moved to CPU to prevent device conflicts
		concrete_inputs = [inp.cpu() if type(inp).__name__ == 'Tensor' else inp for inp in concrete_inputs]

		# Normalize op name (e.g., "aten::add" -> "add")
		aten_prefix = "aten::"
		base_name = op_name[len(aten_prefix):] if op_name.startswith(aten_prefix) else op_name

		# Try torch.ops.aten first
		try:
			aten_ns = getattr(torch.ops, "aten")
			aten_op = getattr(aten_ns, base_name)
			# For reductions with dim provided, aten returns (values, indices) for some ops; handle consistently
			out = aten_op(*concrete_inputs, **kwargs)
			# Normalize common tuple returns to tensor where tests expect tensor
			if base_name in {"max", "min", "argmax", "argmin"} and isinstance(out, tuple) and len(out) > 0:
				return out[0]
			return out
		except Exception as e_ops:
			logger.debug(f"torch.ops.aten fallback failed for {op_name}: {e_ops}")
			# Try functional/torch namespace as secondary fallback
			try:
				torch_api = getattr(torch, base_name)
				out = torch_api(*concrete_inputs, **kwargs)
				if base_name in {"max", "min", "argmax", "argmin"} and isinstance(out, tuple) and len(out) > 0:
					return out[0]
				return out
			except Exception as e_torch:
				# Track failure
				self.stats['failures'].append((op_name, str(e_torch)))

				# ✅ FAIL LOUD with actionable error
				raise UnsupportedOperationError(
					f"Operation '{op_name}' failed to execute.\n"
					f"  Inputs: {[type(i).__name__ for i in concrete_inputs]}\n"
					f"  Tried: torch.ops.aten.{base_name}, torch.{base_name}\n"
					f"  Add handler in SimpleExecutor.operation_handlers\n"
					f"  torch.ops error: {e_ops}\n"
					f"  torch API error: {e_torch}"
				) from e_torch

	# Operation handlers
	def _execute_add(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		self._track_operation("aten::add")
		if len(inputs) < 2:
			return torch.tensor(0.0)
		alpha = kwargs.get("alpha", 1)
		x = self._ensure_concrete(inputs[0])
		y = self._ensure_concrete(inputs[1])
		return torch.add(x, y, alpha=alpha)

	def _execute_sub(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		self._track_operation("aten::sub")
		if len(inputs) < 2:
			return torch.tensor(0.0)
		alpha = kwargs.get("alpha", 1)
		x = self._ensure_concrete(inputs[0])
		y = self._ensure_concrete(inputs[1])
		return torch.sub(x, y, alpha=alpha)

	def _execute_mul(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		self._track_operation("aten::mul")
		if len(inputs) < 2:
			return torch.tensor(0.0)
		x = self._ensure_concrete(inputs[0])
		y = self._ensure_concrete(inputs[1])
		return torch.mul(x, y)

	def _execute_div(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		self._track_operation("aten::div")
		if len(inputs) < 2:
			return torch.tensor(0.0)
		x = self._ensure_concrete(inputs[0])
		y = self._ensure_concrete(inputs[1])
		return torch.div(x, y)

	def _execute_matmul(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if len(inputs) < 2:
			return torch.tensor(0.0)
		x = self._ensure_concrete(inputs[0])
		y = self._ensure_concrete(inputs[1])
		return torch.matmul(x, y)

	def _execute_alias(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if not inputs:
			return torch.tensor(0.0)
		# Pass-through; ensure tensor
		val = self._ensure_concrete(inputs[0])
		if isinstance(val, torch.Tensor):
			return val
		return torch.tensor(val)

	# Intentionally rely on fallback for mm/bmm (unified eager path)

	def _execute_randn(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		# Extract size from inputs or kwargs
		if inputs:
			first_input = inputs[0]
			if type(first_input).__name__ in ('tuple', 'list'):
				size = first_input
			else:
				size = inputs
		else:
			size = (1,)

		dtype = kwargs.get("dtype", torch.float32)
		device = kwargs.get("device", "cpu")
		requires_grad = kwargs.get("requires_grad", False)

		return torch.randn(*size, dtype=dtype, device=device, requires_grad=requires_grad)

	def _execute_zeros(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if inputs and type(inputs[0]).__name__ in ('tuple', 'list'):
			size = inputs[0]
		elif inputs:
			size = inputs
		else:
			size = (1,)

		dtype = kwargs.get("dtype", torch.float32)
		device = kwargs.get("device", "cpu")
		requires_grad = kwargs.get("requires_grad", False)

		return torch.zeros(*size, dtype=dtype, device=device, requires_grad=requires_grad)

	def _execute_ones(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if inputs and type(inputs[0]).__name__ in ('tuple', 'list'):
			size = inputs[0]
		elif inputs:
			size = inputs
		else:
			size = (1,)

		dtype = kwargs.get("dtype", torch.float32)
		device = kwargs.get("device", "cpu")
		requires_grad = kwargs.get("requires_grad", False)

		return torch.ones(*size, dtype=dtype, device=device, requires_grad=requires_grad)

	def _execute_relu(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if not inputs:
			return torch.tensor(0.0)
		x = self._ensure_concrete(inputs[0])
		return torch.relu(x)

	def _execute_sigmoid(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if not inputs:
			return torch.tensor(0.0)
		x = self._ensure_concrete(inputs[0])
		return torch.sigmoid(x)

	def _execute_softmax(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		x = self._ensure_concrete(inputs[0])
		dim = kwargs.get("dim", -1)
		dtype = kwargs.get("dtype", None)
		return torch.softmax(x, dim=dim, dtype=dtype)

	def _execute_argmax(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		x = self._ensure_concrete(inputs[0])
		dim = kwargs.get("dim", None)
		keepdim = kwargs.get("keepdim", False)
		return torch.argmax(x, dim=dim, keepdim=keepdim)

	def _execute_argmin(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		x = self._ensure_concrete(inputs[0])
		dim = kwargs.get("dim", None)
		keepdim = kwargs.get("keepdim", False)
		return torch.argmin(x, dim=dim, keepdim=keepdim)

	def _execute_tanh(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if not inputs:
			return torch.tensor(0.0)
		x = self._ensure_concrete(inputs[0])
		return torch.tanh(x)

	def _execute_conv2d(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if len(inputs) < 2:
			return torch.tensor(0.0)

		input_tensor = self._ensure_concrete(inputs[0])
		weight = self._ensure_concrete(inputs[1])
		bias = self._ensure_concrete(inputs[2]) if len(inputs) > 2 else None

		stride = kwargs.get("stride", 1)
		padding = kwargs.get("padding", 0)
		dilation = kwargs.get("dilation", 1)
		groups = kwargs.get("groups", 1)

		return torch.conv2d(input_tensor, weight, bias, stride, padding, dilation, groups)

	def _execute_leaky_relu(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if not inputs:
			return torch.tensor(0.0)
		x = self._ensure_concrete(inputs[0])
		negative_slope = kwargs.get("negative_slope", 0.01)
		return torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)

	def _execute_sum(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		x = self._ensure_concrete(inputs[0])
		dim = kwargs.get("dim", None)
		keepdim = kwargs.get("keepdim", False)
		dtype = kwargs.get("dtype", None)
		return torch.sum(x, dim=dim, keepdim=keepdim, dtype=dtype)

	def _execute_mean(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		x = self._ensure_concrete(inputs[0])
		dim = kwargs.get("dim", None)
		keepdim = kwargs.get("keepdim", False)
		dtype = kwargs.get("dtype", None)
		return torch.mean(x, dim=dim, keepdim=keepdim, dtype=dtype)

	def _execute_var(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		x = self._ensure_concrete(inputs[0])
		dim = kwargs.get("dim", None)
		keepdim = kwargs.get("keepdim", False)
		# Prefer correction if provided; else map unbiased to correction
		if "correction" in kwargs:
			correction = kwargs.get("correction", 1)
		else:
			correction = 1 if kwargs.get("unbiased", True) else 0
		return torch.var(x, dim=dim, keepdim=keepdim, correction=correction)

	def _execute_std(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		x = self._ensure_concrete(inputs[0])
		dim = kwargs.get("dim", None)
		keepdim = kwargs.get("keepdim", False)
		if "correction" in kwargs:
			correction = kwargs.get("correction", 1)
		else:
			correction = 1 if kwargs.get("unbiased", True) else 0
		return torch.std(x, dim=dim, keepdim=keepdim, correction=correction)

	def get_stats(self) -> Dict[str, Any]:
		"""Get executor statistics."""
		return {
			"execution_count": self.execution_count,
			"supported_operations": len(self.operation_handlers)
		}
	
	# FX-based execution removed for Phase 1 - simplifies debugging and reduces complexity


# Global device assignment (for co-location)
_device_assignments = {}  # colocation_group -> device


def _get_device_for_node(lazy_tensor) -> str:
	"""
	Get device assignment for a node.

	Respects co-location hints from optimizer.
	"""
	# Check if node has co-location metadata
	if hasattr(lazy_tensor, 'metadata') and lazy_tensor.metadata:
		metadata = lazy_tensor.metadata

		# Check for colocation_group
		if hasattr(metadata, 'colocation_group') and metadata.colocation_group:
			group = metadata.colocation_group

			# Get or assign device for this group
			if group not in _device_assignments:
				_device_assignments[group] = os.getenv('GENIE_SERVER_URL', 'http://localhost:8888')
				logger.info(f"Assigned colocation group '{group}' to {_device_assignments[group]}")

			return _device_assignments[group]

	# Default: use env variable or default
	return os.getenv('GENIE_SERVER_URL', 'http://localhost:8888')


def _execute_local_creation(lazy_tensor) -> torch.Tensor:
	"""
	Execute tensor creation operations locally.
	These operations need to run locally first, then can be moved to remote device.
	"""
	from genie.core.lazy_tensor import LazyTensor

	logger.debug(f"Local creation: {lazy_tensor.operation}")

	# Materialize inputs (should be shape/dtype parameters)
	materialized_inputs = []
	for inp in lazy_tensor.inputs:
		if isinstance(inp, LazyTensor):
			materialized_input = inp.materialize()
			# Ensure we get a concrete tensor, not another LazyTensor
			if isinstance(materialized_input, LazyTensor):
				materialized_input = materialized_input.materialize()
			materialized_inputs.append(materialized_input)
		elif isinstance(inp, torch.Tensor):
			materialized_inputs.append(inp)
		else:
			# Convert scalars to tensors
			materialized_inputs.append(torch.tensor(inp))

	# Execute locally using torch operations
	operation = lazy_tensor.operation.replace("aten::", "")
	operation_base = operation.replace('aten::', '') if operation.startswith('aten::') else operation

	try:
		if operation_base == "randn":
			result = torch.randn(*materialized_inputs, **lazy_tensor.kwargs)
		elif operation_base == "zeros":
			result = torch.zeros(*materialized_inputs, **lazy_tensor.kwargs)
		elif operation_base == "ones":
			result = torch.ones(*materialized_inputs, **lazy_tensor.kwargs)
		elif operation_base == "empty":
			result = torch.empty(*materialized_inputs, **lazy_tensor.kwargs)
		else:
			raise ValueError(f"Unknown creation operation: {operation}")

		logger.debug(f"Created tensor: shape={result.shape}, dtype={result.dtype}")
		return result

	except Exception as e:
		logger.error(f"Local creation failed: {e}")
		raise


def _execute_remote(lazy_tensor) -> torch.Tensor:
	"""
	Execute LazyTensor on remote server via HTTP.

	Phase 1 limitations:
	- Only single-input operations
	- Only supported operations (relu, sigmoid, tanh, abs)
	"""
	from genie.runtime.simple_client import get_client
	from genie.core.lazy_tensor import LazyTensor
	import os

	logger.info(f"🌐 Remote execution: {lazy_tensor.operation}")
	logger.debug(f"   Tensor ID: {lazy_tensor.id}")

	# Get device for this node (respects co-location)
	server_url = _get_device_for_node(lazy_tensor)
	logger.debug(f"   Server: {server_url}")

	# Check for co-location metadata
	if hasattr(lazy_tensor, 'metadata') and lazy_tensor.metadata:
		if hasattr(lazy_tensor.metadata, 'colocation_group') and lazy_tensor.metadata.colocation_group:
			logger.info(f"   🔗 Co-location enabled: group={lazy_tensor.metadata.colocation_group}")

	# Get operation name first for early checks
	operation = lazy_tensor.operation.replace("aten::", "")

	# Check if this is a tensor creation operation (needs local execution first)
	TENSOR_CREATION_OPS = {'randn', 'zeros', 'ones', 'empty'}
	operation_base = operation.replace('aten::', '') if operation.startswith('aten::') else operation

	if operation_base in TENSOR_CREATION_OPS:
		# Execute locally first, then we'll handle the device placement
		logger.info(f"Local tensor creation: {operation}")
		input_tensor = _execute_local_creation(lazy_tensor)
		# For tensor creation, we don't need to send to remote - just return the tensor
		# But we need to mark it as materialized and return it
		lazy_tensor.concrete_value = input_tensor
		lazy_tensor.materialized = True
		return input_tensor

	# Materialize inputs first (recursive)
	materialized_inputs = []
	for inp in lazy_tensor.inputs:
		if isinstance(inp, LazyTensor):
			logger.debug(f"   Materializing input: {inp.id}")
			materialized_input = inp.materialize()
			# Ensure we get a concrete tensor, not another LazyTensor
			if isinstance(materialized_input, LazyTensor):
				# If we got a LazyTensor back, materialize it again
				materialized_input = materialized_input.materialize()
			materialized_inputs.append(materialized_input)
		elif isinstance(inp, torch.Tensor):
			materialized_inputs.append(inp)
		else:
			# Convert scalars to tensors
			materialized_inputs.append(torch.tensor(inp))

	# Phase 1: Only support single-input operations
	if len(materialized_inputs) != 1:
		raise NotImplementedError(
			f"Remote execution currently supports single-input operations only. "
			f"Got {len(materialized_inputs)} inputs for {lazy_tensor.operation}. "
			f"\n"
			f"This will be fixed in Phase 2 (multi-input support)."
		)

	input_tensor = materialized_inputs[0]

	# Ensure input_tensor is a concrete torch.Tensor, not a LazyTensor
	if isinstance(input_tensor, LazyTensor):
		input_tensor = input_tensor.materialize()

	# Define supported operations
	SUPPORTED_OPS = {'relu', 'sigmoid', 'tanh', 'abs', 'neg', 'exp', 'log', 'sqrt', 'alias'}

	if operation_base == "alias":
		# Alias operation - just return the input tensor
		logger.info(f"Alias operation: {operation}")
		input_tensor = materialized_inputs[0]
		lazy_tensor.concrete_value = input_tensor
		lazy_tensor.materialized = True
		return input_tensor
	elif operation_base in SUPPORTED_OPS:
		# These operations will be executed remotely via HTTP
		pass  # Continue to HTTP execution below
	else:
		raise NotImplementedError(
			f"Operation '{operation_base}' not supported for remote execution. "
			f"Supported: {SUPPORTED_OPS}. "
			f"Tensor creation ops: {TENSOR_CREATION_OPS}. "
			f"\n"
			f"This will be expanded in Phase 2."
		)

	# Execute via HTTP
	client = get_client(server_url=server_url)

	try:
		result = client.execute(
			operation=operation,
			tensor=input_tensor,
			timeout=30.0
		)

		logger.info(f"✅ Remote execution successful: {input_tensor.shape} -> {result.shape}")
		return result

	except Exception as e:
		logger.error(f"❌ Remote execution failed: {e}")
		raise RuntimeError(
			f"Remote execution of {operation} failed: {e}\n"
			f"Make sure server is running: python -m genie.runtime.simple_server"
		)


# Global executor instance
_executor = SimpleExecutor()


def execute_subgraph(target_lazy_tensor) -> torch.Tensor:  # noqa: ANN001
	"""Execute computation graph up to target tensor."""
	return _executor.execute_subgraph(target_lazy_tensor)


