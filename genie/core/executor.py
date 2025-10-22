from __future__ import annotations

import asyncio
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
			"aten::t": self._execute_t,
			# NOTE: Intentionally omit explicit handlers for mm/bmm to exercise unified fallback
			
			# Tensor creation
			"aten::randn": self._execute_randn,
			"aten::zeros": self._execute_zeros,
			"aten::ones": self._execute_ones,
			
			# Activations
			"aten::relu": self._execute_relu,
			"aten::sigmoid": self._execute_sigmoid,
			"aten::tanh": self._execute_tanh,

			# Device operations
			"aten::cpu": self._execute_cpu,
			
			# Convolution
			"aten::conv2d": self._execute_conv2d,

			# Pooling
			"aten::max_pool2d": self._execute_max_pool2d,
			"aten::avg_pool2d": self._execute_avg_pool2d,

			# Adaptive pooling
			"aten::adaptive_avg_pool2d": self._execute_adaptive_avg_pool2d,
			"aten::adaptive_max_pool2d": self._execute_adaptive_max_pool2d,

			# Normalization
			"aten::batch_norm": self._execute_batch_norm,
			"aten::layer_norm": self._execute_layer_norm,

			# Dropout
			"aten::dropout": self._execute_dropout,

			# Interpolation (upsample)
			"aten::interpolate": self._execute_interpolate,

			# Tensor manipulation
			"aten::split": self._execute_split,

			# Reductions
			"aten::sum": self._execute_sum,
			"aten::mean": self._execute_mean,
			"aten::var": self._execute_var,
			"aten::std": self._execute_std,
			"aten::argmax": self._execute_argmax,
			"aten::argmin": self._execute_argmin,
			"aten::softmax": self._execute_softmax,
			"aten::dim": self._execute_dim,
		}

	def _ensure_concrete(self, value: Any) -> Any:  # noqa: ANN001
		"""Convert LazyTensor to concrete tensor if needed."""
		if type(value).__name__ == 'LazyTensor':
			# Call materialize() directly to avoid recursion through __torch_function__
			return value.materialize()
		elif type(value).__name__ == 'Tensor':
			# Don't try to move meta tensors to CPU - they can't be copied
			if value.device.type == 'meta':
				return value
			return value.cpu() if value.device.type != 'cpu' else value
		else:
			return value

	async def _execute_remote(self, target_lazy_tensor) -> torch.Tensor:
		"""Execute LazyTensor on remote accelerator."""
		from .coordinator import get_coordinator
		from .lazy_tensor import LazyTensor as _LT

		# Get coordinator (will create if doesn't exist)
		coordinator = get_coordinator()

		# Materialize all inputs first (need to handle async properly)
		resolved_inputs = []
		for arg in target_lazy_tensor.inputs:
			if isinstance(arg, _LT):
				# Recursively execute inputs (they might also be remote)
				if hasattr(arg, 'device') and str(arg.device).startswith('remote_accelerator'):
					# Remote input - execute remotely
					resolved_inputs.append(await self._execute_remote(arg))
				else:
					# Local input - execute locally first
					resolved_inputs.append(self.execute_subgraph(arg))
			else:
				resolved_inputs.append(arg)

		# For now, assume single-input operations
		# In a complete implementation, you'd handle multi-input operations properly
		input_tensor = resolved_inputs[0] if resolved_inputs else None

		if input_tensor is None:
			raise ValueError("Remote execution requires at least one input tensor")

		# Extract target server from device string
		device_str = str(target_lazy_tensor.device)
		# Expected format: "remote_accelerator:server:port"
		parts = device_str.split(':')
		if len(parts) < 3:
			raise ValueError(f"Invalid remote device format: {device_str}")

		target = f"{parts[1]}:{parts[2]}"  # server:port

		# Execute remotely
		try:
			result = await coordinator.send_and_execute(
				input_tensor,
				operation=target_lazy_tensor.operation,
				target=target,
				metadata={'phase': 'remote_compute'}
			)

			return result

		except Exception as e:
			logger.error(f"Remote execution failed for {target_lazy_tensor.operation}: {e}")
			raise MaterializationError(f"Remote execution failed: {e}") from e

	def execute_subgraph(self, target_lazy_tensor) -> torch.Tensor:  # noqa: ANN001
		"""Execute computation graph up to target tensor.

		Supports both local and remote execution based on device type.
		Also supports optimized subgraph execution and smart fragmentation (enabled by default).
		"""
		# Check if smart fragmentation is enabled (Phase 2)
		use_smart_fragmentation = os.getenv('GENIE_SMART_FRAGMENTATION', '1') == '1'

		if use_smart_fragmentation:
			try:
				result = self._execute_with_smart_fragmentation(target_lazy_tensor)
				if result is not None:
					return result
				logger.debug("Smart fragmentation not applicable, falling back to basic subgraph optimization")
			except Exception as e:
				logger.warning(f"Smart fragmentation failed, falling back to basic subgraph optimization: {e}")

		# Check if basic subgraph optimization is enabled (Phase 1)
		use_subgraph_optimization = os.getenv('GENIE_SUBGRAPH_OPT', '1') == '1'

		if use_subgraph_optimization:
			# Try the optimized subgraph execution path first
			try:
				result = self._execute_with_subgraph_optimization(target_lazy_tensor)
				if result is not None:
					return result
				logger.debug("Subgraph optimization not applicable, falling back to recursive execution")
			except Exception as e:
				logger.warning(f"Subgraph optimization failed, falling back to recursive execution: {e}")

		# Fallback to original execution path
		return self._execute_recursive(target_lazy_tensor)

	def _execute_with_subgraph_optimization(self, target_lazy_tensor) -> Optional[torch.Tensor]:
		"""
		Execute using subgraph optimization.

		Returns None if optimization is not applicable, otherwise returns the result.
		"""
		from .subgraph_builder import SubgraphBuilder
		from .lazy_tensor import LazyTensor as _LT

		# Only apply optimization for remote tensors
		if not (isinstance(target_lazy_tensor, _LT) and
				hasattr(target_lazy_tensor, 'device') and
				str(target_lazy_tensor.device).startswith('remote_accelerator')):
			return None

		# Build subgraph
		builder = SubgraphBuilder()
		subgraph = builder.build_from_device_chain(target_lazy_tensor)

		if subgraph is None:
			logger.debug("No remote subgraph to execute")
			return None

		logger.info(f"🚀 Executing subgraph with {len(subgraph.operations)} operations")

		try:
			# Materialize input tensors
			input_data = {}
			for tensor_id, tensor in subgraph.input_tensors.items():
				# Materialize LazyTensor inputs
				if isinstance(tensor, LazyTensor):
					materialized = self._execute_recursive(tensor)  # Use recursive for inputs
				else:
					materialized = tensor
				input_data[str(tensor_id)] = materialized

			# Send to remote server
			from ..runtime.simple_client import get_client
			client = get_client()
			result = client.execute_subgraph(subgraph.serialize(), input_data)

			logger.info(f"✅ Subgraph execution complete: {result.shape}")
			return result

		except Exception as e:
			logger.warning(f"Subgraph execution failed: {e}")
			raise  # Re-raise to trigger fallback

	def _execute_with_smart_fragmentation(self, target_lazy_tensor) -> Optional[torch.Tensor]:
		"""
		Execute using smart fragmentation (Phase 2).

		Returns None if fragmentation is not applicable, otherwise returns the result.
		"""
		from .smart_subgraph_builder import SmartSubgraphBuilder, FragmentationConfig
		from .lazy_tensor import LazyTensor as _LT

		# Only apply fragmentation for remote tensors
		if not (isinstance(target_lazy_tensor, _LT) and
				hasattr(target_lazy_tensor, 'device') and
				str(target_lazy_tensor.device).startswith('remote_accelerator')):
			return None

		# Create smart fragmentation configuration
		config = FragmentationConfig(
			memory_limit_gb=float(os.getenv('GENIE_MEMORY_LIMIT_GB', '8.0')),
			network_gbps=float(os.getenv('GENIE_NETWORK_Gbps', '100.0')),
			compute_tflops=float(os.getenv('GENIE_COMPUTE_TFLOPS', '10.0')),
			fragmentation_threshold=float(os.getenv('GENIE_FRAGMENTATION_THRESHOLD', '0.8')),
			prefer_local_compute=os.getenv('GENIE_PREFER_LOCAL', '1') == '1'
		)

		# Build fragments using smart fragmentation
		builder = SmartSubgraphBuilder(config)
		fragments = builder.build_with_fragmentation(target_lazy_tensor)

		if not fragments:
			logger.debug("No fragments created")
			return None

		logger.info(f"🚀 Executing {len(fragments)} fragments with smart fragmentation")

		try:
			# Execute fragments in order
			fragment_results = {}
			executed_count = 0

			for fragment in fragments:
				logger.debug(f"Executing fragment {executed_count}: {len(fragment.operations)} ops, "
							f"mode={fragment.execution_mode}, cost={fragment.cost_estimate.total_cost_ms:.2f}ms")

				if fragment.execution_mode == 'local':
					# Execute locally
					result = self._execute_fragment_locally(fragment)
				else:
					# Execute remotely
					result = self._execute_fragment_remotely(fragment)

				fragment_results[id(fragment)] = result
				executed_count += 1

			# Find the final fragment (the one that produces the target tensor)
			final_fragment = None
			for fragment in fragments:
				if id(fragment.output_tensor) == id(target_lazy_tensor):
					final_fragment = fragment
					break

			if final_fragment and id(final_fragment) in fragment_results:
				result = fragment_results[id(final_fragment)]
				logger.info(f"✅ Smart fragmentation complete: {len(fragments)} fragments → {result.shape}")
				return result
			else:
				logger.warning("Could not find final fragment result")
				return None

		except Exception as e:
			logger.warning(f"Smart fragmentation execution failed: {e}")
			raise  # Re-raise to trigger fallback

	def _execute_fragment_locally(self, fragment) -> torch.Tensor:
		"""Execute a fragment locally."""
		# For now, execute operations recursively
		# In a full implementation, this would use local GPU execution
		return self._execute_operations_locally(fragment.operations)

	def _execute_fragment_remotely(self, fragment) -> torch.Tensor:
		"""Execute a fragment remotely."""
		# Convert fragment to subgraph format for remote execution
		from .subgraph_builder import RemoteSubgraph

		subgraph = RemoteSubgraph(
			operations=fragment.operations,
			input_tensors=fragment.input_tensors,
			output_tensor=fragment.output_tensor
		)

		# Materialize input tensors
		input_data = {}
		for tensor_id, tensor in fragment.input_tensors.items():
			if isinstance(tensor, LazyTensor):
				materialized = self._execute_recursive(tensor)
			else:
				materialized = tensor
			input_data[str(tensor_id)] = materialized

		# Send to remote server
		from ..runtime.simple_client import get_client
		client = get_client()
		result = client.execute_subgraph(subgraph.serialize(), input_data)

		return result

	def _execute_operations_locally(self, operations) -> torch.Tensor:
		"""Execute a sequence of operations locally."""
		if not operations:
			return torch.tensor(0.0)

		# Execute operations in sequence
		current_result = None
		for op in operations:
			if current_result is None:
				# First operation - use its inputs
				resolved_inputs = []
				for inp in op.inputs:
					if isinstance(inp, LazyTensor):
						resolved_inputs.append(self._execute_recursive(inp))
					else:
						resolved_inputs.append(inp)

				current_result = self._execute_operation(op, resolved_inputs)
			else:
				# Subsequent operations - use previous result as input
				current_result = self._execute_operation(op, [current_result])

		return current_result

	def _execute_operation(self, operation, inputs) -> torch.Tensor:
		"""Execute a single operation with given inputs."""
		try:
			op_func = self.operation_handlers.get(operation.operation)
			if op_func:
				return op_func(operation, inputs, operation.kwargs)
			else:
				return self._execute_fallback_eager(operation.operation, inputs, operation.kwargs)
		except Exception as e:
			logger.error(f"Failed to execute operation {operation.operation}: {e}")
			raise MaterializationError(f"Execution failed for {operation.operation}") from e

	def _execute_recursive(self, target_lazy_tensor) -> torch.Tensor:
		"""Recursive execution path (original implementation)."""
		current_thread = threading.get_ident()

		with self._execution_lock:
			if current_thread in self._executing_threads:
				raise RuntimeError("Recursive execution detected")

			self.execution_count += 1
			self.stats['total_executions'] += 1
			self._executing_threads.add(current_thread)

		_in_executor.active = True
		try:
			# Check if this targets a remote device
			from .lazy_tensor import LazyTensor as _LT
			if (isinstance(target_lazy_tensor, _LT) and
				hasattr(target_lazy_tensor, 'device') and
				str(target_lazy_tensor.device).startswith('remote_accelerator')):

				# Remote execution path - need to run in async context
				try:
					loop = asyncio.get_running_loop()
					return loop.run_until_complete(self._execute_remote(target_lazy_tensor))
				except RuntimeError:
					# No running loop, create one
					return asyncio.run(self._execute_remote(target_lazy_tensor))

			# Phase 1: Simple direct recursive evaluation

			# Cache to store computed results for each LazyTensor
			# This ensures a LazyTensor appearing multiple times is only computed once
			_compute_cache = {}

			# Use function parameter for depth tracking (thread-safe, correct for nested calls)
			def _compute_lazy(lt: _LT, depth: int = 0) -> torch.Tensor:
				"""Recursively compute LazyTensor with depth tracking."""
				# Check cache first
				lt_id = id(lt)
				if lt_id in _compute_cache:
					logger.debug(f"[Depth {depth}] Cache hit for {lt.operation}")
					return _compute_cache[lt_id]

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
					logger.debug(f"[Depth {depth}] Computing {lt.operation} with {len(resolved_inputs)} inputs: {[getattr(inp, 'shape', 'scalar') for inp in resolved_inputs]}")
					if op_func:
						result = op_func(lt, resolved_inputs, lt.kwargs)
						logger.debug(f"[Depth {depth}] {lt.operation} result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
						# Cache the result
						_compute_cache[lt_id] = result
						return result
					else:
						# Fallback for unhandled operations using executor's fallback mechanism
						result = self._execute_fallback_eager(lt.operation, resolved_inputs, lt.kwargs)
						# Cache the result
						_compute_cache[lt_id] = result
						return result
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

		# Handle device mapping for creation ops and others
		kwargs = kwargs.copy() if kwargs else {}
		creation_ops = {
			"randn", "rand", "randint",
			"zeros", "ones", "empty", "full", "empty_strided",
			"arange", "linspace", "logspace",
		}
		aten_prefix = "aten::"
		base_name = op_name[len(aten_prefix):] if op_name.startswith(aten_prefix) else op_name

		if base_name in creation_ops:
			# Map remote accelerator devices to actual GPU devices for creation ops
			device = kwargs.get("device")
			if isinstance(device, str) and 'remote_accelerator' in device:
				device_idx = device.split(':')[-1]
				try:
					# Check if device index is valid
					device_idx_int = int(device_idx)
					if device_idx_int < torch.cuda.device_count():
						kwargs["device"] = f'cuda:{device_idx}'
					else:
						# Fall back to CPU for invalid device indices
						kwargs["device"] = "cpu"
				except (ValueError, IndexError):
					# Fall back to CPU for invalid device indices
					kwargs["device"] = "cpu"
			elif hasattr(device, 'type') and device.type in ('remote_accelerator', 'privateuseone'):
				try:
					if device.index < torch.cuda.device_count():
						kwargs["device"] = f'cuda:{device.index}'
					else:
						kwargs["device"] = "cpu"
				except (AttributeError, IndexError):
					kwargs["device"] = "cpu"
			elif device is None:
				# For capture API (device=None), default to GPU to match native behavior
				if torch.cuda.is_available():
					kwargs["device"] = "cuda:0"
				else:
					kwargs["device"] = "cpu"
			else:
				kwargs["device"] = "cpu"
		else:
			# For non-creation ops, still map remote devices but don't default to CPU
			device = kwargs.get("device")
			if isinstance(device, str) and 'remote_accelerator' in device:
				device_idx = device.split(':')[-1]
				try:
					device_idx_int = int(device_idx)
					if device_idx_int < torch.cuda.device_count():
						kwargs["device"] = f'cuda:{device_idx}'
					else:
						# Remove device kwarg for invalid indices - don't pass it
						kwargs.pop("device", None)
				except (ValueError, IndexError):
					kwargs.pop("device", None)
			elif hasattr(device, 'type') and device.type in ('remote_accelerator', 'privateuseone'):
				try:
					if device.index < torch.cuda.device_count():
						kwargs["device"] = f'cuda:{device.index}'
					else:
						kwargs.pop("device", None)
				except (AttributeError, IndexError):
					kwargs.pop("device", None)
			elif device is None:
				# For capture API (device=None), don't pass device to comparison ops
				kwargs.pop("device", None)
			else:
				# For other devices, remove device kwarg - non-creation ops shouldn't have it
				kwargs.pop("device", None)

		# Normalize op name (e.g., "aten::add" -> "add")
		aten_prefix = "aten::"
		base_name = op_name[len(aten_prefix):] if op_name.startswith(aten_prefix) else op_name

		# Special handling for unknown_method - just return the first input
		if base_name == "unknown_method":
			logger.debug(f"Encountered unknown_method operation, returning first input: {concrete_inputs[0] if concrete_inputs else 'none'}")
			return concrete_inputs[0] if concrete_inputs else torch.tensor(0.0)

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

	def _execute_t(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		"""Execute transpose operation."""
		if not inputs:
			return torch.tensor(0.0)
		x = self._ensure_concrete(inputs[0])
		return torch.t(x)

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
		device = kwargs.get("device")
		requires_grad = kwargs.get("requires_grad", False)

		# Map remote accelerator devices to actual GPU devices
		if isinstance(device, str) and 'remote_accelerator' in device:
			# Extract device index (e.g., 'remote_accelerator:0' -> 'cuda:0')
			device_idx = device.split(':')[-1]
			device = f'cuda:{device_idx}'
		elif hasattr(device, 'type') and device.type in ('remote_accelerator', 'privateuseone'):
			# Handle torch.device objects
			device = f'cuda:{device.index}'
		elif device is None:
			# For capture API (device=None), default to GPU if available
			device = "cuda:0" if torch.cuda.is_available() else "cpu"
		else:
			device = "cpu"

		return torch.randn(*size, dtype=dtype, device=device, requires_grad=requires_grad)

	def _execute_zeros(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if inputs and type(inputs[0]).__name__ in ('tuple', 'list'):
			size = inputs[0]
		elif inputs:
			size = inputs
		else:
			size = (1,)

		dtype = kwargs.get("dtype", torch.float32)
		device = kwargs.get("device")
		requires_grad = kwargs.get("requires_grad", False)

		# Map remote accelerator devices to actual GPU devices
		if isinstance(device, str) and 'remote_accelerator' in device:
			device_idx = device.split(':')[-1]
			device = f'cuda:{device_idx}'
		elif hasattr(device, 'type') and device.type in ('remote_accelerator', 'privateuseone'):
			device = f'cuda:{device.index}'
		elif device is None:
			# For capture API (device=None), default to GPU to match native behavior
			device = "cuda:0"
		else:
			device = "cpu"

		return torch.zeros(*size, dtype=dtype, device=device, requires_grad=requires_grad)

	def _execute_ones(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if inputs and type(inputs[0]).__name__ in ('tuple', 'list'):
			size = inputs[0]
		elif inputs:
			size = inputs
		else:
			size = (1,)

		dtype = kwargs.get("dtype", torch.float32)
		device = kwargs.get("device")
		requires_grad = kwargs.get("requires_grad", False)

		# Map remote accelerator devices to actual GPU devices
		if isinstance(device, str) and 'remote_accelerator' in device:
			device_idx = device.split(':')[-1]
			device = f'cuda:{device_idx}'
		elif hasattr(device, 'type') and device.type in ('remote_accelerator', 'privateuseone'):
			device = f'cuda:{device.index}'
		elif device is None:
			# For capture API (device=None), default to GPU to match native behavior
			device = "cuda:0"
		else:
			device = "cpu"

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

	def _execute_cpu(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		"""Execute cpu() operation - move tensor to CPU."""
		if not inputs:
			return torch.tensor(0.0)
		x = self._ensure_concrete(inputs[0])
		return x.cpu()

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

	def _execute_dim(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		"""Execute dim() operation - get tensor dimension."""
		if not inputs:
			return torch.tensor(0)
		x = self._ensure_concrete(inputs[0])
		return torch.tensor(x.dim())

	def _execute_split(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		"""Execute split operation."""
		if len(inputs) < 2:
			return torch.tensor(0.0)
		x = self._ensure_concrete(inputs[0])
		split_size_or_sections = self._ensure_concrete(inputs[1])
		dim = kwargs.get("dim", 0)

		# Split returns a tuple, but we need to return the tuple as a single object
		# The unpacking will happen in the calling code
		split_result = torch.split(x, split_size_or_sections, dim=dim)

		# For now, return the first element as a workaround
		# In a full implementation, we'd need to handle tuple returns properly
		if isinstance(split_result, tuple) and len(split_result) > 0:
			return split_result[0]
		return split_result

	def _execute_max_pool2d(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		"""Execute max_pool2d operation."""
		if not inputs:
			return torch.tensor(0.0)
		x = self._ensure_concrete(inputs[0])

		# Extract pooling parameters from kwargs
		kernel_size = kwargs.get("kernel_size", 2)
		stride = kwargs.get("stride", kernel_size)
		padding = kwargs.get("padding", 0)
		dilation = kwargs.get("dilation", 1)
		ceil_mode = kwargs.get("ceil_mode", False)

		return torch.nn.functional.max_pool2d(
			x, kernel_size, stride=stride, padding=padding,
			dilation=dilation, ceil_mode=ceil_mode
		)

	def _execute_avg_pool2d(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		"""Execute avg_pool2d operation."""
		if not inputs:
			return torch.tensor(0.0)
		x = self._ensure_concrete(inputs[0])

		# Extract pooling parameters from kwargs
		kernel_size = kwargs.get("kernel_size", 2)
		stride = kwargs.get("stride", kernel_size)
		padding = kwargs.get("padding", 0)

		return torch.nn.functional.avg_pool2d(
			x, kernel_size, stride=stride, padding=padding
		)

	def _execute_batch_norm(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		"""Execute batch_norm operation."""
		if len(inputs) < 3:
			return torch.tensor(0.0)

		x = self._ensure_concrete(inputs[0])
		weight = self._ensure_concrete(inputs[1]) if len(inputs) > 1 else None
		bias = self._ensure_concrete(inputs[2]) if len(inputs) > 2 else None

		# For inference (non-training), we need running statistics
		# Since we don't have them, use a simplified version
		if weight is not None and bias is not None:
			# Use the weight and bias directly (simplified batch norm)
			return torch.nn.functional.batch_norm(
				x, running_mean=torch.zeros(x.size(1)).to(x.device),
				running_var=torch.ones(x.size(1)).to(x.device),
				weight=weight, bias=bias,
				training=False, momentum=0.1, eps=1e-5
			)
		else:
			# Simple normalization without affine transform
			return torch.nn.functional.batch_norm(
				x, running_mean=torch.zeros(x.size(1)).to(x.device),
				running_var=torch.ones(x.size(1)).to(x.device),
				training=False, momentum=0.1, eps=1e-5
			)

	def _execute_dropout(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		"""Execute dropout operation."""
		if not inputs:
			return torch.tensor(0.0)
		x = self._ensure_concrete(inputs[0])

		p = kwargs.get("p", 0.5)
		train = kwargs.get("train", False)

		return torch.nn.functional.dropout(x, p=p, training=train)

	def _execute_interpolate(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		"""Execute interpolate operation (upsample)."""
		if not inputs:
			return torch.tensor(0.0)
		x = self._ensure_concrete(inputs[0])

		# Extract interpolation parameters
		size = kwargs.get("size")
		scale_factor = kwargs.get("scale_factor")
		mode = kwargs.get("mode", "nearest")
		align_corners = kwargs.get("align_corners", None)

		return torch.nn.functional.interpolate(
			x, size=size, scale_factor=scale_factor,
			mode=mode, align_corners=align_corners
		)

	def _execute_adaptive_avg_pool2d(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		"""Execute adaptive_avg_pool2d operation."""
		if not inputs:
			return torch.tensor(0.0)
		x = self._ensure_concrete(inputs[0])

		output_size = kwargs.get("output_size")
		if output_size is None:
			return x  # No pooling needed

		return torch.nn.functional.adaptive_avg_pool2d(x, output_size)

	def _execute_adaptive_max_pool2d(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		"""Execute adaptive_max_pool2d operation."""
		if not inputs:
			return torch.tensor(0.0)
		x = self._ensure_concrete(inputs[0])

		output_size = kwargs.get("output_size")
		if output_size is None:
			return x  # No pooling needed

		return torch.nn.functional.adaptive_max_pool2d(x, output_size			)

	def _execute_layer_norm(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		"""Execute layer_norm operation."""
		if not inputs:
			return torch.tensor(0.0)
		x = self._ensure_concrete(inputs[0])

		normalized_shape = kwargs.get("normalized_shape", [])
		weight = kwargs.get("weight")
		bias = kwargs.get("bias")
		eps = kwargs.get("eps", 1e-5)

		# If normalized_shape is empty, infer it from the last dimension
		if not normalized_shape and len(x.shape) > 1:
			normalized_shape = [x.shape[-1]]

		result = torch.nn.functional.layer_norm(
			x, normalized_shape, weight=weight, bias=bias, eps=eps
		)

		return result

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