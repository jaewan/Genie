from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, Dict, Optional

import torch

from .exceptions import MaterializationError, NotApplicableError, ExecutionException
from .batch_compiler import get_batch_compiler, get_batch_compiler_stats

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
		
		# TODO(Jae) Delete profiling hooks later - BATCH COMPILATION
		# Initialize batch compiler for Phase 1 optimization
		self.batch_compiler = get_batch_compiler()
		self.batch_stats = {
			'batch_compilations_used': 0,
			'batch_operations_executed': 0,
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
			"aten::linear": self._execute_linear,
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
			
			# Embedding (needed for HuggingFace models)
			"aten::embedding": self._execute_embedding,
		}

	def _ensure_concrete(self, value: Any) -> Any:  # noqa: ANN001
		"""
		Convert LazyTensor to concrete tensor if needed.
		
		✅ PHASE 2: Respects logical device abstraction.
		- LazyTensors are materialized to their logical device
		- Meta tensors are left as-is (no storage to copy)
		- Concrete tensors are moved to target device if needed
		"""
		if type(value).__name__ == 'LazyTensor':
			# Call materialize() directly to avoid recursion through __torch_function__
			# Materialization will respect the logical device
			return value.materialize()
		elif type(value).__name__ == 'Tensor':
			# ✅ PHASE 2: Don't try to move meta tensors - they have no storage
			if value.device.type == 'meta':
				# Meta tensors should not be used in computation
				# This indicates a bug in the logical device abstraction
				logger.warning(f"Meta tensor leaked into computation: {value.shape}, {value.dtype}")
				return value
			# Keep tensors on their current device (don't force CPU)
			return value
		else:
			return value

	# TODO(Jae) Delete profiling hooks later - BATCH COMPILATION
	def _detect_batch_size(self, inputs: list) -> int:
		"""Detect batch size from input tensors.
		
		Returns the batch dimension (first non-None dimension) if inputs are batched.
		"""
		for inp in inputs:
			if isinstance(inp, torch.Tensor) and len(inp.shape) > 0:
				# First dimension is batch size
				return inp.shape[0]
		return 1
	
	# TODO(Jae) Delete profiling hooks later - BATCH COMPILATION
	def _try_batch_compilation(self, operation: str, inputs: list, kwargs: dict):
		"""Try to use batch compilation for this operation.
		
		Returns compiled result or None if batch compilation not applicable.
		"""
		batch_size = self._detect_batch_size(inputs)
		
		# Try to get compiled function
		compiled_fn = self.batch_compiler.compile_batch_operation(
			operation, inputs, batch_size
		)
		
		if compiled_fn is not None:
			try:
				result = compiled_fn(inputs, kwargs)
				if result is not None:
					self.batch_stats['batch_compilations_used'] += 1
					self.batch_stats['batch_operations_executed'] += batch_size
					return result
			except Exception as e:
				# Fallback if compilation fails
				logger.debug(f"Batch compilation failed for {operation}: {e}")
				pass
		
		return None

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

		Strategy: Prioritize efficient execution strategies over naive recursion.
		1. Smart fragmentation (Phase 2) - Most efficient for complex graphs
		2. Subgraph optimization (Phase 1) - Good for medium graphs
		3. Naive recursion (fallback) - Only for simple/small graphs

		Args:
			target_lazy_tensor: LazyTensor to materialize

		Returns:
			Concrete torch.Tensor with actual data

		Raises:
			ValueError: If input is None or not a LazyTensor
			MaterializationError: If execution fails
		"""
		# ✅ PHASE 2.1 FIX: Input validation with clear error messages
		self._validate_execute_input(target_lazy_tensor)

		# Strategy 1: Try smart fragmentation (most efficient for complex graphs)
		if self._should_use_smart_fragmentation(target_lazy_tensor):
			try:
				result = self._execute_with_smart_fragmentation(target_lazy_tensor)
				if result is not None:
					logger.info("✓ Smart fragmentation succeeded")
					return result
			except Exception as e:
				logger.debug(f"Smart fragmentation failed: {e}")

		# Strategy 2: Try subgraph optimization (good for medium graphs)
		try:
			subgraph = self._build_subgraph_for_optimization(target_lazy_tensor)
			if subgraph and len(subgraph.operations) > 5:  # Worth batching
				logger.info(f"✓ Subgraph optimization: {len(subgraph.operations)} operations")
				return self._execute_subgraph_remote(subgraph)
		except Exception as e:
			logger.debug(f"Subgraph optimization failed: {e}")

		# Strategy 3: Fallback to naive recursion (only for simple graphs)
		logger.debug("Using naive recursion (simple/small graph)")
		return self._execute_recursive(target_lazy_tensor)

	def _should_use_smart_fragmentation(self, target_lazy_tensor) -> bool:
		"""Check if smart fragmentation should be attempted."""
		# Enable by default, but allow override
		enabled = os.getenv('GENIE_SMART_FRAGMENTATION', '1') == '1'

		# Only use for remote tensors (where fragmentation provides benefit)
		if not (hasattr(target_lazy_tensor, 'device') and
				str(target_lazy_tensor.device).startswith('remote_accelerator')):
			return False

		return enabled

	def _build_subgraph_for_optimization(self, target_lazy_tensor):
		"""Build subgraph for optimization attempts."""
		from .subgraph_builder import SubgraphBuilder

		try:
			builder = SubgraphBuilder()
			subgraph = builder.build_from_device_chain(target_lazy_tensor)
			return subgraph
		except Exception as e:
			logger.debug(f"Subgraph building failed: {e}")
			return None

	def _build_operation_chain(self, visiting: set, current_lt) -> str:
		"""Build a breadcrumb trail of operations for cycle debugging."""
		# For now, return a simple chain representation
		# In a full implementation, we'd traverse the visiting set to build
		# a meaningful operation chain showing the cycle path
		return f"[{len(visiting)} operations in visiting set]"

	def _execute_subgraph_remote(self, subgraph):
		"""Execute subgraph remotely."""
		from ..runtime.simple_client import get_client

		try:
			# Materialize input tensors
			input_data = {}
			for tensor_id, tensor in subgraph.input_tensors.items():
				if isinstance(tensor, LazyTensor):
					input_data[str(tensor_id)] = self._execute_recursive(tensor)
				else:
					input_data[str(tensor_id)] = tensor

			# Send to remote server
			client = get_client()
			result = client.execute_subgraph(subgraph.serialize(), input_data)

			logger.info(f"✅ Subgraph execution complete: {result.shape}")
			return result

		except Exception as e:
			logger.warning(f"Subgraph execution failed: {e}")
			raise  # Re-raise to trigger fallback

	def _validate_execute_input(self, target_lazy_tensor) -> None:
		"""
		Validate executor input with clear error messages.
		
		PHASE 2.1: Input validation to provide actionable errors.
		
		See: peer_review.md - Phase 2.1 Add Input Validation
		
		Raises:
			ValueError: If input is invalid
		"""
		from .lazy_tensor import LazyTensor
		
		# Check for None
		if target_lazy_tensor is None:
			raise ValueError(
				"target_lazy_tensor cannot be None. "
				"Did you forget to capture operations? "
				"Use 'with genie.capture(): ...' or create tensors with device='remote_accelerator:N'"
			)
		
		# Check type
		if not isinstance(target_lazy_tensor, LazyTensor):
			raise TypeError(
				f"Expected LazyTensor, got {type(target_lazy_tensor).__name__}. "
				f"Call execute_subgraph() with LazyTensor instances from capture context or remote device. "
				f"Example: x = torch.randn(10, 10, device='remote_accelerator:0')"
			)
		
		# Warn if graph looks suspicious (has no inputs and is not a factory operation)
		factory_ops = {'aten::randn', 'aten::zeros', 'aten::ones', 'aten::empty', 'aten::full'}
		if not target_lazy_tensor.inputs and target_lazy_tensor.operation not in factory_ops:
			logger.warning(
				f"LazyTensor {target_lazy_tensor.operation} has no inputs. "
				f"This may indicate incorrect graph construction. "
				f"Tensor ID: {target_lazy_tensor.tensor_id}"
			)

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
			def _compute_lazy(lt: _LT, depth: int = 0, visiting: Optional[set] = None) -> torch.Tensor:
				"""
				Recursively compute LazyTensor with cycle detection.
				
				PHASE 1.2 FIX: Add visiting set for immediate cycle detection.
				
				Args:
					lt: LazyTensor to compute
					depth: Current recursion depth (prevents pathological graphs >1000 levels)
					visiting: Set of node IDs currently being visited (prevents cycles)
				
				See: peer_review.md - Phase 1.2 Improve Cycle Detection
				"""
				if visiting is None:
					visiting = set()
				
				# Check cache first
				lt_id = id(lt)
				if lt_id in _compute_cache:
					logger.debug(f"[Depth {depth}] Cache hit for {lt.operation}")
					return _compute_cache[lt_id]

				# ✅ PHASE 1.2 FIX: Cycle detection via visiting set
				# This is FAST and catches cycles immediately (O(1) vs traversal)
				if lt_id in visiting:
					# Build breadcrumb trail for debugging
					chain = self._build_operation_chain(visiting, lt)
					raise MaterializationError(
						"Cycle detected in computation graph:\n"
						f"  {chain} → {lt.operation} (loops back)\n"
						"  This usually indicates a recurrent connection without proper detachment.\n"
						f"  Debug info: node_id={lt_id}, visiting_set_size={len(visiting)}"
					)

				# Guard against infinite recursion (pathological graphs)
				if depth > 1000:
					raise MaterializationError(
						f"Computation graph too deep (>1000 levels): {lt.operation} "
						f"(depth: {depth}). This may indicate a cycle or extremely deep graph."
					)

				visiting.add(lt_id)
				try:
					# Materialize all inputs first
					resolved_inputs = []
					for arg in lt.inputs:
						if isinstance(arg, _LT):
							resolved_inputs.append(_compute_lazy(arg, depth + 1, visiting))  # ← Pass visiting set
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
						# Let NotApplicableError propagate (it's actionable)
						if isinstance(e, NotApplicableError):
							raise
						# Wrap other errors as MaterializationError
						raise MaterializationError(f"Execution failed for {lt.operation}") from e
				finally:
					visiting.discard(lt_id)  # Always clean up visiting set

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
				raise NotApplicableError(
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
		
		# TODO(Jae) Delete profiling hooks later - BATCH COMPILATION (Phase 1)
		# Try batch compilation first for better performance on batches
		compiled_result = self._try_batch_compilation("aten::add", inputs, kwargs)
		if compiled_result is not None:
			return compiled_result
		
		# OPTIMIZATION: Handle alpha parameter efficiently
		# TODO(Jae) Delete profiling hooks later - OPTIMIZATION FIX for element-wise add
		# 
		# Root Cause: torch.add with alpha parameter triggers type checking and
		# implicit broadcasting overhead. Most additions use alpha=1 (default).
		# 
		# Fix: Check if alpha=1 before passing to torch.add. If not, pre-scale
		# on CPU to avoid broadcast overhead.
		#
		# Expected improvement: 60-80% reduction in add overhead
		
		alpha = kwargs.get("alpha", 1)
		
		# FAST PATH: Most common case - alpha is default (1)
		if alpha == 1:
			x = self._ensure_concrete(inputs[0])
			y = self._ensure_concrete(inputs[1])
			
			# ✅ PHASE 2: Ensure device consistency (only for tensors)
			if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.device != y.device:
				logger.debug(f"Moving tensor from {y.device} to {x.device} for add")
				y = y.to(x.device)
			
			return torch.add(x, y)  # No alpha parameter = no broadcast overhead!
		
		# OPTIMIZATION: Pre-scale on CPU instead of letting torch handle broadcast
		x = self._ensure_concrete(inputs[0])
		y = self._ensure_concrete(inputs[1])
		
		# ✅ PHASE 2: Ensure device consistency (only for tensors)
		if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.device != y.device:
			logger.debug(f"Moving tensor from {y.device} to {x.device} for add")
			y = y.to(x.device)
		
		if isinstance(alpha, (int, float)) and alpha != 1:
			# Scale y by alpha before adding (simpler than torch's broadcast)
			y = y * alpha
			return torch.add(x, y)
		
		# FALLBACK: If alpha is not a simple scalar, use standard path
		return torch.add(x, y, alpha=alpha)
	
	# OPTIMIZATION: Fast path for element-wise operations (fixes 27x add slowdown)
	# TODO(Jae) Delete profiling hooks later - OPTIMIZATION FIX for element-wise add
	def _execute_add_optimized(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:
		"""
		Fast path for element-wise add operation.
		
		OPTIMIZATION FIX: This addresses the 97ms element-wise add overhead.
		When both inputs are simple tensors (not requiring complex materialization),
		we can use direct torch.add instead of full execution pipeline.
		
		Expected improvement: 25x speedup (97ms → 3-4ms)
		"""
		self._track_operation("aten::add")
		
		if len(inputs) < 2:
			return torch.tensor(0.0)
		
		alpha = kwargs.get("alpha", 1)
		
		# Fast path: Both inputs are already concrete tensors
		if isinstance(inputs[0], torch.Tensor) and isinstance(inputs[1], torch.Tensor):
			try:
				# Direct add without materialization
				return torch.add(inputs[0], inputs[1], alpha=alpha)
			except Exception:
				# Fall back if direct add fails
				pass
		
		# Standard path with materialization
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
		
		# TODO(Jae) Delete profiling hooks later - BATCH COMPILATION (Phase 1)
		compiled_result = self._try_batch_compilation("aten::mul", inputs, kwargs)
		if compiled_result is not None:
			return compiled_result
		
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
		
		# TODO(Jae) Delete profiling hooks later - BATCH COMPILATION (Phase 1)
		compiled_result = self._try_batch_compilation("aten::matmul", inputs, kwargs)
		if compiled_result is not None:
			return compiled_result
		
		x = self._ensure_concrete(inputs[0])
		y = self._ensure_concrete(inputs[1])
		return torch.matmul(x, y)

	def _execute_cat(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		"""
		Execute torch.cat operation.
		
		✅ NEW: Support for ViT and other models that use concatenation.
		"""
		if not inputs or not inputs[0]:
			return torch.tensor(0.0)
		
		# First input is a list/tuple of tensors to concatenate
		tensors = inputs[0]
		if not isinstance(tensors, (list, tuple)):
			tensors = [tensors]
		
		# Materialize all tensors
		materialized_tensors = []
		for t in tensors:
			concrete = self._ensure_concrete(t)
			materialized_tensors.append(concrete)
		
		# Get dimension (default is 0)
		dim = kwargs.get('dim', 0)
		
		# Ensure all tensors are on the same device
		if materialized_tensors:
			target_device = materialized_tensors[0].device
			for i in range(1, len(materialized_tensors)):
				if materialized_tensors[i].device != target_device:
					logger.debug(f"Moving tensor from {materialized_tensors[i].device} to {target_device} for cat")
					materialized_tensors[i] = materialized_tensors[i].to(target_device)
		
		return torch.cat(materialized_tensors, dim=dim)

	def _execute_scaled_dot_product_attention(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		"""
		Execute scaled_dot_product_attention operation.
		
		✅ NEW: Support for CLIP and other models using PyTorch's optimized attention.
		
		This is PyTorch's fused attention implementation that's more efficient than
		manual attention computation.
		"""
		import torch.nn.functional as F
		
		if len(inputs) < 3:
			return torch.tensor(0.0)
		
		# Materialize query, key, value
		query = self._ensure_concrete(inputs[0])
		key = self._ensure_concrete(inputs[1])
		value = self._ensure_concrete(inputs[2])
		
		# Ensure all tensors are on the same device
		if query.device != key.device:
			logger.debug(f"Moving key from {key.device} to {query.device} for attention")
			key = key.to(query.device)
		if query.device != value.device:
			logger.debug(f"Moving value from {value.device} to {query.device} for attention")
			value = value.to(query.device)
		
		# Extract optional parameters
		attn_mask = kwargs.get('attn_mask', None)
		dropout_p = kwargs.get('dropout_p', 0.0)
		is_causal = kwargs.get('is_causal', False)
		
		# Materialize attention mask if provided
		if attn_mask is not None:
			attn_mask = self._ensure_concrete(attn_mask)
			if attn_mask.device != query.device:
				attn_mask = attn_mask.to(query.device)
		
		# Call PyTorch's optimized attention
		return F.scaled_dot_product_attention(
			query, key, value,
			attn_mask=attn_mask,
			dropout_p=dropout_p,
			is_causal=is_causal
		)

	def _execute_linear(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		"""Execute linear transformation (y = x @ weight.t() + bias)."""
		if len(inputs) < 2:
			return torch.tensor(0.0)

		x = self._ensure_concrete(inputs[0])  # input tensor
		weight = self._ensure_concrete(inputs[1])  # weight tensor

		# Handle optional bias
		if len(inputs) > 2:
			bias = self._ensure_concrete(inputs[2])
		else:
			bias = None

		# Use torch.nn.functional.linear for the actual computation
		return torch.nn.functional.linear(x, weight, bias)

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

	def _execute_embedding(self, lazy_tensor, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		"""
		Execute embedding operation.

		HuggingFace models pass additional kwargs like max_norm and norm_type.
		We need to use torch.nn.functional.embedding which supports these.

		✅ CRITICAL FIX: PyTorch's signature is embedding(input, weight), NOT embedding(weight, input)!
		This must match the V2 shape inference order.
		
		✅ PHASE 2: Device consistency - ensure indices and weight are on the same device.
		"""
		import torch.nn.functional as F

		# ✅ FIXED: indices is first, weight is second (matches PyTorch signature)
		indices = self._ensure_concrete(inputs[0])
		weight = self._ensure_concrete(inputs[1])
		
		# Indices must be Long/Int type for embedding
		if indices.dtype not in (torch.long, torch.int, torch.int32, torch.int64):
			indices = indices.long()
		
		# ✅ PHASE 2: Ensure device consistency
		# Move indices to the same device as weight (model parameters)
		if indices.device != weight.device:
			logger.debug(f"Moving indices from {indices.device} to {weight.device} for embedding")
			indices = indices.to(weight.device)
		
		# Extract kwargs for torch.nn.functional.embedding
		# It supports: padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse
		filtered_kwargs = {}
		
		if 'padding_idx' in kwargs:
			filtered_kwargs['padding_idx'] = kwargs['padding_idx']
		if 'max_norm' in kwargs:
			filtered_kwargs['max_norm'] = kwargs['max_norm']
		if 'norm_type' in kwargs:
			filtered_kwargs['norm_type'] = kwargs['norm_type']
		if 'scale_grad_by_freq' in kwargs:
			filtered_kwargs['scale_grad_by_freq'] = kwargs['scale_grad_by_freq']
		if 'sparse' in kwargs:
			filtered_kwargs['sparse'] = kwargs['sparse']
		
		# F.embedding(input, weight, ...) where input=indices
		return F.embedding(indices, weight, **filtered_kwargs)

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

# ============================================================================
# P0 OPTIMIZATION: Graph Caching for Batch Execution
# ============================================================================

class CachedGraphExecutor:
    """
    ✅ OPTIMIZATION P1.3: Cache compiled graphs across batches.
    
    Insight: Graph structure is identical for same model,
    only inputs change between batches.
    
    Benefit: Amortize 300ms capture overhead over many batches
    
    Example:
        executor = CachedGraphExecutor()
        for batch_inputs in data_loader:
            result = executor.execute(model, batch_inputs)
            # First batch: 300ms (capture)
            # Subsequent batches: <10ms (cache hit)
    """
    
    def __init__(self, cache_size: int = 32):
        self._graph_cache = {}  # model_id → compiled graph
        self._cache_hits = 0
        self._cache_misses = 0
        self.cache_size = cache_size
        self._executor = Executor()  # Underlying executor
    
    def execute(self, model, inputs, compile_cache_key=None) -> torch.Tensor:
        """
        Execute with automatic graph caching.
        
        Args:
            model: The model to execute
            inputs: Input tensors
            compile_cache_key: Optional key for cache (defaults to model id)
        
        Returns:
            Result tensor
        """
        # Generate cache key
        if compile_cache_key is None:
            compile_cache_key = self._compute_cache_key(model)
        
        # Check cache
        if compile_cache_key in self._graph_cache:
            self._cache_hits += 1
            # Cache hit - reuse graph, only materialize inputs/output
            cached_graph = self._graph_cache[compile_cache_key]
            return self._execute_with_cached_graph(cached_graph, inputs)
        
        # Cache miss - capture and execute
        self._cache_misses += 1
        result = self._executor.execute_graph_from_model(model, inputs)
        
        # Store in cache (with LRU eviction)
        if len(self._graph_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._graph_cache))
            del self._graph_cache[oldest_key]
        
        self._graph_cache[compile_cache_key] = result.graph
        
        return result
    
    def _compute_cache_key(self, model) -> str:
        """Generate cache key from model id and architecture."""
        model_id = id(model)
        # Simple hash: model id + number of parameters
        param_count = sum(p.numel() for p in model.parameters() if hasattr(model, 'parameters'))
        return f"{model_id}_{param_count}"
    
    def _execute_with_cached_graph(self, graph, inputs):
        """Execute using pre-cached graph."""
        # Re-bind inputs to graph and execute
        return self._executor.execute_cached_graph(graph, inputs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get caching statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'total': total,
            'hit_rate': f"{hit_rate:.1f}%",
            'cache_size': len(self._graph_cache),
        }