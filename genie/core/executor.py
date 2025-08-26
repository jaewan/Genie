from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from .graph import GraphBuilder
from .errors import MaterializationError, UnsupportedOperationError

logger = logging.getLogger(__name__)


class SimpleExecutor:
	"""Simple local executor for Phase 1.
	
	Executes LazyTensor graphs eagerly on CPU for validation and testing.
	In later phases, this will be replaced with remote execution.
	"""

	def __init__(self):
		self.execution_count = 0
		self.operation_handlers = self._build_operation_handlers()

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
			"aten::mm": self._execute_mm,
			"aten::bmm": self._execute_bmm,
			
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
		}

	def execute_subgraph(self, target_lazy_tensor) -> torch.Tensor:  # noqa: ANN001
		"""Execute computation graph up to target tensor."""
		self.execution_count += 1
		
		graph = GraphBuilder.current().get_graph()
		# Compute minimal ancestor subgraph required for target
		needed: Dict[str, bool] = {}
		target_id = getattr(target_lazy_tensor, "id", None)
		if target_id is None:
			return torch.tensor(0.0)
		
		# Build reverse adjacency list once
		rev_adj: Dict[str, list] = {}
		for src, dst in graph.edges:
			rev_adj.setdefault(dst, []).append(src)
		
		def mark_ancestors(node_id: str) -> None:
			if needed.get(node_id):
				return
			needed[node_id] = True
			for parent in rev_adj.get(node_id, []):
				# Only traverse LazyTensor nodes present in graph
				if parent in graph.nodes:
					mark_ancestors(parent)
		
		mark_ancestors(target_id)
		
		# Global order, then filter to needed nodes
		order_all = graph.topological_sort()
		order = [nid for nid in order_all if needed.get(nid)]

		# Map from node id to materialized tensor
		materialized: Dict[str, torch.Tensor] = {}

		for node_id in order:
			node = graph.nodes[node_id]
			
			# Skip if already materialized
			if node_id in materialized:
				continue
				
			try:
				# Execute the operation
				result = self._execute_operation(node, materialized)
				materialized[node_id] = result
				
				# Mark the corresponding LazyTensor as materialized if we can find it
				self._mark_tensor_materialized(node_id, result)
				
			except UnsupportedOperationError as e:
				logger.warning(f"Unsupported operation {node.operation}: {e}")
				materialized[node_id] = torch.tensor(0.0)
			except Exception as e:
				logger.error(f"Execution failed for {node.operation}: {e}")
				raise MaterializationError(str(e))

		return materialized.get(target_lazy_tensor.id, torch.tensor(0.0))

	def _mark_tensor_materialized(self, node_id: str, result: torch.Tensor) -> None:
		"""Mark the LazyTensor corresponding to this node as materialized."""
		# This is a bit of a hack - we need to find the LazyTensor that corresponds to this node
		# In a better implementation, we'd maintain a bidirectional mapping
		graph_builder = GraphBuilder.current()
		for tensor_id, tensor_node in graph_builder.tensor_to_node.items():
			if tensor_node.id == node_id:
				# Try to find the actual LazyTensor object
				# This is tricky because we don't have a direct reference
				# For now, we'll just mark it in a separate tracking dict
				if not hasattr(graph_builder, '_materialized_tensors'):
					graph_builder._materialized_tensors = {}
				graph_builder._materialized_tensors[node_id] = result
				break

	def _execute_operation(self, node, materialized: Dict[str, torch.Tensor]) -> torch.Tensor:
		"""Execute a single operation."""
		handler = self.operation_handlers.get(node.operation)
		
		# Resolve inputs: allow literals (ints, tuples, etc.) and node ids
		inputs = []
		for arg in node.inputs:
			# Literal or object inputs
			if not isinstance(arg, str):
				if isinstance(arg, torch.Tensor):
					inputs.append(arg)
				elif hasattr(arg, "materialize"):
					try:
						inputs.append(arg.materialize())
					except Exception:
						inputs.append(torch.tensor(0.0))
				else:
					inputs.append(arg)
				continue
			# Node ids
			if arg in materialized:
				inputs.append(materialized[arg])
				continue
			concrete = GraphBuilder.current().get_concrete(arg)
			if concrete is not None:
				if hasattr(concrete, "materialize"):
					inputs.append(concrete.materialize())
				else:
					inputs.append(concrete)
				continue
			logger.warning(f"Could not resolve input: {arg}")
			inputs.append(torch.tensor(0.0))
		
		kwargs = node.metadata.get("kwargs", {})
		# Only pass device for creation ops; strip for others to avoid invalid kwargs
		kwargs = kwargs.copy() if kwargs else {}
		# Broaden creation ops set
		aten_prefix = "aten::"
		base_name = node.operation[len(aten_prefix):] if node.operation.startswith(aten_prefix) else node.operation
		creation_ops = {
			"randn", "rand", "randint",
			"zeros", "ones", "empty", "full", "empty_strided",
			"arange", "linspace", "logspace",
		}
		if base_name in creation_ops:
			kwargs["device"] = "cpu"
		else:
			kwargs.pop("device", None)
		
		if handler is not None:
			return handler(inputs, kwargs)
		
		# Fallback path: materialize inputs and execute eagerly via torch.ops or torch API
		return self._execute_fallback_eager(node.operation, inputs, kwargs)

	def _execute_fallback_eager(self, op_name: str, inputs, kwargs) -> torch.Tensor:
		"""Fallback to eager execution using torch.ops.aten or torch API.

		This implements the graceful degradation described in the spec: if an
		operation isn't intercepted, materialize inputs and run it eagerly.
		"""
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
		concrete_inputs = [inp.cpu() if isinstance(inp, torch.Tensor) else inp for inp in concrete_inputs]

		# Normalize op name (e.g., "aten::add" -> "add")
		aten_prefix = "aten::"
		base_name = op_name[len(aten_prefix):] if op_name.startswith(aten_prefix) else op_name

		# Try torch.ops.aten first
		try:
			aten_ns = getattr(torch.ops, "aten")
			aten_op = getattr(aten_ns, base_name)
			return aten_op(*concrete_inputs, **kwargs)
		except Exception as e_ops:
			logger.debug(f"torch.ops.aten fallback failed for {op_name}: {e_ops}")
			# Try functional/torch namespace as secondary fallback
			try:
				torch_api = getattr(torch, base_name)
				return torch_api(*concrete_inputs, **kwargs)
			except Exception as e_torch:
				logger.warning(f"Unsupported operation {op_name}, returning zero tensor fallback: {e_torch}")
				# Last resort: zero tensor matching first input shape/dtype
				if len(concrete_inputs) > 0 and isinstance(concrete_inputs[0], torch.Tensor):
					ref = concrete_inputs[0]
					return torch.zeros_like(ref)
				return torch.tensor(0.0)

	# Operation handlers
	def _execute_add(self, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if len(inputs) < 2:
			return torch.tensor(0.0)
		alpha = kwargs.get("alpha", 1)
		# Force CPU execution to prevent device conflicts
		x = inputs[0].cpu() if isinstance(inputs[0], torch.Tensor) else inputs[0]
		y = inputs[1].cpu() if isinstance(inputs[1], torch.Tensor) else inputs[1]
		return torch.add(x, y, alpha=alpha)

	def _execute_sub(self, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if len(inputs) < 2:
			return torch.tensor(0.0)
		alpha = kwargs.get("alpha", 1)
		# Force CPU execution to prevent device conflicts
		x = inputs[0].cpu() if isinstance(inputs[0], torch.Tensor) else inputs[0]
		y = inputs[1].cpu() if isinstance(inputs[1], torch.Tensor) else inputs[1]
		return torch.sub(x, y, alpha=alpha)

	def _execute_mul(self, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if len(inputs) < 2:
			return torch.tensor(0.0)
		# Force CPU execution to prevent device conflicts
		x = inputs[0].cpu() if isinstance(inputs[0], torch.Tensor) else inputs[0]
		y = inputs[1].cpu() if isinstance(inputs[1], torch.Tensor) else inputs[1]
		return torch.mul(x, y)

	def _execute_div(self, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if len(inputs) < 2:
			return torch.tensor(0.0)
		# Force CPU execution to prevent device conflicts
		x = inputs[0].cpu() if isinstance(inputs[0], torch.Tensor) else inputs[0]
		y = inputs[1].cpu() if isinstance(inputs[1], torch.Tensor) else inputs[1]
		return torch.div(x, y)

	def _execute_matmul(self, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if len(inputs) < 2:
			return torch.tensor(0.0)
		# Force CPU execution to prevent device conflicts
		x = inputs[0].cpu() if isinstance(inputs[0], torch.Tensor) else inputs[0]
		y = inputs[1].cpu() if isinstance(inputs[1], torch.Tensor) else inputs[1]
		return torch.matmul(x, y)

	def _execute_alias(self, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if not inputs:
			return torch.tensor(0.0)
		# Pass-through; ensure tensor
		val = inputs[0]
		if isinstance(val, torch.Tensor):
			return val
		return torch.tensor(val)

	def _execute_mm(self, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if len(inputs) < 2:
			return torch.tensor(0.0)
		return torch.mm(inputs[0], inputs[1])

	def _execute_bmm(self, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if len(inputs) < 2:
			return torch.tensor(0.0)
		return torch.bmm(inputs[0], inputs[1])

	def _execute_randn(self, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		# Extract size from inputs or kwargs
		if inputs and isinstance(inputs[0], (tuple, list)):
			size = inputs[0]
		elif inputs:
			size = inputs
		else:
			size = (1,)
		
		dtype = kwargs.get("dtype", torch.float32)
		device = kwargs.get("device", "cpu")
		requires_grad = kwargs.get("requires_grad", False)
		
		return torch.randn(*size, dtype=dtype, device=device, requires_grad=requires_grad)

	def _execute_zeros(self, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if inputs and isinstance(inputs[0], (tuple, list)):
			size = inputs[0]
		elif inputs:
			size = inputs
		else:
			size = (1,)
		
		dtype = kwargs.get("dtype", torch.float32)
		device = kwargs.get("device", "cpu")
		requires_grad = kwargs.get("requires_grad", False)
		
		return torch.zeros(*size, dtype=dtype, device=device, requires_grad=requires_grad)

	def _execute_ones(self, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if inputs and isinstance(inputs[0], (tuple, list)):
			size = inputs[0]
		elif inputs:
			size = inputs
		else:
			size = (1,)
		
		dtype = kwargs.get("dtype", torch.float32)
		device = kwargs.get("device", "cpu")
		requires_grad = kwargs.get("requires_grad", False)
		
		return torch.ones(*size, dtype=dtype, device=device, requires_grad=requires_grad)

	def _execute_relu(self, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if not inputs:
			return torch.tensor(0.0)
		return torch.relu(inputs[0])

	def _execute_sigmoid(self, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if not inputs:
			return torch.tensor(0.0)
		return torch.sigmoid(inputs[0])

	def _execute_tanh(self, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if not inputs:
			return torch.tensor(0.0)
		return torch.tanh(inputs[0])

	def _execute_conv2d(self, inputs, kwargs) -> torch.Tensor:  # noqa: ANN001
		if len(inputs) < 2:
			return torch.tensor(0.0)
		
		input_tensor = inputs[0]
		weight = inputs[1]
		bias = inputs[2] if len(inputs) > 2 else None
		
		stride = kwargs.get("stride", 1)
		padding = kwargs.get("padding", 0)
		dilation = kwargs.get("dilation", 1)
		groups = kwargs.get("groups", 1)
		
		return torch.conv2d(input_tensor, weight, bias, stride, padding, dilation, groups)

	def get_stats(self) -> Dict[str, Any]:
		"""Get executor statistics."""
		return {
			"execution_count": self.execution_count,
			"supported_operations": len(self.operation_handlers)
		}


# Global executor instance
_executor = SimpleExecutor()


def execute_subgraph(target_lazy_tensor) -> torch.Tensor:  # noqa: ANN001
	"""Execute computation graph up to target tensor."""
	return _executor.execute_subgraph(target_lazy_tensor)


