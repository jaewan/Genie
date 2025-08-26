from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import threading
from torch import fx


@dataclass
class ComputationNode:
	id: str
	operation: str
	inputs: List[str]
	outputs: List[str]
	metadata: Dict


@dataclass
class ComputationGraph:
	nodes: Dict[str, ComputationNode]
	edges: List[Tuple[str, str]]  # (source_id, target_id)
	entry_points: Set[str]

	def topological_sort(self) -> List[str]:
		visited: Set[str] = set()
		stack: List[str] = []

		def visit(node_id: str) -> None:
			if node_id in visited:
				return
			visited.add(node_id)
			node = self.nodes[node_id]
			for inp in node.inputs:
				if inp in self.nodes:
					visit(inp)
			stack.append(node_id)

		for node_id in self.nodes:
			visit(node_id)
		return stack

	def validate_invariants(self) -> Tuple[bool, List[str]]:
		"""Validate basic invariants for nodes and edges.

		Checks:
		- All edge endpoints exist in nodes
		- Node ids are unique
		- No self-loops
		- Inputs referenced as strings must exist as node ids
		"""
		errors: List[str] = []
		# Unique ids
		if len(set(self.nodes.keys())) != len(self.nodes):
			errors.append("Duplicate node ids detected")
		# Edge endpoints
		for src, dst in self.edges:
			if src == dst:
				errors.append(f"Self-loop detected: {src}")
			if src not in self.nodes:
				errors.append(f"Edge src missing node: {src}")
			if dst not in self.nodes:
				errors.append(f"Edge dst missing node: {dst}")
		# Input references
		for node in self.nodes.values():
			for inp in node.inputs:
				if isinstance(inp, str) and inp.startswith("lt_") and inp not in self.nodes:
					errors.append(f"Missing input node: {inp} for {node.id}")
		return (len(errors) == 0, errors)


class GraphBuilder:
	"""Thread-local graph builder."""

	_thread_local = threading.local()

	def __init__(self) -> None:
		self.nodes: Dict[str, ComputationNode] = {}
		self.edges: List[Tuple[str, str]] = []
		self.tensor_to_node: Dict[str, ComputationNode] = {}
		self._concrete_map: Dict[str, object] = {}
		self._lazy_tensors: Dict[str, object] = {}
		self._fx_graph: Optional[fx.Graph] = None

	@classmethod
	def current(cls) -> "GraphBuilder":
		if not hasattr(cls._thread_local, "builder"):
			cls._thread_local.builder = cls()
		return cls._thread_local.builder

	@classmethod
	def reset_current(cls) -> None:
		"""Reset the thread-local builder to a fresh instance."""
		cls._thread_local.builder = cls()

	def add_tensor(self, lazy_tensor) -> None:  # noqa: ANN001
		node = ComputationNode(
			id=lazy_tensor.id,
			operation=lazy_tensor.operation,
			inputs=[self._get_tensor_id(inp) for inp in lazy_tensor.inputs],
			outputs=[lazy_tensor.id],
			metadata={
				"shape": lazy_tensor.shape, 
				"dtype": lazy_tensor.dtype,
				"kwargs": lazy_tensor.kwargs,
				"device": lazy_tensor.device
			},
		)
		self.nodes[node.id] = node
		self.tensor_to_node[lazy_tensor.id] = node
		self._lazy_tensors[lazy_tensor.id] = lazy_tensor
		for inp in lazy_tensor.inputs:
			inp_id = getattr(inp, "id", None)
			if inp_id:
				self.edges.append((inp_id, lazy_tensor.id))

	def reset(self) -> None:
		"""Clear all graph state for a fresh build cycle."""
		self.nodes.clear()
		self.edges.clear()
		self.tensor_to_node.clear()
		self._concrete_map.clear()
		self._lazy_tensors.clear()
		self._fx_graph = None

	def _get_tensor_id(self, tensor) -> str:  # noqa: ANN001
		tid = getattr(tensor, "id", None)
		if tid is not None:
			return tid
		key = f"concrete_{id(tensor)}"
		self._concrete_map[key] = tensor
		return key

	def get_graph(self) -> ComputationGraph:
		return ComputationGraph(
			nodes=self.nodes.copy(),
			edges=self.edges.copy(),
			entry_points={nid for nid in self.nodes if not any(e[1] == nid for e in self.edges)},
		)

	def get_concrete(self, tensor_id: str):
		return self._concrete_map.get(tensor_id)

	def get_lazy_tensor(self, tensor_id: str):
		return self._lazy_tensors.get(tensor_id)

	# FX integration hooks (optional for P1)
	def set_fx_graph(self, fx_graph: fx.Graph) -> None:
		"""Record an FX graph to allow cross-referencing with LazyTensor graph."""
		self._fx_graph = fx_graph

	def get_fx_graph(self) -> Optional[fx.Graph]:
		return self._fx_graph


