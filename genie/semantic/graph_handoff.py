from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set

from genie.core.graph import ComputationGraph, GraphBuilder


@dataclass
class GraphHandoff:
	"""Data passed from LazyTensor to Semantic Analyzer (per interface contract)."""
	graph: ComputationGraph
	lazy_tensors: Dict[str, object]
	materialization_frontier: Set[str]
	metadata_version: str = "1.0"

	def validate(self) -> bool:
		# All nodes have corresponding tensors
		for node_id in self.graph.nodes:
			if node_id not in self.lazy_tensors:
				return False
		# No dangling references
		for source, target in self.graph.edges:
			if source not in self.graph.nodes or target not in self.graph.nodes:
				return False
		return True


def build_graph_handoff() -> GraphHandoff:
	"""Helper to assemble GraphHandoff from current GraphBuilder state."""
	gb = GraphBuilder.current()
	graph = gb.get_graph()
	lazy_map = getattr(gb, "_lazy_tensors", {})
	frontier: Set[str] = set()
	# Approximate frontier as nodes with no outgoing edges
	with_outgoing = {src for src, _ in gb.edges}
	for node_id in graph.nodes:
		if node_id not in with_outgoing:
			frontier.add(node_id)
	return GraphHandoff(graph=graph, lazy_tensors=dict(lazy_map), materialization_frontier=frontier)


