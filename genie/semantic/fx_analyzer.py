from __future__ import annotations

from typing import Dict, Any

from torch import fx

from genie.core.fx_tracer import trace_module
from genie.core.graph import ComputationGraph, GraphBuilder


class FXAnalyzer:
	"""Lightweight FX-based structural analyzer (scaffold)."""

	def analyze_structure(self, graph: ComputationGraph) -> Dict[str, Any]:
		fx_graph = GraphBuilder.current().get_fx_graph()
		if fx_graph is None:
			return {"modules": {}, "architecture": None, "depth": 0, "width": 0, "parameters": 0}
		modules = self.extract_module_hierarchy(fx_graph)
		return {
			"modules": modules,
			"architecture": None,
			"depth": len(list(fx_graph.nodes)),
			"width": max((len(list(n.users)) for n in fx_graph.nodes), default=0),
			"parameters": 0,
		}

	def extract_module_hierarchy(self, fx_graph: fx.Graph) -> Dict[str, Any]:
		hierarchy: Dict[str, Any] = {}
		for node in fx_graph.nodes:
			if node.op == "call_module":
				module_path = str(node.target)
				hierarchy[module_path] = {
					"inputs": [getattr(arg, "name", str(arg)) for arg in node.args],
					"users": [user.name for user in node.users],
				}
		return hierarchy


