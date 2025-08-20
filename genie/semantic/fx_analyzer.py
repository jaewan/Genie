from __future__ import annotations

from typing import Dict, Any

from torch import fx

from genie.core.fx_tracer import trace_module
from genie.core.graph import ComputationGraph, GraphBuilder
from genie.semantic.workload import StructuralInfo
import logging


class FXAnalyzer:
	"""Lightweight FX-based structural analyzer (scaffold)."""

	def analyze_structure(self, graph: ComputationGraph) -> StructuralInfo:
		fx_graph = GraphBuilder.current().get_fx_graph()
		if fx_graph is None:
			return StructuralInfo(modules={}, architecture=None, depth=0, width=0, parameters=0)
		modules = self.extract_module_hierarchy(fx_graph)
		architecture = self.identify_architecture(fx_graph, modules)
		return StructuralInfo(
			modules=modules,
			architecture=architecture,
			depth=len(list(fx_graph.nodes)),
			width=max((len(list(n.users)) for n in fx_graph.nodes), default=0),
			parameters=0,
		)

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

	def identify_architecture(self, fx_graph: fx.Graph, modules: Dict[str, Any]) -> str | None:
		"""Heuristically identify high-level architecture from FX graph.

		Returns one of: 'resnet', 'cnn', 'vit', 'mlp', or None.
		"""
		try:
			# Count function ops
			func_counts: Dict[str, int] = {}
			for node in fx_graph.nodes:
				if node.op == "call_function":
					name = getattr(node.target, "__name__", str(node.target))
					func_counts[name] = func_counts.get(name, 0) + 1

			conv2d = func_counts.get("conv2d", 0)
			relu = func_counts.get("relu", 0)
			add = func_counts.get("add", 0)
			bn = func_counts.get("batch_norm", 0)
			matmul = func_counts.get("matmul", 0) + func_counts.get("mm", 0) + func_counts.get("bmm", 0)
			softmax = func_counts.get("softmax", 0)
			layer_norm = func_counts.get("layer_norm", 0)
			permute = func_counts.get("permute", 0)
			reshape = func_counts.get("reshape", 0) + func_counts.get("view", 0)

			# Module name hints
			module_names = ",".join(modules.keys()).lower()
			if any(k in module_names for k in ["bottleneck", "basicblock", "layer1", "layer2", "layer3", "layer4"]):
				return "resnet"

			# ViT heuristic: attention substructures + layer norm + frequent matmul/softmax
			if matmul >= 2 and softmax >= 1 and (layer_norm >= 1 or "layernorm" in module_names):
				return "vit"

			# ResNet heuristic: many convs, relu, and adds (residual connections) or batch norm
			if conv2d >= 6 and relu >= 2 and (add >= 1 or bn >= 2):
				return "resnet"

			# CNN heuristic: presence of convs and activations but not enough add/bn for resnet
			if conv2d >= 2 and relu >= 1:
				return "cnn"

			# MLP heuristic: linear/addmm and activations without convs
			linear = func_counts.get("linear", 0) + func_counts.get("addmm", 0)
			if linear >= 2 and conv2d == 0 and matmul == 0:
				return "mlp"
		except Exception:
			logging.getLogger(__name__).debug("FX architecture identification failed", exc_info=True)
		return None


