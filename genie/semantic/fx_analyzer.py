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
		parameters = self.count_parameters(fx_graph)
		depth, width = self._depth_and_width(fx_graph)
		return StructuralInfo(
			modules=modules,
			architecture=architecture,
			depth=depth,
			width=width,
			parameters=parameters,
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

	def count_parameters(self, fx_graph: fx.Graph) -> int:
		"""Best-effort parameter count from owning GraphModule if available."""
		try:
			# Prefer an attribute on the graph that points to GraphModule
			gm = getattr(fx_graph, 'owning_module', None)
			if gm is None:
				from genie.core.graph import GraphBuilder as _GB
				gm = getattr(_GB.current(), '_fx_graph', None)
				gm = getattr(gm, 'module', gm)
			if gm is None or not hasattr(gm, 'parameters'):
				return 0
			return sum(int(p.numel()) for p in gm.parameters())
		except Exception:
			logging.getLogger(__name__).debug("Parameter counting failed", exc_info=True)
			return 0

	def _depth_and_width(self, fx_graph: fx.Graph) -> tuple[int, int]:
		try:
			nodes = list(fx_graph.nodes)
			depth = len(nodes)
			width = max((len(list(n.users)) for n in nodes), default=0)
			return depth, width
		except Exception:
			return 0, 0


