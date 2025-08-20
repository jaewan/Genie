from __future__ import annotations

from typing import Dict, Any

from genie.core.graph import ComputationGraph
from genie.semantic.pattern_registry import PatternRegistry
from genie.semantic.workload import WorkloadProfile, WorkloadType
from genie.semantic.workload import MatchedPattern as _MatchedPattern
from genie.semantic.fx_analyzer import FXAnalyzer
from genie.semantic.hooks import HookManager


class SemanticAnalyzer:
	"""Three-tier semantic analyzer (dynamic, FX, hooks)."""

	def __init__(self, pattern_registry: PatternRegistry | None = None) -> None:
		self.pattern_registry = pattern_registry or PatternRegistry()
		self.fx_analyzer = FXAnalyzer()
		self.hook_manager = HookManager()

	def analyze_graph(self, graph: ComputationGraph) -> WorkloadProfile:
		ops_metadata = self._analyze_operations(graph)
		structural_info = self.fx_analyzer.analyze_structure(graph)
		semantic_context = self.hook_manager.get_context(graph)
		patterns = self.pattern_registry.match_patterns(graph)
		workload_type = self._classify_workload(patterns)
		return WorkloadProfile(
			workload_type=workload_type,
			patterns=patterns,
			metadata=ops_metadata,
			structure=structural_info,
			context=semantic_context,
		)

	def _analyze_operations(self, graph: ComputationGraph) -> Dict[str, Any]:
		# Minimal metadata summary: op histogram
		hist: Dict[str, int] = {}
		for node in graph.nodes.values():
			hist[node.operation] = hist.get(node.operation, 0) + 1
		return {"op_histogram": hist, "num_nodes": len(graph.nodes)}

	def _classify_workload(self, patterns) -> WorkloadType:  # noqa: ANN001
		# Minimal heuristic per spec scaffold
		scores = {p.pattern_name: p.confidence for p in patterns}
		if scores.get("llm", 0.0) > 0.8:
			return WorkloadType.LLM
		if scores.get("vision", 0.0) > 0.8:
			return WorkloadType.VISION
		if scores.get("multimodal", 0.0) > 0.7:
			return WorkloadType.MULTIMODAL
		if scores.get("recsys", 0.0) > 0.7:
			return WorkloadType.RECSYS
		return WorkloadType.UNKNOWN


