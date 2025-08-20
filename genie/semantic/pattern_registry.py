from __future__ import annotations

from typing import Dict, List

from genie.core.graph import ComputationGraph
from genie.semantic.workload import MatchedPattern
from genie.patterns.base import PatternPlugin
from genie.patterns.matmul_pattern import MatMulPattern, ConvolutionPattern


class PatternRegistry:
	"""Registry and orchestrator for pattern plugins."""

	def __init__(self) -> None:
		self._patterns: Dict[str, PatternPlugin] = {}
		self.register_builtin_patterns()

	def register_pattern(self, pattern: PatternPlugin) -> None:
		self._patterns[pattern.name] = pattern

	def register_builtin_patterns(self) -> None:
		self.register_pattern(MatMulPattern())
		self.register_pattern(ConvolutionPattern())

	def match_patterns(self, graph: ComputationGraph) -> List[MatchedPattern]:
		matches: List[MatchedPattern] = []
		for pattern in self._patterns.values():
			match = pattern.match(graph)
			if match is None:
				continue
			matches.append(
				MatchedPattern(
					pattern_name=pattern.name,
					confidence=match.confidence,
					subgraph=None,
					optimization_hints=getattr(pattern, "get_hints", lambda: {})(),
					metadata=getattr(match, "metadata", None),
				)
			)
		# Sort by confidence
		matches.sort(key=lambda m: m.confidence, reverse=True)
		return matches


