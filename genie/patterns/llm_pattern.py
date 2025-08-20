from __future__ import annotations

from typing import List, Optional, Dict, Set, Tuple

from genie.core.graph import ComputationGraph
from .base import PatternMatch, PatternPlugin


class LLMAttentionPattern(PatternPlugin):
	"""Detects a minimal self-attention motif typical in LLMs.

	Heuristic:
	- Find at least two matmul-like ops (matmul/mm/bmm)
	- Find a softmax op
	- There exists matmul_A -> softmax -> matmul_B (by graph edges)
	If satisfied, return pattern 'llm' with high confidence.
	"""

	@property
	def name(self) -> str:
		return "llm"

	def match(self, graph: ComputationGraph) -> Optional[PatternMatch]:
		matmul_ops: Set[str] = set()
		softmax_ops: Set[str] = set()

		for node_id, node in graph.nodes.items():
			op = node.operation
			if op in {"aten::matmul", "aten::mm", "aten::bmm"}:
				matmul_ops.add(node_id)
			elif op == "aten::softmax":
				softmax_ops.add(node_id)

		if len(matmul_ops) < 2 or not softmax_ops:
			return None

		# Build quick adjacency from producer -> consumers
		adj: Dict[str, Set[str]] = {}
		for src, dst in graph.edges:
			adj.setdefault(src, set()).add(dst)

		# Check for matmul -> softmax -> matmul chain
		matched_nodes: List[str] = []
		for mm1 in matmul_ops:
			# mm1 output flows to some softmax
			for soft in softmax_ops:
				if mm1 in graph.nodes[soft].inputs:
					# softmax output flows to some other matmul
					for mm2 in matmul_ops:
						if soft in graph.nodes[mm2].inputs and mm2 != mm1:
							matched_nodes = [mm1, soft, mm2]
							confidence = 0.9
							return PatternMatch(pattern_name=self.name, confidence=confidence, matched_nodes=matched_nodes)

		# Fallback: presence of >=3 matmul-like ops can still indicate transformer blocks
		if len(matmul_ops) >= 3:
			return PatternMatch(pattern_name=self.name, confidence=0.75, matched_nodes=list(matmul_ops)[:3])

		return None


