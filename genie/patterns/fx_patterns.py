from __future__ import annotations

from typing import List, Optional
from torch import fx


class SubgraphMatch:
	def __init__(self, nodes: List[fx.Node]) -> None:
		self.nodes = nodes


def find_matmul_chain(gm: fx.GraphModule, min_len: int = 2) -> Optional[SubgraphMatch]:
	"""Very simple matmul chain detection on FX graphs.

	Looks for consecutive calls to aten::matmul/mm/bmm based on node targets.
	"""
	chain: List[fx.Node] = []
	linear_targets = {"matmul", "mm", "bmm"}
	for node in gm.graph.nodes:
		if node.op == "call_function" and getattr(node.target, "__name__", "") in linear_targets:
			# Extend or start chain
			if chain and node.args and chain[-1] in node.args:
				chain.append(node)
			else:
				if len(chain) >= min_len:
					return SubgraphMatch(chain)
				chain = [node]
	# Check at end
	if len(chain) >= min_len:
		return SubgraphMatch(chain)
	return None


