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


def _is_call_function(node: fx.Node, name: str) -> bool:
	return node.op == "call_function" and getattr(node.target, "__name__", "") == name


def find_attention_pattern_fx(gm: fx.GraphModule) -> Optional[SubgraphMatch]:
	"""Find attention subgraph: matmul -> softmax -> matmul.

	This intentionally stays simple and robust.
	"""
	matmul_nodes: List[fx.Node] = []
	softmax_nodes: List[fx.Node] = []

	for node in gm.graph.nodes:
		if _is_call_function(node, "matmul") or _is_call_function(node, "mm") or _is_call_function(node, "bmm"):
			matmul_nodes.append(node)
		elif _is_call_function(node, "softmax"):
			softmax_nodes.append(node)

	# Look for matmul -> softmax -> matmul chain by argument use
	for mm1 in matmul_nodes:
		for soft in softmax_nodes:
			if mm1 in soft.all_input_nodes:
				for mm2 in matmul_nodes:
					if soft in mm2.all_input_nodes and mm2 is not mm1:
						return SubgraphMatch([mm1, soft, mm2])
	return None


def find_conv_activation_pattern_fx(gm: fx.GraphModule) -> Optional[SubgraphMatch]:
	"""Find conv2d -> activation pattern on FX graphs."""
	conv_nodes: List[fx.Node] = []
	act_nodes: List[fx.Node] = []
	activations = {"relu", "sigmoid", "tanh", "gelu"}

	for node in gm.graph.nodes:
		if _is_call_function(node, "conv2d"):
			conv_nodes.append(node)
		elif node.op == "call_function" and getattr(node.target, "__name__", "") in activations:
			act_nodes.append(node)

	for conv in conv_nodes:
		for act in act_nodes:
			if conv in act.all_input_nodes:
				return SubgraphMatch([conv, act])
	return None


def find_mlp_pattern_fx(gm: fx.GraphModule) -> Optional[SubgraphMatch]:
	"""Find linear -> activation -> linear pattern."""
	linear_nodes: List[fx.Node] = []
	act_nodes: List[fx.Node] = []
	activations = {"relu", "gelu", "sigmoid"}

	for node in gm.graph.nodes:
		if _is_call_function(node, "linear") or _is_call_function(node, "addmm"):
			linear_nodes.append(node)
		elif node.op == "call_function" and getattr(node.target, "__name__", "") in activations:
			act_nodes.append(node)

	for l1 in linear_nodes:
		for act in act_nodes:
			if l1 in act.all_input_nodes:
				for l2 in linear_nodes:
					if act in l2.all_input_nodes and l2 is not l1:
						return SubgraphMatch([l1, act, l2])
	return None


