from __future__ import annotations

from typing import List, Optional

from genie.core.graph import ComputationGraph
from .base import PatternMatch, PatternPlugin
from .fx_patterns import find_matmul_chain
from torch import fx


class MatMulPattern(PatternPlugin):
	"""Detect matrix multiplication patterns and chains."""

	@property
	def name(self) -> str:
		return "matmul_chain"

	def match(self, graph: ComputationGraph) -> Optional[PatternMatch]:
		"""Detect matmul chains and compute-intensive patterns."""
		matmul_nodes: List[str] = []
		mm_nodes: List[str] = []
		bmm_nodes: List[str] = []
		
		for node_id, node in graph.nodes.items():
			if node.operation == "aten::matmul":
				matmul_nodes.append(node_id)
			elif node.operation == "aten::mm":
				mm_nodes.append(node_id)
			elif node.operation == "aten::bmm":
				bmm_nodes.append(node_id)
		
		all_linear_ops = matmul_nodes + mm_nodes + bmm_nodes
		
		if len(all_linear_ops) >= 2:
			# Check if they form a chain
			chain_length = self._analyze_chain_structure(graph, all_linear_ops)
			confidence = min(0.95, 0.7 + (chain_length * 0.1))
			
			return PatternMatch(
				pattern_name=self.name,
				confidence=confidence,
				matched_nodes=all_linear_ops
			)
		elif len(all_linear_ops) == 1:
			# Single large matmul might still be worth optimizing
			node_id = all_linear_ops[0]
			node = graph.nodes[node_id]
			if self._is_large_operation(node):
				return PatternMatch(
					pattern_name="large_matmul",
					confidence=0.8,
					matched_nodes=[node_id]
				)
		
		return None

	def match_fx(self, gm: fx.GraphModule) -> Optional[PatternMatch]:
		match = find_matmul_chain(gm)
		if match is None:
			return None
		return PatternMatch(
			pattern_name=self.name,
			confidence=0.9,
			matched_nodes=[n.name for n in match.nodes],
		)

	def _analyze_chain_structure(self, graph: ComputationGraph, linear_ops: List[str]) -> int:
		"""Analyze the chain structure of linear operations."""
		# Simple heuristic: count how many ops are connected
		connected_count = 0
		
		for op_id in linear_ops:
			# Check if this op's output is input to another linear op
			for other_id in linear_ops:
				if op_id != other_id:
					other_node = graph.nodes[other_id]
					if op_id in other_node.inputs:
						connected_count += 1
						break
		
		return connected_count

	def _is_large_operation(self, node) -> bool:
		"""Check if this is a large matrix operation worth optimizing."""
		# Estimate operation size from metadata
		shape = node.metadata.get("shape")
		if shape and len(shape) >= 2:
			# Rough FLOP estimate for matmul
			if len(shape) == 2:
				flops = shape[0] * shape[1] * shape[1]  # Simplified
			else:
				flops = shape[-2] * shape[-1] * shape[-1]  # Simplified
			
			# Consider "large" if > 1M FLOPs
			return flops > 1_000_000
		
		return False


class ConvolutionPattern(PatternPlugin):
	"""Detect convolution patterns typical in CNNs."""

	@property
	def name(self) -> str:
		return "conv_pattern"

	def match(self, graph: ComputationGraph) -> Optional[PatternMatch]:
		"""Detect convolution + activation patterns."""
		conv_nodes: List[str] = []
		activation_nodes: List[str] = []
		
		for node_id, node in graph.nodes.items():
			if node.operation == "aten::conv2d":
				conv_nodes.append(node_id)
			elif node.operation in ["aten::relu", "aten::sigmoid", "aten::tanh"]:
				activation_nodes.append(node_id)
		
		if not conv_nodes:
			return None
		
		# Look for conv + activation patterns
		conv_activation_pairs = []
		for conv_id in conv_nodes:
			for act_id in activation_nodes:
				act_node = graph.nodes[act_id]
				if conv_id in act_node.inputs:
					conv_activation_pairs.append((conv_id, act_id))
		
		if conv_activation_pairs:
			all_nodes = []
			for conv_id, act_id in conv_activation_pairs:
				all_nodes.extend([conv_id, act_id])
			
			return PatternMatch(
				pattern_name="conv_activation",
				confidence=0.9,
				matched_nodes=all_nodes
			)
		elif conv_nodes:
			# Just convolutions without clear activation pattern
			return PatternMatch(
				pattern_name="conv_only",
				confidence=0.7,
				matched_nodes=conv_nodes
			)
		
		return None

	def match_fx(self, gm: fx.GraphModule) -> Optional[PatternMatch]:
		# Placeholder FX-based conv+activation detection (future enhancement)
		return None


