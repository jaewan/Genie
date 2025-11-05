from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Optional, Union
import torch.fx as fx

# Backward compatibility imports
from djinn.core.graph import ComputationGraph, GraphBuilder


@dataclass
class GraphHandoff:
	"""Data passed from LazyTensor to Semantic Analyzer (per interface contract).
	
	Updated for Refactoring #3: Now supports both old ComputationGraph and new
	FX GraphModule for gradual migration.
	"""
	# New format (Refactoring #3)
	fx_graph: Optional[fx.GraphModule] = None
	
	# Old format (backward compatibility)
	graph: Optional[ComputationGraph] = None
	
	# Common metadata
	lazy_tensors: Dict[str, object] = None
	materialization_frontier: Set[str] = None
	metadata_version: str = "2.0"  # Updated version for FX support
	
	def __post_init__(self):
		"""Initialize default values."""
		if self.lazy_tensors is None:
			self.lazy_tensors = {}
		if self.materialization_frontier is None:
			self.materialization_frontier = set()
	
	@property
	def uses_fx(self) -> bool:
		"""Check if using new FX graph format."""
		return self.fx_graph is not None
	
	def validate(self) -> bool:
		"""Validate handoff data integrity."""
		# Must have either old or new format
		if self.graph is None and self.fx_graph is None:
			return False
		
		# Validate old format if present
		if self.graph is not None and self.fx_graph is None:
			# Only validate tensor mapping if using old format exclusively
			# (Allow empty lazy_tensors for basic validation)
			if self.lazy_tensors:
				# All nodes should have corresponding tensors if map is non-empty
				for node_id in self.graph.nodes:
					if node_id not in self.lazy_tensors:
						return False
			# No dangling references
			for source, target in self.graph.edges:
				if source not in self.graph.nodes or target not in self.graph.nodes:
					return False
		
		# Validate new format if present
		if self.fx_graph is not None:
			# Basic validation - graph should have nodes
			if not hasattr(self.fx_graph, 'graph'):
				return False
		
		return True
	
	def get_graph_for_analysis(self) -> Union[ComputationGraph, fx.GraphModule]:
		"""Get graph in preferred format for analysis.
		
		Returns:
			FX GraphModule if available, otherwise ComputationGraph
		"""
		if self.fx_graph is not None:
			return self.fx_graph
		elif self.graph is not None:
			return self.graph
		else:
			raise ValueError("No graph available in handoff")


def build_graph_handoff() -> GraphHandoff:
	"""Helper to assemble GraphHandoff from current GraphBuilder state.
	
	Updated for Refactoring #3: Attempts to build FX graph first, falls back
	to ComputationGraph for backward compatibility.
	"""
	gb = GraphBuilder.current()
	lazy_map = getattr(gb, "_lazy_tensors", {})
	
	# Try to get FX graph first (new format)
	fx_graph = None
	try:
		from djinn.core.fx_graph_builder import FXGraphBuilder
		fx_builder = FXGraphBuilder.current()
		
		# Check if FX builder has content
		if hasattr(fx_builder, 'value_map') and fx_builder.value_map:
			fx_graph = fx_builder.to_graph_module()
	except (ImportError, AttributeError, Exception) as e:
		# FX not available or empty, fall back to old format
		pass
	
	# Get old format for backward compatibility
	graph = gb.get_graph()
	
	# Compute materialization frontier
	frontier: Set[str] = set()
	with_outgoing = {src for src, _ in gb.edges}
	for node_id in graph.nodes:
		if node_id not in with_outgoing:
			frontier.add(node_id)
	
	return GraphHandoff(
		fx_graph=fx_graph,
		graph=graph,
		lazy_tensors=dict(lazy_map),
		materialization_frontier=frontier
	)


