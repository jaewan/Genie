from __future__ import annotations

import time
from typing import Dict, Any, Union
import torch.fx as fx

from genie.core.graph import ComputationGraph
from genie.semantic.pattern_registry import PatternRegistry
from genie.semantic.workload import WorkloadProfile, WorkloadType, WorkloadClassifier
from genie.semantic.fx_analyzer import FXAnalyzer
from genie.semantic.hooks import HookManager
from genie.semantic.graph_utils import analyze_operations_advanced, track_performance, compute_graph_id
import logging
import os


class SemanticAnalyzer:
	"""Three-tier semantic analyzer (dynamic, FX, hooks).
	
	Updated for Refactoring #3: Now supports both ComputationGraph and FX GraphModule.
	Updated for Refactoring #5: Now supports dependency injection for pattern matchers.
	"""

	def __init__(
		self, 
		pattern_registry: PatternRegistry | None = None,
		pattern_matcher=None
	) -> None:
		"""Initialize semantic analyzer.
		
		Args:
			pattern_registry: (Deprecated) Legacy pattern registry. Use pattern_matcher instead.
			pattern_matcher: IPatternMatcher instance for pattern matching.
						   If None, uses default NetworkXPatternMatcher.
		"""
		# Support both old and new initialization
		if pattern_matcher is not None:
			# New approach: Use injected pattern matcher
			self.pattern_matcher = pattern_matcher
			self.pattern_registry = None  # Deprecated
		elif pattern_registry is not None:
			# Old approach: Wrap pattern_registry in NetworkXPatternMatcher
			from genie.semantic.pattern_matcher import NetworkXPatternMatcher
			self.pattern_matcher = NetworkXPatternMatcher(pattern_registry)
			self.pattern_registry = pattern_registry  # Keep for backward compat
		else:
			# Default: Use NetworkX matcher with default patterns
			from genie.semantic.pattern_matcher import get_default_pattern_matcher
			self.pattern_matcher = get_default_pattern_matcher()
			self.pattern_registry = None
		
		self.fx_analyzer = FXAnalyzer()
		self.hook_manager = HookManager()
		self._analysis_stats: Dict[str, float] = {}

	@track_performance
	def analyze_graph(self, graph: Union[ComputationGraph, fx.GraphModule]) -> WorkloadProfile:
		"""Analyze graph with performance tracking and advanced algorithms."""
		start_time = time.perf_counter()
		logger = logging.getLogger(__name__)
		
		# Stable graph-id caching (best-effort in-process)
		# Note: Currently only supports ComputationGraph hashing
		cache_enabled = os.getenv("GENIE_ANALYZER_CACHE", "1") == "1"
		graph_id: str | None = None
		if cache_enabled and isinstance(graph, ComputationGraph):
			try:
				graph_id = compute_graph_id(graph)
			except Exception:
				graph_id = None
				logger.debug("Graph ID computation failed", exc_info=True)
		if cache_enabled and graph_id:
			cached = getattr(self, "_cache", {}).get(graph_id)
			if cached is not None:
				return cached
		
		# Use advanced operation analysis
		ops_metadata = analyze_operations_advanced(graph)
		
		structural_info = self.fx_analyzer.analyze_structure(graph)
		semantic_context = self.hook_manager.get_context(graph)
		
		# Match patterns using injected pattern matcher (Refactoring #5)
		pattern_result = self.pattern_matcher.match_patterns(graph)
		if pattern_result.is_ok:
			patterns = pattern_result.unwrap()
		else:
			# Log error but continue with empty patterns
			logger.warning(f"Pattern matching had errors: {pattern_result.error}")
			patterns = []
		
		workload_type = WorkloadClassifier().classify(patterns)
		
		total_time = time.perf_counter() - start_time
		self._analysis_stats['last_analysis_time'] = total_time
		
		# Feature-flagged logging
		slow_ms_env = os.getenv("GENIE_ANALYZER_SLOW_MS", "100")
		try:
			slow_threshold = float(slow_ms_env) / 1000.0
		except Exception:
			slow_threshold = 0.1
		debug_enabled = os.getenv("GENIE_ANALYZER_DEBUG", "0") == "1"
		if total_time > slow_threshold:
			logger.warning("Graph analysis took %.1fms (>%.1fms target)", total_time * 1000.0, slow_threshold * 1000.0)
		elif debug_enabled:
			logger.debug("Graph analysis took %.3fms", total_time * 1000.0)
		
		profile = WorkloadProfile(
			workload_type=workload_type,
			patterns=patterns,
			metadata=ops_metadata,
			structure=structural_info,
			context=semantic_context,
		)
		if cache_enabled and graph_id:
			cache = getattr(self, "_cache", None)
			if cache is None:
				self._cache = {}
				cache = self._cache
			cache[graph_id] = profile
		return profile

	def get_performance_report(self) -> Dict[str, Any]:
		"""Get comprehensive performance report."""
		return {
			"analyzer_stats": self._analysis_stats,
			"pattern_stats": self.pattern_matcher.get_performance_report()
		}

	def generate_stub_plan(self, graph: Union[ComputationGraph, fx.GraphModule], profile: "WorkloadProfile") -> "ExecutionPlan":
		"""Emit a minimal ExecutionPlan with placement hints.

		This is a Phase-1 stub that wraps the entire graph into a single fragment
		with conservative placement based on workload type.
		"""
		from genie.semantic.workload import ExecutionPlan, PlanFragment, WorkloadType
		fragment_id = "fragment_0"
		fragment = PlanFragment(fragment_id=fragment_id, subgraph=graph, inputs=[], outputs=[])
		# Simple placement heuristic
		placement_device = {
			WorkloadType.VISION: "remote_accelerator:0",
			WorkloadType.LLM: "remote_accelerator:0",
			WorkloadType.MULTIMODAL: "remote_accelerator:0",
			WorkloadType.RECSYS: "cpu",
			WorkloadType.UNKNOWN: "cpu",
		}.get(profile.workload_type, "cpu")
		plan = ExecutionPlan(
			plan_id="plan_stub_0",
			fragments=[fragment],
			placement={fragment_id: placement_device},
			transfers=[],
			feature_flags={"overlap_io": False, "micro_batching": False},
		)
		return plan