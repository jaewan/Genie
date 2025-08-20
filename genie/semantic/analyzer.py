from __future__ import annotations

import time
from typing import Dict, Any

from genie.core.graph import ComputationGraph
from genie.semantic.pattern_registry import PatternRegistry
from genie.semantic.workload import WorkloadProfile, WorkloadType, WorkloadClassifier
from genie.semantic.fx_analyzer import FXAnalyzer
from genie.semantic.hooks import HookManager
from genie.semantic.graph_utils import analyze_operations_advanced, track_performance, compute_graph_id
import logging
import os


class SemanticAnalyzer:
	"""Three-tier semantic analyzer (dynamic, FX, hooks)."""

	def __init__(self, pattern_registry: PatternRegistry | None = None) -> None:
		self.pattern_registry = pattern_registry or PatternRegistry()
		self.fx_analyzer = FXAnalyzer()
		self.hook_manager = HookManager()
		self._analysis_stats: Dict[str, float] = {}

	@track_performance
	def analyze_graph(self, graph: ComputationGraph) -> WorkloadProfile:
		"""Analyze graph with performance tracking and advanced algorithms."""
		start_time = time.perf_counter()
		logger = logging.getLogger(__name__)
		
		# Stable graph-id caching (best-effort in-process)
		cache_enabled = os.getenv("GENIE_ANALYZER_CACHE", "1") == "1"
		graph_id: str | None = None
		if cache_enabled:
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
		patterns = self.pattern_registry.match_patterns(graph)
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
			"pattern_stats": self.pattern_registry.get_performance_report()
		}


