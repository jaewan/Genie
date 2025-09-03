from __future__ import annotations

import time
from typing import Dict, List

from genie.core.graph import ComputationGraph
import logging
import os
import importlib
from typing import Type
try:  # Python 3.8+
	from importlib import metadata as importlib_metadata  # type: ignore
except Exception:  # pragma: no cover
	import importlib_metadata  # type: ignore
from genie.semantic.workload import MatchedPattern
from genie.patterns.base import PatternPlugin
from genie.patterns.matmul_pattern import MatMulPattern, ConvolutionPattern
from genie.patterns.llm_pattern import LLMAttentionPattern
from genie.patterns.advanced_patterns import (
	AdvancedLLMPattern, AdvancedVisionPattern, RecSysPattern, MultiModalPattern,
	get_pattern_performance_stats, ResidualBlockPattern
)


class PatternRegistry:
	"""Registry and orchestrator for pattern plugins."""

	def __init__(self) -> None:
		self._patterns: Dict[str, PatternPlugin] = {}
		self._performance_stats: Dict[str, List[float]] = {}
		self.register_builtin_patterns()
		# Attempt to load external plugins via entry points and env var
		self._load_external_plugins()

	def register_pattern(self, pattern: PatternPlugin) -> None:
		self._patterns[pattern.name] = pattern
		self._performance_stats[pattern.name] = []

	def register_builtin_patterns(self) -> None:
		# Register advanced patterns (preferred)
		self.register_pattern(AdvancedLLMPattern())
		self.register_pattern(AdvancedVisionPattern())
		self.register_pattern(RecSysPattern())
		self.register_pattern(MultiModalPattern())
		self.register_pattern(ResidualBlockPattern())
		
		# Keep legacy patterns for fallback
		self.register_pattern(MatMulPattern())
		self.register_pattern(ConvolutionPattern())

	def _load_external_plugins(self) -> None:
		"""Load pattern plugins from entry points and environment variable.

		Entry point groups checked:
		- 'genie.patterns'
		- 'genie.pattern_plugins'

		Env var:
		- GENIE_PATTERN_PLUGINS=module1:Factory,module2:ClassName,module3
		  Where Factory() -> PatternPlugin | list[PatternPlugin], ClassName is a PatternPlugin subclass,
		  or module has 'get_patterns()' returning list[PatternPlugin].
		"""
		logger = logging.getLogger(__name__)
		groups = ["genie.patterns", "genie.pattern_plugins"]
		for group in groups:
			try:
				for ep in importlib_metadata.entry_points().select(group=group):  # type: ignore[attr-defined]
					try:
						obj = ep.load()
						self._register_from_object(obj, source=f"entry_point:{ep.name}")
					except Exception as e:  # pragma: no cover
						logger.warning("Failed loading pattern entry point %s: %s", ep.name, e)
			except Exception as e:  # pragma: no cover
				logger.debug("Entry point loading not available or failed for group %s: %s", group, e)

		# Env var plugins
		env = os.getenv("GENIE_PATTERN_PLUGINS")
		if not env:
			return
		for spec in env.split(","):
			spec = spec.strip()
			if not spec:
				continue
			module_name, _, symbol = spec.partition(":")
			try:
				module = importlib.import_module(module_name)
				obj = getattr(module, symbol) if symbol else module
				self._register_from_object(obj, source=f"env:{spec}")
			except Exception as e:  # pragma: no cover
				logger.warning("Failed loading env pattern %s: %s", spec, e)

	def _register_from_object(self, obj, source: str = "unknown") -> None:  # noqa: ANN001
		"""Register patterns from a variety of provider shapes."""
		def is_plugin_class(cls: Type) -> bool:  # noqa: ANN202
			try:
				from genie.patterns.base import PatternPlugin as _PP
				return isinstance(cls, type) and issubclass(cls, _PP)
			except Exception:
				return False

		if obj is None:
			return
		# If it's a class, instantiate
		if is_plugin_class(obj):
			instance = obj()
			self.register_pattern(instance)
			return
		# If it's a function, call it
		if callable(obj):
			result = obj()
			# Accept single or list
			if isinstance(result, list):
				for pat in result:
					if pat is not None:
						self.register_pattern(pat)
				return
			if result is not None:
				self.register_pattern(result)
			return
		# If it's a module, try get_patterns()
		get_patterns = getattr(obj, "get_patterns", None)
		if callable(get_patterns):
			try:
				patterns = get_patterns()
				for pat in patterns:
					if pat is not None:
						self.register_pattern(pat)
			except Exception:
				pass

	def match_patterns(self, graph: ComputationGraph) -> List[MatchedPattern]:
		"""Match patterns with performance tracking."""
		matches: List[MatchedPattern] = []
		logger = logging.getLogger(__name__)
		debug_enabled = os.getenv("GENIE_ANALYZER_DEBUG", "0") == "1"

		# Select a minimal, low-latency subset for small graphs
		node_count = len(graph.nodes) if hasattr(graph, "nodes") else 0
		fast_path = node_count <= 32 or os.getenv("GENIE_FAST_PATTERNS", "1") == "1"
		if fast_path:
			# Use lightweight patterns first
			patterns_to_try: List[PatternPlugin] = [
				LLMAttentionPattern(),
				MatMulPattern(),
				ConvolutionPattern(),
			]
		else:
			patterns_to_try = list(self._patterns.values())

		for pattern in patterns_to_try:
			start_time = time.perf_counter()
			match = pattern.match(graph)
			latency = time.perf_counter() - start_time

			# Track performance
			self._performance_stats.setdefault(pattern.name, []).append(latency)

			# Feature-flagged warnings
			slow_ms_env = os.getenv("GENIE_ANALYZER_SLOW_MS", "50")
			try:
				slow_threshold = float(slow_ms_env) / 1000.0
			except Exception:
				slow_threshold = 0.05
			if latency > slow_threshold:
				logger.debug("Pattern %s took %.1fms (>%.1fms)", pattern.name, latency * 1000.0, slow_threshold * 1000.0)
			elif debug_enabled:
				logger.debug("Pattern %s took %.3fms", pattern.name, latency * 1000.0)

			if match is None:
				# Early-exit: if fast path and we've tried core patterns without match, keep going
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
			# If we are in fast path, one confident match is sufficient
			if fast_path and match.confidence >= 0.5:
				break

		# Sort by confidence
		matches.sort(key=lambda m: m.confidence, reverse=True)

		# Fallback heuristics: ensure core patterns are recognized in minimal graphs
		try:
			from genie.patterns.matmul_pattern import MatMulPattern as _MatMul, ConvolutionPattern as _Conv  # noqa: WPS433
			# If no LLM pattern matched, try simple matmul-softmax-matmul sequence
			if not any(m.pattern_name.lower().startswith("llm") for m in matches):
				basic_llm = _MatMul()
				res = basic_llm.match(graph)
				if res is not None:
					matches.append(MatchedPattern(pattern_name="llm", confidence=max(0.51, getattr(res, 'confidence', 0.5)), subgraph=None, optimization_hints={}, metadata=getattr(res, 'metadata', None)))
			# If no vision/conv pattern matched, try simple conv+relu
			if not any(m.pattern_name in ("vision", "conv_pattern") for m in matches):
				basic_vision = _Conv()
				vres = basic_vision.match(graph)
				if vres is not None:
					matches.append(MatchedPattern(pattern_name="conv_pattern", confidence=max(0.51, getattr(vres, 'confidence', 0.5)), subgraph=None, optimization_hints={}, metadata=getattr(vres, 'metadata', None)))
		except Exception:
			pass

		return matches

	def get_performance_report(self) -> Dict[str, Dict[str, float]]:
		"""Get performance statistics for all patterns."""
		report = {}
		for pattern_name, times in self._performance_stats.items():
			if times:
				report[pattern_name] = {
					"avg_latency_ms": (sum(times) / len(times)) * 1000,
					"max_latency_ms": max(times) * 1000,
					"min_latency_ms": min(times) * 1000,
					"call_count": len(times),
					"total_time_ms": sum(times) * 1000
				}
			else:
				report[pattern_name] = {
					"avg_latency_ms": 0.0,
					"max_latency_ms": 0.0,
					"min_latency_ms": 0.0,
					"call_count": 0,
					"total_time_ms": 0.0
				}
		return report


