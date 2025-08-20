from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from genie.core.graph import ComputationGraph
from torch import fx


class PatternPlugin(ABC):
	"""Base class for pattern recognition plugins."""

	@property
	@abstractmethod
	def name(self) -> str:
		...

	@abstractmethod
	def match(self, graph: ComputationGraph) -> Optional["PatternMatch"]:
		...

	# Optional: FX-based matching
	def match_fx(self, gm: fx.GraphModule) -> Optional["PatternMatch"]:  # noqa: D401
		"""Override to support FX-based pattern detection."""
		return None


@dataclass
class PatternMatch:
	pattern_name: str
	confidence: float
	matched_nodes: List[str]


