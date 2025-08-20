from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any


class WorkloadType(str, Enum):
	LLM = "llm"
	VISION = "vision"
	MULTIMODAL = "multimodal"
	RECSYS = "recsys"
	UNKNOWN = "unknown"


@dataclass
class MatchedPattern:
	pattern_name: str
	confidence: float
	subgraph: Any | None = None
	optimization_hints: Dict[str, Any] | None = None
	metadata: Dict[str, Any] | None = None


@dataclass
class WorkloadProfile:
	workload_type: WorkloadType
	patterns: List[MatchedPattern]
	metadata: Dict[str, Any] | None = None
	structure: Dict[str, Any] | None = None
	context: Dict[str, Any] | None = None

	# Optional fields to align with interface-contracts for future phases
	confidence: float | None = None
	phases: Dict[str, List[str]] | None = None
	modalities: Dict[str, Any] | None = None
	dependencies: List[Any] | None = None
	compute_intensity: float | None = None
	memory_bandwidth: float | None = None
	latency_sensitivity: str | None = None
	optimization_hints: Dict[str, Any] | None = None


