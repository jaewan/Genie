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


