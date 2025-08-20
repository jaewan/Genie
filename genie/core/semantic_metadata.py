from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
import time
import torch


@dataclass
class SemanticMetadata:
	"""Rich semantic metadata for tensors and FX nodes.

	This is the canonical definition used across the project (LazyTensor and FX).
	"""
	operation_type: str
	tensor_shape: Optional[torch.Size] = None
	dtype: Optional[torch.dtype] = None
	device_hint: str = "remote_accelerator:0"

	# Semantic enrichment
	module_path: Optional[str] = None
	semantic_role: Optional[str] = None
	execution_phase: Optional[str] = None
	workload_hints: Optional[Dict] = None

	# Performance hints
	compute_intensity: float = 1.0
	memory_access: str = "sequential"
	recompute_cost: Optional[float] = None

	# Metadata versioning
	metadata_version: str = "1.0"
	created_at: float = 0.0

	def __post_init__(self) -> None:
		if not self.created_at:
			self.created_at = time.time()


