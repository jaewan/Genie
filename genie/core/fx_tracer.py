from __future__ import annotations

from typing import Any, Dict, Optional
import logging

import torch
from torch import fx

from .semantic_metadata import SemanticMetadata

logger = logging.getLogger(__name__)


class SemanticTracer(fx.Tracer):
	"""FX tracer that attaches semantic metadata to nodes.

	This tracer aims to capture operation intent at the FX level and propagate
	shapes/dtypes when available. It uses PyTorch's FX to avoid reinventing LazyTensor graphs.
	"""

	def __init__(self) -> None:
		super().__init__()
		self.module_path_stack = []  # Track current module path

	def call_module(self, m: torch.nn.Module, forward, args, kwargs):  # noqa: ANN001
		self.module_path_stack.append(type(m).__name__)
		try:
			return super().call_module(m, forward, args, kwargs)
		finally:
			self.module_path_stack.pop()

	def create_node(self, kind: str, target: Any, args: Any, kwargs: Dict[str, Any], name: Optional[str] = None, type_expr: Optional[Any] = None):  # noqa: ANN001
		node = super().create_node(kind, target, args, kwargs, name, type_expr)
		# Attach minimal semantic metadata to node.meta
		try:
			op_type = str(target) if isinstance(target, (str,)) else getattr(target, "__name__", str(target))
			module_path = ".".join(self.module_path_stack) if self.module_path_stack else None
			# FX meta can carry arbitrary info
			node.meta.setdefault("semantic", {})
			node.meta["semantic"]["metadata"] = SemanticMetadata(
				operation_type=op_type,
				module_path=module_path,
			). __dict__
		except Exception as e:
			logger.debug(f"Failed to attach semantic metadata to FX node: {e}")
		return node


def trace_module(module: torch.nn.Module) -> fx.GraphModule:
	"""Trace a torch.nn.Module with semantic metadata attached to FX nodes."""
	tracer = SemanticTracer()
	graph = tracer.trace(module)
	gm = fx.GraphModule(module, graph)
	return gm


