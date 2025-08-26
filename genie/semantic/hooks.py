from __future__ import annotations

from typing import Callable, Dict, Any, List, Tuple

import torch.nn as nn


class HookManager:
	"""Minimal hook manager scaffold for Phase 1 P1.

	Collects simple per-module context via forward hooks when injected.
	"""

	def __init__(self) -> None:
		self.hooks: Dict[str, Callable] = {}
		self.context: Dict[str, Any] = {}
		self._captures: List[Tuple[str, Dict[str, Any]]] = []

	def register_hook(self, module_name: str, hook_fn: Callable) -> None:
		self.hooks[module_name] = hook_fn

	def inject_hooks(self, model: nn.Module) -> None:
		for name, module in model.named_modules():
			# Always capture minimal context
			module.register_forward_hook(lambda m, i, o, n=name: self.capture_context(n, m, i, o))
			# Run additional custom hook if provided for this module type
			hook_fn = self.hooks.get(type(module).__name__)
			if hook_fn is not None:
				module.register_forward_hook(lambda m, i, o, n=name, f=hook_fn: f(n, m, i, o))

	def capture_context(self, name: str, module: nn.Module, inputs, output) -> None:
		in_shapes = [getattr(i, "shape", None) for i in inputs] if isinstance(inputs, (list, tuple)) else [getattr(inputs, "shape", None)]
		out_shape = getattr(output, "shape", None)
		info = {
			"module_path": name,
			"module_type": type(module).__name__,
			"input_shapes": in_shapes,
			"output_shape": out_shape,
			"execution_phase": self._detect_phase(module, inputs),
			"semantic_role": self._infer_semantic_role(name, module),
		}
		self.context[name] = info
		self._captures.append((name, info))

	def get_context(self, graph=None):  # noqa: ANN001
		return self.context

	def get_captures(self) -> List[Tuple[str, Dict[str, Any]]]:
		return list(self._captures)

	def _detect_phase(self, module: nn.Module, inputs) -> str:  # noqa: ANN001
		# Placeholder phase detection: identify common train/eval or encode/decode hints
		if hasattr(module, 'training'):
			return 'train' if module.training else 'eval'
		return 'unknown'

	def _infer_semantic_role(self, name: str, module: nn.Module) -> str:  # noqa: ANN001
		mtype = type(module).__name__.lower()
		if 'attn' in mtype or 'attention' in mtype:
			return 'attention'
		if 'conv' in mtype:
			return 'conv'
		if 'norm' in mtype or 'layernorm' in mtype or 'batchnorm' in mtype:
			return 'normalization'
		if 'embedding' in mtype:
			return 'embedding'
		return 'module'


