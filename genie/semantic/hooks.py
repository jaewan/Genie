from __future__ import annotations

from typing import Callable, Dict, Any

import torch.nn as nn


class HookManager:
	"""Minimal hook manager scaffold for Phase 1 P1.

	Collects simple per-module context via forward hooks when injected.
	"""

	def __init__(self) -> None:
		self.hooks: Dict[str, Callable] = {}
		self.context: Dict[str, Any] = {}

	def register_hook(self, module_name: str, hook_fn: Callable) -> None:
		self.hooks[module_name] = hook_fn

	def inject_hooks(self, model: nn.Module) -> None:
		for name, module in model.named_modules():
			hook_fn = self.hooks.get(type(module).__name__)
			if hook_fn is None:
				continue
			module.register_forward_hook(lambda m, i, o, n=name: self.capture_context(n, m, i, o))

	def capture_context(self, name: str, module: nn.Module, inputs, output) -> None:
		self.context[name] = {
			"module_type": type(module).__name__,
			"input_shapes": [getattr(i, "shape", None) for i in inputs] if isinstance(inputs, (list, tuple)) else [getattr(inputs, "shape", None)],
			"output_shape": getattr(output, "shape", None),
		}

	def get_context(self, graph=None):  # noqa: ANN001
		return self.context


