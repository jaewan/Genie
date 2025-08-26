from __future__ import annotations

import logging
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


def _is_remote_device(dev: Any) -> bool:
	if dev is None:
		return False
	if isinstance(dev, str):
		return dev.startswith("remote_accelerator") or dev.startswith("privateuseone")
	if isinstance(dev, torch.device):
		return dev.type in ("privateuseone", "remote_accelerator")
	return False


def _install_factory_interceptor(fn_name: str) -> None:
	original: Callable[..., Any] = getattr(torch, fn_name)

	def wrapper(*args: Any, **kwargs: Any):
		device = kwargs.get("device", None)
		from .lazy_tensor import LazyTensor  # Local import to avoid cycles
		# Interception conditions:
		# 1) Explicit remote device requested
		# 2) Any input is a LazyTensor (covers *_like ops) even without device kwarg
		has_lazy_input = any(isinstance(a, LazyTensor) for a in args) or any(isinstance(v, LazyTensor) for v in kwargs.values())
		if _is_remote_device(device) or has_lazy_input:
			op_name = f"aten::{fn_name}"
			lt = LazyTensor(operation=op_name, inputs=list(args), kwargs=kwargs)
			import os
			if os.getenv("GENIE_LOG_INTERCEPTS", "0") == "1":
				try:
					shapes = []
					for a in args:
						s = getattr(a, "shape", None)
						shapes.append(tuple(s) if s is not None else None)
					logger.info(f"[Genie] Factory intercepted torch.{fn_name} device={device}, args_shapes={shapes} -> {lt}")
				except Exception:
					logger.info(f"[Genie] Factory intercepted torch.{fn_name} device={device}")
			return lt
		# If device is explicitly CPU or not specified, call original
		return original(*args, **kwargs)

	setattr(torch, fn_name, wrapper)
	logger.debug(f"Installed factory interceptor for torch.{fn_name}")


def install_all() -> None:
	"""Install interceptors for common factory functions.

	Phase 1 shim to ensure creation on remote_accelerator produces LazyTensor
	without relying on full dispatcher coverage for factory ops.
	"""
	for name in (
		"randn",
		"rand",
		"randint",
		"zeros",
		"ones",
		"empty",
		"full",
		"empty_strided",
		"rand_like",
		"randint_like",
		"zeros_like",
		"ones_like",
		"empty_like",
		"full_like",
		"arange",
		"linspace",
		"logspace",
	):
		try:
			_install_factory_interceptor(name)
		except Exception:
			logger.debug(f"Could not install interceptor for torch.{name}")


