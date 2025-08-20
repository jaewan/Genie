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
		if _is_remote_device(device):
			from .lazy_tensor import LazyTensor  # Local import to avoid cycles
			op_name = f"aten::{fn_name}"
			return LazyTensor(operation=op_name, inputs=list(args), kwargs=kwargs)
		# If device is explicitly CPU or not specified, call original
		return original(*args, **kwargs)

	setattr(torch, fn_name, wrapper)
	logger.debug(f"Installed factory interceptor for torch.{fn_name}")


def install_all() -> None:
	"""Install interceptors for common factory functions.

	Phase 1 shim to ensure creation on remote_accelerator produces LazyTensor
	without relying on full dispatcher coverage for factory ops.
	"""
	for name in ("randn", "zeros", "ones", "empty", "full"):
		try:
			_install_factory_interceptor(name)
		except Exception:
			logger.debug(f"Could not install interceptor for torch.{name}")


