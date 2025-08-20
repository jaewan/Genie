from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import logging

import torch

logger = logging.getLogger(__name__)


_OP_MAP = {
	"aten::add": torch.add,
	"aten::sub": torch.sub,
	"aten::mul": torch.mul,
	"aten::div": torch.div,
	"aten::pow": torch.pow,
	"aten::matmul": torch.matmul,
	"aten::mm": torch.mm,
	"aten::bmm": torch.bmm,
	"aten::addmm": torch.addmm,
	"aten::linear": torch.nn.functional.linear,
	"aten::relu": torch.relu,
	"aten::sigmoid": torch.sigmoid,
	"aten::tanh": torch.tanh,
	"aten::softmax": torch.softmax,
	"aten::log_softmax": torch.log_softmax,
	"aten::gelu": torch.nn.functional.gelu,
	"aten::leaky_relu": torch.nn.functional.leaky_relu,
	"aten::conv2d": torch.conv2d,
	"aten::conv1d": torch.conv1d,
	"aten::batch_norm": torch.batch_norm,
	"aten::layer_norm": torch.layer_norm,
	"aten::dropout": torch.dropout,
	"aten::view": torch.Tensor.view,
	"aten::reshape": torch.reshape,
	"aten::transpose": torch.transpose,
	"aten::permute": torch.Tensor.permute,
	"aten::cat": torch.cat,
	"aten::stack": torch.stack,
	"aten::split": torch.split,
	"aten::squeeze": torch.squeeze,
	"aten::unsqueeze": torch.unsqueeze,
	"aten::sum": torch.sum,
	"aten::mean": torch.mean,
	"aten::max": torch.max,
	"aten::min": torch.min,
	"aten::eq": torch.eq,
	"aten::ne": torch.ne,
	"aten::lt": torch.lt,
	"aten::gt": torch.gt,
	"aten::le": torch.le,
	"aten::ge": torch.ge,
	# Creation ops handled separately
}


def _to_meta_tensor(x: Any) -> Any:
	"""Convert inputs to meta tensors when possible.

	- torch.Tensor -> empty meta tensor with same shape/dtype
	- LazyTensor -> empty meta tensor if it has shape/dtype
	- Tuple/list of ints remains as-is (sizes)
	- Other types returned unchanged
	"""
	from .lazy_tensor import LazyTensor  # Local import to avoid cycles at module import time

	if isinstance(x, torch.Tensor):
		return torch.empty_like(x, device="meta")
	if isinstance(x, LazyTensor):
		if x.shape is not None and x.dtype is not None:
			return torch.empty(*x.shape, dtype=x.dtype, device="meta")
		return x  # Not enough info
	if isinstance(x, (list, tuple)):
		return type(x)(_to_meta_tensor(v) for v in x)
	return x


def infer_via_meta(operation: str, inputs: List[Any], kwargs: Optional[Dict[str, Any]] = None) -> Tuple[Optional[torch.Size], Optional[torch.dtype]]:
	"""Infer shape and dtype by executing the op on meta tensors.

	Returns (shape, dtype) or (None, None) if inference fails.
	"""
	kwargs = kwargs or {}

	# Handle creation ops explicitly
	if operation in {"aten::randn", "aten::zeros", "aten::ones", "aten::empty", "aten::full"}:
		try:
			if operation == "aten::randn":
				y = torch.randn(*inputs, device="meta", **kwargs)
			elif operation == "aten::zeros":
				y = torch.zeros(*inputs, device="meta", **kwargs)
			elif operation == "aten::ones":
				y = torch.ones(*inputs, device="meta", **kwargs)
			elif operation == "aten::empty":
				y = torch.empty(*inputs, device="meta", **kwargs)
			elif operation == "aten::full":
				# inputs: (size, fill_value)
				y = torch.full(inputs[0], inputs[1], device="meta", **kwargs)
			return y.shape, y.dtype  # type: ignore[return-value]
		except Exception as e:
			logger.debug(f"Meta creation inference failed for {operation}: {e}")
			return None, None

	func = _OP_MAP.get(operation)
	if func is None:
		return None, None

	try:
		meta_args = [_to_meta_tensor(a) for a in inputs]
		# Some bound methods (Tensor.view/permutation) need instance first
		if func is torch.Tensor.view:
			result = meta_args[0].view(meta_args[1])  # type: ignore[index]
		elif func is torch.Tensor.permute:
			result = meta_args[0].permute(meta_args[1])  # type: ignore[index]
		else:
			result = func(*meta_args, **kwargs)
		# Most ops return Tensor or tuple; handle Tensor only here
		if isinstance(result, torch.Tensor):
			return result.shape, result.dtype  # type: ignore[return-value]
		return None, None
	except Exception as e:
		logger.debug(f"Meta inference failed for {operation}: {e}")
		return None, None


