import torch

import genie
from genie.core.lazy_tensor import LazyTensor


def setup_module(module):  # noqa: D401
	"""Ensure lazy mode is enabled for tests."""
	genie.set_lazy_mode(True)


def test_unsupported_op_fallback():
	# Create a LazyTensor for an op the executor doesn't implement explicitly
	# Use a reduction like sum which isn't in SimpleExecutor map
	x = torch.randn(4, 4, device="remote_accelerator:0")
	y = LazyTensor("aten::sum", [x], {"dim": 0, "keepdim": False})

	# Materialization should fallback to eager torch.ops and return a real tensor
	out = y.cpu()
	assert isinstance(out, torch.Tensor)
	assert out.ndim == 1
	assert out.shape[0] == 4



