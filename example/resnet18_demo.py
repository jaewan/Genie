from __future__ import annotations

import time
from typing import Tuple

import torch

from genie.core.graph import GraphBuilder
from genie.core.fx_tracer import trace_module
from genie.semantic.graph_handoff import build_graph_handoff
from genie.version_validator import validate_environment
from genie.core.lazy_tensor import LazyTensor


def print_header(title: str) -> None:
	print("\n" + "=" * 80)
	print(title)
	print("=" * 80)


def build_resnet_weights() -> Tuple[torch.Tensor, torch.Tensor | None, dict]:
	"""Load torchvision resnet18 and extract first conv layer weights/bias and params."""
	import torchvision.models as models

	model = models.resnet18(weights=None)
	model.eval()

	conv = model.conv1
	weight = conv.weight.detach()
	bias = conv.bias.detach() if conv.bias is not None else None
	params = {"stride": conv.stride, "padding": conv.padding, "dilation": conv.dilation, "groups": conv.groups}
	return weight, bias, params


def run_lazy_conv_relu() -> None:
	print_header("LazyTensor demo: conv2d -> relu using ResNet-18 conv1 weights")

	# Input tensor (CPU). Executor will materialize on CPU in Phase 1.
	input_tensor = torch.randn(1, 3, 224, 224)
	weight, bias, params = build_resnet_weights()

	# Build LazyTensor graph: conv2d -> relu
	conv_lt = LazyTensor(
		"aten::conv2d",
		[input_tensor, weight] + ([bias] if bias is not None else []),
		{"stride": params["stride"], "padding": params["padding"], "dilation": params["dilation"], "groups": params["groups"]},
	)
	relu_lt = LazyTensor("aten::relu", [conv_lt], {})

	# Materialize and time it
	start = time.time()
	result = relu_lt.cpu()
	elapsed_ms = (time.time() - start) * 1000.0

	print(f"Output shape: {tuple(result.shape)}  |  dtype: {result.dtype}")
	print(f"Materialization time: {elapsed_ms:.2f} ms")

	# Graph info
	graph = GraphBuilder.current().get_graph()
	print(f"Graph nodes: {len(graph.nodes)} | edges: {len(graph.edges)}")


def run_fx_trace() -> None:
	print_header("FX trace of ResNet-18 (structure only)")
	import torchvision.models as models

	model = models.resnet18(weights=None).eval()
	gm = trace_module(model)
	GraphBuilder.current().set_fx_graph(gm.graph)
	print(f"FX nodes: {len(list(gm.graph.nodes))}")
	# Show first few nodes for confirmation
	for i, n in enumerate(gm.graph.nodes):
		if i >= 8:
			break
		print(f"  {i:02d}: {n.op}::{n.target}")


def main() -> None:
	print_header("Genie ResNet-18 Demo")
	ok = validate_environment(strict=False)
	print(f"Environment validation: {'OK' if ok else 'WARN'}  |  torch={torch.__version__}")

	# Run a tiny LazyTensor graph based on ResNet conv1
	run_lazy_conv_relu()

	# FX tracing for full model structure (no execution)
	run_fx_trace()

	# Build GraphHandoff preview
	gh = build_graph_handoff()
	print_header("GraphHandoff summary")
	print(f"handoff.valid={gh.validate()}  |  nodes={len(gh.graph.nodes)}  |  frontier={len(gh.materialization_frontier)}")


if __name__ == "__main__":
	main()


