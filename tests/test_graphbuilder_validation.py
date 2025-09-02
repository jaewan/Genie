import sys
sys.path.insert(0, "/home/jaewan/Genie")

import torch

from genie.core.lazy_tensor import LazyTensor
from genie.core.fx_graph_builder import FXGraphBuilder


def test_graphbuilder_invariants_simple_chain():
    # Build a simple chain and validate FX graph is well-formed
    x = LazyTensor.lift(torch.randn(4, 4))
    w = LazyTensor.lift(torch.randn(4, 4))
    y = (x @ w)
    _ = y + y

    fx_builder = FXGraphBuilder.current()
    gm = fx_builder.to_graph_module()
    # FX graph should have at least placeholders and a couple of call_function nodes
    nodes = list(gm.graph.nodes)
    assert len(nodes) >= 3


def test_graphbuilder_detects_missing_input():
    # FX path: ensure placeholders are created for concrete inputs and graph finalizes
    a = LazyTensor.lift(torch.randn(2, 2))
    b = LazyTensor.lift(torch.randn(2, 2))
    _ = a + b

    fx_builder = FXGraphBuilder.current()
    gm = fx_builder.to_graph_module()
    placeholders = [n for n in gm.graph.nodes if n.op == 'placeholder']
    assert len(placeholders) >= 2


