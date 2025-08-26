import sys
sys.path.insert(0, "/home/jaewan/Genie")

import torch

from genie.core.graph import GraphBuilder
from genie.core.lazy_tensor import LazyTensor


def test_graphbuilder_invariants_simple_chain():
    GraphBuilder.reset_current()
    x = LazyTensor.lift(torch.randn(4, 4))
    w = LazyTensor.lift(torch.randn(4, 4))
    y = (x @ w)
    _ = y + y

    graph = GraphBuilder.current().get_graph()
    ok, errors = graph.validate_invariants()
    assert ok, f"Graph invariants failed: {errors}"


def test_graphbuilder_detects_missing_input():
    GraphBuilder.reset_current()
    # Manually construct a broken state
    gb = GraphBuilder.current()
    # Create one node via LazyTensor
    a = LazyTensor.lift(torch.randn(2, 2))
    # Inject a fake node referencing a missing input id
    from genie.core.graph import ComputationNode
    fake = ComputationNode(id="lt_99999", operation="aten::add", inputs=["lt_00000"], outputs=["lt_9999"], metadata={})  # type: ignore[arg-type]
    gb.nodes[fake.id] = fake

    graph = gb.get_graph()
    ok, errors = graph.validate_invariants()
    assert not ok
    assert any("Missing input node" in e for e in errors)


