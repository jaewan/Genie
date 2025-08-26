import sys
sys.path.insert(0, "/home/jaewan/Genie")

import torch

from genie.core.graph import GraphBuilder
from genie.core.lazy_tensor import LazyTensor
from genie.semantic.analyzer import SemanticAnalyzer
from genie.semantic.workload import WorkloadType


def _simple_graph():
    GraphBuilder.reset_current()
    x = LazyTensor.lift(torch.randn(2, 2))
    w = LazyTensor.lift(torch.randn(2, 2))
    _ = x @ w
    return GraphBuilder.current().get_graph()


def test_generate_stub_plan():
    graph = _simple_graph()
    analyzer = SemanticAnalyzer()
    profile = analyzer.analyze_graph(graph)
    plan = analyzer.generate_stub_plan(graph, profile)

    assert plan.plan_id.startswith("plan_stub_")
    assert len(plan.fragments) == 1
    frag = plan.fragments[0]
    assert frag.fragment_id in plan.placement
    assert plan.placement[frag.fragment_id] in ("cpu", "remote_accelerator:0")


