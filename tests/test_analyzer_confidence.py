import sys
sys.path.insert(0, "/home/jaewan/Genie")

import torch

from genie.core.graph import GraphBuilder
from genie.core.lazy_tensor import LazyTensor
from genie.semantic.analyzer import SemanticAnalyzer
from genie.semantic.workload import WorkloadType


def _build_mlp_graph(layers=3):
    GraphBuilder.reset_current()
    x = LazyTensor.lift(torch.randn(4, 16))
    w = LazyTensor.lift(torch.randn(16, 16))
    t = x @ w
    for _ in range(layers - 1):
        w2 = LazyTensor.lift(torch.randn(16, 16))
        t = (t @ w2).relu()
    return GraphBuilder.current().get_graph()


def test_confidence_for_mlp_like_graph():
    graph = _build_mlp_graph(3)
    analyzer = SemanticAnalyzer()
    profile = analyzer.analyze_graph(graph)
    assert profile.workload_type in (WorkloadType.LLM, WorkloadType.UNKNOWN, WorkloadType.VISION)
    # Should have at least one matched pattern with confidence score
    assert any(p.confidence > 0.0 for p in profile.patterns)


