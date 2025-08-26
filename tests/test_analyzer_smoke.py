import sys
sys.path.insert(0, "/home/jaewan/Genie")

import torch

from genie.core.graph import GraphBuilder
from genie.core.lazy_tensor import LazyTensor
from genie.semantic.analyzer import SemanticAnalyzer


def _simple_cnn_graph():
    GraphBuilder.reset_current()
    x = LazyTensor.lift(torch.randn(1, 3, 8, 8))
    w = LazyTensor.lift(torch.randn(4, 3, 3, 3))
    y = LazyTensor("aten::conv2d", [x, w], {"stride": 1, "padding": 1})
    z = y.relu()
    return GraphBuilder.current().get_graph()


def test_analyzer_smoke_on_cnn():
    graph = _simple_cnn_graph()
    analyzer = SemanticAnalyzer()
    profile = analyzer.analyze_graph(graph)
    assert profile is not None
    assert isinstance(profile.patterns, list)
    # Vision patterns may or may not be detected; just ensure analysis completes fast
    stats = analyzer.get_performance_report()
    assert "analyzer_stats" in stats


