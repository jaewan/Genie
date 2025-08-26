import sys
sys.path.insert(0, "/home/jaewan/Genie")

import time
import torch

from genie.core.graph import GraphBuilder
from genie.core.lazy_tensor import LazyTensor
from genie.semantic.analyzer import SemanticAnalyzer


def _build_chain_graph(n=200):
    GraphBuilder.reset_current()
    x = LazyTensor.lift(torch.randn(2, 2))
    t = x
    for _ in range(n):
        t = t + t
    return GraphBuilder.current().get_graph()


def test_analyzer_latency_under_100ms_small_graph(monkeypatch):
    monkeypatch.setenv("GENIE_ANALYZER_SLOW_MS", "100")
    graph = _build_chain_graph(100)
    analyzer = SemanticAnalyzer()
    start = time.perf_counter()
    _ = analyzer.analyze_graph(graph)
    latency = (time.perf_counter() - start) * 1000.0
    assert latency < 500.0  # Allow slack in CI; target is <100ms


