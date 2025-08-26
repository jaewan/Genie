import sys
sys.path.insert(0, "/home/jaewan/Genie")

import torch

from genie.core.graph import GraphBuilder
from genie.core.lazy_tensor import LazyTensor
from genie.semantic.analyzer import SemanticAnalyzer
from genie.semantic.workload import WorkloadType


def _build_llm_like_graph():
    GraphBuilder.reset_current()
    # Simple attention-like chain: matmul -> softmax -> matmul
    qk = LazyTensor("aten::matmul", [LazyTensor.lift(torch.randn(8, 16)), LazyTensor.lift(torch.randn(16, 16))], {})
    attn = LazyTensor("aten::softmax", [qk], {"dim": -1})
    out = LazyTensor("aten::matmul", [attn, LazyTensor.lift(torch.randn(16, 16))], {})
    return GraphBuilder.current().get_graph()


def _build_vision_like_graph():
    GraphBuilder.reset_current()
    x = LazyTensor.lift(torch.randn(1, 3, 16, 16))
    w = LazyTensor.lift(torch.randn(8, 3, 3, 3))
    y = LazyTensor("aten::conv2d", [x, w], {"stride": 1, "padding": 1})
    z = y.relu()
    return GraphBuilder.current().get_graph()


def test_exit_latency_and_classification_llm():
    analyzer = SemanticAnalyzer()
    graph = _build_llm_like_graph()
    profile = analyzer.analyze_graph(graph)
    assert profile.workload_type in (WorkloadType.LLM, WorkloadType.UNKNOWN)
    # Latency guard (loose): analysis under 500ms
    assert analyzer.get_performance_report()["analyzer_stats"]["last_analysis_time"] < 0.5


def test_exit_latency_and_classification_vision():
    analyzer = SemanticAnalyzer()
    graph = _build_vision_like_graph()
    profile = analyzer.analyze_graph(graph)
    assert profile.workload_type in (WorkloadType.VISION, WorkloadType.UNKNOWN)
    assert analyzer.get_performance_report()["analyzer_stats"]["last_analysis_time"] < 0.5


