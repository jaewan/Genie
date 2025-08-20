"""Benchmark and accuracy tests for Semantic Analyzer patterns.

These tests validate:
- Pattern detection for LLM attention and Vision conv+activation
- Analysis latency target (<100ms for ~1K nodes; here much smaller graphs)

Notes:
- Avoid unsupported ops on remote_accelerator (e.g., slicing, cat, embedding).
- Keep graphs minimal and composed of supported ops.
"""

import time
import torch

import genie
from genie.core.graph import GraphBuilder
from genie.semantic.analyzer import SemanticAnalyzer


def setup_module(module):
    """Enable lazy mode for tests."""
    genie.set_lazy_mode(True)


def test_llm_attention_pattern_latency_and_accuracy():
    """Attention-like graph should be detected with good confidence under target latency."""
    # Build attention-like pattern using supported ops
    Q = torch.randn(32, 64, device="remote_accelerator:0")
    K = torch.randn(64, 32, device="remote_accelerator:0")
    V = torch.randn(32, 64, device="remote_accelerator:0")

    scores = Q @ K  # matmul
    attn = torch.softmax(scores, dim=-1)  # softmax
    out = attn @ V  # matmul

    graph = GraphBuilder.current().get_graph()
    analyzer = SemanticAnalyzer()

    start = time.perf_counter()
    profile = analyzer.analyze_graph(graph)
    elapsed = time.perf_counter() - start

    # Latency target
    assert elapsed < 0.1, f"Analysis took {elapsed*1000:.1f}ms (>100ms)"

    # Accuracy: presence of LLM pattern with reasonable confidence
    llm_patterns = [p for p in profile.patterns if p.pattern_name == "llm"]
    assert llm_patterns, "Expected LLM pattern to be detected"
    assert llm_patterns[0].confidence >= 0.75


def test_vision_conv_activation_pattern_latency_and_accuracy():
    """Conv + ReLU graph should be detected with good confidence under target latency."""
    x = torch.randn(1, 3, 64, 64, device="remote_accelerator:0")
    w = torch.randn(8, 3, 3, 3, device="remote_accelerator:0")

    y = torch.conv2d(x, w, stride=1, padding=1)
    z = torch.relu(y)

    graph = GraphBuilder.current().get_graph()
    analyzer = SemanticAnalyzer()

    start = time.perf_counter()
    profile = analyzer.analyze_graph(graph)
    elapsed = time.perf_counter() - start

    # Latency target
    assert elapsed < 0.1, f"Analysis took {elapsed*1000:.1f}ms (>100ms)"

    # Accuracy: presence of Vision pattern or conv_pattern
    vision_patterns = [p for p in profile.patterns if p.pattern_name in ("vision", "conv_pattern")]
    assert vision_patterns, "Expected Vision/Conv pattern to be detected"
    assert vision_patterns[0].confidence >= 0.7


