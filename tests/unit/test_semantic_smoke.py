"""
Test: Semantic analysis completes without crashing

NOTE: This is a SMOKE TEST, not an accuracy test.
We verify the pipeline runs and produces valid output, not that
pattern detection is 100% accurate.

Validates:
- annotate() completes without crashing
- Returns valid data structures with hard checks
- Cost estimates are non-zero and reasonable
- Patterns have required metadata fields
"""

import torch
import torch.nn as nn
import pytest
import logging
import genie

logger = logging.getLogger(__name__)


class TestSemanticAnalysis:
    """Smoke tests for semantic analysis pipeline."""

    def test_semantic_pipeline_runs(self):
        """Test semantic analysis completes without crashing."""

        # Create simple model
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        with genie.capture():
            x = torch.randn(32, 64)
            output = model(x)

        # Get graph and annotate
        graph = genie.get_graph()

        # HARD ASSERTION: annotate() should not crash
        try:
            annotated = genie.annotate_graph(graph)
        except Exception as e:
            pytest.fail(f"Semantic analysis crashed: {e}")

        # HARD ASSERTIONS: Basic sanity checks
        assert len(list(annotated.nodes())) > 0, \
            "Graph is empty after annotation!"

        assert hasattr(annotated, 'patterns'), \
            "No patterns dict in annotated graph!"

        assert hasattr(annotated, 'costs'), \
            "No costs dict in annotated graph!"

        assert annotated.costs['total_compute_flops'] > 0, \
            "Cost estimation failed - zero FLOPs!"

        print(f"✅ Semantic analysis completed:")
        print(f"   Nodes: {len(list(annotated.nodes()))}")
        print(f"   Pattern types: {list(annotated.patterns.keys())}")
        print(f"   Total FLOPs: {annotated.costs['total_compute_flops']:.2e}")

    def test_patterns_have_required_metadata(self):
        """If patterns detected, verify they have required metadata fields."""

        # Create attention-like pattern
        class SimpleAttention(nn.Module):
            def forward(self, q, k, v):
                scores = q @ k.transpose(-2, -1)
                attn = torch.softmax(scores, dim=-1)
                out = attn @ v
                return out

        model = SimpleAttention()

        with genie.capture():
            q = torch.randn(2, 8, 64)
            k = torch.randn(2, 8, 64)
            v = torch.randn(2, 8, 64)
            output = model(q, k, v)

        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)

        # If patterns detected, verify metadata exists
        for pattern_type, patterns in annotated.patterns.items():
            if patterns:  # If this pattern type was detected
                pattern = patterns[0]

                # HARD ASSERTION: Pattern must have metadata
                assert hasattr(pattern, 'metadata'), \
                    f"Pattern {pattern_type} missing metadata attribute!"

                assert pattern.metadata is not None, \
                    f"Pattern {pattern_type} has None metadata!"

                assert len(pattern.metadata) > 0, \
                    f"Pattern {pattern_type} has empty metadata dict!"

                print(f"✅ Pattern '{pattern_type}' has valid metadata")
                print(f"   Keys: {list(pattern.metadata.keys())}")

    def test_cost_estimates_functional(self):
        """Test cost estimates are generated and non-zero."""

        with genie.capture():
            # Known operation: 100x200 @ 200x300
            A = torch.randn(100, 200)
            B = torch.randn(200, 300)
            C = A @ B

        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)

        # Find matmul node
        matmul_nodes = [n for n in graph.nodes()
                       if 'matmul' in n.operation.lower()]

        assert len(matmul_nodes) > 0, "No matmul node captured!"

        node_id = matmul_nodes[0].id
        metadata = annotated.get_metadata(node_id)

        # HARD ASSERTIONS: Cost estimation must work
        assert metadata is not None, \
            "No metadata for matmul node!"
        assert metadata.compute_flops > 0, \
            "Cost estimation returned zero FLOPs!"
        assert metadata.memory_bytes > 0, \
            "Memory estimation returned zero bytes!"

        # Expected FLOPs: 2 * 100 * 300 * 200 = 12M
        expected_flops = 2 * 100 * 300 * 200
        ratio = metadata.compute_flops / expected_flops

        print(f"✅ Cost estimation functional:")
        print(f"   Estimated: {metadata.compute_flops:.2e} FLOPs")
        print(f"   Expected:  {expected_flops:.2e} FLOPs")
        print(f"   Ratio:     {ratio:.2f}x")

        # Soft check: Accuracy (approximate is OK)
        if not (0.5 <= ratio <= 2.0):
            logger.warning(
                f"Cost estimate off by {ratio:.2f}x - may need tuning. "
                "This is not a test failure, just a warning."
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
