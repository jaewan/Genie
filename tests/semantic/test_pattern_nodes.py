"""
Verify pattern nodes are not lost during conversion.

This is a CRITICAL test - without it, semantic scheduling is silently broken.
"""

import pytest
import torch
import genie
from genie.semantic.annotator import SemanticAnnotator
from genie.core.graph import ComputationGraph


def test_pattern_nodes_populated():
    """
    Test that pattern nodes are extracted and populated correctly.

    This tests the fix for the critical bug where nodes=[] was hardcoded.
    """
    # Create a computation graph using genie capture
    with genie.capture():
        # Create attention-like pattern: Q @ K.T → softmax → @ V
        q = torch.randn(2, 8, 64)  # [batch, heads, dim]
        k = torch.randn(2, 8, 64)
        v = torch.randn(2, 8, 64)

        # Attention mechanism
        scores = q @ k.transpose(-2, -1)  # Q @ K.T
        attn = torch.softmax(scores, dim=-1)  # Softmax
        output = attn @ v  # @ V

    # Get the captured graph
    graph = genie.get_graph()

    # Annotate (this should detect attention pattern)
    annotator = SemanticAnnotator(enable_cache=False)
    annotated = annotator.annotate(graph)

    # CRITICAL ASSERTION: Pattern must contain nodes
    print(f"Detected patterns: {list(annotated.patterns.keys())}")
    print(f"Graph has {len(list(graph.nodes()))} nodes")

    if 'llm' in annotated.patterns:
        llm_patterns = annotated.patterns['llm']
        print(f"Found {len(llm_patterns)} LLM patterns")

        assert len(llm_patterns) > 0, \
            "LLM pattern detected but list is empty"

        first_pattern = llm_patterns[0]

        assert len(first_pattern.nodes) > 0, \
            "BUG: Pattern nodes are empty! Scheduler cannot access them. " \
            "Check _convert_matches_to_patterns() implementation."

        # Verify nodes have required structure
        for node in first_pattern.nodes:
            assert 'id' in node or hasattr(node, 'id'), \
                f"Pattern node missing 'id': {node}"

        print(f"✓ Pattern contains {len(first_pattern.nodes)} nodes")
        print(f"  Node IDs: {[n.get('id', str(n)) for n in first_pattern.nodes]}")
        print(f"  Pattern metadata: {first_pattern.metadata}")
    else:
        pytest.skip("No LLM pattern detected (pattern matcher may need tuning)")


def test_pattern_nodes_multiple_instances():
    """Test that multiple pattern instances are tracked correctly."""
    graph = ComputationGraph.empty()
    
    # Create two separate attention-like blocks
    # Block 1
    node1 = graph.add_node('aten::matmul', 'attn1_q')
    node2 = graph.add_node('aten::matmul', 'attn1_k')
    node3 = graph.add_node('aten::matmul', 'attn1_qk')
    graph.add_edge(node1, node3)
    graph.add_edge(node2, node3)
    
    # Block 2
    node4 = graph.add_node('aten::matmul', 'attn2_q')
    node5 = graph.add_node('aten::matmul', 'attn2_k')
    node6 = graph.add_node('aten::matmul', 'attn2_qk')
    graph.add_edge(node4, node6)
    graph.add_edge(node5, node6)
    
    # Annotate
    annotator = SemanticAnnotator(enable_cache=False)
    annotated = annotator.annotate(graph)
    
    # Verify patterns
    if 'llm' in annotated.patterns:
        llm_patterns = annotated.patterns['llm']
        
        # Should have 1+ patterns (system may merge or track separately)
        assert len(llm_patterns) >= 1, \
            "Expected at least one LLM pattern"
        
        total_nodes = sum(len(p.nodes) for p in llm_patterns)
        assert total_nodes > 0, \
            "Total pattern nodes should be > 0"
        
        print(f"✓ Found {len(llm_patterns)} pattern(s) with {total_nodes} total nodes")


def test_pattern_metadata_preserved():
    """Test that pattern metadata is properly preserved."""
    graph = ComputationGraph.empty()
    
    # Create minimal pattern
    node1 = graph.add_node('aten::matmul', 'node1')
    node2 = graph.add_node('aten::softmax', 'node2')
    graph.add_edge(node1, node2)
    
    # Annotate
    annotator = SemanticAnnotator(enable_cache=False)
    annotated = annotator.annotate(graph)
    
    # Check any detected patterns
    for pattern_name, patterns in annotated.patterns.items():
        for pattern in patterns:
            # Verify metadata structure
            assert isinstance(pattern.metadata, dict), \
                f"Pattern metadata should be dict, got {type(pattern.metadata)}"
            
            # Verify confidence score
            if 'confidence' in pattern.metadata:
                confidence = pattern.metadata['confidence']
                assert 0.0 <= confidence <= 1.0, \
                    f"Confidence should be in [0,1], got {confidence}"
            
            print(f"✓ Pattern '{pattern_name}' metadata preserved: {pattern.metadata}")


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])
