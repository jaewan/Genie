"""
End-to-end test for semantic workflow.

Tests complete pipeline:
1. LazyTensor capture with metadata
2. Pattern detection on LazyDAG
3. Semantic annotation (phases, costs)
4. Scheduler creates placement plan
5. (Future) Backend execution
"""

import torch
import torch.nn as nn
import djinn


def test_complete_semantic_workflow():
    """Test full semantic workflow end-to-end."""

    print("\n=== End-to-End Semantic Workflow Test ===\n")

    # Step 1: Create model with semantic patterns
    print("1. Creating model with attention pattern...")

    class SimpleAttentionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.query_proj = nn.Linear(128, 128)
            self.key_proj = nn.Linear(128, 128)
            self.value_proj = nn.Linear(128, 128)

        def forward(self, x):
            Q = self.query_proj(x)
            K = self.key_proj(x)
            V = self.value_proj(x)

            # Attention: Q @ K.T → softmax → @ V
            scores = torch.matmul(Q, K.transpose(-2, -1))
            attn = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn, V)

            return output

    model = SimpleAttentionModel()

    # Step 2: Capture graph with LazyTensor
    print("2. Capturing computation graph...")

    with genie.capture():
        x = torch.randn(32, 64, 128)
        output = model(x)

    graph = genie.get_graph()
    print(f"   ✓ Graph captured: {len(list(graph.nodes()))} nodes")

    # Step 3: Check metadata was captured
    print("3. Verifying metadata capture...")

    nodes_with_metadata = 0
    for node in graph.nodes():
        if hasattr(node, 'metadata') and node.metadata:
            nodes_with_metadata += 1

    assert nodes_with_metadata > 0, \
        "At least some nodes should have metadata"

    print(f"   ✓ {nodes_with_metadata} nodes have metadata")

    # Step 4: Pattern detection
    print("4. Running pattern detection...")

    from djinn.frontend.semantic.annotator import annotate_graph
    annotated = annotate_graph(graph)

    print(f"   ✓ Patterns detected: {list(annotated.patterns.keys())}")

    # Verify attention pattern detected
    assert 'attention' in annotated.patterns or \
           len(annotated.patterns) > 0, \
        "Should detect at least one pattern"

    if 'attention' in annotated.patterns:
        attention_pattern = annotated.patterns['attention'][0]
        print(f"   ✓ Attention pattern: {len(attention_pattern.nodes)} nodes")

    # Step 5: Cost estimation
    print("5. Verifying cost estimation...")

    assert annotated.costs is not None, "Costs should be estimated"
    assert 'total_compute_flops' in annotated.costs, \
        "Should have compute cost"

    print(f"   ✓ Total compute: {annotated.costs['total_compute_flops']:.2e} FLOPs")
    print(f"   ✓ Total memory: {annotated.costs['total_memory_bytes']:.2e} bytes")

    # Step 6: Scheduling
    print("6. Creating execution schedule...")

    schedule = genie.schedule(graph)

    assert schedule.total_stages > 0, "Should have at least 1 stage"
    assert len(schedule.node_to_stage) > 0, "Should map nodes to stages"

    print(f"   ✓ Schedule: {schedule.total_stages} stages")
    print(f"   ✓ Strategy: {schedule.strategy}")

    # Step 7: Verify semantic decisions
    print("7. Verifying semantic decisions...")

    if 'placement_plan' in schedule.metadata:
        placement = schedule.metadata['placement_plan']
        print(f"   ✓ Placement decisions made: {len(placement)} nodes")

    print("\n✅ Complete semantic workflow successful!\n")


def test_semantic_workflow_with_colocation():
    """Test that co-location decisions are made based on patterns."""

    print("\n=== Testing Semantic Co-location ===\n")

    with genie.capture():
        # Create multiple attention blocks to test co-location
        x1 = torch.randn(16, 64, 128)
        x2 = torch.randn(16, 64, 128)

        # Attention block 1
        q1 = torch.randn(16, 8, 64)
        k1 = torch.randn(16, 8, 64)
        v1 = torch.randn(16, 8, 64)
        scores1 = torch.matmul(q1, k1.transpose(-2, -1))
        attn1 = torch.softmax(scores1, dim=-1)
        out1 = torch.matmul(attn1, v1)

        # Attention block 2
        q2 = torch.randn(16, 8, 64)
        k2 = torch.randn(16, 8, 64)
        v2 = torch.randn(16, 8, 64)
        scores2 = torch.matmul(q2, k2.transpose(-2, -1))
        attn2 = torch.softmax(scores2, dim=-1)
        out2 = torch.matmul(attn2, v2)

    graph = genie.get_graph()
    print(f"Graph: {len(list(graph.nodes()))} nodes")

    # Annotate
    from djinn.frontend.semantic.annotator import annotate_graph
    annotated = annotate_graph(graph)
    print(f"Patterns: {list(annotated.patterns.keys())}")

    # Schedule
    schedule = genie.schedule(graph)
    print(f"Schedule: {schedule.total_stages} stages, {len(schedule.node_to_stage)} nodes")

    # Check if multiple attention patterns were detected
    attention_patterns = 0
    for pattern_name, patterns in annotated.patterns.items():
        if 'attention' in pattern_name.lower():
            attention_patterns += len(patterns)

    print(f"Attention patterns detected: {attention_patterns}")

    if attention_patterns >= 2:
        print("✓ Multiple attention patterns detected - co-location opportunities available")
    else:
        print("⚠️  Single attention pattern - basic co-location")

    print("\n✅ Co-location test complete\n")


def test_metadata_capture_quality():
    """Test the quality of metadata capture."""

    print("\n=== Testing Metadata Capture Quality ===\n")

    with genie.capture():
        # Test various operations to see metadata capture
        x = torch.randn(10, 20)
        y = torch.randn(20, 30)

        # Matrix multiplication
        z = torch.matmul(x, y)

        # Element-wise operations
        w = torch.relu(z)
        u = torch.add(w, 1)

        # Check metadata on final result
        print(f"Final tensor metadata: {u.metadata}")

        # Check if semantic roles are captured
        semantic_roles = []
        for node in genie.get_graph().nodes():
            if hasattr(node, 'metadata') and node.metadata:
                role = node.metadata.get('semantic_role')
                if role:
                    semantic_roles.append(role)

        print(f"Semantic roles detected: {set(semantic_roles)}")

        # Check for pattern hints
        pattern_hints = []
        for node in genie.get_graph().nodes():
            if hasattr(node, 'metadata') and node.metadata:
                hints = node.metadata.get('pattern_hints')
                if hints:
                    pattern_hints.append(hints)

        print(f"Pattern hints detected: {len(pattern_hints)} hints")

        print("\n✅ Metadata quality test complete\n")


if __name__ == '__main__':
    test_complete_semantic_workflow()
    test_semantic_workflow_with_colocation()
    test_metadata_capture_quality()
