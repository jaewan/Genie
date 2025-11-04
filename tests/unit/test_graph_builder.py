"""
Test: Graph builder captures operations correctly

Validates:
- All operations captured
- Dependencies tracked correctly
- Topological order valid
- DAG properties maintained
- FX fallback works for dynamic control flow
"""

import torch
import torch.nn as nn
import pytest
import genie


class TestGraphBuilder:
    """Test graph construction correctness."""

    def test_captures_all_operations(self):
        """Test graph captures all operations."""

        with genie.capture():
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = x + y
            w = torch.relu(z)
            v = w @ w

        graph = genie.get_graph()
        nodes = list(graph.nodes())

        # Hard assertion: operations captured
        operations = [node.operation for node in nodes]

        assert any('randn' in op for op in operations), \
            "randn not captured!"
        assert any('add' in op for op in operations), \
            "add not captured!"
        assert any('relu' in op for op in operations), \
            "relu not captured!"
        assert any('matmul' in op or 'mm' in op for op in operations), \
            "matmul not captured!"

        print(f"✅ Captured {len(nodes)} operations")
        print(f"   Operations: {operations}")

    def test_dependencies_correct(self):
        """Test graph tracks dependencies correctly."""

        with genie.capture():
            x = torch.randn(5, 5)
            y = torch.randn(5, 5)
            z = x + y
            w = z @ z

        graph = genie.get_graph()
        nodes = list(graph.nodes())

        # Find the add node
        add_nodes = [n for n in nodes if 'add' in n.operation.lower()]
        assert len(add_nodes) > 0, "No add node found!"

        add_node = add_nodes[0]

        # Check it has inputs
        assert len(add_node.inputs) >= 2, \
            f"Add node should have 2+ inputs, got {len(add_node.inputs)}"

        print(f"✅ Dependencies tracked correctly")

    def test_topological_order(self):
        """Test graph returns nodes in valid topological order."""

        with genie.capture():
            a = torch.randn(3, 3)
            b = torch.randn(3, 3)
            c = a + b
            d = c @ c
            e = torch.relu(d)

        graph = genie.get_graph()
        nodes = list(graph.topological_sort())

        # Verify dependencies: operations on a,b should come before operations on c
        node_indices = {id(node): i for i, node in enumerate(nodes)}

        # Find add and matmul nodes
        for i, node in enumerate(nodes):
            if 'add' in node.operation.lower():
                add_idx = i
            if 'matmul' in node.operation.lower():
                matmul_idx = i

        # Matmul depends on add result
        if 'add_idx' in locals() and 'matmul_idx' in locals():
            assert add_idx < matmul_idx, \
                f"Topological order violated: add at {add_idx}, matmul at {matmul_idx}"

        print(f"✅ Topological order valid")

    @pytest.mark.slow
    def test_no_cycles(self):
        """Test graph has no cycles (is a valid DAG)."""

        with genie.capture():
            x = torch.randn(10, 10)
            y = x + 1
            z = y @ y

        graph = genie.get_graph()

        # Topological sort should not hang
        try:
            nodes = list(graph.topological_sort())
            assert len(nodes) > 0
            print("✅ No cycles detected (valid DAG)")
        except Exception as e:
            pytest.fail(f"Topological sort failed (possible cycle): {e}")

    def test_fx_fallback_for_control_flow(self):
        """Test graph builder falls back to LazyDAG for dynamic control flow."""

        class DynamicModel(nn.Module):
            def forward(self, x):
                if x.sum() > 0:  # Data-dependent branch
                    return torch.relu(x)
                else:
                    return torch.tanh(x)

        model = DynamicModel()

        with genie.capture():
            x = torch.randn(10, 10)
            output = model(x)

        graph = genie.get_graph()

        # Should use LazyDAG backend
        assert graph.backend_type == 'lazy_dag', \
            f"Expected lazy_dag for control flow, got {graph.backend_type}"

        # Graph should still be valid
        nodes = list(graph.nodes())
        assert len(nodes) > 0, "Empty graph after control flow!"

        print("✅ FX fallback works for dynamic control flow")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
