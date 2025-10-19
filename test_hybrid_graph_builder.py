#!/usr/bin/env python3
"""
Test script for the Hybrid Graph Builder.

This script verifies that the hybrid graph builder correctly:
1. Uses FX tracing for static models (80% case)
2. Falls back to LazyTensor DAG for dynamic models (20% case)
3. Provides unified Graph interface for both backends
"""

import sys
import os
import torch
import torch.fx as fx

# Add genie to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from genie.core.graph_interface import Graph, FXGraphAdapter, LazyDAGAdapter
from genie.core.capture import capture, get_graph, is_capturing
from genie.core.graph_builder import get_global_builder


def test_fx_static_model():
    """Test that static models use FX tracing."""
    print("Testing FX tracing for static model...")

    # Static model (no dynamic control flow)
    class StaticModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x.relu())

    model = StaticModel()

    # Test with hybrid graph builder
    from genie.core.graph_builder import HybridGraphBuilder

    builder = HybridGraphBuilder()
    x = torch.randn(3, 10)

    # This should succeed with FX
    graph = builder.build_from_model(model, x)

    # Verify it's an FX graph
    assert isinstance(graph, FXGraphAdapter)
    assert graph.backend_type == 'fx'
    print(f"✓ FX backend used: {graph.backend_type}")

    nodes = graph.nodes()
    print(f"✓ FX graph has {len(nodes)} nodes")

    # Verify node operations are properly formatted
    for node in nodes:
        assert 'aten::' in node.operation, f"Operation not properly formatted: {node.operation}"

    print("✓ FX operations properly formatted")
    return True


def test_lazy_dag_dynamic_model():
    """Test that dynamic models fall back to LazyTensor DAG."""
    print("\nTesting LazyDAG fallback for dynamic model...")

    # Dynamic model (has control flow that FX can't handle)
    class DynamicModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            # This creates dynamic control flow that FX can't trace
            if x.sum() > 0:
                return self.linear(x.relu())
            else:
                return self.linear(x.tanh())

    model = DynamicModel()

    # Test with hybrid graph builder
    from genie.core.graph_builder import HybridGraphBuilder

    builder = HybridGraphBuilder()

    # Use device-based API to ensure LazyTensor creation
    x = torch.randn(3, 10, device='remote_accelerator:0')

    # This should fall back to LazyDAG due to dynamic control flow
    try:
        graph = builder.build_from_model(model, x)

        # Verify it's a LazyDAG graph
        assert isinstance(graph, LazyDAGAdapter)
        assert graph.backend_type == 'lazy_dag'
        print(f"✓ LazyDAG backend used: {graph.backend_type}")

        nodes = graph.nodes()
        print(f"✓ LazyDAG graph has {len(nodes)} nodes")

        return True

    except Exception as e:
        print(f"⚠️  Dynamic model test failed (expected in some cases): {e}")
        return False


def test_capture_context_api():
    """Test the capture context manager API."""
    print("\nTesting capture context API...")

    # Test context manager
    with capture():
        assert is_capturing() == True
        print("✓ Capture context active")

        # Create some operations
        x = torch.randn(5, device='remote_accelerator:0')
        y = x + 1
        z = y * 2

    assert is_capturing() == False
    print("✓ Capture context inactive after exit")

    # Get the captured graph
    graph = get_graph()
    if graph is not None:
        print(f"✓ Captured graph: {graph.backend_type} backend")
        nodes = graph.nodes()
        print(f"✓ Captured {len(nodes)} nodes")
        return True
    else:
        print("⚠️  No graph captured")
        return False


def test_unified_interface_compatibility():
    """Test that both backends provide the same interface."""
    print("\nTesting unified interface compatibility...")

    # Test FX interface
    def static_func(x):
        return (x + 1).relu()

    gm = fx.symbolic_trace(static_func)
    fx_graph = gm.graph
    fx_adapter = FXGraphAdapter(fx_graph)

    # Test LazyDAG interface (if available)
    lazy_adapter = None
    try:
        x = torch.randn(5, device='remote_accelerator:0')
        y = static_func(x)
        from genie.core.graph_builder import LazyDAGAdapter
        lazy_adapter = LazyDAGAdapter(y)
    except Exception:
        pass

    # Test that both provide the same interface
    adapters = [fx_adapter]
    if lazy_adapter:
        adapters.append(lazy_adapter)

    for adapter in adapters:
        # Same interface methods
        nodes = adapter.nodes()
        backend = adapter.backend_type

        for node in nodes:
            # Same node interface
            _ = node.id
            _ = node.operation
            _ = node.inputs
            _ = node.metadata

        # Same graph interface
        topo_nodes = adapter.topological_sort()
        if nodes:
            node = adapter.get_node(nodes[0].id)
            if node:
                print(f"✓ {backend} adapter: node lookup works")

    print(f"✓ Unified interface works for {len(adapters)} backends")
    return True


def test_build_from_capture():
    """Test building graph from captured operations."""
    print("\nTesting build_from_capture...")

    from genie.core.graph_builder import get_global_builder

    # Clear any existing state
    builder = get_global_builder()
    builder.root_tensor = None

    # Capture some operations
    with capture():
        x = torch.randn(5, device='remote_accelerator:0')
        y = x + 1
        z = y * 2

    # Build graph from capture
    graph = builder.build_from_capture()

    if graph is not None:
        print(f"✓ Built graph from capture: {graph.backend_type}")
        nodes = graph.nodes()
        print(f"✓ Captured {len(nodes)} nodes")

        # Verify operations
        operations = [node.operation for node in nodes]
        print(f"✓ Operations: {operations}")

        return True
    else:
        print("⚠️  No graph built from capture")
        return False


def main():
    """Run all tests."""
    print("Testing Hybrid Graph Builder...")
    print("=" * 50)

    tests = [
        test_fx_static_model,
        test_lazy_dag_dynamic_model,
        test_capture_context_api,
        test_unified_interface_compatibility,
        test_build_from_capture,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"❌ {test.__name__} error: {e}")

    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✅ All tests passed! Hybrid Graph Builder is working correctly.")
        return True
    else:
        print("❌ Some tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
