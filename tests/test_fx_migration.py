"""Test FX migration for Phase 1.2."""

import torch
import torch.nn as nn
import torch.fx as fx
import sys
import os

# Add parent directory to path to import genie
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genie.core.lazy_tensor import LazyTensor
from genie.core.fx_graph_builder import FXGraphBuilder, migrate_from_computation_graph
from genie.core.fx_executor import FXExecutor, OptimizingFXExecutor, execute_fx_graph
from genie.core.semantic_metadata import ExecutionPhase, MemoryPattern


def test_basic_fx_graph_building():
    """Test basic FX graph construction from LazyTensors."""
    print("\n=== Test Basic FX Graph Building ===")
    
    # Reset builder and counter
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    
    # Create LazyTensors
    a = LazyTensor("aten::randn", [[2, 3]], {"dtype": torch.float32})
    b = LazyTensor("aten::randn", [[3, 4]], {"dtype": torch.float32})
    c = LazyTensor("aten::matmul", [a, b])
    d = LazyTensor("aten::relu", [c])
    
    # Get FX builder
    fx_builder = FXGraphBuilder.current()
    
    # Mark output
    fx_builder.mark_output(d)
    
    # Convert to GraphModule
    gm = fx_builder.to_graph_module()
    
    # Verify graph structure
    nodes = list(gm.graph.nodes)
    print(f"Graph has {len(nodes)} nodes")
    
    # Check node types
    placeholders = [n for n in nodes if n.op == 'placeholder']
    call_functions = [n for n in nodes if n.op == 'call_function']
    outputs = [n for n in nodes if n.op == 'output']
    
    print(f"Placeholders: {len(placeholders)}")
    print(f"Function calls: {len(call_functions)}")
    print(f"Outputs: {len(outputs)}")
    
    # Verify operations
    assert len(call_functions) == 4  # 2 randn, 1 matmul, 1 relu
    assert len(outputs) == 1
    
    # Check semantic metadata preservation
    for node in call_functions:
        if 'semantic' in node.meta:
            metadata = node.meta['semantic']
            print(f"Node {node.name}: {metadata.operation_type}, intensity={metadata.compute_intensity}")
    
    print("✓ FX graph building working")


def test_semantic_metadata_in_fx():
    """Test that semantic metadata is preserved in FX nodes."""
    print("\n=== Test Semantic Metadata in FX ===")
    
    # Reset builder and counter
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    
    # Create operations with different semantic properties
    conv = LazyTensor("aten::conv2d", [torch.randn(1, 3, 32, 32)], {})
    pool = LazyTensor("aten::max_pool2d", [conv], {})
    relu = LazyTensor("aten::relu", [pool])
    
    # Get FX builder and convert
    fx_builder = FXGraphBuilder.current()
    fx_builder.mark_output(relu)
    gm = fx_builder.to_graph_module()
    
    # Check metadata preservation
    for node in gm.graph.nodes:
        if node.op == 'call_function' and 'semantic' in node.meta:
            metadata = node.meta['semantic']
            print(f"\nNode: {node.name}")
            print(f"  Operation: {metadata.operation_type}")
            print(f"  Memory pattern: {metadata.memory_pattern}")
            print(f"  Compute intensity: {metadata.compute_intensity}")
            print(f"  Can parallelize: {metadata.can_parallelize}")
            
            # Verify expected properties
            if 'conv2d' in str(node.target):
                assert metadata.memory_pattern == MemoryPattern.STREAMING
                assert metadata.compute_intensity == 10.0
    
    print("✓ Semantic metadata preserved in FX graph")


def test_fx_executor():
    """Test FX executor with semantic tracking."""
    print("\n=== Test FX Executor ===")
    
    # Reset builder and counter
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    
    # Create simple computation
    a = LazyTensor("aten::ones", [[2, 3]], {})
    b = LazyTensor("aten::ones", [[2, 3]], {})
    c = LazyTensor("aten::add", [a, b])
    d = LazyTensor("aten::mul", [c, 2.0])
    
    # Build FX graph
    fx_builder = FXGraphBuilder.current()
    fx_builder.mark_output(d)
    
    # Execute with tracking
    result, summary = execute_fx_graph(fx_builder, optimize=False)
    
    print(f"\nExecution summary:")
    print(f"  Total operations: {summary['total_operations']}")
    print(f"  Operations executed: {summary['semantic_stats']['operations_count']}")
    print(f"  Memory patterns: {summary['semantic_stats']['memory_patterns']}")
    print(f"  Total compute intensity: {summary['semantic_stats']['compute_intensity_total']}")
    
    # Verify result
    assert isinstance(result, torch.Tensor)
    expected = torch.ones(2, 3) * 4.0  # (1 + 1) * 2
    assert torch.allclose(result, expected)
    
    print("✓ FX executor working correctly")


def test_optimizing_executor():
    """Test optimizing FX executor with semantic optimizations."""
    print("\n=== Test Optimizing Executor ===")
    
    # Reset builder and counter
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    
    # Create computation with optimization opportunities
    # Simulate attention-like pattern
    q = LazyTensor("aten::randn", [[2, 8, 64]], {})
    k = LazyTensor("aten::randn", [[2, 8, 64]], {})
    v = LazyTensor("aten::randn", [[2, 8, 64]], {})
    
    # Attention scores (need to transpose k)
    k_t = LazyTensor("aten::transpose", [k], {"dim0": -2, "dim1": -1})
    scores = LazyTensor("aten::matmul", [q, k_t])
    weights = LazyTensor("aten::softmax", [scores], {"dim": -1})
    output = LazyTensor("aten::matmul", [weights, v])
    
    # Build and execute with optimizations
    fx_builder = FXGraphBuilder.current()
    fx_builder.mark_output(output)
    
    result, summary = execute_fx_graph(fx_builder, optimize=True)
    
    print(f"\nOptimization summary:")
    print(f"  Total operations: {summary['total_operations']}")
    print(f"  Cached operations: {summary['optimization_stats']['cached_operations']}")
    print(f"  Fused operations: {summary['optimization_stats']['fused_operations']}")
    print(f"  Parallelized operations: {summary['optimization_stats']['parallelized_operations']}")
    
    assert isinstance(result, torch.Tensor)
    print("✓ Optimizing executor working")


def test_execution_order():
    """Test that execution order is topologically correct."""
    print("\n=== Test Execution Order ===")
    
    # Reset builder and counter
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    
    # Create DAG of operations
    a = LazyTensor("aten::randn", [[2, 3]], {})
    b = LazyTensor("aten::randn", [[2, 3]], {})
    c = LazyTensor("aten::add", [a, b])  # c depends on a, b
    d = LazyTensor("aten::mul", [a, 2.0])  # d depends on a
    e = LazyTensor("aten::sub", [c, d])  # e depends on c, d
    
    # Get execution order
    fx_builder = FXGraphBuilder.current()
    fx_builder.mark_output(e)
    order = fx_builder.get_execution_order()
    
    print(f"Execution order: {order}")
    
    # Verify dependencies are respected
    a_idx = order.index(a.id)
    b_idx = order.index(b.id)
    c_idx = order.index(c.id)
    d_idx = order.index(d.id)
    e_idx = order.index(e.id)
    
    assert a_idx < c_idx  # a before c
    assert b_idx < c_idx  # b before c
    assert a_idx < d_idx  # a before d
    assert c_idx < e_idx  # c before e
    assert d_idx < e_idx  # d before e
    
    print("✓ Execution order is topologically correct")


def test_graph_summary():
    """Test semantic summary generation."""
    print("\n=== Test Graph Summary ===")
    
    # Reset builder and counter
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    
    # Create diverse operations
    conv1 = LazyTensor("aten::conv2d", [torch.randn(1, 3, 32, 32)], {})
    relu1 = LazyTensor("aten::relu", [conv1])
    pool1 = LazyTensor("aten::max_pool2d", [relu1], {})
    
    conv2 = LazyTensor("aten::conv2d", [pool1], {})
    relu2 = LazyTensor("aten::relu", [conv2])
    
    linear = LazyTensor("aten::linear", [relu2], {})
    softmax = LazyTensor("aten::softmax", [linear], {"dim": -1})
    
    # Get summary
    fx_builder = FXGraphBuilder.current()
    fx_builder.mark_output(softmax)
    summary = fx_builder.get_semantic_summary()
    
    print(f"\nGraph summary:")
    print(f"  Total nodes: {summary['num_nodes']}")
    print(f"  Operations: {summary['operations']}")
    print(f"  Memory patterns: {summary['memory_patterns']}")
    print(f"  High intensity ops: {len(summary['compute_intensity']['high'])}")
    print(f"  Medium intensity ops: {len(summary['compute_intensity']['medium'])}")
    print(f"  Low intensity ops: {len(summary['compute_intensity']['low'])}")
    
    # Verify summary contents
    assert summary['num_nodes'] > 0
    assert len(summary['operations']) > 0
    assert len(summary['compute_intensity']['high']) > 0  # conv2d should be high
    
    print("✓ Graph summary generation working")


def test_migration_from_old_graph():
    """Test migration from old ComputationGraph to FX."""
    print("\n=== Test Migration from Old Graph ===")
    
    # Import old graph system
    from genie.core.graph import ComputationGraph, ComputationNode
    
    # Create a fresh computation graph
    old_graph = ComputationGraph(nodes={}, edges=[], entry_points=set())
    
    # Manually create nodes (simulating old system)
    node1 = ComputationNode(
        id="node1",
        operation="aten::randn",
        inputs=[],
        outputs=["node1_out"],
        metadata={"size": [2, 3]}
    )
    node2 = ComputationNode(
        id="node2",
        operation="aten::relu",
        inputs=["node1"],
        outputs=["node2_out"],
        metadata={}
    )
    
    # Add nodes and edges directly
    old_graph.nodes["node1"] = node1
    old_graph.nodes["node2"] = node2
    old_graph.edges.append(("node1", "node2"))
    old_graph.entry_points.add("node1")
    
    # Reset FX builder for new LazyTensors
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    
    # Create LazyTensors for mapping
    lt1 = LazyTensor("aten::randn", [[2, 3]], {})
    lt2 = LazyTensor("aten::relu", [lt1], {})
    
    # Add lazy_tensor_map as an attribute (not part of original dataclass)
    setattr(old_graph, 'lazy_tensor_map', {
        "node1": lt1,
        "node2": lt2
    })
    
    # Migrate to FX
    fx_builder = migrate_from_computation_graph(old_graph)
    
    # Verify migration
    gm = fx_builder.to_graph_module()
    nodes = list(gm.graph.nodes)
    
    print(f"Migrated {len(nodes)} nodes to FX graph")
    
    # Check that operations are preserved
    call_nodes = [n for n in nodes if n.op == 'call_function']
    assert len(call_nodes) >= 2  # At least randn and relu
    
    print("✓ Migration from old graph working")


def test_complex_model_fx():
    """Test FX with a more complex model structure."""
    print("\n=== Test Complex Model with FX ===")
    
    # Reset builder and counter
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    
    # Simulate transformer block
    # Input
    x = LazyTensor("aten::randn", [[1, 10, 256]], {})  # [batch, seq_len, hidden]
    
    # Multi-head attention
    q = LazyTensor("aten::linear", [x], {})
    k = LazyTensor("aten::linear", [x], {})
    v = LazyTensor("aten::linear", [x], {})
    
    # Reshape for heads
    q_heads = LazyTensor("aten::reshape", [q], {"shape": [1, 10, 8, 32]})
    k_heads = LazyTensor("aten::reshape", [k], {"shape": [1, 10, 8, 32]})
    v_heads = LazyTensor("aten::reshape", [v], {"shape": [1, 10, 8, 32]})
    
    # Attention computation
    scores = LazyTensor("aten::matmul", [q_heads, k_heads])
    attn_weights = LazyTensor("aten::softmax", [scores], {"dim": -1})
    attn_output = LazyTensor("aten::matmul", [attn_weights, v_heads])
    
    # Reshape back
    attn_flat = LazyTensor("aten::reshape", [attn_output], {"shape": [1, 10, 256]})
    
    # Output projection
    output = LazyTensor("aten::linear", [attn_flat], {})
    
    # Residual connection
    residual = LazyTensor("aten::add", [x, output])
    
    # Layer norm
    final = LazyTensor("aten::layer_norm", [residual], {"normalized_shape": [256]})
    
    # Build and verify
    fx_builder = FXGraphBuilder.current()
    fx_builder.mark_output(final)
    
    gm = fx_builder.to_graph_module()
    summary = fx_builder.get_semantic_summary()
    
    print(f"\nTransformer block graph:")
    print(f"  Total operations: {summary['num_nodes']}")
    print(f"  Linear layers: {summary['operations'].get('linear', 0)}")
    print(f"  MatMuls: {summary['operations'].get('matmul', 0)}")
    print(f"  Reshapes: {summary['operations'].get('reshape', 0)}")
    
    # Verify structure
    assert summary['operations'].get('linear', 0) >= 4  # Q, K, V, Output projections
    assert summary['operations'].get('matmul', 0) >= 2  # Attention scores and output
    assert summary['operations'].get('softmax', 0) >= 1
    
    print("✓ Complex model FX graph building working")


def test_multiple_outputs():
    """Test FX graph with multiple outputs."""
    print("\n=== Test Multiple Outputs ===")
    
    # Reset builder and counter
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    
    # Create graph with multiple outputs
    a = LazyTensor("aten::randn", [[2, 3]], {})
    b = LazyTensor("aten::randn", [[2, 3]], {})
    
    sum_out = LazyTensor("aten::add", [a, b])
    prod_out = LazyTensor("aten::mul", [a, b])
    
    # Mark both as outputs
    fx_builder = FXGraphBuilder.current()
    fx_builder.mark_output(sum_out)
    fx_builder.mark_output(prod_out)
    
    # Execute
    results, summary = execute_fx_graph(fx_builder)
    
    print(f"Executed graph with {len(fx_builder.outputs)} outputs")
    
    # Verify we got tuple of results
    assert isinstance(results, tuple)
    assert len(results) == 2
    
    print("✓ Multiple outputs handling working")


def run_all_tests():
    """Run all FX migration tests."""
    print("=" * 60)
    print("Testing FX Migration (Phase 1.2)")
    print("=" * 60)
    
    test_basic_fx_graph_building()
    test_semantic_metadata_in_fx()
    test_fx_executor()
    test_optimizing_executor()
    test_execution_order()
    test_graph_summary()
    test_migration_from_old_graph()
    test_complex_model_fx()
    test_multiple_outputs()
    
    print("\n" + "=" * 60)
    print("✅ All Phase 1.2 tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
