"""
Quick demonstration of Genie's key features with minimal setup.

This example shows the core capabilities in a lightweight format:
- Semantic metadata capture
- Phase detection
- Pattern matching
- Optimization planning
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genie.core.lazy_tensor import LazyTensor
from genie.core.fx_graph_builder import FXGraphBuilder
from genie.semantic.enhanced_analyzer import EnhancedSemanticAnalyzer
from genie.semantic.optimizer import SemanticOptimizer
from genie.semantic.phase_detector import get_phase_detector
from genie.core.semantic_metadata import ExecutionPhase


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_llm_attention():
    """Demonstrate LLM attention pattern detection and optimization."""
    print_header("LLM Attention Pattern Demo")
    
    # Reset state
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    phase_detector = get_phase_detector()
    phase_detector.reset()
    
    print("\n1. Building attention computation graph...")
    
    # Simulate multi-head attention
    batch_size, seq_len, hidden_dim = 2, 10, 256
    num_heads = 8
    
    # Input tensor
    x = LazyTensor("aten::randn", [[batch_size, seq_len, hidden_dim]], {})
    
    # QKV projections
    q = LazyTensor("aten::linear", [x], {})
    k = LazyTensor("aten::linear", [x], {})
    v = LazyTensor("aten::linear", [x], {})
    
    # Scaled dot-product attention
    scale = LazyTensor("aten::scalar", [[1.0 / (hidden_dim ** 0.5)]], {})
    k_t = LazyTensor("aten::transpose", [k], {"dim0": -2, "dim1": -1})
    scores = LazyTensor("aten::matmul", [q, k_t], {})
    scores_scaled = LazyTensor("aten::mul", [scores, scale], {})
    weights = LazyTensor("aten::softmax", [scores_scaled], {"dim": -1})
    attn_output = LazyTensor("aten::matmul", [weights, v], {})
    
    # Output projection
    output = LazyTensor("aten::linear", [attn_output], {})
    
    print(f"✓ Created {8} lazy operations")
    
    # Check metadata
    print("\n2. Semantic metadata captured:")
    if attn_output.metadata:
        print(f"   - Operation type: {attn_output.metadata.operation_type}")
        print(f"   - Execution phase: {attn_output.metadata.execution_phase}")
        print(f"   - Memory pattern: {attn_output.metadata.memory_pattern}")
        print(f"   - Can parallelize: {attn_output.metadata.can_parallelize}")
    
    # Build FX graph
    fx_builder = FXGraphBuilder.current()
    fx_builder.mark_output(output)
    fx_graph = fx_builder.to_graph_module()
    
    print(f"\n3. FX graph built with {len(list(fx_graph.graph.nodes))} nodes")
    
    # Analyze and optimize
    analyzer = EnhancedSemanticAnalyzer(use_dynamo=False)
    profile = analyzer.analyze_fx_graph(fx_graph)
    
    print(f"\n4. Workload detected: {profile.workload_type}")
    
    optimizer = SemanticOptimizer()
    optimized_graph, opt_plan = optimizer.optimize(fx_graph, profile)
    
    print(f"\n5. Optimizations applied:")
    for opt in opt_plan.optimizations:
        print(f"   - {opt.value}")
    
    # Check phase detection
    phase_stats = phase_detector.get_phase_statistics()
    print(f"\n6. Phase detection:")
    print(f"   - Current phase: {phase_stats['current_phase']}")
    print(f"   - Token position: {phase_stats['token_position']}")
    
    return optimized_graph, opt_plan


def demo_cnn_pipeline():
    """Demonstrate CNN pipeline optimization."""
    print_header("CNN Pipeline Demo")
    
    # Reset state
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    phase_detector = get_phase_detector()
    phase_detector.reset()
    
    print("\n1. Building CNN pipeline...")
    
    batch_size = 4
    
    # Input image
    x = LazyTensor("aten::randn", [[batch_size, 3, 32, 32]], {})
    
    # Conv block 1
    conv1 = LazyTensor("aten::conv2d", [x], {"stride": 1, "padding": 1})
    relu1 = LazyTensor("aten::relu", [conv1], {})
    pool1 = LazyTensor("aten::max_pool2d", [relu1], {"kernel_size": 2})
    
    # Conv block 2
    conv2 = LazyTensor("aten::conv2d", [pool1], {"stride": 1, "padding": 1})
    relu2 = LazyTensor("aten::relu", [conv2], {})
    pool2 = LazyTensor("aten::max_pool2d", [relu2], {"kernel_size": 2})
    
    # Conv block 3
    conv3 = LazyTensor("aten::conv2d", [pool2], {"stride": 1, "padding": 1})
    relu3 = LazyTensor("aten::relu", [conv3], {})
    
    # Global pooling
    gap = LazyTensor("aten::adaptive_avg_pool2d", [relu3], {"output_size": [1, 1]})
    flat = LazyTensor("aten::flatten", [gap], {"start_dim": 1})
    output = LazyTensor("aten::linear", [flat], {})
    
    print(f"✓ Created 3-stage CNN pipeline")
    
    # Check phase detection for conv operations
    print("\n2. Vision phase detection:")
    if conv1.metadata:
        print(f"   - Conv1 phase: {conv1.metadata.execution_phase}")
    if conv3.metadata:
        print(f"   - Conv3 phase: {conv3.metadata.execution_phase}")
    if gap.metadata:
        print(f"   - GAP phase: {gap.metadata.execution_phase}")
    
    # Build and analyze
    fx_builder = FXGraphBuilder.current()
    fx_builder.mark_output(output)
    fx_graph = fx_builder.to_graph_module()
    
    analyzer = EnhancedSemanticAnalyzer(use_dynamo=False)
    profile = analyzer.analyze_fx_graph(fx_graph)
    
    print(f"\n3. Workload detected: {profile.workload_type}")
    
    # Apply pipeline optimization
    from genie.semantic.scheduling import PipelineScheduler
    
    pipeline_scheduler = PipelineScheduler(num_stages=3)
    pipeline_schedule = pipeline_scheduler.create_pipeline_schedule(fx_graph)
    
    print(f"\n4. Pipeline scheduling:")
    print(f"   - Total stages: {pipeline_schedule.total_stages}")
    print(f"   - Pipeline depth: {pipeline_schedule.metadata.get('pipeline_depth', 0)}")
    
    # Check vision stages
    phase_stats = phase_detector.get_phase_statistics()
    print(f"\n5. Vision tracking:")
    print(f"   - Vision stages processed: {phase_stats['vision_stages']}")
    print(f"   - Current phase: {phase_stats['current_phase']}")
    
    return fx_graph, pipeline_schedule


def demo_kv_cache_detection():
    """Demonstrate KV cache detection for LLMs."""
    print_header("KV Cache Detection Demo")
    
    # Reset state
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    phase_detector = get_phase_detector()
    phase_detector.reset()
    
    print("\n1. Simulating KV cache operations...")
    
    batch_size, hidden_dim = 2, 256
    
    # Existing KV cache (small - indicates few tokens cached)
    k_cache = LazyTensor("aten::zeros", [[batch_size, 5, hidden_dim]], {})
    v_cache = LazyTensor("aten::zeros", [[batch_size, 5, hidden_dim]], {})
    
    # New keys and values (prefill - multiple tokens)
    new_seq_len = 10
    new_k = LazyTensor("aten::randn", [[batch_size, new_seq_len, hidden_dim]], {})
    new_v = LazyTensor("aten::randn", [[batch_size, new_seq_len, hidden_dim]], {})
    
    # Concatenate to update cache
    k_updated = LazyTensor("aten::cat", [k_cache, new_k], {"dim": 1})
    v_updated = LazyTensor("aten::cat", [v_cache, new_v], {"dim": 1})
    
    print(f"✓ KV cache update: 5 cached + {new_seq_len} new tokens")
    
    # Check metadata
    print("\n2. KV cache metadata:")
    if k_updated.metadata:
        print(f"   - KV cache related: {k_updated.metadata.kv_cache_related}")
        print(f"   - Execution phase: {k_updated.metadata.execution_phase}")
        print(f"   - Priority: {k_updated.metadata.priority}")
    
    # Now simulate decode (single token)
    print("\n3. Simulating decode phase...")
    
    phase_detector.mark_batch_boundary()
    
    single_k = LazyTensor("aten::randn", [[batch_size, 1, hidden_dim]], {})
    single_v = LazyTensor("aten::randn", [[batch_size, 1, hidden_dim]], {})
    
    k_decode = LazyTensor("aten::cat", [k_updated, single_k], {"dim": 1})
    v_decode = LazyTensor("aten::cat", [v_updated, single_v], {"dim": 1})
    
    print("✓ Single token decode update")
    
    # Check phase transition
    phase_stats = phase_detector.get_phase_statistics()
    print(f"\n4. Phase transitions:")
    print(f"   - Total transitions: {phase_stats['transitions']}")
    print(f"   - Current phase: {phase_stats['current_phase']}")
    print(f"   - Token position: {phase_stats['token_position']}")
    
    # Build graph for optimization
    fx_builder = FXGraphBuilder.current()
    fx_builder.mark_output(v_decode)
    fx_graph = fx_builder.to_graph_module()
    
    # Optimize with KV cache awareness
    analyzer = EnhancedSemanticAnalyzer(use_dynamo=False)
    profile = analyzer.analyze_fx_graph(fx_graph)
    
    optimizer = SemanticOptimizer()
    _, opt_plan = optimizer.optimize(fx_graph, profile)
    
    print(f"\n5. KV cache optimizations:")
    if 'kv_cache' in opt_plan.colocation_groups:
        print(f"   - KV cache operations co-located: {len(opt_plan.colocation_groups['kv_cache'])}")
    else:
        print("   - No specific KV cache colocation detected")
    
    return fx_graph, opt_plan


def demo_multi_modal_fusion():
    """Demonstrate multi-modal fusion detection."""
    print_header("Multi-Modal Fusion Demo")
    
    # Reset state
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    phase_detector = get_phase_detector()
    phase_detector.reset()
    
    print("\n1. Building multi-modal branches...")
    
    batch_size = 2
    
    # Vision branch
    image = LazyTensor("aten::randn", [[batch_size, 3, 32, 32]], {})
    conv = LazyTensor("aten::conv2d", [image], {"stride": 1})
    pool = LazyTensor("aten::adaptive_avg_pool2d", [conv], {"output_size": [1, 1]})
    vision_feat = LazyTensor("aten::flatten", [pool], {"start_dim": 1})
    
    print("✓ Vision branch created")
    
    # Text branch
    text = LazyTensor("aten::randn", [[batch_size, 10, 128]], {})
    text_feat = LazyTensor("aten::mean", [text], {"dim": 1})
    
    print("✓ Text branch created")
    
    # Cross-modal fusion
    fused = LazyTensor("aten::cat", [vision_feat, text_feat], {"dim": -1})
    output = LazyTensor("aten::linear", [fused], {})
    
    print("✓ Cross-modal fusion created")
    
    # Check modality detection
    phase_stats = phase_detector.get_phase_statistics()
    print(f"\n2. Modality detection:")
    print(f"   - Detected modalities: {phase_stats['detected_modalities']}")
    
    # Check fusion metadata
    print("\n3. Fusion point metadata:")
    if fused.metadata:
        print(f"   - Operation: {fused.metadata.operation_type}")
        print(f"   - Execution phase: {fused.metadata.execution_phase}")
    
    # Build and analyze
    fx_builder = FXGraphBuilder.current()
    fx_builder.mark_output(output)
    fx_graph = fx_builder.to_graph_module()
    
    analyzer = EnhancedSemanticAnalyzer(use_dynamo=False)
    profile = analyzer.analyze_fx_graph(fx_graph)
    
    print(f"\n4. Workload detected: {profile.workload_type}")
    
    # Check for fusion optimizations
    optimizer = SemanticOptimizer()
    _, opt_plan = optimizer.optimize(fx_graph, profile)
    
    print(f"\n5. Fusion optimizations:")
    print(f"   - Fusion points identified: {len(opt_plan.fusion_points)}")
    for opt in opt_plan.optimizations:
        if 'fusion' in opt.value.lower():
            print(f"   - {opt.value} applied")
    
    return fx_graph, opt_plan


def main():
    """Run all quick demos."""
    print("=" * 60)
    print("  GENIE QUICK DEMONSTRATION")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    
    try:
        # Run demos
        attention_graph, attention_plan = demo_llm_attention()
        cnn_graph, pipeline_schedule = demo_cnn_pipeline()
        kv_graph, kv_plan = demo_kv_cache_detection()
        fusion_graph, fusion_plan = demo_multi_modal_fusion()
        
        # Summary
        print_header("DEMONSTRATION SUMMARY")
        
        print("\n✅ Successfully demonstrated:")
        print("   1. LLM attention pattern detection and optimization")
        print("   2. CNN pipeline scheduling (3 stages)")
        print("   3. KV cache detection for prefill/decode phases")
        print("   4. Multi-modal fusion detection and optimization")
        
        print("\n📊 Features showcased:")
        print("   - Enhanced semantic metadata capture")
        print("   - FX graph generation and analysis")
        print("   - Execution phase detection")
        print("   - Workload-specific optimizations")
        print("   - Pipeline and parallel scheduling")
        
        phase_detector = get_phase_detector()
        final_stats = phase_detector.get_phase_statistics()
        
        print(f"\n📈 Statistics:")
        print(f"   - Total operations tracked: {sum(final_stats['operation_counts'].values())}")
        print(f"   - Phase transitions: {final_stats['transitions']}")
        print(f"   - Vision stages: {final_stats['vision_stages']}")
        print(f"   - Modalities detected: {final_stats['detected_modalities']}")
        
        print("\n✨ Genie is ready for semantic-aware ML optimization!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
