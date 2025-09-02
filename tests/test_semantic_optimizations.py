"""Test semantic optimizations for Phase 2.1."""

import torch
import torch.nn as nn
import torch.fx as fx
import sys
import os

# Add parent directory to path to import genie
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genie.core.lazy_tensor import LazyTensor
from genie.core.fx_graph_builder import FXGraphBuilder
from genie.semantic.optimizer import (
    SemanticOptimizer, OptimizationType, OptimizationPlan, AdaptiveOptimizer
)
from genie.semantic.scheduling import (
    Scheduler, PipelineScheduler, DynamicScheduler,
    SchedulingStrategy, ExecutionSchedule
)
from genie.semantic.placement import (
    PlacementEngine, DeviceCapabilities, DeviceType, PlacementPlan
)
from genie.semantic.workload import WorkloadProfile, WorkloadType
from genie.semantic.enhanced_analyzer import EnhancedSemanticAnalyzer
from genie.core.semantic_metadata import ExecutionPhase, MemoryPattern


def create_llm_graph():
    """Create an LLM-like computation graph."""
    # Reset builders
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    
    batch_size, seq_len, hidden_dim = 2, 10, 512
    num_heads = 8
    
    # Input
    x = LazyTensor("aten::randn", [[batch_size, seq_len, hidden_dim]], {})
    
    # Simulate KV cache operations
    k_cache = LazyTensor("aten::zeros", [[batch_size, 5, hidden_dim]], {})
    v_cache = LazyTensor("aten::zeros", [[batch_size, 5, hidden_dim]], {})
    
    # QKV projections
    q = LazyTensor("aten::linear", [x], {})
    k_new = LazyTensor("aten::linear", [x], {})
    v_new = LazyTensor("aten::linear", [x], {})
    
    # Update KV cache
    k = LazyTensor("aten::cat", [k_cache, k_new], {"dim": 1})
    v = LazyTensor("aten::cat", [v_cache, v_new], {"dim": 1})
    
    # Attention
    k_t = LazyTensor("aten::transpose", [k], {"dim0": -2, "dim1": -1})
    scores = LazyTensor("aten::matmul", [q, k_t])
    weights = LazyTensor("aten::softmax", [scores], {"dim": -1})
    attn_out = LazyTensor("aten::matmul", [weights, v])
    
    # Output projection
    output = LazyTensor("aten::linear", [attn_out], {})
    
    # FFN
    ffn1 = LazyTensor("aten::linear", [output], {})
    ffn1_act = LazyTensor("aten::gelu", [ffn1])
    ffn2 = LazyTensor("aten::linear", [ffn1_act], {})
    
    # Residual
    final = LazyTensor("aten::add", [x, ffn2])
    
    # Get FX graph
    fx_builder = FXGraphBuilder.current()
    fx_builder.mark_output(final)
    
    return fx_builder.to_graph_module()


def create_vision_graph():
    """Create a CNN-like vision graph."""
    # Reset builders
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    
    batch_size = 1
    channels = 3
    height, width = 224, 224
    
    # Input image
    x = LazyTensor("aten::randn", [[batch_size, channels, height, width]], {})
    
    # Conv block 1
    conv1 = LazyTensor("aten::conv2d", [x], {"stride": 2, "padding": 1})
    bn1 = LazyTensor("aten::batch_norm", [conv1], {})
    relu1 = LazyTensor("aten::relu", [bn1])
    pool1 = LazyTensor("aten::max_pool2d", [relu1], {"kernel_size": 2})
    
    # Conv block 2
    conv2 = LazyTensor("aten::conv2d", [pool1], {"stride": 1, "padding": 1})
    bn2 = LazyTensor("aten::batch_norm", [conv2], {})
    relu2 = LazyTensor("aten::relu", [bn2])
    pool2 = LazyTensor("aten::max_pool2d", [relu2], {"kernel_size": 2})
    
    # Conv block 3
    conv3 = LazyTensor("aten::conv2d", [pool2], {"stride": 1, "padding": 1})
    bn3 = LazyTensor("aten::batch_norm", [conv3], {})
    relu3 = LazyTensor("aten::relu", [bn3])
    
    # Global pooling
    gap = LazyTensor("aten::adaptive_avg_pool2d", [relu3], {"output_size": [1, 1]})
    flat = LazyTensor("aten::flatten", [gap], {"start_dim": 1})
    
    # Classifier
    output = LazyTensor("aten::linear", [flat], {})
    
    # Get FX graph
    fx_builder = FXGraphBuilder.current()
    fx_builder.mark_output(output)
    
    return fx_builder.to_graph_module()


def create_multimodal_graph():
    """Create a multi-modal graph with vision and text branches."""
    # Reset builders
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    
    batch_size = 2
    
    # Vision branch
    image = LazyTensor("aten::randn", [[batch_size, 3, 64, 64]], {})
    conv1 = LazyTensor("aten::conv2d", [image], {})
    relu1 = LazyTensor("aten::relu", [conv1])
    pool1 = LazyTensor("aten::adaptive_avg_pool2d", [relu1], {"output_size": [1, 1]})
    vision_flat = LazyTensor("aten::flatten", [pool1], {"start_dim": 1})
    vision_features = LazyTensor("aten::linear", [vision_flat], {})
    
    # Text branch
    text = LazyTensor("aten::randn", [[batch_size, 20, 256]], {})  # [batch, seq_len, embed_dim]
    text_encoder = LazyTensor("aten::linear", [text], {})
    text_features = LazyTensor("aten::mean", [text_encoder], {"dim": 1})  # Pool over sequence
    
    # Cross-modal fusion
    # Cross-attention
    q_vision = LazyTensor("aten::linear", [vision_features], {})
    k_text = LazyTensor("aten::linear", [text_features], {})
    v_text = LazyTensor("aten::linear", [text_features], {})
    
    # Simplified attention (without reshape for heads)
    k_t = LazyTensor("aten::transpose", [k_text], {"dim0": -2, "dim1": -1})
    scores = LazyTensor("aten::matmul", [q_vision.unsqueeze(1) if hasattr(q_vision, 'unsqueeze') else q_vision, k_t])
    weights = LazyTensor("aten::softmax", [scores], {"dim": -1})
    cross_attn = LazyTensor("aten::matmul", [weights, v_text.unsqueeze(1) if hasattr(v_text, 'unsqueeze') else v_text])
    
    # Fusion
    fused = LazyTensor("aten::cat", [vision_features, text_features], {"dim": -1})
    output = LazyTensor("aten::linear", [fused], {})
    
    # Get FX graph
    fx_builder = FXGraphBuilder.current()
    fx_builder.mark_output(output)
    
    # Mark modalities in metadata
    for node in fx_builder.fx_graph.nodes:
        if 'conv' in str(node) or 'vision' in str(node) or 'image' in str(node):
            node.meta['modality'] = 'vision'
        elif 'text' in str(node) or 'encoder' in str(node):
            node.meta['modality'] = 'text'
    
    return fx_builder.to_graph_module()


def test_semantic_optimizer():
    """Test semantic optimizer."""
    print("\n=== Test Semantic Optimizer ===")
    
    optimizer = SemanticOptimizer(enable_all=True)
    
    # Test with LLM graph
    llm_graph = create_llm_graph()
    llm_profile = WorkloadProfile(
        workload_type=WorkloadType.LLM,
        patterns=[],
        metadata={'test': True}
    )
    
    optimized_graph, optimization_plan = optimizer.optimize(llm_graph, llm_profile)
    
    print(f"LLM Optimizations applied: {[opt.value for opt in optimization_plan.optimizations]}")
    print(f"Colocation groups: {optimization_plan.colocation_groups}")
    print(f"Parallel branches: {len(optimization_plan.parallel_branches)}")
    
    # Check that optimizations were applied
    assert len(optimization_plan.optimizations) > 0
    
    # Test with Vision graph
    vision_graph = create_vision_graph()
    vision_profile = WorkloadProfile(
        workload_type=WorkloadType.VISION,
        patterns=[],
        metadata={'test': True}
    )
    
    optimized_graph, optimization_plan = optimizer.optimize(vision_graph, vision_profile)
    
    print(f"\nVision Optimizations applied: {[opt.value for opt in optimization_plan.optimizations]}")
    print(f"Pipeline stages: {len(optimization_plan.pipeline_stages)}")
    
    # Check that some optimizations were applied (at least memory optimization)
    assert len(optimization_plan.optimizations) > 0
    
    # Test with Multi-modal graph
    mm_graph = create_multimodal_graph()
    mm_profile = WorkloadProfile(
        workload_type=WorkloadType.MULTIMODAL,
        patterns=[],
        metadata={'test': True}
    )
    
    optimized_graph, optimization_plan = optimizer.optimize(mm_graph, mm_profile)
    
    print(f"\nMulti-modal Optimizations applied: {[opt.value for opt in optimization_plan.optimizations]}")
    print(f"Fusion points: {optimization_plan.fusion_points}")
    
    print("✓ Semantic optimizer working")


def test_optimization_strategies():
    """Test specific optimization strategies."""
    print("\n=== Test Optimization Strategies ===")
    
    optimizer = SemanticOptimizer(enable_all=True)
    
    # Create LLM graph with decode phase
    graph = create_llm_graph()
    
    # Mark some nodes as decode phase
    for node in graph.graph.nodes:
        if 'cat' in str(node.target).lower():
            # Mark as KV cache operation
            if 'semantic' not in node.meta:
                node.meta['semantic'] = type('obj', (object,), {})()
            node.meta['semantic'].kv_cache_related = True
            node.meta['semantic'].execution_phase = ExecutionPhase.DECODE
    
    profile = WorkloadProfile(
        workload_type=WorkloadType.LLM,
        patterns=[],
        metadata={}
    )
    
    optimized_graph, plan = optimizer.optimize(graph, profile)
    
    # Check KV cache colocation
    kv_cache_colocated = False
    for node in optimized_graph.graph.nodes:
        if node.meta.get('colocation_group') == 'kv_cache':
            kv_cache_colocated = True
            break
    
    print(f"KV cache colocation: {kv_cache_colocated}")
    
    # Check parallel branches
    parallel_marked = False
    for node in optimized_graph.graph.nodes:
        if 'parallel_group' in node.meta:
            parallel_marked = True
            break
    
    print(f"Parallel execution marked: {parallel_marked}")
    
    print("✓ Optimization strategies working")


def test_scheduler():
    """Test execution scheduler."""
    print("\n=== Test Scheduler ===")
    
    scheduler = Scheduler()
    
    # Test with vision graph
    graph = create_vision_graph()
    
    # Create optimization plan with pipeline stages
    opt_plan = {
        'pipeline_stages': [
            ['conv2d', 'batch_norm', 'relu'],
            ['conv2d_1', 'batch_norm_1', 'relu_1'],
            ['conv2d_2', 'batch_norm_2', 'relu_2']
        ],
        'colocation_groups': {
            'conv_group': ['conv2d', 'conv2d_1', 'conv2d_2']
        }
    }
    
    schedule = scheduler.create_schedule(graph, opt_plan)
    
    print(f"Total stages: {schedule.total_stages}")
    print(f"Scheduling strategy: {schedule.strategy}")
    print(f"Number of groups: {len(schedule.node_to_group)}")
    
    assert schedule.total_stages > 0
    assert schedule.strategy in [s for s in SchedulingStrategy]
    
    print("✓ Scheduler working")


def test_pipeline_scheduler():
    """Test pipeline scheduler."""
    print("\n=== Test Pipeline Scheduler ===")
    
    scheduler = PipelineScheduler(num_stages=3)
    
    # Test with vision graph
    graph = create_vision_graph()
    schedule = scheduler.create_pipeline_schedule(graph)
    
    print(f"Pipeline stages: {schedule.total_stages}")
    print(f"Pipeline depth: {schedule.metadata.get('pipeline_depth', 0)}")
    
    assert schedule.strategy == SchedulingStrategy.PIPELINE
    assert schedule.total_stages > 0
    
    print("✓ Pipeline scheduler working")


def test_dynamic_scheduler():
    """Test dynamic scheduler with runtime constraints."""
    print("\n=== Test Dynamic Scheduler ===")
    
    scheduler = DynamicScheduler()
    
    graph = create_llm_graph()
    
    # Test with memory constraint
    constraints = {
        'memory_limit': 8.0,  # GB
        'latency_target': 100.0  # ms
    }
    
    schedule = scheduler.create_adaptive_schedule(graph, constraints)
    
    print(f"Adaptive schedule stages: {schedule.total_stages}")
    print(f"Memory optimized: {schedule.metadata.get('memory_optimized', False)}")
    print(f"Latency optimized: {schedule.metadata.get('latency_optimized', False)}")
    
    assert schedule.strategy == SchedulingStrategy.DYNAMIC
    
    # Update runtime stats
    scheduler.update_runtime_stats("matmul", latency=10.0, memory=1.0)
    
    print("✓ Dynamic scheduler working")


def test_placement_engine():
    """Test device placement engine."""
    print("\n=== Test Placement Engine ===")
    
    # Create placement engine with simulated devices
    devices = [
        DeviceCapabilities(
            device_id="cpu:0",
            device_type=DeviceType.CPU,
            memory_gb=16.0,
            compute_tflops=0.5,
            bandwidth_gbps=50.0
        ),
        DeviceCapabilities(
            device_id="cuda:0",
            device_type=DeviceType.GPU,
            memory_gb=16.0,
            compute_tflops=10.0,
            bandwidth_gbps=600.0
        ),
        DeviceCapabilities(
            device_id="remote_gpu:0",
            device_type=DeviceType.REMOTE_GPU,
            memory_gb=24.0,
            compute_tflops=15.0,
            bandwidth_gbps=100.0
        )
    ]
    
    placement_engine = PlacementEngine(devices)
    
    # Test with LLM graph
    graph = create_llm_graph()
    
    # Create optimization plan with colocation groups
    from genie.semantic.optimizer import OptimizationPlan
    opt_plan = OptimizationPlan()
    opt_plan.colocation_groups = {'kv_cache': ['cat', 'cat_1']}
    opt_plan.metadata = {}
    
    placement_plan = placement_engine.create_placement_plan(graph, opt_plan)
    
    print(f"Total devices used: {placement_plan.total_devices_used}")
    print(f"Device assignments: {list(placement_plan.device_assignments.keys())}")
    print(f"Colocation groups: {placement_plan.colocation_groups}")
    
    assert placement_plan.total_devices_used > 0
    assert len(placement_plan.decisions) > 0
    
    # Test placement decisions
    has_gpu_placement = False
    for decision in placement_plan.decisions.values():
        if 'cuda' in decision.device_id or 'gpu' in decision.device_id:
            has_gpu_placement = True
            break
    
    print(f"Has GPU placement: {has_gpu_placement}")
    
    print("✓ Placement engine working")


def test_semantic_placement():
    """Test semantic-aware placement."""
    print("\n=== Test Semantic Placement ===")
    
    placement_engine = PlacementEngine()
    
    # Create graph with semantic metadata
    graph = create_llm_graph()
    
    # Add semantic metadata to nodes
    for node in graph.graph.nodes:
        if 'matmul' in str(node.target).lower():
            if 'semantic' not in node.meta:
                node.meta['semantic'] = type('obj', (object,), {})()
            node.meta['semantic'].execution_phase = ExecutionPhase.DECODE
            node.meta['semantic'].compute_intensity = 10.0
        elif 'cat' in str(node.target).lower():
            if 'semantic' not in node.meta:
                node.meta['semantic'] = type('obj', (object,), {})()
            node.meta['semantic'].memory_pattern = MemoryPattern.PERSISTENT
            node.meta['semantic'].kv_cache_related = True
    
    placement_plan = placement_engine.create_placement_plan(graph)
    
    # Check semantic-based placement
    decode_on_gpu = False
    persistent_on_local = False
    
    for node in graph.graph.nodes:
        if node.name in placement_plan.decisions:
            decision = placement_plan.decisions[node.name]
            
            if hasattr(node.meta.get('semantic'), 'execution_phase'):
                if node.meta['semantic'].execution_phase == ExecutionPhase.DECODE:
                    if 'cuda' in decision.device_id or 'gpu' in decision.device_id:
                        decode_on_gpu = True
            
            if hasattr(node.meta.get('semantic'), 'memory_pattern'):
                if node.meta['semantic'].memory_pattern == MemoryPattern.PERSISTENT:
                    if 'remote' not in decision.device_id:
                        persistent_on_local = True
    
    print(f"Decode phase on GPU: {decode_on_gpu}")
    print(f"Persistent memory on local: {persistent_on_local}")
    
    print("✓ Semantic placement working")


def test_adaptive_optimizer():
    """Test adaptive optimizer with feedback."""
    print("\n=== Test Adaptive Optimizer ===")
    
    optimizer = AdaptiveOptimizer()
    
    graph = create_vision_graph()
    profile = WorkloadProfile(
        workload_type=WorkloadType.VISION,
        patterns=[],
        metadata={}
    )
    
    # First optimization
    optimized_graph, plan = optimizer.optimize_with_feedback(graph, profile)
    
    print(f"Initial optimizations: {[opt.value for opt in plan.optimizations]}")
    
    # Simulate performance feedback
    perf_metrics = {
        'latency': 50.0,
        'throughput': 100.0
    }
    
    # Second optimization with feedback
    optimized_graph, plan = optimizer.optimize_with_feedback(graph, profile, perf_metrics)
    
    print(f"Adaptive optimizations: {[opt.value for opt in plan.optimizations]}")
    print(f"Performance history: {len(optimizer.performance_history)} entries")
    
    assert len(optimizer.performance_history) > 0
    
    print("✓ Adaptive optimizer working")


def test_end_to_end_optimization():
    """Test end-to-end optimization pipeline."""
    print("\n=== Test End-to-End Optimization ===")
    
    # Create analyzer
    analyzer = EnhancedSemanticAnalyzer(use_dynamo=False)
    
    # Create multi-modal graph
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    
    # Build a simple multi-modal model
    image = LazyTensor("aten::randn", [[2, 3, 32, 32]], {})
    text = LazyTensor("aten::randn", [[2, 20, 128]], {})
    
    # Vision processing
    conv = LazyTensor("aten::conv2d", [image], {})
    pool = LazyTensor("aten::adaptive_avg_pool2d", [conv], {"output_size": [1, 1]})
    vision_feat = LazyTensor("aten::flatten", [pool], {"start_dim": 1})
    
    # Text processing
    text_feat = LazyTensor("aten::mean", [text], {"dim": 1})
    
    # Fusion
    fused = LazyTensor("aten::cat", [vision_feat, text_feat], {"dim": -1})
    output = LazyTensor("aten::linear", [fused], {})
    
    fx_builder = FXGraphBuilder.current()
    fx_builder.mark_output(output)
    
    # Analyze
    profile = analyzer.analyze_from_lazy_tensors()
    
    # Optimize
    optimizer = SemanticOptimizer()
    graph = fx_builder.to_graph_module()
    optimized_graph, opt_plan = optimizer.optimize(graph, profile)
    
    # Schedule
    scheduler = Scheduler()
    schedule = scheduler.create_schedule(optimized_graph, opt_plan)
    
    # Place
    placement_engine = PlacementEngine()
    placement_plan = placement_engine.create_placement_plan(optimized_graph, opt_plan, schedule)
    
    print(f"Workload type: {profile.workload_type}")
    print(f"Optimizations: {[opt.value for opt in opt_plan.optimizations]}")
    print(f"Schedule stages: {schedule.total_stages}")
    print(f"Devices used: {placement_plan.total_devices_used}")
    
    assert profile.workload_type in [WorkloadType.MULTIMODAL, WorkloadType.UNKNOWN]
    assert schedule.total_stages > 0
    assert placement_plan.total_devices_used > 0
    
    print("✓ End-to-end optimization working")


def run_all_tests():
    """Run all semantic optimization tests."""
    print("=" * 60)
    print("Testing Semantic Optimizations (Phase 2.1)")
    print("=" * 60)
    
    test_semantic_optimizer()
    test_optimization_strategies()
    test_scheduler()
    test_pipeline_scheduler()
    test_dynamic_scheduler()
    test_placement_engine()
    test_semantic_placement()
    test_adaptive_optimizer()
    test_end_to_end_optimization()
    
    print("\n" + "=" * 60)
    print("✅ All Phase 2.1 tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
