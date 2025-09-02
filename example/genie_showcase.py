"""
Comprehensive showcase of Genie's frontend and scheduling capabilities.

This example demonstrates all implemented features:
- Phase 1.1: Enhanced semantic metadata capture
- Phase 1.2: FX graph migration
- Phase 1.3: TorchDynamo pattern matching
- Phase 2.1: Semantic optimizations
- Phase 2.2: Execution phase detection

With real CUDA execution for various workload types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import time
from typing import Dict, Any, Optional, Tuple
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genie.core.lazy_tensor import LazyTensor
from genie.core.fx_graph_builder import FXGraphBuilder
from genie.semantic.enhanced_analyzer import EnhancedSemanticAnalyzer
from genie.semantic.optimizer import SemanticOptimizer
from genie.semantic.scheduling import Scheduler, PipelineScheduler
from genie.semantic.placement import PlacementEngine
from genie.semantic.phase_detector import get_phase_detector
from genie.semantic.hooks import HookManager
from genie.semantic.workload import WorkloadType
from genie.core.fx_executor import OptimizingFXExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title: str) -> None:
    """Print a sub-header."""
    print(f"\n>>> {title}")
    print("-" * 40)


class LLMModel(nn.Module):
    """Simple LLM-like model for demonstration."""
    
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ffn2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, kv_cache=None):
        # Embedding
        x = self.embedding(x)
        
        # Self-attention with optional KV cache
        residual = x
        x = self.layer_norm1(x)
        
        if kv_cache is not None:
            # Decode mode with KV cache
            k_cache, v_cache = kv_cache
            # Simplified KV cache concatenation
            x, _ = self.attention(x, x, x)
        else:
            # Prefill mode
            x, _ = self.attention(x, x, x)
        
        x = x + residual
        
        # FFN
        residual = x
        x = self.layer_norm2(x)
        x = self.ffn1(x)
        x = F.gelu(x)
        x = self.ffn2(x)
        x = x + residual
        
        # Output projection
        x = self.output(x)
        return x


class VisionModel(nn.Module):
    """Simple CNN model for vision tasks."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # CNN backbone
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Stage 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Stage 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Classification head
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        
        return x


class MultiModalModel(nn.Module):
    """Simple multi-modal model combining vision and text."""
    
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__()
        # Vision branch
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, hidden_dim)
        )
        
        # Text branch
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.text_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Cross-modal fusion
        self.cross_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, image, text):
        # Vision encoding
        vision_features = self.vision_encoder(image)  # [batch, hidden_dim]
        vision_features = vision_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Text encoding
        text_embed = self.text_embedding(text)  # [batch, seq_len, hidden_dim]
        text_features, _ = self.text_encoder(text_embed)
        text_pooled = text_features.mean(dim=1, keepdim=True)  # [batch, 1, hidden_dim]
        
        # Cross-modal attention
        fused, _ = self.cross_attention(vision_features, text_pooled, text_pooled)
        
        # Final fusion
        combined = torch.cat([fused.squeeze(1), text_pooled.squeeze(1)], dim=-1)
        fused_features = self.fusion_layer(combined)
        output = self.output(fused_features)
        
        return output


def test_llm_workload(device: str = "cuda"):
    """Test LLM workload with semantic analysis."""
    print_header("LLM Workload Testing")
    
    # Create model
    model = LLMModel().to(device)
    model.eval()
    
    # Reset for clean test
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    phase_detector = get_phase_detector()
    phase_detector.reset()
    
    print_subheader("1. Prefill Phase (Processing prompt)")
    
    # Prefill: Process multiple tokens
    batch_size = 2
    seq_len = 20  # Prompt length
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    
    # Hook injection for phase detection
    hook_manager = HookManager(enable_phase_detection=True)
    hook_manager.inject_hooks(model)
    
    # Run prefill
    with torch.no_grad():
        start_time = time.time()
        output_prefill = model(input_ids)
        prefill_time = (time.time() - start_time) * 1000
    
    print(f"Prefill output shape: {output_prefill.shape}")
    print(f"Prefill time: {prefill_time:.2f} ms")
    
    # Check phase detection
    phase_stats = phase_detector.get_phase_statistics()
    print(f"Detected phase: {phase_stats['current_phase']}")
    print(f"Token position: {phase_stats['token_position']}")
    
    print_subheader("2. Decode Phase (Generating tokens)")
    
    # Mark batch boundary
    phase_detector.mark_batch_boundary()
    
    # Decode: Generate single tokens
    decode_times = []
    for i in range(5):  # Generate 5 tokens
        single_token = torch.randint(0, 1000, (batch_size, 1)).to(device)
        
        with torch.no_grad():
            start_time = time.time()
            output_decode = model(single_token, kv_cache=None)  # Simplified, no real cache
            decode_time = (time.time() - start_time) * 1000
            decode_times.append(decode_time)
    
    print(f"Decode output shape: {output_decode.shape}")
    print(f"Average decode time: {sum(decode_times)/len(decode_times):.2f} ms")
    
    # Build LazyTensor graph for analysis
    print_subheader("3. Semantic Analysis")
    
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    
    # Create LazyTensor computation
    x = LazyTensor("aten::randn", [[batch_size, seq_len, 512]], {})
    
    # Simulate attention pattern
    q = LazyTensor("aten::linear", [x], {})
    k = LazyTensor("aten::linear", [x], {})
    v = LazyTensor("aten::linear", [x], {})
    
    # KV cache simulation
    k_cache = LazyTensor("aten::zeros", [[batch_size, 5, 512]], {})
    v_cache = LazyTensor("aten::zeros", [[batch_size, 5, 512]], {})
    k_updated = LazyTensor("aten::cat", [k_cache, k], {"dim": 1})
    v_updated = LazyTensor("aten::cat", [v_cache, v], {"dim": 1})
    
    # Attention computation
    k_t = LazyTensor("aten::transpose", [k_updated], {"dim0": -2, "dim1": -1})
    scores = LazyTensor("aten::matmul", [q, k_t], {})
    weights = LazyTensor("aten::softmax", [scores], {"dim": -1})
    attn_out = LazyTensor("aten::matmul", [weights, v_updated], {})
    
    # FFN
    ffn1 = LazyTensor("aten::linear", [attn_out], {})
    ffn1_act = LazyTensor("aten::gelu", [ffn1], {})
    output = LazyTensor("aten::linear", [ffn1_act], {})
    
    # Get FX graph
    fx_builder = FXGraphBuilder.current()
    fx_builder.mark_output(output)
    fx_graph = fx_builder.to_graph_module()
    
    # Analyze with enhanced analyzer
    analyzer = EnhancedSemanticAnalyzer(use_dynamo=False, enable_phase_detection=True)
    profile = analyzer.analyze_fx_graph(fx_graph)
    
    print(f"Workload type detected: {profile.workload_type}")
    print(f"Phase info: {profile.metadata.get('phase_info', {})}")
    
    # Apply semantic optimizations
    print_subheader("4. Semantic Optimizations")
    
    optimizer = SemanticOptimizer(enable_all=True)
    optimized_graph, opt_plan = optimizer.optimize(fx_graph, profile)
    
    print(f"Optimizations applied: {[opt.value for opt in opt_plan.optimizations]}")
    print(f"KV cache colocation groups: {opt_plan.colocation_groups}")
    print(f"Parallel branches: {len(opt_plan.parallel_branches)}")
    
    # Schedule execution
    scheduler = Scheduler()
    schedule = scheduler.create_schedule(optimized_graph, opt_plan)
    
    print(f"Execution stages: {schedule.total_stages}")
    print(f"Scheduling strategy: {schedule.strategy}")
    
    # Device placement
    placement_engine = PlacementEngine()
    placement_plan = placement_engine.create_placement_plan(optimized_graph, opt_plan, schedule)
    
    print(f"Devices used: {placement_plan.total_devices_used}")
    print(f"Device assignments: {list(placement_plan.device_assignments.keys())}")
    
    return profile, opt_plan


def test_vision_workload(device: str = "cuda"):
    """Test vision CNN workload."""
    print_header("Vision CNN Workload Testing")
    
    # Create model
    model = VisionModel().to(device)
    model.eval()
    
    # Test input
    batch_size = 4
    input_image = torch.randn(batch_size, 3, 64, 64).to(device)
    
    print_subheader("1. Forward Pass")
    
    # Run forward pass
    with torch.no_grad():
        start_time = time.time()
        output = model(input_image)
        forward_time = (time.time() - start_time) * 1000
    
    print(f"Output shape: {output.shape}")
    print(f"Forward time: {forward_time:.2f} ms")
    
    # Build LazyTensor graph
    print_subheader("2. Pipeline Analysis")
    
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    phase_detector = get_phase_detector()
    phase_detector.reset()
    
    # Create CNN pipeline in LazyTensor
    x = LazyTensor("aten::randn", [[batch_size, 3, 64, 64]], {})
    
    # Stage 1
    conv1 = LazyTensor("aten::conv2d", [x], {"stride": 1, "padding": 1})
    bn1 = LazyTensor("aten::batch_norm", [conv1], {})
    relu1 = LazyTensor("aten::relu", [bn1], {})
    pool1 = LazyTensor("aten::max_pool2d", [relu1], {"kernel_size": 2})
    
    # Stage 2
    conv2 = LazyTensor("aten::conv2d", [pool1], {"stride": 1, "padding": 1})
    bn2 = LazyTensor("aten::batch_norm", [conv2], {})
    relu2 = LazyTensor("aten::relu", [bn2], {})
    pool2 = LazyTensor("aten::max_pool2d", [relu2], {"kernel_size": 2})
    
    # Stage 3
    conv3 = LazyTensor("aten::conv2d", [pool2], {"stride": 1, "padding": 1})
    bn3 = LazyTensor("aten::batch_norm", [conv3], {})
    relu3 = LazyTensor("aten::relu", [bn3], {})
    
    # Classification head
    gap = LazyTensor("aten::adaptive_avg_pool2d", [relu3], {"output_size": [1, 1]})
    flat = LazyTensor("aten::flatten", [gap], {"start_dim": 1})
    output = LazyTensor("aten::linear", [flat], {})
    
    # Get FX graph
    fx_builder = FXGraphBuilder.current()
    fx_builder.mark_output(output)
    fx_graph = fx_builder.to_graph_module()
    
    # Analyze
    analyzer = EnhancedSemanticAnalyzer(use_dynamo=False)
    profile = analyzer.analyze_fx_graph(fx_graph)
    
    print(f"Workload type detected: {profile.workload_type}")
    
    # Apply pipeline optimization
    print_subheader("3. Pipeline Optimization")
    
    optimizer = SemanticOptimizer(enable_all=True)
    optimized_graph, opt_plan = optimizer.optimize(fx_graph, profile)
    
    print(f"Optimizations applied: {[opt.value for opt in opt_plan.optimizations]}")
    print(f"Pipeline stages: {len(opt_plan.pipeline_stages)}")
    
    # Pipeline scheduling
    pipeline_scheduler = PipelineScheduler(num_stages=3)
    pipeline_schedule = pipeline_scheduler.create_pipeline_schedule(optimized_graph)
    
    print(f"Pipeline depth: {pipeline_schedule.metadata.get('pipeline_depth', 0)}")
    print(f"Total stages: {pipeline_schedule.total_stages}")
    
    # Check phase detection
    phase_stats = phase_detector.get_phase_statistics()
    print(f"Vision stages detected: {phase_stats['vision_stages']}")
    print(f"Current phase: {phase_stats['current_phase']}")
    
    return profile, opt_plan


def test_multimodal_workload(device: str = "cuda"):
    """Test multi-modal workload."""
    print_header("Multi-Modal Workload Testing")
    
    # Create model
    model = MultiModalModel().to(device)
    model.eval()
    
    # Test inputs
    batch_size = 2
    image = torch.randn(batch_size, 3, 32, 32).to(device)
    text = torch.randint(0, 1000, (batch_size, 10)).to(device)
    
    print_subheader("1. Multi-Modal Forward Pass")
    
    # Run forward pass
    with torch.no_grad():
        start_time = time.time()
        output = model(image, text)
        forward_time = (time.time() - start_time) * 1000
    
    print(f"Output shape: {output.shape}")
    print(f"Forward time: {forward_time:.2f} ms")
    
    # Build LazyTensor graph
    print_subheader("2. Cross-Modal Analysis")
    
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    phase_detector = get_phase_detector()
    phase_detector.reset()
    
    # Vision branch
    img_tensor = LazyTensor("aten::randn", [[batch_size, 3, 32, 32]], {})
    conv = LazyTensor("aten::conv2d", [img_tensor], {"stride": 1, "padding": 1})
    relu = LazyTensor("aten::relu", [conv], {})
    pool = LazyTensor("aten::adaptive_avg_pool2d", [relu], {"output_size": [1, 1]})
    vision_flat = LazyTensor("aten::flatten", [pool], {"start_dim": 1})
    vision_features = LazyTensor("aten::linear", [vision_flat], {})
    
    # Text branch
    text_tensor = LazyTensor("aten::randint", [[batch_size, 10]], {"low": 0, "high": 1000})
    text_embed = LazyTensor("aten::embedding", [text_tensor], {})
    text_features = LazyTensor("aten::lstm", [text_embed], {})
    text_pooled = LazyTensor("aten::mean", [text_features], {"dim": 1})
    
    # Cross-modal fusion
    # Cross-attention
    q_vision = LazyTensor("aten::linear", [vision_features], {})
    k_text = LazyTensor("aten::linear", [text_pooled], {})
    v_text = LazyTensor("aten::linear", [text_pooled], {})
    
    # Attention scores
    k_t = LazyTensor("aten::transpose", [k_text], {"dim0": -2, "dim1": -1})
    scores = LazyTensor("aten::matmul", [q_vision, k_t], {})
    weights = LazyTensor("aten::softmax", [scores], {"dim": -1})
    cross_attn = LazyTensor("aten::matmul", [weights, v_text], {})
    
    # Final fusion
    fused = LazyTensor("aten::cat", [vision_features, text_pooled], {"dim": -1})
    output = LazyTensor("aten::linear", [fused], {})
    
    # Get FX graph
    fx_builder = FXGraphBuilder.current()
    fx_builder.mark_output(output)
    fx_graph = fx_builder.to_graph_module()
    
    # Analyze
    analyzer = EnhancedSemanticAnalyzer(use_dynamo=False)
    profile = analyzer.analyze_fx_graph(fx_graph)
    
    print(f"Workload type detected: {profile.workload_type}")
    
    # Apply multi-modal optimizations
    print_subheader("3. Fusion Optimization")
    
    optimizer = SemanticOptimizer(enable_all=True)
    optimized_graph, opt_plan = optimizer.optimize(fx_graph, profile)
    
    print(f"Optimizations applied: {[opt.value for opt in opt_plan.optimizations]}")
    print(f"Fusion points: {opt_plan.fusion_points}")
    
    # Check modality detection
    phase_stats = phase_detector.get_phase_statistics()
    print(f"Detected modalities: {phase_stats['detected_modalities']}")
    
    # Placement for fusion
    placement_engine = PlacementEngine()
    placement_plan = placement_engine.create_placement_plan(optimized_graph, opt_plan)
    
    # Check fusion device placement
    fusion_devices = set()
    for node_name, decision in placement_plan.decisions.items():
        if 'fusion' in node_name.lower() or 'cat' in node_name.lower():
            fusion_devices.add(decision.device_id)
    
    print(f"Fusion operations placed on: {fusion_devices}")
    
    return profile, opt_plan


def test_fx_execution():
    """Test FX graph execution with optimizations."""
    print_header("FX Graph Execution Testing")
    
    print_subheader("1. Building Test Graph")
    
    # Reset and build a simple graph
    FXGraphBuilder.reset()
    LazyTensor.reset_id_counter()
    
    # Create a simple computation
    x = LazyTensor("aten::randn", [[2, 4, 4]], {})
    w1 = LazyTensor("aten::randn", [[4, 8]], {})
    
    # Linear transformation
    y = LazyTensor("aten::matmul", [x, w1], {})
    y_act = LazyTensor("aten::relu", [y], {})
    
    # Another linear
    w2 = LazyTensor("aten::randn", [[8, 4]], {})
    z = LazyTensor("aten::matmul", [y_act, w2], {})
    
    # Get FX graph
    fx_builder = FXGraphBuilder.current()
    fx_builder.mark_output(z)
    fx_graph = fx_builder.to_graph_module()
    
    print(f"Graph nodes: {len(list(fx_graph.graph.nodes))}")
    
    print_subheader("2. Executing with Optimizations")
    
    # Create optimizing executor
    executor = OptimizingFXExecutor(fx_graph, enable_optimizations=True)
    
    # Execute the graph
    result = executor.run()
    
    print(f"Execution completed")
    
    # Check if the executor has a get_summary method
    if hasattr(executor, 'get_summary'):
        summary = executor.get_summary()
        print(f"Total operations: {summary['total_operations']}")
        print(f"Operation counts: {summary['op_counts']}")
        print(f"Metadata summary: {summary['metadata_summary']}")
    else:
        # Fallback - just use basic info
        summary = {
            'total_operations': len(list(fx_graph.graph.nodes)),
            'op_counts': executor.op_counts if hasattr(executor, 'op_counts') else {},
            'metadata_summary': executor.metadata_summary if hasattr(executor, 'metadata_summary') else {}
        }
        print(f"Total operations: {summary['total_operations']}")
        if summary['op_counts']:
            print(f"Operation counts: {dict(summary['op_counts'])}")
    
    if result is not None and hasattr(result, 'shape'):
        print(f"Result shape: {result.shape}")
    
    return summary


def main():
    """Main demonstration function."""
    print_header("GENIE COMPREHENSIVE SHOWCASE")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Test LLM workload
        llm_profile, llm_plan = test_llm_workload(device)
        
        # Test Vision workload
        vision_profile, vision_plan = test_vision_workload(device)
        
        # Test Multi-modal workload
        mm_profile, mm_plan = test_multimodal_workload(device)
        
        # Test FX execution
        exec_summary = test_fx_execution()
        
        # Summary
        print_header("SUMMARY")
        
        print("\n📊 Workload Detection:")
        print(f"  - LLM: {llm_profile.workload_type}")
        print(f"  - Vision: {vision_profile.workload_type}")
        print(f"  - Multi-modal: {mm_profile.workload_type}")
        
        print("\n🔧 Optimizations Applied:")
        print(f"  - LLM: {len(llm_plan.optimizations)} optimizations")
        print(f"  - Vision: {len(vision_plan.optimizations)} optimizations")
        print(f"  - Multi-modal: {len(mm_plan.optimizations)} optimizations")
        
        print("\n🎯 Phase Detection:")
        phase_detector = get_phase_detector()
        final_stats = phase_detector.get_phase_statistics()
        print(f"  - Total phases detected: {final_stats['total_phases']}")
        print(f"  - Transitions: {final_stats['transitions']}")
        print(f"  - Vision stages: {final_stats['vision_stages']}")
        print(f"  - Modalities: {final_stats['detected_modalities']}")
        
        print("\n✅ All features demonstrated successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
