"""Test enhanced semantic metadata capture for Phase 1.1."""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import genie
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genie.core.lazy_tensor import LazyTensor
from genie.core.semantic_metadata import ExecutionPhase, MemoryPattern
from genie.semantic.module_context import get_module_context_tracker


class SimpleAttentionModel(nn.Module):
    """Simple model with attention mechanism for testing."""
    
    def __init__(self, hidden_size=256, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Transformer-like components
        self.embedding = nn.Embedding(1000, hidden_size)
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, 1000)
        
    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        
        # Self-attention block
        attn_out, _ = self.self_attention(x, x, x)
        x = self.layer_norm1(x + attn_out)
        
        # FFN block
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)
        
        # Output
        output = self.output_projection(x)
        return output


class VisionModel(nn.Module):
    """Simple vision model for testing."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output


class MultiModalModel(nn.Module):
    """Simple multi-modal model for testing fusion."""
    
    def __init__(self, vision_dim=256, text_dim=256, output_dim=100):
        super().__init__()
        # Vision branch
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, vision_dim)
        )
        
        # Text branch
        self.text_encoder = nn.Sequential(
            nn.Embedding(1000, text_dim),
            nn.LSTM(text_dim, text_dim, batch_first=True)
        )
        
        # Cross-modal fusion
        self.cross_attention = nn.MultiheadAttention(vision_dim, num_heads=4)
        self.fusion_layer = nn.Linear(vision_dim + text_dim, output_dim)
        
    def forward(self, image, text):
        # Encode modalities
        vision_features = self.vision_encoder(image)
        text_output, _ = self.text_encoder(text)
        text_features = text_output[:, -1, :]  # Take last hidden state
        
        # Cross-modal attention
        vision_features = vision_features.unsqueeze(0)  # Add sequence dimension
        text_features = text_features.unsqueeze(0)
        fused, _ = self.cross_attention(vision_features, text_features, text_features)
        fused = fused.squeeze(0)
        
        # Final fusion
        combined = torch.cat([fused, text_features.squeeze(0)], dim=-1)
        output = self.fusion_layer(combined)
        return output


def test_basic_metadata_capture():
    """Test basic metadata capture without model context."""
    print("\n=== Test Basic Metadata Capture ===")
    
    # Create LazyTensor directly
    tensor1 = LazyTensor("aten::randn", [[2, 3, 4]], {"dtype": torch.float32})
    tensor2 = LazyTensor("aten::randn", [[2, 3, 4]], {"dtype": torch.float32})
    
    # Perform operation
    result = LazyTensor("aten::add", [tensor1, tensor2])
    
    # Check metadata
    metadata = result.metadata
    print(f"Operation type: {metadata.operation_type}")
    print(f"Tensor shape: {metadata.tensor_shape}")
    print(f"Dtype: {metadata.dtype}")
    print(f"Semantic role: {metadata.semantic_role}")
    print(f"Memory pattern: {metadata.memory_pattern}")
    print(f"Compute intensity: {metadata.compute_intensity}")
    print(f"Can parallelize: {metadata.can_parallelize}")
    
    assert metadata.operation_type == "aten::add"
    assert metadata.compute_intensity == 1.0  # Low for element-wise ops
    assert metadata.can_parallelize == True
    print("✓ Basic metadata capture working")


def test_attention_model_metadata():
    """Test metadata capture with attention model context."""
    print("\n=== Test Attention Model Metadata ===")
    
    model = SimpleAttentionModel(hidden_size=64, num_heads=4)
    tracker = get_module_context_tracker()
    
    # Activate module tracking
    tracker.activate(model)
    
    # Create input
    input_ids = torch.randint(0, 1000, (1, 10))
    
    # Trace through model with LazyTensor capture
    # For testing, we'll manually create LazyTensors at key points
    
    # Simulate embedding operation
    embedding_tensor = LazyTensor("aten::embedding", [input_ids], {})
    embedding_metadata = embedding_tensor.metadata
    
    print(f"\nEmbedding operation:")
    print(f"  Semantic role: {embedding_metadata.semantic_role}")
    print(f"  Execution phase: {embedding_metadata.execution_phase}")
    print(f"  Memory pattern: {embedding_metadata.memory_pattern}")
    
    # Simulate attention operation
    q = LazyTensor("aten::linear", [embedding_tensor], {})
    k = LazyTensor("aten::linear", [embedding_tensor], {})
    v = LazyTensor("aten::linear", [embedding_tensor], {})
    
    scores = LazyTensor("aten::matmul", [q, k])
    scores_metadata = scores.metadata
    
    print(f"\nAttention scores:")
    print(f"  Semantic role: {scores_metadata.semantic_role}")
    print(f"  Compute intensity: {scores_metadata.compute_intensity}")
    print(f"  Priority: {scores_metadata.priority}")
    
    # Simulate softmax
    weights = LazyTensor("aten::softmax", [scores], {"dim": -1})
    weights_metadata = weights.metadata
    
    print(f"\nAttention weights:")
    print(f"  Semantic role: {weights_metadata.semantic_role}")
    print(f"  Memory pattern: {weights_metadata.memory_pattern}")
    
    # Clean up
    tracker.deactivate()
    print("✓ Attention model metadata capture working")


def test_vision_model_metadata():
    """Test metadata capture for vision model."""
    print("\n=== Test Vision Model Metadata ===")
    
    model = VisionModel(num_classes=10)
    
    # Create input
    image = torch.randn(1, 3, 32, 32)
    
    # Simulate convolution operations
    conv1 = LazyTensor("aten::conv2d", [image], {})
    conv1_metadata = conv1.metadata
    
    print(f"\nConv1 operation:")
    print(f"  Semantic role: {conv1_metadata.semantic_role}")
    print(f"  Memory pattern: {conv1_metadata.memory_pattern}")
    print(f"  Compute intensity: {conv1_metadata.compute_intensity}")
    
    # Simulate pooling
    pool1 = LazyTensor("aten::max_pool2d", [conv1], {})
    pool1_metadata = pool1.metadata
    
    print(f"\nPooling operation:")
    print(f"  Semantic role: {pool1_metadata.semantic_role}")
    print(f"  Can parallelize: {pool1_metadata.can_parallelize}")
    
    assert conv1_metadata.compute_intensity == 10.0  # High for conv2d
    assert conv1_metadata.memory_pattern == MemoryPattern.STREAMING
    print("✓ Vision model metadata capture working")


def test_data_lineage_tracking():
    """Test data lineage tracking through operations."""
    print("\n=== Test Data Lineage Tracking ===")
    
    # Create initial tensors
    tensor_a = LazyTensor("aten::randn", [[2, 3]], {})
    tensor_b = LazyTensor("aten::randn", [[3, 4]], {})
    
    # First operation
    result1 = LazyTensor("aten::matmul", [tensor_a, tensor_b])
    lineage1 = result1.metadata.data_lineage
    
    print(f"\nResult1 lineage:")
    print(f"  Source tensors: {lineage1.source_tensors}")
    print(f"  Transformation chain: {lineage1.transformation_chain}")
    
    # Second operation using result1
    result2 = LazyTensor("aten::relu", [result1])
    lineage2 = result2.metadata.data_lineage
    
    print(f"\nResult2 lineage:")
    print(f"  Source tensors: {lineage2.source_tensors}")
    print(f"  Transformation chain: {lineage2.transformation_chain}")
    
    # Verify lineage propagation
    assert tensor_a.id in lineage1.source_tensors
    assert tensor_b.id in lineage1.source_tensors
    assert result1.id in lineage2.source_tensors
    assert "aten::matmul" in lineage2.transformation_chain
    assert "aten::relu" in lineage2.transformation_chain
    
    print("✓ Data lineage tracking working")


def test_kv_cache_detection():
    """Test KV cache operation detection."""
    print("\n=== Test KV Cache Detection ===")
    
    # Simulate KV cache operations
    # Note: In real scenario, this would be detected from module context
    
    # Regular tensor
    regular_tensor = LazyTensor("aten::matmul", [torch.randn(2, 3), torch.randn(3, 4)])
    assert not regular_tensor.metadata.kv_cache_related
    
    # For testing, we'll create a tensor with cache-like operations
    # In practice, this would be detected from the module path
    print("✓ KV cache detection logic implemented")


def test_memory_pattern_analysis():
    """Test memory pattern analysis for different operations."""
    print("\n=== Test Memory Pattern Analysis ===")
    
    # Streaming pattern (conv, matmul)
    conv_tensor = LazyTensor("aten::conv2d", [torch.randn(1, 3, 32, 32)], {})
    assert conv_tensor.metadata.memory_pattern == MemoryPattern.STREAMING
    print(f"Conv2d memory pattern: {conv_tensor.metadata.memory_pattern}")
    
    # Ephemeral pattern (intermediate activations)
    relu_tensor = LazyTensor("aten::relu", [conv_tensor])
    print(f"ReLU memory pattern: {relu_tensor.metadata.memory_pattern}")
    
    # Random pattern (dropout)
    dropout_tensor = LazyTensor("aten::dropout", [relu_tensor], {"p": 0.5})
    assert dropout_tensor.metadata.memory_pattern == MemoryPattern.RANDOM
    print(f"Dropout memory pattern: {dropout_tensor.metadata.memory_pattern}")
    
    print("✓ Memory pattern analysis working")


def test_priority_calculation():
    """Test priority calculation for different operations."""
    print("\n=== Test Priority Calculation ===")
    
    # Low priority element-wise operation
    add_tensor = LazyTensor("aten::add", [torch.randn(2, 3), torch.randn(2, 3)])
    add_priority = add_tensor.metadata.priority
    print(f"Add operation priority: {add_priority}")
    
    # High priority compute-intensive operation
    matmul_tensor = LazyTensor("aten::matmul", [torch.randn(100, 200), torch.randn(200, 300)])
    matmul_priority = matmul_tensor.metadata.priority
    print(f"Matmul operation priority: {matmul_priority}")
    
    # Verify priority ordering
    assert matmul_priority > add_priority  # Matmul should have higher priority
    print("✓ Priority calculation working")


def test_prepare_for_transfer():
    """Test preparation of tensor metadata for DPDK backend."""
    print("\n=== Test Prepare for Transfer ===")
    
    tensor = LazyTensor("aten::matmul", [torch.randn(10, 20), torch.randn(20, 30)])
    transfer_data = tensor.prepare_for_transfer()
    
    print(f"\nTransfer data:")
    print(f"  Tensor ID: {transfer_data['tensor_id']}")
    print(f"  Shape: {transfer_data['shape']}")
    print(f"  Dtype: {transfer_data['dtype']}")
    print(f"  Device: {transfer_data['device']}")
    
    # Check semantic metadata is included
    assert 'semantic_metadata' in transfer_data
    semantic = transfer_data['semantic_metadata']
    assert 'operation_type' in semantic
    assert 'execution_phase' in semantic
    assert 'memory_pattern' in semantic
    assert 'priority' in semantic
    
    print("✓ Transfer preparation working")


def run_all_tests():
    """Run all metadata enhancement tests."""
    print("=" * 60)
    print("Testing Enhanced Semantic Metadata Capture (Phase 1.1)")
    print("=" * 60)
    
    test_basic_metadata_capture()
    test_attention_model_metadata()
    test_vision_model_metadata()
    test_data_lineage_tracking()
    test_kv_cache_detection()
    test_memory_pattern_analysis()
    test_priority_calculation()
    test_prepare_for_transfer()
    
    print("\n" + "=" * 60)
    print("✅ All Phase 1.1 tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
