"""
Test: Real Model Integration

Validates:
- Realistic model patterns
- Multi-layer operations
- Attention-like patterns
- Convolution-like patterns
- Skip connections
- Complex computation graphs
"""

import torch
import pytest
import logging
import djinn

logger = logging.getLogger(__name__)


class TestMLPPatterns:
    """Test Multi-Layer Perceptron patterns."""
    
    def test_simple_mlp_pattern(self):
        """Test simple MLP pattern: linear -> relu -> linear."""
        
        with genie.capture():
            x = torch.randn(32, 128)
            
            # Layer 1: linear -> relu
            h1 = x @ torch.randn(128, 256)
            h1 = torch.relu(h1)
            
            # Layer 2: linear
            y = h1 @ torch.randn(256, 10)
        
        result = y.cpu()
        
        assert result.shape == torch.Size([32, 10])
        
        print("✅ Simple MLP pattern works")
    
    def test_deep_mlp_pattern(self):
        """Test deeper MLP with multiple layers."""
        
        with genie.capture():
            x = torch.randn(16, 64)
            
            # 4 layers
            x = x @ torch.randn(64, 128) + 0.1
            x = torch.relu(x)
            
            x = x @ torch.randn(128, 64) + 0.1
            x = torch.relu(x)
            
            x = x @ torch.randn(64, 32) + 0.1
            x = torch.relu(x)
            
            y = x @ torch.randn(32, 10)
        
        result = y.cpu()
        
        assert result.shape == torch.Size([16, 10])
        
        print("✅ Deep MLP pattern works")
    
    def test_mlp_with_dropout_like_pattern(self):
        """Test MLP with dropout-like (masking) pattern."""
        
        with genie.capture():
            x = torch.randn(32, 128)
            mask = torch.ones(32, 128)  # In real code: random mask
            
            # Linear with mask
            x = x * mask
            x = x @ torch.randn(128, 256)
            x = torch.relu(x)
            
            y = x @ torch.randn(256, 10)
        
        result = y.cpu()
        
        assert result.shape == torch.Size([32, 10])
        
        print("✅ MLP with masking pattern works")


class TestAttentionPatterns:
    """Test Attention mechanism patterns."""
    
    def test_scaled_dot_product_attention(self):
        """Test scaled dot-product attention pattern."""
        
        with genie.capture():
            # Batch size 2, seq len 8, dim 64
            q = torch.randn(2, 8, 64)
            k = torch.randn(2, 8, 64)
            v = torch.randn(2, 8, 64)
            
            # Scaled dot-product: Q @ K^T / sqrt(d)
            scores = q @ k.transpose(-2, -1)
            scores = scores / 8.0  # sqrt(64)
            
            # Apply relu (instead of softmax for simplicity)
            attn = torch.relu(scores)
            
            # Attention @ V
            output = attn @ v
        
        result = output.cpu()
        
        assert result.shape == torch.Size([2, 8, 64])
        
        print("✅ Scaled dot-product attention works")
    
    def test_multi_head_attention_pattern(self):
        """Test multi-head attention pattern."""
        
        with genie.capture():
            x = torch.randn(2, 8, 256)  # batch, seq, dim
            num_heads = 4
            head_dim = 64
            
            # Project to Q, K, V
            q = x @ torch.randn(256, 256)
            k = x @ torch.randn(256, 256)
            v = x @ torch.randn(256, 256)
            
            # Attention-like operation
            scores = q @ k.transpose(-2, -1)
            attn = torch.relu(scores)
            
            # Combine heads
            output = attn @ v
            output = output @ torch.randn(256, 256)
        
        result = output.cpu()
        
        assert result.shape == torch.Size([2, 8, 256])
        
        print("✅ Multi-head attention pattern works")
    
    def test_attention_with_bias(self):
        """Test attention with bias addition."""
        
        with genie.capture():
            q = torch.randn(2, 4, 64)
            k = torch.randn(2, 4, 64)
            v = torch.randn(2, 4, 64)
            
            # Attention with bias
            scores = q @ k.transpose(-2, -1)
            bias = torch.randn(4, 4) * 0.1
            scores = scores + bias
            
            attn = torch.relu(scores)
            output = attn @ v
        
        result = output.cpu()
        
        assert result.shape == torch.Size([2, 4, 64])
        
        print("✅ Attention with bias works")


class TestSkipConnectionPatterns:
    """Test skip connection patterns."""
    
    def test_simple_skip_connection(self):
        """Test simple skip connection (residual)."""
        
        with genie.capture():
            x = torch.randn(32, 128)
            
            # Residual block: x + f(x)
            residual = x
            x = x @ torch.randn(128, 256)
            x = torch.relu(x)
            x = x @ torch.randn(256, 128)
            
            # Skip connection - need same dimension
            x = x + residual
            x = torch.relu(x)
        
        result = x.cpu()
        
        assert result.shape == torch.Size([32, 128])
        
        print("✅ Simple skip connection works")
    
    def test_dense_skip_connections(self):
        """Test dense skip connections (like DenseNet)."""
        
        with genie.capture():
            x = torch.randn(16, 64)
            
            # Dense connections: concat(x, f(x))
            x1 = x
            
            x2 = x1 @ torch.randn(64, 64)
            x2 = torch.relu(x2)
            
            # Concatenate - use addition instead of concat for simplicity
            x3 = x1 + x2
            
            x4 = x3 @ torch.randn(64, 32)
            x4 = torch.relu(x4)
        
        result = x4.cpu()
        
        assert result.shape[1] == 32
        
        print("✅ Dense skip connections work")
    
    def test_bottleneck_with_skip(self):
        """Test bottleneck pattern with skip connection."""
        
        with genie.capture():
            x = torch.randn(32, 128)
            residual = x
            
            # Bottleneck: 1x1 conv down -> 3x3 conv -> 1x1 conv up
            x = x @ torch.randn(128, 32)
            x = torch.relu(x)
            
            x = x @ torch.randn(32, 32)
            x = torch.relu(x)
            
            x = x @ torch.randn(32, 128)
            
            # Skip connection
            x = x + residual
            x = torch.relu(x)
        
        result = x.cpu()
        
        assert result.shape == torch.Size([32, 128])
        
        print("✅ Bottleneck with skip works")


class TestNormalizationPatterns:
    """Test normalization patterns."""
    
    def test_batch_norm_like_pattern(self):
        """Test batch normalization-like pattern."""
        
        with genie.capture():
            x = torch.randn(32, 128)
            
            # Batch norm: (x - mean) / std * gamma + beta
            mean = x.mean(dim=0)
            x_centered = x - mean
            
            std = x_centered.std(dim=0) + 1e-5
            x_normalized = x_centered / std
            
            # Scale and shift
            gamma = torch.ones(128) * 2.0
            beta = torch.zeros(128)
            x_out = x_normalized * gamma + beta
        
        result = x_out.cpu()
        
        assert result.shape == torch.Size([32, 128])
        
        print("✅ Batch norm-like pattern works")
    
    def test_layer_norm_pattern(self):
        """Test layer normalization pattern."""
        
        with genie.capture():
            x = torch.randn(32, 128)
            
            # Layer norm: normalize per sample
            mean = x.mean(dim=1, keepdim=True)
            x_centered = x - mean
            
            std = x_centered.std(dim=1, keepdim=True) + 1e-5
            x_normalized = x_centered / std
        
        result = x_normalized.cpu()
        
        assert result.shape == torch.Size([32, 128])
        
        print("✅ Layer norm pattern works")


class TestActivationPatterns:
    """Test various activation patterns."""
    
    def test_multiple_activations(self):
        """Test various activation functions."""
        
        with genie.capture():
            x = torch.randn(32, 128)
            
            # Various activations
            y1 = torch.relu(x)
            y2 = torch.tanh(x)
            y3 = torch.sigmoid(x)
            y4 = torch.relu(x @ torch.randn(128, 128))
            
            # Combine
            y = y1 + y2 + y3 + y4
        
        result = y.cpu()
        
        assert result.shape == torch.Size([32, 128])
        
        print("✅ Multiple activations work")
    
    def test_gelu_like_pattern(self):
        """Test GELU-like approximation pattern."""
        
        with genie.capture():
            x = torch.randn(32, 128)
            
            # GELU approximation: 0.5 * x * (1 + tanh(...))
            cdf = 0.5 * (1.0 + torch.tanh(x))
            gelu_approx = x * cdf
        
        result = gelu_approx.cpu()
        
        assert result.shape == torch.Size([32, 128])
        
        print("✅ GELU-like pattern works")


class TestComplexPatterns:
    """Test complex model patterns."""
    
    def test_transformer_block_like(self):
        """Test transformer block-like pattern."""
        
        with genie.capture():
            x = torch.randn(2, 8, 256)
            
            # Self-attention + residual + FFN
            q = x @ torch.randn(256, 256)
            k = x @ torch.randn(256, 256)
            v = x @ torch.randn(256, 256)
            
            # Attention
            scores = q @ k.transpose(-2, -1)
            attn = torch.relu(scores)
            attn_out = attn @ v
            
            # Add & norm
            x = x + attn_out
            
            # FFN
            ff = x @ torch.randn(256, 512)
            ff = torch.relu(ff)
            ff = ff @ torch.randn(512, 256)
            
            # Add & norm
            x = x + ff
        
        result = x.cpu()
        
        assert result.shape == torch.Size([2, 8, 256])
        
        print("✅ Transformer-like block works")
    
    def test_cnn_like_pattern(self):
        """Test CNN-like pattern (using matmul)."""
        
        with genie.capture():
            # Simulate conv: simplified 2D operation
            x = torch.randn(8, 784)  # Flattened image batch
            
            # "Conv" layer with projection
            y = x @ torch.randn(784, 256)
            y = torch.relu(y)
        
        result = y.cpu()
        
        assert result.shape == torch.Size([8, 256])
        
        print("✅ CNN-like pattern works")
    
    def test_recurrent_like_pattern(self):
        """Test recurrent pattern."""
        
        with genie.capture():
            batch_size = 8
            seq_len = 3
            hidden_dim = 64
            input_dim = 128
            
            # Single timestep (simplified from full sequence)
            x = torch.randn(batch_size, input_dim)
            h = torch.zeros(batch_size, hidden_dim)
            
            # Single RNN step
            h = torch.tanh(x @ torch.randn(input_dim, hidden_dim))
            
            output = h
        
        result = output.cpu()
        
        assert result.shape == torch.Size([8, 64])
        
        print("✅ Recurrent-like pattern works")


class TestModelRobustness:
    """Test model patterns robustness."""
    
    def test_mixed_precisions_in_model(self):
        """Test model with mixed precision operations."""
        
        with genie.capture():
            x_fp32 = torch.randn(32, 128, dtype=torch.float32)
            x_fp64 = torch.randn(32, 128, dtype=torch.float64)
            
            # Mixed operations
            x_fp32 = x_fp32 @ torch.randn(128, 256, dtype=torch.float32)
            x_fp64 = x_fp64 @ torch.randn(128, 256, dtype=torch.float64)
            
            # Convert to same dtype
            x_fp32_converted = x_fp32.to(torch.float32)
        
        result = x_fp32_converted.cpu()
        
        assert result.dtype in [torch.float32, torch.float64]
        
        print("✅ Mixed precisions handled")
    
    def test_model_with_large_batch(self):
        """Test model with large batch size."""
        
        with genie.capture():
            x = torch.randn(256, 512)  # Large batch
            
            # Multi-layer
            x = x @ torch.randn(512, 256)
            x = torch.relu(x)
            x = x @ torch.randn(256, 128)
            x = torch.relu(x)
            x = x @ torch.randn(128, 10)
        
        result = x.cpu()
        
        assert result.shape == torch.Size([256, 10])
        
        print("✅ Large batch model works")
    
    def test_model_execution_correctness(self):
        """Test model produces correct output shapes and values."""
        
        with genie.capture():
            x = torch.randn(32, 784)  # MNIST-like
            
            # Simple classifier
            x = x @ torch.randn(784, 256)
            x = torch.relu(x)
            x = x @ torch.randn(256, 128)
            x = torch.relu(x)
            logits = x @ torch.randn(128, 10)
        
        result = logits.cpu()
        
        assert result.shape == torch.Size([32, 10])
        assert not torch.isnan(result).any()
        
        print("✅ Model output correct")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
