"""Pattern Definition DSL for Genie.

This module provides a declarative DSL for defining semantic patterns
that can be recognized in computation graphs using TorchDynamo.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any, Tuple
import torch
import torch.fx as fx
from torch.fx import subgraph_rewriter
from enum import Enum

from ..core.semantic_metadata import ExecutionPhase, SemanticMetadata


class PatternType(Enum):
    """Types of patterns that can be recognized."""
    ATTENTION = "attention"
    CONVOLUTION = "convolution"
    LINEAR_LAYER = "linear_layer"
    NORMALIZATION = "normalization"
    ACTIVATION = "activation"
    POOLING = "pooling"
    EMBEDDING = "embedding"
    FUSION = "fusion"
    CUSTOM = "custom"


@dataclass
class PatternDescriptor:
    """Describes a semantic pattern to be recognized."""
    name: str
    type: PatternType
    pattern_fn: Callable  # Function defining the pattern
    replacement_fn: Optional[Callable] = None  # Optional replacement
    metadata: Dict[str, Any] = None
    confidence_threshold: float = 0.8
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SemanticOp:
    """Semantic operations that replace recognized patterns."""
    
    @staticmethod
    def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                  phase: str = "unknown", num_heads: Optional[int] = None) -> torch.Tensor:
        """Semantic attention operation with metadata."""
        # Simplified without shape access for FX compatibility
        scores = torch.matmul(q, k.transpose(-2, -1))
        # Can't access q.shape in FX patterns
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)
        
        # Metadata would be attached during pattern replacement
        return output
    
    @staticmethod
    def layer_norm(x: torch.Tensor, normalized_shape: List[int], 
                   weight: Optional[torch.Tensor] = None,
                   bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Semantic layer normalization."""
        output = torch.nn.functional.layer_norm(x, normalized_shape, weight, bias)
        
        if hasattr(output, 'meta'):
            output.meta['semantic_op'] = 'layer_norm'
        
        return output
    
    @staticmethod
    def gelu_activation(x: torch.Tensor) -> torch.Tensor:
        """Semantic GELU activation."""
        output = torch.nn.functional.gelu(x)
        
        if hasattr(output, 'meta'):
            output.meta['semantic_op'] = 'gelu_activation'
        
        return output
    
    @staticmethod
    def conv_block(x: torch.Tensor, weight: torch.Tensor, 
                   bias: Optional[torch.Tensor] = None,
                   stride: int = 1, padding: int = 0) -> torch.Tensor:
        """Semantic convolution block."""
        output = torch.nn.functional.conv2d(x, weight, bias, stride, padding)
        
        if hasattr(output, 'meta'):
            output.meta['semantic_op'] = 'conv_block'
            output.meta['stride'] = stride
            output.meta['padding'] = padding
        
        return output
    
    @staticmethod
    def cross_modal_fusion(vision_features: torch.Tensor, 
                          text_features: torch.Tensor,
                          fusion_type: str = "concat") -> torch.Tensor:
        """Semantic cross-modal fusion operation."""
        if fusion_type == "concat":
            output = torch.cat([vision_features, text_features], dim=-1)
        elif fusion_type == "add":
            output = vision_features + text_features
        elif fusion_type == "multiply":
            output = vision_features * text_features
        else:
            output = torch.cat([vision_features, text_features], dim=-1)
        
        if hasattr(output, 'meta'):
            output.meta['semantic_op'] = 'cross_modal_fusion'
            output.meta['fusion_type'] = fusion_type
        
        return output


class PatternBuilder:
    """Builder for creating pattern descriptors."""
    
    def __init__(self, name: str):
        self.name = name
        self.type = PatternType.CUSTOM
        self.pattern_fn = None
        self.replacement_fn = None
        self.metadata = {}
        self.confidence_threshold = 0.8
    
    def with_type(self, pattern_type: PatternType) -> "PatternBuilder":
        """Set the pattern type."""
        self.type = pattern_type
        return self
    
    def with_pattern(self, pattern_fn: Callable) -> "PatternBuilder":
        """Set the pattern function."""
        self.pattern_fn = pattern_fn
        return self
    
    def with_replacement(self, replacement_fn: Callable) -> "PatternBuilder":
        """Set the replacement function."""
        self.replacement_fn = replacement_fn
        return self
    
    def with_metadata(self, **kwargs) -> "PatternBuilder":
        """Add metadata to the pattern."""
        self.metadata.update(kwargs)
        return self
    
    def with_confidence(self, threshold: float) -> "PatternBuilder":
        """Set confidence threshold."""
        self.confidence_threshold = threshold
        return self
    
    def build(self) -> PatternDescriptor:
        """Build the pattern descriptor."""
        if self.pattern_fn is None:
            raise ValueError(f"Pattern function required for {self.name}")
        
        return PatternDescriptor(
            name=self.name,
            type=self.type,
            pattern_fn=self.pattern_fn,
            replacement_fn=self.replacement_fn,
            metadata=self.metadata,
            confidence_threshold=self.confidence_threshold
        )


# Pre-defined pattern templates
class PatternTemplates:
    """Common pattern templates for quick pattern creation."""
    
    @staticmethod
    def attention_pattern() -> PatternDescriptor:
        """Standard attention pattern (Q, K, V)."""
        def pattern(q, k, v):
            # Simplified pattern without shape access (FX limitation)
            scores = torch.matmul(q, k.transpose(-2, -1))
            weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(weights, v)
            return output
        
        def replacement(q, k, v):
            # Simplified replacement
            scores = torch.matmul(q, k.transpose(-2, -1))
            weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(weights, v)
            return output
        
        return PatternBuilder("attention") \
            .with_type(PatternType.ATTENTION) \
            .with_pattern(pattern) \
            .with_replacement(replacement) \
            .with_metadata(compute_intensity=10.0, can_parallelize=True) \
            .build()
    
    @staticmethod
    def transformer_block_pattern() -> PatternDescriptor:
        """Transformer block pattern (attention + FFN)."""
        def pattern(x, q_weight, k_weight, v_weight, o_weight,
                   ffn1_weight, ffn2_weight, ln1_weight, ln2_weight):
            # Multi-head attention
            q = torch.matmul(x, q_weight)
            k = torch.matmul(x, k_weight)
            v = torch.matmul(x, v_weight)
            
            scores = torch.matmul(q, k.transpose(-2, -1))
            weights = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(weights, v)
            attn_out = torch.matmul(attn_out, o_weight)
            
            # Add & Norm
            x = torch.nn.functional.layer_norm(x + attn_out, x.shape[-1:], ln1_weight)
            
            # FFN
            ffn_out = torch.matmul(x, ffn1_weight)
            ffn_out = torch.nn.functional.gelu(ffn_out)
            ffn_out = torch.matmul(ffn_out, ffn2_weight)
            
            # Add & Norm
            output = torch.nn.functional.layer_norm(x + ffn_out, x.shape[-1:], ln2_weight)
            
            return output
        
        return PatternBuilder("transformer_block") \
            .with_type(PatternType.ATTENTION) \
            .with_pattern(pattern) \
            .with_metadata(
                compute_intensity=15.0,
                memory_pattern="streaming",
                execution_phase="prefill"
            ) \
            .build()
    
    @staticmethod
    def conv_relu_pattern() -> PatternDescriptor:
        """Convolution + ReLU pattern."""
        def pattern(x, weight, bias):
            conv = torch.nn.functional.conv2d(x, weight, bias)
            output = torch.relu(conv)
            return output
        
        return PatternBuilder("conv_relu") \
            .with_type(PatternType.CONVOLUTION) \
            .with_pattern(pattern) \
            .with_metadata(
                compute_intensity=10.0,
                memory_pattern="streaming",
                can_fuse=True
            ) \
            .build()
    
    @staticmethod
    def residual_block_pattern() -> PatternDescriptor:
        """Residual block pattern."""
        def pattern(x, conv1_weight, conv2_weight, bn1_weight, bn2_weight):
            # First conv + bn + relu
            out = torch.nn.functional.conv2d(x, conv1_weight)
            out = torch.nn.functional.batch_norm(out, None, bn1_weight)
            out = torch.relu(out)
            
            # Second conv + bn
            out = torch.nn.functional.conv2d(out, conv2_weight)
            out = torch.nn.functional.batch_norm(out, None, bn2_weight)
            
            # Residual connection
            out = out + x
            out = torch.relu(out)
            
            return out
        
        return PatternBuilder("residual_block") \
            .with_type(PatternType.CONVOLUTION) \
            .with_pattern(pattern) \
            .with_metadata(
                compute_intensity=12.0,
                memory_pattern="reused",
                skip_connection=True
            ) \
            .build()
    
    @staticmethod
    def cross_attention_pattern() -> PatternDescriptor:
        """Cross-attention pattern for multi-modal models."""
        def pattern(q_modal1, k_modal2, v_modal2):
            # Simplified without shape access
            scores = torch.matmul(q_modal1, k_modal2.transpose(-2, -1))
            weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(weights, v_modal2)
            return output
        
        def replacement(q_modal1, k_modal2, v_modal2):
            # Simplified replacement
            scores = torch.matmul(q_modal1, k_modal2.transpose(-2, -1))
            weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(weights, v_modal2)
            return output
        
        return PatternBuilder("cross_attention") \
            .with_type(PatternType.FUSION) \
            .with_pattern(pattern) \
            .with_replacement(replacement) \
            .with_metadata(
                compute_intensity=10.0,
                execution_phase="multimodal_fusion",
                is_cross_modal=True
            ) \
            .build()


def register_pattern(graph_module: fx.GraphModule, 
                    pattern: PatternDescriptor) -> int:
    """Register a pattern with the FX subgraph rewriter.
    
    Args:
        graph_module: The FX GraphModule to apply patterns to
        pattern: The pattern descriptor to register
        
    Returns:
        Number of pattern matches found and replaced
    """
    matches = subgraph_rewriter.replace_pattern(
        graph_module,
        pattern.pattern_fn,
        pattern.replacement_fn or pattern.pattern_fn
    )
    
    # Attach metadata to matched nodes
    for node in graph_module.graph.nodes:
        if hasattr(node, 'target') and hasattr(node.target, '__name__'):
            if pattern.name in node.target.__name__:
                node.meta.update(pattern.metadata)
                node.meta['pattern_name'] = pattern.name
                node.meta['pattern_type'] = pattern.type.value
    
    return len(matches) if matches else 0
