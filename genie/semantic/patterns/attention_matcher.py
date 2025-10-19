"""
Matchers for attention and convolution patterns.

Detects patterns like:
- Multi-head attention: Q@K.T softmax @ V
- Query-Key-Value projections
- Convolutional operations
"""

import logging
from typing import List, Optional
from .base_matcher import PatternMatcher, Pattern

logger = logging.getLogger(__name__)


class AttentionMatcher(PatternMatcher):
    """Detects attention patterns in computation graphs."""
    
    @property
    def pattern_name(self) -> str:
        return "attention"
    
    def match(self, graph) -> List[Pattern]:
        """
        Detect attention patterns in graph.
        
        Key indicators:
        1. Batch matrix multiplication (Q @ K.T)
        2. Softmax normalization
        3. Weighted sum (scores @ V)
        4. Linear projections (QKV)
        """
        patterns = []
        nodes = list(graph.nodes())
        
        # Look for softmax operations (characteristic of attention)
        for i, node in enumerate(nodes):
            if self._is_softmax_like(node):
                # Found potential attention - backtrack to find Q@K.T pattern
                matmul_nodes = self._find_matmul_chain(node, nodes, i)
                if matmul_nodes:
                    # Found attention pattern
                    pattern = self._extract_attention_pattern(matmul_nodes, node)
                    if pattern:
                        patterns.append(pattern)
        
        return patterns
    
    def _is_softmax_like(self, node) -> bool:
        """Check if node is softmax or related normalization."""
        op = node.operation.lower()
        return any(x in op for x in ['softmax', 'log_softmax', 'exp', 'normalize'])
    
    def _find_matmul_chain(self, softmax_node, nodes, idx) -> List:
        """Trace back from softmax to find Q@K.T @ V pattern."""
        matmul_nodes = []

        # Look at softmax inputs (should be Q@K.T matmul)
        for inp_node in softmax_node.inputs:
            if self._is_matmul_like(inp_node):
                matmul_nodes.append(inp_node)

        # Look at softmax consumers (should be scores@V matmul)
        # Note: This requires get_consumers() method in GraphNode interface
        if hasattr(softmax_node, 'get_consumers'):
            for consumer in softmax_node.get_consumers():
                if self._is_matmul_like(consumer):
                    matmul_nodes.append(consumer)

        return matmul_nodes

    def _is_matmul_like(self, node) -> bool:
        """Check if node is matrix multiplication."""
        op = node.operation.lower()
        return 'matmul' in op or 'mm' in op or 'bmm' in op
    
    def _extract_attention_pattern(self, matmul_nodes, softmax_node) -> Optional[Pattern]:
        """Extract attention pattern with metadata."""
        if not matmul_nodes:
            return None
            
        # Determine attention type
        attention_type = self._classify_attention(matmul_nodes)
        
        # All nodes involved in attention
        all_nodes = matmul_nodes + [softmax_node]
        
        return Pattern(
            name="attention",
            nodes=all_nodes,
            metadata={
                'semantic_role': 'attention',
                'attention_type': attention_type,  # 'self' or 'cross'
                'modality': self._infer_modality(matmul_nodes),
                'parallelizable': True,
                'memory_intensive': True,
                'compute_bound': True,
            }
        )
    
    def _classify_attention(self, matmul_nodes) -> str:
        """Determine if self-attention or cross-attention."""
        # Cross-attention if Q and K have different sources
        # This would check metadata on input nodes
        return "self_attention"  # Simplified
    
    def _infer_modality(self, nodes) -> str:
        """Infer if this is vision, text, or multimodal attention."""
        # Check metadata from preceding operations
        return "text"  # Simplified


class ConvolutionMatcher(PatternMatcher):
    """Detects convolutional patterns."""
    
    @property
    def pattern_name(self) -> str:
        return "convolution"
    
    def match(self, graph) -> List[Pattern]:
        """Detect convolutional patterns."""
        patterns = []
        
        for node in graph.nodes():
            if self._is_conv_like(node):
                pattern = self._extract_conv_pattern(node)
                patterns.append(pattern)
        
        return patterns
    
    def _is_conv_like(self, node) -> bool:
        """Check if node is convolution operation."""
        op = node.operation.lower()
        return any(x in op for x in ['conv', 'convolution', 'conv1d', 'conv2d', 'conv3d'])
    
    def _extract_conv_pattern(self, conv_node) -> Pattern:
        """Extract convolution pattern."""
        return Pattern(
            name="convolution",
            nodes=[conv_node],
            metadata={
                'semantic_role': 'convolution',
                'modality': 'vision',
                'can_fuse_bn': self._can_fuse_with_bn(conv_node),
                'can_pipeline': True,
                'memory_footprint': 'high',
                'optimization_hint': 'fusion_opportunity',
            }
        )
    
    def _can_fuse_with_bn(self, conv_node) -> bool:
        """Check if conv can be fused with BatchNorm."""
        # Check if next operation is BatchNorm
        return False  # Simplified


class KVCacheMatcher(PatternMatcher):
    """Detects KV cache accumulation patterns (LLM decode phase)."""
    
    @property
    def pattern_name(self) -> str:
        return "kv_cache"
    
    def match(self, graph) -> List[Pattern]:
        """
        Detect KV cache patterns.
        
        Signature:
            cache_t+1 = torch.cat([cache_t, new_kv], dim=seq_dim)
        
        Where cache feeds back into next iteration.
        """
        patterns = []
        
        for node in graph.nodes():
            if self._is_cat_like(node):
                if self._is_recurrent(node, graph):
                    pattern = self._extract_kv_cache_pattern(node)
                    patterns.append(pattern)
        
        return patterns
    
    def _is_cat_like(self, node) -> bool:
        """Check if node is concatenation."""
        op = node.operation.lower()
        return 'cat' in op or 'concat' in op
    
    def _is_recurrent(self, cat_node, graph) -> bool:
        """Check if output feeds back as input (recurrent pattern)."""
        if not hasattr(cat_node, 'get_consumers'):
            return False

        # Check if cat_node output is used as input to similar cat operations
        for consumer in cat_node.get_consumers():
            if (self._is_cat_like(consumer) and
                consumer.operation == cat_node.operation):
                # Found recurrent pattern
                return True

        return False
    
    def _extract_kv_cache_pattern(self, cat_node) -> Pattern:
        """Extract KV cache pattern."""
        return Pattern(
            name="kv_cache",
            nodes=[cat_node],
            metadata={
                'execution_phase': 'llm_decode',
                'residency': 'persistent_kv_cache',
                'requires_colocation': True,
                'sequential': True,
                'memory_intensive': True,
                'optimization_hint': 'colocate_with_decoder',
            }
        )
