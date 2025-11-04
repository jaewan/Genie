"""
PHASE 3.1: Decode Phase Detection for LLM Workloads

This module provides detection of the LLM decode phase by analyzing computation
graphs for characteristic patterns:

1. KV cache usage (concatenation + attention with existing cache)
2. Single token input (batch dimension is 1 or 2)
3. Attention pattern with mature cache (large K/V dimensions)
4. Reduced compute-to-memory ratio (decode is memory-bound)

Key insight: Decode phase has fundamentally different characteristics than
prefill, enabling co-location optimizations that the scheduler can use.

Expected impact: 1.01x → 5x speedup for LLM decode by keeping KV cache
and decoder layers on the same remote GPU to minimize network transfers.
"""

import torch
import torch.fx as fx
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
import logging

from ..core.types import ExecutionPhase, MemoryPattern

logger = logging.getLogger(__name__)


@dataclass
class DecodePhaseAnalysis:
    """Results of decode phase detection."""
    is_decode: bool  # Whether this is likely a decode phase workload
    confidence: float  # 0.0-1.0 confidence score
    
    # Decode phase characteristics
    kv_cache_usage: bool  # Uses KV cache for attention
    single_token_input: bool  # Processing single token
    attention_operations: int  # Number of attention ops
    kv_concat_operations: int  # Number of KV concatenations
    
    # Compute characteristics
    compute_to_memory_ratio: float  # Lower in decode than prefill
    estimated_memory_bound: float  # Percentage of time waiting for memory (>50% = memory bound)
    
    # Optimization hints
    can_colocate: bool  # Whether co-location would help
    kv_cache_size_estimate: int  # Estimated KV cache size (bytes)
    
    # Graph characteristics
    graph_depth: int  # Maximum depth of computation DAG
    num_nodes: int  # Total nodes in graph
    num_attention_heads: Optional[int] = None
    sequence_length: Optional[int] = None
    
    def __str__(self) -> str:
        """Human-readable analysis summary."""
        status = "✅ DECODE" if self.is_decode else "❌ NOT DECODE"
        return f"""
{status} (confidence: {self.confidence:.1%})
  KV Cache: {self.kv_cache_usage}
  Single Token: {self.single_token_input}
  Attention Ops: {self.attention_operations}
  KV Concats: {self.kv_concat_operations}
  Compute-to-Memory: {self.compute_to_memory_ratio:.2f}x
  Memory Bound: {self.estimated_memory_bound:.1%}
  Can Co-locate: {self.can_colocate}
  KV Cache Size: {self.kv_cache_size_estimate / 1024 / 1024:.1f}MB
  Graph Depth: {self.graph_depth}
  Nodes: {self.num_nodes}
        """


class DecodePhaseDetector:
    """
    Detects LLM decode phase in computation graphs.
    
    PHASE 3.1 Optimization Strategy:
    
    1. Analyze graph structure for decode patterns
    2. Check for KV cache usage (concatenation operations)
    3. Measure compute intensity (ops per memory access)
    4. Verify single-token or small-batch processing
    5. Estimate memory bandwidth requirements
    6. Return confidence score and co-location recommendations
    
    This enables the Phase 3.2 co-location scheduler to make informed
    decisions about keeping KV cache and decoder on same GPU.
    """
    
    def __init__(self):
        """Initialize detector with default thresholds."""
        # Detection thresholds
        self.min_confidence_for_decode = 0.7
        self.memory_bound_threshold = 0.5  # >50% waiting for memory
        self.max_single_token_batch_size = 8  # Larger batches = prefill
        
        # Sequence length heuristics
        self.min_kv_cache_length = 100  # At least 100 tokens in cache = decode
        self.max_input_length_for_decode = 10  # Single token generation
    
    def analyze_graph(self, graph_module: fx.GraphModule, 
                     input_shapes: Optional[Dict[str, torch.Size]] = None) -> DecodePhaseAnalysis:
        """
        Analyze FX graph to detect decode phase.
        
        Args:
            graph_module: The FX GraphModule to analyze
            input_shapes: Optional input tensor shapes for better analysis
            
        Returns:
            DecodePhaseAnalysis with detection results
        """
        # Collect metrics
        attention_ops = 0
        kv_concat_ops = 0
        softmax_ops = 0
        linear_ops = 0
        matmul_ops = 0
        
        nodes_by_op = {}
        graph_depth = self._compute_graph_depth(graph_module)
        
        # Scan graph for operations
        for node in graph_module.graph.nodes:
            if node.op == 'call_function':
                op_name = getattr(node.target, '__name__', str(node.target))
                
                # Track operations
                if 'matmul' in op_name or 'mm' in op_name:
                    matmul_ops += 1
                    nodes_by_op['matmul'] = nodes_by_op.get('matmul', 0) + 1
                
                if 'softmax' in op_name:
                    softmax_ops += 1
                    nodes_by_op['softmax'] = nodes_by_op.get('softmax', 0) + 1
                
                if 'linear' in op_name:
                    linear_ops += 1
                    nodes_by_op['linear'] = nodes_by_op.get('linear', 0) + 1
                
                if 'cat' in op_name or 'concat' in op_name:
                    kv_concat_ops += 1
                    nodes_by_op['concat'] = nodes_by_op.get('concat', 0) + 1
        
        num_nodes = len([n for n in graph_module.graph.nodes if n.op in ('call_function', 'call_module')])
        
        # Detect KV cache usage
        kv_cache_usage = kv_concat_ops > 0  # Concatenation ops suggest KV cache
        
        # Analyze input shapes to detect single-token processing
        single_token = False
        input_shape_analysis = self._analyze_input_shapes(graph_module, input_shapes)
        
        # Compute characteristics
        compute_to_memory_ratio = self._estimate_compute_intensity(
            num_nodes, matmul_ops, linear_ops, softmax_ops
        )
        
        # Estimate if memory-bound (decode is typically memory-bound)
        memory_bound_estimate = 1.0 - (1.0 / (1.0 + max(0, 10 - compute_to_memory_ratio)))
        
        # Determine decode phase
        is_decode, confidence = self._determine_decode_phase(
            kv_cache_usage=kv_cache_usage,
            attention_ops=matmul_ops + softmax_ops,
            concat_ops=kv_concat_ops,
            compute_ratio=compute_to_memory_ratio,
            memory_bound=memory_bound_estimate,
            input_shape_analysis=input_shape_analysis
        )
        
        # Estimate KV cache size
        kv_cache_size = self._estimate_kv_cache_size(
            graph_module, input_shape_analysis, kv_cache_usage
        )
        
        # Check if co-location would help
        can_colocate = is_decode and kv_cache_usage and (kv_cache_size > 10 * 1024 * 1024)  # >10MB
        
        return DecodePhaseAnalysis(
            is_decode=is_decode,
            confidence=confidence,
            kv_cache_usage=kv_cache_usage,
            single_token_input=input_shape_analysis['single_token'],
            attention_operations=matmul_ops + softmax_ops,
            kv_concat_operations=kv_concat_ops,
            compute_to_memory_ratio=compute_to_memory_ratio,
            estimated_memory_bound=memory_bound_estimate,
            can_colocate=can_colocate,
            kv_cache_size_estimate=kv_cache_size,
            graph_depth=graph_depth,
            num_nodes=num_nodes,
            num_attention_heads=input_shape_analysis.get('num_attention_heads'),
            sequence_length=input_shape_analysis.get('sequence_length')
        )
    
    def _compute_graph_depth(self, graph_module: fx.GraphModule) -> int:
        """Compute maximum depth of computation DAG."""
        node_depth = {}
        
        for node in graph_module.graph.nodes:
            if node.op in ('placeholder', 'get_attr'):
                node_depth[node] = 0
            else:
                # Depth is 1 + max depth of inputs
                input_depths = []
                for arg in node.all_input_nodes:
                    input_depths.append(node_depth.get(arg, 0))
                
                node_depth[node] = (max(input_depths) if input_depths else 0) + 1
        
        return max(node_depth.values()) if node_depth else 0
    
    def _analyze_input_shapes(self, graph_module: fx.GraphModule,
                             input_shapes: Optional[Dict[str, torch.Size]]) -> Dict[str, Any]:
        """Analyze input shapes to detect single-token processing."""
        analysis = {
            'single_token': False,
            'batch_size': None,
            'sequence_length': None,
            'num_attention_heads': None,
            'hidden_size': None,
        }
        
        # Try to infer from placeholders
        try:
            for node in graph_module.graph.nodes:
                if node.op == 'placeholder':
                    # Get shape from metadata or input_shapes
                    if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
                        shape = node.meta['tensor_meta'].shape
                    elif input_shapes and node.name in input_shapes:
                        shape = input_shapes[node.name]
                    else:
                        continue
                    
                    # Check for single-token pattern (seq_len=1 or batch=1)
                    if len(shape) >= 2:
                        batch_size = shape[0]
                        seq_len = shape[1] if len(shape) > 1 else 1
                        
                        # Single token: batch size <= 8, sequence length = 1
                        if seq_len == 1 and batch_size <= 8:
                            analysis['single_token'] = True
                            analysis['batch_size'] = batch_size
                            analysis['sequence_length'] = 1
                            break
                        
                        # For embedding-like shapes [batch, hidden]
                        if len(shape) == 2 and batch_size <= 8 and seq_len > 100:
                            # Likely KV cache: [seq_len, hidden]
                            analysis['sequence_length'] = seq_len
        except Exception as e:
            logger.debug(f"Could not infer shapes: {e}")
        
        return analysis
    
    def _estimate_compute_intensity(self, num_nodes: int, matmul_ops: int,
                                    linear_ops: int, softmax_ops: int) -> float:
        """
        Estimate compute intensity (ops per memory access).
        
        Lower ratio = more memory-bound (typical for decode)
        Higher ratio = more compute-bound (typical for prefill)
        """
        # Matrix multiplications are compute-heavy
        total_ops = max(1, matmul_ops * 10 + linear_ops * 5 + softmax_ops * 2 + (num_nodes - matmul_ops - linear_ops))
        
        # Memory accesses (rough estimate)
        memory_accesses = num_nodes
        
        compute_intensity = total_ops / max(1, memory_accesses)
        return compute_intensity
    
    def _determine_decode_phase(self,
                               kv_cache_usage: bool,
                               attention_ops: int,
                               concat_ops: int,
                               compute_ratio: float,
                               memory_bound: float,
                               input_shape_analysis: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Determine if this is a decode phase workload.
        
        Returns: (is_decode, confidence_score)
        """
        score = 0.0
        
        # Factor 1: KV cache usage (strong indicator)
        if kv_cache_usage and concat_ops > 0:
            score += 0.4  # Strong indicator
        
        # Factor 2: Single token input (strong indicator)
        if input_shape_analysis['single_token']:
            score += 0.3  # Strong indicator
        
        # Factor 3: Attention operations present (moderate indicator)
        if attention_ops > 0:
            score += 0.15
        
        # Factor 4: Memory bound (decode is memory-bound, prefill is compute-bound)
        if memory_bound > 0.5:
            score += 0.15
        
        # Final decision
        is_decode = score >= self.min_confidence_for_decode
        confidence = min(score, 1.0)
        
        return is_decode, confidence
    
    def _estimate_kv_cache_size(self, graph_module: fx.GraphModule,
                               input_shape_analysis: Dict[str, Any],
                               kv_cache_usage: bool) -> int:
        """Estimate KV cache size in bytes."""
        if not kv_cache_usage:
            return 0
        
        try:
            # Typical KV cache: [seq_len, num_heads, head_dim]
            # For LLMs: seq_len ~1k-4k, num_heads ~8-32, head_dim ~64-128
            
            # Conservative estimate
            seq_len = input_shape_analysis.get('sequence_length', 1024)
            num_heads = input_shape_analysis.get('num_attention_heads', 16)
            head_dim = 64  # Common default
            
            # KV cache is 2x (K and V), 2 bytes per value (fp16)
            kv_cache_size = seq_len * num_heads * head_dim * 2 * 2
            
            return kv_cache_size
        except Exception:
            return 0
    
    def can_benefit_from_colocation(self, analysis: DecodePhaseAnalysis) -> bool:
        """
        Determine if co-location optimization would be beneficial.
        
        Co-location is beneficial when:
        1. Decode phase detected with high confidence
        2. KV cache is significant (>10MB)
        3. Network is enabled (multi-device setup)
        """
        return (analysis.is_decode and 
                analysis.confidence > 0.75 and 
                analysis.kv_cache_size_estimate > 10 * 1024 * 1024)


# Global instance for convenience
_global_detector: Optional[DecodePhaseDetector] = None


def get_decode_detector() -> DecodePhaseDetector:
    """Get the global decode phase detector instance."""
    global _global_detector
    if _global_detector is None:
        _global_detector = DecodePhaseDetector()
    return _global_detector


def detect_decode_phase(graph_module: fx.GraphModule,
                       input_shapes: Optional[Dict[str, torch.Size]] = None) -> DecodePhaseAnalysis:
    """
    Convenience function to detect decode phase.
    
    Args:
        graph_module: The FX GraphModule to analyze
        input_shapes: Optional input tensor shapes
        
    Returns:
        DecodePhaseAnalysis with detection results
    """
    detector = get_decode_detector()
    return detector.analyze_graph(graph_module, input_shapes)
