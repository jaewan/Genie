"""TorchDynamo-based pattern matching for Genie.

This module provides declarative pattern matching using TorchDynamo's
subgraph rewriter to identify and tag semantic patterns in computation graphs.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.fx as fx
from torch.fx import subgraph_rewriter
from collections import defaultdict

from .pattern_dsl import (
    PatternDescriptor, PatternBuilder, PatternTemplates,
    PatternType, SemanticOp, register_pattern
)
from ..core.types import ExecutionPhase, MemoryPattern

logger = logging.getLogger(__name__)


class DynamoPatternMatcher:
    """Pattern matcher using TorchDynamo's subgraph rewriter.
    
    This class manages pattern registration and matching for semantic
    recognition in FX graphs using declarative patterns.
    """
    
    def __init__(self, use_cache: bool = False):
        self.patterns: Dict[str, PatternDescriptor] = {}
        self.match_stats: Dict[str, int] = defaultdict(int)
        self._initialize_patterns()
        self.use_cache = use_cache  # Enable caching
        self._cache_valid = False   # Cache validity flag
        self._last_graph_hash = None  # Track graph hash for invalidation
    
    def _initialize_patterns(self):
        """Initialize default patterns."""
        # Register pre-defined patterns
        self.register_llm_patterns()
        self.register_vision_patterns()
        self.register_multimodal_patterns()
        
        logger.info(f"Initialized {len(self.patterns)} patterns")
    
    def register_llm_patterns(self):
        """Register patterns specific to LLM workloads."""
        
        # Standard attention pattern
        attention = PatternTemplates.attention_pattern()
        self.patterns["attention"] = attention
        
        # Self-attention with KV cache pattern (simplified for FX)
        def kv_cache_attention_pattern(q, k_cache, v_cache, k_new, v_new):
            # Concatenate new K, V with cache
            k = torch.cat([k_cache, k_new], dim=1)
            v = torch.cat([v_cache, v_new], dim=1)
            
            # Compute attention (simplified)
            scores = torch.matmul(q, k.transpose(-2, -1))
            weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(weights, v)
            
            return output, k, v  # Return updated cache
        
        def kv_cache_replacement(q, k_cache, v_cache, k_new, v_new):
            # Concatenate with cache
            k = torch.cat([k_cache, k_new], dim=1)
            v = torch.cat([v_cache, v_new], dim=1)
            
            # Simplified computation
            scores = torch.matmul(q, k.transpose(-2, -1))
            weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(weights, v)
            
            return output, k, v
        
        kv_pattern = PatternBuilder("kv_cache_attention") \
            .with_type(PatternType.ATTENTION) \
            .with_pattern(kv_cache_attention_pattern) \
            .with_replacement(kv_cache_replacement) \
            .with_metadata(
                kv_cache_related=True,
                execution_phase=ExecutionPhase.DECODE,
                memory_pattern=MemoryPattern.PERSISTENT,
                priority=10
            ) \
            .build()
        
        self.patterns["kv_cache_attention"] = kv_pattern
        
        # Rotary position embedding pattern
        def rope_pattern(q, k, cos, sin):
            # Simplified RoPE pattern
            q_rot = q * cos + rotate_half(q) * sin
            k_rot = k * cos + rotate_half(k) * sin
            return q_rot, k_rot
        
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat([-x2, x1], dim=-1)
        
        rope = PatternBuilder("rotary_embedding") \
            .with_type(PatternType.EMBEDDING) \
            .with_pattern(rope_pattern) \
            .with_metadata(
                execution_phase=ExecutionPhase.EMBEDDING,
                compute_intensity=2.0
            ) \
            .build()
        
        self.patterns["rotary_embedding"] = rope
        
        # Transformer block pattern
        transformer = PatternTemplates.transformer_block_pattern()
        self.patterns["transformer_block"] = transformer
        
        logger.debug(f"Registered {len([p for p in self.patterns.values() if p.type == PatternType.ATTENTION])} LLM patterns")
    
    def register_vision_patterns(self):
        """Register patterns specific to vision workloads."""
        
        # Conv + ReLU pattern
        conv_relu = PatternTemplates.conv_relu_pattern()
        self.patterns["conv_relu"] = conv_relu
        
        # Conv + BatchNorm + ReLU pattern
        def conv_bn_relu_pattern(x, conv_weight, conv_bias, bn_weight, bn_bias):
            conv = torch.nn.functional.conv2d(x, conv_weight, conv_bias)
            bn = torch.nn.functional.batch_norm(
                conv, None, bn_weight, bn_bias, training=False
            )
            output = torch.relu(bn)
            return output
        
        conv_bn_relu = PatternBuilder("conv_bn_relu") \
            .with_type(PatternType.CONVOLUTION) \
            .with_pattern(conv_bn_relu_pattern) \
            .with_metadata(
                compute_intensity=11.0,
                memory_pattern=MemoryPattern.STREAMING,
                can_fuse=True,
                execution_phase=ExecutionPhase.VISION_BACKBONE
            ) \
            .build()
        
        self.patterns["conv_bn_relu"] = conv_bn_relu
        
        # Residual block pattern
        residual = PatternTemplates.residual_block_pattern()
        self.patterns["residual_block"] = residual
        
        # Depthwise separable convolution pattern (simplified)
        def depthwise_separable_pattern(x, dw_weight, pw_weight):
            # Simplified - can't use x.shape in FX pattern
            # Just do two convolutions
            dw = torch.nn.functional.conv2d(x, dw_weight)
            output = torch.nn.functional.conv2d(dw, pw_weight)
            return output
        
        depthwise = PatternBuilder("depthwise_separable") \
            .with_type(PatternType.CONVOLUTION) \
            .with_pattern(depthwise_separable_pattern) \
            .with_metadata(
                compute_intensity=8.0,
                memory_pattern=MemoryPattern.STREAMING,
                is_mobile_friendly=True
            ) \
            .build()
        
        self.patterns["depthwise_separable"] = depthwise
        
        # Spatial pyramid pooling pattern - DISABLED due to FX limitations with loops
        # FX can't handle loops in pattern matching
        # def spp_pattern(x, pool_sizes):
        #     pools = []
        #     for size in pool_sizes:
        #         pool = torch.nn.functional.adaptive_avg_pool2d(x, size)
        #         pools.append(pool.flatten(1))
        #     output = torch.cat(pools, dim=1)
        #     return output
        # 
        # spp = PatternBuilder("spatial_pyramid_pooling") \
        #     .with_type(PatternType.POOLING) \
        #     .with_pattern(spp_pattern) \
        #     .with_metadata(
        #         compute_intensity=3.0,
        #         memory_pattern=MemoryPattern.REUSED,
        #         execution_phase=ExecutionPhase.VISION_HEAD
        #     ) \
        #     .build()
        # 
        # self.patterns["spatial_pyramid_pooling"] = spp
        
        logger.debug(f"Registered {len([p for p in self.patterns.values() if p.type == PatternType.CONVOLUTION])} vision patterns")
    
    def register_multimodal_patterns(self):
        """Register patterns specific to multi-modal workloads."""
        
        # Cross-attention pattern
        cross_attention = PatternTemplates.cross_attention_pattern()
        self.patterns["cross_attention"] = cross_attention
        
        # Vision-language fusion pattern
        def vl_fusion_pattern(vision_features, text_features, 
                            fusion_weight, fusion_bias):
            # Concatenate features
            combined = torch.cat([vision_features, text_features], dim=-1)
            # Linear fusion
            fused = torch.nn.functional.linear(combined, fusion_weight, fusion_bias)
            # Activation
            output = torch.nn.functional.gelu(fused)
            return output
        
        def vl_fusion_replacement(vision_features, text_features,
                                 fusion_weight, fusion_bias):
            # Simplified replacement without metadata assignment
            combined = torch.cat([vision_features, text_features], dim=-1)
            fused = torch.nn.functional.linear(combined, fusion_weight, fusion_bias)
            output = torch.nn.functional.gelu(fused)
            return output
        
        vl_fusion = PatternBuilder("vision_language_fusion") \
            .with_type(PatternType.FUSION) \
            .with_pattern(vl_fusion_pattern) \
            .with_replacement(vl_fusion_replacement) \
            .with_metadata(
                compute_intensity=5.0,
                memory_pattern=MemoryPattern.STREAMING,
                execution_phase=ExecutionPhase.MULTIMODAL_FUSION,
                requires_sync=True
            ) \
            .build()
        
        self.patterns["vision_language_fusion"] = vl_fusion
        
        # CLIP-style contrastive pattern
        def clip_contrastive_pattern(image_features, text_features, temperature):
            # Normalize features
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
            
            # Compute similarity
            logits = torch.matmul(image_features, text_features.t()) / temperature
            
            return logits
        
        clip = PatternBuilder("clip_contrastive") \
            .with_type(PatternType.FUSION) \
            .with_pattern(clip_contrastive_pattern) \
            .with_metadata(
                compute_intensity=4.0,
                memory_pattern=MemoryPattern.REUSED,
                is_contrastive=True,
                execution_phase=ExecutionPhase.MULTIMODAL_FUSION
            ) \
            .build()
        
        self.patterns["clip_contrastive"] = clip
        
        logger.debug(f"Registered {len([p for p in self.patterns.values() if p.type == PatternType.FUSION])} multi-modal patterns")
    
    def add_custom_pattern(self, pattern: PatternDescriptor):
        """Add a custom pattern to the matcher.
        
        Args:
            pattern: PatternDescriptor to add
        """
        self.patterns[pattern.name] = pattern
        logger.info(f"Added custom pattern: {pattern.name}")
    
    def match_patterns(self, graph_module: fx.GraphModule,
                      pattern_names: Optional[List[str]] = None) -> Dict[str, int]:
        """Match patterns in an FX graph.
        
        Args:
            graph_module: The FX GraphModule to analyze
            pattern_names: Optional list of specific patterns to match.
                         If None, matches all registered patterns.
        
        Returns:
            Dictionary mapping pattern names to match counts
        """
        patterns_to_match = pattern_names or list(self.patterns.keys())
        match_counts = {}
        
        for pattern_name in patterns_to_match:
            if pattern_name not in self.patterns:
                logger.warning(f"Pattern {pattern_name} not registered")
                continue
            
            pattern = self.patterns[pattern_name]
            
            try:
                # Apply pattern matching
                count = register_pattern(graph_module, pattern)
                match_counts[pattern_name] = count
                self.match_stats[pattern_name] += count
                
                if count > 0:
                    logger.debug(f"Found {count} matches for pattern {pattern_name}")
                    
            except Exception as e:
                logger.error(f"Error matching pattern {pattern_name}: {e}")
                match_counts[pattern_name] = 0
        
        return match_counts
    
    def analyze_graph(self, graph_module: fx.GraphModule) -> Dict[str, Any]:
        """
        Comprehensive analysis of patterns in a graph with Phase 2.3 lazy caching.
        
        PHASE 2.3 OPTIMIZATION: Cache pattern matching results based on graph hash.
        Only re-scan if graph structure changes.
        
        Args:
            graph_module: The FX GraphModule to analyze
            
        Returns:
            Analysis results including pattern matches and statistics (cached if possible)
        """
        import time
        
        # Phase 2.3: Lazy caching implementation
        if self.use_cache:
            # Compute graph hash (fingerprint of structure)
            graph_hash = _compute_graph_hash(graph_module)
            
            # Check cache (fast path, no lock needed)
            if graph_hash in _global_pattern_cache and self._cache_valid:
                with _pattern_cache_lock:
                    _pattern_cache_stats['hits'] += 1
                    _pattern_cache_stats['cache_size'] = len(_global_pattern_cache)
                
                cached_result = _global_pattern_cache[graph_hash]
                if cached_result is not None:
                    return cached_result
            
            # Cache miss: compute patterns
            with _pattern_cache_lock:
                _pattern_cache_stats['misses'] += 1
            
            start_time = time.perf_counter()
        
        # Match all patterns
        matches = self.match_patterns(graph_module)
        
        # Collect statistics
        analysis = {
            'total_patterns_matched': sum(matches.values()),
            'pattern_counts': matches,
            'pattern_types': defaultdict(int),
            'execution_phases': defaultdict(int),
            'high_priority_ops': [],
            'fusion_opportunities': [],
            'optimization_hints': []
        }
        
        # Analyze matched patterns
        for node in graph_module.graph.nodes:
            if 'pattern_name' in node.meta:
                pattern_name = node.meta['pattern_name']
                pattern = self.patterns.get(pattern_name)
                
                if pattern:
                    # Count by type
                    analysis['pattern_types'][pattern.type.value] += 1
                    
                    # Track execution phases
                    if 'execution_phase' in pattern.metadata:
                        phase = pattern.metadata['execution_phase']
                        if isinstance(phase, ExecutionPhase):
                            phase = phase.value
                        analysis['execution_phases'][phase] += 1
                    
                    # Identify high priority operations
                    if pattern.metadata.get('priority', 0) >= 8:
                        analysis['high_priority_ops'].append({
                            'node': node.name,
                            'pattern': pattern_name,
                            'priority': pattern.metadata.get('priority')
                        })
                    
                    # Identify fusion opportunities
                    if pattern.metadata.get('can_fuse', False):
                        analysis['fusion_opportunities'].append({
                            'node': node.name,
                            'pattern': pattern_name
                        })
        
        # Cache result if using cache (Phase 2.3)
        if self.use_cache:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            with _pattern_cache_lock:
                # Store in cache
                _global_pattern_cache[graph_hash] = analysis
                _pattern_cache_stats['cache_size'] = len(_global_pattern_cache)
                
                # Automatic eviction if cache too large
                if len(_global_pattern_cache) > _pattern_cache_stats['max_size']:
                    # Remove oldest 30% (FIFO)
                    keys_to_remove = list(_global_pattern_cache.keys())[:int(len(_global_pattern_cache) * 0.3)]
                    for k in keys_to_remove:
                        del _global_pattern_cache[k]
            
            # Mark cache as valid until invalidation
            self._cache_valid = True
            self._last_graph_hash = graph_hash
        
        return analysis
    
    def get_match_statistics(self) -> Dict[str, int]:
        """Get cumulative match statistics.
        
        Returns:
            Dictionary of pattern names to total match counts
        """
        return dict(self.match_stats)
    
    def reset_statistics(self):
        """Reset match statistics."""
        self.match_stats.clear()
        logger.info("Match statistics reset")

    def invalidate_cache(self):
        """Invalidate pattern matching cache (call when graph structure changes)."""
        self._cache_valid = False
        self._last_graph_hash = None


# ============================================================================
# P0 OPTIMIZATION: Lazy Pattern Matching Cache (Phase 2.3)
# ============================================================================

import hashlib
import threading
from typing import Optional, Dict, Any

# Global pattern cache statistics
_pattern_cache_stats = {
    'hits': 0,
    'misses': 0,
    'cache_size': 0,
    'max_size': 1000,
}
_pattern_cache_lock = threading.Lock()
_global_pattern_cache: Dict[str, Optional[Dict[str, Any]]] = {}


def get_pattern_matching_stats() -> Dict[str, Any]:
    """Get pattern matching cache statistics."""
    with _pattern_cache_lock:
        total = _pattern_cache_stats['hits'] + _pattern_cache_stats['misses']
        hit_rate = (_pattern_cache_stats['hits'] / total * 100) if total > 0 else 0
        return {
            'hits': _pattern_cache_stats['hits'],
            'misses': _pattern_cache_stats['misses'],
            'total': total,
            'hit_rate': f"{hit_rate:.1f}%",
            'cache_size': _pattern_cache_stats['cache_size'],
        }


def _compute_graph_hash(graph_module: fx.GraphModule) -> str:
    """
    Compute deterministic hash of graph structure.
    
    Uses operation sequence and connectivity, not values.
    Two structurally identical graphs (same ops, same connections) have same hash.
    """
    try:
        # Build hash from graph structure
        ops_str = ""
        edges_str = ""
        
        for node in graph_module.graph.nodes:
            ops_str += f"{node.op}:{node.target};"
            for arg in node.all_input_nodes:
                edges_str += f"{arg.name}->{node.name};"
        
        # Combine and hash
        structure = ops_str + edges_str
        return hashlib.md5(structure.encode()).hexdigest()[:16]
    except Exception:
        # Fallback: compute hash from module id
        return hashlib.md5(str(id(graph_module)).encode()).hexdigest()[:16]


# Global instance for convenience
_global_matcher = None


def get_pattern_matcher() -> DynamoPatternMatcher:
    """Get the global pattern matcher instance.
    
    Returns:
        Global DynamoPatternMatcher instance with lazy caching enabled
    """
    global _global_matcher
    if _global_matcher is None:
        _global_matcher = DynamoPatternMatcher(use_cache=True)
    return _global_matcher


def match_patterns_in_graph(graph_module: fx.GraphModule) -> Dict[str, Any]:
    """
    Convenience function to match patterns in a graph with lazy caching.
    
    Phase 2.3 Optimization: Caches pattern matching results based on graph structure.
    
    Args:
        graph_module: The FX GraphModule to analyze
        
    Returns:
        Analysis results from pattern matching (cached if graph structure unchanged)
    """
    matcher = get_pattern_matcher()
    return matcher.analyze_graph(graph_module)
