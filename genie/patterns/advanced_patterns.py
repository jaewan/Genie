"""Advanced pattern implementations using NetworkX for better accuracy and performance."""

from __future__ import annotations

from typing import List, Optional, Dict, Set
import networkx as nx

from genie.core.graph import ComputationGraph
from .base import PatternMatch, PatternPlugin
from genie.semantic.graph_utils import (
    graph_to_networkx, find_attention_pattern, find_conv_activation_pattern,
    find_mlp_pattern, find_embedding_pattern, analyze_graph_complexity,
    track_performance
)
from genie.core.graph import GraphBuilder
from .fx_patterns import (
    find_attention_pattern_fx, find_conv_activation_pattern_fx, find_mlp_pattern_fx
)


class AdvancedLLMPattern(PatternPlugin):
    """Advanced LLM pattern detection using NetworkX subgraph matching."""

    @property
    def name(self) -> str:
        return "llm"

    @track_performance
    def match(self, graph: ComputationGraph) -> Optional[PatternMatch]:
        """Detect LLM patterns using sophisticated graph analysis."""
        # Prefer FX-based matching when available
        fx_graph = GraphBuilder.current().get_fx_graph()
        if fx_graph is not None:
            fx_match = find_attention_pattern_fx(GraphBuilder.current()._fx_graph.module) if hasattr(GraphBuilder.current()._fx_graph, 'module') else None
            fx_match = find_attention_pattern_fx(GraphBuilder.current()._fx_graph) if fx_match is None else fx_match
            if fx_match is not None:
                return PatternMatch(
                    pattern_name=self.name,
                    confidence=0.9,
                    matched_nodes=[n.name for n in fx_match.nodes]
                )

        G = graph_to_networkx(graph)
        
        # Find attention patterns
        attention_matches = find_attention_pattern(G)
        
        # Find MLP patterns (common in transformers)
        mlp_matches = find_mlp_pattern(G)
        
        # Analyze overall graph characteristics
        complexity = analyze_graph_complexity(G)
        
        # Scoring based on pattern density and characteristics
        attention_score = min(len(attention_matches) * 0.6, 1.0)
        mlp_score = min(len(mlp_matches) * 0.2, 0.8)
        complexity_score = complexity["compute_intensity"] * 0.5
        
        total_score = attention_score + mlp_score + complexity_score
        
        if total_score > 0.8 or len(attention_matches) >= 1:
            confidence = min(total_score, 0.95)
            if len(attention_matches) >= 1 and confidence < 0.85:
                confidence = 0.88
            matched_nodes = []
            
            # Collect matched node IDs
            for match in attention_matches[:3]:  # Limit to top 3
                matched_nodes.extend(match.values())
            for match in mlp_matches[:2]:  # Limit to top 2
                matched_nodes.extend(match.values())
            
            return PatternMatch(
                pattern_name=self.name,
                confidence=confidence,
                matched_nodes=list(set(matched_nodes))  # Remove duplicates
            )
        
        # Fallback for simpler transformer patterns
        if attention_matches or (len(mlp_matches) >= 2):
            return PatternMatch(
                pattern_name=self.name,
                confidence=0.7,
                matched_nodes=[list(match.values())[0] for match in attention_matches[:1]]
            )
        
        return None


class AdvancedVisionPattern(PatternPlugin):
    """Advanced vision pattern detection for CNNs and Vision Transformers."""

    @property
    def name(self) -> str:
        return "vision"

    @track_performance
    def match(self, graph: ComputationGraph) -> Optional[PatternMatch]:
        """Detect vision patterns including CNNs and ViTs."""
        fx_graph = GraphBuilder.current().get_fx_graph()
        if fx_graph is not None:
            fx_match = find_conv_activation_pattern_fx(fx_graph)
            if fx_match is not None:
                return PatternMatch(
                    pattern_name=self.name,
                    confidence=0.85,
                    matched_nodes=[n.name for n in fx_match.nodes]
                )

        G = graph_to_networkx(graph)
        
        # Find convolutional patterns
        conv_matches = find_conv_activation_pattern(G)
        
        # Find attention patterns (for Vision Transformers)
        attention_matches = find_attention_pattern(G)
        
        # Analyze for vision-specific operations
        vision_ops = {"aten::conv2d", "aten::conv1d", "aten::conv3d", 
                     "aten::max_pool2d", "aten::avg_pool2d", "aten::adaptive_avg_pool2d"}
        
        vision_node_count = sum(1 for _, data in G.nodes(data=True) 
                               if data.get('operation') in vision_ops)
        
        total_nodes = len(G.nodes())
        vision_ratio = vision_node_count / total_nodes if total_nodes > 0 else 0
        
        # Scoring
        conv_score = min(len(conv_matches) * 0.4, 1.0)
        vision_ops_score = vision_ratio * 0.8
        attention_score = min(len(attention_matches) * 0.2, 0.6)  # ViT component
        
        total_score = conv_score + vision_ops_score + attention_score
        
        if total_score > 0.8:
            confidence = min(total_score, 0.95)
            matched_nodes = []
            
            # Collect matched nodes
            for match in conv_matches[:5]:
                matched_nodes.extend(match.values())
            for match in attention_matches[:2]:
                matched_nodes.extend(match.values())
            
            return PatternMatch(
                pattern_name=self.name,
                confidence=confidence,
                matched_nodes=list(set(matched_nodes))
            )
        
        # Fallback for basic CNN patterns
        if conv_matches or vision_ratio > 0.3:
            return PatternMatch(
                pattern_name=self.name,
                confidence=0.72,
                matched_nodes=[list(match.values())[0] for match in conv_matches[:1]]
            )
        
        return None


class RecSysPattern(PatternPlugin):
    """Recommendation system pattern detection."""

    @property
    def name(self) -> str:
        return "recsys"

    @track_performance
    def match(self, graph: ComputationGraph) -> Optional[PatternMatch]:
        """Detect RecSys patterns: embeddings + MLPs."""
        fx_graph = GraphBuilder.current().get_fx_graph()
        if fx_graph is not None:
            fx_match = find_mlp_pattern_fx(fx_graph)
            if fx_match is not None:
                return PatternMatch(
                    pattern_name=self.name,
                    confidence=0.8,
                    matched_nodes=[n.name for n in fx_match.nodes]
                )

        G = graph_to_networkx(graph)
        
        # Find embedding patterns
        embedding_matches = find_embedding_pattern(G)
        
        # Find MLP patterns
        mlp_matches = find_mlp_pattern(G)
        
        # Look for RecSys-specific operations
        recsys_ops = {"aten::embedding", "aten::embedding_bag", "aten::gather", 
                     "aten::index_select", "aten::cat"}
        
        recsys_node_count = sum(1 for _, data in G.nodes(data=True) 
                               if data.get('operation') in recsys_ops)
        
        total_nodes = len(G.nodes())
        recsys_ratio = recsys_node_count / total_nodes if total_nodes > 0 else 0
        
        # Scoring
        embedding_score = min(len(embedding_matches) * 0.5, 1.0)
        mlp_score = min(len(mlp_matches) * 0.3, 0.8)
        recsys_ops_score = recsys_ratio * 0.7
        
        total_score = embedding_score + mlp_score + recsys_ops_score
        
        if total_score > 0.7:
            confidence = min(total_score, 0.95)
            matched_nodes = []
            
            # Collect matched nodes
            for match in embedding_matches[:3]:
                matched_nodes.extend(match.values())
            for match in mlp_matches[:2]:
                matched_nodes.extend(match.values())
            
            return PatternMatch(
                pattern_name=self.name,
                confidence=confidence,
                matched_nodes=list(set(matched_nodes))
            )
        
        return None


class MultiModalPattern(PatternPlugin):
    """Multi-modal pattern detection for models combining vision and language."""

    @property
    def name(self) -> str:
        return "multimodal"

    @track_performance
    def match(self, graph: ComputationGraph) -> Optional[PatternMatch]:
        """Detect multi-modal patterns combining vision and language components."""
        # Multi-modal uses both FX and NetworkX paths
        fx_graph = GraphBuilder.current().get_fx_graph()
        if fx_graph is not None:
            attn_fx = find_attention_pattern_fx(fx_graph)
            conv_fx = find_conv_activation_pattern_fx(fx_graph)
            if attn_fx is not None and conv_fx is not None:
                matched_nodes = [n.name for n in (attn_fx.nodes + conv_fx.nodes)]
                return PatternMatch(
                    pattern_name=self.name,
                    confidence=0.85,
                    matched_nodes=list(set(matched_nodes))
                )

        G = graph_to_networkx(graph)
        
        # Find vision patterns
        conv_matches = find_conv_activation_pattern(G)
        
        # Find language/attention patterns
        attention_matches = find_attention_pattern(G)
        
        # Find MLP patterns (fusion layers)
        mlp_matches = find_mlp_pattern(G)
        
        # Look for multi-modal specific operations
        vision_ops = {"aten::conv2d", "aten::max_pool2d", "aten::avg_pool2d"}
        language_ops = {"aten::embedding", "aten::linear"}
        fusion_ops = {"aten::cat", "aten::add", "aten::mul"}  # Common fusion operations
        
        vision_count = sum(1 for _, data in G.nodes(data=True) 
                          if data.get('operation') in vision_ops)
        language_count = sum(1 for _, data in G.nodes(data=True) 
                            if data.get('operation') in language_ops)
        fusion_count = sum(1 for _, data in G.nodes(data=True) 
                          if data.get('operation') in fusion_ops)
        
        total_nodes = len(G.nodes())
        if total_nodes == 0:
            return None
        
        # Multi-modal requires both vision and language components
        has_vision = (vision_count > 0 or len(conv_matches) > 0)
        has_language = (language_count > 0 or len(attention_matches) > 0)
        has_fusion = fusion_count > 0
        
        if has_vision and has_language:
            # Scoring based on component balance and fusion
            vision_score = min(len(conv_matches) * 0.3, 0.8)
            language_score = min(len(attention_matches) * 0.3, 0.8)
            fusion_score = min(fusion_count / total_nodes, 0.4)
            mlp_score = min(len(mlp_matches) * 0.2, 0.6)
            
            total_score = vision_score + language_score + fusion_score + mlp_score
            
            if total_score > 0.7:
                confidence = min(total_score, 0.95)
                matched_nodes = []
                
                # Collect representative nodes from each modality
                for match in conv_matches[:2]:
                    matched_nodes.extend(match.values())
                for match in attention_matches[:2]:
                    matched_nodes.extend(match.values())
                for match in mlp_matches[:1]:
                    matched_nodes.extend(match.values())
                
                return PatternMatch(
                    pattern_name=self.name,
                    confidence=confidence,
                    matched_nodes=list(set(matched_nodes))
                )
        
        return None


# Performance tracking utilities
def get_pattern_performance_stats(pattern_class) -> Dict[str, float]:
    """Get performance statistics for a pattern class."""
    if hasattr(pattern_class.match, '_performance_stats'):
        stats = pattern_class.match._performance_stats
        return {
            "avg_latency": sum(stats) / len(stats),
            "max_latency": max(stats),
            "min_latency": min(stats),
            "call_count": len(stats)
        }
    return {"avg_latency": 0.0, "max_latency": 0.0, "min_latency": 0.0, "call_count": 0}
