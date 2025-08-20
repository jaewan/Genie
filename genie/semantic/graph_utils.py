"""Graph analysis utilities using NetworkX for efficient pattern matching."""

from __future__ import annotations

import time
from functools import wraps, lru_cache
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import Counter

import networkx as nx
import logging
import os
import hashlib

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    # Fallback entropy calculation
    import math
    def entropy(counts):
        if not counts:
            return 0.0
        total = sum(counts)
        if total == 0:
            return 0.0
        return -sum((c/total) * math.log2(c/total) for c in counts if c > 0)
    
    class stats:
        entropy = staticmethod(entropy)

from genie.core.graph import ComputationGraph


logger = logging.getLogger(__name__)


def track_performance(func):
    """Decorator to track function performance for <100ms requirement."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        latency = time.perf_counter() - start
        
        # Log performance metrics (could be enhanced with proper logging)
        if hasattr(wrapper, '_performance_stats'):
            wrapper._performance_stats.append(latency)
        else:
            wrapper._performance_stats = [latency]
            
        # Feature flags
        slow_ms_env = os.getenv("GENIE_ANALYZER_SLOW_MS", "100")
        try:
            slow_threshold = float(slow_ms_env) / 1000.0
        except Exception:
            slow_threshold = 0.1

        debug_enabled = os.getenv("GENIE_ANALYZER_DEBUG", "0") == "1"

        # Warn if exceeding threshold
        if latency > slow_threshold:
            logger.warning("%s took %.1fms (>%.1fms target)", func.__name__, latency * 1000.0, slow_threshold * 1000.0)
        elif debug_enabled:
            logger.debug("%s took %.3fms", func.__name__, latency * 1000.0)
            
        return result
    return wrapper


@lru_cache(maxsize=128)
def convert_to_networkx(graph_hash: str, nodes_tuple: tuple, edges_tuple: tuple) -> nx.DiGraph:
    """Convert ComputationGraph to NetworkX DiGraph with caching.
    
    Args:
        graph_hash: Hash of the graph for cache key
        nodes_tuple: Tuple of (node_id, operation, metadata_hash) for all nodes
        edges_tuple: Tuple of all edges
    """
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for node_id, operation, metadata_hash in nodes_tuple:
        G.add_node(node_id, operation=operation, metadata_hash=metadata_hash)
    
    # Add edges
    G.add_edges_from(edges_tuple)
    
    return G


def graph_to_networkx(graph: ComputationGraph) -> nx.DiGraph:
    """Convert ComputationGraph to NetworkX DiGraph with caching."""
    # Create cache-friendly representation
    nodes_tuple = tuple(
        (node_id, node.operation, hash(str(node.metadata)))
        for node_id, node in graph.nodes.items()
    )
    edges_tuple = tuple(graph.edges)
    graph_hash = compute_graph_id_from_parts(nodes_tuple, edges_tuple)
    
    return convert_to_networkx(graph_hash, nodes_tuple, edges_tuple)


def compute_graph_id(graph: ComputationGraph) -> str:
    """Compute a stable content hash for a ComputationGraph.

    Uses SHA1 over sorted nodes (id, op, metadata) and sorted edges to ensure stability.
    """
    nodes_items = []
    for node_id in sorted(graph.nodes.keys()):
        node = graph.nodes[node_id]
        nodes_items.append((node_id, node.operation, str(sorted(node.metadata.items()))))
    edges_items = sorted((str(s), str(t)) for s, t in graph.edges)
    h = hashlib.sha1()
    for nid, op, md in nodes_items:
        h.update(nid.encode()); h.update(b"|"); h.update(op.encode()); h.update(b"|"); h.update(md.encode()); h.update(b"\n")
    h.update(b"#edges\n")
    for s, t in edges_items:
        h.update(s.encode()); h.update(b"->"); h.update(t.encode()); h.update(b"\n")
    return h.hexdigest()


def compute_graph_id_from_parts(nodes_tuple: Tuple[Tuple[str, str, int], ...], edges_tuple: Tuple[Tuple[str, str], ...]) -> str:
    """Compute a stable graph id from node/edge tuples (helper for cached conversion)."""
    h = hashlib.sha1()
    for node_id, op, mdh in sorted(nodes_tuple):
        h.update(node_id.encode()); h.update(b"|"); h.update(op.encode()); h.update(b"|"); h.update(str(mdh).encode()); h.update(b"\n")
    h.update(b"#edges\n")
    for s, t in sorted(edges_tuple):
        h.update(str(s).encode()); h.update(b"->"); h.update(str(t).encode()); h.update(b"\n")
    return h.hexdigest()


@track_performance
def analyze_operations_advanced(graph: ComputationGraph) -> Dict[str, Any]:
    """Advanced operation analysis using statistical methods."""
    ops = [node.operation for node in graph.nodes.values()]
    op_counts = Counter(ops)
    
    # Statistical analysis
    counts = list(op_counts.values())
    entropy = stats.entropy(counts) if counts else 0.0
    
    # Graph topology metrics
    G = graph_to_networkx(graph)
    
    return {
        "op_histogram": dict(op_counts),
        "num_nodes": len(graph.nodes),
        "num_edges": len(graph.edges),
        "entropy": entropy,
        "diversity": len(op_counts),
        "avg_degree": sum(dict(G.degree()).values()) / len(G.nodes()) if G.nodes() else 0,
        "is_dag": nx.is_directed_acyclic_graph(G),
        "strongly_connected_components": len(list(nx.strongly_connected_components(G))),
    }


def find_subgraph_patterns(G: nx.DiGraph, pattern_nodes: List[str], 
                          pattern_edges_idx: List[Tuple[int, int]]) -> List[Dict[str, str]]:
    """Find all instances of a pattern in the graph using subgraph isomorphism.

    Args:
        G: The main graph
        pattern_nodes: List of node operations in the pattern (by index)
        pattern_edges_idx: List of edges in the pattern as (src_index, dst_index)

    Returns:
        List of mappings from pattern nodes (p0, p1, ...) to actual node IDs
    """
    # Create pattern graph
    pattern_G = nx.DiGraph()
    for i, op in enumerate(pattern_nodes):
        pattern_G.add_node(f"p{i}", operation=op)

    for src_idx, dst_idx in pattern_edges_idx:
        pattern_G.add_edge(f"p{src_idx}", f"p{dst_idx}")
    
    # Use NetworkX subgraph isomorphism
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        G, pattern_G,
        node_match=lambda n1, n2: n1.get('operation') == n2.get('operation')
    )
    
    matches = []
    for mapping in matcher.subgraph_isomorphisms_iter():
        # Reverse mapping: pattern node -> actual node
        reverse_mapping = {v: k for k, v in mapping.items()}
        matches.append(reverse_mapping)
        
        # Limit matches to avoid performance issues
        if len(matches) >= 10:
            break
    
    return matches


def find_attention_pattern(G: nx.DiGraph) -> List[Dict[str, str]]:
    """Find attention patterns: matmul -> softmax -> matmul."""
    pattern_nodes = ["aten::matmul", "aten::softmax", "aten::matmul"]
    pattern_edges_idx = [(0, 1), (1, 2)]
    return find_subgraph_patterns(G, pattern_nodes, pattern_edges_idx)


def find_conv_activation_pattern(G: nx.DiGraph) -> List[Dict[str, str]]:
    """Find conv -> activation patterns."""
    # Try multiple activation types
    activations = ["aten::relu", "aten::sigmoid", "aten::tanh", "aten::gelu"]
    
    all_matches = []
    for activation in activations:
        pattern_nodes = ["aten::conv2d", activation]
        pattern_edges_idx = [(0, 1)]
        matches = find_subgraph_patterns(G, pattern_nodes, pattern_edges_idx)
        all_matches.extend(matches)
    
    return all_matches


def find_mlp_pattern(G: nx.DiGraph) -> List[Dict[str, str]]:
    """Find MLP patterns: linear -> activation -> linear."""
    activations = ["aten::relu", "aten::gelu", "aten::sigmoid"]
    
    all_matches = []
    for activation in activations:
        pattern_nodes = ["aten::linear", activation, "aten::linear"]
        pattern_edges_idx = [(0, 1), (1, 2)]
        matches = find_subgraph_patterns(G, pattern_nodes, pattern_edges_idx)
        all_matches.extend(matches)
    
    return all_matches


def find_embedding_pattern(G: nx.DiGraph) -> List[Dict[str, str]]:
    """Find embedding lookup patterns typical in RecSys."""
    pattern_nodes = ["aten::embedding"]
    # Just find embedding nodes for now
    matches = []
    for node_id, data in G.nodes(data=True):
        if data.get('operation') == 'aten::embedding':
            matches.append({'p0': node_id})
    
    return matches


def analyze_graph_complexity(G: nx.DiGraph) -> Dict[str, float]:
    """Analyze computational complexity indicators."""
    if not G.nodes():
        return {"compute_intensity": 0.0, "memory_bandwidth": 0.0, "parallelism": 0.0}
    
    # Compute intensity heuristics
    compute_ops = {"aten::matmul", "aten::mm", "aten::bmm", "aten::conv2d", "aten::linear"}
    memory_ops = {"aten::embedding", "aten::gather", "aten::index_select"}
    
    compute_nodes = sum(1 for _, data in G.nodes(data=True) 
                       if data.get('operation') in compute_ops)
    memory_nodes = sum(1 for _, data in G.nodes(data=True) 
                      if data.get('operation') in memory_ops)
    
    total_nodes = len(G.nodes())
    compute_intensity = compute_nodes / total_nodes
    memory_bandwidth = memory_nodes / total_nodes
    
    # Parallelism potential (based on graph width)
    try:
        levels = list(nx.topological_generations(G))
        max_width = max(len(level) for level in levels) if levels else 1
        parallelism = max_width / total_nodes
    except nx.NetworkXError:
        # Not a DAG
        parallelism = 1.0 / total_nodes
    
    return {
        "compute_intensity": compute_intensity,
        "memory_bandwidth": memory_bandwidth,
        "parallelism": parallelism,
    }
