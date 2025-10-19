"""
Graph annotator: Combines all semantic analysis to produce annotated graph.

Main public API for semantic analysis.
"""

import logging
from typing import Optional, Dict, List
from .patterns.base_matcher import get_pattern_registry
from .phase_detector import PhaseDetector, PhaseAnnotator, ExecutionPhase
from .cost_estimator import GraphCostEstimator
from .metadata_registry import (
    get_metadata_registry, NodeMetadata, MetadataRegistry
)

logger = logging.getLogger(__name__)


class SemanticAnnotator:
    """
    Annotates computation graphs with semantic information.

    Combines:
    1. Pattern detection (attention, convolution, KV cache)
    2. Phase detection (prefill, decode, forward)
    3. Cost estimation (FLOPs, memory, intensity)
    4. Metadata storage

    Uses caching to avoid redundant analysis of identical graphs.
    """

    def __init__(self):
        self.pattern_registry = get_pattern_registry()
        self.phase_detector = PhaseDetector(self.pattern_registry)
        self.phase_annotator = PhaseAnnotator(self.phase_detector)
        self.cost_estimator = GraphCostEstimator()
        self.metadata_registry = get_metadata_registry()

        # Caching for performance
        self._analysis_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def annotate(self, graph) -> 'AnnotatedGraph':
        """
        Fully annotate a computation graph.

        Args:
            graph: Unified Graph (FX or LazyDAG)

        Returns:
            AnnotatedGraph with semantic metadata
        """
        # Check cache first
        graph_hash = self._compute_graph_hash(graph)
        if graph_hash in self._analysis_cache:
            self._cache_hits += 1
            logger.debug(f"Cache hit for graph hash {graph_hash[:8]}...")
            return self._analysis_cache[graph_hash]

        self._cache_misses += 1
        logger.info(f"Starting semantic analysis for {len(list(graph.nodes()))} nodes (cache miss)")
        logger.debug(f"Cache stats: {self._cache_hits} hits, {self._cache_misses} misses")
        
        # Single-pass analysis: pattern detection, phase detection, and cost estimation
        logger.info("Performing single-pass semantic analysis...")

        # Step 1: Run pattern matching on the entire graph
        patterns = self.pattern_registry.match_all(graph)
        logger.debug(f"Detected patterns: {list(patterns.keys())}")

        # Step 2: Detect phases
        phases = self.phase_detector.detect_phases(graph)
        logger.debug(f"Phase distribution: {self._count_phases(phases)}")

        # Step 3: Create metadata for each node (without cost info first)
        logger.info("Creating node metadata...")
        for node in graph.nodes():
            phase_obj = phases.get(node.id)
            phase_value = phase_obj.value if hasattr(phase_obj, 'value') else str(phase_obj) if phase_obj else None

            metadata = NodeMetadata(
                node_id=node.id,
                phase=phase_value,
                semantic_role=getattr(node, 'metadata', {}).get('semantic_role'),
                modality=getattr(node, 'metadata', {}).get('modality'),
            )

            # Add optimization hints
            node_meta = getattr(node, 'metadata', {})
            metadata.optimization_hints = node_meta.get('optimization_hints', {})

            # Register metadata (without cost info for now)
            self.metadata_registry.register_metadata(node.id, metadata)

        # Step 4: Estimate costs (now metadata exists for cost estimator)
        cost_results = self.cost_estimator.estimate_graph(graph)
        costs = cost_results['per_node']
        total_compute = cost_results['total_compute_flops']
        total_memory = cost_results['total_memory_bytes']
        total_data_movement = cost_results['total_data_movement_bytes']

        # Step 5: Update metadata with cost information
        logger.info("Updating metadata with cost estimates...")
        for node in graph.nodes():
            node_cost = costs.get(node.id)
            if node_cost:
                self.metadata_registry.update_metadata(node.id, {
                    'compute_flops': node_cost.compute_flops,
                    'memory_bytes': node_cost.memory_bytes,
                    'operational_intensity': node_cost.operational_intensity,
                    'data_movement_bytes': node_cost.data_movement_bytes,
                })

        # Summary costs
        graph_costs = {
            'per_node': costs,
            'total_compute_flops': total_compute,
            'total_memory_bytes': total_memory,
            'total_data_movement_bytes': total_data_movement,
            'mean_operational_intensity': (total_compute / total_memory if total_memory > 0 else 0),
        }

        logger.info(f"Analysis complete: {len(patterns)} pattern types, {self._count_phases(phases)} phases")
        logger.debug(f"  Total FLOPs: {graph_costs['total_compute_flops']:.2e}")
        logger.debug(f"  Total Memory: {graph_costs['total_memory_bytes']:.2e} bytes")
        
        logger.info("Semantic analysis complete")

        # Cache result
        result = AnnotatedGraph(graph, patterns, phases, graph_costs, self.metadata_registry)
        self._analysis_cache[graph_hash] = result

        return result

    def _compute_graph_hash(self, graph) -> str:
        """Compute stable hash of graph structure for caching."""
        # Create hash from node operations and shapes
        hash_components = []

        for node in graph.nodes():
            op = node.operation
            shape = getattr(node, 'shape', None)
            if shape:
                shape_str = str(tuple(shape))
            else:
                shape_str = "unknown"

            # Include operation and shape in hash
            hash_components.append(f"{op}:{shape_str}")

        # Create stable hash
        graph_structure = "|".join(sorted(hash_components))
        return hash(graph_structure)

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0,
            'cache_size': len(self._analysis_cache)
        }

    def clear_cache(self):
        """Clear analysis cache."""
        self._analysis_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def _match_node_patterns(self, node, graph) -> List:
        """Match patterns for a single node."""
        patterns = []

        # Use pattern matchers that support node-level matching
        for matcher in self.pattern_registry._matchers:
            if hasattr(matcher, 'is_match') and matcher.is_match(node):
                # Create pattern for this node
                pattern = Pattern(
                    name=matcher.pattern_name,
                    nodes=[node],
                    metadata={'semantic_role': matcher.pattern_name}
                )
                patterns.append(pattern)

        return patterns

    def _detect_node_phase(self, node, node_patterns) -> ExecutionPhase:
        """Detect phase for a single node."""
        # Check if node is part of any patterns
        for pattern in node_patterns:
            if pattern.name == "kv_cache":
                return ExecutionPhase.LLM_DECODE
            elif pattern.name == "attention":
                # Check if this is parallel attention (prefill)
                if self._is_parallel_attention_node(node):
                    return ExecutionPhase.LLM_PREFILL

        # Default to forward
        return ExecutionPhase.FORWARD

    def _is_parallel_attention_node(self, node) -> bool:
        """Check if attention node processes multiple positions in parallel."""
        # Heuristic: If batch size in sequence dimension > 1
        shape = getattr(node, 'shape', None)
        if shape and len(shape) >= 2:
            # Assume dimension 1 is sequence for [batch, seq, ...]
            seq_dim = shape[1] if len(shape) >= 3 else shape[0]
            return seq_dim > 1
        return False
    
    def _count_phases(self, phases) -> dict:
        """Count nodes by phase."""
        counts = {}
        for phase in phases.values():
            phase_name = phase.value if hasattr(phase, 'value') else str(phase)
            counts[phase_name] = counts.get(phase_name, 0) + 1
        return counts


class AnnotatedGraph:
    """
    Computation graph with semantic annotations.
    
    Wrapper around a Graph that includes semantic metadata.
    """
    
    def __init__(self, base_graph, patterns, phases, costs, metadata_registry):
        self.base_graph = base_graph
        self.patterns = patterns
        self.phases = phases
        self.costs = costs
        self.metadata_registry = metadata_registry
    
    def nodes(self):
        """Get all annotated nodes."""
        return self.base_graph.nodes()
    
    def get_node(self, node_id):
        """Get node by ID."""
        return self.base_graph.get_node(node_id)
    
    def get_metadata(self, node_id) -> Optional[NodeMetadata]:
        """Get metadata for a node."""
        return self.metadata_registry.get_metadata(node_id)
    
    @property
    def backend_type(self):
        """Backend type (FX or lazy_dag)."""
        return self.base_graph.backend_type
    
    def summary(self) -> str:
        """Get human-readable summary of annotations."""
        lines = [
            f"AnnotatedGraph ({self.backend_type} backend)",
            f"  Total nodes: {len(list(self.nodes()))}",
            f"  Total compute: {self.costs['total_compute_flops']:.2e} FLOPs",
            f"  Total memory: {self.costs['total_memory_bytes']:.2e} bytes",
            f"  Mean intensity: {self.costs['mean_operational_intensity']:.2f} FLOPs/byte",
            f"Patterns detected:",
        ]
        
        for pattern_name, pattern_list in self.patterns.items():
            lines.append(f"  - {pattern_name}: {len(pattern_list)} occurrences")
        
        return "\n".join(lines)


# Main public API
def annotate_graph(graph) -> AnnotatedGraph:
    """
    Annotate a computation graph with semantic information.
    
    Convenience function - creates annotator and runs full analysis.
    
    Args:
        graph: Unified Graph (from genie.get_graph())
    
    Returns:
        AnnotatedGraph with full semantic metadata
    """
    annotator = SemanticAnnotator()
    return annotator.annotate(graph)
