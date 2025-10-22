"""
Graph annotator: Combines all semantic analysis to produce annotated graph.

Main public API for semantic analysis with content-addressed caching.
"""

import logging
import hashlib
from typing import Optional, Dict, List
from .pattern_registry import PatternRegistry
from .phase_detector import PhaseDetector, PhaseAnnotator, ExecutionPhase
from .cost_estimator import GraphCostEstimator
from .metadata_registry import (
    get_metadata_registry, NodeMetadata, MetadataRegistry
)
from .cache import SemanticAnnotatorCache

logger = logging.getLogger(__name__)


class SemanticAnnotator:
    """
    Annotates computation graphs with semantic information.

    Combines:
    1. Pattern detection (attention, convolution, KV cache)
    2. Phase detection (prefill, decode, forward)
    3. Cost estimation (FLOPs, memory, intensity)
    4. Metadata storage

    Uses content-addressed caching for optimal performance.
    """

    def __init__(
        self,
        enable_cache: bool = True,
        cache_dir: str = ".genie_cache",
        max_memory_mb: int = 500
    ):
        # Use the improved PatternRegistry with hierarchical indexing
        from .pattern_registry import PatternRegistry
        self.pattern_registry = PatternRegistry()
        self.phase_detector = PhaseDetector(self.pattern_registry)
        self.phase_annotator = PhaseAnnotator(self.phase_detector)
        self.cost_estimator = GraphCostEstimator()
        self.metadata_registry = get_metadata_registry()

        # Content-addressed cache for full analysis results
        self._pattern_cache: Dict[str, List] = {}
        self._graph_cache: Dict[str, 'AnnotatedGraph'] = {}
        self._cache_size_limit = 1000  # Bound memory usage

        # Initialize legacy cache for backward compatibility
        self.legacy_cache = SemanticAnnotatorCache(
            max_memory_mb=max_memory_mb,
            cache_dir=cache_dir
        ) if enable_cache else None

        # Load persisted cache if available
        if self.legacy_cache:
            self.legacy_cache.load()

        # Metrics for content-addressed cache
        self._cache_hits = 0
        self._cache_misses = 0
    
    def annotate(self, graph) -> 'AnnotatedGraph':
        """
        Fully annotate a computation graph with content-addressed caching.

        Args:
            graph: Unified Graph (FX or LazyDAG)

        Returns:
            AnnotatedGraph with semantic metadata
        """
        # Compute content hash (structural + operations + shapes)
        graph_key = self._compute_content_hash(graph)

        # Check content-addressed cache first
        if graph_key in self._graph_cache:
            self._cache_hits += 1
            logger.debug(f"Content cache hit: {graph_key[:8]}...")
            return self._graph_cache[graph_key]

        self._cache_misses += 1

        # Full analysis (cache miss)
        result = self._analyze_graph(graph)

        # Cache result (with LRU eviction)
        self._add_to_cache(graph_key, result)

        return result

    def _compute_content_hash(self, graph) -> str:
        """
        Compute structure-based hash of graph.

        Design: Hash structure + operations + shapes (not tensor values).
        This makes structurally identical graphs hit cache even with different tensor data.
        """
        hasher = hashlib.blake2b(digest_size=16)  # Fast, good distribution

        # Hash structure (edges by node relationships)
        node_inputs = []
        for node in graph.nodes():
            input_ids = sorted([getattr(inp, 'id', str(id(inp))) for inp in node.inputs])
            node_inputs.append((getattr(node, 'id', str(id(node))), node.operation, tuple(input_ids)))
        node_inputs.sort()
        hasher.update(str(node_inputs).encode())

        # Hash operations sequence
        ops = sorted([
            (getattr(node, 'id', str(id(node))), getattr(node, 'operation', 'unknown'))
            for node in graph.nodes()
        ])
        hasher.update(str(ops).encode())

        # Hash shapes (if available)
        shapes = sorted([
            (getattr(node, 'id', str(id(node))), str(getattr(node, 'shape', None)))
            for node in graph.nodes()
        ])
        hasher.update(str(shapes).encode())

        return hasher.hexdigest()

    def _add_to_cache(self, key: str, result: 'AnnotatedGraph'):
        """Add to cache with LRU eviction."""
        if len(self._graph_cache) >= self._cache_size_limit:
            # Evict oldest (simple FIFO for now; could use LRU)
            oldest_key = next(iter(self._graph_cache))
            del self._graph_cache[oldest_key]
            logger.debug(f"Cache eviction: {oldest_key[:8]}...")

        self._graph_cache[key] = result

    def _analyze_graph(self, graph) -> 'AnnotatedGraph':
        """Full semantic analysis without caching."""
        logger.info(f"Starting semantic analysis for {len(list(graph.nodes()))} nodes")

        # Pattern matching with hierarchical index and early termination
        from .pattern_registry import MatchingMode
        pattern_result = self.pattern_registry.match_patterns(graph, mode=MatchingMode.EXHAUSTIVE)
        if not pattern_result.is_ok:
            logger.warning(f"Pattern matching failed: {pattern_result.error}")
            # Convert to old format for backward compatibility
            patterns = {}
        else:
            matches = pattern_result.unwrap()
            # Convert to old format expected by phase detector
            patterns = self._convert_matches_to_patterns(matches)

        logger.debug(f"Detected patterns: {list(patterns.keys())}")

        # Phase detection
        phases = self.phase_detector.detect_phases(graph, patterns)
        logger.debug(f"Phase distribution: {self._count_phases(phases)}")

        # Cost estimation and metadata
        return self._compute_costs(graph, patterns, phases)

    def _convert_matches_to_patterns(self, matches):
        """Convert new MatchedPattern format to old Pattern format."""
        from .patterns.base_matcher import Pattern

        patterns = {}
        for match in matches:
            # Create a simple Pattern object from MatchedPattern
            pattern = Pattern(
                name=match.pattern_name,
                nodes=[],  # TODO: Extract nodes from match if available
                metadata={
                    'confidence': match.confidence,
                    'optimization_hints': match.optimization_hints,
                    'metadata': match.metadata
                }
            )
            patterns[match.pattern_name] = [pattern]

        return patterns

    def _compute_topology(self, graph):
        """Compute topology-level analysis (structure only)."""
        logger.info(f"Computing topology analysis for {len(list(graph.nodes()))} nodes")
        
        patterns = self.pattern_registry.match_all(graph)
        logger.debug(f"Detected patterns: {list(patterns.keys())}")

        phases = self.phase_detector.detect_phases(graph)
        logger.debug(f"Phase distribution: {self._count_phases(phases)}")
        
        return (patterns, phases)
    
    def _compute_costs(self, graph, patterns, phases):
        """Compute shape-specific costs."""
        logger.info("Computing shape-specific costs...")
        
        for node in graph.nodes():
            phase_obj = phases.get(node.id)
            phase_value = phase_obj.value if hasattr(phase_obj, 'value') else str(phase_obj) if phase_obj else None

            metadata = NodeMetadata(
                node_id=node.id,
                phase=phase_value,
                semantic_role=getattr(node, 'metadata', {}).get('semantic_role'),
                modality=getattr(node, 'metadata', {}).get('modality'),
            )

            node_meta = getattr(node, 'metadata', {})
            metadata.optimization_hints = node_meta.get('optimization_hints', {})

            self.metadata_registry.register_metadata(node.id, metadata)

        cost_results = self.cost_estimator.estimate_graph(graph)
        costs = cost_results['per_node']
        total_compute = cost_results['total_compute_flops']
        total_memory = cost_results['total_memory_bytes']
        total_data_movement = cost_results['total_data_movement_bytes']

        for node in graph.nodes():
            node_cost = costs.get(node.id)
            if node_cost:
                self.metadata_registry.update_metadata(node.id, {
                    'compute_flops': node_cost.compute_flops,
                    'memory_bytes': node_cost.memory_bytes,
                    'operational_intensity': node_cost.operational_intensity,
                    'data_movement_bytes': node_cost.data_movement_bytes,
                })

        graph_costs = {
            'per_node': costs,
            'total_compute_flops': total_compute,
            'total_memory_bytes': total_memory,
            'total_data_movement_bytes': total_data_movement,
            'mean_operational_intensity': (total_compute / total_memory if total_memory > 0 else 0),
        }

        logger.info(f"Cost estimation complete")
        
        return AnnotatedGraph(graph, patterns, phases, graph_costs, self.metadata_registry)
    
    def _annotate_uncached(self, graph) -> 'AnnotatedGraph':
        """Original uncached implementation."""
        logger.info(f"Starting semantic analysis for {len(list(graph.nodes()))} nodes")
        
        patterns = self.pattern_registry.match_all(graph)
        logger.debug(f"Detected patterns: {list(patterns.keys())}")

        phases = self.phase_detector.detect_phases(graph)
        logger.debug(f"Phase distribution: {self._count_phases(phases)}")

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

            node_meta = getattr(node, 'metadata', {})
            metadata.optimization_hints = node_meta.get('optimization_hints', {})

            self.metadata_registry.register_metadata(node.id, metadata)

        cost_results = self.cost_estimator.estimate_graph(graph)
        costs = cost_results['per_node']
        total_compute = cost_results['total_compute_flops']
        total_memory = cost_results['total_memory_bytes']
        total_data_movement = cost_results['total_data_movement_bytes']

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

        graph_costs = {
            'per_node': costs,
            'total_compute_flops': total_compute,
            'total_memory_bytes': total_memory,
            'total_data_movement_bytes': total_data_movement,
            'mean_operational_intensity': (total_compute / total_memory if total_memory > 0 else 0),
        }

        logger.info(f"Semantic analysis complete")
        
        return AnnotatedGraph(graph, patterns, phases, graph_costs, self.metadata_registry)

    def get_cache_stats(self) -> Optional[Dict]:
        """Get cache statistics including content-addressed cache."""
        stats = {
            'content_addressed_cache': {
                'hits': self._cache_hits,
                'misses': self._cache_misses,
                'hit_rate': (self._cache_hits / (self._cache_hits + self._cache_misses)
                           if (self._cache_hits + self._cache_misses) > 0 else 0.0),
                'cache_size': len(self._graph_cache),
                'memory_mb': self._estimate_cache_memory(),
            }
        }

        # Include legacy cache stats if available
        if self.legacy_cache:
            legacy_stats = self.legacy_cache.get_stats()
            stats['legacy_cache'] = legacy_stats

        return stats

    def _estimate_cache_memory(self) -> float:
        """Estimate cache memory usage (rough)."""
        import sys

        if not self._graph_cache:
            return 0.0

        # Sample one entry to estimate average size
        sample = next(iter(self._graph_cache.values()))
        sample_size = sys.getsizeof(sample)

        return (sample_size * len(self._graph_cache)) / (1024 * 1024)

    def save_cache(self):
        """Save cache to disk."""
        if self.cache:
            self.cache.save()

    def clear_cache(self):
        """Clear all caches."""
        if self.cache:
            self.cache.clear()

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


def annotate_graph(graph) -> AnnotatedGraph:
    """
    Annotate a computation graph with semantic information.
    
    Uses advanced two-level caching for performance.
    
    Args:
        graph: Unified Graph (from genie.get_graph())
    
    Returns:
        AnnotatedGraph with full semantic metadata
    """
    annotator = SemanticAnnotator(enable_cache=True)
    return annotator.annotate(graph)
