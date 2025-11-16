# DEPRECATED: This package has been restructured.
# Semantic analysis components moved to djinn.frontend.semantic
# Execution components moved to djinn.backend.server
#
# For backward compatibility, imports are available from:
# - djinn.frontend.semantic for analysis/annotation
# - djinn.backend.server for execution optimizations

# Re-export for backward compatibility
from djinn.frontend.semantic.analyzer import SemanticAnalyzer  # noqa: F401
from djinn.frontend.semantic.pattern_registry import PatternRegistry  # noqa: F401
from djinn.frontend.semantic.workload import WorkloadProfile, WorkloadType  # noqa: F401
from djinn.frontend.semantic.hooks import HookManager  # noqa: F401
from djinn.frontend.semantic.annotator import (  # noqa: F401
        SemanticAnnotator,
        AnnotatedGraph,
        annotate_graph,
    )
from djinn.frontend.semantic.metadata_registry import (  # noqa: F401
        MetadataRegistry,
        NodeMetadata,
        get_metadata_registry,
    )
from djinn.frontend.semantic.phase_detector import (  # noqa: F401
        ExecutionPhase,
        PhaseDetector,
        PhaseAnnotator,
    )
from djinn.scheduler.core.cost_estimator import (  # noqa: F401
        CostEstimate,
        CostEstimator,
        GraphCostEstimator,
    )
from djinn.frontend.semantic.patterns import Pattern  # noqa: F401
# PatternMatcher, AttentionMatcher, ConvolutionMatcher, KVCacheMatcher removed - unused
# get_pattern_registry removed - use PatternRegistry from pattern_registry.py instead
from djinn.server.optimizations.phase_executor import (  # noqa: F401
        PhaseAwareExecutor,
        PrefillExecutionStrategy,
        DecodeExecutionStrategy,
        VisionExecutionStrategy,
        ExecutionPhaseStrategy,
        ExecutionStrategy,
    )

__all__ = [
    'SemanticAnalyzer',
    'PatternRegistry',
    'WorkloadProfile',
    'WorkloadType',
    'HookManager',
    'SemanticAnnotator',
    'AnnotatedGraph',
    'annotate_graph',
    'MetadataRegistry',
    'NodeMetadata',
    'get_metadata_registry',
    'ExecutionPhase',
    'PhaseDetector',
    'PhaseAnnotator',
    'CostEstimate',
    'CostEstimator',
    'GraphCostEstimator',
    'Pattern',  # Internal DTO - will be migrated to MatchedPattern
    # PatternMatcher, AttentionMatcher, ConvolutionMatcher, KVCacheMatcher removed
    # get_pattern_registry removed - use PatternRegistry from pattern_registry.py
    'PhaseAwareExecutor',
    'PrefillExecutionStrategy',
    'DecodeExecutionStrategy',
    'VisionExecutionStrategy',
    'ExecutionPhaseStrategy',
    'ExecutionStrategy',
]