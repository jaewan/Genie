from .analyzer import SemanticAnalyzer  # noqa: F401
from .pattern_registry import PatternRegistry  # noqa: F401
from .workload import WorkloadProfile, WorkloadType  # noqa: F401
from .hooks import HookManager  # noqa: F401
from .fx_analyzer import FXAnalyzer  # noqa: F401
from .scheduling import Scheduler, ExecutionSchedule, SchedulingStrategy  # noqa: F401

# Phase 3 modules
try:
    from .annotator import (  # noqa: F401
        SemanticAnnotator,
        AnnotatedGraph,
        annotate_graph,
    )
    from .metadata_registry import (  # noqa: F401
        MetadataRegistry,
        NodeMetadata,
        get_metadata_registry,
    )
    from .phase_detector import (  # noqa: F401
        ExecutionPhase,
        PhaseDetector,
        PhaseAnnotator,
    )
    from .cost_estimator import (  # noqa: F401
        CostEstimate,
        CostEstimator,
        GraphCostEstimator,
    )
    from .patterns import (  # noqa: F401
        PatternMatcher,
        Pattern,
        get_pattern_registry,
        AttentionMatcher,
        ConvolutionMatcher,
        KVCacheMatcher,
    )
except ImportError as e:
    # Phase 3 not available yet
    pass

__all__ = [
    'SemanticAnalyzer',
    'PatternRegistry',
    'WorkloadProfile',
    'WorkloadType',
    'HookManager',
    'FXAnalyzer',
    'Scheduler',
    'ExecutionSchedule',
    'SchedulingStrategy',
    # Phase 3 exports
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
    'PatternMatcher',
    'Pattern',
    'get_pattern_registry',
    'AttentionMatcher',
    'ConvolutionMatcher',
    'KVCacheMatcher',
]