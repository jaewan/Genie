"""
Genie: Semantic Disaggregated Execution Framework

Clean initialization:
1. Register device backend (optional, non-blocking)
2. Wrap factory functions
3. Set up LazyTensor graph builder
"""

import logging
logger = logging.getLogger(__name__)

# Version
__version__ = "0.2.0"

# Core interception setup
def _initialize():
    """Initialize Genie interception layer."""
    try:
        # Step 1: Try to register C++ backend (optional)
        try:
            from . import _C
            _C.register_remote_accelerator_device()
            logger.info("C++ backend registered")
        except ImportError:
            logger.info("C++ backend not available (Python-only mode)")

        # Step 2: Wrap factory functions (REQUIRED)
        from .core.factory_interceptor import wrap_factories
        wrap_factories()

        # Step 3: Initialize graph builder
        from .core.graph_builder import initialize_global_builder
        initialize_global_builder()

        logger.info("Genie initialized successfully")

    except Exception as e:
        logger.error(f"Genie initialization failed: {e}")
        raise

# Initialize on import
_initialize()

# Public API - Phase 1 (Core)
from .core.lazy_tensor import LazyTensor
from .core.capture import capture, get_graph, is_capturing
from .core.subgraph_builder import SubgraphBuilder, RemoteSubgraph

# Public API - Phase 2 (Smart Fragmentation)
from .core.smart_subgraph_builder import (
    SmartSubgraphBuilder,
    FragmentationConfig,
    SubgraphFragment,
    CostEstimate,
    MemoryEstimator,
    CostCalculator
)

# Public API - Phase 2 (Semantic Analysis & Scheduling)
from .semantic import (
    SemanticAnalyzer,
    Scheduler,
    WorkloadProfile,
    WorkloadType,
    PatternRegistry,
    ExecutionSchedule,
    SchedulingStrategy,
)
from .semantic.annotator import annotate_graph, AnnotatedGraph

# Convenience functions for common workflows
def analyze(graph=None):
    """
    Analyze a computation graph for semantic information.

    Args:
        graph: Graph to analyze. If None, uses the most recently captured graph.

    Returns:
        WorkloadProfile with semantic analysis results
    """
    if graph is None:
        graph = get_graph()
        if graph is None:
            raise RuntimeError("No graph available. Call get_graph() after capture or pass graph explicitly.")

    analyzer = SemanticAnalyzer()
    # For LazyDAG graphs, we'll need to handle this differently
    # For now, return a basic profile
    return WorkloadProfile(
        workload_type=WorkloadType.UNKNOWN,
        patterns=[],
        metadata={'graph_type': graph.backend_type}
    )

def schedule(graph=None, profile=None):
    """
    Create an execution schedule for a computation graph.

    Args:
        graph: Graph to schedule. If None, uses the most recently captured graph.
        profile: WorkloadProfile from semantic analysis. If None, runs analysis first.

    Returns:
        ExecutionSchedule with scheduling decisions
    """
    if graph is None:
        graph = get_graph()
        if graph is None:
            raise RuntimeError("No graph available. Call get_graph() after capture or pass graph explicitly.")

    if profile is None:
        profile = analyze(graph)

    scheduler = Scheduler()
    # For LazyDAG graphs, create a simple schedule
    return ExecutionSchedule(
        stages=[],
        node_to_stage={},
        node_to_group={},
        total_stages=1,
        strategy=SchedulingStrategy.SEQUENTIAL,
        metadata={'graph_type': graph.backend_type}
    )

__all__ = [
    # Phase 1 (Core)
    'LazyTensor',
    'capture',
    'get_graph',
    'is_capturing',
    'SubgraphBuilder',
    'RemoteSubgraph',

    # Phase 2 (Smart Fragmentation)
    'SmartSubgraphBuilder',
    'FragmentationConfig',
    'SubgraphFragment',
    'CostEstimate',
    'MemoryEstimator',
    'CostCalculator',

    # Phase 2 (Semantic Analysis & Scheduling)
    'SemanticAnalyzer',
    'Scheduler',
    'WorkloadProfile',
    'WorkloadType',
    'PatternRegistry',
    'ExecutionSchedule',
    'SchedulingStrategy',

    # Phase 3
    'annotate_graph',
    'AnnotatedGraph',

    # Convenience Functions
    'analyze',
    'schedule',
]