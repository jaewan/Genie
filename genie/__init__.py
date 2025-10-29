"""
Genie: Semantic Disaggregated Execution Framework

INITIALIZATION DESIGN (Senior Engineer):
==========================================

This library implements async-first, eager initialization that triggers on first
Genie API call. The design philosophy prioritizes:

1. TRANSPARENCY: Works out of the box without explicit calls
2. EARLY DETECTION: Initializes when user first touches Genie code
3. NON-BLOCKING: Initialization happens concurrently, doesn't block operations
4. MEASURABLE: Explicit init() API allows users to measure init cost separately
5. ONCE-ONLY: Guarantee of single initialization across all threads/async tasks

INITIALIZATION FLOW:
====================

Path 1: Explicit Initialization (Recommended for Benchmarking)
  import genie
  result = genie.init(server_address='localhost:5556')  ← User controls timing
  # Now benchmarking doesn't include init time
  model = model.to('remote_accelerator:0')  ← Already initialized

Path 2: Implicit/Auto-Initialization (Works automatically)
  import genie
  model = model.to('remote_accelerator:0')  ← Triggers auto-init
  #  _ensure_async_init() called from FactoryInterceptor
  #  Initialization happens in background
  x = model(inputs)  ← Operation may wait for init to complete

INITIALIZATION TRIGGERS:
========================

Early triggers (non-blocking):
  1. torch.randn(device='remote_accelerator:0')  → FactoryInterceptor
  2. model.to('remote_accelerator:0')            → FactoryInterceptor  
  3. with genie.capture():                       → CaptureContext.__enter__
  
Late triggers (blocking):
  4. LazyTensor operation (x @ y)                → __torch_dispatch__
  5. Graph execution (.materialize())            → materialize()
  6. Thread pool access (get_thread_pool())      → getter function
  7. Explicit call (genie.init())                → public API

ASYNC-FIRST DESIGN:
===================

Key insight: Use asyncio.create_task() to start init in background
- Returns immediately (non-blocking)
- Initialization happens concurrently
- Operations wait if they need resources before init completes
- No busy-spinning or polling

Thread Safety:
- Double-check locking pattern in _ensure_async_init()
- Once-only guarantee via initialization_task tracking
- Thread-local storage for event loop detection
- Background thread created if no event loop exists

BENCHMARK RECOMMENDATIONS:
==========================

For publication-quality benchmarks:

  from benchmarks.comprehensive_evaluation import ComprehensiveEvaluation
  
  # PHASE 1: Measure just init cost
  import genie
  import time
  
  start = time.perf_counter()
  result = genie.init(server_address='localhost:5556')
  init_time = (time.perf_counter() - start) * 1000
  print(f"Initialization: {init_time:.1f}ms")
  
  # PHASE 2: Measure workload (init already done)
  eval = ComprehensiveEvaluation(use_real_models=True, spawn_server=True)
  await eval.run_all(num_runs=10, num_warmup=3)

This separates initialization cost from workload cost for honest measurement.
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

# Public API - Phase 3 (Async-First Runtime Initialization)
from .runtime.initialization import (
    init as init_sync,
    init_async,
    ensure_initialized,
    get_runtime_state,
    is_initialized,
    get_thread_pool,
    get_coordinator,
    get_initialization_time_ms,
    register_init_hook,
)

# Convenience wrapper for synchronous code
def init(
    server_address=None,
    auto_connect=True,
    thread_pool_size=4,
    profiling=False
):
    """
    Initialize Genie runtime explicitly.
    
    RECOMMENDED FOR BENCHMARKING: Call this before measurements to separate
    initialization cost from workload cost.
    
    If not called, auto-init will trigger on first Genie API call.
    
    Args:
        server_address: Remote server address (e.g., 'localhost:5556')
                       If None, checks GENIE_SERVER_ADDRESS env var
        auto_connect: Whether to connect to remote server automatically
        thread_pool_size: Number of threads in execution pool (default: 4)
        profiling: Enable performance profiling (default: False)
    
    Returns:
        Dictionary with initialization result:
        {
            'status': 'success' or 'error',
            'initialized': bool,
            'server_address': Optional[str],
            'gpu_count': Optional[int],
            'thread_pool_size': int,
            'duration_ms': float,  # Initialization time
            'error': Optional[str]
        }
    
    Example:
        >>> import genie
        >>> result = genie.init(server_address='localhost:5556')
        >>> if result['status'] == 'success':
        ...     print(f"Initialized in {result['duration_ms']:.1f}ms")
    """
    return init_sync(
        server_address=server_address,
        auto_connect=auto_connect,
        thread_pool_size=thread_pool_size,
        profiling=profiling
    )


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

    # ✅ FIXED: Actually use the semantic scheduler
    from .semantic.scheduling import Scheduler
    from .semantic.cost_estimator import GraphCostEstimator, NetworkTopology

    scheduler = Scheduler(
        cost_estimator=GraphCostEstimator(),
        network_topology=NetworkTopology()
    )

    # Create actual schedule using semantic analysis
    return scheduler.create_schedule(graph, profile)

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

    # Phase 3 (Async-First Runtime Initialization)
    'init',                          # ✅ Public API (synchronous wrapper)
    'init_async',                    # ✅ Public API (async version)
    'init_sync',                     # ✅ Public API (core implementation)
    'ensure_initialized',            # ✅ Trigger auto-init from sync contexts
    'get_runtime_state',             # ✅ Get global runtime state
    'is_initialized',                # ✅ Check if initialized
    'get_thread_pool',               # ✅ Access thread pool
    'get_coordinator',               # ✅ Access remote coordinator
    'get_initialization_time_ms',    # ✅ Get init duration
    'register_init_hook',            # ✅ Register custom init handlers

    # Phase 3
    'annotate_graph',
    'AnnotatedGraph',

    # Convenience Functions
    'analyze',
    'schedule',
]