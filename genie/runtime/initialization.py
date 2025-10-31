"""
Genie Runtime Initialization Module

DESIGN PRINCIPLES (Senior Engineer):
1. Async-first: Non-blocking initialization on first Genie API use
2. Early trigger: Initialize when ANY Genie code is first invoked, not at materialize
3. Once-only: Guarantee single initialization with double-check locking
4. Transparent: Automatic for users, but explicit control available
5. Measurable: Initialization cost tracked separately from workload

INITIALIZATION TRIGGERS:
- FactoryInterceptor.wrap() - tensor creation
- LazyTensor.__torch_dispatch__() - operations
- capture context entry - graph capture
- ensure_initialized() - explicit call

FLOW:
  User imports Genie
    ↓
  _initialize() wraps factories (synchronous, fast)
    ↓
  User creates tensor / captures / runs operation
    ↓
  Interception code calls _ensure_async_init()
    ↓
  Async init happens once, non-blocking
    ↓
  Operation continues
"""

import asyncio
import logging
import threading
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL RUNTIME STATE
# ============================================================================

@dataclass
class GenieRuntimeState:
    """Global runtime state singleton."""
    initialized: bool = False
    coordinator: Optional[Any] = None  # GenieCoordinator instance
    thread_pool: Optional[ThreadPoolExecutor] = None
    server_address: Optional[str] = None
    gpu_capabilities: Optional[Dict[str, Any]] = None
    config: Optional[Any] = None
    event_loop: Optional[asyncio.AbstractEventLoop] = None
    
    # Initialization tracking
    initialization_lock: threading.Lock = field(default_factory=threading.Lock)
    initialization_event: Optional[asyncio.Event] = None
    initialization_task: Optional[asyncio.Task] = None
    initialization_error: Optional[Exception] = None
    
    # Initialization timing
    init_start_time: Optional[float] = None
    init_end_time: Optional[float] = None


# Global state instance
_runtime_state = GenieRuntimeState()

# Initialization hooks registry
_init_hooks: List[Callable] = []


# ============================================================================
# REGISTRATION SYSTEM FOR EARLY INITIALIZATION
# ============================================================================

def register_init_hook(hook: Callable) -> None:
    """
    Register a callback to be called during initialization.
    
    This allows different parts of the library to hook into initialization
    and perform setup tasks.
    
    Args:
        hook: Callable that takes (runtime_state, config) and returns None
    """
    _init_hooks.append(hook)


def _call_init_hooks(runtime_state: GenieRuntimeState, config: Any) -> None:
    """Call all registered initialization hooks."""
    for hook in _init_hooks:
        try:
            hook(runtime_state, config)
            logger.debug(f"✓ Init hook {hook.__name__} completed")
        except Exception as e:
            logger.warning(f"Init hook {hook.__name__} failed: {e}")


# ============================================================================
# PUBLIC API - EXPLICIT INITIALIZATION
# ============================================================================

def init(
    server_address: Optional[str] = None,
    auto_connect: bool = True,
    thread_pool_size: int = 4,
    profiling: bool = False
) -> Dict[str, Any]:
    """
    Initialize Genie runtime explicitly.
    
    Call this BEFORE your benchmarks/workload for better control and measurement.
    If not called, auto-init will trigger on first Genie API call.
    
    DESIGN NOTE (Senior Engineer):
    - This is a convenience wrapper around async initialization
    - For async code, use init_async() instead for non-blocking init
    - Blocking this call for synchronous compatibility
    
    Args:
        server_address: Remote server address (e.g., 'localhost:5556')
                       If None, checks GENIE_SERVER_ADDRESS env var
        auto_connect: Whether to connect to remote server automatically
        thread_pool_size: Number of threads in execution pool
        profiling: Enable performance profiling
    
    Returns:
        Dictionary with initialization result
    """
    try:
        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, caller should use init_async() directly
            raise RuntimeError(
                "Cannot call genie.init() from within an async context. "
                "Use 'await genie.runtime.initialization.init_async()' instead."
            )
        except RuntimeError as e:
            if "Cannot call genie.init()" in str(e):
                raise
            # No running loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run async initialization
        result = loop.run_until_complete(
            init_async(
                server_address=server_address,
                auto_connect=auto_connect,
                thread_pool_size=thread_pool_size,
                profiling=profiling
            )
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to initialize Genie: {e}", exc_info=True)
        return {
            'status': 'error',
            'initialized': False,
            'error': str(e)
        }


async def init_async(
    server_address: Optional[str] = None,
    auto_connect: bool = True,
    thread_pool_size: int = 4,
    profiling: bool = False
) -> Dict[str, Any]:
    """
    Initialize Genie runtime asynchronously (non-blocking).
    
    DESIGN NOTE (Senior Engineer):
    - This is the core async initialization
    - Allows clean integration with async code
    - Guarantees once-only initialization
    - Waits for any in-progress initialization
    
    Args:
        server_address: Remote server address
        auto_connect: Whether to auto-connect to remote server
        thread_pool_size: Number of threads in execution pool
        profiling: Enable performance profiling
    
    Returns:
        Dictionary with initialization result
    """
    with _runtime_state.initialization_lock:
        if _runtime_state.initialized:
            logger.debug("Genie already initialized")
            return {
                'status': 'success',
                'initialized': True,
                'server_address': _runtime_state.server_address,
                'gpu_count': _runtime_state.gpu_capabilities.get('gpu_count') if _runtime_state.gpu_capabilities else None,
                'duration_ms': (_runtime_state.init_end_time - _runtime_state.init_start_time) * 1000 if _runtime_state.init_end_time else None,
            }
        
        if _runtime_state.initialization_task is not None:
            # Already in progress, wait for it
            logger.debug("Initialization in progress, waiting...")
            pass  # Will wait below
    
    # If initialization is in progress, wait for it
    if _runtime_state.initialization_task is not None:
        try:
            result = await asyncio.wait_for(_runtime_state.initialization_task, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            return {
                'status': 'error',
                'initialized': False,
                'error': 'Initialization timeout after 30 seconds'
            }
    
    # Create and run initialization task
    _runtime_state.initialization_task = asyncio.create_task(
        _genie_init_async_impl(
            server_address=server_address,
            auto_connect=auto_connect,
            thread_pool_size=thread_pool_size,
            profiling=profiling
        )
    )
    
    return await _runtime_state.initialization_task


async def _genie_init_async_impl(
    server_address: Optional[str] = None,
    auto_connect: bool = True,
    thread_pool_size: int = 4,
    profiling: bool = False
) -> Dict[str, Any]:
    """
    Core async initialization implementation.
    
    This is called by init_async() and _ensure_async_init().
    """
    import time
    _runtime_state.init_start_time = time.time()
    
    try:
        with _runtime_state.initialization_lock:
            if _runtime_state.initialized:
                return {'status': 'success', 'initialized': True}
        
        logger.info("=" * 80)
        logger.info("🚀 Initializing Genie Runtime")
        logger.info("=" * 80)
        
        # Step 1: Load configuration
        logger.info("[1/5] Loading configuration...")
        from ..config import get_config
        config = get_config()
        
        # Resolve server address (from param > env > config > None)
        if server_address is None:
            server_address = os.environ.get('GENIE_SERVER_ADDRESS')
        if server_address is None and hasattr(config, 'network'):
            if hasattr(config.network, 'remote_server_address'):
                server_address = config.network.remote_server_address
        
        _runtime_state.config = config
        _runtime_state.server_address = server_address
        logger.info(f"  ✓ Configuration loaded")
        if server_address:
            logger.info(f"  ✓ Remote server: {server_address}")
        
        # Step 2: Create thread pool
        logger.info(f"[2/5] Creating thread pool (size={thread_pool_size})...")
        _runtime_state.thread_pool = ThreadPoolExecutor(
            max_workers=thread_pool_size,
            thread_name_prefix="genie-worker-"
        )
        logger.info(f"  ✓ Thread pool created")
        
        # Step 3: Call initialization hooks
        logger.info("[3/5] Calling initialization hooks...")
        _call_init_hooks(_runtime_state, config)
        logger.info(f"  ✓ {len(_init_hooks)} hooks executed")
        
        # Step 4: Connect to remote server (if configured)
        if server_address and auto_connect:
            logger.info(f"[4/5] Connecting to remote server at {server_address}...")
            try:
                from ..core.coordinator import GenieCoordinator, CoordinatorConfig
                
                coordinator_config = CoordinatorConfig(
                    node_id='genie-client',
                    tcp_fallback=True
                )
                
                coordinator = GenieCoordinator(coordinator_config)
                await coordinator.start()
                
                _runtime_state.coordinator = coordinator
                logger.info(f"  ✓ Connected to remote server")
            
            except Exception as e:
                logger.warning(f"  ✗ Failed to connect: {e}")
                logger.info("  Falling back to local execution")
                _runtime_state.coordinator = None
        else:
            logger.info("[4/5] Remote server not configured, using local execution")
        
        # Step 5: Query GPU capabilities (if remote connected)
        logger.info("[5/5] Querying capabilities...")
        if _runtime_state.coordinator:
            try:
                capabilities = await _runtime_state.coordinator.get_capabilities()
                _runtime_state.gpu_capabilities = capabilities
                gpu_count = capabilities.get('gpu_count', '?')
                logger.info(f"  ✓ Remote has {gpu_count} GPUs")
            except Exception as e:
                logger.warning(f"  ✗ Failed to query capabilities: {e}")
                _runtime_state.gpu_capabilities = None
        else:
            logger.info("  ℹ Using local execution (no remote capabilities)")
        
        # Mark as initialized
        _runtime_state.initialized = True
        _runtime_state.init_end_time = time.time()
        
        init_duration = (_runtime_state.init_end_time - _runtime_state.init_start_time) * 1000
        logger.info("=" * 80)
        logger.info(f"✅ Genie runtime initialized in {init_duration:.1f}ms")
        logger.info("=" * 80)
        
        return {
            'status': 'success',
            'initialized': True,
            'server_address': _runtime_state.server_address,
            'gpu_count': _runtime_state.gpu_capabilities.get('gpu_count') if _runtime_state.gpu_capabilities else None,
            'thread_pool_size': thread_pool_size,
            'profiling': profiling,
            'duration_ms': init_duration,
        }
    
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        _runtime_state.initialization_error = e
        _runtime_state.init_end_time = time.time()
        
        return {
            'status': 'error',
            'initialized': False,
            'error': str(e),
        }


# ============================================================================
# AUTOMATIC INITIALIZATION - Triggered on First Genie API Call
# ============================================================================

_auto_init_task: Optional[asyncio.Task] = None
_auto_init_lock = threading.Lock()


def _ensure_async_init() -> None:
    """
    DESIGN NOTE (Senior Engineer):
    
    This is the KEY to the entire system. It's called from:
    - FactoryInterceptor._create_wrapper() - tensor creation
    - LazyTensor.__torch_dispatch__() - operations
    - capture() context manager - graph capture
    
    It ensures:
    1. Non-blocking initialization (returns immediately)
    2. Actually starts async initialization if needed
    3. Continues with user's operation (doesn't block)
    4. Initialization happens concurrently
    
    For proper timing measurement in benchmarks:
    - Call genie.init() explicitly BEFORE benchmarks
    - This way benchmarks don't include init time
    """
    global _auto_init_task
    
    # Quick check without lock
    if _runtime_state.initialized:
        return
    
    # Check if initialization is already in progress
    with _auto_init_lock:
        if _runtime_state.initialized or _auto_init_task is not None:
            return
        
        logger.debug("🚀 Auto-initialization triggered on first Genie API call")
        
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop, create one in background thread
                loop = asyncio.new_event_loop()
                
                def run_background_init():
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(_create_auto_init_task())
                
                thread = threading.Thread(target=run_background_init, daemon=True)
                thread.start()
                return
            
            # We're in an async context, create task
            _auto_init_task = loop.create_task(_create_auto_init_task())
        
        except Exception as e:
            logger.warning(f"Failed to start auto-init: {e}")


async def _create_auto_init_task() -> None:
    """Create and run auto-initialization task."""
    try:
        # Load server address from config/env
        server_address = os.environ.get('GENIE_SERVER_ADDRESS')
        
        if server_address is None:
            from ..config import get_config
            config = get_config()
            if hasattr(config, 'network') and hasattr(config.network, 'remote_server_address'):
                server_address = config.network.remote_server_address
        
        # Initialize
        await init_async(
            server_address=server_address,
            auto_connect=True,
            thread_pool_size=4,
            profiling=False
        )
    
    except Exception as e:
        logger.warning(f"Auto-initialization failed (continuing with local execution): {e}")


def ensure_initialized() -> None:
    """
    Synchronous wrapper that ensures initialization is triggered.
    
    This is called from:
    - LazyTensor.materialize() - graph execution
    - get_thread_pool() - thread pool access
    - get_coordinator() - coordinator access
    
    DESIGN NOTE (Senior Engineer):
    - In async code, use await init_async() instead
    - This ensures initialization happens, even in sync contexts
    - For benchmarking: call genie.init() before measurements
    """
    _ensure_async_init()
    
    # Brief wait to allow background initialization to start
    # (but doesn't block on completion - that happens in executor/materialize)
    if _runtime_state.initialization_error:
        logger.warning("Genie initialization had errors, continuing with local execution")


# ============================================================================
# GETTER FUNCTIONS
# ============================================================================

def get_runtime_state() -> GenieRuntimeState:
    """Get current runtime state."""
    ensure_initialized()
    return _runtime_state


def is_initialized() -> bool:
    """Check if Genie runtime is initialized."""
    return _runtime_state.initialized


def get_thread_pool() -> ThreadPoolExecutor:
    """Get the global thread pool."""
    ensure_initialized()
    
    if _runtime_state.thread_pool is None:
        raise RuntimeError("Thread pool not initialized")
    
    return _runtime_state.thread_pool


def get_coordinator() -> Optional[Any]:
    """Get the global coordinator (if connected to remote server)."""
    ensure_initialized()
    return _runtime_state.coordinator


def get_initialization_time_ms() -> Optional[float]:
    """Get the initialization time in milliseconds (if it has completed)."""
    if _runtime_state.init_start_time and _runtime_state.init_end_time:
        return (_runtime_state.init_end_time - _runtime_state.init_start_time) * 1000
    return None


# ============================================================================
# SHUTDOWN & CLEANUP
# ============================================================================

async def shutdown() -> None:
    """Gracefully shutdown Genie runtime."""
    logger.info("Shutting down Genie runtime...")
    
    # Shutdown coordinator
    if _runtime_state.coordinator:
        try:
            await _runtime_state.coordinator.stop()
            logger.debug("✓ Coordinator shutdown complete")
        except Exception as e:
            logger.warning(f"Error shutting down coordinator: {e}")
    
    # Shutdown thread pool
    if _runtime_state.thread_pool:
        try:
            _runtime_state.thread_pool.shutdown(wait=True)
            logger.debug("✓ Thread pool shutdown complete")
        except Exception as e:
            logger.warning(f"Error shutting down thread pool: {e}")
    
    _runtime_state.initialized = False
    logger.info("✓ Genie runtime shutdown complete")


