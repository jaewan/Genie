"""
Capture context manager with thread-local state signaling.

Uses thread-local storage for thread safety. Signals to factory
interceptor when operations should return LazyTensors instead of concrete tensors.

Threading Behavior:
- Each thread has its own capture state
- Capture contexts don't interfere across threads
- Nested contexts work correctly within each thread
- State is properly isolated and restored

Example:
    >>> # Thread-safe usage
    >>> def worker():
    ...     with genie.capture():
    ...         x = torch.randn(10)  # LazyTensor in this thread only
    ...
    >>> import threading
    >>> t1 = threading.Thread(target=worker)
    >>> t2 = threading.Thread(target=worker)
    >>> t1.start(); t2.start()  # Independent captures

    >>> # Nested contexts
    >>> with genie.capture():
    ...     x = torch.randn(10)  # Outer capture
    ...     with genie.capture():
    ...         y = torch.randn(10)  # Inner capture
    ...     z = torch.randn(10)  # Back to outer
"""
import logging
import threading
from contextlib import contextmanager
from typing import Optional

from .graph_interface import Graph
from .graph_builder import get_global_builder

logger = logging.getLogger(__name__)

# Thread-local state for capture context
# This signals to factory interceptor that we're in capture mode
_capture_context = threading.local()


class CaptureContext:
    """
    Context for capturing operations into a computation graph.

    This context manager signals to the factory interceptor that operations
    should return LazyTensors instead of concrete tensors. The captured
    operations are recorded in a computation graph for later analysis,
    scheduling, and execution.

    **Thread Safety:** Each thread has its own capture state via thread-local
    storage. Capture contexts don't interfere across threads, and nested
    contexts work correctly within each thread.

    **Graph State Management:** Properly saves and restores graph builder state
    to handle nested capture contexts. Each context gets a fresh graph capture
    session while preserving the parent context's state.

    Example:
        >>> with genie.capture():
        ...     x = torch.randn(10, 10)
        ...     y = model(x)
        >>> graph = genie.get_graph()  # Get captured graph
    """

    def __init__(self):
        self.builder = get_global_builder()
        self.prev_root = None
        self.prev_active = False

    def __enter__(self):
        # Signal to factory interceptor that we're in capture mode
        self.prev_active = getattr(_capture_context, 'active', False)
        _capture_context.active = True

        # Save previous state - get the CURRENT root_tensor before clearing
        self.prev_root = self.builder.root_tensor

        # Start fresh capture for this context
        self.builder.root_tensor = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore capture state (ALWAYS, even on exception)."""
        try:
            # Restore the capture active state
            _capture_context.active = self.prev_active

            # ✅ FIX: Always restore the previous root_tensor
            # This ensures nested contexts properly restore to parent context
            self.builder.root_tensor = self.prev_root

        except Exception as e:
            logger.error(f"Failed to restore capture context: {e}")
            # Don't suppress original exception
        return False  # Propagate exceptions


@contextmanager
def capture():
    """
    Capture operations into a computation graph.

    This context manager enables transparent interception of tensor operations,
    capturing them into a computation graph for semantic analysis and scheduling.

    **Usage:**
        >>> with genie.capture():
        ...     x = torch.randn(10, 10)  # Returns LazyTensor
        ...     y = model(x)             # Operations captured
        >>> graph = genie.get_graph()    # Get captured graph

    **Behavior:**
    - Factory functions (torch.randn, torch.zeros, etc.) return LazyTensors
    - Operations on LazyTensors are deferred until materialization
    - Graph is built incrementally as operations are performed
    - Outside capture context, normal PyTorch behavior resumes

    **Thread Safety:** Each thread has independent capture state. Contexts
    don't interfere across threads.

    **Nested Contexts:** Properly handles nested capture contexts by saving
    and restoring graph state. Each context gets a fresh capture session.

    Example:
        >>> # Outer context captures entire model
        >>> with genie.capture():
        ...     x = torch.randn(batch_size, seq_len)
        ...     with genie.capture():  # Inner context for sub-computation
        ...         y = x @ weight_matrix
        ...     output = model(y)
        >>> graph = genie.get_graph()  # Contains entire computation
    """
    ctx = CaptureContext()
    with ctx:
        yield ctx


def get_graph() -> Optional[Graph]:
    """Get the most recently captured graph."""
    builder = get_global_builder()
    return builder.get_graph()


def is_capturing() -> bool:
    """Check if currently inside a capture context."""
    return getattr(_capture_context, 'active', False)