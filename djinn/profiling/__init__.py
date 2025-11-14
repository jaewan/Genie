"""
Profiling module for Djinn.

NOTE: This module is deprecated. Use djinn.server.profiling_context instead.

The new profiling system uses ProfilingContext and record_phase() context manager
for lightweight, thread-safe phase-level timing measurements.
"""

# Keep imports for backward compatibility but mark as deprecated
import warnings

warnings.warn(
    "djinn.profiling module is deprecated. Use djinn.server.profiling_context instead.",
    DeprecationWarning,
    stacklevel=2
)

# Provide stub implementations for backward compatibility
class DjinnProfiler:
    """Deprecated - use ProfilingContext instead."""
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "DjinnProfiler is deprecated. Use ProfilingContext instead.",
            DeprecationWarning,
            stacklevel=2
        )
    
    def profile_operation(self, *args, **kwargs):
        """Deprecated - no-op."""
        from contextlib import nullcontext
        return nullcontext()


def get_detailed_profiler():
    """Deprecated - returns None."""
    warnings.warn(
        "get_detailed_profiler() is deprecated. Use ProfilingContext instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return None


__all__ = ['DjinnProfiler', 'get_detailed_profiler']
