"""
Unified Caching API for Djinn.

Provides consistent interfaces to all caching functionality:
- Graph caching (client-side with differential updates)
- Weight caching (GPU persistent storage)
- Differential updates (delta computation protocol)

Usage:
    from djinn.caching import get_cache

    # Get graph cache
    graph_cache = get_cache('graph')

    # Get weight cache
    weight_cache = get_cache('weight')

    # Get differential protocol
    diff_protocol = get_differential_protocol()
"""

from typing import Union, Any, Optional
from .graph_cache import GraphCache, get_graph_cache
from .weight_cache import GPUWeightCache, get_gpu_weight_cache
from .differential_updates import DifferentialGraphProtocol

# Re-export for convenience
__all__ = [
    'get_cache',
    'get_differential_protocol',
    'GraphCache',
    'GPUWeightCache',
    'DifferentialGraphProtocol'
]


def get_cache(cache_type: str, **kwargs) -> Union[GraphCache, GPUWeightCache]:
    """
    Unified cache factory function.

    Args:
        cache_type: Type of cache ('graph', 'weight', 'gpu')
        **kwargs: Cache-specific initialization parameters

    Returns:
        Cache instance

    Raises:
        ValueError: If cache_type is not supported
    """
    if cache_type in ('graph', 'graphs'):
        return get_graph_cache()
    elif cache_type in ('weight', 'weights', 'gpu_weight'):
        return get_gpu_weight_cache(**kwargs)
    else:
        raise ValueError(
            f"Unsupported cache type: {cache_type}. "
            f"Supported types: 'graph', 'weight'"
        )


def get_differential_protocol() -> DifferentialGraphProtocol:
    """
    Get the differential graph protocol for network-efficient updates.

    Returns:
        DifferentialGraphProtocol instance
    """
    return DifferentialGraphProtocol()

