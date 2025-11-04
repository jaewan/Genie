"""
Graph cache for parsed computation graphs.

Caches parsed operation graphs to avoid repeated JSON parsing and operation
reconstruction on subsequent requests with the same graph structure.
"""

import hashlib
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GraphCache:
    """
    Cache parsed computation graphs.
    
    This eliminates the ~7ms graph parsing overhead on subsequent requests
    with the same graph structure (common in autoregressive generation).
    
    Expected savings: 7ms per cache hit
    """

    def __init__(self, max_graphs: int = 100):
        """
        Initialize graph cache.
        
        Args:
            max_graphs: Maximum number of graphs to cache (LRU eviction)
        """
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.max_graphs = max_graphs
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }
        logger.info("GraphCache initialized: max_graphs=%d", max_graphs)

    def get_graph(
        self,
        graph_json: bytes,
        parser: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Get cached graph or parse.
        
        Args:
            graph_json: Serialized graph specification (bytes)
            parser: Optional parser function (defaults to torch.load)
            
        Returns:
            Parsed graph dictionary
        """
        # Compute hash of graph structure
        graph_hash = hashlib.md5(graph_json).hexdigest()

        # Cache hit - move to end (LRU)
        if graph_hash in self.cache:
            self.stats["hits"] += 1
            self.cache.move_to_end(graph_hash)
            logger.debug(
                "Graph cache HIT for hash=%s (hit_rate=%.1f%%)",
                graph_hash[:8],
                100 * self.stats["hits"] / (self.stats["hits"] + self.stats["misses"]),
            )
            return self.cache[graph_hash]

        # Cache miss - parse graph
        self.stats["misses"] += 1
        logger.info(
            "Graph cache MISS for hash=%s, parsing graph",
            graph_hash[:8],
        )

        # Parse graph (this is the expensive part we're caching)
        if parser is None:
            import torch
            import io
            parsed = torch.load(io.BytesIO(graph_json))
        else:
            parsed = parser(graph_json)

        # Evict oldest graph if at capacity
        if len(self.cache) >= self.max_graphs:
            evicted_hash = next(iter(self.cache))
            self.cache.pop(evicted_hash)
            self.stats["evictions"] += 1
            logger.debug(
                "Graph cache evicted hash=%s",
                evicted_hash[:8],
            )

        self.cache[graph_hash] = parsed
        logger.debug(
            "Graph cache stored hash=%s (cached_graphs=%d)",
            graph_hash[:8],
            len(self.cache),
        )

        return parsed

    def clear(self) -> None:
        """Clear all cached graphs."""
        self.cache.clear()
        logger.info("Graph cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (
            100 * self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        )
        return {
            **self.stats,
            "cached_graphs": len(self.cache),
            "hit_rate_percent": hit_rate,
        }


# Global cache instance (singleton pattern for simple_server.py)
_global_cache: Optional[GraphCache] = None


def get_global_graph_cache(max_graphs: int = 100) -> GraphCache:
    """Get or create the global graph cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = GraphCache(max_graphs=max_graphs)
    return _global_cache


def clear_global_graph_cache() -> None:
    """Clear the global graph cache."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()

