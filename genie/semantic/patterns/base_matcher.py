"""
Base class for pattern matchers.

Pattern matchers identify high-level computational patterns in the computation
graph and annotate nodes with semantic metadata.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """Represents a detected pattern in the computation graph."""
    name: str
    nodes: List[Any]  # List of GraphNode
    metadata: Dict[str, Any]


class PatternMatcher(ABC):
    """
    Abstract base class for pattern matchers.
    
    Subclasses implement specific pattern detection logic.
    """
    
    @property
    @abstractmethod
    def pattern_name(self) -> str:
        """Name of the pattern being detected."""
        pass
    
    @abstractmethod
    def match(self, graph) -> List[Pattern]:
        """
        Detect pattern occurrences in graph.
        
        Args:
            graph: Unified Graph interface (FX or LazyDAG)
        
        Returns:
            List of detected Pattern instances
        """
        pass
    
    def is_match(self, node) -> bool:
        """
        Check if a single node matches this pattern.
        
        Override for simple node-level matching.
        """
        return False


class PatternRegistry:
    """
    Registry of all pattern matchers.
    
    Manages the lifecycle of pattern matchers and coordinates detection.
    """
    
    def __init__(self):
        self._matchers: List[PatternMatcher] = []
        self._patterns_cache = {}
    
    def register(self, matcher: PatternMatcher):
        """Register a new pattern matcher."""
        self._matchers.append(matcher)
        logger.debug(f"Registered pattern matcher: {matcher.pattern_name}")
    
    def match_all(self, graph) -> Dict[str, List[Pattern]]:
        """
        Run all pattern matchers on graph.
        
        Returns:
            Dict mapping pattern name → list of detected patterns
        """
        results = {}
        for matcher in self._matchers:
            try:
                patterns = matcher.match(graph)
                results[matcher.pattern_name] = patterns
            except Exception as e:
                logger.warning(f"Pattern matcher {matcher.pattern_name} failed: {e}")
        
        return results


# Global pattern registry
_global_pattern_registry = PatternRegistry()


def get_pattern_registry() -> PatternRegistry:
    """Get the global pattern registry."""
    return _global_pattern_registry
