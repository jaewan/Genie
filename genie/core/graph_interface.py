"""
Unified graph interface for semantic-aware computation graphs.

Design: Single abstraction point that works with different representations:
- FX GraphModule (PyTorch traced)
- LazyTensor DAG (deferred execution)
- Materialized DAG (post-optimization)

All graph algorithms use this interface, enabling multiple backends.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator, Optional, List, Dict, Any, Union
import torch
import logging

logger = logging.getLogger(__name__)

from .types import NodeProtocol, ConcreteNode, DictNodeAdapter

# ============================================================================
# ABSTRACT GRAPH INTERFACE
# ============================================================================

class GenieGraph(ABC):
    """
    Abstract base for all graph representations in Genie.
    
    Provides common operations: iteration, lookup, ordering.
    Concrete implementations adapt to FX, LazyTensor, or other formats.
    """
    
    @abstractmethod
    def nodes(self) -> Iterator[NodeProtocol]:
        """Iterate over all nodes in the graph."""
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[NodeProtocol]:
        """Get node by ID (O(1) lookup for efficiency)."""
        pass

    @abstractmethod
    def topological_order(self) -> List[NodeProtocol]:
        """Get nodes in topological execution order."""
        pass

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        """Number of nodes (O(1) for efficiency)."""
        pass

    @property
    @abstractmethod
    def num_edges(self) -> int:
        """Number of edges (dependencies)."""
        pass
    
    def get_roots(self) -> List[NodeProtocol]:
        """Get root nodes (inputs with no predecessors)."""
        roots = []
        for node in self.nodes():
            if not node.inputs:
                roots.append(node)
        return roots
    
    def get_leaves(self) -> List[NodeProtocol]:
        """Get leaf nodes (outputs with no successors)."""
        leaves = []
        for node in self.nodes():
            if not node.outputs:
                leaves.append(node)
        return leaves
    
    def get_subgraph(self, node_ids: set[str]) -> GenieGraph:
        """Extract subgraph containing only specified nodes."""
        # This might be overridden by subclasses for efficiency
        nodes_map = {}
        for node_id in node_ids:
            node = self.get_node(node_id)
            if node:
                nodes_map[node_id] = node
        
        # Create subgraph adapter
        return _SubgraphAdapter(self, node_ids)


# ============================================================================
# CONCRETE IMPLEMENTATIONS
# ============================================================================

class ConcreteGraphImpl(GenieGraph):
    """
    Concrete implementation backed by ConcreteNode objects.
    
    Most efficient: direct access to nodes, precomputed orderings.
    Used for scheduler output and post-optimization graphs.
    """
    
    def __init__(self, nodes: Optional[List[ConcreteNode]] = None):
        self._nodes: Dict[str, ConcreteNode] = {}
        self._node_list: List[ConcreteNode] = []
        self._topological_order: Optional[List[ConcreteNode]] = None
        self._num_edges: Optional[int] = None
        
        if nodes:
            for node in nodes:
                self._add_node(node)
            self._compute_topological_order()
    
    def _add_node(self, node: ConcreteNode) -> None:
        """Add node to graph."""
        if node.id not in self._nodes:
            self._nodes[node.id] = node
            self._node_list.append(node)
            # Invalidate cached orderings
            self._topological_order = None
            self._num_edges = None
    
    def nodes(self) -> Iterator[ConcreteNode]:
        """Iterate nodes (in insertion order)."""
        return iter(self._node_list)
    
    def get_node(self, node_id: str) -> Optional[ConcreteNode]:
        """Get node by ID (O(1))."""
        return self._nodes.get(node_id)
    
    def topological_order(self) -> List[ConcreteNode]:
        """Get nodes in execution order."""
        if self._topological_order is None:
            self._compute_topological_order()
        return self._topological_order
    
    def _compute_topological_order(self) -> None:
        """Compute topological order using Kahn's algorithm."""
        # Build in-degree map
        in_degree = {node.id: 0 for node in self._node_list}
        for node in self._node_list:
            for inp in node.inputs:
                if inp.id in in_degree:
                    in_degree[node.id] += 1
        
        # Topological sort (Kahn)
        queue = [node for node in self._node_list if in_degree[node.id] == 0]
        order = []
        
        while queue:
            node = queue.pop(0)
            order.append(node)
            
            for output in node.outputs:
                if output.id in in_degree:
                    in_degree[output.id] -= 1
                    if in_degree[output.id] == 0:
                        queue.append(output)
        
        # Check for cycles
        if len(order) != len(self._node_list):
            logger.warning(f"Cycle detected in graph ({len(order)}/{len(self._node_list)} nodes)")
        
        self._topological_order = order

    @property
    def num_nodes(self) -> int:
        """Number of nodes (O(1))."""
        return len(self._node_list)

    @property
    def num_edges(self) -> int:
        """Number of edges (O(N))."""
        if self._num_edges is None:
            self._num_edges = sum(len(node.inputs) for node in self._node_list)
        return self._num_edges


class FXGraphAdapter(GenieGraph):
    """
    Adapter for torch.fx.GraphModule.
    
    Translates FX representation to GenieGraph interface.
    """
    
    def __init__(self, fx_graph: torch.fx.Graph):
        self._fx_graph = fx_graph
        self._node_cache: Optional[Dict[str, NodeProtocol]] = None
        self._topological_order_cache: Optional[List[NodeProtocol]] = None
    
    def nodes(self) -> Iterator[NodeProtocol]:
        """Iterate FX nodes."""
        for fx_node in self._fx_graph.nodes:
            if fx_node.op in ('call_function', 'call_method', 'call_module'):
                yield _FXNodeAdapter(fx_node)
            elif fx_node.op == 'placeholder':
                yield _FXNodeAdapter(fx_node)
            elif fx_node.op == 'output':
                yield _FXNodeAdapter(fx_node)
    
    def get_node(self, node_id: str) -> Optional[NodeProtocol]:
        """Get node by ID (lazy caching)."""
        if self._node_cache is None:
            self._build_node_cache()
        return self._node_cache.get(node_id)
    
    def _build_node_cache(self) -> None:
        """Build cache for O(1) lookups."""
        self._node_cache = {}
        for fx_node in self._fx_graph.nodes:
            adapter = _FXNodeAdapter(fx_node)
            self._node_cache[adapter.id] = adapter
    
    def topological_order(self) -> List[NodeProtocol]:
        """Get nodes in execution order."""
        if self._topological_order_cache is None:
            self._topological_order_cache = list(self.nodes())
        return self._topological_order_cache

    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return len(list(self._fx_graph.nodes))

    @property
    def num_edges(self) -> int:
        """Number of edges."""
        return sum(len(list(node.all_input_nodes)) for node in self._fx_graph.nodes)


class DictGraphAdapter(GenieGraph):
    """
    Adapter for dict-based graph representations (legacy).
    
    Enables gradual migration to ConcreteGraph.
    """
    
    def __init__(self, graph_dict: Dict[str, Any]):
        self._dict = graph_dict
        self._nodes = graph_dict.get('nodes', [])
        self._node_cache: Optional[Dict[str, DictNodeAdapter]] = None
    
    def nodes(self) -> Iterator[DictNodeAdapter]:
        """Iterate over nodes."""
        for node_dict in self._nodes:
            yield DictNodeAdapter(node_dict)
    
    def get_node(self, node_id: str) -> Optional[DictNodeAdapter]:
        """Get node by ID."""
        if self._node_cache is None:
            self._build_node_cache()
        return self._node_cache.get(node_id)
    
    def _build_node_cache(self) -> None:
        """Build node cache."""
        self._node_cache = {}
        for node_dict in self._nodes:
            adapter = DictNodeAdapter(node_dict)
            self._node_cache[adapter.id] = adapter
    
    def topological_order(self) -> List[DictNodeAdapter]:
        """Get nodes in order (assuming dict preserves order)."""
        return list(self.nodes())

    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        """Number of edges."""
        return sum(len(DictNodeAdapter(n).inputs) for n in self._nodes)


# ============================================================================
# INTERNAL ADAPTERS
# ============================================================================

class _FXNodeAdapter:
    """Adapter to make torch.fx.Node look like NodeProtocol."""
    
    def __init__(self, fx_node: torch.fx.Node):
        self._node = fx_node

    @property
    def id(self) -> str:
        return self._node.name

    @property
    def operation(self) -> str:
        if self._node.op == 'call_function':
            return str(self._node.target)
        elif self._node.op == 'call_method':
            return f"method:{self._node.target}"
        elif self._node.op == 'call_module':
            return f"module:{self._node.target}"
        return self._node.op

    @property
    def inputs(self) -> List[NodeProtocol]:
        return [_FXNodeAdapter(inp) for inp in self._node.all_input_nodes]
    
    @property
    def outputs(self) -> List[NodeProtocol]:
        return [_FXNodeAdapter(user) for user, _ in self._node.users]
    
    @property
    def shape(self) -> Optional[torch.Size]:
        meta = self._node.meta
        if 'tensor_meta' in meta:
            return meta['tensor_meta'].shape
        return None
    
    @property
    def dtype(self) -> Optional[torch.dtype]:
        meta = self._node.meta
        if 'tensor_meta' in meta:
            return meta['tensor_meta'].dtype
        return None

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._node.meta.get('genie_metadata', {})


class _SubgraphAdapter(GenieGraph):
    """Adapter for viewing a subgraph of another graph."""
    
    def __init__(self, parent_graph: GenieGraph, node_ids: set[str]):
        self._parent = parent_graph
        self._node_ids = node_ids
        self._nodes_cache: Optional[List[NodeProtocol]] = None
    
    def nodes(self) -> Iterator[NodeProtocol]:
        """Iterate only over selected nodes."""
        for node in self._parent.nodes():
            if node.id in self._node_ids:
                yield node
    
    def get_node(self, node_id: str) -> Optional[NodeProtocol]:
        """Get node if it's in subgraph."""
        if node_id in self._node_ids:
            return self._parent.get_node(node_id)
        return None
    
    def topological_order(self) -> List[NodeProtocol]:
        """Get subgraph nodes in topological order."""
        return [n for n in self._parent.topological_order() if n.id in self._node_ids]
    
    @property
    def num_nodes(self) -> int:
        return len(self._node_ids)
    
    @property
    def num_edges(self) -> int:
        count = 0
        for node in self.nodes():
            for inp in node.inputs:
                if inp.id in self._node_ids:
                    count += 1
        return count


# ============================================================================
# CONVENIENCE CONSTRUCTORS
# ============================================================================

def create_concrete_graph(nodes: List[ConcreteNode]) -> ConcreteGraphImpl:
    """Create a concrete graph from nodes."""
    return ConcreteGraphImpl(nodes)


def create_fx_graph(fx_module: torch.fx.GraphModule) -> FXGraphAdapter:
    """Create adapter for FX module."""
    return FXGraphAdapter(fx_module.graph)


def create_dict_graph(graph_dict: Dict[str, Any]) -> DictGraphAdapter:
    """Create adapter for dict graph."""
    return DictGraphAdapter(graph_dict)


# ============================================================================
# BACKWARD COMPATIBILITY: Legacy Names
# ============================================================================

# Alias old names to new implementations for backward compatibility
Graph = GenieGraph  # Old name -> New name
FXGraphAdapter = FXGraphAdapter  # Already correct

# Create LazyDAGAdapter as a backward-compatible alias
class LazyDAGAdapter(GenieGraph):
    """
    Adapter for LazyTensor DAG (backward compatibility).
    
    This is a stub that delegates to the parent graph interface.
    In practice, should wrap a LazyTensor root and traverse it.
    """
    
    def __init__(self, root_tensor=None):
        """Initialize with root LazyTensor."""
        self.root_tensor = root_tensor
        self._nodes_list = []
        if root_tensor is not None:
            self._collect_nodes()
    
    def _collect_nodes(self):
        """Collect all nodes from LazyTensor graph."""
        if self.root_tensor is None:
            return
        
        # Import here to avoid circular imports
        try:
            from .lazy_tensor import LazyTensor
            visited = set()
            
            def traverse(node):
                if not isinstance(node, LazyTensor):
                    return
                node_id = id(node)
                if node_id in visited:
                    return
                visited.add(node_id)
                # Visit inputs first
                for inp in node.inputs:
                    traverse(inp)
                # Create adapter wrapper for each node
                self._nodes_list.append(node)
            
            traverse(self.root_tensor)
        except Exception:
            pass  # Graceful degradation
    
    def nodes(self) -> Iterator:
        """Iterate over nodes."""
        return iter(self._nodes_list)
    
    def get_node(self, node_id: str) -> Optional:
        """Get node by ID."""
        for node in self._nodes_list:
            if str(id(node)) == node_id:
                return node
        return None
    
    def topological_order(self) -> List:
        """Get nodes in topological order."""
        # Already collected in reverse topological order
        return list(reversed(self._nodes_list))

    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return len(self._nodes_list)

    @property
    def num_edges(self) -> int:
        """Number of edges."""
        count = 0
        for node in self._nodes_list:
            if hasattr(node, 'inputs'):
                count += len(node.inputs)
        return count


__all__ = [
    'GenieGraph',
    'ConcreteGraphImpl',
    'FXGraphAdapter',
    'DictGraphAdapter',
    'LazyDAGAdapter',
    'Graph',  # Backward compat
    'create_concrete_graph',
    'create_fx_graph',
    'create_dict_graph',
    'NodeProtocol',
]