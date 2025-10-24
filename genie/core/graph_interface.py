"""
Unified graph interface supporting both FX and LazyTensor DAG.

This module provides a common API that abstracts over the underlying representation,
enabling the hybrid graph strategy specified in the enhancement plan.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import torch.fx as fx
import logging

logger = logging.getLogger(__name__)


class GraphNode(ABC):
    """Abstract node in computation graph."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique node identifier."""
        pass

    @property
    @abstractmethod
    def operation(self) -> str:
        """Operation name (e.g., 'aten::add')."""
        pass

    @property
    @abstractmethod
    def inputs(self) -> List['GraphNode']:
        """Input nodes."""
        pass

    @abstractmethod
    def get_consumers(self) -> List['GraphNode']:
        """Nodes that consume this node's output."""
        pass

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Semantic metadata."""
        pass


class Graph(ABC):
    """Abstract computation graph."""

    @abstractmethod
    def nodes(self) -> List[GraphNode]:
        """Get all nodes in topological order."""
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID."""
        pass

    @abstractmethod
    def topological_sort(self) -> List[GraphNode]:
        """Get nodes in execution order."""
        pass

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Backend type: 'fx' or 'lazy_dag'."""
        pass


class FXGraphAdapter(Graph):
    """Adapter for torch.fx.Graph with optimized performance."""

    def __init__(self, fx_graph: fx.Graph):
        self.fx_graph = fx_graph
        self._nodes_cache = None
        self._node_map = {}  # O(1) lookup optimization

    def nodes(self) -> List[GraphNode]:
        if self._nodes_cache is None:
            # Use FX's built-in node filtering for better performance
            operation_nodes = [node for node in self.fx_graph.nodes
                             if node.op in ('call_function', 'call_method')]
            self._nodes_cache = [FXNodeAdapter(node) for node in operation_nodes]

            # Build node map for O(1) lookup
            for node_adapter in self._nodes_cache:
                self._node_map[node_adapter.id] = node_adapter

        return self._nodes_cache

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        # O(1) hash-based lookup instead of O(n) linear search
        if not self._node_map:
            # Ensure nodes are loaded
            _ = self.nodes()
        return self._node_map.get(node_id)

    def topological_sort(self) -> List[GraphNode]:
        """Get nodes in topological execution order."""
        try:
            from torch.fx.passes.graph_utils import get_topo_order

            ordered_nodes = get_topo_order(self.fx_graph)

            # Filter to only operation nodes and adapt
            return [FXNodeAdapter(node) for node in ordered_nodes
                   if node.op in ('call_function', 'call_method')]

        except (ImportError, AttributeError, Exception) as e:
            # Fallback: assume FX is already sorted
            logger.debug(f"get_topo_order not available: {e}, using cached order")
            return self.nodes()

    @property
    def backend_type(self) -> str:
        return 'fx'


class FXNodeAdapter(GraphNode):
    """Adapter for torch.fx.Node with improved operation handling."""

    def __init__(self, fx_node: fx.Node):
        self.fx_node = fx_node

    @property
    def id(self) -> str:
        return self.fx_node.name

    @property
    def operation(self) -> str:
        # Better operation name formatting using FX utilities
        if self.fx_node.op == 'call_function':
            # Use function name if available, otherwise use target string
            if hasattr(self.fx_node.target, '__name__'):
                return f"aten::{self.fx_node.target.__name__}"
            else:
                return f"aten::{str(self.fx_node.target)}"
        elif self.fx_node.op == 'call_method':
            return f"aten::{self.fx_node.target}"
        else:
            return str(self.fx_node.target)

    @property
    def inputs(self) -> List[GraphNode]:
        # Use FX's built-in user tracking for more accurate inputs
        inputs = []

        # Add explicit args that are nodes
        for arg in self.fx_node.args:
            if isinstance(arg, fx.Node):
                inputs.append(FXNodeAdapter(arg))

        # Add explicit kwargs that are nodes
        for kwarg in self.fx_node.kwargs.values():
            if isinstance(kwarg, fx.Node):
                inputs.append(FXNodeAdapter(kwarg))

        return inputs

    def get_consumers(self) -> List[GraphNode]:
        """Nodes that consume this node's output."""
        consumers = []
        for user in self.fx_node.users:
            consumers.append(FXNodeAdapter(user))
        return consumers

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.fx_node.meta.get('semantic', {})


class LazyDAGAdapter(Graph):
    """Adapter for LazyTensor DAG with optimized performance."""

    def __init__(self, root_tensor):
        from .lazy_tensor import LazyTensor
        self.root = root_tensor
        self._nodes_cache = None
        self._node_map = {}  # O(1) lookup optimization

    def nodes(self) -> List[GraphNode]:
        if self._nodes_cache is None:
            self._nodes_cache = self._collect_nodes()

            # Build node map for O(1) lookup
            for node_adapter in self._nodes_cache:
                self._node_map[node_adapter.id] = node_adapter

        return self._nodes_cache

    def _collect_nodes(self) -> List[GraphNode]:
        """Collect all nodes reachable from root using optimized traversal."""
        from .lazy_tensor import LazyTensor

        visited = set()
        result = []

        def visit(tensor):
            if not isinstance(tensor, LazyTensor):
                return
            tensor_id = id(tensor)
            if tensor_id in visited:
                return
            visited.add(tensor_id)

            # Visit inputs first (post-order traversal for topological order)
            for inp in tensor.inputs:
                visit(inp)

            # Create adapter and add to result
            node_adapter = LazyDAGNodeAdapter(tensor)
            node_adapter._parent_adapter = self  # Set parent reference for consumer lookup
            result.append(node_adapter)

        visit(self.root)
        return result

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        # O(1) hash-based lookup instead of O(n) linear search
        if not self._node_map:
            # Ensure nodes are loaded
            _ = self.nodes()
        return self._node_map.get(node_id)

    def topological_sort(self) -> List[GraphNode]:
        # Nodes are already collected in topological order from _collect_nodes()
        return self.nodes()

    @property
    def backend_type(self) -> str:
        return 'lazy_dag'


class LazyDAGNodeAdapter(GraphNode):
    """Adapter for LazyTensor node."""

    def __init__(self, lazy_tensor):
        self.tensor = lazy_tensor
        self._metadata_cache = None  # Cache for metadata lookups

    @property
    def id(self) -> str:
        return str(self.tensor.tensor_id)

    @property
    def operation(self) -> str:
        return self.tensor.operation

    @property
    def inputs(self) -> List[GraphNode]:
        from .lazy_tensor import LazyTensor
        return [
            LazyDAGNodeAdapter(inp)
            for inp in self.tensor.inputs
            if isinstance(inp, LazyTensor)
        ]

    def get_consumers(self) -> List[GraphNode]:
        """Nodes that consume this node's output."""
        # For LazyDAG, we need to traverse the graph to find consumers
        # This is expensive but necessary for pattern matching
        consumers = []

        # Get the adapter's parent graph to traverse
        parent_adapter = getattr(self, '_parent_adapter', None)
        if parent_adapter and hasattr(parent_adapter, 'nodes'):
            # Look through all nodes to find consumers
            for node in parent_adapter.nodes():
                for inp in node.inputs:
                    if inp.id == self.id:
                        consumers.append(node)
                        break

        return consumers

    @property
    def metadata(self) -> Dict[str, Any]:
        # First try to get metadata from the tensor itself (new approach)
        if hasattr(self.tensor, 'metadata'):
            self._metadata_cache = self.tensor.metadata
            return self._metadata_cache

        # Fallback: metadata stored separately in registry with caching
        if self._metadata_cache is None:
            try:
                from genie.semantic.metadata_registry import get_metadata_registry
                registry = get_metadata_registry()
                meta = registry.get_metadata(self.id)
                self._metadata_cache = meta.to_dict() if meta else {}
            except Exception:
                self._metadata_cache = {}
        return self._metadata_cache