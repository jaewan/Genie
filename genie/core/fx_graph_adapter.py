"""FX Graph Adapter - Unified graph representation with metadata integration.

This module implements Refactoring #3 (Unify Graph Representation) and coordinates
with Refactoring #2 (Separate Semantic Metadata). It provides:

1. FX Graph as single source of truth for graph structure
2. Integration with MetadataRegistry for semantic metadata
3. Backward compatibility during migration from ComputationGraph
4. Clean API for graph consumers

Design Coordination (Refactoring #2 + #3):
- FX meta stores: tensor_id, operation, shape, dtype (structural data)
- MetadataRegistry stores: semantic_role, execution_phase, etc. (semantic data)
- Bridge: Use tensor_id from FX meta to lookup in MetadataRegistry
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, Set, Tuple
import torch
import torch.fx as fx
from torch.fx import GraphModule, Node

logger = logging.getLogger(__name__)


class FXGraphAdapter:
    """Adapter for unified FX Graph access with MetadataRegistry integration.
    
    This class bridges FX Graph (structural) with MetadataRegistry (semantic),
    implementing the coordination strategy from Refactoring #2 and #3.
    
    Usage:
        adapter = FXGraphAdapter(fx_graph_module)
        
        # Access structural data from FX
        tensor_id = adapter.get_tensor_id(node)
        operation = adapter.get_operation(node)
        
        # Access semantic data from registry
        semantic_meta = adapter.get_semantic_metadata(node)
        if semantic_meta:
            role = semantic_meta.semantic_role
            phase = semantic_meta.execution_phase
    """
    
    def __init__(self, graph_module: Optional[GraphModule] = None):
        """Initialize adapter with optional GraphModule.
        
        Args:
            graph_module: FX GraphModule to adapt. If None, creates empty adapter.
        """
        self.graph_module = graph_module
        self._metadata_registry = None  # Lazy init
        
    @property
    def graph(self) -> fx.Graph:
        """Get underlying FX graph."""
        if self.graph_module is None:
            raise ValueError("No GraphModule attached to adapter")
        return self.graph_module.graph
    
    @property
    def metadata_registry(self):
        """Get metadata registry (lazy initialization)."""
        if self._metadata_registry is None:
            try:
                from genie.semantic.metadata_registry import get_metadata_registry
                self._metadata_registry = get_metadata_registry()
            except ImportError:
                logger.warning("MetadataRegistry not available (Refactoring #2 incomplete?)")
                return None
        return self._metadata_registry
    
    # === Structural Data Access (from FX meta) ===
    
    def get_tensor_id(self, node: Node) -> Optional[str]:
        """Get tensor ID for a node from FX meta.
        
        Args:
            node: FX node
            
        Returns:
            Tensor ID string (e.g., "lt_123") or None
        """
        # Check new format (Refactoring #3)
        if 'genie' in node.meta and 'tensor_id' in node.meta['genie']:
            return node.meta['genie']['tensor_id']
        
        # Check old format (backward compatibility)
        if 'lazy_tensor_id' in node.meta:
            return node.meta['lazy_tensor_id']
        
        return None
    
    def get_operation(self, node: Node) -> Optional[str]:
        """Get operation name for a node.
        
        Args:
            node: FX node
            
        Returns:
            Operation name (e.g., "aten::matmul") or None
        """
        # Check new format
        if 'genie' in node.meta and 'operation' in node.meta['genie']:
            return node.meta['genie']['operation']
        
        # Fallback: infer from node target
        if node.op == 'call_function':
            func_name = getattr(node.target, '__name__', str(node.target))
            return f"aten::{func_name}"
        elif node.op == 'call_method':
            return f"aten::{node.target}"
        
        return None
    
    def get_shape(self, node: Node) -> Optional[torch.Size]:
        """Get tensor shape for a node.
        
        Args:
            node: FX node
            
        Returns:
            torch.Size or None
        """
        # Check new format
        if 'genie' in node.meta and 'shape' in node.meta['genie']:
            shape = node.meta['genie']['shape']
            return torch.Size(shape) if isinstance(shape, (list, tuple)) else shape
        
        # Check old format
        if 'shape' in node.meta:
            shape = node.meta['shape']
            return torch.Size(shape) if isinstance(shape, (list, tuple)) else shape
        
        return None
    
    def get_dtype(self, node: Node) -> Optional[torch.dtype]:
        """Get tensor dtype for a node.
        
        Args:
            node: FX node
            
        Returns:
            torch.dtype or None
        """
        # Check new format
        if 'genie' in node.meta and 'dtype' in node.meta['genie']:
            dtype = node.meta['genie']['dtype']
            return dtype if isinstance(dtype, torch.dtype) else None
        
        # Check old format
        if 'dtype' in node.meta:
            dtype = node.meta['dtype']
            return dtype if isinstance(dtype, torch.dtype) else None
        
        return None
    
    # === Semantic Data Access (from MetadataRegistry) ===
    
    def get_semantic_metadata(self, node: Node):
        """Get semantic metadata for a node from MetadataRegistry.
        
        This bridges to Refactoring #2's MetadataRegistry using tensor_id.
        
        Args:
            node: FX node
            
        Returns:
            SemanticMetadata object or None
        """
        tensor_id = self.get_tensor_id(node)
        if not tensor_id:
            return None
        
        registry = self.metadata_registry
        if not registry:
            return None
        
        return registry.get_metadata(tensor_id)
    
    def has_semantic_metadata(self, node: Node) -> bool:
        """Check if node has semantic metadata available.
        
        Args:
            node: FX node
            
        Returns:
            True if semantic metadata exists
        """
        return self.get_semantic_metadata(node) is not None
    
    # === Graph Analysis ===
    
    def get_all_nodes(self) -> List[Node]:
        """Get all nodes in the graph.
        
        Returns:
            List of FX nodes
        """
        return list(self.graph.nodes)
    
    def get_operation_nodes(self) -> List[Node]:
        """Get all operation nodes (call_function, call_method).
        
        Returns:
            List of operation nodes
        """
        return [node for node in self.graph.nodes 
                if node.op in ('call_function', 'call_method')]
    
    def get_nodes_by_operation(self, operation: str) -> List[Node]:
        """Get all nodes with specific operation.
        
        Args:
            operation: Operation name (e.g., "aten::matmul")
            
        Returns:
            List of matching nodes
        """
        return [node for node in self.get_operation_nodes()
                if self.get_operation(node) == operation]
    
    def get_execution_order(self) -> List[Node]:
        """Get nodes in execution order (topological).
        
        Returns:
            List of nodes in execution order
        """
        # FX graphs are already in topological order
        return self.get_operation_nodes()
    
    def get_dependencies(self, node: Node) -> List[Node]:
        """Get direct dependencies (inputs) of a node.
        
        Args:
            node: FX node
            
        Returns:
            List of nodes that this node depends on
        """
        deps = []
        for arg in node.args:
            if isinstance(arg, Node):
                deps.append(arg)
        for kwarg in node.kwargs.values():
            if isinstance(kwarg, Node):
                deps.append(kwarg)
        return deps
    
    def get_users(self, node: Node) -> List[Node]:
        """Get nodes that use this node's output.
        
        Args:
            node: FX node
            
        Returns:
            List of user nodes
        """
        return list(node.users.keys())
    
    # === Graph Statistics ===
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics.
        
        Returns:
            Dictionary with graph statistics
        """
        op_nodes = self.get_operation_nodes()
        
        # Operation counts
        operation_counts = {}
        for node in op_nodes:
            op = self.get_operation(node)
            if op:
                operation_counts[op] = operation_counts.get(op, 0) + 1
        
        # Semantic statistics (if available)
        semantic_stats = {
            'nodes_with_semantic': 0,
            'execution_phases': {},
            'semantic_roles': {},
        }
        
        for node in op_nodes:
            semantic = self.get_semantic_metadata(node)
            if semantic:
                semantic_stats['nodes_with_semantic'] += 1
                
                if semantic.execution_phase:
                    phase = str(semantic.execution_phase)
                    semantic_stats['execution_phases'][phase] = \
                        semantic_stats['execution_phases'].get(phase, 0) + 1
                
                if semantic.semantic_role:
                    role = str(semantic.semantic_role)
                    semantic_stats['semantic_roles'][role] = \
                        semantic_stats['semantic_roles'].get(role, 0) + 1
        
        return {
            'num_nodes': len(self.get_all_nodes()),
            'num_operations': len(op_nodes),
            'operation_counts': operation_counts,
            'semantic_coverage': semantic_stats['nodes_with_semantic'] / len(op_nodes) if op_nodes else 0.0,
            'semantic_stats': semantic_stats,
        }
    
    # === Migration Helpers ===
    
    def ensure_genie_meta(self, node: Node, lazy_tensor=None) -> None:
        """Ensure node has proper genie meta structure.
        
        This helps migrate from old format to new unified format.
        
        Args:
            node: FX node to update
            lazy_tensor: Optional LazyTensor to extract metadata from
        """
        # Skip if already has new format
        if 'genie' in node.meta and 'tensor_id' in node.meta['genie']:
            return
        
        # Create genie meta structure
        genie_meta = {}
        
        # Extract tensor_id
        if lazy_tensor:
            genie_meta['tensor_id'] = lazy_tensor.id
            genie_meta['operation'] = lazy_tensor.operation
            genie_meta['shape'] = lazy_tensor.shape
            genie_meta['dtype'] = lazy_tensor.dtype
        else:
            # Try to extract from old format
            if 'lazy_tensor_id' in node.meta:
                genie_meta['tensor_id'] = node.meta['lazy_tensor_id']
            if 'shape' in node.meta:
                genie_meta['shape'] = node.meta['shape']
            if 'dtype' in node.meta:
                genie_meta['dtype'] = node.meta['dtype']
            
            # Infer operation
            if node.op == 'call_function':
                func_name = getattr(node.target, '__name__', str(node.target))
                genie_meta['operation'] = f"aten::{func_name}"
            elif node.op == 'call_method':
                genie_meta['operation'] = f"aten::{node.target}"
        
        # Set unified meta
        node.meta['genie'] = genie_meta
    
    def migrate_all_nodes(self) -> int:
        """Migrate all nodes to new genie meta format.
        
        Returns:
            Number of nodes migrated
        """
        migrated = 0
        for node in self.get_operation_nodes():
            if 'genie' not in node.meta or 'tensor_id' not in node.meta['genie']:
                self.ensure_genie_meta(node)
                migrated += 1
        
        logger.info(f"Migrated {migrated} nodes to new genie meta format")
        return migrated
    
    # === Compatibility with Old ComputationGraph ===
    
    @classmethod
    def from_computation_graph(cls, comp_graph, graph_builder=None) -> FXGraphAdapter:
        """Create FXGraphAdapter from old ComputationGraph.
        
        This provides backward compatibility during migration.
        
        Args:
            comp_graph: Legacy ComputationGraph
            graph_builder: Optional GraphBuilder for LazyTensor access
            
        Returns:
            FXGraphAdapter instance
        """
        from .fx_graph_builder import FXGraphBuilder
        
        # Build FX graph from computation graph
        fx_builder = FXGraphBuilder()
        
        # Process nodes in topological order
        node_order = comp_graph.topological_sort()
        
        for node_id in node_order:
            if node_id not in comp_graph.nodes:
                continue
            
            node = comp_graph.nodes[node_id]
            
            # Get LazyTensor if available
            lazy_tensor = None
            if graph_builder:
                lazy_tensor = graph_builder.get_lazy_tensor(node_id)
            
            if lazy_tensor:
                fx_builder.add_lazy_tensor(lazy_tensor)
        
        # Convert to GraphModule
        graph_module = fx_builder.to_graph_module()
        
        # Create adapter
        adapter = cls(graph_module)
        adapter.migrate_all_nodes()
        
        return adapter
    
    # === String Representation ===
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"FXGraphAdapter(nodes={stats['num_nodes']}, "
                f"operations={stats['num_operations']}, "
                f"semantic_coverage={stats['semantic_coverage']:.1%})")


def get_current_fx_adapter() -> Optional[FXGraphAdapter]:
    """Get FX adapter for current graph builder.
    
    This provides a migration path from GraphBuilder.current() to FXGraphAdapter.
    
    Returns:
        FXGraphAdapter for current graph or None
    """
    try:
        from .fx_graph_builder import FXGraphBuilder
        
        fx_builder = FXGraphBuilder.current()
        graph_module = fx_builder.to_graph_module()
        
        adapter = FXGraphAdapter(graph_module)
        adapter.migrate_all_nodes()
        
        return adapter
    except Exception as e:
        logger.debug(f"Could not get current FX adapter: {e}")
        return None

