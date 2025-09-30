"""FX-based graph builder for Genie.

This module replaces the custom ComputationGraph with PyTorch FX's native
graph representation, enabling better integration with PyTorch's optimization
and analysis tools.
"""

import threading
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.fx as fx
from torch.fx import Graph, Node, GraphModule
import logging

from .semantic_metadata import SemanticMetadata

logger = logging.getLogger(__name__)


class FXGraphBuilder:
    """Build FX graphs from LazyTensor operations.
    
    This replaces the custom GraphBuilder with FX-based implementation,
    providing native PyTorch graph representation with semantic metadata.
    """
    
    _thread_local = threading.local()
    
    def __init__(self):
        self.fx_graph = Graph()
        self.value_map: Dict[str, Node] = {}  # LazyTensor.id -> fx.Node mapping
        self.lazy_tensor_map: Dict[str, Any] = {}  # LazyTensor.id -> LazyTensor
        self.placeholders: List[Node] = []  # Input nodes
        self.outputs: List[Node] = []  # Output nodes
        self._finalized = False
        
    @classmethod
    def current(cls) -> "FXGraphBuilder":
        """Get thread-local FX graph builder."""
        if not hasattr(cls._thread_local, "builder"):
            cls._thread_local.builder = cls()
        return cls._thread_local.builder
    
    @classmethod
    def reset(cls) -> None:
        """Reset thread-local builder."""
        if hasattr(cls._thread_local, "builder"):
            delattr(cls._thread_local, "builder")
    
    def add_lazy_tensor(self, lazy_tensor) -> Node:
        """Add a LazyTensor operation to the FX graph.
        
        Args:
            lazy_tensor: LazyTensor instance to add
            
        Returns:
            FX Node representing this operation
        """
        # Avoid circular import
        from .lazy_tensor import LazyTensor
        
        if self._finalized:
            # Allow retrieving already-added nodes after finalization
            if lazy_tensor.id in self.value_map:
                return self.value_map[lazy_tensor.id]
            # Re-open the graph to accept new nodes (test-friendly behavior)
            self._finalized = False
        
        # Check if already added
        if lazy_tensor.id in self.value_map:
            return self.value_map[lazy_tensor.id]
        
        # Store lazy tensor reference
        self.lazy_tensor_map[lazy_tensor.id] = lazy_tensor
        
        # Process inputs and build dependency nodes
        fx_args = []
        fx_kwargs = {}
        
        for inp in lazy_tensor.inputs:
            if isinstance(inp, LazyTensor):
                # Recursively add dependency
                inp_node = self.add_lazy_tensor(inp)
                fx_args.append(inp_node)
            elif isinstance(inp, (torch.Tensor, torch.nn.Parameter)):
                # Create placeholder for concrete tensor
                placeholder = self._get_or_create_placeholder(inp)
                fx_args.append(placeholder)
            else:
                # Constant value
                fx_args.append(inp)
        
        # Process kwargs
        for key, value in lazy_tensor.kwargs.items():
            if isinstance(value, LazyTensor):
                fx_kwargs[key] = self.add_lazy_tensor(value)
            elif isinstance(value, (torch.Tensor, torch.nn.Parameter)):
                fx_kwargs[key] = self._get_or_create_placeholder(value)
            else:
                fx_kwargs[key] = value
        
        # Map operation name to torch function
        torch_func = self._get_torch_function(lazy_tensor.operation)
        
        # Create FX node
        if torch_func:
            node = self.fx_graph.call_function(
                torch_func,
                tuple(fx_args) if fx_args else (),
                fx_kwargs if fx_kwargs else {}
            )
        else:
            # Fallback to method call
            node = self.fx_graph.call_method(
                lazy_tensor.operation.split("::")[-1],  # Extract method name
                tuple(fx_args) if fx_args else (),
                fx_kwargs if fx_kwargs else {}
            )
        
        # Attach metadata in new unified format (Refactoring #3)
        # Uses 'genie' namespace to coordinate with Refactoring #2's MetadataRegistry
        node.meta['genie'] = {
            'tensor_id': lazy_tensor.id,      # Key for MetadataRegistry lookup
            'operation': lazy_tensor.operation,  # Duplicated for convenience
            'shape': lazy_tensor.shape,       # Graph-structural info
            'dtype': lazy_tensor.dtype,       # Graph-structural info
            'device': lazy_tensor.device,     # Graph-structural info
        }
        
        # Keep old format for backward compatibility (will be removed in Refactoring #3 complete)
        node.meta['semantic'] = lazy_tensor.metadata
        node.meta['lazy_tensor_id'] = lazy_tensor.id
        node.meta['shape'] = lazy_tensor.shape
        node.meta['dtype'] = lazy_tensor.dtype
        node.meta['device'] = lazy_tensor.device
        
        # Store mapping
        self.value_map[lazy_tensor.id] = node
        
        logger.debug(f"Added FX node for {lazy_tensor.operation} (id={lazy_tensor.id})")
        
        return node
    
    def _get_or_create_placeholder(self, tensor: torch.Tensor) -> Node:
        """Get or create a placeholder node for a concrete tensor."""
        # Use tensor's data pointer as unique ID
        tensor_id = str(tensor.data_ptr())
        
        if tensor_id in self.value_map:
            return self.value_map[tensor_id]
        
        # Create placeholder
        placeholder = self.fx_graph.placeholder(f"input_{len(self.placeholders)}")
        placeholder.meta['tensor_meta'] = {
            'shape': list(tensor.shape),
            'dtype': tensor.dtype,
            'device': str(tensor.device),
            'requires_grad': tensor.requires_grad
        }
        
        self.value_map[tensor_id] = placeholder
        self.placeholders.append(placeholder)
        
        return placeholder
    
    def _get_torch_function(self, operation: str) -> Optional[Any]:
        """Map operation name to torch function."""
        # Remove aten:: prefix
        op_name = operation.replace("aten::", "")
        
        # Common operations mapping
        op_map = {
            "add": torch.add,
            "sub": torch.sub,
            "mul": torch.mul,
            "div": torch.div,
            "matmul": torch.matmul,
            "mm": torch.mm,
            "bmm": torch.bmm,
            "relu": torch.relu,
            "gelu": torch.nn.functional.gelu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "softmax": torch.softmax,
            "layer_norm": torch.nn.functional.layer_norm,
            "dropout": torch.nn.functional.dropout,
            "conv2d": torch.nn.functional.conv2d,
            "max_pool2d": torch.nn.functional.max_pool2d,
            "linear": torch.nn.functional.linear,
            "embedding": torch.nn.functional.embedding,
            "transpose": torch.transpose,
            "reshape": torch.reshape,
            "view": torch.Tensor.view,
            "squeeze": torch.squeeze,
            "unsqueeze": torch.unsqueeze,
            "cat": torch.cat,
            "stack": torch.stack,
            "sum": torch.sum,
            "mean": torch.mean,
            "randn": torch.randn,
            "zeros": torch.zeros,
            "ones": torch.ones,
            "alias": lambda x: x,  # Identity for alias
        }
        
        if op_name in op_map:
            return op_map[op_name]
        
        # Try to get from torch.ops.aten
        if hasattr(torch.ops.aten, op_name):
            return getattr(torch.ops.aten, op_name)
        
        # Try direct torch attribute
        if hasattr(torch, op_name):
            return getattr(torch, op_name)
        
        return None
    
    def mark_output(self, lazy_tensor) -> None:
        """Mark a LazyTensor as graph output."""
        from .lazy_tensor import LazyTensor
        
        if not isinstance(lazy_tensor, LazyTensor):
            return
        
        # Ensure it's in the graph
        node = self.add_lazy_tensor(lazy_tensor)
        
        if node not in self.outputs:
            self.outputs.append(node)
    
    def finalize(self) -> None:
        """Finalize the graph construction."""
        if self._finalized:
            return
        
        # Create output node
        if self.outputs:
            if len(self.outputs) == 1:
                self.fx_graph.output(self.outputs[0])
            else:
                # Multiple outputs - create tuple
                self.fx_graph.output(tuple(self.outputs))
        else:
            # No explicit outputs - use all nodes without users
            leaf_nodes = []
            for node in self.fx_graph.nodes:
                if node.op == 'call_function' and not node.users:
                    leaf_nodes.append(node)
            
            if leaf_nodes:
                if len(leaf_nodes) == 1:
                    self.fx_graph.output(leaf_nodes[0])
                else:
                    self.fx_graph.output(tuple(leaf_nodes))
        
        self._finalized = True
        logger.info(f"Finalized FX graph with {len(list(self.fx_graph.nodes))} nodes")
    
    def to_graph_module(self, trace_module: Optional[torch.nn.Module] = None) -> GraphModule:
        """Convert to FX GraphModule.
        
        Args:
            trace_module: Optional module to use as container
            
        Returns:
            GraphModule that can be executed or optimized
        """
        self.finalize()
        
        # Create GraphModule
        if trace_module is None:
            trace_module = torch.nn.Module()
        
        gm = GraphModule(trace_module, self.fx_graph)
        
        # Attach semantic metadata to GraphModule
        gm.meta['semantic_metadata'] = {
            node.name: node.meta.get('semantic', None)
            for node in self.fx_graph.nodes
            if 'semantic' in node.meta
        }
        
        return gm
    
    def get_execution_order(self) -> List[str]:
        """Get topological execution order of LazyTensor IDs."""
        self.finalize()
        
        order = []
        for node in self.fx_graph.nodes:
            if 'lazy_tensor_id' in node.meta:
                order.append(node.meta['lazy_tensor_id'])
        
        return order
    
    def visualize(self, filename: str = "fx_graph") -> None:
        """Visualize the FX graph.
        
        Args:
            filename: Output filename (without extension)
        """
        try:
            from torch.fx.passes.graph_drawer import FxGraphDrawer
            
            self.finalize()
            gm = self.to_graph_module()
            
            drawer = FxGraphDrawer(gm, filename)
            drawer.get_dot_graph().render(filename, format='pdf')
            logger.info(f"Graph visualization saved to {filename}.pdf")
        except ImportError:
            logger.warning("Graph visualization requires graphviz. Install with: pip install graphviz")
    
    def get_semantic_summary(self) -> Dict[str, Any]:
        """Get semantic summary of the graph."""
        self.finalize()
        
        summary = {
            'num_nodes': len(list(self.fx_graph.nodes)),
            'num_placeholders': len(self.placeholders),
            'num_outputs': len(self.outputs),
            'operations': {},
            'execution_phases': {},
            'memory_patterns': {},
            'compute_intensity': {
                'high': [],
                'medium': [],
                'low': []
            }
        }
        
        for node in self.fx_graph.nodes:
            if node.op == 'call_function':
                # Count operations
                op_name = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)
                summary['operations'][op_name] = summary['operations'].get(op_name, 0) + 1
                
                # Analyze semantic metadata
                if 'semantic' in node.meta:
                    metadata = node.meta['semantic']
                    
                    # Execution phases
                    if metadata.execution_phase:
                        phase = str(metadata.execution_phase.value if hasattr(metadata.execution_phase, 'value') else metadata.execution_phase)
                        summary['execution_phases'][phase] = summary['execution_phases'].get(phase, 0) + 1
                    
                    # Memory patterns
                    if metadata.memory_pattern:
                        pattern = str(metadata.memory_pattern.value if hasattr(metadata.memory_pattern, 'value') else metadata.memory_pattern)
                        summary['memory_patterns'][pattern] = summary['memory_patterns'].get(pattern, 0) + 1
                    
                    # Compute intensity
                    if metadata.compute_intensity:
                        lt_id = node.meta.get('lazy_tensor_id', node.name)
                        if metadata.compute_intensity >= 10:
                            summary['compute_intensity']['high'].append(lt_id)
                        elif metadata.compute_intensity >= 5:
                            summary['compute_intensity']['medium'].append(lt_id)
                        else:
                            summary['compute_intensity']['low'].append(lt_id)
        
        return summary


def migrate_from_computation_graph(comp_graph) -> FXGraphBuilder:
    """Migrate from old ComputationGraph to FX-based graph.
    
    Args:
        comp_graph: Legacy ComputationGraph instance
        
    Returns:
        FXGraphBuilder with migrated graph
    """
    fx_builder = FXGraphBuilder()
    
    # Import here to avoid circular dependency
    from .graph import ComputationNode
    
    # Build node order for topological processing
    node_order = comp_graph.topological_sort() if hasattr(comp_graph, 'topological_sort') else list(comp_graph.nodes.keys())
    
    # Process nodes in topological order
    for node_id in node_order:
        if node_id in comp_graph.nodes:
            node = comp_graph.nodes[node_id]
            
            # Get corresponding LazyTensor if available
            lazy_tensor = comp_graph.lazy_tensor_map.get(node_id)
            
            if lazy_tensor:
                fx_builder.add_lazy_tensor(lazy_tensor)
    
    logger.info(f"Migrated {len(comp_graph.nodes)} nodes from ComputationGraph to FX")
    
    return fx_builder
