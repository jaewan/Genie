"""
Differential Graph Transfer Protocol - Week 3 Optimization.

Problem: In iterative generation (e.g., LLM decoding), the computation graph changes
slightly each iteration, but we send the entire 5MB+ graph every time.

Solution: Send only the differences between consecutive graphs, reducing network
traffic from 5MB to ~500KB per request (10x reduction).

Usage:
    protocol = DifferentialGraphProtocol()
    
    # First inference - send full graph
    protocol.send_graph("gpt2_model", full_graph, is_update=False)
    
    # Second inference - send only delta (10% of first)
    protocol.send_graph("gpt2_model", updated_graph, is_update=True)
    
    # Result: 90% network traffic reduction
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class GraphDelta:
    """Represents the differences between two computation graphs."""
    graph_id: str
    version: int  # Version number of this graph
    
    # Operations that were added
    added_operations: List[Dict[str, Any]]
    
    # Operations that were removed
    removed_operation_ids: List[int]
    
    # Operations whose metadata changed (but structure stayed same)
    modified_operations: Dict[int, Dict[str, Any]]
    
    # Input tensor changes
    input_changes: Dict[str, Any]
    
    # Size in bytes when serialized
    serialized_size: int = 0
    
    def estimate_compression(self, full_graph_size: int) -> float:
        """Estimate compression ratio vs full graph."""
        if full_graph_size > 0:
            return self.serialized_size / full_graph_size
        return 1.0


class DifferentialGraphProtocol:
    """
    Protocol for efficient graph transfer in iterative workloads.
    
    Maintains server-side cache of previously sent graphs and computes
    only the differences needed for updates.
    """
    
    def __init__(self, max_cache_entries: int = 100):
        """
        Initialize the differential protocol.
        
        Args:
            max_cache_entries: Maximum number of cached graphs on server (must be >= 1)
        
        Raises:
            ValueError: If max_cache_entries is invalid
        """
        if max_cache_entries < 1:
            raise ValueError(
                f"max_cache_entries must be >= 1, got {max_cache_entries}"
            )
        self.max_cache_entries = max_cache_entries
        
        # Client-side tracking (use OrderedDict for LRU support)
        from collections import OrderedDict
        self.client_graphs: 'OrderedDict[str, Dict[str, Any]]' = OrderedDict()  # graph_id -> graph_data
        self.client_versions: Dict[str, int] = {}  # graph_id -> version number
        
        # Server-side cache (mirrored locally for computing diffs, use OrderedDict for LRU)
        self.server_cache: 'OrderedDict[str, Dict[str, Any]]' = OrderedDict()
        self.server_versions: Dict[str, int] = {}
        
        # Statistics
        self.stats = {
            'full_graphs_sent': 0,
            'delta_updates_sent': 0,
            'total_traffic_saved_mb': 0.0,
            'total_full_size_mb': 0.0,
            'total_delta_size_mb': 0.0,
        }
    
    def send_graph(self, 
                   graph_id: str,
                   graph: Dict[str, Any],
                   is_update: bool = False) -> Dict[str, Any]:
        """
        Send graph to server, using differential update if possible.
        
        Args:
            graph_id: Unique identifier for this graph (e.g., model class name)
            graph: The full computation graph (operations + metadata)
            is_update: Whether this is an update to a previously sent graph
            
        Returns:
            Message to send to server (either full graph or delta)
        """
        graph_size = self._estimate_size(graph)
        
        # First time - send full graph
        if not is_update or graph_id not in self.server_cache:
            # Enforce max cache size (LRU eviction)
            if len(self.server_cache) >= self.max_cache_entries:
                # Evict least recently used
                evicted_id, _ = self.server_cache.popitem(last=False)
                # Also remove from client cache and versions
                self.client_graphs.pop(evicted_id, None)
                self.client_versions.pop(evicted_id, None)
                self.server_versions.pop(evicted_id, None)
                logger.debug(f"Evicted LRU graph from differential protocol: {evicted_id}")
            
            logger.info(f"✓ First graph for {graph_id}: sending full ({graph_size/1024:.1f}KB)")
            self.client_graphs[graph_id] = graph
            self.client_versions[graph_id] = 1
            self.server_cache[graph_id] = graph
            self.server_versions[graph_id] = 1
            # Move to end (most recently used)
            self.client_graphs.move_to_end(graph_id)
            self.server_cache.move_to_end(graph_id)
            
            self.stats['full_graphs_sent'] += 1
            self.stats['total_full_size_mb'] += graph_size / (1024 * 1024)
            
            return {
                'type': 'full_graph',
                'graph_id': graph_id,
                'version': 1,
                'graph': graph
            }
        
        # Update - compute and send delta
        cached_graph = self.server_cache[graph_id]
        delta = self._compute_delta(graph_id, cached_graph, graph)
        
        compression = delta.estimate_compression(graph_size)
        delta_size = delta.serialized_size
        saved = graph_size - delta_size
        
        logger.info(f"✓ Update for {graph_id}: delta {delta_size/1024:.1f}KB " 
                   f"({compression*100:.1f}% of full)")
        
        # Update caches
        self.client_graphs[graph_id] = graph
        self.client_versions[graph_id] = delta.version
        self.server_cache[graph_id] = graph
        self.server_versions[graph_id] = delta.version
        # Move to end (most recently used)
        self.client_graphs.move_to_end(graph_id)
        self.server_cache.move_to_end(graph_id)
        
        # Update statistics
        self.stats['delta_updates_sent'] += 1
        self.stats['total_delta_size_mb'] += delta_size / (1024 * 1024)
        self.stats['total_traffic_saved_mb'] += saved / (1024 * 1024)
        
        return {
            'type': 'delta_update',
            'graph_id': graph_id,
            'version': delta.version,
            'delta': {
                'added_operations': delta.added_operations,
                'removed_operation_ids': delta.removed_operation_ids,
                'modified_operations': delta.modified_operations,
                'input_changes': delta.input_changes,
            }
        }
    
    def _compute_delta(self, 
                       graph_id: str,
                       old_graph: Dict[str, Any],
                       new_graph: Dict[str, Any]) -> GraphDelta:
        """
        Compute the differences between two graphs.
        
        Returns only what changed - typically 10% of full graph size.
        """
        old_ops = {op['op_id']: op for op in old_graph.get('operations', [])}
        new_ops = {op['op_id']: op for op in new_graph.get('operations', [])}
        
        old_op_ids = set(old_ops.keys())
        new_op_ids = set(new_ops.keys())
        
        # Identify changes
        added_op_ids = new_op_ids - old_op_ids
        removed_op_ids = old_op_ids - new_op_ids
        common_op_ids = old_op_ids & new_op_ids
        
        added_operations = [new_ops[op_id] for op_id in added_op_ids]
        removed_operation_ids = list(removed_op_ids)
        
        # Find modified operations (same ID but different metadata)
        modified_operations = {}
        for op_id in common_op_ids:
            old_op = old_ops[op_id]
            new_op = new_ops[op_id]
            
            # Compare operation content (ignore timing info)
            if (old_op.get('operation') != new_op.get('operation') or
                old_op.get('inputs') != new_op.get('inputs')):
                modified_operations[op_id] = new_op
        
        # Track input changes
        old_inputs = old_graph.get('input_tensors', {})
        new_inputs = new_graph.get('input_tensors', {})
        input_changes = {}
        
        for input_id in set(old_inputs.keys()) | set(new_inputs.keys()):
            if old_inputs.get(input_id) != new_inputs.get(input_id):
                input_changes[input_id] = new_inputs.get(input_id)
        
        # Create delta
        delta = GraphDelta(
            graph_id=graph_id,
            version=self.server_versions.get(graph_id, 0) + 1,
            added_operations=added_operations,
            removed_operation_ids=removed_operation_ids,
            modified_operations=modified_operations,
            input_changes=input_changes
        )
        
        # Estimate serialized size
        delta.serialized_size = (
            len(added_operations) * 150 +  # ~150 bytes per added op
            len(removed_operation_ids) * 8 +  # ~8 bytes per removed ID
            len(modified_operations) * 200 +  # ~200 bytes per modified op
            len(input_changes) * 100  # ~100 bytes per input change
        )
        
        return delta
    
    @staticmethod
    def _estimate_size(graph: Dict[str, Any]) -> int:
        """
        Estimate graph size in bytes.
        
        Args:
            graph: Graph dictionary
            
        Returns:
            Estimated size in bytes
        """
        if not isinstance(graph, dict):
            logger.warning(f"Invalid graph type for size estimation: {type(graph)}")
            return 1000  # Conservative default
        
        try:
            return len(json.dumps(graph).encode('utf-8'))
        except (TypeError, ValueError) as e:
            # Fallback: estimate based on structure
            logger.debug(f"JSON serialization failed for size estimation: {e}")
            ops_size = len(graph.get('operations', [])) * 150
            inputs_size = len(graph.get('input_tensors', {})) * 100
            return ops_size + inputs_size + 1000

# NOTE: MegaNodeExpander class was removed (2024) - it was unused dead code (~100 lines).
# If mega-node expansion is needed in the future, it should be implemented in the server
# execution layer where it would actually be used, not in the differential updates module.
