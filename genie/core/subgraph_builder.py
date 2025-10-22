"""
SubgraphBuilder: Extracts executable subgraphs from LazyTensor DAG.

This module implements the core optimization described in the network enhancement plan:
instead of executing operations one at a time (causing O(n) network round-trips),
we extract entire computation subgraphs and execute them remotely in a single request.

Key benefits:
- O(1) network round-trips instead of O(n)
- Intermediates stay on GPU (no CPU round-trips)
- Reduced serialization overhead
- Better batching opportunities

Usage:
    builder = SubgraphBuilder()
    subgraph = builder.build_remote_subgraph(target_tensor)
    # Send subgraph to remote server for execution
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Set, Optional
import torch

from .lazy_tensor import LazyTensor

logger = logging.getLogger(__name__)


@dataclass
class RemoteSubgraph:
    """
    Subgraph to execute remotely.

    Contains:
    - Topologically sorted operations
    - External inputs (tensors from outside this subgraph)
    - Output tensor (final result to return)
    """
    operations: List[LazyTensor]  # Sorted in execution order
    input_tensors: Dict[int, LazyTensor]  # tensor_id → LazyTensor (external inputs)
    output_tensor: LazyTensor  # Final output

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize to JSON-compatible dict for network transfer.

        Returns:
            {
                'operations': [
                    {
                        'op_id': int,
                        'operation': str (e.g., 'aten::matmul'),
                        'inputs': [int, ...],  # IDs of input tensors
                        'kwargs': {...}
                    },
                    ...
                ],
                'input_tensors': {
                    'tensor_id': {
                        'shape': [int, ...],
                        'dtype': str
                    },
                    ...
                },
                'output_id': int
            }
        """
        return {
            'operations': [
                {
                    'op_id': id(op),
                    'operation': op.operation,
                    'inputs': [
                        id(inp) for inp in op.inputs
                        if isinstance(inp, LazyTensor)
                    ],
                    'kwargs': op.kwargs,
                    'shape': list(op.shape) if op.shape else None,
                    'dtype': str(op.dtype) if op.dtype else None
                }
                for op in self.operations
            ],
            'input_tensors': {
                str(tensor_id): {
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype)
                }
                for tensor_id, tensor in self.input_tensors.items()
            },
            'output_id': id(self.output_tensor)
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any],
                   input_data: Dict[str, torch.Tensor]) -> 'RemoteSubgraph':
        """
        Reconstruct subgraph from serialized data.

        This is used on the server side to rebuild the computation graph.
        """
        # This would be implemented on the server side
        # For now, just a placeholder
        raise NotImplementedError("Server-side deserialization not yet implemented")


class SubgraphBuilder:
    """
    Extracts executable subgraphs from LazyTensor DAG.

    Strategy (Phase 1 - Simple & Correct):
    - Extract ALL ancestors of target tensor
    - Execute entire chain remotely
    - Only transfer final result back

    Strategy (Phase 2 - Smart):
    - Cost-based fragmentation
    - Mix local/remote execution
    - Optimize transfer vs compute tradeoff
    """

    def build_remote_subgraph(self, target_tensor: LazyTensor) -> RemoteSubgraph:
        """
        Extract subgraph for remote execution.

        Algorithm:
        1. Start from target tensor
        2. Traverse backwards (BFS/DFS) to find all ancestors
        3. Identify external inputs (factory ops or leaf tensors)
        4. Sort operations in topological order
        5. Return RemoteSubgraph
        """
        operations = []
        visited = set()
        input_tensors = {}

        def collect_ancestors(tensor):
            if id(tensor) in visited:
                return
            visited.add(id(tensor))

            # Check if this is an external input (factory operation or leaf)
            if self._is_external_input(tensor):
                input_tensors[id(tensor)] = tensor
                return

            # Collect inputs first (post-order traversal for topological order)
            for inp in tensor.inputs:
                if isinstance(inp, LazyTensor):
                    collect_ancestors(inp)

            # Add this operation
            operations.append(tensor)

        collect_ancestors(target_tensor)

        logger.info(f"Built subgraph: {len(operations)} operations, "
                   f"{len(input_tensors)} external inputs")

        return RemoteSubgraph(
            operations=operations,  # Already in topological order
            input_tensors=input_tensors,
            output_tensor=target_tensor
        )

    def _is_external_input(self, tensor: LazyTensor) -> bool:
        """Check if tensor is an external input (not computed in this subgraph)."""
        # Factory operations create tensors from nothing
        factory_ops = {
            'aten::randn', 'aten::zeros', 'aten::ones', 'aten::empty',
            'aten::tensor', 'aten::as_tensor', 'aten::from_numpy',
            'aten::arange', 'aten::linspace', 'aten::logspace',
            'aten::eye', 'aten::full'
        }
        return tensor.operation in factory_ops

    def _should_include_in_subgraph(self, tensor: LazyTensor) -> bool:
        """Check if tensor should be included in remote subgraph."""
        # For Phase 1: include all tensors targeting remote devices

        # Check current device property (this should work for remote devices)
        if hasattr(tensor, 'device') and tensor.device:
            device_str = str(tensor.device)
            if ('remote_accelerator' in device_str or
                'privateuseone' in device_str):
                return True

        # Check original device (for remote devices mapped to meta)
        if hasattr(tensor, '_original_device') and tensor._original_device:
            device_str = str(tensor._original_device)
            if ('remote_accelerator' in device_str or
                'privateuseone' in device_str):
                return True

        # Check device string from kwargs
        if hasattr(tensor, '_kwargs') and tensor._kwargs:
            device = tensor._kwargs.get('device')
            if device:
                device_str = str(device)
                return ('remote_accelerator' in device_str or
                       'privateuseone' in device_str)

        return False

    def build_from_device_chain(self, target_tensor: LazyTensor) -> Optional[RemoteSubgraph]:
        """
        Build subgraph only for tensors that target remote devices.

        This is the main entry point for the optimization - only extract
        subgraphs for operations that actually need remote execution.
        """
        # Check if this tensor chain targets remote devices by checking if any
        # ancestor is a remote device
        has_remote_ancestor = self._has_remote_ancestor(target_tensor)

        if not has_remote_ancestor:
            return None

        # Build the subgraph
        return self.build_remote_subgraph(target_tensor)

    def _has_remote_ancestor(self, tensor: LazyTensor) -> bool:
        """Check if tensor has any remote device ancestor."""
        visited = set()

        def check_ancestors(t):
            if id(t) in visited:
                return False
            visited.add(id(t))

            # Check if this tensor is remote
            if self._should_include_in_subgraph(t):
                return True

            # Check inputs
            for inp in t.inputs:
                if isinstance(inp, LazyTensor):
                    if check_ancestors(inp):
                        return True

            return False

        return check_ancestors(tensor)
