"""
Hybrid graph builder: Tries FX first, falls back to LazyTensor DAG.

This provides the "single source of truth" while maintaining compatibility
with models that have dynamic control flow.
"""

import torch
import torch.fx as fx
import threading
from typing import Optional, Any
import logging

from .graph_interface import Graph, FXGraphAdapter, LazyDAGAdapter
from .lazy_tensor import LazyTensor

logger = logging.getLogger(__name__)


class HybridGraphBuilder:
    """
    Thread-local graph builder.

    Each thread gets its own graph builder instance to avoid
    race conditions during concurrent capture.

    Strategy:
    1. Try torch.fx.symbolic_trace (covers ~80% of models)
    2. If that fails, use LazyTensor DAG (always works)

    Both representations are exposed through unified Graph interface.
    """

    # Thread-local storage for per-thread builders
    _thread_local = threading.local()

    def __init__(self):
        self.fx_graph: Optional[fx.Graph] = None
        self.fx_module: Optional[fx.GraphModule] = None
        self.use_fx = True

        # LazyTensor tracking (FIX: Remove unused all_tensors dict)
        self.root_tensor: Optional[LazyTensor] = None

    def build_from_model(self, model: torch.nn.Module, *args) -> Graph:
        """
        Build graph from model using hybrid strategy.

        Args:
            model: PyTorch model to trace
            *args: Example inputs

        Returns:
            Unified Graph interface
        """
        # Try FX first
        try:
            logger.info("Attempting FX symbolic trace...")
            self.fx_module = fx.symbolic_trace(model)
            self.fx_graph = self.fx_module.graph
            self.use_fx = True
            logger.info(f"✓ FX trace successful ({len(list(self.fx_graph.nodes))} nodes)")
            return FXGraphAdapter(self.fx_graph)

        except Exception as e:
            # FX failed - fall back to LazyTensor DAG
            logger.info(f"FX trace failed: {e}")
            logger.info("Falling back to LazyTensor DAG capture...")

            self.use_fx = False

            # Capture using LazyTensor
            output = model(*args)

            if not isinstance(output, LazyTensor):
                raise RuntimeError(
                    "Model output is not a LazyTensor. "
                    "Make sure tensors are on remote_accelerator device."
                )

            self.root_tensor = output
            logger.info(f"✓ LazyTensor DAG built successfully")
            return LazyDAGAdapter(self.root_tensor)

    def build_from_capture(self) -> Graph:
        """
        Build graph from captured LazyTensors.

        Used with context manager:
            with genie.capture():
                output = model(input)
            graph = builder.build_from_capture()

        Strategy: Try FX tracing first, fallback to LazyDAG if that fails.
        """
        if self.root_tensor is None:
            raise RuntimeError("No LazyTensor captured")

        # Try FX tracing on the captured computation
        # We need to reconstruct the model from the LazyTensor DAG
        try:
            logger.info("Attempting FX symbolic trace on captured computation...")
            # For now, if we have a LazyDAG, we'll use it
            # TODO: Implement FX reconstruction from LazyDAG for better semantic analysis
            logger.info("Using LazyDAG (FX reconstruction not yet implemented)")
            return LazyDAGAdapter(self.root_tensor)

        except Exception as e:
            logger.info(f"FX reconstruction failed: {e}")
            logger.info("Using LazyTensor DAG fallback")
            return LazyDAGAdapter(self.root_tensor)

    def add_operation(self, tensor: LazyTensor):
        """
        Register LazyTensor operation (called from LazyTensor.__init__).

        This tracks all tensors as they're created, enabling DAG construction.

        Thread-safe: Each thread has its own builder instance.
        """
        # FIX: Remove unused all_tensors tracking (memory leak)
        self.root_tensor = tensor  # Track most recent (used as output)

    def get_graph(self) -> Optional[Graph]:
        """Get the current graph (FX or LazyDAG)."""
        if self.use_fx and self.fx_graph is not None:
            return FXGraphAdapter(self.fx_graph)
        elif self.root_tensor is not None:
            return LazyDAGAdapter(self.root_tensor)
        else:
            return None

    def materialize(self, target_tensor) -> torch.Tensor:
        """
        Materialize a LazyTensor by executing its computation graph.

        Design: Graph builder is responsible for graph construction and semantic
        analysis. Executor is responsible for execution. Clean separation of concerns.

        Args:
            target_tensor: LazyTensor to materialize

        Returns:
            Concrete torch.Tensor with computed values

        Note:
            This delegates to the executor module, which handles:
            - Recursive graph traversal
            - Operation execution
            - Fallback for unsupported ops
            - Error handling and reporting
        """
        from .executor import execute_subgraph
        return execute_subgraph(target_tensor)


def get_global_builder() -> HybridGraphBuilder:
    """Get thread-local graph builder."""
    if not hasattr(HybridGraphBuilder._thread_local, 'builder'):
        HybridGraphBuilder._thread_local.builder = HybridGraphBuilder()
    return HybridGraphBuilder._thread_local.builder


def initialize_global_builder():
    """Initialize thread-local graph builder (called on import)."""
    # Create builder for main thread
    builder = get_global_builder()

    # Note: Don't set LazyTensor._graph_builder anymore since we use thread-local
    # LazyTensor instances will get the thread-local builder when needed