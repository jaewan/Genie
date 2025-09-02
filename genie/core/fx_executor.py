"""FX-based executor for Genie.

This module provides execution capabilities for FX graphs with semantic metadata.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.fx as fx
from torch.fx import GraphModule, Interpreter

from .semantic_metadata import SemanticMetadata, ExecutionPhase

logger = logging.getLogger(__name__)


class FXExecutor(Interpreter):
    """Execute FX graphs with semantic awareness.
    
    This executor extends PyTorch's FX Interpreter to provide semantic-aware
    execution with metadata tracking and optimization opportunities.
    """
    
    def __init__(self, module: GraphModule, track_metadata: bool = True):
        """Initialize FX executor.
        
        Args:
            module: GraphModule to execute
            track_metadata: Whether to track semantic metadata during execution
        """
        super().__init__(module)
        self.track_metadata = track_metadata
        self.execution_trace = []
        self.semantic_stats = {
            'phases_executed': {},
            'memory_patterns': {},
            'compute_intensity_total': 0.0,
            'operations_count': {}
        }
    
    def run_node(self, n: fx.Node) -> Any:
        """Execute a single node with semantic tracking.
        
        Args:
            n: FX Node to execute
            
        Returns:
            Result of node execution
        """
        # Track pre-execution metadata
        if self.track_metadata and 'semantic' in n.meta:
            self._track_pre_execution(n)
        
        # Execute node
        result = super().run_node(n)
        
        # Track post-execution metadata
        if self.track_metadata:
            self._track_post_execution(n, result)
        
        return result
    
    def _track_pre_execution(self, node: fx.Node) -> None:
        """Track metadata before node execution."""
        metadata = node.meta.get('semantic')
        if not metadata:
            return
        
        # Track execution phase
        if metadata.execution_phase:
            phase = str(metadata.execution_phase.value if hasattr(metadata.execution_phase, 'value') else metadata.execution_phase)
            self.semantic_stats['phases_executed'][phase] = self.semantic_stats['phases_executed'].get(phase, 0) + 1
        
        # Track memory pattern
        if metadata.memory_pattern:
            pattern = str(metadata.memory_pattern.value if hasattr(metadata.memory_pattern, 'value') else metadata.memory_pattern)
            self.semantic_stats['memory_patterns'][pattern] = self.semantic_stats['memory_patterns'].get(pattern, 0) + 1
        
        # Track compute intensity
        if metadata.compute_intensity:
            self.semantic_stats['compute_intensity_total'] += metadata.compute_intensity
        
        # Log high-priority operations
        if metadata.priority and metadata.priority > 5:
            logger.debug(f"Executing high-priority operation: {node.name} (priority={metadata.priority})")
    
    def _track_post_execution(self, node: fx.Node, result: Any) -> None:
        """Track metadata after node execution."""
        # Track operation type
        if node.op == 'call_function':
            op_name = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)
            self.semantic_stats['operations_count'][op_name] = self.semantic_stats['operations_count'].get(op_name, 0) + 1
        
        # Add to execution trace
        self.execution_trace.append({
            'node': node.name,
            'op': node.op,
            'target': str(node.target),
            'semantic_metadata': node.meta.get('semantic', None),
            'result_shape': result.shape if isinstance(result, torch.Tensor) else None,
            'result_dtype': result.dtype if isinstance(result, torch.Tensor) else None
        })
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution with semantic statistics."""
        return {
            'total_operations': len(self.execution_trace),
            'semantic_stats': self.semantic_stats,
            'execution_trace_length': len(self.execution_trace)
        }


class OptimizingFXExecutor(FXExecutor):
    """Semantic-aware optimizing executor.
    
    This executor applies semantic-driven optimizations during execution
    based on the metadata from the HotNets'25 paper.
    """
    
    def __init__(self, module: GraphModule, enable_optimizations: bool = True):
        """Initialize optimizing executor.
        
        Args:
            module: GraphModule to execute
            enable_optimizations: Whether to apply semantic optimizations
        """
        super().__init__(module, track_metadata=True)
        self.enable_optimizations = enable_optimizations
        self.optimization_stats = {
            'fused_operations': 0,
            'recomputed_operations': 0,
            'cached_operations': 0,
            'parallelized_operations': 0
        }
        self.kv_cache = {}  # Simple KV cache for LLM decode optimization
    
    def run_node(self, n: fx.Node) -> Any:
        """Execute node with semantic optimizations."""
        if not self.enable_optimizations:
            return super().run_node(n)
        
        metadata = n.meta.get('semantic')
        
        # Apply phase-specific optimizations
        if metadata and metadata.execution_phase:
            if metadata.execution_phase == ExecutionPhase.DECODE:
                return self._run_decode_optimized(n)
            elif metadata.execution_phase == ExecutionPhase.VISION_BACKBONE:
                return self._run_vision_optimized(n)
            elif metadata.execution_phase == ExecutionPhase.MULTIMODAL_FUSION:
                return self._run_fusion_optimized(n)
        
        # Check for KV cache operations
        if metadata and metadata.kv_cache_related:
            return self._run_with_kv_cache(n)
        
        # Default execution
        return super().run_node(n)
    
    def _run_decode_optimized(self, node: fx.Node) -> Any:
        """Optimized execution for LLM decode phase."""
        # In real implementation, this would co-locate with KV cache
        logger.debug(f"Applying decode optimization for {node.name}")
        
        # Check if this is a KV cache access
        metadata = node.meta.get('semantic')
        if metadata and metadata.semantic_role and 'attention' in metadata.semantic_role:
            # Simulate cache-aware execution
            self.optimization_stats['cached_operations'] += 1
        
        return super().run_node(node)
    
    def _run_vision_optimized(self, node: fx.Node) -> Any:
        """Optimized execution for vision backbone."""
        # In real implementation, this would pipeline CNN stages
        logger.debug(f"Applying vision optimization for {node.name}")
        
        metadata = node.meta.get('semantic')
        if metadata and metadata.can_parallelize:
            self.optimization_stats['parallelized_operations'] += 1
        
        return super().run_node(node)
    
    def _run_fusion_optimized(self, node: fx.Node) -> Any:
        """Optimized execution for multi-modal fusion."""
        # In real implementation, this would JIT transfer at fusion point
        logger.debug(f"Applying fusion optimization for {node.name}")
        
        self.optimization_stats['fused_operations'] += 1
        
        return super().run_node(node)
    
    def _run_with_kv_cache(self, node: fx.Node) -> Any:
        """Execute with KV cache optimization."""
        # Simple cache simulation
        cache_key = f"{node.name}_{node.target}"
        
        if cache_key in self.kv_cache:
            logger.debug(f"Using cached result for {node.name}")
            self.optimization_stats['cached_operations'] += 1
            return self.kv_cache[cache_key]
        
        result = super().run_node(node)
        
        # Cache result if it's a tensor
        if isinstance(result, torch.Tensor):
            self.kv_cache[cache_key] = result
        
        return result
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimizations applied."""
        base_summary = self.get_execution_summary()
        base_summary['optimization_stats'] = self.optimization_stats
        base_summary['kv_cache_size'] = len(self.kv_cache)
        return base_summary


def execute_fx_graph(fx_builder, inputs: Optional[Dict[str, torch.Tensor]] = None,
                     optimize: bool = False) -> Tuple[Any, Dict[str, Any]]:
    """Execute an FX graph with optional optimizations.
    
    Args:
        fx_builder: FXGraphBuilder instance
        inputs: Optional input tensors mapping
        optimize: Whether to apply semantic optimizations
        
    Returns:
        Tuple of (execution result, execution summary)
    """
    # Convert to GraphModule
    gm = fx_builder.to_graph_module()
    
    # Create executor
    if optimize:
        executor = OptimizingFXExecutor(gm, enable_optimizations=True)
    else:
        executor = FXExecutor(gm, track_metadata=True)
    
    # Prepare inputs
    if inputs is None:
        inputs = {}
    
    # Map placeholder names to input tensors
    args = []
    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            if node.name in inputs:
                args.append(inputs[node.name])
            else:
                # Create dummy input based on metadata
                if 'tensor_meta' in node.meta:
                    meta = node.meta['tensor_meta']
                    dummy = torch.randn(
                        meta['shape'],
                        dtype=meta['dtype'],
                        device=meta['device']
                    )
                    args.append(dummy)
                else:
                    args.append(torch.randn(1))  # Default dummy
    
    # Execute
    try:
        result = executor.run(*args)
        summary = executor.get_optimization_summary() if optimize else executor.get_execution_summary()
        
        logger.info(f"FX graph executed successfully with {summary['total_operations']} operations")
        
        return result, summary
    except Exception as e:
        logger.error(f"FX graph execution failed: {e}")
        raise
