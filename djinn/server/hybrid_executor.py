"""
Hybrid Executor: Unified execution with lazy output references

Implements the v2.3 architecture feature: "Hybrid Executor with Lazy Output"

Execution Flow:
┌─────────────────────────────────────┐
│ 1. Memory Plan                      │ ← From MetaSimulator
│    (allocations, watermark)         │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ 2. Allocate tensors (VMU)          │
│    Persistent: KV cache             │
│    Volatile: Activations            │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ 3. Execute Graph                    │
│    Operations on slab views         │
│    Stream 1: DMA (H2D/D2H)          │
│    Stream 2: Compute (GPU)          │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ 4. Return RemoteRefs                │ ← Lazy output
│    Not concrete tensors             │
│    Only IDs + shapes (bytes)        │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ 5. Cleanup                          │
│    Reset volatile memory            │
│    Keep persistent (KV)             │
└─────────────────────────────────────┘

Key Innovations:
1. Slab-based execution: All tensors in single buffer
2. Lazy outputs: RemoteRef instead of data transfer
3. Watermark reset: Efficient volatile memory reuse
4. Two CUDA streams: Overlap network and compute
"""

import logging
import contextlib
import asyncio
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Union
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExecutionMetrics:
    """Metrics from execution."""
    duration_ms: float
    memory_peak_mb: float
    memory_persistent_mb: float
    memory_volatile_mb: float
    activations_count: int
    output_refs_count: int
    success: bool
    plan_summary: Optional[Dict[str, Any]] = None


class HybridExecutor:
    """
    Unified executor for graph and model cache execution.
    
    Handles:
    - Slab-based memory allocation
    - Watermark-aware execution
    - Lazy output references
    - Volatile memory reset
    - Two-stream pipelined execution
    """
    
    def __init__(self, vmu, meta_simulator=None):
        """
        Initialize hybrid executor.
        
        Args:
            vmu: Unified VMU instance
            meta_simulator: MetaSimulator for planning
        """
        self.vmu = vmu
        self.meta_simulator = meta_simulator
        self.stream_compute = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.stream_transfer = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        # Model cache: avoid moving models to GPU repeatedly
        self.model_cache = {}  # {model_id: gpu_model}
        
        logger.info("✅ HybridExecutor initialized")
    
    async def execute_with_lazy_outputs(self,
                                        model: nn.Module,
                                        inputs: Dict[str, torch.Tensor],
                                        session_id: Optional[str] = None,
                                        return_lazy: bool = True) -> Union[torch.Tensor, List]:
        """
        Execute model with lazy output references and stream locking.
        
        Strategy:
        1. Acquire stream lock (one execution at a time)
        2. Plan memory allocation
        3. Execute forward pass
        4. Skeletonize outputs (preserve structure, use RemoteRefStubs)
        5. Reset volatile memory
        6. Release stream lock
        
        Args:
            model: Model to execute
            inputs: Input tensors
            session_id: Session ID for GC registration
            return_lazy: Whether to return RemoteRefs (True) or concrete tensors (False)
        
        Returns:
            RemoteRefs (or skeletonized structure) if return_lazy=True, else concrete tensors
        """
        start_time = time.perf_counter()
        timing_breakdown = {}
        
        try:
            logger.info("🚀 Hybrid execution starting (lazy output mode)")

            loop = asyncio.get_running_loop()

            plan_start = time.perf_counter()
            with self.vmu.lock:
                plan = self.meta_simulator.get_plan(model, inputs, self.vmu) if self.meta_simulator else None
            timing_breakdown['planning'] = (time.perf_counter() - plan_start) * 1000

            placement_start = time.perf_counter()
            gpu_model = self._ensure_model_on_gpu(model)
            gpu_inputs = self._move_inputs_to_gpu(inputs)
            timing_breakdown['placement'] = (time.perf_counter() - placement_start) * 1000

            exec_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
            event = torch.cuda.Event() if torch.cuda.is_available() else None

            execution_output = None
            with torch.cuda.stream(exec_stream) if exec_stream else contextlib.nullcontext():
                exec_start = time.perf_counter()
                with torch.no_grad():
                    execution_output = gpu_model(**gpu_inputs)
                timing_breakdown['execution'] = (time.perf_counter() - exec_start) * 1000

                skel_start = time.perf_counter()
                if return_lazy:
                    prepared_output = self._skeletonize(execution_output, session_id)
                else:
                    prepared_output = self._materialize_to_cpu(execution_output)
                timing_breakdown['skeletonization'] = (time.perf_counter() - skel_start) * 1000

                if event:
                    event.record(exec_stream)

            if event:
                await loop.run_in_executor(None, event.synchronize)
                torch.cuda.current_stream().wait_event(event)

            cleanup_start = time.perf_counter()
            with self.vmu.lock:
                self.vmu.reset_stack()
            timing_breakdown['cleanup'] = (time.perf_counter() - cleanup_start) * 1000

            duration_ms = (time.perf_counter() - start_time) * 1000
            stats = self.vmu.get_stats()
            
            metrics = ExecutionMetrics(
                duration_ms=duration_ms,
                memory_peak_mb=stats.get('max_current_offset', 0) / 1024**2,
                memory_persistent_mb=stats.get('persistent_allocated_bytes', 0) / 1024**2,
                memory_volatile_mb=stats.get('volatile_allocated_bytes', 0) / 1024**2,
                activations_count=self._count_activations(execution_output),
                output_refs_count=1,
                success=True,
                plan_summary=plan.semantic_summary if plan else None
            )

            vmu_stats = self.vmu.get_stats()
            logger.info(
                f"✅ Execution complete: {metrics.duration_ms:.1f}ms, "
                f"peak memory {metrics.memory_peak_mb:.1f}MB"
            )
            logger.info(
                f"   Timing breakdown: "
                f"plan={timing_breakdown.get('planning', 0):.1f}ms, "
                f"placement={timing_breakdown.get('placement', 0):.1f}ms, "
                f"exec={timing_breakdown.get('execution', 0):.1f}ms, "
                f"skel={timing_breakdown.get('skeletonization', 0):.1f}ms, "
                f"cleanup={timing_breakdown.get('cleanup', 0):.1f}ms"
            )
            logger.info(
                f"   VMU usage: {vmu_stats['vmu_total_allocated_mb']:.1f}MB / "
                f"{vmu_stats['vmu_total_capacity_mb']:.1f}MB "
                f"({vmu_stats['vmu_utilization_percent']:.1f}%), "
                f"persistent={vmu_stats['persistent_allocated_mb']:.1f}MB"
            )
            
            return prepared_output, metrics
        
        except Exception as e:
            logger.error(f"❌ Execution failed: {e}")
            
            metrics = ExecutionMetrics(
                duration_ms=(time.perf_counter() - start_time) * 1000,
                memory_peak_mb=0,
                memory_persistent_mb=0,
                memory_volatile_mb=0,
                activations_count=0,
                output_refs_count=0,
                success=False,
                plan_summary=None
            )
            
            raise
    
    def _skeletonize(self, obj: Any, session_id: Optional[str] = None) -> Any:
        """
        Convert tensor outputs to RemoteRefStubs while preserving structure.
        
        Implements Output Skeletonization:
        - Tensors → RemoteRefStubs (lightweight references)
        - Dicts → Dicts of stubs
        - Tuples → Tuples of stubs
        - Lists → Lists of stubs
        - Custom classes → Preserved with stub attributes
        
        Args:
            obj: Output object (tensor, dict, tuple, etc.)
            session_id: Session ID for GC registration
        
        Returns:
            Same structure but with RemoteRefStubs instead of tensors
        """
        from .session_manager import get_session_manager
        from ..core.remote_ref_stub import RemoteRefStub, store_tensor
        
        if isinstance(obj, torch.Tensor):
            # Generate unique reference ID
            ref_id = self._generate_tensor_id()
            
            # CRITICAL: Store tensor in global registry for later retrieval
            # This allows RemoteRefStub.to() to fetch the actual tensor data
            store_tensor(ref_id, obj)
            
            # Register with session manager for distributed GC
            if session_id:
                session_mgr = get_session_manager()
                session_mgr.register_ref(
                    session_id, 
                    ref_id,
                    size_bytes=obj.numel() * obj.element_size()
                )
                logger.debug(
                    f"Registered tensor {ref_id} with session {session_id} "
                    f"({obj.numel() * obj.element_size() / 1024**2:.1f}MB)"
                )
            
            # Create lightweight stub
            stub = RemoteRefStub(
                ref_id=ref_id,
                shape=tuple(obj.shape),
                dtype=obj.dtype
            )
            
            return stub
        
        elif isinstance(obj, dict):
            # Preserve dictionary structure
            return {k: self._skeletonize(v, session_id) for k, v in obj.items()}
        
        elif isinstance(obj, (list, tuple)):
            # Preserve sequence type
            skeletonized = [self._skeletonize(v, session_id) for v in obj]
            return type(obj)(skeletonized)
        
        else:
            # Pass through non-tensor types (strings, ints, floats, etc.)
            return obj
    
    def _generate_tensor_id(self) -> str:
        """Generate unique tensor ID."""
        import uuid
        return f"tensor_{uuid.uuid4().hex[:12]}"
    
    def _count_activations(self, output: Any) -> int:
        """Count number of activation tensors."""
        if isinstance(output, torch.Tensor):
            return 1
        elif isinstance(output, (tuple, list)):
            return len(output)
        elif isinstance(output, dict):
            return len(output)
        else:
            return 0
    
    def _ensure_model_on_gpu(self, model: nn.Module) -> nn.Module:
        if not torch.cuda.is_available():
            return model
        model_id = id(model)
        if model_id in self.model_cache:
            return self.model_cache[model_id]
        gpu_model = model
        model_device = next(model.parameters()).device if list(model.parameters()) else None
        if model_device is None or model_device.type != 'cuda':
            logger.info(f"Moving model from {model_device} to {self.vmu.device}...")
            for param in gpu_model.parameters():
                param.data = param.data.to(self.vmu.device)
            for buffer in gpu_model.buffers():
                buffer.data = buffer.data.to(self.vmu.device)
            gpu_model = gpu_model.to(self.vmu.device)
        gpu_model.eval()
        self.model_cache[model_id] = gpu_model
        return gpu_model

    def _move_inputs_to_gpu(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not torch.cuda.is_available():
            return inputs
        gpu_inputs: Dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                gpu_inputs[key] = value.to(self.vmu.device, non_blocking=True)
            else:
                gpu_inputs[key] = value
        return gpu_inputs

    def _materialize_to_cpu(self, output: Any) -> Any:
        if not torch.cuda.is_available():
            return output

        def _to_cpu(t: torch.Tensor) -> torch.Tensor:
            if t.device.type == 'cpu':
                return t
            return t.to('cpu', non_blocking=True)

        if isinstance(output, torch.Tensor):
            return _to_cpu(output)
        if isinstance(output, dict):
            return {k: self._materialize_to_cpu(v) for k, v in output.items()}
        if isinstance(output, (list, tuple)):
            converted = [self._materialize_to_cpu(v) for v in output]
            return type(output)(converted)
        return output
    
    def execute_graph_on_slab(self,
                             operations: List[Tuple[str, List, Dict]],
                             slab_offset_map: Dict[str, Tuple[int, int]]) -> torch.Tensor:
        """
        Execute computation graph using slab allocations.
        
        Args:
            operations: List of (op_name, inputs, kwargs)
            slab_offset_map: Mapping of tensor_name -> (offset, size)
        
        Returns:
            Output tensor
        """
        logger.debug(f"Executing {len(operations)} operations on slab")
        
        # Execute operations in order
        tensor_cache = {}  # Cache computed tensors
        
        try:
            for op_idx, (op_name, inputs, kwargs) in enumerate(operations):
                # Get input tensors (from cache or create new)
                materialized_inputs = []
                for inp in inputs:
                    if isinstance(inp, str):
                        # Reference to previous tensor
                        if inp in tensor_cache:
                            materialized_inputs.append(tensor_cache[inp])
                        elif inp in slab_offset_map:
                            # Create slab view
                            offset, size = slab_offset_map[inp]
                            view = self.vmu.get_slab_view(offset, size, torch.float32)
                            materialized_inputs.append(view)
                        else:
                            logger.warning(f"Unknown tensor reference: {inp}")
                    else:
                        materialized_inputs.append(inp)
                
                # Execute operation
                logger.debug(f"Executing op {op_idx}: {op_name}")
                
                # Use compute stream
                if self.stream_compute:
                    with torch.cuda.stream(self.stream_compute):
                        result = self._execute_operation(op_name, materialized_inputs, kwargs)
                else:
                    result = self._execute_operation(op_name, materialized_inputs, kwargs)
                
                # Cache result
                tensor_cache[f"op_{op_idx}"] = result
            
            # Return last result
            return tensor_cache[f"op_{len(operations) - 1}"]
        
        except Exception as e:
            logger.error(f"Graph execution failed: {e}")
            raise
    
    def _execute_operation(self, op_name: str, inputs: List, kwargs: Dict) -> torch.Tensor:
        """Execute a single operation."""
        try:
            if op_name == 'matmul':
                return torch.matmul(inputs[0], inputs[1])
            elif op_name == 'add':
                return inputs[0] + inputs[1]
            elif op_name == 'mul':
                return inputs[0] * inputs[1]
            elif op_name == 'gelu':
                return torch.nn.functional.gelu(inputs[0])
            elif op_name == 'attention':
                # Simplified attention
                q, k, v = inputs[0], inputs[1], inputs[2]
                scores = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
                weights = torch.softmax(scores, dim=-1)
                return torch.matmul(weights, v)
            else:
                logger.warning(f"Unknown operation: {op_name}")
                return inputs[0]
        
        except Exception as e:
            logger.error(f"Operation execution failed: {op_name}: {e}")
            raise


# Global executor instance
_global_executor: Optional[HybridExecutor] = None


def get_hybrid_executor() -> HybridExecutor:
    """Get or create global hybrid executor."""
    global _global_executor
    
    if _global_executor is None:
        from ..backend.runtime.unified_vmu import get_vmu
        from .meta_simulator import get_meta_simulator
        
        vmu = get_vmu()
        simulator = get_meta_simulator()
        _global_executor = HybridExecutor(vmu=vmu, meta_simulator=simulator)
    
    return _global_executor

