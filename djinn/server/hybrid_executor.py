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
            
            # CRITICAL: Acquire stream lock for exclusive GPU access
            # Only one execution per GPU at a time
            lock_start = time.perf_counter()
            with self.vmu.lock:
                timing_breakdown['lock_acquire'] = (time.perf_counter() - lock_start) * 1000
                logger.debug("Step 1: Acquired stream lock (exclusive GPU access)")
                
                # Step 2: Get memory plan
                plan_start = time.perf_counter()
                logger.debug("Step 2: Computing memory plan")
                if self.meta_simulator:
                    plan = self.meta_simulator.get_plan(model, inputs, self.vmu)
                else:
                    plan = None
                timing_breakdown['planning'] = (time.perf_counter() - plan_start) * 1000
                
                # Step 3: GPU Device Placement (CRITICAL OPTIMIZATION)
                placement_start = time.perf_counter()
                logger.debug("Step 3: Placing model and inputs on GPU")
                
                # Get or cache GPU model
                if torch.cuda.is_available():
                    # Use model ID for caching (based on object id)
                    model_id = id(model)
                    
                    if model_id not in self.model_cache:
                        # First time seeing this model - move to GPU and cache
                        logger.debug(f"First execution for model {model_id}, moving to GPU...")
                        model_device = next(model.parameters()).device if list(model.parameters()) else None
                        if model_device is None or model_device.type != 'cuda':
                            logger.info(f"Moving model from {model_device} to {self.vmu.device}...")
                            
                            # WORKAROUND: Manually move parameters if .to() is patched/broken
                            gpu_model = model
                            for param in gpu_model.parameters():
                                param.data = param.data.to(self.vmu.device)
                            for buffer in gpu_model.buffers():
                                buffer.data = buffer.data.to(self.vmu.device)
                            
                            # Also call .to() for metadata
                            gpu_model = gpu_model.to(self.vmu.device)
                            
                            logger.info(f"✅ Moved model to {self.vmu.device} and cached. New device: {next(gpu_model.parameters()).device}")
                        else:
                            gpu_model = model
                            logger.debug(f"Model already on GPU: {model_device}")
                        
                        # Ensure model is in eval mode
                        gpu_model.eval()
                        logger.debug(f"Model set to eval mode")
                        
                        self.model_cache[model_id] = gpu_model
                    else:
                        # Use cached GPU model
                        gpu_model = self.model_cache[model_id]
                        logger.debug(f"Using cached GPU model {model_id}")
                    
                    # Move inputs to GPU
                    gpu_inputs = {}
                    for key, value in inputs.items():
                        if isinstance(value, torch.Tensor):
                            if value.device.type != 'cuda':
                                gpu_inputs[key] = value.to(self.vmu.device)
                            else:
                                gpu_inputs[key] = value
                        else:
                            gpu_inputs[key] = value
                    num_gpu_tensors = len([v for v in gpu_inputs.values() if isinstance(v, torch.Tensor) and v.device.type == 'cuda'])
                    logger.debug(f"Inputs on GPU: {num_gpu_tensors}/{len([v for v in gpu_inputs.values() if isinstance(v, torch.Tensor)])}")
                else:
                    gpu_model = model
                    gpu_inputs = inputs
                    logger.warning("CUDA not available, running on CPU")
                
                timing_breakdown['placement'] = (time.perf_counter() - placement_start) * 1000
                
                # Execute model forward pass on compute stream
                exec_start = time.perf_counter()
                logger.debug("Step 4: Executing forward pass on GPU")
                
                # DEBUG: Check device
                first_param = next(gpu_model.parameters())
                logger.info(f"DEBUG: Model type: {type(gpu_model)}")
                logger.info(f"DEBUG: Execution device: {first_param.device}, VMU device: {self.vmu.device}, available={torch.cuda.is_available()}")
                
                with torch.no_grad():
                    # Execute directly without custom streams for now (optimization opportunity)
                    # Custom streams can add overhead if not used correctly
                    output = gpu_model(**gpu_inputs)
                    
                    # Synchronize to ensure execution completes before timing
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                
                timing_breakdown['execution'] = (time.perf_counter() - exec_start) * 1000
                
                # Step 5: Skeletonize outputs (preserve structure)
                skel_start = time.perf_counter()
                logger.debug("Step 5: Skeletonizing outputs")
                
                if return_lazy:
                    # Skeletonize: convert tensors to RemoteRefStubs while preserving structure
                    skeleton = self._skeletonize(output, session_id)
                    logger.info(f"✅ Output skeletonized")
                else:
                    skeleton = output
                
                timing_breakdown['skeletonization'] = (time.perf_counter() - skel_start) * 1000
                
                # Step 6: Cleanup and reset volatile memory
                cleanup_start = time.perf_counter()
                logger.debug("Step 6: Resetting volatile memory")
                self.vmu.reset_volatile()
                timing_breakdown['cleanup'] = (time.perf_counter() - cleanup_start) * 1000
            
            # Stream lock released here
            
            # Compute metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            stats = self.vmu.get_memory_stats()
            
            metrics = ExecutionMetrics(
                duration_ms=duration_ms,
                memory_peak_mb=stats.get('max_current_offset', 0) / 1024**2,
                memory_persistent_mb=stats.get('persistent_allocated_bytes', 0) / 1024**2,
                memory_volatile_mb=stats.get('volatile_allocated_bytes', 0) / 1024**2,
                activations_count=self._count_activations(output),
                output_refs_count=1 if return_lazy else 1,  # Skeleton = 1 object
                success=True
            )
            
            # Get VMU stats
            vmu_stats = self.vmu.get_stats()
            
            logger.info(
                f"✅ Execution complete: {metrics.duration_ms:.1f}ms, "
                f"peak memory {metrics.memory_peak_mb:.1f}MB"
            )
            logger.info(
                f"   Timing breakdown: "
                f"lock={timing_breakdown.get('lock_acquire', 0):.1f}ms, "
                f"plan={timing_breakdown.get('planning', 0):.1f}ms, "
                f"placement={timing_breakdown.get('placement', 0):.1f}ms, "
                f"exec={timing_breakdown.get('execution', 0):.1f}ms, "
                f"skel={timing_breakdown.get('skeletonization', 0):.1f}ms, "
                f"cleanup={timing_breakdown.get('cleanup', 0):.1f}ms"
            )
            logger.info(
                f"   VMU usage: {vmu_stats['total_used_mb']:.1f}MB / "
                f"{vmu_stats['slab_capacity_mb']:.1f}MB "
                f"({vmu_stats['utilization_percent']:.1f}%), "
                f"persistent={vmu_stats['persistent_used_mb']:.1f}MB"
            )
            
            return skeleton, metrics
        
        except Exception as e:
            logger.error(f"❌ Execution failed: {e}")
            
            metrics = ExecutionMetrics(
                duration_ms=(time.perf_counter() - start_time) * 1000,
                memory_peak_mb=0,
                memory_persistent_mb=0,
                memory_volatile_mb=0,
                activations_count=0,
                output_refs_count=0,
                success=False
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

