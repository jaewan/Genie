"""
Stub scheduler for Phase 1 - Intentionally simple.

Phase 1: Always use remote GPU 0 (no placement decisions)
Phase 2: Add multi-GPU placement
Phase 3: Add cost-based optimization
Phase 4: Add semantic-aware scheduling

This replaces the 1500+ LOC semantic scheduler to avoid unnecessary
complexity in Phase 1. We can enable advanced scheduling in Phase 2
once the basic integration is working.
"""

import logging
from typing import Dict, Any, List, Optional
from genie.core.graph_interface import GenieGraph
from genie.core.types import WorkloadType

logger = logging.getLogger(__name__)


# ============================================================================
# BASE SCHEDULER INTERFACE (For consistency with semantic_scheduler.py)
# ============================================================================

class BaseScheduler:
    """Abstract base for schedulers (for compatibility)."""
    def schedule(self, graph: GenieGraph, workload_type: WorkloadType = WorkloadType.GENERIC):
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        raise NotImplementedError


class StubScheduler(BaseScheduler):
    """
    Phase 1 scheduler: Always use remote GPU 0.
    
    This is intentionally simple. Advanced scheduling (multi-GPU placement,
    cost-based optimization, semantic awareness) comes in Phase 2+.
    
    Design principle: Do the simplest thing that could possibly work.
    """
    
    def __init__(self):
        """Initialize stub scheduler with caching."""
        self.decision_count = 0
        self.device_usage = {}  # Track per-device load
        self.kv_cache_devices = {}  # Track KV cache placement

        # ✅ OPTIMIZATION: Placement decision caching
        self._placement_cache: Dict[str, Dict] = {}  # Cache placement decisions
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 1000  # Limit cache size

        # Known servers (populated by coordinator)
        self.known_servers = []

        # Feature toggles for ablation studies
        self.enable_colocation = True
        self.enable_pattern_detection = True
        self.enable_phase_detection = True
        self.enable_cost_model = True

        # Network topology (for cost estimation)
        self.network_bandwidth_gbps = 100.0
        self.network_latency_ms = 1.0

        # Discover available devices
        self.available_devices = self._discover_devices()
        logger.info(f"StubScheduler initialized (Phase 1: semantic-aware, {len(self.available_devices)} devices)")
    
    # ========================================================================
    # NEW API (BaseScheduler interface)
    # ========================================================================
    
    def schedule_graph(self, graph: GenieGraph, 
                      workload_type: WorkloadType = WorkloadType.GENERIC) -> 'Ok[ExecutionPlan]':
        """✅ NEW: Unified interface compatible with SemanticScheduler."""
        try:
            from genie.core.exceptions import Ok
            from genie.scheduler.semantic_scheduler import ExecutionPlan
            
            # Convert graph to placement decisions
            placements = {}
            for node in graph.nodes():
                # Use legacy scheduling logic
                decision = self.schedule_operation(
                    operation=node.operation,
                    metadata=node.metadata or {}
                )
                placements[node.id] = decision['device']
            
            plan = ExecutionPlan(placements=placements)
            return Ok(plan)
        except Exception as e:
            from genie.core.exceptions import Err, SchedulingError
            return Err(SchedulingError(f"Scheduling failed: {e}"))
    
    # ========================================================================
    # LEGACY API (operation-based interface)
    # ========================================================================
    
    def schedule_operation(
        self,
        operation: str,
        inputs: List = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        ⚠️ LEGACY: Schedule operation (old API).

        Phase 1: Enhanced with caching and basic semantic awareness:
        1. LLM decode → co-locate with KV cache
        2. Large tensors → prefer device with most free memory
        3. Default → round-robin
        4. Caching → reuse previous decisions when possible

        Args:
            operation: Operation name (e.g., 'aten::add')
            inputs: Input tensors (for shape analysis)
            metadata: Optional metadata (shapes, dtypes, phase info)

        Returns:
            Scheduling decision dict with device placement and explanation
        """
        self.decision_count += 1
        metadata = metadata or {}

        # ✅ OPTIMIZATION: Create cache key based on operation characteristics
        cache_key = self._create_cache_key(operation, metadata)

        # Check cache first
        if cache_key in self._placement_cache:
            cached_decision = self._placement_cache[cache_key]
            # Validate cached decision (device still available)
            if cached_decision['device'] in self.available_devices:
                self._cache_hits += 1
                logger.debug(f"Cache hit for {operation} → {cached_decision['device']}")
                return cached_decision
            else:
                # Remove invalid cached decision
                del self._placement_cache[cache_key]
                self._cache_misses += 1

        # Cache miss - make new decision
        self._cache_misses += 1

        # Rule 1: LLM decode co-location (only if semantic features enabled)
        if self.enable_colocation and self.enable_phase_detection and self._is_llm_decode(operation, metadata):
            # Try to place on same device as previous decode ops
            cache_key = metadata.get('model_id', 'default')
            if cache_key in self.kv_cache_devices:
                device = self.kv_cache_devices[cache_key]
                logger.debug(f"LLM decode: co-locate on {device}")
                decision = {
                    'device': device,
                    'strategy': 'colocate_kv_cache',
                    'priority': 10,
                    'timeout': 30.0,
                    'explanation': f'Co-located with KV cache for {cache_key}'
                }
            else:
                # First decode op - pick device and remember it
                device = self._pick_least_loaded_device()
                self.kv_cache_devices[cache_key] = device
                logger.debug(f"LLM decode: new cache on {device}")
                decision = {
                    'device': device,
                    'strategy': 'new_kv_cache',
                    'priority': 10,
                    'timeout': 30.0,
                    'explanation': f'New KV cache on {device}'
                }

        # Rule 2: Large tensor → prefer device with memory
        elif self._is_large_tensor(metadata):
            device = self._pick_device_by_memory()
            decision = {
                'device': device,
                'strategy': 'memory_aware',
                'priority': 5,
                'timeout': 60.0,
                'explanation': f'Large tensor → memory-optimized device'
            }

        # Rule 3: Default round-robin
        else:
            device = self._pick_round_robin()
            decision = {
                'device': device,
                'strategy': 'round_robin',
                'priority': 1,
                'timeout': 30.0,
                'explanation': 'Default round-robin placement'
            }

        # ✅ OPTIMIZATION: Cache the decision (with size limit)
        if len(self._placement_cache) < self._max_cache_size:
            self._placement_cache[cache_key] = decision
        else:
            # Remove oldest entries (simple LRU approximation)
            oldest_key = next(iter(self._placement_cache))
            del self._placement_cache[oldest_key]
            self._placement_cache[cache_key] = decision

        logger.debug(f"New decision for {operation} → {decision['device']} (cache: {self._cache_hits}/{self._cache_hits + self._cache_misses})")
        return decision
    
    # ✅ BACKWARD COMPAT: Old method name
    def schedule(self, operation_or_graph=None, inputs: List = None,
                metadata: Optional[Dict[str, Any]] = None, workload_type: WorkloadType = None):
        """Backward compatible schedule method that routes to correct implementation."""
        if isinstance(operation_or_graph, str):
            # Old API: schedule(operation, inputs, metadata)
            return self.schedule_operation(operation_or_graph, inputs, metadata)
        elif hasattr(operation_or_graph, 'nodes'):
            # New API: schedule(graph, workload_type)
            return self.schedule_graph(operation_or_graph, workload_type or WorkloadType.GENERIC)
        else:
            raise TypeError("schedule() requires either operation str or GenieGraph")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics including cache performance."""
        cache_total = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / cache_total if cache_total > 0 else 0

        return {
            'scheduler_type': 'stub',
            'phase': 1,
            'decisions_made': self.decision_count,
            'available_devices': len(self.available_devices),
            'device_usage': dict(self.device_usage),
            'kv_cache_models': len(self.kv_cache_devices),
            'cache_size': len(self._placement_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': cache_hit_rate,
            'status': 'Phase 1 - Semantic-aware placement with caching'
        }
    
    def _is_llm_decode(self, operation: str, metadata: Dict) -> bool:
        """Detect LLM decode pattern."""
        # Heuristic 1: Small batch size (decode is sequential)
        shapes = metadata.get('input_shapes', [])
        if shapes and len(shapes[0]) >= 2:
            batch_size = shapes[0][0]
            if batch_size == 1:
                return True

        # Heuristic 2: Explicit phase annotation
        if metadata.get('phase') == 'llm_decode':
            return True

        # Heuristic 3: Operation is attention + small batch
        if 'attention' in operation.lower() or 'matmul' in operation.lower():
            if shapes and shapes[0][0] == 1:
                return True

        return False

    def _create_cache_key(self, operation: str, metadata: Dict) -> str:
        """Create cache key based on operation characteristics."""
        # Create deterministic key from operation properties
        key_parts = [operation]

        # Include tensor size information
        total_bytes = 0
        shapes = metadata.get('input_shapes', [])
        dtypes = metadata.get('input_dtypes', [])

        for shape, dtype in zip(shapes, dtypes):
            if shape and dtype:
                numel = 1
                for dim in shape:
                    numel *= dim
                bytes_per_elem = {'torch.float32': 4, 'torch.float16': 2, 'torch.int64': 8, 'torch.int32': 4}.get(dtype, 4)
                total_bytes += numel * bytes_per_elem

        # Categorize by size (small, medium, large)
        if total_bytes > 1_000_000_000:  # 1GB
            key_parts.append('large')
        elif total_bytes > 100_000_000:  # 100MB
            key_parts.append('medium')
        else:
            key_parts.append('small')

        # Include phase information
        phase = metadata.get('phase', 'unknown')
        key_parts.append(phase)

        # Include model ID for co-location
        model_id = metadata.get('model_id', 'default')
        key_parts.append(model_id)

        return '|'.join(key_parts)

    def _is_large_tensor(self, metadata: Dict) -> bool:
        """Check if tensor is large enough for memory-aware placement."""
        total_bytes = sum(
            self._estimate_tensor_size(shape, dtype)
            for shape, dtype in zip(
                metadata.get('input_shapes', []),
                metadata.get('input_dtypes', [])
            )
        )

        return total_bytes > 1_000_000_000  # 1GB threshold

    def _discover_devices(self) -> List[str]:
        """Discover available remote devices."""
        devices = []

        # In Phase 1, we work with actual server addresses
        # For testing: use localhost with different ports to simulate multiple devices
        # In production: would discover actual remote servers
        base_port = 5556  # Default data port

        # Create logical devices mapped to localhost (for Phase 1 testing)
        # Each "device" is just a different port on localhost
        for i in range(4):
            devices.append(f'localhost:{base_port + i}')

        return devices

    def _pick_least_loaded_device(self) -> str:
        """Pick device with lowest load."""
        if not self.device_usage:
            return self.available_devices[0]

        return min(self.device_usage, key=self.device_usage.get)

    def _pick_device_by_memory(self) -> str:
        """Pick device with most free memory."""
        # Simplified: just pick GPU 0 for now
        # In production: query actual memory usage
        return self.available_devices[0]

    def _pick_round_robin(self) -> str:
        """Round-robin across devices."""
        idx = self.decision_count % len(self.available_devices)
        device = self.available_devices[idx]
        self.device_usage[device] = self.device_usage.get(device, 0) + 1
        return device

    def _estimate_tensor_size(self, shape, dtype_str) -> int:
        """Estimate tensor size in bytes."""
        if not shape:
            return 0

        numel = 1
        for dim in shape:
            numel *= dim

        # Dtype size mapping
        dtype_sizes = {
            'torch.float32': 4,
            'torch.float16': 2,
            'torch.int64': 8,
            'torch.int32': 4,
        }

        bytes_per_elem = dtype_sizes.get(dtype_str, 4)
        return numel * bytes_per_elem

    def register_server(self, server_address: str):
        """Register a known server address."""
        if server_address not in self.known_servers:
            self.known_servers.append(server_address)
            logger.info(f"Registered server: {server_address}")
            # Update available devices if needed
            self._update_available_devices()

    def _update_available_devices(self):
        """Update available devices based on known servers."""
        # If we have known servers, use them as devices
        if self.known_servers:
            self.available_devices = self.known_servers.copy()
        else:
            # Fallback to localhost discovery
            self.available_devices = self._discover_devices()

    def update_network_topology(self, bandwidth_gbps: float, latency_ms: float):
        """Update network topology for cost estimation."""
        self.network_bandwidth_gbps = bandwidth_gbps
        self.network_latency_ms = latency_ms
        logger.info(f"Updated network topology: {bandwidth_gbps}Gbps, {latency_ms}ms")

    def reset_stats(self):
        """Reset statistics counters."""
        self.decision_count = 0
        self.device_usage.clear()
        self.kv_cache_devices.clear()
        logger.info("StubScheduler stats reset")


# Singleton instance
_scheduler_instance: Optional[StubScheduler] = None


def get_scheduler() -> StubScheduler:
    """Get global scheduler instance (singleton)."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = StubScheduler()
    return _scheduler_instance


def reset_scheduler():
    """Reset global scheduler instance."""
    global _scheduler_instance
    _scheduler_instance = None
