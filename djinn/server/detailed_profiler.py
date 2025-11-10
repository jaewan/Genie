"""
Fine-grained profiler for remote execution.

Tracks:
- Per-operation execution time
- Network transfer breakdown
- Serialization overhead
- GPU memory usage
- Phase-specific metrics

Design principles:
- Thread-safe (lock-free fast path)
- Low overhead (<1% when enabled)
- Hierarchical (request → operation)
- Statistical (percentiles, not just averages)
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import json
import logging

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class OperationProfile:
    """Profile for a single operation execution."""
    operation: str
    execution_time_ms: float
    input_shapes: List[tuple]
    output_shape: tuple
    device: str
    execution_phase: str
    memory_allocated_mb: float
    memory_reserved_mb: float


@dataclass
class RequestProfile:
    """Profile for entire request."""
    request_id: str
    total_time_ms: float

    # Breakdown
    serialization_ms: float = 0.0
    network_send_ms: float = 0.0
    server_queue_ms: float = 0.0
    execution_ms: float = 0.0
    network_return_ms: float = 0.0
    deserialization_ms: float = 0.0

    # Execution details
    operations: List[OperationProfile] = field(default_factory=list)
    num_operations: int = 0
    execution_strategy: str = ""

    # Optimization
    tensorrt_used: bool = False
    fusion_used: bool = False
    cache_hit: bool = False


class DetailedProfiler:
    """
    Detailed profiler with minimal overhead.

    Design principles:
    - Thread-safe (lock-free fast path)
    - Low overhead (<1% when enabled)
    - Hierarchical (request → operation)
    - Statistical (percentiles, not just averages)
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._request_profiles: Dict[str, RequestProfile] = {}
        self._lock = threading.Lock()

        # Aggregated statistics
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        self.phase_stats: Dict[str, List[float]] = defaultdict(list)

    def start_request(self, request_id: str) -> 'RequestContext':
        """Start profiling a request."""
        if not self.enabled:
            return RequestContext(None, request_id)

        profile = RequestProfile(
            request_id=request_id,
            total_time_ms=0.0,
            serialization_ms=0.0,
            network_send_ms=0.0,
            server_queue_ms=0.0,
            execution_ms=0.0,
            network_return_ms=0.0,
            deserialization_ms=0.0
        )

        with self._lock:
            self._request_profiles[request_id] = profile

        return RequestContext(self, request_id)

    def record_operation(
        self,
        request_id: str,
        operation: str,
        execution_time_ms: float,
        input_shapes: List[tuple],
        output_shape: tuple,
        execution_phase: str = "unknown"
    ):
        """Record operation execution."""
        if not self.enabled:
            return

        profile = OperationProfile(
            operation=operation,
            execution_time_ms=execution_time_ms,
            input_shapes=input_shapes,
            output_shape=output_shape,
            device=str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu",
            execution_phase=execution_phase,
            memory_allocated_mb=torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            memory_reserved_mb=torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0
        )

        with self._lock:
            if request_id in self._request_profiles:
                self._request_profiles[request_id].operations.append(profile)

            # Update aggregated stats
            self.operation_stats[operation].append(execution_time_ms)
            self.phase_stats[execution_phase].append(execution_time_ms)

    def get_hotspots(self, top_k: int = 10) -> List[tuple]:
        """
        Identify performance hotspots.

        Returns:
            List of (operation, avg_time_ms, count, total_time_ms)
        """
        hotspots = []
        for op, times in self.operation_stats.items():
            avg = sum(times) / len(times)
            count = len(times)
            total = sum(times)
            hotspots.append((op, avg, count, total))

        # Sort by total time (biggest impact)
        hotspots.sort(key=lambda x: x[3], reverse=True)
        return hotspots[:top_k]

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        import numpy as np

        stats = {
            'total_requests': len(self._request_profiles),
            'total_operations': sum(len(p.operations) for p in self._request_profiles.values()),
            'hotspots': self.get_hotspots(10),
            'phase_breakdown': {},
            'optimization_impact': {}
        }

        # Phase breakdown
        for phase, times in self.phase_stats.items():
            if times:
                stats['phase_breakdown'][phase] = {
                    'count': len(times),
                    'mean_ms': np.mean(times),
                    'median_ms': np.median(times),
                    'p95_ms': np.percentile(times, 95),
                    'p99_ms': np.percentile(times, 99),
                    'total_ms': sum(times)
                }

        # Optimization impact
        tensorrt_requests = [p for p in self._request_profiles.values() if p.tensorrt_used]
        baseline_requests = [p for p in self._request_profiles.values() if not p.tensorrt_used]

        if tensorrt_requests and baseline_requests:
            stats['optimization_impact']['tensorrt_speedup'] = (
                np.mean([p.execution_ms for p in baseline_requests]) /
                np.mean([p.execution_ms for p in tensorrt_requests])
            )

        return stats

    def export_to_json(self, filepath: str):
        """Export profiles to JSON for analysis."""
        data = {
            'profiles': [
                {
                    'request_id': p.request_id,
                    'total_time_ms': p.total_time_ms,
                    'breakdown': {
                        'serialization': p.serialization_ms,
                        'network_send': p.network_send_ms,
                        'execution': p.execution_ms,
                        'network_return': p.network_return_ms,
                        'deserialization': p.deserialization_ms
                    },
                    'operations': [
                        {
                            'op': op.operation,
                            'time_ms': op.execution_time_ms,
                            'phase': op.execution_phase
                        }
                        for op in p.operations
                    ]
                }
                for p in self._request_profiles.values()
            ],
            'statistics': self.get_statistics()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class RequestContext:
    """Context manager for request profiling."""

    def __init__(self, profiler: Optional[DetailedProfiler], request_id: str):
        self.profiler = profiler
        self.request_id = request_id
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler and self.start_time:
            elapsed = (time.perf_counter() - self.start_time) * 1000
            with self.profiler._lock:
                if self.request_id in self.profiler._request_profiles:
                    self.profiler._request_profiles[self.request_id].total_time_ms = elapsed

    def record_phase(self, phase: str, duration_ms: float):
        """Record timing for a phase."""
        if self.profiler:
            with self.profiler._lock:
                if self.request_id in self.profiler._request_profiles:
                    profile = self.profiler._request_profiles[self.request_id]
                    setattr(profile, f"{phase}_ms", duration_ms)


# Global profiler instance
_global_profiler = DetailedProfiler(enabled=False)


def get_profiler() -> DetailedProfiler:
    """Get global profiler instance."""
    return _global_profiler


def enable_profiling():
    """Enable global profiling."""
    global _global_profiler
    _global_profiler.enabled = True


def disable_profiling():
    """Disable global profiling."""
    global _global_profiler
    _global_profiler.enabled = False
