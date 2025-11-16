"""Utilities for Djinn benchmarking."""

from benchmarks.utils.server_spawner import RemoteServerManager
from benchmarks.utils.common import *

# ✅ FIX: Lazy import models to avoid transformers dependency if not available
try:
    from benchmarks.utils.models import *
except ImportError:
    # transformers not available - models module will fail gracefully
    pass

from benchmarks.utils.metrics import *

__all__ = [
    # Server management
    'RemoteServerManager',

    # Common utilities
    'setup_logging',
    'get_gpu_utilization',
    'get_gpu_memory_usage',
    'measure_utilization_during_execution',
    'BenchmarkResult',
    'ProfilingData',
    'BenchmarkOutputManager',
    'torch_device_info',
    'ensure_cuda_device',
    'cleanup_gpu_memory',

    # Model utilities
    'ModelManager',
    'model_manager',
    'get_model_memory_usage',
    'generate_sample_input',
    'benchmark_model_forward',

    # Metrics
    'LatencyMetrics',
    'ThroughputMetrics',
    'MemoryMetrics',
    'BenchmarkMetrics',
    'measure_function_latency',
    'calculate_efficiency_metrics',
    'format_metrics_for_display',
]
