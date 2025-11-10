"""
Common utilities shared across all benchmarks.

This module contains shared functionality used by multiple benchmark types:
- GPU monitoring utilities
- Logging setup
- Result data structures
- Output directory management
- Common imports and setup
"""

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
import sys
import os

# Add Djinn to path for all benchmarks
sys.path.insert(0, '/home/jae/Genie')


def setup_logging(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Standardized logging setup for all benchmarks."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(name)


def get_gpu_utilization(device_id: int = 0) -> float:
    """Get GPU utilization for a specific device."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--id={device_id}", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2
        )
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        return 0.0


def get_gpu_memory_usage(device_id: int = 0) -> float:
    """Get GPU memory usage in MB for a specific device."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--id={device_id}", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2
        )
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        return 0.0


def measure_utilization_during_execution(device_id: int, duration_ms: int = 1000,
                                       interval_ms: int = 100) -> float:
    """Measure GPU utilization during execution."""
    measurements = []
    start = time.time()

    while (time.time() - start) * 1000 < duration_ms:
        util = get_gpu_utilization(device_id)
        measurements.append(util)
        time.sleep(interval_ms / 1000.0)

    return sum(measurements) / len(measurements) if measurements else 0.0


@dataclass
class BenchmarkResult:
    """Base class for benchmark results."""
    success: bool
    error_msg: Optional[str] = None
    execution_time_ms: Optional[float] = None
    gpu_utilization: Optional[float] = None
    memory_usage_mb: Optional[float] = None


@dataclass
class ProfilingData:
    """System profiling data shared across benchmarks."""
    network_packets_sent: int = 0
    network_packets_received: int = 0
    network_bytes_sent: int = 0
    network_bytes_received: int = 0
    gpu_utilization_avg: float = 0.0
    gpu_memory_used_mb: float = 0.0
    cpu_utilization_avg: float = 0.0
    server_requests_handled: int = 0
    client_requests_made: int = 0
    remote_execution_success: bool = False
    component_usage: Dict[str, Any] = field(default_factory=dict)


class BenchmarkOutputManager:
    """Manages output directories and file writing for benchmarks."""

    def __init__(self, output_dir: str, create_dir: bool = True):
        self.output_dir = Path(output_dir)
        if create_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, filename: str) -> Path:
        """Get full path for a file in the output directory."""
        return self.output_dir / filename

    def write_json(self, data: Any, filename: str) -> Path:
        """Write data as JSON to the output directory."""
        import json
        path = self.get_path(filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def write_text(self, text: str, filename: str) -> Path:
        """Write text to a file in the output directory."""
        path = self.get_path(filename)
        with open(path, 'w') as f:
            f.write(text)
        return path


def torch_device_info() -> Dict[str, Any]:
    """Get comprehensive PyTorch device information."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_info = {
                'name': torch.cuda.get_device_name(i),
                'capability': torch.cuda.get_device_capability(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory / 1024**3,  # GB
                'memory_free': torch.cuda.mem_get_info(i)[0] / 1024**3,  # GB
            }
            info[f'device_{i}'] = device_info

    return info


def ensure_cuda_device(device: Optional[int] = None):
    """Ensure we're on the correct CUDA device."""
    if torch.cuda.is_available():
        target_device = device if device is not None else 0
        torch.cuda.set_device(target_device)
        return torch.device(f'cuda:{target_device}')
    return torch.device('cpu')


def cleanup_gpu_memory():
    """Aggressive GPU memory cleanup."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    import gc
    gc.collect()
