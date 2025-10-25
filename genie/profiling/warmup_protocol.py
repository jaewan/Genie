"""
PHASE 4.2: Improved Warm-up Protocol

Implements a comprehensive warm-up protocol to reduce measurement variance by:
1. CPU pinning for deterministic process placement
2. GPU memory clearing to eliminate cold-start effects
3. Cache warm-up for stable performance
4. Variance measurement and validation

Expected impact: Reduce measurement variance from 136ms std dev to <50ms
"""

import os
import psutil
import torch
import threading
import logging
from typing import Optional, List, Callable
from dataclasses import dataclass
import time
import gc

logger = logging.getLogger(__name__)


@dataclass
class WarmupStats:
    """Statistics for warm-up process."""
    cpu_cores_pinned: int
    gpu_memory_cleared: bool
    warmup_runs: int
    variance_before_ms: float  # Std dev before warmup
    variance_after_ms: float  # Std dev after warmup
    improvement_percent: float  # % reduction in variance


class CPUPinner:
    """
    Handles CPU affinity to pin process to specific cores.
    
    Benefit: Prevents OS scheduler from moving process between cores,
    which causes cache misses and inconsistent performance.
    """
    
    def __init__(self, num_cores: Optional[int] = None):
        """
        Initialize CPU pinner.
        
        Args:
            num_cores: Number of cores to pin to. If None, uses all available cores.
        """
        try:
            total_cores = psutil.cpu_count(logical=False)
        except:
            total_cores = os.cpu_count() or 4
        
        if num_cores is None:
            num_cores = max(1, total_cores // 2)  # Use half of available cores
        
        self.num_cores = min(num_cores, total_cores)
        self.original_affinity = None
    
    def pin_process(self):
        """Pin current process to specific CPU cores."""
        try:
            p = psutil.Process()
            self.original_affinity = p.cpu_affinity()
            
            # Pin to first N cores
            cores_to_use = list(range(self.num_cores))
            p.cpu_affinity(cores_to_use)
            
            logger.info(f"Process pinned to {self.num_cores} CPU cores: {cores_to_use}")
            return True
        except Exception as e:
            logger.warning(f"Failed to pin CPU: {e}")
            return False
    
    def unpin_process(self):
        """Restore original CPU affinity."""
        if self.original_affinity is None:
            return
        
        try:
            p = psutil.Process()
            p.cpu_affinity(self.original_affinity)
            logger.info("CPU affinity restored")
        except Exception as e:
            logger.warning(f"Failed to restore CPU affinity: {e}")


class GPUWarmer:
    """Handles GPU warm-up and memory clearing."""
    
    def __init__(self, device: str = "cuda:0"):
        """
        Initialize GPU warmer.
        
        Args:
            device: GPU device to warm up
        """
        self.device = device
        self.has_cuda = torch.cuda.is_available()
    
    def warm_up_gpu(self, num_runs: int = 3):
        """
        Warm up GPU with sample workloads.
        
        Args:
            num_runs: Number of warm-up runs
        """
        if not self.has_cuda:
            logger.debug("CUDA not available, skipping GPU warm-up")
            return
        
        try:
            for i in range(num_runs):
                # Allocate and deallocate memory to warm up GPU
                _ = torch.randn(1024, 1024, device=self.device)
                
                # Run a sample computation
                a = torch.randn(512, 512, device=self.device)
                b = torch.randn(512, 512, device=self.device)
                _ = torch.matmul(a, b)
                
                # Synchronize to ensure computation completes
                torch.cuda.synchronize(self.device)
            
            logger.info(f"GPU {self.device} warmed up with {num_runs} runs")
        except Exception as e:
            logger.warning(f"GPU warm-up failed: {e}")
    
    def clear_gpu_memory(self):
        """Clear GPU memory caches."""
        if not self.has_cuda:
            return
        
        try:
            # Clear torch cache
            torch.cuda.empty_cache()
            
            # Clear GPU memory stats
            torch.cuda.reset_peak_memory_stats(self.device)
            
            # Synchronize to ensure cache clear completes
            torch.cuda.synchronize(self.device)
            
            logger.info(f"GPU {self.device} memory cleared")
        except Exception as e:
            logger.warning(f"GPU memory clear failed: {e}")


class SystemWarmer:
    """Handles system-level warm-up (CPU, memory, caches)."""
    
    @staticmethod
    def warm_up_system():
        """Warm up system caches and memory."""
        # Force garbage collection
        gc.collect()
        
        # Warm up memory allocation
        try:
            _ = bytearray(1024 * 1024)  # 1MB
        except:
            pass
        
        # Warm up CPU cache by doing some work
        result = 0
        for i in range(1000000):
            result += i * i
        
        logger.info("System caches warmed up")


class WarmupProtocol:
    """
    Comprehensive warm-up protocol for deterministic measurements.
    
    PHASE 4.2 Strategy:
    
    1. Pin process to CPU cores (deterministic scheduling)
    2. Warm up CPU caches and memory
    3. Warm up GPU with sample workloads
    4. Clear GPU memory for fresh start
    5. Repeat workload multiple times for stable measurements
    6. Track variance reduction
    
    Expected result: Reduce measurement variance by 70%+
    """
    
    def __init__(self, device: str = "cuda:0", num_cores: Optional[int] = None):
        """
        Initialize warm-up protocol.
        
        Args:
            device: GPU device to warm up
            num_cores: Number of CPU cores to pin to
        """
        self.device = device
        self.cpu_pinner = CPUPinner(num_cores)
        self.gpu_warmer = GPUWarmer(device)
        self.system_warmer = SystemWarmer()
    
    def execute(self, workload_fn: Callable, num_warmup_runs: int = 5, 
               num_measurement_runs: int = 10) -> WarmupStats:
        """
        Execute complete warm-up protocol.
        
        Args:
            workload_fn: Function to run as workload. Should return latency in ms.
            num_warmup_runs: Number of warm-up runs
            num_measurement_runs: Number of measurement runs after warm-up
            
        Returns:
            WarmupStats with variance reduction metrics
        """
        
        # Phase 1: Measure baseline variance (before optimization)
        logger.info("Phase 1: Measuring baseline variance...")
        baseline_latencies = []
        for i in range(5):
            latency = workload_fn()
            baseline_latencies.append(latency)
        
        baseline_variance = self._calculate_std_dev(baseline_latencies)
        logger.info(f"Baseline std dev: {baseline_variance:.2f}ms")
        
        # Phase 2: Apply optimizations
        logger.info("Phase 2: Applying warm-up optimizations...")
        
        # Pin CPU
        cpu_pinned = self.cpu_pinner.pin_process()
        
        # Warm up system
        self.system_warmer.warm_up_system()
        
        # Warm up GPU
        self.gpu_warmer.warm_up_gpu(num_runs=num_warmup_runs)
        
        # Clear GPU memory
        self.gpu_warmer.clear_gpu_memory()
        
        # Phase 3: Run workload for actual measurement
        logger.info("Phase 3: Running measurements after warm-up...")
        warmup_latencies = []
        for i in range(num_measurement_runs):
            latency = workload_fn()
            warmup_latencies.append(latency)
        
        # Phase 4: Calculate improvement
        logger.info("Phase 4: Calculating variance reduction...")
        warmup_variance = self._calculate_std_dev(warmup_latencies)
        improvement = (baseline_variance - warmup_variance) / baseline_variance * 100
        
        logger.info(f"After warm-up std dev: {warmup_variance:.2f}ms")
        logger.info(f"Variance reduction: {improvement:.1f}%")
        
        return WarmupStats(
            cpu_cores_pinned=self.cpu_pinner.num_cores if cpu_pinned else 0,
            gpu_memory_cleared=True,
            warmup_runs=num_warmup_runs,
            variance_before_ms=baseline_variance,
            variance_after_ms=warmup_variance,
            improvement_percent=improvement
        )
    
    def cleanup(self):
        """Clean up after warm-up (restore CPU affinity, etc.)."""
        self.cpu_pinner.unpin_process()
    
    @staticmethod
    def _calculate_std_dev(values: List[float]) -> float:
        """Calculate standard deviation of latencies."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        std_dev = variance ** 0.5
        
        return std_dev


class MeasurementSession:
    """
    Context manager for measurement sessions with warm-up.
    
    Usage:
        with MeasurementSession(device="cuda:0") as session:
            # Warm-up already applied
            latency = run_workload()
    """
    
    def __init__(self, device: str = "cuda:0", num_cores: Optional[int] = None,
                enable_warmup: bool = True):
        """
        Initialize measurement session.
        
        Args:
            device: GPU device
            num_cores: CPU cores to pin to
            enable_warmup: Whether to enable warm-up
        """
        self.device = device
        self.enable_warmup = enable_warmup
        self.protocol = WarmupProtocol(device, num_cores) if enable_warmup else None
    
    def __enter__(self):
        """Start measurement session."""
        if self.protocol:
            # Start basic warm-up
            self.protocol.cpu_pinner.pin_process()
            self.protocol.system_warmer.warm_up_system()
            self.protocol.gpu_warmer.warm_up_gpu(num_runs=3)
            self.protocol.gpu_warmer.clear_gpu_memory()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End measurement session."""
        if self.protocol:
            self.protocol.cleanup()
    
    def run_workload(self, workload_fn: Callable) -> float:
        """Run workload during measurement session."""
        return workload_fn()
