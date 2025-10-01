"""
Network Discovery Service

Automatically discovers and tests available network backends:
- TCP (always available)
- DPDK (zero-copy, requires configuration)
- DPDK GPUDirect (zero-copy + GPU direct access)
- RDMA (InfiniBand, requires hardware)

Recommends the fastest available backend based on:
1. Availability
2. Capabilities  
3. Performance characteristics
"""

import asyncio
import logging
import socket
import struct
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class BackendCapability:
    """Capabilities of a network backend"""
    name: str  # 'tcp', 'dpdk', 'dpdk_gpudev', 'rdma'
    available: bool
    latency_us: Optional[float] = None  # Measured latency in microseconds
    bandwidth_gbps: Optional[float] = None  # Measured bandwidth
    supports_zero_copy: bool = False
    supports_gpu_direct: bool = False
    priority: int = 0  # Higher is better
    error_message: Optional[str] = None


async def discover_network_capabilities(
    target_addr: str,
    target_port: int,
    timeout: float = 5.0,
    test_backends: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Discover available network backends and recommend the best one.
    
    Args:
        target_addr: Address of remote node to test against
        target_port: Port of remote node
        timeout: Timeout for discovery in seconds
        test_backends: List of backends to test (None = test all)
    
    Returns:
        Dictionary with:
            - 'available_backends': List[str] - Available backend names
            - 'recommended_backend': str - Best backend to use
            - 'backend_details': Dict[str, BackendCapability] - Details per backend
            - 'discovery_time': float - Time taken for discovery
    
    Example:
        ```python
        info = await discover_network_capabilities(
            target_addr='192.168.1.100',
            target_port=5555
        )
        print(f"Use backend: {info['recommended_backend']}")
        ```
    """
    start_time = time.time()
    
    if test_backends is None:
        test_backends = ['tcp', 'dpdk', 'dpdk_gpudev', 'rdma']
    
    logger.info(f"Discovering network capabilities for {target_addr}:{target_port}")
    logger.info(f"Testing backends: {test_backends}")
    
    # Test each backend
    capabilities: Dict[str, BackendCapability] = {}
    
    # Always test TCP first (baseline)
    if 'tcp' in test_backends:
        capabilities['tcp'] = await _test_tcp_backend(target_addr, target_port, timeout)
    
    # Test DPDK
    if 'dpdk' in test_backends:
        capabilities['dpdk'] = await _test_dpdk_backend(timeout)
    
    # Test DPDK with GPUDirect
    if 'dpdk_gpudev' in test_backends:
        capabilities['dpdk_gpudev'] = await _test_dpdk_gpudev_backend(timeout)
    
    # Test RDMA
    if 'rdma' in test_backends:
        capabilities['rdma'] = await _test_rdma_backend(target_addr, timeout)
    
    # Filter available backends
    available = [
        name for name, cap in capabilities.items()
        if cap.available
    ]
    
    # Select recommended backend (highest priority among available)
    if available:
        recommended = max(
            available,
            key=lambda name: capabilities[name].priority
        )
    else:
        # Fallback to TCP even if test failed
        logger.warning("No backends available, forcing TCP fallback")
        recommended = 'tcp'
        if 'tcp' not in capabilities:
            capabilities['tcp'] = BackendCapability(
                name='tcp',
                available=True,
                priority=1
            )
        else:
            capabilities['tcp'].available = True
    
    discovery_time = time.time() - start_time
    
    logger.info(f"Discovery complete in {discovery_time:.2f}s")
    logger.info(f"Available: {available}")
    logger.info(f"Recommended: {recommended}")
    
    return {
        'available_backends': available if available else ['tcp'],
        'recommended_backend': recommended,
        'backend_details': {
            name: {
                'available': cap.available,
                'latency_us': cap.latency_us,
                'bandwidth_gbps': cap.bandwidth_gbps,
                'supports_zero_copy': cap.supports_zero_copy,
                'supports_gpu_direct': cap.supports_gpu_direct,
                'priority': cap.priority,
                'error': cap.error_message
            }
            for name, cap in capabilities.items()
        },
        'discovery_time': discovery_time
    }


async def _test_tcp_backend(
    target_addr: str,
    target_port: int,
    timeout: float
) -> BackendCapability:
    """Test TCP backend availability and latency"""
    logger.debug("Testing TCP backend...")
    
    try:
        # Attempt connection
        start = time.time()
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(target_addr, target_port),
            timeout=timeout
        )
        latency = (time.time() - start) * 1_000_000  # Convert to microseconds
        
        # Close connection
        writer.close()
        await writer.wait_closed()
        
        logger.info(f"  TCP: Available (latency: {latency:.0f}µs)")
        
        return BackendCapability(
            name='tcp',
            available=True,
            latency_us=latency,
            bandwidth_gbps=1.0,  # Typical 1GbE
            supports_zero_copy=False,
            supports_gpu_direct=False,
            priority=1  # Lowest priority (fallback)
        )
        
    except asyncio.TimeoutError:
        logger.warning(f"  TCP: Connection timeout to {target_addr}:{target_port}")
        return BackendCapability(
            name='tcp',
            available=False,
            priority=1,
            error_message=f"Connection timeout to {target_addr}:{target_port}"
        )
    except Exception as e:
        logger.warning(f"  TCP: Connection failed: {e}")
        return BackendCapability(
            name='tcp',
            available=False,
            priority=1,
            error_message=str(e)
        )


async def _test_dpdk_backend(timeout: float) -> BackendCapability:
    """Test DPDK backend availability"""
    logger.debug("Testing DPDK backend...")
    
    try:
        # Check if DPDK library is available
        try:
            import genie_data_plane  # noqa: F401
            dpdk_available = True
        except ImportError:
            dpdk_available = False
            raise ImportError("genie_data_plane module not available")
        
        # Check if huge pages are configured
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                if 'HugePages_Total' in meminfo:
                    for line in meminfo.split('\n'):
                        if 'HugePages_Total' in line:
                            total_pages = int(line.split()[1])
                            if total_pages == 0:
                                raise RuntimeError("Huge pages not configured")
                else:
                    raise RuntimeError("Huge pages not available")
        except Exception as e:
            logger.debug(f"  DPDK: Huge pages check failed: {e}")
            raise
        
        # Check for /dev/hugepages
        import os
        if not os.path.exists('/dev/hugepages'):
            raise RuntimeError("/dev/hugepages not mounted")
        
        logger.info("  DPDK: Available (zero-copy capable)")
        
        return BackendCapability(
            name='dpdk',
            available=True,
            latency_us=5.0,  # Typical DPDK latency
            bandwidth_gbps=10.0,  # 10GbE typical
            supports_zero_copy=True,
            supports_gpu_direct=False,
            priority=3  # High priority
        )
        
    except Exception as e:
        logger.info(f"  DPDK: Not available ({e})")
        return BackendCapability(
            name='dpdk',
            available=False,
            priority=3,
            error_message=str(e)
        )


async def _test_dpdk_gpudev_backend(timeout: float) -> BackendCapability:
    """Test DPDK with GPUDirect backend"""
    logger.debug("Testing DPDK+GPUDirect backend...")
    
    try:
        # First check if DPDK is available
        dpdk_cap = await _test_dpdk_backend(timeout)
        if not dpdk_cap.available:
            raise RuntimeError("DPDK not available")
        
        # Check if CUDA/GPU is available
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                raise RuntimeError("No GPUs detected")
        except ImportError:
            raise RuntimeError("PyTorch not available")
        
        # Check if GPUDirect RDMA kernel module is loaded
        try:
            with open('/proc/modules', 'r') as f:
                modules = f.read()
                # Check for nvidia_p2p or gdrdrv module
                if 'nvidia_p2p' not in modules and 'gdrdrv' not in modules:
                    logger.debug("  GPUDirect kernel modules not loaded")
                    # Not fatal - can still work without optimal performance
        except Exception:
            pass
        
        logger.info("  DPDK+GPUDirect: Available (zero-copy + GPU direct)")
        
        return BackendCapability(
            name='dpdk_gpudev',
            available=True,
            latency_us=3.0,  # Lower latency with GPU direct
            bandwidth_gbps=25.0,  # 25GbE with GPU direct
            supports_zero_copy=True,
            supports_gpu_direct=True,
            priority=5  # Highest priority
        )
        
    except Exception as e:
        logger.info(f"  DPDK+GPUDirect: Not available ({e})")
        return BackendCapability(
            name='dpdk_gpudev',
            available=False,
            priority=5,
            error_message=str(e)
        )


async def _test_rdma_backend(target_addr: str, timeout: float) -> BackendCapability:
    """Test RDMA backend availability"""
    logger.debug("Testing RDMA backend...")
    
    try:
        # Check for InfiniBand devices
        import os
        import glob
        
        ib_devices = glob.glob('/sys/class/infiniband/*')
        if not ib_devices:
            raise RuntimeError("No InfiniBand devices found")
        
        # Check for rdma_cm module
        try:
            with open('/proc/modules', 'r') as f:
                modules = f.read()
                if 'rdma_cm' not in modules:
                    raise RuntimeError("RDMA kernel modules not loaded")
        except Exception:
            raise RuntimeError("Cannot check RDMA modules")
        
        # Check for libibverbs
        try:
            import ctypes
            ctypes.CDLL('libibverbs.so.1')
        except Exception:
            raise RuntimeError("libibverbs not available")
        
        logger.info("  RDMA: Available (InfiniBand detected)")
        
        return BackendCapability(
            name='rdma',
            available=True,
            latency_us=1.5,  # Very low latency
            bandwidth_gbps=100.0,  # InfiniBand typical
            supports_zero_copy=True,
            supports_gpu_direct=True,
            priority=4  # Second highest priority
        )
        
    except Exception as e:
        logger.info(f"  RDMA: Not available ({e})")
        return BackendCapability(
            name='rdma',
            available=False,
            priority=4,
            error_message=str(e)
        )


async def test_bandwidth(
    backend: str,
    target_addr: str,
    target_port: int,
    duration: float = 1.0
) -> float:
    """
    Test actual bandwidth for a backend.
    
    Args:
        backend: Backend name to test
        target_addr: Target address
        target_port: Target port
        duration: Test duration in seconds
    
    Returns:
        Bandwidth in Gbps
    """
    # TODO: Implement actual bandwidth testing
    # This would send test data and measure throughput
    logger.warning("Bandwidth testing not yet implemented")
    return 0.0


def get_backend_priority() -> List[str]:
    """
    Get backend priority order (best to worst).
    
    Returns:
        List of backend names in priority order
    """
    return [
        'dpdk_gpudev',  # Best: zero-copy + GPU direct
        'rdma',          # Second: InfiniBand zero-copy
        'dpdk',          # Third: DPDK zero-copy
        'tcp',           # Fallback: standard TCP
    ]


def select_best_backend(
    available: List[str],
    user_preference: Optional[str] = None
) -> str:
    """
    Select the best backend from available options.
    
    Args:
        available: List of available backend names
        user_preference: User's preferred backend (if any)
    
    Returns:
        Selected backend name
    """
    # If user specified a preference and it's available, use it
    if user_preference and user_preference in available:
        return user_preference
    
    # Otherwise, select highest priority available
    priority_order = get_backend_priority()
    
    for backend in priority_order:
        if backend in available:
            return backend
    
    # Fallback to TCP
    return 'tcp'

