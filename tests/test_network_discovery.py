"""Tests for network discovery service"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, mock_open

from genie.runtime.network_discovery import (
    discover_network_capabilities,
    BackendCapability,
    get_backend_priority,
    select_best_backend,
    _test_tcp_backend,
    _test_dpdk_backend,
    _test_dpdk_gpudev_backend,
    _test_rdma_backend,
)


@pytest.mark.asyncio
async def test_tcp_backend_success():
    """Test TCP backend detection when connection succeeds"""
    
    # Mock successful connection
    async def mock_open_connection(host, port):
        reader = AsyncMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        return reader, writer
    
    with patch('asyncio.open_connection', side_effect=mock_open_connection):
        cap = await _test_tcp_backend('localhost', 5555, timeout=1.0)
    
    assert cap.available
    assert cap.name == 'tcp'
    assert cap.latency_us is not None
    assert cap.latency_us > 0
    assert cap.priority == 1
    assert not cap.supports_zero_copy
    assert not cap.supports_gpu_direct


@pytest.mark.asyncio
async def test_tcp_backend_timeout():
    """Test TCP backend detection when connection times out"""
    
    async def mock_timeout(host, port):
        await asyncio.sleep(10)  # Will timeout
    
    with patch('asyncio.open_connection', side_effect=mock_timeout):
        cap = await _test_tcp_backend('unreachable', 5555, timeout=0.1)
    
    assert not cap.available
    assert cap.name == 'tcp'
    assert cap.error_message is not None
    assert 'timeout' in cap.error_message.lower()


@pytest.mark.asyncio
async def test_tcp_backend_connection_refused():
    """Test TCP backend when connection is refused"""
    
    async def mock_refused(host, port):
        raise ConnectionRefusedError("Connection refused")
    
    with patch('asyncio.open_connection', side_effect=mock_refused):
        cap = await _test_tcp_backend('localhost', 9999, timeout=1.0)
    
    assert not cap.available
    assert cap.error_message is not None


@pytest.mark.asyncio
async def test_dpdk_backend_not_available():
    """Test DPDK backend when module not available"""
    
    with patch('builtins.__import__', side_effect=ImportError("No module genie_data_plane")):
        cap = await _test_dpdk_backend(timeout=1.0)
    
    assert not cap.available
    assert cap.name == 'dpdk'
    assert cap.priority == 3
    assert 'genie_data_plane' in cap.error_message


@pytest.mark.asyncio
async def test_dpdk_backend_available():
    """Test DPDK backend when all requirements met"""
    
    # Mock DPDK module available
    mock_module = MagicMock()
    
    # Mock meminfo with huge pages
    meminfo_content = """
    MemTotal:       16384000 kB
    HugePages_Total:     512
    HugePages_Free:      512
    """
    
    m_open = mock_open(read_data=meminfo_content)
    
    with patch('builtins.__import__', return_value=mock_module), \
         patch('builtins.open', m_open), \
         patch('os.path.exists', return_value=True):
        
        cap = await _test_dpdk_backend(timeout=1.0)
    
    assert cap.available
    assert cap.name == 'dpdk'
    assert cap.supports_zero_copy
    assert not cap.supports_gpu_direct
    assert cap.priority == 3
    assert cap.bandwidth_gbps == 10.0


@pytest.mark.asyncio
async def test_dpdk_backend_no_huge_pages():
    """Test DPDK backend when huge pages not configured"""
    
    # Mock DPDK module available but no huge pages
    mock_module = MagicMock()
    meminfo_content = """
    MemTotal:       16384000 kB
    HugePages_Total:     0
    """
    
    m_open = mock_open(read_data=meminfo_content)
    
    with patch('builtins.__import__', return_value=mock_module), \
         patch('builtins.open', m_open):
        
        cap = await _test_dpdk_backend(timeout=1.0)
    
    assert not cap.available
    assert 'Huge pages' in cap.error_message


@pytest.mark.asyncio
async def test_dpdk_gpudev_backend_not_available():
    """Test DPDK+GPUDirect when DPDK not available"""
    
    with patch('genie.runtime.network_discovery._test_dpdk_backend') as mock_dpdk:
        mock_dpdk.return_value = BackendCapability(
            name='dpdk',
            available=False,
            priority=3,
            error_message="DPDK not available"
        )
        
        cap = await _test_dpdk_gpudev_backend(timeout=1.0)
    
    assert not cap.available
    assert cap.name == 'dpdk_gpudev'
    assert 'DPDK not available' in cap.error_message


@pytest.mark.asyncio
async def test_dpdk_gpudev_backend_no_cuda():
    """Test DPDK+GPUDirect when CUDA not available"""
    
    with patch('genie.runtime.network_discovery._test_dpdk_backend') as mock_dpdk:
        mock_dpdk.return_value = BackendCapability(
            name='dpdk',
            available=True,
            priority=3
        )
        
        with patch('builtins.__import__', side_effect=ImportError("No module torch")):
            cap = await _test_dpdk_gpudev_backend(timeout=1.0)
    
    assert not cap.available
    assert 'PyTorch not available' in cap.error_message


@pytest.mark.asyncio
async def test_dpdk_gpudev_backend_available():
    """Test DPDK+GPUDirect when all requirements met"""
    
    # Mock DPDK available
    mock_dpdk_cap = BackendCapability(
        name='dpdk',
        available=True,
        priority=3
    )
    
    # Mock torch with CUDA
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 2
    
    # Mock /proc/modules
    modules_content = "nvidia_p2p 16384 0 - Live 0xffffffffc0a00000\n"
    m_open = mock_open(read_data=modules_content)
    
    with patch('genie.runtime.network_discovery._test_dpdk_backend', return_value=mock_dpdk_cap), \
         patch('builtins.__import__', return_value=mock_torch), \
         patch('builtins.open', m_open):
        
        cap = await _test_dpdk_gpudev_backend(timeout=1.0)
    
    assert cap.available
    assert cap.name == 'dpdk_gpudev'
    assert cap.supports_zero_copy
    assert cap.supports_gpu_direct
    assert cap.priority == 5  # Highest priority


@pytest.mark.asyncio
async def test_rdma_backend_not_available():
    """Test RDMA backend when hardware not present"""
    
    with patch('glob.glob', return_value=[]):  # No IB devices
        cap = await _test_rdma_backend('localhost', timeout=1.0)
    
    assert not cap.available
    assert cap.name == 'rdma'
    assert 'InfiniBand' in cap.error_message


@pytest.mark.asyncio
async def test_rdma_backend_available():
    """Test RDMA backend when hardware present"""
    
    # Mock InfiniBand devices
    ib_devices = ['/sys/class/infiniband/mlx5_0']
    
    # Mock /proc/modules with RDMA
    modules_content = "rdma_cm 65536 0 - Live 0xffffffffc0b00000\n"
    m_open = mock_open(read_data=modules_content)
    
    # Mock libibverbs
    mock_cdll = MagicMock()
    
    with patch('glob.glob', return_value=ib_devices), \
         patch('builtins.open', m_open), \
         patch('ctypes.CDLL', return_value=mock_cdll):
        
        cap = await _test_rdma_backend('localhost', timeout=1.0)
    
    assert cap.available
    assert cap.name == 'rdma'
    assert cap.supports_zero_copy
    assert cap.supports_gpu_direct
    assert cap.priority == 4
    assert cap.bandwidth_gbps == 100.0


@pytest.mark.asyncio
async def test_discover_network_capabilities_tcp_only():
    """Test full discovery when only TCP available"""
    
    async def mock_tcp(addr, port, timeout):
        return BackendCapability(name='tcp', available=True, priority=1)
    
    async def mock_dpdk(timeout):
        return BackendCapability(name='dpdk', available=False, priority=3)
    
    async def mock_dpdk_gpu(timeout):
        return BackendCapability(name='dpdk_gpudev', available=False, priority=5)
    
    async def mock_rdma(addr, timeout):
        return BackendCapability(name='rdma', available=False, priority=4)
    
    with patch('genie.runtime.network_discovery._test_tcp_backend', side_effect=mock_tcp), \
         patch('genie.runtime.network_discovery._test_dpdk_backend', side_effect=mock_dpdk), \
         patch('genie.runtime.network_discovery._test_dpdk_gpudev_backend', side_effect=mock_dpdk_gpu), \
         patch('genie.runtime.network_discovery._test_rdma_backend', side_effect=mock_rdma):
        
        result = await discover_network_capabilities('localhost', 5555, timeout=1.0)
    
    assert result['recommended_backend'] == 'tcp'
    assert 'tcp' in result['available_backends']
    assert len(result['available_backends']) == 1
    assert result['discovery_time'] > 0


@pytest.mark.asyncio
async def test_discover_network_capabilities_dpdk_preferred():
    """Test discovery prefers DPDK over TCP"""
    
    async def mock_tcp(addr, port, timeout):
        return BackendCapability(name='tcp', available=True, priority=1)
    
    async def mock_dpdk(timeout):
        return BackendCapability(name='dpdk', available=True, priority=3)
    
    async def mock_dpdk_gpu(timeout):
        return BackendCapability(name='dpdk_gpudev', available=False, priority=5)
    
    async def mock_rdma(addr, timeout):
        return BackendCapability(name='rdma', available=False, priority=4)
    
    with patch('genie.runtime.network_discovery._test_tcp_backend', side_effect=mock_tcp), \
         patch('genie.runtime.network_discovery._test_dpdk_backend', side_effect=mock_dpdk), \
         patch('genie.runtime.network_discovery._test_dpdk_gpudev_backend', side_effect=mock_dpdk_gpu), \
         patch('genie.runtime.network_discovery._test_rdma_backend', side_effect=mock_rdma):
        
        result = await discover_network_capabilities('localhost', 5555, timeout=1.0)
    
    assert result['recommended_backend'] == 'dpdk'
    assert 'tcp' in result['available_backends']
    assert 'dpdk' in result['available_backends']


@pytest.mark.asyncio
async def test_discover_network_capabilities_gpudev_best():
    """Test discovery prefers DPDK+GPUDirect when available"""
    
    async def mock_tcp(addr, port, timeout):
        return BackendCapability(name='tcp', available=True, priority=1)
    
    async def mock_dpdk(timeout):
        return BackendCapability(name='dpdk', available=True, priority=3, supports_zero_copy=True)
    
    async def mock_dpdk_gpu(timeout):
        return BackendCapability(
            name='dpdk_gpudev',
            available=True,
            priority=5,
            supports_zero_copy=True,
            supports_gpu_direct=True
        )
    
    async def mock_rdma(addr, timeout):
        return BackendCapability(name='rdma', available=False, priority=4)
    
    with patch('genie.runtime.network_discovery._test_tcp_backend', side_effect=mock_tcp), \
         patch('genie.runtime.network_discovery._test_dpdk_backend', side_effect=mock_dpdk), \
         patch('genie.runtime.network_discovery._test_dpdk_gpudev_backend', side_effect=mock_dpdk_gpu), \
         patch('genie.runtime.network_discovery._test_rdma_backend', side_effect=mock_rdma):
        
        result = await discover_network_capabilities('localhost', 5555, timeout=1.0)
    
    assert result['recommended_backend'] == 'dpdk_gpudev'
    assert len(result['available_backends']) == 3
    assert 'backend_details' in result
    gpudev_details = result['backend_details']['dpdk_gpudev']
    assert gpudev_details['supports_gpu_direct'] is True


@pytest.mark.asyncio
async def test_discover_network_fallback_tcp():
    """Test discovery falls back to TCP when all tests fail"""
    
    async def mock_tcp_fail(addr, port, timeout):
        return BackendCapability(name='tcp', available=False, priority=1, error_message="Failed")
    
    async def mock_dpdk(timeout):
        return BackendCapability(name='dpdk', available=False, priority=3)
    
    async def mock_dpdk_gpu(timeout):
        return BackendCapability(name='dpdk_gpudev', available=False, priority=5)
    
    async def mock_rdma(addr, timeout):
        return BackendCapability(name='rdma', available=False, priority=4)
    
    with patch('genie.runtime.network_discovery._test_tcp_backend', side_effect=mock_tcp_fail), \
         patch('genie.runtime.network_discovery._test_dpdk_backend', side_effect=mock_dpdk), \
         patch('genie.runtime.network_discovery._test_dpdk_gpudev_backend', side_effect=mock_dpdk_gpu), \
         patch('genie.runtime.network_discovery._test_rdma_backend', side_effect=mock_rdma):
        
        result = await discover_network_capabilities('localhost', 5555, timeout=1.0)
    
    # Should force TCP fallback
    assert result['recommended_backend'] == 'tcp'
    assert 'tcp' in result['available_backends']


def test_get_backend_priority():
    """Test backend priority ordering"""
    priority = get_backend_priority()
    
    assert len(priority) == 4
    assert priority[0] == 'dpdk_gpudev'  # Best
    assert priority[1] == 'rdma'
    assert priority[2] == 'dpdk'
    assert priority[3] == 'tcp'  # Worst (fallback)


def test_select_best_backend_user_preference():
    """Test backend selection respects user preference"""
    available = ['tcp', 'dpdk', 'rdma']
    
    # User prefers DPDK
    selected = select_best_backend(available, user_preference='dpdk')
    assert selected == 'dpdk'
    
    # User prefers unavailable backend - ignore and pick best
    selected = select_best_backend(available, user_preference='dpdk_gpudev')
    assert selected == 'rdma'  # Highest priority among available


def test_select_best_backend_priority():
    """Test backend selection by priority"""
    # Only TCP available
    selected = select_best_backend(['tcp'])
    assert selected == 'tcp'
    
    # TCP and DPDK available - prefer DPDK
    selected = select_best_backend(['tcp', 'dpdk'])
    assert selected == 'dpdk'
    
    # All available except GPUDirect - prefer RDMA
    selected = select_best_backend(['tcp', 'dpdk', 'rdma'])
    assert selected == 'rdma'
    
    # All available - prefer GPUDirect
    selected = select_best_backend(['tcp', 'dpdk', 'rdma', 'dpdk_gpudev'])
    assert selected == 'dpdk_gpudev'


def test_select_best_backend_empty():
    """Test backend selection with empty list falls back to TCP"""
    selected = select_best_backend([])
    assert selected == 'tcp'


@pytest.mark.asyncio
async def test_discover_custom_backends():
    """Test discovery with custom backend list"""
    
    async def mock_tcp(addr, port, timeout):
        return BackendCapability(name='tcp', available=True, priority=1)
    
    async def mock_dpdk(timeout):
        return BackendCapability(name='dpdk', available=True, priority=3)
    
    # Only test TCP and DPDK
    with patch('genie.runtime.network_discovery._test_tcp_backend', side_effect=mock_tcp), \
         patch('genie.runtime.network_discovery._test_dpdk_backend', side_effect=mock_dpdk):
        
        result = await discover_network_capabilities(
            'localhost', 5555,
            timeout=1.0,
            test_backends=['tcp', 'dpdk']  # Skip RDMA and GPUDirect
        )
    
    assert 'tcp' in result['available_backends']
    assert 'dpdk' in result['available_backends']
    # Should not have tested these
    assert 'rdma' not in result['backend_details']
    assert 'dpdk_gpudev' not in result['backend_details']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

