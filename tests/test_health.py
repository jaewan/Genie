"""Tests for health check service"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

from genie.cluster.health import (
    HealthChecker,
    HealthCheck,
    HealthReport,
    HealthStatus,
    create_health_checker
)
from genie.cluster.node_info import (
    NodeInfo,
    NodeRole,
    GPUInfo,
    create_local_node_info
)


@pytest.fixture
def mock_node():
    """Create mock node"""
    node = create_local_node_info('test-node', role=NodeRole.SERVER)
    # Add GPUs
    node.gpus = [
        GPUInfo(
            gpu_id=0,
            name='Test GPU 0',
            memory_total_mb=8000,
            memory_used_mb=2000,
            memory_free_mb=6000,
            utilization_percent=25.0,
            temperature_celsius=65.0,
            is_available=True
        ),
        GPUInfo(
            gpu_id=1,
            name='Test GPU 1',
            memory_total_mb=8000,
            memory_used_mb=3000,
            memory_free_mb=5000,
            utilization_percent=37.5,
            temperature_celsius=70.0,
            is_available=True
        )
    ]
    return node


def test_health_checker_creation(mock_node):
    """Test HealthChecker creation"""
    checker = HealthChecker(mock_node)
    
    assert checker.local_node == mock_node
    assert checker.last_report is None


def test_create_health_checker(mock_node):
    """Test factory function"""
    checker = create_health_checker(mock_node)
    
    assert isinstance(checker, HealthChecker)


def test_health_check_to_dict():
    """Test HealthCheck serialization"""
    check = HealthCheck(
        check_name='test_check',
        status=HealthStatus.HEALTHY,
        message='All good',
        details={'value': 42}
    )
    
    d = check.to_dict()
    
    assert d['check_name'] == 'test_check'
    assert d['status'] == 'healthy'
    assert d['message'] == 'All good'
    assert d['details']['value'] == 42
    assert 'timestamp' in d


def test_health_report_to_dict(mock_node):
    """Test HealthReport serialization"""
    checks = [
        HealthCheck('check1', HealthStatus.HEALTHY, 'OK'),
        HealthCheck('check2', HealthStatus.DEGRADED, 'Warning'),
        HealthCheck('check3', HealthStatus.UNHEALTHY, 'Error')
    ]
    
    report = HealthReport(
        overall_status=HealthStatus.DEGRADED,
        checks=checks,
        node_id='test-node'
    )
    
    d = report.to_dict()
    
    assert d['overall_status'] == 'degraded'
    assert d['node_id'] == 'test-node'
    assert len(d['checks']) == 3
    assert d['summary']['total_checks'] == 3
    assert d['summary']['healthy'] == 1
    assert d['summary']['degraded'] == 1
    assert d['summary']['unhealthy'] == 1


@pytest.mark.asyncio
async def test_gpu_health_check_all_healthy(mock_node):
    """Test GPU health check when all GPUs healthy"""
    checker = HealthChecker(mock_node)
    
    check = await checker._check_gpu_health()
    
    assert check.check_name == 'gpu_health'
    assert check.status == HealthStatus.HEALTHY
    assert check.details['total_gpus'] == 2
    assert check.details['healthy_gpus'] == 2


@pytest.mark.asyncio
async def test_gpu_health_check_some_unhealthy(mock_node):
    """Test GPU health check with some unhealthy GPUs"""
    # Make one GPU unhealthy by setting is_available=False
    mock_node.gpus[1].is_available = False
    mock_node.gpus[1].error_state = 'test_error'
    
    checker = HealthChecker(mock_node)
    
    check = await checker._check_gpu_health()
    
    assert check.check_name == 'gpu_health'
    # 1 out of 2 healthy = 50%, should be UNHEALTHY
    assert check.status == HealthStatus.UNHEALTHY
    assert check.details['healthy_gpus'] == 1


@pytest.mark.asyncio
async def test_gpu_health_check_overheated(mock_node):
    """Test GPU health check with overheating"""
    # Make one GPU overheated
    mock_node.gpus[0].temperature_celsius = 90.0
    
    checker = HealthChecker(mock_node)
    
    check = await checker._check_gpu_health()
    
    assert check.check_name == 'gpu_health'
    assert check.status == HealthStatus.DEGRADED
    assert 0 in check.details['overheated_gpus']


@pytest.mark.asyncio
async def test_gpu_health_check_no_gpus():
    """Test GPU health check with no GPUs"""
    node = create_local_node_info('client-node', role=NodeRole.CLIENT)
    node.gpus = []
    
    checker = HealthChecker(node)
    
    check = await checker._check_gpu_health()
    
    assert check.status == HealthStatus.HEALTHY
    assert check.details['gpu_count'] == 0
    assert 'normal for client' in check.message.lower()


@pytest.mark.asyncio
async def test_cpu_usage_check():
    """Test CPU usage check"""
    node = create_local_node_info('test-node')
    
    with patch('psutil.cpu_percent', return_value=45.0), \
         patch('psutil.cpu_count', return_value=8):
        
        checker = HealthChecker(node)
        check = await checker._check_cpu_usage()
    
    assert check.check_name == 'cpu_usage'
    assert check.status == HealthStatus.HEALTHY
    assert check.details['cpu_percent'] == 45.0
    assert check.details['cpu_count'] == 8


@pytest.mark.asyncio
async def test_cpu_usage_check_high():
    """Test CPU usage check when high"""
    node = create_local_node_info('test-node')
    
    with patch('psutil.cpu_percent', return_value=88.0):
        checker = HealthChecker(node)
        check = await checker._check_cpu_usage()
    
    assert check.status == HealthStatus.UNHEALTHY


@pytest.mark.asyncio
async def test_memory_check():
    """Test memory check"""
    node = create_local_node_info('test-node')
    
    # Mock memory info
    mock_memory = MagicMock()
    mock_memory.total = 16 * (1024**3)  # 16 GB
    mock_memory.available = 8 * (1024**3)  # 8 GB
    mock_memory.percent = 50.0
    
    with patch('psutil.virtual_memory', return_value=mock_memory):
        checker = HealthChecker(node)
        check = await checker._check_memory()
    
    assert check.check_name == 'memory'
    assert check.status == HealthStatus.HEALTHY
    assert check.details['used_percent'] == 50.0


@pytest.mark.asyncio
async def test_memory_check_critical():
    """Test memory check when critical"""
    node = create_local_node_info('test-node')
    
    mock_memory = MagicMock()
    mock_memory.total = 16 * (1024**3)
    mock_memory.available = 0.5 * (1024**3)  # Very low
    mock_memory.percent = 97.0
    
    with patch('psutil.virtual_memory', return_value=mock_memory):
        checker = HealthChecker(node)
        check = await checker._check_memory()
    
    assert check.status == HealthStatus.CRITICAL


@pytest.mark.asyncio
async def test_disk_space_check():
    """Test disk space check"""
    node = create_local_node_info('test-node')
    
    mock_disk = MagicMock()
    mock_disk.total = 500 * (1024**3)  # 500 GB
    mock_disk.free = 250 * (1024**3)   # 250 GB
    mock_disk.percent = 50.0
    
    with patch('psutil.disk_usage', return_value=mock_disk):
        checker = HealthChecker(node)
        check = await checker._check_disk_space()
    
    assert check.check_name == 'disk_space'
    assert check.status == HealthStatus.HEALTHY
    assert check.details['used_percent'] == 50.0


@pytest.mark.asyncio
async def test_disk_space_check_high():
    """Test disk space check when high"""
    node = create_local_node_info('test-node')
    
    mock_disk = MagicMock()
    mock_disk.total = 500 * (1024**3)
    mock_disk.free = 10 * (1024**3)  # Very low
    mock_disk.percent = 98.0
    
    with patch('psutil.disk_usage', return_value=mock_disk):
        checker = HealthChecker(node)
        check = await checker._check_disk_space()
    
    assert check.status == HealthStatus.CRITICAL


@pytest.mark.asyncio
async def test_network_check():
    """Test network check"""
    node = create_local_node_info('test-node')
    
    mock_net = MagicMock()
    mock_net.bytes_sent = 1000000
    mock_net.bytes_recv = 2000000
    mock_net.packets_sent = 10000
    mock_net.packets_recv = 15000
    mock_net.errin = 5
    mock_net.errout = 3
    
    with patch('psutil.net_io_counters', return_value=mock_net):
        checker = HealthChecker(node)
        check = await checker._check_network()
    
    assert check.check_name == 'network'
    assert check.status == HealthStatus.HEALTHY
    assert check.details['error_rate_percent'] < 1.0


@pytest.mark.asyncio
async def test_network_check_high_errors():
    """Test network check with high error rate"""
    node = create_local_node_info('test-node')
    
    mock_net = MagicMock()
    mock_net.bytes_sent = 1000000
    mock_net.bytes_recv = 2000000
    mock_net.packets_sent = 10000
    mock_net.packets_recv = 10000
    mock_net.errin = 600  # 6% error rate (>5% is UNHEALTHY)
    mock_net.errout = 600
    
    with patch('psutil.net_io_counters', return_value=mock_net):
        checker = HealthChecker(node)
        check = await checker._check_network()
    
    assert check.status == HealthStatus.UNHEALTHY  # 6% error rate is UNHEALTHY


@pytest.mark.asyncio
async def test_perform_health_check(mock_node):
    """Test full health check"""
    # Mock system calls
    mock_memory = MagicMock()
    mock_memory.total = 16 * (1024**3)
    mock_memory.available = 8 * (1024**3)
    mock_memory.percent = 50.0
    
    mock_disk = MagicMock()
    mock_disk.total = 500 * (1024**3)
    mock_disk.free = 250 * (1024**3)
    mock_disk.percent = 50.0
    
    mock_net = MagicMock()
    mock_net.bytes_sent = 1000000
    mock_net.bytes_recv = 2000000
    mock_net.packets_sent = 10000
    mock_net.packets_recv = 15000
    mock_net.errin = 5
    mock_net.errout = 3
    
    with patch('psutil.cpu_percent', return_value=45.0), \
         patch('psutil.cpu_count', return_value=8), \
         patch('psutil.virtual_memory', return_value=mock_memory), \
         patch('psutil.disk_usage', return_value=mock_disk), \
         patch('psutil.net_io_counters', return_value=mock_net):
        
        checker = HealthChecker(mock_node)
        report = await checker.perform_health_check()
    
    assert report.overall_status == HealthStatus.HEALTHY
    assert len(report.checks) == 5  # GPU, CPU, Memory, Disk, Network
    assert report.node_id == 'test-node'
    
    # Check summary
    summary = report.to_dict()['summary']
    assert summary['total_checks'] == 5
    assert summary['healthy'] > 0


@pytest.mark.asyncio
async def test_determine_overall_status_critical(mock_node):
    """Test overall status when any check is critical"""
    checker = HealthChecker(mock_node)
    
    checks = [
        HealthCheck('check1', HealthStatus.HEALTHY, 'OK'),
        HealthCheck('check2', HealthStatus.CRITICAL, 'Critical'),  # One critical
        HealthCheck('check3', HealthStatus.HEALTHY, 'OK')
    ]
    
    status = checker._determine_overall_status(checks)
    
    assert status == HealthStatus.CRITICAL


@pytest.mark.asyncio
async def test_determine_overall_status_unhealthy(mock_node):
    """Test overall status when majority unhealthy"""
    checker = HealthChecker(mock_node)
    
    checks = [
        HealthCheck('check1', HealthStatus.UNHEALTHY, 'Bad'),
        HealthCheck('check2', HealthStatus.UNHEALTHY, 'Bad'),
        HealthCheck('check3', HealthStatus.HEALTHY, 'OK')
    ]
    
    status = checker._determine_overall_status(checks)
    
    assert status == HealthStatus.UNHEALTHY


@pytest.mark.asyncio
async def test_determine_overall_status_degraded(mock_node):
    """Test overall status when some degraded"""
    checker = HealthChecker(mock_node)
    
    checks = [
        HealthCheck('check1', HealthStatus.HEALTHY, 'OK'),
        HealthCheck('check2', HealthStatus.DEGRADED, 'Warning'),
        HealthCheck('check3', HealthStatus.HEALTHY, 'OK')
    ]
    
    status = checker._determine_overall_status(checks)
    
    assert status == HealthStatus.DEGRADED


def test_get_last_report(mock_node):
    """Test getting last health report"""
    checker = HealthChecker(mock_node)
    
    assert checker.get_last_report() is None
    
    # Perform check
    report = HealthReport(
        overall_status=HealthStatus.HEALTHY,
        checks=[],
        node_id='test-node'
    )
    checker.last_report = report
    
    assert checker.get_last_report() == report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

