"""Tests for cluster node information"""

import pytest
import time
from genie.cluster.node_info import (
    NodeInfo, NodeStatus, NodeRole, GPUInfo,
    create_local_node_info
)


def test_gpu_info_creation():
    """Test GPUInfo creation"""
    gpu = GPUInfo(
        gpu_id=0,
        name="Tesla V100",
        memory_total_mb=16384,
        memory_used_mb=4096,
        memory_free_mb=12288,
        utilization_percent=50.0
    )
    
    assert gpu.gpu_id == 0
    assert gpu.memory_utilization_percent == 25.0  # 4096/16384
    assert gpu.is_available


def test_gpu_info_memory_utilization():
    """Test GPU memory utilization calculation"""
    gpu = GPUInfo(
        gpu_id=0,
        name="Test GPU",
        memory_total_mb=10000,
        memory_used_mb=5000,
        memory_free_mb=5000,
        utilization_percent=50.0
    )
    
    assert gpu.memory_utilization_percent == 50.0
    
    # Test with zero total memory
    gpu_zero = GPUInfo(
        gpu_id=1,
        name="Test GPU",
        memory_total_mb=0,
        memory_used_mb=0,
        memory_free_mb=0,
        utilization_percent=0.0
    )
    assert gpu_zero.memory_utilization_percent == 0.0


def test_node_info_creation():
    """Test NodeInfo creation"""
    node = NodeInfo(
        node_id="test-node",
        hostname="test-host",
        role=NodeRole.CLIENT,
        host="127.0.0.1",
        control_port=5555,
        data_port=5556,
        network_backend="tcp"
    )
    
    assert node.node_id == "test-node"
    assert node.status == NodeStatus.UNKNOWN
    assert node.gpu_count == 0
    assert len(node.gpus) == 0


def test_node_health_check():
    """Test node health checking"""
    node = NodeInfo(
        node_id="test-node",
        hostname="test-host",
        role=NodeRole.SERVER,
        host="127.0.0.1",
        control_port=5555,
        data_port=5556,
        network_backend="tcp",
        status=NodeStatus.ACTIVE
    )
    
    # Fresh heartbeat - should be healthy
    assert node.is_healthy(timeout=60.0)
    
    # Old heartbeat - should be unhealthy
    node.last_heartbeat = time.time() - 100
    assert not node.is_healthy(timeout=60.0)


def test_node_health_check_status():
    """Test health check considers node status"""
    node = NodeInfo(
        node_id="test-node",
        hostname="test-host",
        role=NodeRole.SERVER,
        host="127.0.0.1",
        control_port=5555,
        data_port=5556,
        network_backend="tcp",
        status=NodeStatus.UNHEALTHY
    )
    
    # Even with fresh heartbeat, UNHEALTHY status makes it unhealthy
    assert not node.is_healthy(timeout=60.0)
    
    # DISCONNECTED status also makes it unhealthy
    node.status = NodeStatus.DISCONNECTED
    assert not node.is_healthy(timeout=60.0)


def test_node_update_from_heartbeat():
    """Test updating node from heartbeat"""
    node = NodeInfo(
        node_id="test-node",
        hostname="test-host",
        role=NodeRole.SERVER,
        host="127.0.0.1",
        control_port=5555,
        data_port=5556,
        network_backend="tcp"
    )
    
    heartbeat_data = {
        'status': 'active',
        'active_transfers': 3,
        'gpus': [
            {
                'gpu_id': 0,
                'name': 'A100',
                'memory_total_mb': 40960,
                'memory_used_mb': 10240,
                'memory_free_mb': 30720,
                'utilization_percent': 80.0
            }
        ]
    }
    
    old_heartbeat = node.last_heartbeat
    node.update_from_heartbeat(heartbeat_data)
    
    assert node.status == NodeStatus.ACTIVE
    assert node.active_transfers == 3
    assert node.gpu_count == 1
    assert node.gpus[0].name == 'A100'
    assert node.last_heartbeat > old_heartbeat


def test_create_local_node_info():
    """Test local node info creation"""
    node = create_local_node_info(
        node_id="local-test",
        role=NodeRole.CLIENT
    )
    
    assert node.node_id == "local-test"
    assert node.role == NodeRole.CLIENT
    assert node.hostname is not None
    assert node.host is not None
    # GPU count depends on local system


def test_node_to_dict():
    """Test node serialization"""
    node = NodeInfo(
        node_id="test-node",
        hostname="test-host",
        role=NodeRole.SERVER,
        host="192.168.1.1",
        control_port=5555,
        data_port=5556,
        network_backend="dpdk"
    )
    
    data = node.to_dict()
    
    assert data['node_id'] == "test-node"
    assert data['role'] == 'server'
    assert data['network_backend'] == 'dpdk'
    assert isinstance(data['gpus'], list)


def test_get_available_gpus():
    """Test getting available GPUs"""
    node = NodeInfo(
        node_id="test-node",
        hostname="test-host",
        role=NodeRole.SERVER,
        host="127.0.0.1",
        control_port=5555,
        data_port=5556,
        network_backend="tcp",
        gpus=[
            GPUInfo(gpu_id=0, name="GPU0", memory_total_mb=8000,
                   memory_used_mb=2000, memory_free_mb=6000,
                   utilization_percent=30.0, is_available=True),
            GPUInfo(gpu_id=1, name="GPU1", memory_total_mb=8000,
                   memory_used_mb=7000, memory_free_mb=1000,
                   utilization_percent=95.0, is_available=False),
        ]
    )
    
    available = node.get_available_gpus()
    assert len(available) == 1
    assert available[0].gpu_id == 0


def test_node_status_enum():
    """Test NodeStatus enum values"""
    assert NodeStatus.UNKNOWN.value == "unknown"
    assert NodeStatus.ACTIVE.value == "active"
    assert NodeStatus.UNHEALTHY.value == "unhealthy"


def test_node_role_enum():
    """Test NodeRole enum values"""
    assert NodeRole.CLIENT.value == "client"
    assert NodeRole.SERVER.value == "server"
    assert NodeRole.WORKER.value == "worker"
    assert NodeRole.MASTER.value == "master"


def test_gpu_info_with_optional_fields():
    """Test GPUInfo with optional fields"""
    gpu = GPUInfo(
        gpu_id=0,
        name="A100",
        memory_total_mb=40960,
        memory_used_mb=10240,
        memory_free_mb=30720,
        utilization_percent=80.0,
        temperature_celsius=65.5,
        power_draw_watts=250.0,
        compute_capability="8.0",
        cuda_cores=6912
    )
    
    assert gpu.temperature_celsius == 65.5
    assert gpu.power_draw_watts == 250.0
    assert gpu.compute_capability == "8.0"
    assert gpu.cuda_cores == 6912


def test_node_info_with_backends():
    """Test NodeInfo with multiple backends"""
    node = NodeInfo(
        node_id="test-node",
        hostname="test-host",
        role=NodeRole.SERVER,
        host="192.168.1.1",
        control_port=5555,
        data_port=5556,
        network_backend="dpdk_gpudev",
        available_backends=['tcp', 'dpdk', 'dpdk_gpudev']
    )
    
    assert node.network_backend == "dpdk_gpudev"
    assert 'tcp' in node.available_backends
    assert 'dpdk_gpudev' in node.available_backends

