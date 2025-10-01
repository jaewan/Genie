"""Tests for cluster initialization API"""

import asyncio
import os
import pytest
from unittest.mock import Mock, patch, AsyncMock

from genie.cluster.init import (
    ClusterConfig,
    ClusterState,
    init,
    shutdown,
    get_cluster_state,
    is_initialized,
    _build_config,
)
from genie.cluster.node_info import NodeRole, NodeStatus


@pytest.fixture
def reset_cluster():
    """Reset cluster state before and after each test"""
    ClusterState.reset_singleton()
    yield
    ClusterState.reset_singleton()


def test_cluster_config_validation():
    """Test ClusterConfig validation"""
    # Valid config
    config = ClusterConfig(
        master_addr='localhost',
        master_port=5555,
        backend='tcp'
    )
    config.validate()  # Should not raise
    
    # Invalid discovery method
    config = ClusterConfig(discovery_method='invalid')
    with pytest.raises(ValueError, match='Invalid discovery_method'):
        config.validate()
    
    # Invalid node role
    config = ClusterConfig(
        master_addr='localhost',
        node_role='invalid'
    )
    with pytest.raises(ValueError, match='Invalid node_role'):
        config.validate()
    
    # Invalid backend
    config = ClusterConfig(
        master_addr='localhost',
        backend='invalid'
    )
    with pytest.raises(ValueError, match='Invalid backend'):
        config.validate()
    
    # Static discovery requires master_addr
    config = ClusterConfig(
        discovery_method='static',
        master_addr=None
    )
    with pytest.raises(ValueError, match='master_addr required'):
        config.validate()


def test_cluster_state_singleton():
    """Test ClusterState singleton pattern"""
    state1 = ClusterState.get()
    state2 = ClusterState.get()
    
    assert state1 is state2
    assert not state1.initialized


def test_cluster_state_reset(reset_cluster):
    """Test ClusterState reset"""
    state = ClusterState.get()
    
    # Add some data
    state.initialized = True
    state.stats['init_time'] = 1.5
    
    # Reset
    state.reset()
    
    assert not state.initialized
    assert state.stats['init_time'] == 0.0


def test_cluster_state_node_management(reset_cluster):
    """Test node management in ClusterState"""
    from genie.cluster.node_info import NodeInfo
    
    state = ClusterState.get()
    
    # Add node
    node = NodeInfo(
        node_id='test-node',
        hostname='test-host',
        role=NodeRole.SERVER,
        host='127.0.0.1',
        control_port=5555,
        data_port=5556,
        network_backend='tcp'
    )
    
    state.add_node(node)
    
    assert 'test-node' in state.nodes
    assert state.get_node('test-node') == node
    assert state.stats['total_nodes_discovered'] == 1
    
    # Remove node
    state.remove_node('test-node')
    
    assert 'test-node' not in state.nodes


def test_cluster_state_healthy_nodes(reset_cluster):
    """Test getting healthy nodes"""
    import time
    from genie.cluster.node_info import NodeInfo
    
    state = ClusterState.get()
    
    # Add healthy node
    node1 = NodeInfo(
        node_id='node1',
        hostname='host1',
        role=NodeRole.SERVER,
        host='127.0.0.1',
        control_port=5555,
        data_port=5556,
        network_backend='tcp',
        status=NodeStatus.ACTIVE
    )
    state.add_node(node1)
    
    # Add unhealthy node (old heartbeat)
    node2 = NodeInfo(
        node_id='node2',
        hostname='host2',
        role=NodeRole.SERVER,
        host='127.0.0.2',
        control_port=5555,
        data_port=5556,
        network_backend='tcp',
        status=NodeStatus.ACTIVE,
        last_heartbeat=time.time() - 100  # Old heartbeat
    )
    state.add_node(node2)
    
    # Get healthy nodes
    healthy = state.get_healthy_nodes(timeout=60.0)
    
    assert len(healthy) == 1
    assert healthy[0].node_id == 'node1'


def test_build_config_from_env():
    """Test building config from environment variables"""
    # Set environment variables
    env_vars = {
        'GENIE_MASTER_ADDR': '192.168.1.100',
        'GENIE_MASTER_PORT': '6666',
        'GENIE_NODE_ID': 'test-node',
        'GENIE_NODE_ROLE': 'server',
        'GENIE_BACKEND': 'dpdk'
    }
    
    with patch.dict(os.environ, env_vars):
        config = _build_config()
    
    assert config.master_addr == '192.168.1.100'
    assert config.master_port == 6666
    assert config.node_id == 'test-node'
    assert config.node_role == 'server'
    assert config.backend == 'dpdk'


def test_build_config_arg_priority():
    """Test that explicit arguments override environment variables"""
    env_vars = {
        'GENIE_MASTER_ADDR': '192.168.1.100',
        'GENIE_MASTER_PORT': '6666',
    }
    
    with patch.dict(os.environ, env_vars):
        config = _build_config(
            master_addr='10.0.0.1',
            master_port=7777
        )
    
    # Explicit args should override env vars
    assert config.master_addr == '10.0.0.1'
    assert config.master_port == 7777


@pytest.mark.asyncio
async def test_init_basic(reset_cluster):
    """Test basic initialization"""
    from genie.cluster.node_info import create_local_node_info
    
    # Mock the initialization phases but create a local node
    async def mock_init_phases(state, config):
        # Simulate phase 1 - create local node
        state.local_node = create_local_node_info('test-node')
    
    with patch('genie.cluster.init._run_initialization_phases', side_effect=mock_init_phases):
        state = await init(
            master_addr='localhost',
            master_port=5555,
            backend='tcp',
            timeout=1.0
        )
        
        assert state.initialized
        assert state.config is not None
        assert state.config.master_addr == 'localhost'
        assert state.local_node is not None


@pytest.mark.asyncio
async def test_init_already_initialized(reset_cluster):
    """Test initialization when already initialized"""
    state = ClusterState.get()
    state.initialized = True
    
    result = await init(master_addr='localhost')
    
    assert result is state
    assert state.initialized


@pytest.mark.asyncio
async def test_init_timeout(reset_cluster):
    """Test initialization timeout"""
    async def slow_init(*args, **kwargs):
        await asyncio.sleep(10)  # Longer than timeout
    
    with patch('genie.cluster.init._run_initialization_phases', side_effect=slow_init):
        with pytest.raises(asyncio.TimeoutError):
            await init(
                master_addr='localhost',
                timeout=0.1  # Very short timeout
            )
    
    # State should be reset after failure
    state = ClusterState.get()
    assert not state.initialized


@pytest.mark.asyncio
async def test_init_error_recovery(reset_cluster):
    """Test that state is reset on initialization error"""
    async def failing_init(*args, **kwargs):
        raise RuntimeError("Initialization failed")
    
    with patch('genie.cluster.init._run_initialization_phases', side_effect=failing_init):
        with pytest.raises(RuntimeError, match="Initialization failed"):
            await init(master_addr='localhost')
    
    # State should be reset after failure
    state = ClusterState.get()
    assert not state.initialized


@pytest.mark.asyncio
async def test_shutdown(reset_cluster):
    """Test shutdown"""
    state = ClusterState.get()
    state.initialized = True
    
    # Create mock monitoring tasks
    task1 = asyncio.create_task(asyncio.sleep(100))
    task2 = asyncio.create_task(asyncio.sleep(100))
    state.monitor_tasks = [task1, task2]
    
    # Mock components
    mock_transport = Mock()
    mock_transport.shutdown = AsyncMock()
    state.transport_coordinator = mock_transport
    
    mock_control = Mock()
    mock_control.shutdown = AsyncMock()
    state.control_integration = mock_control
    
    await shutdown()
    
    # Verify tasks were cancelled
    assert task1.cancelled()
    assert task2.cancelled()
    
    # Verify components were shut down
    mock_transport.shutdown.assert_called_once()
    mock_control.shutdown.assert_called_once()
    
    # Verify state was reset
    assert not state.initialized
    assert state.transport_coordinator is None
    assert state.control_integration is None


@pytest.mark.asyncio
async def test_shutdown_not_initialized(reset_cluster):
    """Test shutdown when not initialized"""
    # Should not raise an error
    await shutdown()


@pytest.mark.asyncio
async def test_shutdown_with_errors(reset_cluster):
    """Test shutdown handles component errors gracefully"""
    state = ClusterState.get()
    state.initialized = True
    
    # Mock components that raise errors
    state.transport_coordinator = Mock()
    state.transport_coordinator.shutdown = AsyncMock(side_effect=RuntimeError("Error"))
    
    state.control_integration = Mock()
    state.control_integration.shutdown = AsyncMock(side_effect=RuntimeError("Error"))
    
    # Should not raise, but should log warnings
    await shutdown()
    
    assert not state.initialized


def test_is_initialized(reset_cluster):
    """Test is_initialized function"""
    assert not is_initialized()
    
    state = ClusterState.get()
    state.initialized = True
    
    assert is_initialized()


def test_get_cluster_state(reset_cluster):
    """Test get_cluster_state function"""
    state = get_cluster_state()
    
    assert isinstance(state, ClusterState)
    assert state is ClusterState.get()


@pytest.mark.asyncio
async def test_init_server_mode(reset_cluster):
    """Test initialization in server mode"""
    from genie.cluster.node_info import create_local_node_info
    
    async def mock_init_phases(state, config):
        state.local_node = create_local_node_info('test-server', role=NodeRole.SERVER)
    
    with patch('genie.cluster.init._run_initialization_phases', side_effect=mock_init_phases):
        state = await init(
            node_role='server',
            master_port=5555,
            backend='tcp',
            timeout=1.0
        )
        
        assert state.initialized
        assert state.config.node_role == 'server'


@pytest.mark.asyncio
async def test_init_auto_node_id(reset_cluster):
    """Test automatic node ID generation"""
    from genie.cluster.node_info import create_local_node_info
    
    async def mock_init_phases(state, config):
        # Simulate the auto-generation that happens in _create_local_node
        if not config.node_id:
            import socket
            hostname = socket.gethostname()
            config.node_id = f"{hostname}-auto"
        state.local_node = create_local_node_info(config.node_id)
    
    with patch('genie.cluster.init._run_initialization_phases', side_effect=mock_init_phases):
        state = await init(
            master_addr='localhost',
            node_id=None,  # Should auto-generate
            timeout=1.0
        )
        
        assert state.config.node_id is not None
        assert len(state.config.node_id) > 0
        assert state.local_node is not None


@pytest.mark.asyncio
async def test_monitoring_tasks_creation(reset_cluster):
    """Test that monitoring tasks are created"""
    state = ClusterState.get()
    
    # Mock the local node
    from genie.cluster.node_info import create_local_node_info
    state.local_node = create_local_node_info('test-node')
    state.initialized = True
    
    # Mock config
    config = ClusterConfig(
        master_addr='localhost',
        enable_heartbeat=True,
        enable_gpu_monitoring=True,
        enable_health_checks=True
    )
    state.config = config
    
    # Import monitoring start function
    from genie.cluster.init import _start_monitoring
    
    await _start_monitoring(state, config)
    
    # Should have created monitoring tasks
    assert len(state.monitor_tasks) > 0
    
    # Clean up
    for task in state.monitor_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

