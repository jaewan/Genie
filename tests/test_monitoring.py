"""Tests for enhanced resource monitoring"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

from genie.cluster.monitoring import (
    ResourceMonitor,
    MonitoringEvent,
    MonitoringEventType,
    create_resource_monitor
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
    # Add a GPU
    node.gpus = [
        GPUInfo(
            gpu_id=0,
            name='Test GPU',
            memory_total_mb=8000,
            memory_used_mb=2000,
            memory_free_mb=6000,
            utilization_percent=25.0,
            temperature_celsius=65.0,
            power_draw_watts=150.0,
            is_available=True
        )
    ]
    return node


def test_resource_monitor_creation(mock_node):
    """Test ResourceMonitor creation"""
    monitor = ResourceMonitor(mock_node, poll_interval=1.0)
    
    assert monitor.local_node == mock_node
    assert monitor.poll_interval == 1.0
    assert not monitor.running
    assert monitor.events_emitted == 0


def test_create_resource_monitor(mock_node):
    """Test factory function"""
    monitor = create_resource_monitor(mock_node, poll_interval=2.0)
    
    assert isinstance(monitor, ResourceMonitor)
    assert monitor.poll_interval == 2.0


@pytest.mark.asyncio
async def test_monitor_start_stop(mock_node):
    """Test starting and stopping monitor"""
    monitor = ResourceMonitor(mock_node)
    
    await monitor.start()
    assert monitor.running
    
    await monitor.stop()
    assert not monitor.running


@pytest.mark.asyncio
async def test_gpu_availability_event(mock_node):
    """Test GPU availability change detection"""
    events = []
    
    def event_callback(event):
        events.append(event)
    
    monitor = ResourceMonitor(mock_node, event_callback=event_callback)
    await monitor.start()
    
    # Initial GPU state (healthy)
    gpu1 = GPUInfo(
        gpu_id=0,
        name='Test GPU',
        memory_total_mb=8000,
        memory_used_mb=2000,
        memory_free_mb=6000,
        utilization_percent=25.0,
        is_available=True
    )
    
    await monitor._detect_gpu_changes([gpu1])
    
    # Change to unhealthy
    gpu2 = GPUInfo(
        gpu_id=0,
        name='Test GPU',
        memory_total_mb=8000,
        memory_used_mb=7500,
        memory_free_mb=500,
        utilization_percent=95.0,
        is_available=False,
        error_state='high_utilization'
    )
    
    await monitor._detect_gpu_changes([gpu2])
    
    # Should have emitted unavailable event (may also get memory pressure)
    assert len(events) >= 1
    unavailable_events = [e for e in events if e.event_type == MonitoringEventType.GPU_UNAVAILABLE]
    assert len(unavailable_events) == 1
    assert unavailable_events[0].data['gpu_id'] == 0


@pytest.mark.asyncio
async def test_gpu_temperature_event(mock_node):
    """Test GPU temperature warning"""
    events = []
    
    def event_callback(event):
        events.append(event)
    
    monitor = ResourceMonitor(mock_node, event_callback=event_callback)
    await monitor.start()
    
    # Initial GPU state (normal temp)
    gpu1 = GPUInfo(
        gpu_id=0,
        name='Test GPU',
        memory_total_mb=8000,
        memory_used_mb=2000,
        memory_free_mb=6000,
        utilization_percent=25.0,
        temperature_celsius=70.0,
        is_available=True
    )
    
    await monitor._detect_gpu_changes([gpu1])
    
    # Increase temperature above threshold
    gpu2 = GPUInfo(
        gpu_id=0,
        name='Test GPU',
        memory_total_mb=8000,
        memory_used_mb=2000,
        memory_free_mb=6000,
        utilization_percent=25.0,
        temperature_celsius=90.0,  # Over 85°C threshold
        is_available=True
    )
    
    await monitor._detect_gpu_changes([gpu2])
    
    # Should have emitted overheating event
    assert len(events) == 1
    assert events[0].event_type == MonitoringEventType.GPU_OVERHEATED
    assert events[0].data['temperature'] == 90.0


@pytest.mark.asyncio
async def test_memory_pressure_event(mock_node):
    """Test memory pressure detection"""
    events = []
    
    def event_callback(event):
        events.append(event)
    
    monitor = ResourceMonitor(mock_node, event_callback=event_callback)
    await monitor.start()
    
    # Initial GPU state (normal memory)
    gpu1 = GPUInfo(
        gpu_id=0,
        name='Test GPU',
        memory_total_mb=10000,
        memory_used_mb=5000,  # 50%
        memory_free_mb=5000,
        utilization_percent=50.0,
        is_available=True
    )
    
    await monitor._detect_gpu_changes([gpu1])
    
    # Increase memory usage above threshold
    gpu2 = GPUInfo(
        gpu_id=0,
        name='Test GPU',
        memory_total_mb=10000,
        memory_used_mb=9500,  # 95% - over 90% threshold
        memory_free_mb=500,
        utilization_percent=95.0,
        is_available=True
    )
    
    await monitor._detect_gpu_changes([gpu2])
    
    # Should have emitted memory pressure event
    assert len(events) == 1
    assert events[0].event_type == MonitoringEventType.MEMORY_PRESSURE
    assert events[0].data['memory_usage_percent'] > 90.0


@pytest.mark.asyncio
async def test_power_limit_event(mock_node):
    """Test power consumption warning"""
    events = []
    
    def event_callback(event):
        events.append(event)
    
    monitor = ResourceMonitor(mock_node, event_callback=event_callback)
    await monitor.start()
    
    # Initial GPU state (normal power)
    gpu1 = GPUInfo(
        gpu_id=0,
        name='Test GPU',
        memory_total_mb=8000,
        memory_used_mb=2000,
        memory_free_mb=6000,
        utilization_percent=25.0,
        power_draw_watts=150.0,
        is_available=True
    )
    
    await monitor._detect_gpu_changes([gpu1])
    
    # Increase power above threshold
    gpu2 = GPUInfo(
        gpu_id=0,
        name='Test GPU',
        memory_total_mb=8000,
        memory_used_mb=2000,
        memory_free_mb=6000,
        utilization_percent=25.0,
        power_draw_watts=350.0,  # Over 300W threshold
        is_available=True
    )
    
    await monitor._detect_gpu_changes([gpu2])
    
    # Should have emitted power limit event
    assert len(events) == 1
    assert events[0].event_type == MonitoringEventType.GPU_POWER_LIMIT
    assert events[0].data['power_usage'] == 350.0


@pytest.mark.asyncio
async def test_metrics_recording(mock_node):
    """Test metrics history recording"""
    monitor = ResourceMonitor(mock_node, poll_interval=0.1)
    await monitor.start()
    
    gpus = [
        GPUInfo(
            gpu_id=0,
            name='Test GPU',
            memory_total_mb=8000,
            memory_used_mb=2000,
            memory_free_mb=6000,
            utilization_percent=25.0,
            is_available=True
        )
    ]
    
    # Record metrics
    await monitor._record_metrics(gpus)
    await monitor._record_metrics(gpus)
    await monitor._record_metrics(gpus)
    
    # Should have history
    assert len(monitor.metrics_history) == 3
    
    recent = monitor.get_recent_metrics(count=2)
    assert len(recent) == 2


def test_monitoring_statistics(mock_node):
    """Test monitoring statistics"""
    monitor = ResourceMonitor(mock_node)
    
    stats = monitor.get_statistics()
    
    assert 'running' in stats
    assert 'poll_interval' in stats
    assert 'events_emitted' in stats
    assert 'events_by_type' in stats
    assert stats['events_emitted'] == 0


def test_monitoring_event_to_dict():
    """Test MonitoringEvent serialization"""
    event = MonitoringEvent(
        event_type=MonitoringEventType.GPU_AVAILABLE,
        node_id='test-node',
        timestamp=123.456,
        data={'gpu_id': 0},
        message='GPU available'
    )
    
    d = event.to_dict()
    
    assert d['event_type'] == 'gpu_available'
    assert d['node_id'] == 'test-node'
    assert d['timestamp'] == 123.456
    assert d['data']['gpu_id'] == 0
    assert d['message'] == 'GPU available'


@pytest.mark.asyncio
async def test_monitor_loop_cancellation(mock_node):
    """Test monitor loop handles cancellation"""
    with patch('genie.cluster.monitoring._detect_local_gpus', return_value=[]):
        monitor = ResourceMonitor(mock_node, poll_interval=0.01)
        await monitor.start()
        
        # Start monitor loop
        task = asyncio.create_task(monitor.monitor_loop())
        
        # Let it run briefly
        await asyncio.sleep(0.05)
        
        # Cancel it
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Should have stopped cleanly
        assert True  # No exception raised


@pytest.mark.asyncio
async def test_event_callback_error_handling(mock_node):
    """Test error handling in event callback"""
    def bad_callback(event):
        raise RuntimeError("Callback error")
    
    monitor = ResourceMonitor(mock_node, event_callback=bad_callback)
    await monitor.start()
    
    # This should not raise even though callback errors
    gpu = GPUInfo(
        gpu_id=0,
        name='Test GPU',
        memory_total_mb=8000,
        memory_used_mb=2000,
        memory_free_mb=6000,
        utilization_percent=25.0,
        temperature_celsius=90.0,  # Will trigger event
        is_available=True
    )
    
    # First pass to initialize
    await monitor._detect_gpu_changes([gpu])
    
    # Second pass with normal temp to trigger event
    gpu2 = GPUInfo(
        gpu_id=0,
        name='Test GPU',
        memory_total_mb=8000,
        memory_used_mb=2000,
        memory_free_mb=6000,
        utilization_percent=25.0,
        temperature_celsius=70.0,
        is_available=True
    )
    
    # Set up initial state with high temp
    monitor.prev_gpu_states[0] = {'temperature_celsius': 80.0, 'health_status': 'healthy', 
                                   'utilization_percent': 25.0, 'power_draw_watts': None,
                                   'memory_used_mb': 2000, 'memory_total_mb': 8000}
    
    gpu_hot = GPUInfo(
        gpu_id=0,
        name='Test GPU',
        memory_total_mb=8000,
        memory_used_mb=2000,
        memory_free_mb=6000,
        utilization_percent=25.0,
        temperature_celsius=90.0,
        is_available=True
    )
    
    # Should not raise despite callback error
    await monitor._detect_gpu_changes([gpu_hot])
    
    # Monitor should still be operational
    assert monitor.running


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

