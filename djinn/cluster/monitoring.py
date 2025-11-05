"""
Enhanced Resource Monitoring for Djinn Cluster

Provides comprehensive monitoring of cluster resources including:
- GPU status tracking with event notifications
- Memory usage monitoring
- Temperature and power monitoring
- Change detection and alerts
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from .node_info import NodeInfo, NodeStatus, GPUInfo, _detect_local_gpus

logger = logging.getLogger(__name__)


class MonitoringEventType(Enum):
    """Types of monitoring events"""
    GPU_AVAILABLE = "gpu_available"
    GPU_UNAVAILABLE = "gpu_unavailable"
    GPU_OVERHEATED = "gpu_overheated"
    GPU_POWER_LIMIT = "gpu_power_limit"
    MEMORY_PRESSURE = "memory_pressure"
    NODE_DEGRADED = "node_degraded"
    NODE_RECOVERED = "node_recovered"


@dataclass
class MonitoringEvent:
    """Event emitted by monitoring system"""
    event_type: MonitoringEventType
    node_id: str
    timestamp: float
    data: Dict
    message: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'event_type': self.event_type.value,
            'node_id': self.node_id,
            'timestamp': self.timestamp,
            'data': self.data,
            'message': self.message
        }


class ResourceMonitor:
    """
    Monitors cluster node resources.
    
    Features:
    - Tracks GPU status changes
    - Detects temperature issues
    - Monitors power consumption
    - Emits events on state changes
    - Provides metrics history
    """
    
    def __init__(
        self,
        local_node: NodeInfo,
        poll_interval: float = 5.0,
        event_callback: Optional[Callable[[MonitoringEvent], None]] = None
    ):
        """
        Initialize resource monitor.
        
        Args:
            local_node: Local node to monitor
            poll_interval: Seconds between polls
            event_callback: Optional callback for events
        """
        self.local_node = local_node
        self.poll_interval = poll_interval
        self.event_callback = event_callback
        self.running = False
        
        # Track previous state for change detection
        self.prev_gpu_states: Dict[int, Dict] = {}
        
        # Metrics history (last N samples)
        self.metrics_history: List[Dict] = []
        self.max_history_size = 100
        
        # Event statistics
        self.events_emitted = 0
        self.events_by_type: Dict[str, int] = {}
    
    async def start(self):
        """Start monitoring"""
        self.running = True
        logger.info(f"Resource monitor started (poll_interval={self.poll_interval}s)")
    
    async def stop(self):
        """Stop monitoring"""
        self.running = False
        logger.info("Resource monitor stopped")
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect GPU metrics
                new_gpus = _detect_local_gpus()
                
                # Detect changes and emit events
                await self._detect_gpu_changes(new_gpus)
                
                # Update local node
                self.local_node.gpus = new_gpus
                
                # Record metrics
                await self._record_metrics(new_gpus)
                
                # Wait for next poll
                await asyncio.sleep(self.poll_interval)
                
            except asyncio.CancelledError:
                logger.debug("Monitor loop cancelled")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)
    
    async def _detect_gpu_changes(self, new_gpus: List[GPUInfo]):
        """Detect GPU state changes and emit events"""
        for gpu in new_gpus:
            prev_state = self.prev_gpu_states.get(gpu.gpu_id)
            
            if prev_state is None:
                # First time seeing this GPU - initialize state
                self.prev_gpu_states[gpu.gpu_id] = self._gpu_to_dict(gpu)
                logger.info(f"GPU {gpu.gpu_id} ({gpu.name}) initialized")
                continue
            
            # Check availability change
            await self._check_availability_change(gpu, prev_state)
            
            # Check temperature warnings
            await self._check_temperature(gpu, prev_state)
            
            # Check power consumption
            await self._check_power(gpu, prev_state)
            
            # Check memory pressure
            await self._check_memory_pressure(gpu, prev_state)
            
            # Update previous state
            self.prev_gpu_states[gpu.gpu_id] = self._gpu_to_dict(gpu)
    
    async def _check_availability_change(self, gpu: GPUInfo, prev_state: Dict):
        """Check for GPU availability changes"""
        was_available = prev_state.get('health_status') == 'healthy'
        is_available = gpu.health_status == 'healthy'
        
        if was_available and not is_available:
            await self._emit_event(MonitoringEvent(
                event_type=MonitoringEventType.GPU_UNAVAILABLE,
                node_id=self.local_node.node_id,
                timestamp=time.time(),
                data={
                    'gpu_id': gpu.gpu_id,
                    'gpu_name': gpu.name,
                    'utilization': gpu.utilization_percent,
                    'memory_used': gpu.memory_used_mb,
                    'memory_total': gpu.memory_total_mb
                },
                message=f"GPU {gpu.gpu_id} became unavailable (util: {gpu.utilization_percent:.1f}%)"
            ))
        elif not was_available and is_available:
            await self._emit_event(MonitoringEvent(
                event_type=MonitoringEventType.GPU_AVAILABLE,
                node_id=self.local_node.node_id,
                timestamp=time.time(),
                data={
                    'gpu_id': gpu.gpu_id,
                    'gpu_name': gpu.name
                },
                message=f"GPU {gpu.gpu_id} became available"
            ))
    
    async def _check_temperature(self, gpu: GPUInfo, prev_state: Dict):
        """Check for temperature warnings"""
        if gpu.temperature_celsius is None:
            return
        
        prev_temp = prev_state.get('temperature_celsius', 0)
        current_temp = gpu.temperature_celsius
        
        # Overheating threshold (85°C)
        if current_temp > 85.0 and prev_temp <= 85.0:
            await self._emit_event(MonitoringEvent(
                event_type=MonitoringEventType.GPU_OVERHEATED,
                node_id=self.local_node.node_id,
                timestamp=time.time(),
                data={
                    'gpu_id': gpu.gpu_id,
                    'gpu_name': gpu.name,
                    'temperature': current_temp,
                    'threshold': 85.0
                },
                message=f"GPU {gpu.gpu_id} overheating: {current_temp:.1f}°C (threshold: 85°C)"
            ))
            logger.warning(f"GPU {gpu.gpu_id} temperature: {current_temp:.1f}°C")
    
    async def _check_power(self, gpu: GPUInfo, prev_state: Dict):
        """Check for power consumption warnings"""
        if gpu.power_draw_watts is None:
            return
        
        # Power limit threshold (e.g., 300W for typical GPU)
        power_limit = 300.0  # Could be made configurable
        prev_power = prev_state.get('power_draw_watts', 0)
        current_power = gpu.power_draw_watts
        
        if current_power > power_limit and prev_power <= power_limit:
            await self._emit_event(MonitoringEvent(
                event_type=MonitoringEventType.GPU_POWER_LIMIT,
                node_id=self.local_node.node_id,
                timestamp=time.time(),
                data={
                    'gpu_id': gpu.gpu_id,
                    'gpu_name': gpu.name,
                    'power_usage': current_power,
                    'power_limit': power_limit
                },
                message=f"GPU {gpu.gpu_id} power usage: {current_power:.1f}W (limit: {power_limit}W)"
            ))
    
    async def _check_memory_pressure(self, gpu: GPUInfo, prev_state: Dict):
        """Check for memory pressure"""
        if gpu.memory_total_mb == 0:
            return
        
        memory_usage_percent = (gpu.memory_used_mb / gpu.memory_total_mb) * 100
        prev_usage_percent = (prev_state.get('memory_used_mb', 0) / 
                             max(prev_state.get('memory_total_mb', 1), 1)) * 100
        
        # Memory pressure threshold (90%)
        if memory_usage_percent > 90.0 and prev_usage_percent <= 90.0:
            await self._emit_event(MonitoringEvent(
                event_type=MonitoringEventType.MEMORY_PRESSURE,
                node_id=self.local_node.node_id,
                timestamp=time.time(),
                data={
                    'gpu_id': gpu.gpu_id,
                    'gpu_name': gpu.name,
                    'memory_used_mb': gpu.memory_used_mb,
                    'memory_total_mb': gpu.memory_total_mb,
                    'memory_usage_percent': memory_usage_percent
                },
                message=f"GPU {gpu.gpu_id} memory pressure: {memory_usage_percent:.1f}%"
            ))
    
    async def _emit_event(self, event: MonitoringEvent):
        """Emit monitoring event"""
        self.events_emitted += 1
        event_type_str = event.event_type.value
        self.events_by_type[event_type_str] = self.events_by_type.get(event_type_str, 0) + 1
        
        # Log event
        logger.info(f"Event: {event.message}")
        
        # Call callback if provided
        if self.event_callback:
            try:
                if asyncio.iscoroutinefunction(self.event_callback):
                    await self.event_callback(event)
                else:
                    self.event_callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}", exc_info=True)
    
    async def _record_metrics(self, gpus: List[GPUInfo]):
        """Record metrics for history"""
        metrics = {
            'timestamp': time.time(),
            'gpu_count': len(gpus),
            'gpus': [
                {
                    'gpu_id': gpu.gpu_id,
                    'utilization': gpu.utilization_percent,
                    'memory_used_mb': gpu.memory_used_mb,
                    'memory_total_mb': gpu.memory_total_mb,
                    'temperature': gpu.temperature_celsius,
                    'power_watts': gpu.power_draw_watts,
                    'health_status': gpu.health_status
                }
                for gpu in gpus
            ]
        }
        
        self.metrics_history.append(metrics)
        
        # Trim history if too large
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    def _gpu_to_dict(self, gpu: GPUInfo) -> Dict:
        """Convert GPUInfo to dict for comparison"""
        return {
            'health_status': gpu.health_status,
            'utilization_percent': gpu.utilization_percent,
            'temperature_celsius': gpu.temperature_celsius,
            'power_draw_watts': gpu.power_draw_watts,
            'memory_used_mb': gpu.memory_used_mb,
            'memory_total_mb': gpu.memory_total_mb
        }
    
    def get_statistics(self) -> Dict:
        """Get monitoring statistics"""
        return {
            'running': self.running,
            'poll_interval': self.poll_interval,
            'events_emitted': self.events_emitted,
            'events_by_type': self.events_by_type.copy(),
            'metrics_history_size': len(self.metrics_history),
            'gpus_monitored': len(self.prev_gpu_states)
        }
    
    def get_recent_metrics(self, count: int = 10) -> List[Dict]:
        """Get recent metrics"""
        return self.metrics_history[-count:] if self.metrics_history else []


def create_resource_monitor(
    local_node: NodeInfo,
    poll_interval: float = 5.0,
    event_callback: Optional[Callable[[MonitoringEvent], None]] = None
) -> ResourceMonitor:
    """
    Create a resource monitor instance.
    
    Args:
        local_node: Local node to monitor
        poll_interval: Seconds between polls (default: 5.0)
        event_callback: Optional callback for events
    
    Returns:
        ResourceMonitor instance
    """
    return ResourceMonitor(local_node, poll_interval, event_callback)

