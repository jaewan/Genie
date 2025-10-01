"""Node information data structures

Defines comprehensive node and GPU information for cluster management.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
import time


class NodeStatus(Enum):
    """Node operational status"""
    UNKNOWN = "unknown"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DISCONNECTED = "disconnected"


class NodeRole(Enum):
    """Node role in cluster"""
    CLIENT = "client"      # Only consumes remote GPUs
    SERVER = "server"      # Only provides GPUs
    WORKER = "worker"      # Both provides and consumes
    MASTER = "master"      # Cluster coordinator


@dataclass
class GPUInfo:
    """GPU device information"""
    gpu_id: int
    name: str                      # e.g., "NVIDIA A100-SXM4-40GB"
    memory_total_mb: float
    memory_used_mb: float
    memory_free_mb: float
    utilization_percent: float     # 0-100
    temperature_celsius: Optional[float] = None
    power_draw_watts: Optional[float] = None
    
    # Compute capabilities
    compute_capability: Optional[str] = None  # e.g., "8.0"
    cuda_cores: Optional[int] = None
    
    # Status
    is_available: bool = True
    error_state: Optional[str] = None
    
    @property
    def memory_utilization_percent(self) -> float:
        """Calculate memory utilization_percent percentage"""
        if self.memory_total_mb > 0:
            return (self.memory_used_mb / self.memory_total_mb) * 100
        return 0.0
    
    @property
    def health_status(self) -> str:
        """Get health status string for compatibility"""
        if self.error_state:
            return 'unhealthy'
        elif self.is_available:
            return 'healthy'
        else:
            return 'unavailable'
    
    @property
    def power_usage_watts(self) -> Optional[float]:
        """Alias for power_draw_watts for compatibility"""
        return self.power_draw_watts


@dataclass
class NodeInfo:
    """Comprehensive node information"""
    # Identity
    node_id: str
    hostname: str
    role: NodeRole
    
    # Network endpoints
    host: str
    control_port: int
    data_port: int
    
    # Network capabilities
    network_backend: str  # 'tcp', 'dpdk', 'dpdk_gpudev', 'rdma'
    available_backends: List[str] = field(default_factory=list)
    network_bandwidth_gbps: float = 10.0
    
    # GPU resources
    gpu_count: int = 0
    gpus: List[GPUInfo] = field(default_factory=list)
    
    # Status and health
    status: NodeStatus = NodeStatus.UNKNOWN
    last_heartbeat: float = field(default_factory=time.time)
    last_health_check: float = field(default_factory=time.time)
    uptime_seconds: float = 0.0
    
    # Transfer statistics
    active_transfers: int = 0
    completed_transfers: int = 0
    failed_transfers: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_from_heartbeat(self, heartbeat_data: Dict[str, Any]):
        """Update node info from heartbeat message"""
        self.last_heartbeat = time.time()
        
        # Update status
        if 'status' in heartbeat_data:
            if isinstance(heartbeat_data['status'], str):
                self.status = NodeStatus(heartbeat_data['status'])
            else:
                self.status = heartbeat_data['status']
        
        # Update GPU info if provided
        if 'gpus' in heartbeat_data:
            self.gpus = [
                GPUInfo(**gpu_data) 
                for gpu_data in heartbeat_data['gpus']
            ]
            self.gpu_count = len(self.gpus)
        
        # Update statistics
        if 'active_transfers' in heartbeat_data:
            self.active_transfers = heartbeat_data['active_transfers']
    
    def is_healthy(self, timeout: float = 60.0) -> bool:
        """Check if node is healthy based on heartbeat"""
        time_since_heartbeat = time.time() - self.last_heartbeat
        return (
            time_since_heartbeat < timeout and
            self.status not in [NodeStatus.UNHEALTHY, NodeStatus.DISCONNECTED]
        )
    
    def get_available_gpus(self) -> List[GPUInfo]:
        """Get list of available (not busy) GPUs"""
        return [gpu for gpu in self.gpus if gpu.is_available]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'node_id': self.node_id,
            'hostname': self.hostname,
            'role': self.role.value,
            'host': self.host,
            'control_port': self.control_port,
            'data_port': self.data_port,
            'network_backend': self.network_backend,
            'available_backends': self.available_backends,
            'gpu_count': self.gpu_count,
            'gpus': [
                {
                    'gpu_id': g.gpu_id,
                    'name': g.name,
                    'memory_total_mb': g.memory_total_mb,
                    'memory_used_mb': g.memory_used_mb,
                    'utilization_percent': g.utilization_percent,
                    'is_available': g.is_available
                }
                for g in self.gpus
            ],
            'status': self.status.value,
            'last_heartbeat': self.last_heartbeat,
            'active_transfers': self.active_transfers,
        }


def create_local_node_info(
    node_id: str,
    role: NodeRole = NodeRole.CLIENT,
    control_port: int = 5555,
    data_port: int = 5556
) -> NodeInfo:
    """
    Create NodeInfo for local node.
    
    Detects local GPUs and network configuration.
    """
    import socket
    
    hostname = socket.gethostname()
    try:
        host = socket.gethostbyname(hostname)
    except:
        host = '127.0.0.1'
    
    # Detect local GPUs
    gpus = _detect_local_gpus()
    
    return NodeInfo(
        node_id=node_id,
        hostname=hostname,
        role=role,
        host=host,
        control_port=control_port,
        data_port=data_port,
        network_backend='tcp',  # Default, will be updated
        gpu_count=len(gpus),
        gpus=gpus,
        status=NodeStatus.INITIALIZING
    )


def _detect_local_gpus() -> List[GPUInfo]:
    """Detect GPUs on local machine"""
    gpus = []
    
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                
                # Get memory info
                mem_total = props.total_memory / (1024**2)  # Convert to MB
                mem_reserved = torch.cuda.memory_reserved(i) / (1024**2)
                mem_allocated = torch.cuda.memory_allocated(i) / (1024**2)
                mem_free = mem_total - mem_reserved
                
                # Get utilization (requires nvidia-smi)
                utilization = _get_gpu_utilization(i)
                
                gpu = GPUInfo(
                    gpu_id=i,
                    name=props.name,
                    memory_total_mb=mem_total,
                    memory_used_mb=mem_allocated,
                    memory_free_mb=mem_free,
                    utilization_percent=utilization,
                    compute_capability=f"{props.major}.{props.minor}",
                    cuda_cores=props.multi_processor_count * 128,  # Approximate
                    is_available=True
                )
                gpus.append(gpu)
    except ImportError:
        pass  # PyTorch not available
    except Exception as e:
        import logging
        logging.warning(f"Failed to detect GPUs: {e}")
    
    return gpus


def _get_gpu_utilization(gpu_id: int) -> float:
    """Get GPU utilization percentage using nvidia-smi"""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', 
             '--format=csv,noheader,nounits', f'--id={gpu_id}'],
            capture_output=True,
            text=True,
            timeout=2.0
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except:
        pass
    
    return 0.0  # Unknown

