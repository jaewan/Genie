"""
Djinn Cluster Management

Public API for cluster initialization and management.
"""

from .node_info import (
    NodeInfo,
    NodeStatus,
    NodeRole,
    GPUInfo,
    create_local_node_info
)

from .init import (
    init,
    shutdown,
    ClusterState,
    ClusterConfig,
    get_cluster_state,
    is_initialized,
)

__all__ = [
    # Node info
    'NodeInfo',
    'NodeStatus',
    'NodeRole',
    'GPUInfo',
    'create_local_node_info',
    
    # Initialization
    'init',
    'shutdown',
    'ClusterState',
    'ClusterConfig',
    'get_cluster_state',
    'is_initialized',
]

__version__ = '0.1.0'

