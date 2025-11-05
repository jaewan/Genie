"""
Djinn Cluster Initialization

Main entry point for connecting to Djinn cluster.
"""

import asyncio
import logging
import os
import uuid
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from .node_info import NodeInfo, NodeStatus, NodeRole, create_local_node_info

logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    """
    Configuration for cluster initialization.
    
    Attributes:
        discovery_method: How to find cluster nodes
            - 'static': Use master_addr/master_port
            - 'env': Read from environment variables
            - 'auto': Try env first, then static
        
        master_addr: Address of cluster master node
        master_port: Port of cluster master
        
        node_id: Unique identifier for this node (auto-generated if None)
        node_role: Role of this node in cluster
        
        backend: Network backend preference
            - 'auto': Discover and select best available
            - 'tcp': Force TCP (always available)
            - 'dpdk': Force DPDK zero-copy
            - 'dpdk_gpudev': Force DPDK with GPUDirect
            - 'rdma': Force RDMA
        
        backend_options: Additional options for backend
        
        enable_heartbeat: Enable periodic heartbeat messages
        heartbeat_interval: Seconds between heartbeats
        heartbeat_timeout: Seconds before declaring node dead
        
        enable_gpu_monitoring: Enable GPU status monitoring
        gpu_poll_interval: Seconds between GPU polls
        
        enable_health_checks: Enable comprehensive health checks
        health_check_interval: Seconds between health checks
        
        timeout: Total initialization timeout in seconds
    """
    # Discovery
    discovery_method: str = "auto"
    master_addr: Optional[str] = None
    master_port: int = 5555
    
    # Node identity
    node_id: Optional[str] = None
    node_role: str = "client"  # client, server, worker, master
    
    # Network backend
    backend: str = "auto"
    backend_options: Dict[str, Any] = field(default_factory=dict)
    
    # Monitoring
    enable_heartbeat: bool = True
    heartbeat_interval: float = 10.0
    heartbeat_timeout: float = 60.0
    
    # GPU monitoring
    enable_gpu_monitoring: bool = True
    gpu_poll_interval: float = 5.0
    
    # Health checks
    enable_health_checks: bool = True
    health_check_interval: float = 30.0
    
    # Timeouts
    timeout: float = 30.0
    
    def validate(self):
        """Validate configuration"""
        if self.discovery_method not in ['static', 'env', 'auto']:
            raise ValueError(f"Invalid discovery_method: {self.discovery_method}")
        
        if self.discovery_method == 'static' and not self.master_addr:
            raise ValueError("master_addr required for static discovery")
        
        if self.node_role not in ['client', 'server', 'worker', 'master']:
            raise ValueError(f"Invalid node_role: {self.node_role}")
        
        if self.backend not in ['auto', 'tcp', 'dpdk', 'dpdk_gpudev', 'rdma']:
            raise ValueError(f"Invalid backend: {self.backend}")


class ClusterState:
    """
    Global cluster state (singleton).
    
    Manages the state of the Djinn cluster including:
    - Connection to master/peers
    - Discovered nodes
    - Transport coordinator
    - Monitoring tasks
    """
    
    _instance: Optional['ClusterState'] = None
    
    def __init__(self):
        self.initialized = False
        self.config: Optional[ClusterConfig] = None
        
        # Local node info
        self.local_node: Optional[NodeInfo] = None
        
        # Discovered nodes (node_id -> NodeInfo)
        self.nodes: Dict[str, NodeInfo] = {}
        
        # Components
        self.transport_coordinator = None
        self.control_integration = None
        
        # Monitoring
        self.monitor_tasks: List[asyncio.Task] = []
        
        # Statistics
        self.stats = {
            'init_time': 0.0,
            'uptime': 0.0,
            'total_nodes_discovered': 0,
            'total_transfers': 0,
        }
    
    @classmethod
    def get(cls) -> 'ClusterState':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_singleton(cls):
        """Reset singleton (for testing)"""
        cls._instance = None
    
    def reset(self):
        """Reset state (for testing)"""
        # Cancel monitoring tasks
        for task in self.monitor_tasks:
            if not task.done():
                task.cancel()
        
        self.initialized = False
        self.config = None
        self.local_node = None
        self.nodes.clear()
        self.transport_coordinator = None
        self.control_integration = None
        self.monitor_tasks.clear()
        self.stats = {
            'init_time': 0.0,
            'uptime': 0.0,
            'total_nodes_discovered': 0,
            'total_transfers': 0,
        }
    
    def get_node(self, node_id: str) -> Optional[NodeInfo]:
        """Get node by ID"""
        return self.nodes.get(node_id)
    
    def add_node(self, node: NodeInfo):
        """Add discovered node"""
        self.nodes[node.node_id] = node
        self.stats['total_nodes_discovered'] += 1
        logger.info(f"Discovered node: {node.node_id} ({node.hostname})")
    
    def remove_node(self, node_id: str):
        """Remove node"""
        if node_id in self.nodes:
            node = self.nodes.pop(node_id)
            logger.info(f"Removed node: {node_id} ({node.hostname})")
    
    def get_healthy_nodes(self, timeout: float = 60.0) -> List[NodeInfo]:
        """Get list of healthy nodes"""
        return [
            node for node in self.nodes.values()
            if node.is_healthy(timeout)
        ]


# Global state accessor
def get_cluster_state() -> ClusterState:
    """Get global cluster state"""
    return ClusterState.get()


def is_initialized() -> bool:
    """Check if cluster is initialized"""
    return ClusterState.get().initialized


async def init(
    master_addr: Optional[str] = None,
    master_port: int = 5555,
    backend: str = "auto",
    node_id: Optional[str] = None,
    node_role: str = "client",
    timeout: float = 30.0,
    **kwargs
) -> ClusterState:
    """
    Initialize Djinn cluster connection.
    
    This is the main entry point for connecting to remote accelerators.
    Call this once at the beginning of your program.
    
    Args:
        master_addr: Address of cluster master/server node.
                     Can be None if GENIE_MASTER_ADDR env var is set.
        master_port: Port of cluster master (default: 5555)
        backend: Network backend - 'auto', 'tcp', 'dpdk', 'rdma'
                 'auto' will discover best available
        node_id: Unique identifier for this node (auto-generated if None)
        node_role: Role - 'client' (default), 'server', 'worker', 'master'
        timeout: Initialization timeout in seconds
        **kwargs: Additional ClusterConfig options
    
    Returns:
        ClusterState object
    
    Raises:
        ValueError: Invalid configuration
        ConnectionError: Cannot connect to cluster
        TimeoutError: Initialization timeout
    
    Example:
        Basic usage:
        ```python
        import djinn
        import torch
        
        # Initialize once
        await genie.init(master_addr='gpu-server.example.com')
        
        # Use remote accelerators
        x = torch.randn(1000, 1000, device='remote_accelerator:0')
        ```
        
        Advanced usage:
        ```python
        await genie.init(
            master_addr='192.168.1.100',
            backend='dpdk_gpudev',  # Force zero-copy
            enable_gpu_monitoring=True,
            heartbeat_interval=5.0
        )
        ```
        
        Server mode:
        ```python
        # Run as server
        await genie.init(node_role='server', master_port=5555)
        # Keep running
        await asyncio.Event().wait()
        ```
    
    Environment Variables:
        GENIE_MASTER_ADDR: Master node address
        GENIE_MASTER_PORT: Master node port (default: 5555)
        GENIE_NODE_ID: Node identifier
        GENIE_NODE_ROLE: Node role (client/server/worker)
        GENIE_BACKEND: Preferred backend (auto/tcp/dpdk/rdma)
    """
    start_time = time.time()
    
    state = ClusterState.get()
    
    # Check if already initialized
    if state.initialized:
        logger.warning("Djinn already initialized, returning existing state")
        return state
    
    # Build configuration
    config = _build_config(
        master_addr=master_addr,
        master_port=master_port,
        backend=backend,
        node_id=node_id,
        node_role=node_role,
        timeout=timeout,
        **kwargs
    )
    
    # Validate
    config.validate()
    
    logger.info("=" * 60)
    logger.info("Initializing Djinn Cluster")
    logger.info("=" * 60)
    logger.info(f"Master: {config.master_addr}:{config.master_port}")
    logger.info(f"Role: {config.node_role}")
    logger.info(f"Backend: {config.backend}")
    
    try:
        # Run initialization phases with timeout
        await asyncio.wait_for(
            _run_initialization_phases(state, config),
            timeout=config.timeout
        )
        
        # Success!
        state.config = config
        state.initialized = True
        state.stats['init_time'] = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("✅ Djinn Initialized Successfully")
        logger.info("=" * 60)
        logger.info(f"Node ID: {state.local_node.node_id}")
        logger.info(f"Backend: {state.local_node.network_backend}")
        logger.info(f"GPUs: {len(state.local_node.gpus)}")
        logger.info(f"Peers: {len(state.nodes)}")
        logger.info(f"Time: {state.stats['init_time']:.2f}s")
        
        return state
        
    except asyncio.TimeoutError:
        logger.error(f"❌ Initialization timed out after {config.timeout}s")
        state.reset()
        raise
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        state.reset()
        raise


async def shutdown():
    """
    Shutdown Djinn cluster connection.
    
    Cleanly shuts down all monitoring tasks, closes connections,
    and resets cluster state.
    """
    state = ClusterState.get()
    
    if not state.initialized:
        logger.warning("Djinn not initialized, nothing to shutdown")
        return
    
    logger.info("Shutting down Djinn cluster connection...")
    
    # Cancel all monitoring tasks
    for task in state.monitor_tasks:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    # Shutdown transport coordinator
    if state.transport_coordinator:
        try:
            await state.transport_coordinator.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down transport coordinator: {e}")
    
    # Shutdown control integration
    if state.control_integration:
        try:
            await state.control_integration.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down control integration: {e}")
    
    # Reset state
    state.reset()
    
    logger.info("✅ Djinn shutdown complete")


def _build_config(**kwargs) -> ClusterConfig:
    """Build configuration from arguments and environment"""
    # Start with defaults
    config_dict = {}
    
    # Override with environment variables
    if os.getenv('GENIE_MASTER_ADDR'):
        config_dict['master_addr'] = os.getenv('GENIE_MASTER_ADDR')
    if os.getenv('GENIE_MASTER_PORT'):
        config_dict['master_port'] = int(os.getenv('GENIE_MASTER_PORT'))
    if os.getenv('GENIE_NODE_ID'):
        config_dict['node_id'] = os.getenv('GENIE_NODE_ID')
    if os.getenv('GENIE_NODE_ROLE'):
        config_dict['node_role'] = os.getenv('GENIE_NODE_ROLE')
    if os.getenv('GENIE_BACKEND'):
        config_dict['backend'] = os.getenv('GENIE_BACKEND')
    
    # Override with explicit arguments
    config_dict.update({k: v for k, v in kwargs.items() if v is not None})
    
    return ClusterConfig(**config_dict)


async def _run_initialization_phases(state: ClusterState, config: ClusterConfig):
    """Run all initialization phases"""
    
    # Phase 1: Create local node info
    logger.info("Phase 1/5: Creating local node info...")
    _create_local_node(state, config)
    
    # Phase 2: Discover network capabilities
    logger.info("Phase 2/5: Discovering network infrastructure...")
    network_info = await _discover_network(state, config)
    
    # Phase 3: Select and initialize backend
    logger.info("Phase 3/5: Initializing network backend...")
    await _initialize_backend(state, config, network_info)
    
    # Phase 4: Connect to cluster
    if config.node_role in ['client', 'worker']:
        logger.info("Phase 4/5: Connecting to cluster...")
        await _connect_to_cluster(state, config)
    else:
        logger.info("Phase 4/5: Skipping cluster connection (server mode)")
    
    # Phase 5: Start monitoring
    logger.info("Phase 5/5: Starting monitoring services...")
    await _start_monitoring(state, config)


def _create_local_node(state: ClusterState, config: ClusterConfig):
    """Create local node information"""
    # Generate node ID if not provided
    if not config.node_id:
        import socket
        hostname = socket.gethostname()
        short_id = str(uuid.uuid4())[:8]
        config.node_id = f"{hostname}-{short_id}"
    
    # Convert role string to enum
    role_map = {
        'client': NodeRole.CLIENT,
        'server': NodeRole.SERVER,
        'worker': NodeRole.WORKER,
        'master': NodeRole.SERVER,  # Master is a type of server
    }
    role = role_map.get(config.node_role, NodeRole.CLIENT)
    
    # Create node info
    state.local_node = create_local_node_info(
        node_id=config.node_id,
        role=role,
        control_port=config.master_port if config.node_role in ['server', 'master'] else 0,
        data_port=config.master_port + 1 if config.node_role in ['server', 'master'] else 0
    )
    
    logger.info(f"  Node ID: {state.local_node.node_id}")
    logger.info(f"  Hostname: {state.local_node.hostname}")
    logger.info(f"  Role: {state.local_node.role.value}")
    logger.info(f"  GPUs: {len(state.local_node.gpus)}")


async def _discover_network(
    state: ClusterState,
    config: ClusterConfig
) -> Dict[str, Any]:
    """Discover network capabilities"""
    
    if config.node_role in ['server', 'master']:
        # Server: test local capabilities (no remote connection needed)
        try:
            from ..runtime.network_discovery import discover_network_capabilities
            
            # Test local backends (no target needed for server)
            network_info = await discover_network_capabilities(
                target_addr='localhost',  # Not used for server
                target_port=config.master_port,
                timeout=2.0,
                test_backends=['dpdk', 'dpdk_gpudev', 'rdma']  # Skip TCP test for server
            )
            
            # Always include TCP for server
            if 'tcp' not in network_info['available_backends']:
                network_info['available_backends'].insert(0, 'tcp')
            
            logger.info(f"  Server capabilities: {network_info['available_backends']}")
            return network_info
            
        except ImportError:
            logger.info("  Network discovery module not available, using TCP")
            return {
                'available_backends': ['tcp'],
                'recommended_backend': 'tcp'
            }
        except Exception as e:
            logger.warning(f"  Network discovery failed: {e}")
            return {
                'available_backends': ['tcp'],
                'recommended_backend': 'tcp'
            }
    
    # Client/worker: discover capabilities by testing connection to master
    try:
        from ..runtime.network_discovery import discover_network_capabilities
        
        network_info = await discover_network_capabilities(
            target_addr=config.master_addr,
            target_port=config.master_port,
            timeout=5.0
        )
        
        logger.info(f"  Available backends: {network_info['available_backends']}")
        logger.info(f"  Recommended: {network_info['recommended_backend']}")
        
        return network_info
        
    except ImportError:
        logger.info("  Network discovery module not available, using TCP")
        return {
            'available_backends': ['tcp'],
            'recommended_backend': 'tcp'
        }
    except Exception as e:
        logger.warning(f"  Network discovery failed: {e}")
        logger.info("  Falling back to TCP")
        return {
            'available_backends': ['tcp'],
            'recommended_backend': 'tcp'
        }


async def _initialize_backend(
    state: ClusterState,
    config: ClusterConfig,
    network_info: Dict[str, Any]
):
    """Initialize network backend"""
    # Select backend
    if config.backend == 'auto':
        selected_backend = network_info['recommended_backend']
    else:
        # User explicitly requested a backend
        selected_backend = config.backend
        if selected_backend not in network_info['available_backends']:
            logger.warning(
                f"  Requested backend '{selected_backend}' not available"
            )
            logger.warning(
                f"  Available: {network_info['available_backends']}"
            )
            logger.warning("  Falling back to TCP")
            selected_backend = 'tcp'
    
    logger.info(f"  Selected backend: {selected_backend}")
    
    # Update local node info
    state.local_node.network_backend = selected_backend
    
    # Initialize transport layer
    if selected_backend in ['dpdk', 'dpdk_gpudev']:
        await _init_dpdk_transport(state, config, selected_backend)
    elif selected_backend == 'rdma':
        await _init_rdma_transport(state, config)
    else:  # tcp
        await _init_tcp_transport(state, config)


async def _init_tcp_transport(state: ClusterState, config: ClusterConfig):
    """Initialize TCP transport"""
    try:
        from ..runtime.control_data_integration import ControlDataIntegration
    except ImportError:
        logger.warning("  ControlDataIntegration not available, creating stub")
        # For testing, we'll allow initialization without the full integration
        state.control_integration = None
        return
    
    logger.info("  Initializing TCP transport...")
    
    state.control_integration = ControlDataIntegration(
        node_id=config.node_id,
        control_host='0.0.0.0' if config.node_role in ['server', 'master'] else None,
        control_port=config.master_port if config.node_role in ['server', 'master'] else 0
    )
    
    if config.node_role in ['server', 'master', 'worker']:
        try:
            await state.control_integration.start()
            logger.info(f"  TCP server listening on port {config.master_port}")
        except Exception as e:
            logger.warning(f"  Failed to start TCP server: {e}")


async def _init_dpdk_transport(
    state: ClusterState,
    config: ClusterConfig,
    backend: str
):
    """Initialize DPDK transport"""
    try:
        from ..runtime.transport_coordinator import TransportCoordinator, DataPlaneConfig
    except ImportError:
        logger.warning("  TransportCoordinator not available, falling back to TCP")
        await _init_tcp_transport(state, config)
        state.local_node.network_backend = 'tcp'
        return
    
    logger.info(f"  Initializing DPDK transport (GPUDev: {backend == 'dpdk_gpudev'})...")
    
    data_config = DataPlaneConfig(
        enable_gpudev=(backend == 'dpdk_gpudev'),
        **config.backend_options
    )
    
    state.transport_coordinator = TransportCoordinator(
        node_id=config.node_id,
        config={'data_plane': data_config}
    )
    
    try:
        success = await state.transport_coordinator.initialize()
        if success:
            logger.info("  DPDK initialized successfully")
        else:
            raise RuntimeError("DPDK initialization returned False")
    except Exception as e:
        logger.warning(f"  DPDK initialization failed: {e}")
        logger.info("  Falling back to TCP...")
        
        # Fallback to TCP
        state.transport_coordinator = None
        await _init_tcp_transport(state, config)
        state.local_node.network_backend = 'tcp'


async def _init_rdma_transport(state: ClusterState, config: ClusterConfig):
    """Initialize RDMA transport"""
    logger.warning("  RDMA transport not yet implemented")
    logger.info("  Falling back to TCP...")
    await _init_tcp_transport(state, config)
    state.local_node.network_backend = 'tcp'


async def _connect_to_cluster(state: ClusterState, config: ClusterConfig):
    """Connect to cluster master"""
    if not state.control_integration:
        logger.warning("  No control integration available, skipping cluster connection")
        return
    
    try:
        # Connect to master node
        success = await state.control_integration.connect_to_node(
            host=config.master_addr,
            port=config.master_port
        )
        
        if not success:
            raise ConnectionError(
                f"Failed to connect to {config.master_addr}:{config.master_port}"
            )
        
        logger.info(f"  Connected to master at {config.master_addr}:{config.master_port}")
        
        # Query peer nodes (TODO: implement in Phase 2)
        
    except Exception as e:
        logger.error(f"  Failed to connect to cluster: {e}")
        raise


async def _start_monitoring(state: ClusterState, config: ClusterConfig):
    """Start monitoring services"""
    
    if config.enable_heartbeat:
        task = asyncio.create_task(
            _heartbeat_monitor(state, config),
            name="heartbeat-monitor"
        )
        state.monitor_tasks.append(task)
        logger.info("  ✓ Heartbeat monitor started")
    
    if config.enable_gpu_monitoring and len(state.local_node.gpus) > 0:
        task = asyncio.create_task(
            _gpu_monitor(state, config),
            name="gpu-monitor"
        )
        state.monitor_tasks.append(task)
        logger.info("  ✓ GPU monitor started")
    
    if config.enable_health_checks:
        task = asyncio.create_task(
            _health_monitor(state, config),
            name="health-monitor"
        )
        state.monitor_tasks.append(task)
        logger.info("  ✓ Health monitor started")


async def _heartbeat_monitor(state: ClusterState, config: ClusterConfig):
    """Monitor heartbeats and detect node failures"""
    
    while state.initialized:
        try:
            await asyncio.sleep(config.heartbeat_interval)
            
            current_time = time.time()
            stale_nodes = []
            
            # Check for stale nodes
            for node_id, node in state.nodes.items():
                time_since_heartbeat = current_time - node.last_heartbeat
                if time_since_heartbeat > config.heartbeat_timeout:
                    stale_nodes.append(node_id)
                    logger.warning(
                        f"Node {node_id} heartbeat timeout "
                        f"({time_since_heartbeat:.1f}s > {config.heartbeat_timeout}s)"
                    )
            
            # Remove stale nodes
            for node_id in stale_nodes:
                state.remove_node(node_id)
            
        except asyncio.CancelledError:
            logger.info("Heartbeat monitor stopped")
            break
        except Exception as e:
            logger.error(f"Heartbeat monitor error: {e}")


async def _gpu_monitor(state: ClusterState, config: ClusterConfig):
    """Monitor GPU status"""
    from .node_info import _detect_local_gpus
    
    while state.initialized:
        try:
            await asyncio.sleep(config.gpu_poll_interval)
            
            # Update local GPU status
            if state.local_node:
                gpus = _detect_local_gpus()
                state.local_node.gpus = gpus
            
        except asyncio.CancelledError:
            logger.info("GPU monitor stopped")
            break
        except Exception as e:
            logger.error(f"GPU monitor error: {e}")


async def _health_monitor(state: ClusterState, config: ClusterConfig):
    """Monitor overall system health"""
    
    while state.initialized:
        try:
            await asyncio.sleep(config.health_check_interval)
            
            # Update local node status
            if state.local_node:
                # Check if we have any issues
                healthy_nodes = len(state.get_healthy_nodes())
                total_nodes = len(state.nodes)
                
                if total_nodes > 0 and healthy_nodes < total_nodes * 0.5:
                    # More than 50% nodes unhealthy
                    state.local_node.status = NodeStatus.UNHEALTHY
                    logger.warning(
                        f"Cluster degraded: {healthy_nodes}/{total_nodes} nodes healthy"
                    )
                elif len(state.local_node.gpus) > 0:
                    # Check GPU availability
                    available_gpus = len([g for g in state.local_node.gpus 
                                         if g.health_status == 'healthy'])
                    if available_gpus == 0:
                        state.local_node.status = NodeStatus.BUSY
                    elif state.local_node.active_transfers > 0:
                        state.local_node.status = NodeStatus.BUSY
                    else:
                        state.local_node.status = NodeStatus.ACTIVE
                else:
                    state.local_node.status = NodeStatus.ACTIVE
            
        except asyncio.CancelledError:
            logger.info("Health monitor stopped")
            break
        except Exception as e:
            logger.error(f"Health monitor error: {e}")

