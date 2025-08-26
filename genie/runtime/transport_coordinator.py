"""
Transport Coordinator for Genie Zero-Copy Transport

This module provides the Python-side coordination between:
1. Control Plane (Python TCP server) - handles transfer negotiation, JSON protocols
2. Data Plane (C++ DPDK library) - handles zero-copy packet processing, GPU memory

ARCHITECTURE SEPARATION:
┌─────────────────────────────────────────────────────────────┐
│                    PYTHON LAYER                            │
├─────────────────────────────────────────────────────────────┤
│  Control Plane (TCP)     │  Transport Coordinator          │
│  - Transfer negotiation  │  - Bridges Python ↔ C++        │
│  - Node capabilities     │  - Transfer lifecycle mgmt      │
│  - JSON protocols        │  - Async coordination           │
│  - Async TCP server      │  - ctypes bindings              │
└─────────────────────────────┬───────────────────────────────┘
                              │ ctypes interface
┌─────────────────────────────┴───────────────────────────────┐
│                     C++ LAYER                              │
├─────────────────────────────────────────────────────────────┤
│  Data Plane (DPDK)                                         │
│  - Zero-copy packet processing                             │
│  - GPU memory registration (GPUDev)                       │
│  - Custom reliable transport                               │
│  - High-performance RX/TX loops                           │
│  - Fragment management                                     │
│  - ACK/NACK reliability                                    │
└─────────────────────────────────────────────────────────────┘

The coordinator is responsible for:
- Initializing and managing the C++ data plane via ctypes
- Coordinating between control plane requests and data plane operations
- Managing transfer lifecycle and state synchronization
- Providing high-level Python API for tensor transfers
- Bridging async Python world with high-performance C++ DPDK world

NOTE: All packet processing, GPU memory management, and network I/O
is handled by the C++ data plane (src/data_plane/genie_data_plane.cpp).
This coordinator only handles coordination and control logic.
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Callable, Any, Tuple
import threading
import weakref

# Import our control plane components
from .control_server import (
    ControlPlaneServer, TransferRequest, TransferResponse, 
    ControlMessage, MessageType, NodeCapabilities
)

logger = logging.getLogger(__name__)

class TransferState(IntEnum):
    """Transfer state tracking (matches C++ enum)"""
    PENDING = 0
    NEGOTIATING = 1
    PREPARING = 2
    SENDING = 3
    RECEIVING = 4
    COMPLETED = 5
    FAILED = 6
    CANCELLED = 7

@dataclass
class DataPlaneConfig:
    """Configuration for C++ data plane"""
    # DPDK configuration
    eal_args: List[str] = field(default_factory=lambda: [
        "genie-transport", "-c", "0x3", "-n", "4", "--huge-dir", "/mnt/huge"
    ])
    port_id: int = 0
    queue_id: int = 0
    mempool_size: int = 8192
    rx_ring_size: int = 1024
    tx_ring_size: int = 1024
    
    # GPU configuration
    gpu_device_id: int = 0
    enable_gpudev: bool = True
    
    # Network configuration
    local_ip: str = "192.168.1.100"
    local_mac: str = "aa:bb:cc:dd:ee:01"
    data_port: int = 5556
    
    # Performance tuning
    burst_size: int = 32
    poll_interval_us: int = 100
    enable_batching: bool = True
    
    # Reliability
    ack_timeout_ms: int = 100
    max_retries: int = 3
    window_size: int = 64
    
    def to_json(self) -> str:
        """Convert to JSON for C++ consumption"""
        return json.dumps({
            'eal_args': self.eal_args,
            'port_id': self.port_id,
            'queue_id': self.queue_id,
            'mempool_size': self.mempool_size,
            'rx_ring_size': self.rx_ring_size,
            'tx_ring_size': self.tx_ring_size,
            'gpu_device_id': self.gpu_device_id,
            'enable_gpudev': self.enable_gpudev,
            'local_ip': self.local_ip,
            'local_mac': self.local_mac,
            'data_port': self.data_port,
            'burst_size': self.burst_size,
            'poll_interval_us': self.poll_interval_us,
            'enable_batching': self.enable_batching,
            'ack_timeout_ms': self.ack_timeout_ms,
            'max_retries': self.max_retries,
            'window_size': self.window_size
        })

@dataclass
class TransferContext:
    """Python-side transfer context"""
    transfer_id: str
    tensor_id: str
    source_node: str
    target_node: str
    size: int
    dtype: str
    shape: List[int]
    state: TransferState = TransferState.PENDING
    
    # Tensor reference (weak to avoid memory leaks)
    tensor_ref: Optional[weakref.ReferenceType] = None
    gpu_ptr: Optional[int] = None
    
    # Timing and callbacks
    start_time: float = field(default_factory=time.time)
    completion_callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    progress_callback: Optional[Callable] = None
    
    # Statistics (updated from C++)
    bytes_transferred: int = 0
    packets_sent: int = 0
    packets_received: int = 0

class DataPlaneBindings:
    """ctypes bindings to C++ data plane"""
    
    def __init__(self):
        self.lib: Optional[ctypes.CDLL] = None
        self.data_plane_ptr: Optional[ctypes.c_void_p] = None
        self._load_library()
        self._setup_function_signatures()
    
    def _load_library(self):
        """Load the C++ data plane shared library"""
        try:
            # Try different possible library paths
            library_paths = [
                "./libgenie_data_plane.so",
                "/usr/local/lib/libgenie_data_plane.so",
                "/opt/genie/lib/libgenie_data_plane.so"
            ]
            
            for path in library_paths:
                try:
                    self.lib = ctypes.CDLL(path)
                    logger.info(f"Loaded data plane library from {path}")
                    break
                except OSError:
                    continue
            
            if self.lib is None:
                logger.warning("C++ data plane library not found, using fallback mode")
                
        except Exception as e:
            logger.error(f"Failed to load data plane library: {e}")
            self.lib = None
    
    def _setup_function_signatures(self):
        """Setup ctypes function signatures"""
        if not self.lib:
            return
        
        try:
            # Data plane lifecycle
            self.lib.genie_data_plane_create.argtypes = [ctypes.c_char_p]
            self.lib.genie_data_plane_create.restype = ctypes.c_void_p
            
            self.lib.genie_data_plane_initialize.argtypes = [ctypes.c_void_p]
            self.lib.genie_data_plane_initialize.restype = ctypes.c_int
            
            self.lib.genie_data_plane_start.argtypes = [ctypes.c_void_p]
            self.lib.genie_data_plane_start.restype = ctypes.c_int
            
            self.lib.genie_data_plane_stop.argtypes = [ctypes.c_void_p]
            self.lib.genie_data_plane_stop.restype = None
            
            self.lib.genie_data_plane_destroy.argtypes = [ctypes.c_void_p]
            self.lib.genie_data_plane_destroy.restype = None
            
            # Transfer operations
            self.lib.genie_send_tensor.argtypes = [
                ctypes.c_void_p,    # data_plane
                ctypes.c_char_p,    # transfer_id
                ctypes.c_char_p,    # tensor_id
                ctypes.c_void_p,    # gpu_ptr
                ctypes.c_size_t,    # size
                ctypes.c_char_p     # target_node
            ]
            self.lib.genie_send_tensor.restype = ctypes.c_int
            
            self.lib.genie_receive_tensor.argtypes = [
                ctypes.c_void_p,    # data_plane
                ctypes.c_char_p,    # transfer_id
                ctypes.c_char_p,    # tensor_id
                ctypes.c_void_p,    # gpu_ptr
                ctypes.c_size_t,    # size
                ctypes.c_char_p     # source_node
            ]
            self.lib.genie_receive_tensor.restype = ctypes.c_int
            
            # Memory management
            self.lib.genie_register_gpu_memory.argtypes = [
                ctypes.c_void_p,        # data_plane
                ctypes.c_void_p,        # gpu_ptr
                ctypes.c_size_t,        # size
                ctypes.POINTER(ctypes.c_uint64)  # iova output
            ]
            self.lib.genie_register_gpu_memory.restype = ctypes.c_int
            
            self.lib.genie_unregister_gpu_memory.argtypes = [
                ctypes.c_void_p,    # data_plane
                ctypes.c_void_p     # gpu_ptr
            ]
            self.lib.genie_unregister_gpu_memory.restype = None
            
            # Status and monitoring
            self.lib.genie_get_statistics.argtypes = [
                ctypes.c_void_p,    # data_plane
                ctypes.c_char_p,    # stats_json buffer
                ctypes.c_size_t     # buffer_size
            ]
            self.lib.genie_get_statistics.restype = None
            
            self.lib.genie_get_transfer_status.argtypes = [
                ctypes.c_void_p,    # data_plane
                ctypes.c_char_p,    # transfer_id
                ctypes.c_char_p,    # status_json buffer
                ctypes.c_size_t     # buffer_size
            ]
            self.lib.genie_get_transfer_status.restype = ctypes.c_int
            
            # Configuration
            self.lib.genie_set_target_node.argtypes = [
                ctypes.c_void_p,    # data_plane
                ctypes.c_char_p,    # node_id
                ctypes.c_char_p,    # ip
                ctypes.c_char_p     # mac
            ]
            self.lib.genie_set_target_node.restype = None
            
            self.lib.genie_remove_target_node.argtypes = [
                ctypes.c_void_p,    # data_plane
                ctypes.c_char_p     # node_id
            ]
            self.lib.genie_remove_target_node.restype = None
            
            logger.info("C++ data plane function signatures configured")
            
        except Exception as e:
            logger.error(f"Failed to setup function signatures: {e}")
            self.lib = None
    
    def is_available(self) -> bool:
        """Check if C++ data plane is available"""
        return self.lib is not None and self.data_plane_ptr is not None
    
    def create(self, config: DataPlaneConfig) -> bool:
        """Create C++ data plane instance"""
        if not self.lib:
            return False
        
        try:
            config_json = config.to_json().encode('utf-8')
            self.data_plane_ptr = self.lib.genie_data_plane_create(config_json)
            return self.data_plane_ptr is not None
        except Exception as e:
            logger.error(f"Failed to create data plane: {e}")
            return False
    
    def initialize(self) -> bool:
        """Initialize C++ data plane"""
        if not self.is_available():
            return False
        
        try:
            result = self.lib.genie_data_plane_initialize(self.data_plane_ptr)
            return result == 0
        except Exception as e:
            logger.error(f"Failed to initialize data plane: {e}")
            return False
    
    def start(self) -> bool:
        """Start C++ data plane"""
        if not self.is_available():
            return False
        
        try:
            result = self.lib.genie_data_plane_start(self.data_plane_ptr)
            return result == 0
        except Exception as e:
            logger.error(f"Failed to start data plane: {e}")
            return False
    
    def stop(self):
        """Stop C++ data plane"""
        if self.is_available():
            try:
                self.lib.genie_data_plane_stop(self.data_plane_ptr)
            except Exception as e:
                logger.error(f"Failed to stop data plane: {e}")
    
    def destroy(self):
        """Destroy C++ data plane"""
        if self.is_available():
            try:
                self.lib.genie_data_plane_destroy(self.data_plane_ptr)
                self.data_plane_ptr = None
            except Exception as e:
                logger.error(f"Failed to destroy data plane: {e}")
    
    def send_tensor(self, transfer_id: str, tensor_id: str, gpu_ptr: int, size: int, target_node: str) -> bool:
        """Send tensor via C++ data plane"""
        if not self.is_available():
            return False
        
        try:
            result = self.lib.genie_send_tensor(
                self.data_plane_ptr,
                transfer_id.encode('utf-8'),
                tensor_id.encode('utf-8'),
                ctypes.c_void_p(gpu_ptr),
                ctypes.c_size_t(size),
                target_node.encode('utf-8')
            )
            return result == 0
        except Exception as e:
            logger.error(f"Failed to send tensor: {e}")
            return False
    
    def receive_tensor(self, transfer_id: str, tensor_id: str, gpu_ptr: int, size: int, source_node: str) -> bool:
        """Receive tensor via C++ data plane"""
        if not self.is_available():
            return False
        
        try:
            result = self.lib.genie_receive_tensor(
                self.data_plane_ptr,
                transfer_id.encode('utf-8'),
                tensor_id.encode('utf-8'),
                ctypes.c_void_p(gpu_ptr),
                ctypes.c_size_t(size),
                source_node.encode('utf-8')
            )
            return result == 0
        except Exception as e:
            logger.error(f"Failed to receive tensor: {e}")
            return False
    
    def register_gpu_memory(self, gpu_ptr: int, size: int) -> Optional[int]:
        """Register GPU memory for DMA"""
        if not self.is_available():
            return None
        
        try:
            iova = ctypes.c_uint64()
            result = self.lib.genie_register_gpu_memory(
                self.data_plane_ptr,
                ctypes.c_void_p(gpu_ptr),
                ctypes.c_size_t(size),
                ctypes.byref(iova)
            )
            return iova.value if result == 0 else None
        except Exception as e:
            logger.error(f"Failed to register GPU memory: {e}")
            return None
    
    def unregister_gpu_memory(self, gpu_ptr: int):
        """Unregister GPU memory"""
        if self.is_available():
            try:
                self.lib.genie_unregister_gpu_memory(
                    self.data_plane_ptr,
                    ctypes.c_void_p(gpu_ptr)
                )
            except Exception as e:
                logger.error(f"Failed to unregister GPU memory: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data plane statistics"""
        if not self.is_available():
            return {}
        
        try:
            buffer_size = 4096
            buffer = ctypes.create_string_buffer(buffer_size)
            self.lib.genie_get_statistics(self.data_plane_ptr, buffer, buffer_size)
            
            stats_json = buffer.value.decode('utf-8')
            return json.loads(stats_json) if stats_json else {}
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def get_transfer_status(self, transfer_id: str) -> Optional[Dict[str, Any]]:
        """Get transfer status from C++ data plane"""
        if not self.is_available():
            return None
        
        try:
            buffer_size = 2048
            buffer = ctypes.create_string_buffer(buffer_size)
            result = self.lib.genie_get_transfer_status(
                self.data_plane_ptr,
                transfer_id.encode('utf-8'),
                buffer,
                buffer_size
            )
            
            if result == 0:
                status_json = buffer.value.decode('utf-8')
                return json.loads(status_json) if status_json else None
            return None
        except Exception as e:
            logger.error(f"Failed to get transfer status: {e}")
            return None
    
    def set_target_node(self, node_id: str, ip: str, mac: str):
        """Configure target node for transfers"""
        if self.is_available():
            try:
                self.lib.genie_set_target_node(
                    self.data_plane_ptr,
                    node_id.encode('utf-8'),
                    ip.encode('utf-8'),
                    mac.encode('utf-8')
                )
            except Exception as e:
                logger.error(f"Failed to set target node: {e}")

class TransportCoordinator:
    """
    Main transport coordinator
    
    Bridges Python control plane and C++ data plane for complete zero-copy transport
    """
    
    def __init__(self, node_id: str, config: Dict[str, Any] = None):
        self.node_id = node_id
        self.config = config or {}
        
        # Component initialization
        self.control_server: Optional[ControlPlaneServer] = None
        self.data_plane: Optional[DataPlaneBindings] = None
        self.data_plane_config: Optional[DataPlaneConfig] = None
        
        # Transfer tracking
        self.active_transfers: Dict[str, TransferContext] = {}
        self.transfer_lock = threading.RLock()
        
        # Status monitoring
        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.transfer_callbacks: Dict[str, List[Callable]] = {
            'started': [],
            'progress': [],
            'completed': [],
            'failed': []
        }
    
    async def initialize(self) -> bool:
        """Initialize transport coordinator"""
        try:
            logger.info(f"Initializing transport coordinator for node {self.node_id}")
            
            # Initialize data plane configuration
            await self._init_data_plane_config()
            
            # Initialize C++ data plane
            await self._init_data_plane()
            
            # Initialize control plane
            await self._init_control_plane()
            
            # Start monitoring
            await self._start_monitoring()
            
            self.running = True
            logger.info("Transport coordinator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize transport coordinator: {e}")
            await self.shutdown()
            return False
    
    async def _init_data_plane_config(self):
        """Initialize data plane configuration"""
        dp_config = self.config.get('data_plane', {})
        
        self.data_plane_config = DataPlaneConfig(
            eal_args=dp_config.get('eal_args', [
                f"genie-{self.node_id}", "-c", "0x3", "-n", "4", "--huge-dir", "/mnt/huge"
            ]),
            port_id=dp_config.get('port_id', 0),
            local_ip=dp_config.get('local_ip', '192.168.1.100'),
            local_mac=dp_config.get('local_mac', 'aa:bb:cc:dd:ee:01'),
            data_port=dp_config.get('data_port', 5556),
            gpu_device_id=dp_config.get('gpu_device_id', 0),
            enable_gpudev=dp_config.get('enable_gpudev', True)
        )
        
        logger.info("Data plane configuration initialized")
    
    async def _init_data_plane(self):
        """Initialize C++ data plane"""
        self.data_plane = DataPlaneBindings()
        
        if not self.data_plane.lib:
            logger.warning("C++ data plane not available, using fallback mode")
            return
        
        # Create data plane instance
        if not self.data_plane.create(self.data_plane_config):
            raise RuntimeError("Failed to create C++ data plane")
        
        # Initialize data plane
        if not self.data_plane.initialize():
            raise RuntimeError("Failed to initialize C++ data plane")
        
        # Start data plane
        if not self.data_plane.start():
            raise RuntimeError("Failed to start C++ data plane")
        
        logger.info("C++ data plane initialized and started")
    
    async def _init_control_plane(self):
        """Initialize control plane server"""
        from .control_server import get_control_server
        
        # Get control server configuration
        control_config = self.config.get('control_plane', {})
        capabilities = NodeCapabilities(
            node_id=self.node_id,
            gpu_count=control_config.get('gpu_count', 1),
            max_transfer_size=control_config.get('max_transfer_size', 10 * 1024 * 1024 * 1024),
            data_port=self.data_plane_config.data_port if self.data_plane_config else 5556
        )
        
        self.control_server = get_control_server(
            node_id=self.node_id,
            host=control_config.get('host', '0.0.0.0'),
            port=control_config.get('port', 5555),
            capabilities=capabilities
        )
        
        # Add transfer callbacks
        self.control_server.add_transfer_callback('request', self._handle_transfer_request)
        self.control_server.add_transfer_callback('ready', self._handle_transfer_ready)
        self.control_server.add_transfer_callback('complete', self._handle_transfer_complete)
        self.control_server.add_transfer_callback('error', self._handle_transfer_error)
        
        # Start control server
        await self.control_server.start()
        
        logger.info(f"Control plane server started on {self.control_server.host}:{self.control_server.port}")
    
    async def _start_monitoring(self):
        """Start background monitoring tasks"""
        self.monitor_task = asyncio.create_task(self._monitor_transfers())
        logger.info("Transfer monitoring started")
    
    async def _monitor_transfers(self):
        """Monitor active transfers and sync with C++ data plane"""
        while self.running:
            try:
                await self._sync_transfer_status()
                await asyncio.sleep(1.0)  # Check every second
            except Exception as e:
                logger.error(f"Transfer monitoring error: {e}")
                await asyncio.sleep(5.0)
    
    async def _sync_transfer_status(self):
        """Sync transfer status between Python and C++"""
        if not self.data_plane or not self.data_plane.is_available():
            return
        
        with self.transfer_lock:
            for transfer_id, context in list(self.active_transfers.items()):
                try:
                    # Get status from C++ data plane
                    cpp_status = self.data_plane.get_transfer_status(transfer_id)
                    
                    if cpp_status:
                        # Update context with C++ status
                        context.bytes_transferred = cpp_status.get('bytes_transferred', 0)
                        context.packets_sent = cpp_status.get('packets_sent', 0)
                        context.packets_received = cpp_status.get('packets_received', 0)
                        
                        # Check for completion or error
                        if cpp_status.get('is_complete', False):
                            context.state = TransferState.COMPLETED
                            await self._handle_transfer_completion(transfer_id, context)
                        elif cpp_status.get('has_error', False):
                            context.state = TransferState.FAILED
                            error_msg = cpp_status.get('error_message', 'Unknown error')
                            await self._handle_transfer_failure(transfer_id, context, error_msg)
                        
                        # Call progress callback
                        if context.progress_callback:
                            try:
                                await context.progress_callback(transfer_id, context)
                            except Exception as e:
                                logger.error(f"Progress callback error: {e}")
                
                except Exception as e:
                    logger.error(f"Failed to sync transfer {transfer_id}: {e}")
    
    async def send_tensor(self, tensor: Any, target_node: str, 
                         completion_callback: Callable = None,
                         error_callback: Callable = None,
                         progress_callback: Callable = None) -> str:
        """
        Send tensor to target node using zero-copy transport
        
        Args:
            tensor: PyTorch tensor to send
            target_node: Target node ID
            completion_callback: Called when transfer completes
            error_callback: Called if transfer fails
            progress_callback: Called periodically with progress updates
            
        Returns:
            Transfer ID for tracking
        """
        try:
            # Generate transfer and tensor IDs
            transfer_id = str(uuid.uuid4())
            tensor_id = str(uuid.uuid4())
            
            # Get tensor properties
            gpu_ptr = tensor.data_ptr()
            size = tensor.numel() * tensor.element_size()
            dtype = str(tensor.dtype)
            shape = list(tensor.shape)
            
            # Create transfer context
            context = TransferContext(
                transfer_id=transfer_id,
                tensor_id=tensor_id,
                source_node=self.node_id,
                target_node=target_node,
                size=size,
                dtype=dtype,
                shape=shape,
                tensor_ref=weakref.ref(tensor),
                gpu_ptr=gpu_ptr,
                completion_callback=completion_callback,
                error_callback=error_callback,
                progress_callback=progress_callback
            )
            
            with self.transfer_lock:
                self.active_transfers[transfer_id] = context
            
            # Send transfer request via control plane
            await self._send_transfer_request(context)
            
            logger.info(f"Started tensor transfer {transfer_id} to {target_node} ({size} bytes)")
            
            # Call started callbacks
            for callback in self.transfer_callbacks['started']:
                try:
                    await callback(transfer_id, context)
                except Exception as e:
                    logger.error(f"Started callback error: {e}")
            
            return transfer_id
            
        except Exception as e:
            logger.error(f"Failed to start tensor transfer: {e}")
            if error_callback:
                try:
                    await error_callback(transfer_id, str(e))
                except:
                    pass
            raise
    
    async def _send_transfer_request(self, context: TransferContext):
        """Send transfer request via control plane"""
        request = TransferRequest(
            transfer_id=context.transfer_id,
            tensor_id=context.tensor_id,
            source_node=context.source_node,
            target_node=context.target_node,
            size=context.size,
            dtype=context.dtype,
            shape=context.shape
        )
        
        context.state = TransferState.NEGOTIATING
        
        # This would be sent via control server to target node
        # For now, we'll simulate immediate acceptance and start data plane transfer
        await self._start_data_plane_send(context)
    
    async def _start_data_plane_send(self, context: TransferContext):
        """Start data plane transfer"""
        if not self.data_plane or not self.data_plane.is_available():
            logger.warning("C++ data plane not available, using fallback")
            context.state = TransferState.FAILED
            return
        
        # Register GPU memory if needed
        if context.gpu_ptr:
            iova = self.data_plane.register_gpu_memory(context.gpu_ptr, context.size)
            if not iova:
                logger.error("Failed to register GPU memory for transfer")
                context.state = TransferState.FAILED
                return
        
        # Start C++ data plane transfer
        success = self.data_plane.send_tensor(
            context.transfer_id,
            context.tensor_id,
            context.gpu_ptr,
            context.size,
            context.target_node
        )
        
        if success:
            context.state = TransferState.SENDING
            logger.info(f"Data plane transfer started for {context.transfer_id}")
        else:
            context.state = TransferState.FAILED
            logger.error(f"Failed to start data plane transfer for {context.transfer_id}")
    
    async def _handle_transfer_request(self, request: TransferRequest):
        """Handle incoming transfer request from control plane"""
        try:
            logger.info(f"Received transfer request {request.transfer_id} from {request.source_node}")
            
            # Create context for incoming transfer
            context = TransferContext(
                transfer_id=request.transfer_id,
                tensor_id=request.tensor_id,
                source_node=request.source_node,
                target_node=request.target_node,
                size=request.size,
                dtype=request.dtype,
                shape=request.shape,
                state=TransferState.PREPARING
            )
            
            with self.transfer_lock:
                self.active_transfers[request.transfer_id] = context
            
            # Prepare to receive (allocate memory, etc.)
            await self._prepare_receive_transfer(context)
            
        except Exception as e:
            logger.error(f"Failed to handle transfer request: {e}")
    
    async def _prepare_receive_transfer(self, context: TransferContext):
        """Prepare to receive incoming transfer"""
        try:
            # For now, just mark as ready to receive
            # In full implementation, this would allocate GPU memory
            
            if self.data_plane and self.data_plane.is_available():
                # Start C++ data plane receive
                success = self.data_plane.receive_tensor(
                    context.transfer_id,
                    context.tensor_id,
                    0,  # Will allocate GPU memory in C++
                    context.size,
                    context.source_node
                )
                
                if success:
                    context.state = TransferState.RECEIVING
                    logger.info(f"Prepared to receive transfer {context.transfer_id}")
                else:
                    context.state = TransferState.FAILED
                    logger.error(f"Failed to prepare receive for {context.transfer_id}")
            else:
                logger.warning("C++ data plane not available for receive")
                context.state = TransferState.FAILED
                
        except Exception as e:
            logger.error(f"Failed to prepare receive transfer: {e}")
            context.state = TransferState.FAILED
    
    async def _handle_transfer_ready(self, transfer_id: str, payload: Dict[str, Any]):
        """Handle transfer ready notification"""
        logger.info(f"Transfer {transfer_id} is ready")
    
    async def _handle_transfer_complete(self, transfer_id: str, payload: Dict[str, Any]):
        """Handle transfer completion notification"""
        with self.transfer_lock:
            if transfer_id in self.active_transfers:
                context = self.active_transfers[transfer_id]
                await self._handle_transfer_completion(transfer_id, context)
    
    async def _handle_transfer_completion(self, transfer_id: str, context: TransferContext):
        """Handle transfer completion"""
        context.state = TransferState.COMPLETED
        
        # Cleanup GPU memory registration
        if context.gpu_ptr and self.data_plane and self.data_plane.is_available():
            self.data_plane.unregister_gpu_memory(context.gpu_ptr)
        
        # Call completion callback
        if context.completion_callback:
            try:
                await context.completion_callback(transfer_id, context)
            except Exception as e:
                logger.error(f"Completion callback error: {e}")
        
        # Call transport callbacks
        for callback in self.transfer_callbacks['completed']:
            try:
                await callback(transfer_id, context)
            except Exception as e:
                logger.error(f"Transport callback error: {e}")
        
        # Remove from active transfers
        with self.transfer_lock:
            if transfer_id in self.active_transfers:
                del self.active_transfers[transfer_id]
        
        logger.info(f"Transfer {transfer_id} completed successfully")
    
    async def _handle_transfer_error(self, transfer_id: str, payload: Dict[str, Any]):
        """Handle transfer error notification"""
        with self.transfer_lock:
            if transfer_id in self.active_transfers:
                context = self.active_transfers[transfer_id]
                error_msg = payload.get('reason', 'Unknown error')
                await self._handle_transfer_failure(transfer_id, context, error_msg)
    
    async def _handle_transfer_failure(self, transfer_id: str, context: TransferContext, error_msg: str):
        """Handle transfer failure"""
        context.state = TransferState.FAILED
        
        # Cleanup GPU memory registration
        if context.gpu_ptr and self.data_plane and self.data_plane.is_available():
            self.data_plane.unregister_gpu_memory(context.gpu_ptr)
        
        # Call error callback
        if context.error_callback:
            try:
                await context.error_callback(transfer_id, error_msg)
            except Exception as e:
                logger.error(f"Error callback error: {e}")
        
        # Call transport callbacks
        for callback in self.transfer_callbacks['failed']:
            try:
                await callback(transfer_id, context)
            except Exception as e:
                logger.error(f"Transport callback error: {e}")
        
        # Remove from active transfers
        with self.transfer_lock:
            if transfer_id in self.active_transfers:
                del self.active_transfers[transfer_id]
        
        logger.error(f"Transfer {transfer_id} failed: {error_msg}")
    
    def add_transfer_callback(self, event: str, callback: Callable):
        """Add callback for transfer events"""
        if event in self.transfer_callbacks:
            self.transfer_callbacks[event].append(callback)
        else:
            raise ValueError(f"Unknown event type: {event}")
    
    def get_transfer_status(self, transfer_id: str) -> Optional[TransferContext]:
        """Get status of active transfer"""
        with self.transfer_lock:
            return self.active_transfers.get(transfer_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get transport statistics"""
        stats = {}
        
        # Get C++ data plane statistics
        if self.data_plane and self.data_plane.is_available():
            stats.update(self.data_plane.get_statistics())
        
        # Add Python-side statistics
        with self.transfer_lock:
            stats['active_transfers'] = len(self.active_transfers)
        
        return stats
    
    def set_target_node(self, node_id: str, ip: str, mac: str):
        """Configure target node"""
        if self.data_plane and self.data_plane.is_available():
            self.data_plane.set_target_node(node_id, ip, mac)
        
        logger.info(f"Configured target node {node_id}: {ip} ({mac})")
    
    async def shutdown(self):
        """Shutdown transport coordinator"""
        logger.info("Shutting down transport coordinator")
        
        self.running = False
        
        # Stop monitoring
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop control server
        if self.control_server:
            await self.control_server.stop()
        
        # Cleanup active transfers
        with self.transfer_lock:
            for transfer_id, context in self.active_transfers.items():
                if context.gpu_ptr and self.data_plane and self.data_plane.is_available():
                    self.data_plane.unregister_gpu_memory(context.gpu_ptr)
            self.active_transfers.clear()
        
        # Stop C++ data plane
        if self.data_plane:
            self.data_plane.stop()
            self.data_plane.destroy()
        
        logger.info("Transport coordinator shutdown complete")

# Global coordinator instance
_transport_coordinator: Optional[TransportCoordinator] = None

def get_transport_coordinator(node_id: str = None, config: Dict[str, Any] = None) -> TransportCoordinator:
    """Get global transport coordinator instance"""
    global _transport_coordinator
    if _transport_coordinator is None:
        if node_id is None:
            import socket
            node_id = f"genie-{socket.gethostname()}"
        _transport_coordinator = TransportCoordinator(node_id, config)
    return _transport_coordinator

async def initialize_transport(node_id: str = None, config: Dict[str, Any] = None) -> TransportCoordinator:
    """Initialize and return transport coordinator"""
    coordinator = get_transport_coordinator(node_id, config)
    await coordinator.initialize()
    return coordinator

async def shutdown_transport():
    """Shutdown global transport coordinator"""
    global _transport_coordinator
    if _transport_coordinator:
        await _transport_coordinator.shutdown()
        _transport_coordinator = None
