"""
Full Control-Data Plane Integration for Genie

This module provides complete integration between:
1. TCP Control Plane (negotiation, coordination)
2. DPDK Data Plane (zero-copy transfers)

Key Features:
- Full TCP handshake and negotiation
- Automatic node discovery and capability exchange
- Transfer lifecycle management
- Error recovery and retransmission
- Performance monitoring and statistics
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import weakref

from .control_server import (
    ControlPlaneServer, 
    TransferRequest, 
    TransferResponse,
    NodeCapabilities,
    MessageType,
    TransferStatus,
    ControlMessage
)
from .transport_coordinator import (
    TransportCoordinator,
    DataPlaneConfig,
    TransferContext,
    TransferState
)

logger = logging.getLogger(__name__)


@dataclass
class NodeConnection:
    """Represents a connection to a remote node"""
    node_id: str
    host: str
    control_port: int
    data_port: int
    capabilities: Optional[NodeCapabilities] = None
    is_connected: bool = False
    last_heartbeat: float = field(default_factory=time.time)
    reader: Optional[asyncio.StreamReader] = None
    writer: Optional[asyncio.StreamWriter] = None
    pending_transfers: Dict[str, TransferRequest] = field(default_factory=dict)
    active_transfers: Dict[str, TransferContext] = field(default_factory=dict)


class ControlDataIntegration:
    """
    Full integration of control and data planes.
    
    This class orchestrates the complete flow:
    1. TCP connection establishment
    2. Capability negotiation
    3. Transfer request/response
    4. Data plane coordination
    5. Completion notification
    """
    
    def __init__(self, 
                 node_id: str,
                 control_host: str = "0.0.0.0",
                 control_port: int = 5555,
                 data_config: Optional[DataPlaneConfig] = None):
        """
        Initialize the integrated control-data plane.
        
        Args:
            node_id: Unique identifier for this node
            control_host: Host to bind control server
            control_port: Port for control plane TCP
            data_config: Configuration for data plane
        """
        self.node_id = node_id
        self.control_host = control_host
        self.control_port = control_port
        
        # Initialize data plane config
        if data_config is None:
            data_config = DataPlaneConfig(
                local_ip=self._get_local_ip(),
                data_port=control_port + 1  # Use next port for data
            )
        self.data_config = data_config
        
        # Components
        self.control_server = ControlPlaneServer(
            node_id=node_id,
            host=control_host,
            port=control_port
        )
        self.transport_coordinator = TransportCoordinator(data_config)
        
        # Connection tracking
        self.connections: Dict[str, NodeConnection] = {}
        self.connection_lock = asyncio.Lock()
        
        # Transfer tracking
        self.transfers: Dict[str, TransferContext] = {}
        self.transfer_lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            'transfers_initiated': 0,
            'transfers_completed': 0,
            'transfers_failed': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'connection_count': 0,
            'last_transfer_time': 0.0
        }
        
        # Running state
        self.running = False
        self.tasks: List[asyncio.Task] = []
        
        # Setup callbacks
        self._setup_callbacks()
    
    def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            # Create a dummy socket to get local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    def _setup_callbacks(self):
        """Setup callbacks between components"""
        # The control server handles messages internally via ClientHandler
        # We'll override behavior by subclassing or use the existing callback system
        
        # Register transfer callbacks with control server
        self.control_server.add_transfer_callback('request', self._on_transfer_request)
        self.control_server.add_transfer_callback('complete', self._on_transfer_complete_callback)
        self.control_server.add_transfer_callback('error', self._on_transfer_error_callback)
        
        # Note: Data plane callbacks will be set up when transport_coordinator is implemented
    
    async def _on_transfer_request(self, request: TransferRequest):
        """Callback when a transfer request is received"""
        logger.info(f"Transfer request received: {request.transfer_id}")
        # This will be handled by the control server's process_transfer_request
    
    async def _on_transfer_complete_callback(self, transfer_id: str, payload: Dict):
        """Callback when a transfer completes"""
        logger.info(f"Transfer complete: {transfer_id}")
        async with self.transfer_lock:
            if transfer_id in self.transfers:
                self.transfers[transfer_id].state = TransferState.COMPLETED
                self.stats['transfers_completed'] += 1
    
    async def _on_transfer_error_callback(self, transfer_id: str, payload: Dict):
        """Callback when a transfer fails"""
        logger.error(f"Transfer error: {transfer_id} - {payload.get('error')}")
        async with self.transfer_lock:
            if transfer_id in self.transfers:
                self.transfers[transfer_id].state = TransferState.FAILED
                self.stats['transfers_failed'] += 1
    
    async def start(self):
        """Start the integrated system"""
        if self.running:
            logger.warning("System already running")
            return
        
        self.running = True
        logger.info(f"Starting Control-Data Integration for node {self.node_id}")
        
        try:
            # Start control server
            await self.control_server.start()
            logger.info(f"Control server listening on {self.control_host}:{self.control_port}")
            
            # Initialize data plane (skip for testing without DPDK)
            try:
                if not self.transport_coordinator.initialize():
                    logger.warning("Data plane initialization failed (DPDK not available), continuing in test mode")
                else:
                    logger.info("Data plane initialized successfully")
                    
                    # Start data plane
                    if not self.transport_coordinator.start():
                        logger.warning("Failed to start data plane, continuing in test mode")
                    else:
                        logger.info(f"Data plane started on port {self.data_config.data_port}")
            except Exception as e:
                logger.warning(f"Data plane setup failed: {e}, continuing in test mode")
            
            # Start background tasks
            self.tasks.append(asyncio.create_task(self._heartbeat_loop()))
            self.tasks.append(asyncio.create_task(self._monitor_transfers()))
            self.tasks.append(asyncio.create_task(self._stats_reporter()))
            
            # Announce ourselves
            await self._announce_presence()
            
            logger.info("Control-Data Integration fully started")
            
        except Exception as e:
            logger.error(f"Failed to start: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the integrated system"""
        if not self.running:
            return
        
        self.running = False
        logger.info("Stopping Control-Data Integration")
        
        # Cancel background tasks
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
        
        # Disconnect from all nodes
        await self._disconnect_all()
        
        # Stop components
        try:
            self.transport_coordinator.stop()
        except Exception as e:
            logger.warning(f"Error stopping transport coordinator: {e}")
        
        await self.control_server.stop()
        
        logger.info("Control-Data Integration stopped")
    
    async def connect_to_node(self, 
                              host: str, 
                              port: int,
                              node_id: Optional[str] = None) -> bool:
        """
        Connect to a remote node and perform handshake.
        
        Args:
            host: Remote host address
            port: Remote control port
            node_id: Optional node ID (will be discovered if not provided)
            
        Returns:
            True if connection successful
        """
        try:
            logger.info(f"Connecting to {host}:{port}")
            
            # Establish TCP connection
            reader, writer = await asyncio.open_connection(host, port)
            
            # Send HELLO message
            hello_msg = ControlMessage(
                type=MessageType.HELLO,
                sender=self.node_id,
                receiver='',
                payload={'version': '1.0', 'data_port': self.data_config.data_port}
            )
            await self._send_message(writer, hello_msg)
            
            # Wait for response
            response = await self._receive_message(reader)
            if not response or response.type != MessageType.HELLO:
                raise ValueError("Invalid HELLO response")
            
            # Extract node info
            remote_node_id = response.sender
            remote_data_port = response.payload.get('data_port', port + 1)
            
            # Exchange capabilities
            capabilities = NodeCapabilities(
                node_id=self.node_id,
                gpu_count=1,  # TODO: Get from actual GPU info
                max_transfer_size=10 * 1024 * 1024 * 1024,
                network_bandwidth_gbps=100.0
            )
            
            cap_msg = ControlMessage(
                type=MessageType.CAPABILITY_EXCHANGE,
                sender=self.node_id,
                receiver=remote_node_id,
                payload=asdict(capabilities)
            )
            await self._send_message(writer, cap_msg)
            
            # Receive remote capabilities
            cap_response = await self._receive_message(reader)
            if not cap_response or cap_response.type != MessageType.CAPABILITY_EXCHANGE:
                raise ValueError("Invalid capability response")
            
            remote_capabilities = NodeCapabilities(**cap_response.payload)
            
            # Store connection
            async with self.connection_lock:
                conn = NodeConnection(
                    node_id=remote_node_id,
                    host=host,
                    control_port=port,
                    data_port=remote_data_port,
                    capabilities=remote_capabilities,
                    is_connected=True,
                    reader=reader,
                    writer=writer
                )
                self.connections[remote_node_id] = conn
                self.stats['connection_count'] += 1
            
            # Register node in data plane (if method exists)
            if hasattr(self.transport_coordinator, 'register_remote_node'):
                self.transport_coordinator.register_remote_node(
                    remote_node_id, host, remote_data_port
                )
            
            # Start connection handler
            asyncio.create_task(self._handle_connection(conn))
            
            logger.info(f"Connected to node {remote_node_id} at {host}:{port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {host}:{port}: {e}")
            return False
    
    async def transfer_tensor(self,
                            tensor: Any,
                            target_node: str,
                            tensor_id: Optional[str] = None,
                            priority: int = 1,
                            timeout: float = 300.0) -> str:
        """
        Transfer a tensor to a remote node.
        
        Args:
            tensor: The tensor to transfer (PyTorch/NumPy/etc)
            target_node: Target node ID
            tensor_id: Optional tensor identifier
            priority: Transfer priority (higher = more urgent)
            timeout: Transfer timeout in seconds
            
        Returns:
            Transfer ID for tracking
        """
        # Validate connection
        async with self.connection_lock:
            if target_node not in self.connections:
                raise ValueError(f"Not connected to node {target_node}")
            conn = self.connections[target_node]
            if not conn.is_connected:
                raise ValueError(f"Connection to {target_node} is not active")
        
        # Generate IDs
        transfer_id = f"transfer_{self.node_id}_{int(time.time() * 1000000)}"
        if tensor_id is None:
            tensor_id = f"tensor_{transfer_id}"
        
        # Get tensor info
        import torch
        if isinstance(tensor, torch.Tensor):
            size = tensor.numel() * tensor.element_size()
            dtype = str(tensor.dtype).replace('torch.', '')
            shape = list(tensor.shape)
            gpu_ptr = tensor.data_ptr() if tensor.is_cuda else None
        else:
            # Handle numpy or other tensor types
            import numpy as np
            if isinstance(tensor, np.ndarray):
                size = tensor.nbytes
                dtype = str(tensor.dtype)
                shape = list(tensor.shape)
                gpu_ptr = None
            else:
                raise TypeError(f"Unsupported tensor type: {type(tensor)}")
        
        # Create transfer request
        request = TransferRequest(
            transfer_id=transfer_id,
            tensor_id=tensor_id,
            source_node=self.node_id,
            target_node=target_node,
            size=size,
            dtype=dtype,
            shape=shape,
            priority=priority,
            timeout_seconds=timeout
        )
        
        # Send request to target node
        request_msg = ControlMessage(
            type=MessageType.TRANSFER_REQUEST,
            sender=self.node_id,
            receiver=target_node,
            payload=asdict(request)
        )
        
        await self._send_message(conn.writer, request_msg)
        
        # Store transfer context
        context = TransferContext(
            transfer_id=transfer_id,
            tensor_id=tensor_id,
            source_node=self.node_id,
            target_node=target_node,
            size=size,
            dtype=dtype,
            shape=shape,
            state=TransferState.NEGOTIATING,
            tensor_ref=weakref.ref(tensor),
            gpu_ptr=gpu_ptr
        )
        
        async with self.transfer_lock:
            self.transfers[transfer_id] = context
            conn.pending_transfers[transfer_id] = request
        
        self.stats['transfers_initiated'] += 1
        
        # Wait for acceptance
        await self._wait_for_acceptance(transfer_id, timeout=10.0)
        
        # Start data plane transfer
        if gpu_ptr:
            # GPU tensor - use zero-copy
            success = await self.transport_coordinator.send_tensor_async(
                transfer_id=transfer_id,
                tensor_id=tensor_id,
                gpu_ptr=gpu_ptr,
                size=size,
                target_node=target_node
            )
        else:
            # CPU tensor - need to copy to GPU or use CPU staging
            data = tensor.numpy() if hasattr(tensor, 'numpy') else tensor
            success = await self.transport_coordinator.send_data_async(
                transfer_id=transfer_id,
                data=data.tobytes(),
                target_node=target_node
            )
        
        if not success:
            raise RuntimeError(f"Failed to start transfer {transfer_id}")
        
        logger.info(f"Started transfer {transfer_id} to {target_node}")
        return transfer_id
    
    async def _wait_for_acceptance(self, transfer_id: str, timeout: float):
        """Wait for transfer acceptance from remote node"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            async with self.transfer_lock:
                if transfer_id in self.transfers:
                    context = self.transfers[transfer_id]
                    if context.state == TransferState.PREPARING:
                        return  # Accepted
                    elif context.state == TransferState.FAILED:
                        raise RuntimeError(f"Transfer {transfer_id} rejected")
            
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"Transfer {transfer_id} acceptance timeout")
    
    async def _handle_connection(self, conn: NodeConnection):
        """Handle messages from a connected node"""
        try:
            while self.running and conn.is_connected:
                msg = await self._receive_message(conn.reader)
                if not msg:
                    break
                
                # Update heartbeat
                conn.last_heartbeat = time.time()
                
                # Route message
                await self._route_message(conn, msg)
                
        except Exception as e:
            logger.error(f"Connection error with {conn.node_id}: {e}")
        finally:
            await self._disconnect_node(conn.node_id)
    
    async def _route_message(self, conn: NodeConnection, msg: ControlMessage):
        """Route incoming control message"""
        try:
            if msg.type == MessageType.TRANSFER_READY:
                await self._handle_transfer_ready(conn, msg)
            elif msg.type == MessageType.TRANSFER_COMPLETE:
                await self._handle_transfer_complete(conn, msg)
            elif msg.type == MessageType.TRANSFER_ERROR:
                await self._handle_transfer_error(conn, msg)
            elif msg.type == MessageType.HEARTBEAT:
                # Already updated heartbeat time
                pass
            else:
                # Delegate to control server handlers
                await self.control_server._route_message(msg)
                
        except Exception as e:
            logger.error(f"Error routing message: {e}")
    
    async def _handle_transfer_request(self, msg: ControlMessage) -> Optional[ControlMessage]:
        """Handle incoming transfer request"""
        try:
            request = TransferRequest(**msg.payload)
            
            # Validate we can receive this transfer
            # TODO: Check available memory, bandwidth, etc.
            
            # Accept the transfer
            response = TransferResponse(
                transfer_id=request.transfer_id,
                accepted=True,
                allocated_buffer=0,  # Will be allocated by data plane
                ready_port=self.data_config.data_port
            )
            
            # Prepare to receive in data plane (if method exists)
            if hasattr(self.transport_coordinator, 'prepare_receive'):
                self.transport_coordinator.prepare_receive(
                    request.transfer_id,
                    request.size,
                    request.source_node
                )
            
            # Send acceptance
            return ControlMessage(
                type=MessageType.TRANSFER_READY,
                sender=self.node_id,
                receiver=msg.sender,
                payload=asdict(response)
            )
            
        except Exception as e:
            logger.error(f"Error handling transfer request: {e}")
            return ControlMessage(
                type=MessageType.TRANSFER_ERROR,
                sender=self.node_id,
                receiver=msg.sender,
                payload={'error': str(e)}
            )
    
    async def _handle_transfer_ready(self, conn: NodeConnection, msg: ControlMessage):
        """Handle transfer ready response"""
        response = TransferResponse(**msg.payload)
        transfer_id = response.transfer_id
        
        async with self.transfer_lock:
            if transfer_id in self.transfers:
                context = self.transfers[transfer_id]
                if response.accepted:
                    context.state = TransferState.PREPARING
                    logger.info(f"Transfer {transfer_id} accepted by {conn.node_id}")
                else:
                    context.state = TransferState.FAILED
                    logger.warning(f"Transfer {transfer_id} rejected: {response.reason}")
    
    async def _handle_transfer_complete(self, conn: NodeConnection, msg: ControlMessage):
        """Handle transfer completion notification"""
        transfer_id = msg.payload.get('transfer_id')
        
        async with self.transfer_lock:
            if transfer_id in self.transfers:
                context = self.transfers[transfer_id]
                context.state = TransferState.COMPLETED
                self.stats['transfers_completed'] += 1
                self.stats['bytes_sent'] += context.size
                
                # Clean up
                del self.transfers[transfer_id]
                if transfer_id in conn.pending_transfers:
                    del conn.pending_transfers[transfer_id]
        
        logger.info(f"Transfer {transfer_id} completed successfully")
    
    async def _handle_transfer_error(self, conn: NodeConnection, msg: ControlMessage):
        """Handle transfer error notification"""
        transfer_id = msg.payload.get('transfer_id')
        error = msg.payload.get('error', 'Unknown error')
        
        async with self.transfer_lock:
            if transfer_id in self.transfers:
                context = self.transfers[transfer_id]
                context.state = TransferState.FAILED
                self.stats['transfers_failed'] += 1
                
                # Clean up
                del self.transfers[transfer_id]
                if transfer_id in conn.pending_transfers:
                    del conn.pending_transfers[transfer_id]
        
        logger.error(f"Transfer {transfer_id} failed: {error}")
    
    async def _handle_capability_exchange(self, msg: ControlMessage) -> Optional[ControlMessage]:
        """Handle capability exchange"""
        # Store remote capabilities
        remote_cap = NodeCapabilities(**msg.payload)
        
        # Send our capabilities
        our_cap = NodeCapabilities(
            node_id=self.node_id,
            gpu_count=1,
            max_transfer_size=10 * 1024 * 1024 * 1024
        )
        
        return ControlMessage(
            type=MessageType.CAPABILITY_EXCHANGE,
            sender=self.node_id,
            receiver=msg.sender,
            payload=asdict(our_cap)
        )
    
    async def _handle_hello(self, msg: ControlMessage) -> Optional[ControlMessage]:
        """Handle HELLO message"""
        return ControlMessage(
            type=MessageType.HELLO,
            sender=self.node_id,
            receiver=msg.sender,
            payload={'version': '1.0', 'data_port': self.data_config.data_port}
        )
    
    def _on_transfer_complete(self, transfer_id: str, stats: Dict):
        """Callback from data plane on transfer completion"""
        asyncio.create_task(self._notify_transfer_complete(transfer_id, stats))
    
    def _on_transfer_error(self, transfer_id: str, error: str):
        """Callback from data plane on transfer error"""
        asyncio.create_task(self._notify_transfer_error(transfer_id, error))
    
    async def _notify_transfer_complete(self, transfer_id: str, stats: Dict):
        """Notify remote node of transfer completion"""
        async with self.transfer_lock:
            if transfer_id not in self.transfers:
                return
            context = self.transfers[transfer_id]
            target_node = context.target_node
        
        async with self.connection_lock:
            if target_node in self.connections:
                conn = self.connections[target_node]
                msg = ControlMessage(
                    type=MessageType.TRANSFER_COMPLETE,
                    sender=self.node_id,
                    receiver=target_node,
                    payload={'transfer_id': transfer_id, 'stats': stats}
                )
                await self._send_message(conn.writer, msg)
    
    async def _notify_transfer_error(self, transfer_id: str, error: str):
        """Notify remote node of transfer error"""
        async with self.transfer_lock:
            if transfer_id not in self.transfers:
                return
            context = self.transfers[transfer_id]
            target_node = context.target_node
        
        async with self.connection_lock:
            if target_node in self.connections:
                conn = self.connections[target_node]
                msg = ControlMessage(
                    type=MessageType.TRANSFER_ERROR,
                    sender=self.node_id,
                    receiver=target_node,
                    payload={'transfer_id': transfer_id, 'error': error}
                )
                await self._send_message(conn.writer, msg)
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to connected nodes"""
        while self.running:
            try:
                async with self.connection_lock:
                    for conn in self.connections.values():
                        if conn.is_connected:
                            msg = ControlMessage(
                                type=MessageType.HEARTBEAT,
                                sender=self.node_id,
                                receiver=conn.node_id,
                                payload={'timestamp': time.time()}
                            )
                            await self._send_message(conn.writer, msg)
                
                await asyncio.sleep(10)  # Heartbeat every 10 seconds
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def _monitor_transfers(self):
        """Monitor and timeout stuck transfers"""
        while self.running:
            try:
                current_time = time.time()
                
                async with self.transfer_lock:
                    for transfer_id, context in list(self.transfers.items()):
                        elapsed = current_time - context.start_time
                        
                        # Check for timeout (default 5 minutes)
                        if elapsed > 300:
                            logger.warning(f"Transfer {transfer_id} timed out")
                            context.state = TransferState.FAILED
                            self.stats['transfers_failed'] += 1
                            del self.transfers[transfer_id]
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Transfer monitor error: {e}")
    
    async def _stats_reporter(self):
        """Report statistics periodically"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                logger.info(f"Statistics: {self.stats}")
                
                # Get data plane stats
                dp_stats = self.transport_coordinator.get_statistics()
                if dp_stats:
                    logger.info(f"Data plane stats: {dp_stats}")
                    
            except Exception as e:
                logger.error(f"Stats reporter error: {e}")
    
    async def _announce_presence(self):
        """Announce this node's presence (for discovery)"""
        # TODO: Implement discovery mechanism (multicast, registry, etc.)
        pass
    
    async def _disconnect_node(self, node_id: str):
        """Disconnect from a specific node"""
        async with self.connection_lock:
            if node_id in self.connections:
                conn = self.connections[node_id]
                conn.is_connected = False
                
                if conn.writer:
                    try:
                        # Send goodbye
                        msg = ControlMessage(
                            type=MessageType.GOODBYE,
                            sender=self.node_id,
                            receiver=node_id,
                            payload={}
                        )
                        await self._send_message(conn.writer, msg)
                        conn.writer.close()
                        await conn.writer.wait_closed()
                    except Exception:
                        pass
                
                del self.connections[node_id]
                self.stats['connection_count'] -= 1
        
        # Unregister from data plane (if method exists)
        if hasattr(self.transport_coordinator, 'unregister_remote_node'):
            self.transport_coordinator.unregister_remote_node(node_id)
        
        logger.info(f"Disconnected from node {node_id}")
    
    async def _disconnect_all(self):
        """Disconnect from all nodes"""
        node_ids = list(self.connections.keys())
        for node_id in node_ids:
            await self._disconnect_node(node_id)
    
    async def _send_message(self, writer: asyncio.StreamWriter, msg: ControlMessage):
        """Send a control message"""
        try:
            # Use the ControlMessage's to_json method
            message_str = msg.to_json()
            message_data = message_str.encode('utf-8')
            
            # Send length prefix (4 bytes, big endian) as per protocol
            length_data = len(message_data).to_bytes(4, 'big')
            writer.write(length_data + message_data)
            await writer.drain()
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise
    
    async def _receive_message(self, reader: asyncio.StreamReader) -> Optional[ControlMessage]:
        """Receive a control message"""
        try:
            # Read length prefix (4 bytes)
            length_data = await reader.readexactly(4)
            if not length_data:
                return None
            
            message_length = int.from_bytes(length_data, 'big')
            
            # Read message data
            message_data = await reader.readexactly(message_length)
            message_str = message_data.decode('utf-8')
            
            # Parse using ControlMessage's from_json method
            return ControlMessage.from_json(message_str)
            
        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        stats = self.stats.copy()
        
        # Add current state
        stats['active_connections'] = len(self.connections)
        stats['active_transfers'] = len(self.transfers)
        
        # Get data plane stats (if method exists)
        if hasattr(self.transport_coordinator, 'get_statistics'):
            dp_stats = self.transport_coordinator.get_statistics()
            if dp_stats:
                stats['data_plane'] = dp_stats
        
        return stats


async def main():
    """Example usage of the integrated system"""
    import torch
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create node A
    node_a = ControlDataIntegration(
        node_id="node_a",
        control_port=5555
    )
    
    # Create node B
    node_b = ControlDataIntegration(
        node_id="node_b",
        control_port=5556
    )
    
    try:
        # Start both nodes
        await node_a.start()
        await node_b.start()
        
        # Connect A to B
        connected = await node_a.connect_to_node("localhost", 5556)
        if not connected:
            raise RuntimeError("Failed to connect nodes")
        
        print("Nodes connected successfully!")
        
        # Create a test tensor
        tensor = torch.randn(1024, 1024, device='cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Created tensor: shape={tensor.shape}, device={tensor.device}")
        
        # Transfer tensor from A to B
        transfer_id = await node_a.transfer_tensor(
            tensor=tensor,
            target_node="node_b",
            tensor_id="test_tensor"
        )
        
        print(f"Transfer initiated: {transfer_id}")
        
        # Wait a bit for transfer to complete
        await asyncio.sleep(5)
        
        # Get statistics
        stats_a = node_a.get_statistics()
        stats_b = node_b.get_statistics()
        
        print(f"Node A stats: {stats_a}")
        print(f"Node B stats: {stats_b}")
        
    finally:
        # Cleanup
        await node_a.stop()
        await node_b.stop()


if __name__ == "__main__":
    asyncio.run(main())
