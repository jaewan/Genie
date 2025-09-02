"""
Control Plane TCP Server for Genie Zero-Copy Transport

This module implements the TCP-based control plane that coordinates
zero-copy tensor transfers between nodes. It handles:

- Transfer request/response negotiation
- Node capability exchange
- Heartbeat and connection monitoring
- Transfer status tracking
- Error handling and recovery

The control plane uses JSON messages over TCP for reliability and
ease of debugging, while the data plane uses custom UDP packets
for maximum performance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Dict, List, Optional, Set, Callable, Any
import socket
import weakref

logger = logging.getLogger(__name__)

# Control Message Types
class MessageType(IntEnum):
    """Control plane message types"""
    # Connection management
    HELLO = 1
    CAPABILITY_EXCHANGE = 2
    HEARTBEAT = 3
    GOODBYE = 4
    
    # Transfer coordination
    TRANSFER_REQUEST = 10
    TRANSFER_READY = 11
    TRANSFER_START = 12
    TRANSFER_COMPLETE = 13
    TRANSFER_ERROR = 14
    TRANSFER_CANCEL = 15
    
    # Status and monitoring
    STATUS_REQUEST = 20
    STATUS_RESPONSE = 21
    NODE_LIST_REQUEST = 22
    NODE_LIST_RESPONSE = 23

class TransferStatus(IntEnum):
    """Transfer status values"""
    PENDING = 0
    NEGOTIATING = 1
    READY = 2
    IN_PROGRESS = 3
    COMPLETED = 4
    FAILED = 5
    CANCELLED = 6

@dataclass
class NodeCapabilities:
    """Node capability information"""
    node_id: str
    gpu_count: int
    max_transfer_size: int = 10 * 1024 * 1024 * 1024  # 10GB default
    supported_dtypes: List[str] = field(default_factory=lambda: ['float32', 'float16', 'int32', 'int64'])
    network_bandwidth_gbps: float = 100.0  # 100 Gbps default
    memory_bandwidth_gbps: float = 900.0   # 900 GB/s default for modern GPUs
    features: List[str] = field(default_factory=lambda: ['fragmentation', 'compression', 'reliability'])
    # Optional data port for UDP plane (used by integration harness)
    data_port: int = 5556

@dataclass
class TransferRequest:
    """Transfer request information"""
    transfer_id: str
    tensor_id: str
    source_node: str
    target_node: str
    size: int
    dtype: str
    shape: List[int]
    source_gpu: int = 0
    target_gpu: int = 0
    priority: int = 1
    timeout_seconds: float = 300.0
    requires_ack: bool = True
    compression: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TransferResponse:
    """Transfer response information"""
    transfer_id: str
    accepted: bool
    reason: str = ""
    estimated_time_seconds: float = 0.0
    allocated_gpu: int = 0
    data_port: int = 5556  # UDP port for data transfer
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ControlMessage:
    """Control plane message"""
    type: MessageType
    sender: str
    # Optional explicit receiver for point-to-point flows (integration harness)
    receiver: str = ""
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        data = asdict(self)
        data['type'] = int(self.type)  # Convert enum to int
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ControlMessage':
        """Create message from JSON string"""
        data = json.loads(json_str)
        data['type'] = MessageType(data['type'])  # Convert int to enum
        return cls(**data)

class ClientHandler:
    """Handles individual client connections"""
    
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, 
                 server: 'ControlPlaneServer'):
        self.reader = reader
        self.writer = writer
        self.server = server
        self.client_id: Optional[str] = None
        self.capabilities: Optional[NodeCapabilities] = None
        self.last_heartbeat = time.time()
        self.connected = True
        self.pending_transfers: Set[str] = set()
        
        # Get client address
        peername = writer.get_extra_info('peername')
        self.client_address = f"{peername[0]}:{peername[1]}" if peername else "unknown"
        
        logger.info(f"New client connection from {self.client_address}")
    
    async def handle_connection(self):
        """Main connection handling loop"""
        try:
            # Send welcome message
            await self.send_hello()
            
            # Process messages
            while self.connected:
                try:
                    message = await self.receive_message()
                    if message:
                        await self.process_message(message)
                    else:
                        break  # Connection closed
                except asyncio.TimeoutError:
                    # Check for heartbeat timeout
                    if time.time() - self.last_heartbeat > self.server.heartbeat_timeout:
                        logger.warning(f"Client {self.client_id} heartbeat timeout")
                        break
                except Exception as e:
                    logger.error(f"Error processing message from {self.client_id}: {e}")
                    break
        
        except Exception as e:
            logger.error(f"Connection error with {self.client_address}: {e}")
        
        finally:
            await self.cleanup()
    
    async def send_hello(self):
        """Send initial hello message"""
        hello_msg = ControlMessage(
            type=MessageType.HELLO,
            sender=self.server.node_id,
            payload={
                'server_version': '1.0',
                'supported_features': ['transfer_coordination', 'heartbeat', 'status_monitoring'],
                'max_concurrent_transfers': self.server.max_concurrent_transfers
            }
        )
        await self.send_message(hello_msg)
    
    async def receive_message(self) -> Optional[ControlMessage]:
        """Receive and parse a message"""
        try:
            # Read message length (4 bytes, big endian)
            length_data = await asyncio.wait_for(
                self.reader.readexactly(4), 
                timeout=self.server.message_timeout
            )
            message_length = int.from_bytes(length_data, 'big')
            
            if message_length > self.server.max_message_size:
                raise ValueError(f"Message too large: {message_length} bytes")
            
            # Read message data
            message_data = await asyncio.wait_for(
                self.reader.readexactly(message_length),
                timeout=self.server.message_timeout
            )
            
            # Parse JSON message
            message_str = message_data.decode('utf-8')
            return ControlMessage.from_json(message_str)
            
        except asyncio.IncompleteReadError:
            # Connection closed
            return None
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None
    
    async def send_message(self, message: ControlMessage):
        """Send a message to the client"""
        try:
            message_str = message.to_json()
            message_data = message_str.encode('utf-8')
            
            # Send length prefix (4 bytes, big endian)
            length_data = len(message_data).to_bytes(4, 'big')
            self.writer.write(length_data + message_data)
            await self.writer.drain()
            
            logger.debug(f"Sent {message.type.name} to {self.client_id}")
            
        except Exception as e:
            logger.error(f"Error sending message to {self.client_id}: {e}")
            self.connected = False
    
    async def process_message(self, message: ControlMessage):
        """Process received message"""
        self.last_heartbeat = time.time()
        
        logger.debug(f"Received {message.type.name} from {message.sender}")
        
        # Update client ID if not set
        if not self.client_id:
            self.client_id = message.sender
            self.server.register_client(self.client_id, self)
        
        # Dispatch message by type
        handler_map = {
            MessageType.CAPABILITY_EXCHANGE: self.handle_capability_exchange,
            MessageType.HEARTBEAT: self.handle_heartbeat,
            MessageType.TRANSFER_REQUEST: self.handle_transfer_request,
            MessageType.TRANSFER_READY: self.handle_transfer_ready,
            MessageType.TRANSFER_START: self.handle_transfer_start,
            MessageType.TRANSFER_COMPLETE: self.handle_transfer_complete,
            MessageType.TRANSFER_ERROR: self.handle_transfer_error,
            MessageType.TRANSFER_CANCEL: self.handle_transfer_cancel,
            MessageType.STATUS_REQUEST: self.handle_status_request,
            MessageType.NODE_LIST_REQUEST: self.handle_node_list_request,
            MessageType.GOODBYE: self.handle_goodbye,
        }
        
        handler = handler_map.get(message.type)
        if handler:
            await handler(message)
        else:
            logger.warning(f"Unknown message type: {message.type}")
    
    async def handle_capability_exchange(self, message: ControlMessage):
        """Handle capability exchange message"""
        try:
            cap_data = message.payload
            self.capabilities = NodeCapabilities(
                node_id=message.sender,
                gpu_count=cap_data.get('gpu_count', 1),
                max_transfer_size=cap_data.get('max_transfer_size', 10 * 1024 * 1024 * 1024),
                supported_dtypes=cap_data.get('supported_dtypes', ['float32']),
                network_bandwidth_gbps=cap_data.get('network_bandwidth_gbps', 100.0),
                memory_bandwidth_gbps=cap_data.get('memory_bandwidth_gbps', 900.0),
                features=cap_data.get('features', [])
            )
            
            logger.info(f"Received capabilities from {self.client_id}: "
                       f"{self.capabilities.gpu_count} GPUs, "
                       f"{self.capabilities.network_bandwidth_gbps} Gbps network")
            
            # Send our capabilities back
            response = ControlMessage(
                type=MessageType.CAPABILITY_EXCHANGE,
                sender=self.server.node_id,
                payload=asdict(self.server.capabilities)
            )
            await self.send_message(response)
            
        except Exception as e:
            logger.error(f"Error handling capability exchange: {e}")
    
    async def handle_heartbeat(self, message: ControlMessage):
        """Handle heartbeat message"""
        # Heartbeat received, update timestamp
        self.last_heartbeat = time.time()
        
        # Send heartbeat response
        response = ControlMessage(
            type=MessageType.HEARTBEAT,
            sender=self.server.node_id,
            payload={'status': 'alive', 'timestamp': time.time()}
        )
        await self.send_message(response)
    
    async def handle_transfer_request(self, message: ControlMessage):
        """Handle transfer request"""
        try:
            req_data = message.payload
            transfer_request = TransferRequest(
                transfer_id=req_data['transfer_id'],
                tensor_id=req_data['tensor_id'],
                source_node=req_data['source_node'],
                target_node=req_data['target_node'],
                size=req_data['size'],
                dtype=req_data['dtype'],
                shape=req_data['shape'],
                source_gpu=req_data.get('source_gpu', 0),
                target_gpu=req_data.get('target_gpu', 0),
                priority=req_data.get('priority', 1),
                timeout_seconds=req_data.get('timeout_seconds', 300.0),
                requires_ack=req_data.get('requires_ack', True),
                compression=req_data.get('compression', False),
                metadata=req_data.get('metadata', {})
            )
            
            # Process transfer request
            response = await self.server.process_transfer_request(transfer_request, self)
            
            # Send response
            response_msg = ControlMessage(
                type=MessageType.TRANSFER_READY if response.accepted else MessageType.TRANSFER_ERROR,
                sender=self.server.node_id,
                payload=asdict(response)
            )
            await self.send_message(response_msg)
            
            if response.accepted:
                self.pending_transfers.add(transfer_request.transfer_id)
            
        except Exception as e:
            logger.error(f"Error handling transfer request: {e}")
            error_response = TransferResponse(
                transfer_id=req_data.get('transfer_id', 'unknown'),
                accepted=False,
                reason=f"Processing error: {e}"
            )
            error_msg = ControlMessage(
                type=MessageType.TRANSFER_ERROR,
                sender=self.server.node_id,
                payload=asdict(error_response)
            )
            await self.send_message(error_msg)
    
    async def handle_transfer_ready(self, message: ControlMessage):
        """Handle transfer ready notification"""
        transfer_id = message.payload.get('transfer_id')
        if transfer_id:
            await self.server.handle_transfer_ready(transfer_id, message.payload)
    
    async def handle_transfer_start(self, message: ControlMessage):
        """Handle transfer start notification"""
        transfer_id = message.payload.get('transfer_id')
        if transfer_id:
            await self.server.handle_transfer_start(transfer_id, message.payload)
    
    async def handle_transfer_complete(self, message: ControlMessage):
        """Handle transfer completion"""
        transfer_id = message.payload.get('transfer_id')
        if transfer_id:
            self.pending_transfers.discard(transfer_id)
            await self.server.handle_transfer_complete(transfer_id, message.payload)
    
    async def handle_transfer_error(self, message: ControlMessage):
        """Handle transfer error"""
        transfer_id = message.payload.get('transfer_id')
        if transfer_id:
            self.pending_transfers.discard(transfer_id)
            await self.server.handle_transfer_error(transfer_id, message.payload)
    
    async def handle_transfer_cancel(self, message: ControlMessage):
        """Handle transfer cancellation"""
        transfer_id = message.payload.get('transfer_id')
        if transfer_id:
            self.pending_transfers.discard(transfer_id)
            await self.server.handle_transfer_cancel(transfer_id, message.payload)
    
    async def handle_status_request(self, message: ControlMessage):
        """Handle status request"""
        status = {
            'node_id': self.server.node_id,
            'uptime_seconds': time.time() - self.server.start_time,
            'active_connections': len(self.server.clients),
            'active_transfers': len(self.server.active_transfers),
            'total_transfers': self.server.total_transfers,
            'capabilities': asdict(self.server.capabilities)
        }
        
        response = ControlMessage(
            type=MessageType.STATUS_RESPONSE,
            sender=self.server.node_id,
            payload=status
        )
        await self.send_message(response)
    
    async def handle_node_list_request(self, message: ControlMessage):
        """Handle node list request"""
        nodes = []
        for client_id, client_handler in self.server.clients.items():
            if client_handler.capabilities:
                nodes.append({
                    'node_id': client_id,
                    'address': client_handler.client_address,
                    'capabilities': asdict(client_handler.capabilities),
                    'connected_time': time.time() - client_handler.last_heartbeat,
                    'pending_transfers': len(client_handler.pending_transfers)
                })
        
        response = ControlMessage(
            type=MessageType.NODE_LIST_RESPONSE,
            sender=self.server.node_id,
            payload={'nodes': nodes}
        )
        await self.send_message(response)
    
    async def handle_goodbye(self, message: ControlMessage):
        """Handle goodbye message"""
        logger.info(f"Client {self.client_id} sent goodbye")
        self.connected = False
    
    async def cleanup(self):
        """Cleanup client connection"""
        try:
            if self.client_id:
                self.server.unregister_client(self.client_id)
                logger.info(f"Client {self.client_id} disconnected")
            
            self.writer.close()
            await self.writer.wait_closed()
            
        except Exception as e:
            logger.error(f"Error during client cleanup: {e}")

class ControlPlaneServer:
    """TCP server for control plane coordination"""
    
    def __init__(self, node_id: str, host: str = '0.0.0.0', port: int = 5555):
        self.node_id = node_id
        self.host = host
        self.port = port
        
        # Server state
        self.server: Optional[asyncio.Server] = None
        self.clients: Dict[str, ClientHandler] = {}
        self.active_transfers: Dict[str, TransferRequest] = {}
        self.start_time = time.time()
        self.total_transfers = 0
        
        # Configuration
        self.max_concurrent_transfers = 100
        self.heartbeat_timeout = 60.0  # seconds
        self.message_timeout = 30.0    # seconds
        self.max_message_size = 1024 * 1024  # 1MB
        
        # Callbacks for transfer events
        self.transfer_callbacks: Dict[str, List[Callable]] = {
            'request': [],
            'ready': [],
            'start': [],
            'complete': [],
            'error': [],
            'cancel': []
        }
        
        # Server capabilities
        self.capabilities = NodeCapabilities(
            node_id=node_id,
            gpu_count=1,  # Will be updated based on actual hardware
            max_transfer_size=10 * 1024 * 1024 * 1024,  # 10GB
            supported_dtypes=['float32', 'float16', 'int32', 'int64', 'uint8'],
            network_bandwidth_gbps=100.0,
            memory_bandwidth_gbps=900.0,
            features=['fragmentation', 'reliability', 'compression', 'heartbeat']
        )
        
        # Background tasks
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        # Server serving task (if we choose to run serve_forever in background)
        self.server_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the control plane server (non-blocking)."""
        try:
            # Create server
            self.server = await asyncio.start_server(
                self.handle_client,
                self.host,
                self.port
            )

            # Begin accepting connections without blocking this coroutine
            await self.server.start_serving()

            # Start background tasks
            self.heartbeat_task = asyncio.create_task(self.heartbeat_loop())
            self.cleanup_task = asyncio.create_task(self.cleanup_loop())

            addr = self.server.sockets[0].getsockname()
            logger.info(f"Control plane server started on {addr[0]}:{addr[1]}")

        except Exception as e:
            logger.error(f"Failed to start control plane server: {e}")
            raise
    
    async def stop(self):
        """Stop the control plane server"""
        logger.info("Stopping control plane server...")
        
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Close all client connections
        for client in list(self.clients.values()):
            client.connected = False
            try:
                client.writer.close()
                await client.writer.wait_closed()
            except:
                pass
        
        # Stop server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("Control plane server stopped")

    def get_capabilities(self) -> NodeCapabilities:
        """Return server capabilities (compat with tests)."""
        return self.capabilities
    
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle new client connection"""
        client_handler = ClientHandler(reader, writer, self)
        await client_handler.handle_connection()
    
    def register_client(self, client_id: str, handler: ClientHandler):
        """Register a new client"""
        self.clients[client_id] = handler
        logger.info(f"Registered client: {client_id}")
    
    def unregister_client(self, client_id: str):
        """Unregister a client"""
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"Unregistered client: {client_id}")
    
    async def process_transfer_request(self, request: TransferRequest, 
                                     client: ClientHandler) -> TransferResponse:
        """Process a transfer request"""
        try:
            # Validate request
            if request.size > self.capabilities.max_transfer_size:
                return TransferResponse(
                    transfer_id=request.transfer_id,
                    accepted=False,
                    reason=f"Transfer size {request.size} exceeds maximum {self.capabilities.max_transfer_size}"
                )
            
            if request.dtype not in self.capabilities.supported_dtypes:
                return TransferResponse(
                    transfer_id=request.transfer_id,
                    accepted=False,
                    reason=f"Unsupported dtype: {request.dtype}"
                )
            
            if len(self.active_transfers) >= self.max_concurrent_transfers:
                return TransferResponse(
                    transfer_id=request.transfer_id,
                    accepted=False,
                    reason="Maximum concurrent transfers reached"
                )
            
            # Estimate transfer time (simple model)
            estimated_time = request.size / (self.capabilities.network_bandwidth_gbps * 1024**3 / 8)
            
            # Accept transfer
            self.active_transfers[request.transfer_id] = request
            self.total_transfers += 1
            
            # Call request callbacks
            for callback in self.transfer_callbacks['request']:
                try:
                    await callback(request)
                except Exception as e:
                    logger.error(f"Transfer request callback error: {e}")
            
            return TransferResponse(
                transfer_id=request.transfer_id,
                accepted=True,
                reason="Transfer accepted",
                estimated_time_seconds=estimated_time,
                allocated_gpu=request.target_gpu,
                data_port=5556  # UDP port for data transfer
            )
            
        except Exception as e:
            logger.error(f"Error processing transfer request: {e}")
            return TransferResponse(
                transfer_id=request.transfer_id,
                accepted=False,
                reason=f"Processing error: {e}"
            )
    
    async def handle_transfer_ready(self, transfer_id: str, payload: Dict[str, Any]):
        """Handle transfer ready notification"""
        logger.info(f"Transfer {transfer_id} is ready")
        for callback in self.transfer_callbacks['ready']:
            try:
                await callback(transfer_id, payload)
            except Exception as e:
                logger.error(f"Transfer ready callback error: {e}")
    
    async def handle_transfer_start(self, transfer_id: str, payload: Dict[str, Any]):
        """Handle transfer start notification"""
        logger.info(f"Transfer {transfer_id} started")
        for callback in self.transfer_callbacks['start']:
            try:
                await callback(transfer_id, payload)
            except Exception as e:
                logger.error(f"Transfer start callback error: {e}")
    
    async def handle_transfer_complete(self, transfer_id: str, payload: Dict[str, Any]):
        """Handle transfer completion"""
        if transfer_id in self.active_transfers:
            del self.active_transfers[transfer_id]
        
        logger.info(f"Transfer {transfer_id} completed")
        for callback in self.transfer_callbacks['complete']:
            try:
                await callback(transfer_id, payload)
            except Exception as e:
                logger.error(f"Transfer complete callback error: {e}")
    
    async def handle_transfer_error(self, transfer_id: str, payload: Dict[str, Any]):
        """Handle transfer error"""
        if transfer_id in self.active_transfers:
            del self.active_transfers[transfer_id]
        
        error_msg = payload.get('reason', 'Unknown error')
        logger.error(f"Transfer {transfer_id} failed: {error_msg}")
        
        for callback in self.transfer_callbacks['error']:
            try:
                await callback(transfer_id, payload)
            except Exception as e:
                logger.error(f"Transfer error callback error: {e}")
    
    async def handle_transfer_cancel(self, transfer_id: str, payload: Dict[str, Any]):
        """Handle transfer cancellation"""
        if transfer_id in self.active_transfers:
            del self.active_transfers[transfer_id]
        
        logger.info(f"Transfer {transfer_id} cancelled")
        for callback in self.transfer_callbacks['cancel']:
            try:
                await callback(transfer_id, payload)
            except Exception as e:
                logger.error(f"Transfer cancel callback error: {e}")
    
    def add_transfer_callback(self, event: str, callback: Callable):
        """Add a callback for transfer events"""
        if event in self.transfer_callbacks:
            self.transfer_callbacks[event].append(callback)
        else:
            raise ValueError(f"Unknown event type: {event}")
    
    async def heartbeat_loop(self):
        """Background heartbeat loop"""
        while True:
            try:
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                
                # Send heartbeat to all connected clients
                heartbeat_msg = ControlMessage(
                    type=MessageType.HEARTBEAT,
                    sender=self.node_id,
                    payload={'timestamp': time.time()}
                )
                
                for client in list(self.clients.values()):
                    try:
                        await client.send_message(heartbeat_msg)
                    except Exception as e:
                        logger.error(f"Failed to send heartbeat to {client.client_id}: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
    
    async def cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                
                current_time = time.time()
                
                # Check for timed out clients
                disconnected_clients = []
                for client_id, client in self.clients.items():
                    if current_time - client.last_heartbeat > self.heartbeat_timeout:
                        disconnected_clients.append(client_id)
                
                # Remove timed out clients
                for client_id in disconnected_clients:
                    logger.warning(f"Removing timed out client: {client_id}")
                    client = self.clients[client_id]
                    client.connected = False
                    self.unregister_client(client_id)
                
                # Check for stale transfers
                stale_transfers = []
                for transfer_id, transfer in self.active_transfers.items():
                    # Simple timeout check - in practice, this would be more sophisticated
                    if current_time - transfer.timeout_seconds > 3600:  # 1 hour default timeout
                        stale_transfers.append(transfer_id)
                
                # Remove stale transfers
                for transfer_id in stale_transfers:
                    logger.warning(f"Removing stale transfer: {transfer_id}")
                    del self.active_transfers[transfer_id]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

# Global server instance
_control_server: Optional[ControlPlaneServer] = None

def get_control_server(node_id: str = None) -> ControlPlaneServer:
    """Get global control server instance"""
    global _control_server
    if _control_server is None:
        if node_id is None:
            node_id = f"genie-node-{socket.gethostname()}"
        _control_server = ControlPlaneServer(node_id)
    return _control_server

async def start_control_server(node_id: str = None, host: str = '0.0.0.0', port: int = 5555):
    """Start the global control server"""
    server = get_control_server(node_id)
    server.host = host
    server.port = port
    await server.start()

async def stop_control_server():
    """Stop the global control server"""
    global _control_server
    if _control_server:
        await _control_server.stop()
        _control_server = None
