"""
TCP Data Server for Genie Zero-Copy Transport

This module implements a TCP server that receives tensor data sent by PinnedTCPTransport.
It works alongside the ControlPlaneServer for coordination and the PinnedTCPTransport for sending.
"""

import asyncio
import logging
import struct
import threading
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ReceivedTensor:
    """Information about a received tensor"""
    tensor_id: str
    size: int
    data: bytes
    received_at: float
    source_address: tuple

class TCPDataServer:
    """TCP server for receiving tensor data"""

    def __init__(self, host: str = '0.0.0.0', port: int = 50555):
        self.host = host
        self.port = port
        self.server: Optional[asyncio.Server] = None
        self.running = False

        # Received tensors storage
        self.received_tensors: Dict[str, ReceivedTensor] = {}

        # Callbacks for received tensors
        self.tensor_callbacks: list[Callable[[ReceivedTensor], None]] = []

    def add_tensor_callback(self, callback: Callable[[ReceivedTensor], None]):
        """Add callback for received tensors"""
        self.tensor_callbacks.append(callback)

    async def start(self):
        """Start the TCP data server"""
        if self.running:
            return

        try:
            self.server = await asyncio.start_server(
                self._handle_client, self.host, self.port
            )
            self.running = True
            logger.info(f"TCP data server started on {self.host}:{self.port}")

        except Exception as e:
            logger.error(f"Failed to start TCP data server: {e}")
            raise

    async def stop(self):
        """Stop the TCP data server"""
        if not self.running:
            return

        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("TCP data server stopped")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a client connection"""
        client_address = writer.get_extra_info('peername')
        logger.debug(f"Client connected from {client_address}")

        try:
            while self.running:
                # Read tensor_id length (4 bytes, big-endian)
                tensor_id_len_data = await reader.readexactly(4)
                if not tensor_id_len_data:
                    break

                tensor_id_len = struct.unpack('>I', tensor_id_len_data)[0]

                # Read tensor_id
                tensor_id_data = await reader.readexactly(tensor_id_len)
                tensor_id = tensor_id_data.decode('utf-8')

                # Read size (8 bytes, big-endian)
                size_data = await reader.readexactly(8)
                size = struct.unpack('>Q', size_data)[0]

                # Read tensor data
                data = await reader.readexactly(size)

                # Store received tensor
                import time
                received_tensor = ReceivedTensor(
                    tensor_id=tensor_id,
                    size=size,
                    data=data,
                    received_at=time.time(),
                    source_address=client_address
                )

                self.received_tensors[tensor_id] = received_tensor

                # Call callbacks
                for callback in self.tensor_callbacks:
                    try:
                        callback(received_tensor)
                    except Exception as e:
                        logger.error(f"Error in tensor callback: {e}")

                logger.info(f"Received tensor {tensor_id} ({size} bytes) from {client_address}")

        except asyncio.IncompleteReadError:
            logger.debug(f"Connection closed by {client_address}")
        except Exception as e:
            logger.error(f"Error handling client {client_address}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    def get_received_tensor(self, tensor_id: str) -> Optional[ReceivedTensor]:
        """Get a received tensor by ID"""
        return self.received_tensors.get(tensor_id)

    def list_received_tensors(self) -> list[str]:
        """List all received tensor IDs"""
        return list(self.received_tensors.keys())

    def clear_received_tensors(self):
        """Clear all received tensors"""
        self.received_tensors.clear()

# Global instance for testing
_tcp_data_server: Optional[TCPDataServer] = None

def get_tcp_data_server(host: str = '127.0.0.1', port: int = 50555) -> TCPDataServer:
    """Get or create global TCP data server instance"""
    global _tcp_data_server
    if _tcp_data_server is None:
        _tcp_data_server = TCPDataServer(host, port)
    return _tcp_data_server

async def start_tcp_data_server(host: str = '127.0.0.1', port: int = 50555) -> TCPDataServer:
    """Start the global TCP data server"""
    server = get_tcp_data_server(host, port)
    await server.start()
    return server

async def stop_tcp_data_server():
    """Stop the global TCP data server"""
    global _tcp_data_server
    if _tcp_data_server:
        await _tcp_data_server.stop()
        _tcp_data_server = None
