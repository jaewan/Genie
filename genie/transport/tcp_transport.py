"""
TCP transport for Genie.

Provides reliable fallback when DPDK is not available.
Uses length-prefixed framing for tensor transfers.

Performance: ~10 Gbps (limited by TCP, not hardware)
"""

import asyncio
import struct
import logging
import json
from typing import Dict, Any, Optional, Callable
import torch
import numpy as np

from .base import Transport

logger = logging.getLogger(__name__)


class TCPTransport(Transport):
    """TCP transport implementation with length-prefixed framing."""

    @property
    def name(self) -> str:
        return "TCP"

    def __init__(self, config):
        self.config = config
        self.data_port = getattr(config, 'data_port', 5556)
        self.server = None

        # Receive handlers
        self._receive_handlers: Dict[str, asyncio.Future] = {}
        self._received_tensors: Dict[str, torch.Tensor] = {}

        # ✅ ADD: Result callback
        self._result_callback: Optional[Callable] = None

        # ✅ ADD: Operation callback
        self._operation_callback: Optional[Callable] = None

    async def initialize(self) -> bool:
        """Start TCP receiver (server-side)."""
        try:
            # Start TCP server for receiving tensors
            self.server = await asyncio.start_server(
                self._handle_connection,
                '0.0.0.0',
                self.data_port
            )

            addrs = ', '.join(str(sock.getsockname()) for sock in self.server.sockets)
            logger.info(f"TCP transport listening on {addrs}")
            return True

        except Exception as e:
            logger.error(f"TCP transport initialization failed: {e}")
            return False

    async def send(
        self,
        tensor: torch.Tensor,
        target: str,
        transfer_id: str,
        metadata: Dict
    ) -> bool:
        """
        Send tensor to target via TCP.

        Protocol:
            [4 bytes: transfer_id length]
            [N bytes: transfer_id (UTF-8)]
            [4 bytes: metadata length]
            [N bytes: metadata (JSON)]
            [8 bytes: tensor size in bytes]
            [N bytes: tensor data]
        """
        try:
            # Parse target (host:port)
            if ':' in target:
                host, port_str = target.rsplit(':', 1)
                port = int(port_str)
            else:
                host = target
                port = self.data_port

            logger.info(f"TCP send: {tensor.shape} to {host}:{port}")

            # Connect
            reader, writer = await asyncio.open_connection(host, port)

            # Move tensor to CPU if needed
            cpu_tensor = tensor.cpu() if tensor.is_cuda else tensor
            cpu_tensor = cpu_tensor.contiguous()

            # Serialize tensor to bytes
            tensor_bytes = cpu_tensor.numpy().tobytes()

            # Serialize metadata
            metadata_json = json.dumps(metadata).encode('utf-8')

            # Send transfer_id
            transfer_id_bytes = transfer_id.encode('utf-8')
            writer.write(struct.pack('>I', len(transfer_id_bytes)))
            writer.write(transfer_id_bytes)

            # Send metadata
            writer.write(struct.pack('>I', len(metadata_json)))
            writer.write(metadata_json)

            # Send tensor size
            writer.write(struct.pack('>Q', len(tensor_bytes)))

            # Send tensor data (in chunks to avoid memory pressure)
            chunk_size = 1024 * 1024  # 1MB chunks
            offset = 0
            while offset < len(tensor_bytes):
                chunk = tensor_bytes[offset:offset + chunk_size]
                writer.write(chunk)
                await writer.drain()
                offset += len(chunk)

            # Close connection
            writer.close()
            await writer.wait_closed()

            logger.info(f"TCP send complete: {len(tensor_bytes)} bytes")
            return True

        except Exception as e:
            logger.error(f"TCP send failed: {e}")
            return False

    async def receive(
        self,
        transfer_id: str,
        metadata: Dict
    ) -> torch.Tensor:
        """
        Wait for tensor to be received.

        This is called AFTER negotiate() has set up the receiver.
        """
        # Create future
        future = asyncio.Future()
        self._receive_handlers[transfer_id] = future

        try:
            # Wait for tensor (timeout 30s)
            await asyncio.wait_for(future, timeout=30.0)

            # Get tensor
            tensor = self._received_tensors.pop(transfer_id)
            return tensor

        except asyncio.TimeoutError:
            raise RuntimeError(f"Timeout waiting for {transfer_id}")
        finally:
            self._receive_handlers.pop(transfer_id, None)

    def is_available(self) -> bool:
        """TCP is always available."""
        return True

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter
    ):
        """Handle incoming tensor transfer."""
        addr = writer.get_extra_info('peername')
        logger.info(f"TCP connection from {addr}")

        try:
            # Read transfer_id
            transfer_id_len_bytes = await reader.readexactly(4)
            transfer_id_len = struct.unpack('>I', transfer_id_len_bytes)[0]
            transfer_id_bytes = await reader.readexactly(transfer_id_len)
            transfer_id = transfer_id_bytes.decode('utf-8')

            # Read metadata
            metadata_len_bytes = await reader.readexactly(4)
            metadata_len = struct.unpack('>I', metadata_len_bytes)[0]
            metadata_bytes = await reader.readexactly(metadata_len)

            metadata = json.loads(metadata_bytes.decode('utf-8'))

            # Read tensor size
            size_bytes = await reader.readexactly(8)
            tensor_size = struct.unpack('>Q', size_bytes)[0]

            logger.info(f"Receiving tensor: {transfer_id}, {tensor_size} bytes")

            # Read tensor data
            tensor_bytes = await reader.readexactly(tensor_size)

            # Reconstruct tensor
            shape = metadata['shape']
            dtype_str = metadata['dtype']

            # Map PyTorch dtype string to NumPy dtype
            dtype_map = {
                'torch.float32': np.float32,
                'torch.float16': np.float16,
                'torch.int64': np.int64,
                'torch.int32': np.int32,
                'torch.float64': np.float64,
            }
            np_dtype = dtype_map.get(dtype_str, np.float32)

            # Create numpy array
            np_array = np.frombuffer(tensor_bytes, dtype=np_dtype)
            np_array = np_array.reshape(shape)

            # Convert to PyTorch tensor
            tensor = torch.from_numpy(np_array.copy())

            # ✅ CHECK: Is this a result?
            if metadata.get('is_result', False):
                # This is a result coming back
                if self._result_callback:
                    await self._result_callback(transfer_id, tensor)
                else:
                    logger.warning(f"Result received but no callback: {transfer_id}")
            # ✅ CHECK: Is this an operation request?
            elif metadata.get('operation'):
                # This needs execution - call server callback
                if self._operation_callback:
                    await self._operation_callback(transfer_id, tensor, metadata)
                else:
                    logger.warning(f"Operation request but no callback: {transfer_id}")
            else:
                # This is a normal transfer
                self._received_tensors[transfer_id] = tensor

                # Signal completion
                if transfer_id in self._receive_handlers:
                    self._receive_handlers[transfer_id].set_result(True)

            logger.info(f"Tensor received: {tensor.shape}")

        except Exception as e:
            logger.error(f"Error receiving tensor: {e}")
        finally:
            writer.close()
            await writer.wait_closed()