"""
Djinn Server - Main server component for disaggregated GPU cluster.

Manages:
- GPU resource discovery and allocation
- Network transport initialization
- Control plane for coordination
- Remote computation execution
"""

import asyncio
import logging
import os
import struct
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from ..core.coordinator import DjinnCoordinator, CoordinatorConfig
from .optimization_executor import OptimizationExecutor
from .capability_provider import CapabilityProvider
from ..core.metadata_types import ResultMetadata, ErrorMetadata, create_result_metadata, create_error_metadata

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for Djinn server."""
    node_id: Optional[str] = None  # Use centralized config if None
    control_port: Optional[int] = None  # Use centralized config if None
    data_port: Optional[int] = None     # Use centralized config if None
    gpu_indices: Optional[List[int]] = None  # Which GPUs to use (None = all available)
    prefer_dpdk: Optional[bool] = None  # Use centralized config if None
    tcp_fallback: Optional[bool] = None  # Use centralized config if None
    max_concurrent_transfers: Optional[int] = None  # Use centralized config if None

    def get_config(self):
        """Get configuration using centralized config as fallback."""
        from ..config import get_config
        config = get_config()

        return {
            'node_id': self.node_id or config.server.node_id,
            'control_port': self.control_port or config.network.control_port,
            'data_port': self.data_port or config.network.data_port,
            'gpu_indices': self.gpu_indices or [],
            'prefer_dpdk': self.prefer_dpdk if self.prefer_dpdk is not None else config.network.prefer_dpdk,
            'tcp_fallback': self.tcp_fallback if self.tcp_fallback is not None else config.network.tcp_fallback,
            'max_concurrent_transfers': self.max_concurrent_transfers or config.server.max_concurrent_transfers,
        }


class DjinnServer:
    """
    Main server for Djinn disaggregated GPU cluster.

    Handles remote tensor execution and data transfers.
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        # Get centralized configuration
        from ..config import get_config
        self._central_config = get_config()

        # Resolve configuration values
        resolved_config = config.get_config()
        self.node_id = resolved_config['node_id']
        self.control_port = resolved_config['control_port']
        self.data_port = resolved_config['data_port']
        self.max_concurrent_transfers = resolved_config['max_concurrent_transfers']

        self.capabilities = None
        self.coordinator = None
        self.control_plane = None
        self.executor = None  # Remote executor for operations
        self.is_running = False
        self.tcp_server = None  # Server's own TCP server
        self.tcp_transport = None  # TCP transport for sending results

    async def start(self) -> bool:
        """Start the Djinn server."""
        try:
            logger.info(f"Starting Djinn server: {self.node_id}")

            # 1. Discover capabilities
            logger.info("Discovering capabilities...")
            self.capabilities = CapabilityProvider.discover()
            logger.info(f"✓ Found {self.capabilities.gpu_count} GPUs")

            # 2. Start TCP server for listening to incoming operation requests
            logger.info("Starting TCP server for operation requests...")
            import asyncio
            self.tcp_server = await asyncio.start_server(
                self._handle_connection,
                '0.0.0.0',
                self.data_port
            )
            logger.info(f"✓ TCP server listening on port {self.data_port}")

            # Set up transport for handling operation requests and sending results
            from .transport.tcp_transport import TCPTransport
            from ..core.coordinator import CoordinatorConfig

            # Create transport config for server operations
            server_config = CoordinatorConfig(
                node_id=f"{self.node_id}-server",
                control_port=self.control_port,
                data_port=self.data_port,
                prefer_dpdk=False,
                tcp_fallback=True,
                is_server=True
            )

            # Initialize transport that can handle both incoming requests and outgoing results
            self.transport = TCPTransport(server_config)
            await self.transport.initialize()

            # ✅ WIRE: Connect operation callback to transport
            self.transport._operation_callback = self._handle_operation_request

            # Also set up result transport for sending responses back to clients
            result_config = CoordinatorConfig(
                node_id=f"{self.node_id}-result-sender",
                control_port=self.control_port + self._central_config.network.result_port_offset,
                data_port=self.data_port + self._central_config.network.result_port_offset,
                prefer_dpdk=False,
                tcp_fallback=True,
                is_server=False
            )

            self.result_transport = TCPTransport(result_config)
            # Don't initialize as server - this is for sending results only
            logger.info("✓ Server transport with operation callback wired")
            logger.info("✓ Result transport configured")

            # Server doesn't need control plane for basic operation
            # Control plane is handled by the coordinator if needed

            # 4. Initialize optimization executor (with tensor registry and fusion compiler)
            logger.info("Initializing optimization executor...")
            self.executor = OptimizationExecutor(gpu_id=0)  # Use first GPU
            logger.info(f"✓ Optimization executor ready (GPU {self.executor.gpu_id}, Registry: enabled, Fusion: enabled)")

            self.is_running = True
            logger.info(f"\n🎉 Djinn server ready on {self.node_id}")
            logger.info(f"   Control plane: {self.control_port}")
            logger.info(f"   Data plane: {self.data_port}")
            logger.info(f"   GPUs: {len(self.capabilities.gpu_indices)}")
            logger.info(f"   Memory: {self.capabilities.total_memory_gb}GB")

            # Start background tasks
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._transfer_handler_loop())

            return True

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            await self.stop()
            return False

    async def stop(self):
        """Stop the Djinn server."""
        logger.info(f"Stopping Djinn server: {self.node_id}")

        self.is_running = False

        # Server doesn't use control plane for basic operation
        # if self.control_plane:
        #     await self.control_plane.stop()

        # Server doesn't use coordinator for basic operation
        # if self.coordinator:
        #     await self.coordinator.stop()

        if hasattr(self, 'tcp_server') and self.tcp_server:
            self.tcp_server.close()
            await self.tcp_server.wait_closed()

        if hasattr(self, 'transport') and self.transport:
            if hasattr(self.transport, 'stop'):
                await self.transport.stop()

        if hasattr(self, 'result_transport') and self.result_transport:
            if self.result_transport.server:
                self.result_transport.server.close()
                await self.result_transport.server.wait_closed()

        logger.info("✓ Server stopped")

    async def _handle_connection(self, reader, writer):
        """Handle incoming TCP connections for operation requests."""
        addr = writer.get_extra_info('peername')
        logger.info(f"TCP CONNECTION: New connection from {addr}")

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
            logger.info(f"DIRECT TCP: Received metadata: {metadata}")
            logger.info(f"DIRECT TCP: Client port in metadata: {metadata.get('client_port')}")
            logger.info(f"DIRECT TCP: Source node in metadata: {metadata.get('source_node')}")

            # Check if this is multi-tensor or single-tensor message
            peek_data = await reader.readexactly(1)
            next_byte_val = struct.unpack('B', peek_data)[0]

            # Multi-tensor: next byte is number of tensors
            is_multi_tensor = 'num_inputs' in metadata and next_byte_val <= 10

            if is_multi_tensor:
                # Multi-tensor protocol
                num_tensors = next_byte_val
                logger.info(f"Receiving {num_tensors} tensors for {transfer_id}")

                tensors = []
                for i in range(num_tensors):
                    # Read tensor size
                    size_bytes = await reader.readexactly(8)
                    tensor_size = struct.unpack('>Q', size_bytes)[0]

                    # Read tensor data
                    tensor_bytes = await reader.readexactly(tensor_size)

                    # Reconstruct tensor
                    shape = metadata['input_shapes'][i]
                    dtype_str = metadata['input_dtypes'][i]

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
                    tensors.append(tensor)

                    logger.debug(f"  Received tensor {i}: {shape}")

                # Handle operation request
                await self._handle_operation_request(transfer_id, tensors, metadata)

            else:
                # Single-tensor protocol (fallback)
                # Re-interpret first byte + next 7 bytes as tensor size
                size_bytes_full = peek_data + await reader.readexactly(7)
                tensor_size = struct.unpack('>Q', size_bytes_full)[0]

                logger.info(f"Receiving single tensor: {transfer_id}, {tensor_size} bytes")

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

                # Handle operation request (single tensor)
                await self._handle_operation_request(transfer_id, tensor, metadata)

        except Exception as e:
            logger.error(f"Error handling connection from {addr}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to connected clients."""
        while self.is_running:
            try:
                # Send heartbeat to all connected clients
                await asyncio.sleep(30)  # Every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def _transfer_handler_loop(self):
        """Handle incoming transfer requests and execute operations."""
        while self.is_running:
            try:
                # Check for completed transfers from coordinator
                # In a real implementation, this would integrate with control_plane
                # to get notified of completed transfers

                # For now, we'll poll the coordinator for completed transfers
                # This is a simplified implementation - in practice you'd use callbacks
                if hasattr(self.coordinator, 'active_transfers'):
                    for transfer_id, transfer in list(self.coordinator.active_transfers.items()):
                        # Check if transfer is complete (simplified check)
                        # In practice, this would be handled by proper callbacks

                        # For demo purposes, we'll execute a simple operation on received tensors
                        try:
                            # Get received tensor (this would be implemented properly in coordinator)
                            # tensor = await self.coordinator.receive_tensor(transfer_id, transfer.metadata)

                            # Get operation from metadata
                            operation = transfer.metadata.get('operation', 'aten::relu')

                            # Execute operation (simplified - would need actual tensor)
                            print(f"Executing {operation} for transfer {transfer_id}")

                            # In a real implementation, you'd:
                            # 1. Receive the actual tensor
                            # 2. Execute the operation using the executor
                            # 3. Send the result back to the client

                        except Exception as e:
                            logger.error(f"Error executing operation for {transfer_id}: {e}")

                await asyncio.sleep(0.1)  # Small delay to avoid busy waiting
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Transfer handler error: {e}")

    async def _handle_operation_request(
        self,
        transfer_id: str,
        tensor_or_tensors,  # Can be single tensor or list of tensors
        metadata: Dict
    ):
        """
        Execute operation on received tensor(s) and send result back.

        This is called when a client sends a remote operation request.

        Args:
            transfer_id: Unique transfer identifier
            tensor_or_tensors: Single tensor or list of tensors from multi-tensor protocol
            metadata: Operation metadata
        """
        import traceback
        import numpy as np

        logger.info(f"OPERATION REQUEST: transfer_id={transfer_id}")
        logger.info(f"OPERATION REQUEST: metadata={metadata}")
        logger.info(f"OPERATION REQUEST: metadata keys={list(metadata.keys())}")

        operation = metadata.get('operation')
        result_id = metadata.get('result_id')
        source_node = metadata.get('source_node', 'unknown')

        # Use client_port if provided (more reliable than source_node)
        client_port = metadata.get('client_port')
        logger.debug(f"Received metadata keys: {list(metadata.keys())}")
        logger.debug(f"Original source_node: {source_node}")
        logger.debug(f"Client port from metadata: {client_port}")

        # Always try to use client_port if available, otherwise use hardcoded for testing
        client_port = metadata.get('client_port')
        logger.info(f"OPERATION REQUEST: transfer_id={transfer_id}")
        logger.info(f"OPERATION REQUEST: metadata={metadata}")
        logger.info(f"OPERATION REQUEST: metadata keys={list(metadata.keys())}")
        logger.info(f"OPERATION REQUEST: original source_node={source_node}")
        logger.info(f"OPERATION REQUEST: client_port={client_port}")

        # Always use the correct client port for localhost connections
        if ':' in source_node:
            client_ip = source_node.split(':')[0]
        else:
            client_ip = '127.0.0.1'  # localhost

        # Use client_port from metadata if available (preferred method)
        client_port = metadata.get('client_port')
        if client_port:
            source_node = f"{client_ip}:{client_port}"
            logger.info(f"OPERATION REQUEST: Using client_port {client_port} from metadata")
        else:
            # Fallback: use the same port as the server for localhost
            source_node = f"{client_ip}:{self.data_port}"
            logger.info(f"OPERATION REQUEST: No client_port in metadata, using server data port {self.data_port}")

        if not all([operation, result_id]):
            logger.warning(f"Missing critical metadata for {transfer_id}: op={operation}, result_id={result_id}")
            return

        try:
            # Handle both single and multi-tensor inputs
            if isinstance(tensor_or_tensors, list):
                tensors = tensor_or_tensors
                logger.info(f"🔧 Executing {operation} on {len(tensors)} tensors: {[t.shape for t in tensors]}")
            else:
                tensors = [tensor_or_tensors]
                logger.info(f"🔧 Executing {operation} on {tensor_or_tensors.shape}")

            # Create subgraph request for the executor
            subgraph_request = {
                'operations': [{
                    'op_id': 0,
                    'operation': operation,
                    'inputs': list(range(len(tensors))),  # Input indices
                    'kwargs': {}
                }],
                'output_id': 0,
                'semantic_metadata': {}  # Add basic semantic metadata
            }

            # Prepare input data dictionary
            input_data = {str(i): tensor for i, tensor in enumerate(tensors)}

            # Call executor with proper signature
            result, stats = await self.executor.execute(
                subgraph_request=subgraph_request,
                input_data=input_data,
                model_id=None,  # Single operation, no model context
                timeout=30.0
            )

            logger.info(f"📤 Sending result {result.shape} to {source_node}")

            # ✅ Send success result with result_id
            result_metadata = create_result_metadata(
                result_id=result_id,
                original_transfer=transfer_id,
                result=result
            ).to_dict()

            # ✅ FIX: Use client's listening port from metadata (not source_node)
            client_port = metadata.get('client_port')
            if client_port:
                # Get client IP from source_node or use localhost
                if ':' in source_node:
                    client_ip = source_node.split(':')[0]
                else:
                    client_ip = '127.0.0.1'  # localhost
                result_target = f"{client_ip}:{client_port}"
                logger.info(f"📤 Sending result {result.shape} to {result_target}")
            else:
                logger.error(f"Missing client_port in metadata for {transfer_id}")
                result_target = source_node
                logger.info(f"📤 Sending result {result.shape} to {result_target} (fallback)")

            # Send result back to client using the result transport
            success = await self.result_transport.send(
                result,
                target=result_target,  # ✅ Use correct target with client port
                transfer_id=result_id,
                metadata=result_metadata
            )

            if success:
                logger.info(f"✅ Result sent for {transfer_id}")
            else:
                logger.error(f"Failed to send result for {transfer_id}")

        except Exception as e:
            logger.error(f"❌ Operation execution failed: {e}")
            logger.error(traceback.format_exc())

            # ✅ Send error response to client
            error_metadata = create_error_metadata(
                result_id=result_id,
                original_transfer=transfer_id,
                error=e
            ).to_dict()

            # Send empty tensor with error metadata
            error_tensor = torch.zeros(0)

            try:
                # ✅ FIX: Use same client port routing for error responses
                client_port = metadata.get('client_port')
                if client_port:
                    if ':' in source_node:
                        client_ip = source_node.split(':')[0]
                    else:
                        client_ip = '127.0.0.1'
                    error_target = f"{client_ip}:{client_port}"
                else:
                    error_target = source_node

                await self.result_transport.send(
                    error_tensor,
                    target=error_target,  # ✅ Use correct target for errors too
                    transfer_id=result_id,
                    metadata=error_metadata
                )
                logger.info(f"📤 Error response sent for {transfer_id}")
            except Exception as send_error:
                logger.error(f"Failed to send error response: {send_error}")


def main():
    """CLI entry point for Djinn server."""
    import argparse

    parser = argparse.ArgumentParser(description="Djinn disaggregated GPU server")
    parser.add_argument("--node-id", required=True, help="Unique server identifier")
    parser.add_argument("--control-port", type=int, default=5555, help="Control plane port")
    parser.add_argument("--data-port", type=int, default=5556, help="Data plane port")
    parser.add_argument("--gpus", nargs="*", type=int, help="GPU indices to use (default: all)")
    parser.add_argument("--no-dpdk", action="store_true", help="Disable DPDK (use TCP only)")

    args = parser.parse_args()

    # Create config (can be None to use centralized defaults)
    config = ServerConfig(
        node_id=args.node_id if args.node_id else None,
        control_port=args.control_port if args.control_port != 5555 else None,
        data_port=args.data_port if args.data_port != 5556 else None,
        gpu_indices=args.gpus,
        prefer_dpdk=not args.no_dpdk if args.no_dpdk else None,
        tcp_fallback=True
    )

    # Start server
    async def run_server():
        server = DjinnServer(config)
        success = await server.start()
        if not success:
            exit(1)

        try:
            # Keep server running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await server.stop()

    # Run server
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
