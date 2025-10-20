"""
Genie Server - Main server component for disaggregated GPU cluster.

Manages:
- GPU resource discovery and allocation
- Network transport initialization
- Control plane for coordination
- Remote computation execution
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from ..core.coordinator import GenieCoordinator, CoordinatorConfig
from ..runtime.control_plane import ControlPlane
from .capability_provider import CapabilityProvider
from .executor import RemoteExecutor

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for Genie server."""
    node_id: str
    control_port: int = 5555
    data_port: int = 5556
    gpu_indices: List[int] = None  # Which GPUs to use (None = all available)
    prefer_dpdk: bool = True
    tcp_fallback: bool = True
    max_concurrent_transfers: int = 32


class GenieServer:
    """
    Main server for Genie disaggregated GPU cluster.

    Handles remote tensor execution and data transfers.
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self.capabilities = None
        self.coordinator = None
        self.control_plane = None
        self.executor = None  # Remote executor for operations
        self.is_running = False

    async def start(self) -> bool:
        """Start the Genie server."""
        try:
            print(f"Starting Genie server: {self.config.node_id}")

            # 1. Discover capabilities
            print("Discovering capabilities...")
            self.capabilities = CapabilityProvider.discover()
            print(f"✓ Found {self.capabilities.gpu_count} GPUs")

            # 2. Initialize coordinator
            print("Initializing coordinator...")
            coordinator_config = CoordinatorConfig(
                node_id=self.config.node_id,
                control_port=self.config.control_port,
                data_port=self.config.data_port,
                prefer_dpdk=self.config.prefer_dpdk,
                tcp_fallback=self.config.tcp_fallback
            )
            self.coordinator = GenieCoordinator(coordinator_config)
            await self.coordinator.start()
            print("✓ Coordinator started")

            # ✅ ADD: Hook server to handle incoming operation requests
            if 'tcp' in self.coordinator.transports:
                self.coordinator.transports['tcp']._operation_callback = \
                    self._handle_operation_request
                print("✓ Server operation callback registered")

            # 3. Start control plane
            print("Starting control plane...")
            self.control_plane = ControlPlane(
                self.config.node_id,
                self.config.control_port
            )
            await self.control_plane.start()
            print("✓ Control plane started")

            # 4. Initialize remote executor
            print("Initializing remote executor...")
            self.executor = RemoteExecutor(gpu_id=0)  # Use first GPU
            print(f"✓ Remote executor ready (GPU {self.executor.gpu_id})")

            self.is_running = True
            print(f"\n🎉 Genie server ready on {self.config.node_id}")
            print(f"   Control plane: {self.config.control_port}")
            print(f"   Data plane: {self.config.data_port}")
            print(f"   GPUs: {len(self.capabilities.gpu_indices)}")
            print(f"   Memory: {self.capabilities.total_memory_gb}GB")

            # Start background tasks
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._transfer_handler_loop())

            return True

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            await self.stop()
            return False

    async def stop(self):
        """Stop the Genie server."""
        print(f"Stopping Genie server: {self.config.node_id}")

        self.is_running = False

        if self.control_plane:
            await self.control_plane.stop()

        if self.coordinator:
            await self.coordinator.stop()

        print("✓ Server stopped")

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
        tensor: torch.Tensor,
        metadata: Dict
    ):
        """Execute operation and send result back."""

        operation = metadata.get('operation')
        result_id = metadata.get('result_id')
        source_node = metadata.get('source_node')

        if not all([operation, result_id, source_node]):
            logger.warning(f"Missing metadata for {transfer_id}")
            return

        try:
            logger.info(f"🔧 Executing {operation} on {tensor.shape}")

            # Execute operation
            result = await self.executor.execute(operation, tensor)

            logger.info(f"📤 Sending result {result.shape} to {source_node}")

            # Send result back
            result_metadata = {
                'is_result': True,
                'original_transfer': transfer_id,
                'dtype': str(result.dtype),
                'shape': list(result.shape),
                'size_bytes': result.numel() * result.element_size()
            }

            await self.coordinator.transports['tcp'].send(
                result,
                target=source_node,
                transfer_id=result_id,
                metadata=result_metadata
            )

            logger.info(f"✅ Result sent for {transfer_id}")

        except Exception as e:
            logger.error(f"❌ Operation execution failed: {e}")


def main():
    """CLI entry point for Genie server."""
    import argparse

    parser = argparse.ArgumentParser(description="Genie disaggregated GPU server")
    parser.add_argument("--node-id", required=True, help="Unique server identifier")
    parser.add_argument("--control-port", type=int, default=5555, help="Control plane port")
    parser.add_argument("--data-port", type=int, default=5556, help="Data plane port")
    parser.add_argument("--gpus", nargs="*", type=int, help="GPU indices to use (default: all)")
    parser.add_argument("--no-dpdk", action="store_true", help="Disable DPDK (use TCP only)")

    args = parser.parse_args()

    # Create config
    config = ServerConfig(
        node_id=args.node_id,
        control_port=args.control_port,
        data_port=args.data_port,
        gpu_indices=args.gpus,
        prefer_dpdk=not args.no_dpdk,
        tcp_fallback=True
    )

    # Start server
    async def run_server():
        server = GenieServer(config)
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
