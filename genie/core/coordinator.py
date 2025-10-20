"""
THE coordinator. All tensor transfers go through here.

Design principles:
1. Try DPDK first (if GPU + hardware available)
2. Automatic TCP fallback (always works)
3. Semantic metadata tracking
4. Simple > Complex
"""

import threading
from dataclasses import dataclass
from typing import Optional, Dict, Callable, Any
import asyncio
import uuid
import torch

# ==========================================
# SINGLETON PATTERN FOR COORDINATOR
# ==========================================

_coordinator_instance: Optional['GenieCoordinator'] = None
_coordinator_lock = threading.Lock()

def get_coordinator() -> 'GenieCoordinator':
    """Get global coordinator instance."""
    global _coordinator_instance
    if _coordinator_instance is None:
        raise RuntimeError(
            "Coordinator not initialized. "
            "Call GenieCoordinator(config).start() first."
        )
    return _coordinator_instance

def set_coordinator(coordinator: 'GenieCoordinator'):
    """Set global coordinator (called by __init__)."""
    global _coordinator_instance
    with _coordinator_lock:
        _coordinator_instance = coordinator

@dataclass
class CoordinatorConfig:
    """Configuration for Genie coordinator."""
    node_id: str
    
    # Ports
    control_port: int = 5555  # TCP control plane
    data_port: int = 5556     # DPDK data plane
    
    # Transport preferences
    prefer_dpdk: bool = True      # Try DPDK first
    require_dpdk: bool = False    # False = fallback to TCP allowed
    tcp_fallback: bool = True     # Always allow TCP fallback
    
    # DPDK config (if available)
    dpdk_eal_args: list = None
    dpdk_port_id: int = 0

class GenieCoordinator:
    """
    The ONE coordinator for all tensor transfers.
    
    Usage:
        coordinator = GenieCoordinator(config)
        await coordinator.start()
        
        # Send tensor (tries DPDK, falls back to TCP)
        transfer_id = await coordinator.send_tensor(
            tensor, 
            target="node-2:5556"
        )
    """
    
    def __init__(self, config: CoordinatorConfig):
        self.config = config
        self.node_id = config.node_id

        # Components (initialized in start())
        self.control_plane = None
        self.transports = {}
        self.memory_manager = None
        self.active_transfers = {}

        # ✅ ADD: Result management
        self._result_queues: Dict[str, asyncio.Queue] = {}
        self._result_handlers: Dict[str, Any] = {}

        # ✅ Register as global instance
        set_coordinator(self)
        
    async def start(self):
        """Initialize coordinator and all transports."""
        print(f"Starting GenieCoordinator: {self.node_id}")
        
        # 1. Initialize control plane (TCP, always available)
        from ..runtime.control_plane import ControlPlane
        self.control_plane = ControlPlane(
            self.node_id,
            self.config.control_port
        )
        await self.control_plane.start()
        print("✓ Control plane started")
        
        # 2. Initialize memory manager
        from genie.memory import GPUMemoryManager
        self.memory_manager = GPUMemoryManager()
        print("✓ Memory manager initialized")
        
        # 3. Try to initialize DPDK transport
        if self.config.prefer_dpdk:
            try:
                from genie.transport.dpdk_transport import DPDKTransport
                self.transports['dpdk'] = DPDKTransport(self.config)
                await self.transports['dpdk'].initialize()
                print("✓ DPDK transport available (GPU Direct)")
            except Exception as e:
                print(f"✗ DPDK not available: {e}")
                if self.config.require_dpdk:
                    raise
                print("  → Will use TCP fallback")
        
        # 4. Always initialize TCP fallback
        if self.config.tcp_fallback:
            from genie.transport.tcp_transport import TCPTransport
            self.transports['tcp'] = TCPTransport(self.config)
            print("✓ TCP fallback available")

        # ✅ ADD: Register result handler with TCP transport
        if 'tcp' in self.transports:
            self.transports['tcp']._result_callback = self._handle_result_received
            print("✓ TCP transport with result callback")
        
        if not self.transports:
            raise RuntimeError("No transports available!")
            
        print(f"GenieCoordinator ready: {list(self.transports.keys())}")
    
    async def send_tensor(
        self, 
        tensor: torch.Tensor,
        target: str,  # "hostname:port" or node_id
        semantic_metadata: Optional[Dict] = None
    ) -> str:
        """
        Send tensor to target node.
        
        This is the ONLY public send API.
        
        Args:
            tensor: PyTorch tensor (CPU or GPU)
            target: Target node (IP:port or node_id)
            semantic_metadata: Optional metadata (phase, dtype, etc.)
            
        Returns:
            transfer_id for tracking
        """
        # Generate transfer ID
        import uuid
        transfer_id = str(uuid.uuid4())
        
        # Extract/enrich semantic metadata
        metadata = self._extract_metadata(tensor, semantic_metadata)
        
        print(f"\n=== Transfer {transfer_id} ===")
        print(f"  Tensor: {tensor.shape} {tensor.dtype} on {tensor.device}")
        print(f"  Target: {target}")
        print(f"  Size: {metadata['size_bytes'] / 1024**2:.2f} MB")
        
        # Select best transport
        transport = self._select_transport(tensor, metadata)
        print(f"  Transport: {transport.name}")
        
        # Register GPU memory (if needed)
        if tensor.is_cuda and hasattr(self.memory_manager, 'register'):
            self.memory_manager.register(tensor, transfer_id)
        
        # Negotiate with target (control plane)
        print("  Negotiating...")
        accepted = await self.control_plane.negotiate_transfer(
            transfer_id, target, metadata
        )
        if not accepted:
            raise RuntimeError(f"Transfer rejected by {target}")
        
        # Execute transfer
        print("  Sending...")
        success = await transport.send(
            tensor, target, transfer_id, metadata
        )
        
        if success:
            print(f"✓ Transfer {transfer_id} complete")
        else:
            print(f"✗ Transfer {transfer_id} failed")
            
        return transfer_id

    async def receive_tensor(
        self,
        transfer_id: str,
        metadata: Dict
    ) -> torch.Tensor:
        """
        Receive tensor (server-side).

        Called when control plane accepts a transfer.
        """
        print(f"\n=== Receiving {transfer_id} ===")

        # Select transport
        transport = self._select_transport_for_metadata(metadata)
        print(f"  Transport: {transport.name}")

        # Receive data
        tensor = await transport.receive(transfer_id, metadata)

        print(f"✓ Received: {tensor.shape}")
        return tensor

    async def send_and_execute(
        self,
        tensor: torch.Tensor,
        operation: str,
        target: str,  # "hostname:port" or node_id
        metadata: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Send tensor, execute operation remotely, receive result.

        This is the full remote execution flow:
        1. Send tensor to remote server
        2. Server executes operation on tensor
        3. Receive result back

        Args:
            tensor: PyTorch tensor to send
            operation: Operation to execute (e.g., 'aten::relu')
            target: Target server (hostname:port or node_id)
            metadata: Additional metadata for the operation

        Returns:
            Result tensor after remote execution
        """
        # Generate transfer ID
        transfer_id = str(uuid.uuid4())
        result_id = f"{transfer_id}_result"

        # Prepare metadata for send + execute
        send_metadata = self._extract_metadata(tensor, metadata or {})
        send_metadata.update({
            'operation': operation,
            'result_id': result_id,
            'expects_result': True
        })

        print(f"\n=== Remote Execution {transfer_id} ===")
        print(f"  Tensor: {tensor.shape} {tensor.dtype} on {tensor.device}")
        print(f"  Target: {target}")
        print(f"  Operation: {operation}")
        print(f"  Size: {send_metadata['size_bytes'] / 1024**2:.2f} MB")

        # Select transport for sending
        transport = self._select_transport(tensor, send_metadata)
        print(f"  Send Transport: {transport.name}")

        # Send tensor to remote server
        print("  Sending tensor...")
        success = await transport.send(
            tensor, target, transfer_id, send_metadata
        )

        if not success:
            raise RuntimeError(f"Failed to send tensor for {transfer_id}")

        # Wait for result (in practice, this would be handled by callbacks)
        print("  Waiting for result...")

        result = await asyncio.wait_for(
            result_queue.get(),
            timeout=30.0
        )

        print(f"✅ Remote execution complete: {result.shape}")
        return result

    async def _handle_result_received(self, result_id: str, tensor: torch.Tensor):
        """Called by transport when result arrives."""
        print(f"  Result received: {result_id}")
        if result_id in self._result_queues:
            await self._result_queues[result_id].put(tensor)
        else:
            print(f"  Warning: No queue for result {result_id}")

    def _select_transport(self, tensor, metadata):
        """
        Select best transport based on context.
        
        Strategy:
        1. If GPU tensor + DPDK available → use DPDK (100 Gbps)
        2. Otherwise → use TCP (10 Gbps, but always works)
        """
        # Prefer DPDK for GPU tensors (zero-copy)
        if tensor.is_cuda and 'dpdk' in self.transports:
            return self.transports['dpdk']
        
        # Otherwise use TCP
        if 'tcp' not in self.transports:
            raise RuntimeError("No available transport for CPU tensor")
        return self.transports['tcp']

    def _select_transport_for_metadata(self, metadata):
        """Select transport based on metadata."""
        # For Phase 1, always use TCP
        return self.transports['tcp']
    
    def _extract_metadata(self, tensor, user_metadata):
        """Extract semantic metadata from tensor."""
        metadata = {
            'dtype': str(tensor.dtype),
            'shape': list(tensor.shape),
            'size_bytes': tensor.numel() * tensor.element_size(),
            'device': str(tensor.device),
            'is_gpu': tensor.is_cuda,
        }
        
        # Add user-provided semantic metadata
        if user_metadata:
            metadata.update(user_metadata)
        
        # Infer phase if not provided (from shape heuristics)
        if 'phase' not in metadata:
            metadata['phase'] = self._infer_phase(tensor)
        
        return metadata
    
    def _infer_phase(self, tensor):
        """Infer execution phase from tensor characteristics."""
        # Simple heuristic: large batch = prefill, small = decode
        if len(tensor.shape) >= 2:
            batch_size = tensor.shape[0]
            if batch_size >= 32:
                return "prefill"
            elif batch_size == 1:
                return "decode"
        return "unknown"
    
    async def stop(self):
        """Shutdown coordinator."""
        print("Stopping GenieCoordinator...")
        
        # Stop control plane
        if self.control_plane:
            await self.control_plane.stop()
        
        # Stop transports
        for name, transport in self.transports.items():
            if hasattr(transport, 'stop'):
                await transport.stop()
        
        print("GenieCoordinator stopped")