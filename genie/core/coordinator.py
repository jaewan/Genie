"""
THE coordinator. All tensor transfers go through here.

Design principles:
1. Try DPDK first (if GPU + hardware available)
2. Automatic TCP fallback (always works)
3. Semantic metadata tracking
4. Simple > Complex
"""

import threading
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Callable, Any, List
import asyncio
import uuid
import torch
from genie.scheduler.stub_scheduler import get_scheduler
from .metadata_types import OperationMetadata, create_operation_metadata

logger = logging.getLogger(__name__)

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

    # Network configuration (delegated to centralized config)
    control_port: Optional[int] = None  # Use centralized config if None
    data_port: Optional[int] = None     # Use centralized config if None

    # Transport preferences (delegated to centralized config)
    prefer_dpdk: Optional[bool] = None      # Use centralized config if None
    require_dpdk: Optional[bool] = None    # Use centralized config if None
    tcp_fallback: Optional[bool] = None    # Use centralized config if None

    # DPDK config (if available)
    dpdk_eal_args: Optional[list] = None
    dpdk_port_id: int = 0

    # Profiling configuration (delegated to centralized config)
    enable_profiling: Optional[bool] = None  # Use centralized config if None

    # Server configuration
    is_server: bool = False  # True if this coordinator is used by a server

    def get_network_config(self):
        """Get network configuration, using centralized config as fallback."""
        from ..config import get_config
        config = get_config()

        return {
            'control_port': self.control_port or config.network.control_port,
            'data_port': self.data_port or config.network.data_port,
            'prefer_dpdk': self.prefer_dpdk if self.prefer_dpdk is not None else config.network.prefer_dpdk,
            'require_dpdk': self.require_dpdk if self.require_dpdk is not None else config.network.require_dpdk,
            'tcp_fallback': self.tcp_fallback if self.tcp_fallback is not None else config.network.tcp_fallback,
        }

    def get_performance_config(self):
        """Get performance configuration."""
        from ..config import get_config
        config = get_config()

        return {
            'enable_profiling': self.enable_profiling if self.enable_profiling is not None else config.performance.enable_profiling,
            'operation_timeout': config.performance.operation_timeout,
            'transfer_timeout': config.performance.transfer_timeout,
        }

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

        # Get centralized configuration
        from ..config import get_config
        self._central_config = get_config()

        # Components (initialized in start())
        self.control_plane = None
        self.transports = {}
        self.memory_manager = None
        self.active_transfers = {}

        # ✅ ADD: Result management
        self._result_queues: Dict[str, asyncio.Queue] = {}
        self._result_handlers: Dict[str, Any] = {}

        # ✅ ADD: Scheduler integration (CRITICAL for semantic awareness)
        self.scheduler = get_scheduler()

        # ✅ ADD: Profiling integration (CRITICAL for performance analysis)
        network_config = config.get_network_config()
        perf_config = config.get_performance_config()
        self.enable_profiling = perf_config['enable_profiling']

        if self.enable_profiling:
            from genie.profiling import GenieProfiler
            self.profiler = GenieProfiler()
            logger.info("✓ Profiling enabled")
        else:
            self.profiler = None

        # ✅ Register as global instance
        set_coordinator(self)
        
    async def start(self):
        """Initialize coordinator and all transports."""
        logger.info(f"Starting GenieCoordinator: {self.node_id}")

        # 1. Initialize control plane (TCP, always available)
        from ..runtime.control_plane import ControlPlane
        network_config = self.config.get_network_config()
        self.control_plane = ControlPlane(
            self.node_id,
            network_config['control_port']
        )
        await self.control_plane.start()
        logger.info("✓ Control plane started")

        # 2. Initialize memory manager
        from genie.memory import GPUMemoryManager
        self.memory_manager = GPUMemoryManager()
        logger.info("✓ Memory manager initialized")
        
        # 3. Try to initialize DPDK transport
        if self.config.prefer_dpdk:
            try:
                from genie.transport.dpdk_transport import DPDKTransport
                self.transports['dpdk'] = DPDKTransport(self.config)
                await self.transports['dpdk'].initialize()
                logger.info("✓ DPDK transport available (GPU Direct)")
            except Exception as e:
                logger.warning(f"✗ DPDK not available: {e}")
                if self.config.require_dpdk:
                    raise
                logger.info("  → Will use TCP fallback")

        # 4. Initialize TCP fallback (but not for server coordinators)
        if self.config.tcp_fallback:
            from genie.transport.tcp_transport import TCPTransport
            self.transports['tcp'] = TCPTransport(self.config)
            # Only initialize TCP server for client coordinators (not servers)
            # Servers will use their own TCP server
            is_server_coordinator = (hasattr(self.config, 'is_server') and self.config.is_server)
            if not is_server_coordinator:
                # Client coordinators need to initialize TCP to receive results from servers
                await self.transports['tcp'].initialize()
                logger.info("✓ TCP fallback available")

        # ✅ ADD: Register result handler with TCP transport (for clients only)
        if 'tcp' in self.transports and not is_server_coordinator:
            self.transports['tcp']._result_callback = self._handle_result_received
            logger.info("✓ TCP transport with result callback")
        
        if not self.transports:
            raise RuntimeError("No transports available!")
            
        logger.info(f"GenieCoordinator ready: {list(self.transports.keys())}")
    
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
        operation_metadata = create_operation_metadata(
            operation=operation,
            result_id=result_id,
            inputs=[tensor]
        )
        send_metadata = self._extract_metadata(tensor, operation_metadata.to_dict())

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
            timeout=self._central_config.performance.operation_timeout
        )

        logger.info(f"✅ Remote execution complete: {result.shape}")
        return result

    async def execute_remote_operation(
        self,
        operation: str,
        inputs: list,
        target: str,
        timeout: Optional[float] = None
    ) -> torch.Tensor:
        """
        Execute operation remotely and wait for result.

        This is the main client-side API for remote execution.

        Args:
            operation: ATen operation (e.g., 'aten::add', 'aten::relu')
            inputs: List of input tensors
            target: Server address ('hostname:port')
            timeout: Max wait time in seconds (uses config default if None)

        Returns:
            Result tensor (on CPU)

        Raises:
            RuntimeError: If execution fails or times out
        """
        # Use centralized timeout configuration
        if timeout is None:
            timeout = self._central_config.performance.operation_timeout

        # ✅ NEW: Wrap with profiler if enabled
        if self.profiler:
            with self.profiler.profile_operation(operation, {
                'target': target,
                'input_shapes': [list(t.shape) for t in inputs],
                'input_dtypes': [str(t.dtype) for t in inputs],
                'num_inputs': len(inputs),
                'timeout': timeout
            }) as measurement:

                # Generate IDs
                transfer_id = str(uuid.uuid4())
                result_id = f"{transfer_id}_result"

                logger.info(f"🚀 Remote execution: {operation} with {len(inputs)} inputs → {target}")

                # Time scheduler decision
                t0 = time.perf_counter()
                primary_tensor = inputs[0]
                operation_metadata = create_operation_metadata(
                    operation=operation,
                    result_id=result_id,
                    inputs=inputs,
                    client_port=self._central_config.network.data_port
                )
                metadata = self._extract_metadata(primary_tensor, operation_metadata.to_dict())
                measurement['timings']['scheduler_time'] = time.perf_counter() - t0

                # ✅ Register target server with scheduler (if not already known)
                if hasattr(self.scheduler, 'register_server'):
                    self.scheduler.register_server(target)

                # ✅ Consult scheduler (THIS IS THE KEY CHANGE)
                scheduling_decision = self.scheduler.schedule(
                    operation=operation,
                    inputs=inputs,
                    metadata=metadata
                )

                # ✅ Use scheduler's device choice
                scheduled_target = scheduling_decision['device']
                logger.info(f"Scheduler placed {operation} on {scheduled_target} "
                           f"(requested: {target}, explanation: {scheduling_decision['explanation']})")

                # Step 6: Select transport
                transport = self._select_transport(primary_tensor, metadata)

                # Step 7: Send multiple tensors using SCHEDULER'S TARGET
                logger.debug(f"  Sending {len(inputs)} tensors to {scheduled_target}")

                # Time network transfer
                t0 = time.perf_counter()
                success = await transport.send_multi_tensor(
                    inputs, scheduled_target, transfer_id, metadata
                )
                measurement['timings']['network_send'] = time.perf_counter() - t0

                if not success:
                    raise RuntimeError(f"Failed to send tensors for {operation}")

                # Step 8: Wait for result (with timeout)
                logger.debug(f"  Waiting for result (timeout={timeout}s)...")

                # Time result waiting
                t0 = time.perf_counter()
                result_queue = self._create_result_queue(result_id)
                result = await asyncio.wait_for(
                    result_queue.get(),
                    timeout=timeout
                )
                measurement['timings']['wait_result'] = time.perf_counter() - t0

                # Step 9: Check if result is an exception
                if isinstance(result, Exception):
                    logger.error(f"Remote execution failed: {result}")
                    raise result

                # Time deserialization (minimal for our protocol)
                t0 = time.perf_counter()
                measurement['timings']['deserialize'] = time.perf_counter() - t0

                logger.info(f"✅ Remote execution complete: {result.shape}")
                return result
        else:
            # Normal execution without profiling
            return await self._execute_internal(operation, inputs, target, timeout)

    async def _execute_internal(
        self,
        operation: str,
        inputs: list,
        target: str,
        timeout: Optional[float] = None
    ) -> torch.Tensor:
        """Internal execution without profiling."""
        # Use centralized timeout configuration
        if timeout is None:
            timeout = self._central_config.performance.operation_timeout

        # Generate IDs
        transfer_id = str(uuid.uuid4())
        result_id = f"{transfer_id}_result"

        logger.info(f"🚀 Remote execution: {operation} with {len(inputs)} inputs → {target}")

        # Step 1: Create result queue BEFORE sending
        result_queue = self._create_result_queue(result_id)

        try:
            # Step 2: Prepare metadata (WITHOUT tensor data - tensors sent separately)
            primary_tensor = inputs[0]
            operation_metadata = create_operation_metadata(
                operation=operation,
                result_id=result_id,
                inputs=inputs,
                client_port=self._central_config.network.data_port  # Tell server where to send result
            )
            metadata = self._extract_metadata(primary_tensor, operation_metadata.to_dict())
            logger.debug(f"Sending operation metadata: {metadata}")
            logger.debug(f"Client port in metadata: {metadata.get('client_port')}")

            # ✅ NEW: Step 3 - Register target server with scheduler (if not already known)
            if hasattr(self.scheduler, 'register_server'):
                self.scheduler.register_server(target)

            # ✅ NEW: Step 4 - Consult scheduler (THIS IS THE KEY CHANGE)
            scheduling_decision = self.scheduler.schedule(
                operation=operation,
                inputs=inputs,
                metadata=metadata
            )

            # ✅ NEW: Step 5 - Use scheduler's device choice
            scheduled_target = scheduling_decision['device']
            logger.info(f"Scheduler placed {operation} on {scheduled_target} "
                       f"(requested: {target}, explanation: {scheduling_decision['explanation']})")

            # Step 6: Select transport
            transport = self._select_transport(primary_tensor, metadata)

            # Step 7: Send multiple tensors using SCHEDULER'S TARGET
            logger.debug(f"  Sending {len(inputs)} tensors to {scheduled_target}")
            success = await transport.send_multi_tensor(
                inputs, scheduled_target, transfer_id, metadata
            )

            if not success:
                raise RuntimeError(f"Failed to send tensors for {operation}")

            # Step 8: Wait for result (with timeout)
            logger.debug(f"  Waiting for result (timeout={timeout}s)...")
            result = await asyncio.wait_for(
                result_queue.get(),
                timeout=timeout
            )

            # Step 9: Check if result is an exception
            if isinstance(result, Exception):
                logger.error(f"Remote execution failed: {result}")
                raise result

            logger.info(f"✅ Remote execution complete: {result.shape}")
            return result

        except asyncio.TimeoutError:
            # Check if result was received via callback (race condition)
            if result_id in self._result_queues:
                # Result arrived via callback during timeout handling
                try:
                    result = await asyncio.wait_for(
                        self._result_queues[result_id].get(),
                        timeout=0.1  # Short timeout to avoid hanging
                    )
                    logger.info(f"✅ Remote execution complete via callback: {result.shape}")
                    return result
                except asyncio.TimeoutError:
                    pass  # No result, proceed with timeout error

            # ✅ NEW: Enhanced timeout error with detailed context
            logger.error(f"Remote execution timeout after {timeout}s")
            logger.error(f"  Operation: {operation}")
            logger.error(f"  Target: {scheduled_target}")
            logger.error(f"  Input shapes: {[list(t.shape) for t in inputs]}")
            logger.error(f"  Queue status: {len(self._result_queues)} active queues")

            raise RuntimeError(
                f"Remote execution timeout after {timeout}s. "
                f"Operation: {operation}, Target: {scheduled_target}. "
                f"Possible causes: server crashed, network partition, operation too slow, "
                f"or result routing failure. Check server logs and network connectivity."
            )
        except Exception as e:
            # ✅ NEW: Enhanced error context for debugging
            logger.error(f"Remote execution failed: {e}")
            logger.error(f"  Operation: {operation}")
            logger.error(f"  Target: {scheduled_target}")
            logger.error(f"  Input count: {len(inputs)}")
            logger.error(f"  Queue cleanup: {len(self._result_queues)} queues remain")

            # Re-raise with enhanced context
            raise RuntimeError(
                f"Remote execution failed for {operation}: {e}. "
                f"Target: {scheduled_target}, Inputs: {len(inputs)} tensors. "
                f"Check server status and network connectivity."
            ) from e
        finally:
            # Step 10: Cleanup queue
            self._result_queues.pop(result_id, None)

    def _create_result_queue(self, result_id: str) -> asyncio.Queue:
        """Create result queue for operation."""
        result_queue = asyncio.Queue(maxsize=1)
        self._result_queues[result_id] = result_queue
        return result_queue

    async def _handle_result_received(self, result_id: str, result):
        """
        Handle result from transport (callback).

        Args:
            result_id: Result identifier (from metadata)
            result: torch.Tensor (success) or Exception (failure)
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"  Result received: {result_id}")
        logger.debug(f"  Available queues: {list(self._result_queues.keys())}")

        if result_id in self._result_queues:
            queue = self._result_queues[result_id]

            try:
                # Put result (tensor or exception) in queue (non-blocking)
                queue.put_nowait(result)

                # Log based on type
                if isinstance(result, Exception):
                    logger.debug(f"  Error delivered: {result}")
                else:
                    logger.debug(f"  Result delivered: {result.shape}")
            except asyncio.QueueFull:
                logger.warning(f"Result queue full for {result_id}")
        else:
            logger.warning(f"No queue for result {result_id}")
            logger.warning(f"Available queue keys: {list(self._result_queues.keys())}")

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
        logger.info("Stopping GenieCoordinator...")

        # Stop control plane
        if self.control_plane:
            await self.control_plane.stop()

        # Stop transports
        for name, transport in self.transports.items():
            if hasattr(transport, 'stop'):
                await transport.stop()

        logger.info("GenieCoordinator stopped")