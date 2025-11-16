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
import time
import json
from dataclasses import dataclass
from typing import Optional, Dict, Callable, Any, List
import asyncio
import uuid
import torch
from ..scheduler import Scheduler
from .metadata_types import OperationMetadata, create_operation_metadata

logger = logging.getLogger(__name__)

# ==========================================
# SINGLETON PATTERN FOR COORDINATOR
# ==========================================

_coordinator_instance: Optional['DjinnCoordinator'] = None
_coordinator_lock = threading.Lock()

def get_coordinator() -> 'DjinnCoordinator':
    """Get global coordinator instance."""
    global _coordinator_instance
    if _coordinator_instance is None:
        raise RuntimeError(
            "Coordinator not initialized. "
            "Call DjinnCoordinator(config).start() first."
        )
    return _coordinator_instance

def set_coordinator(coordinator: 'DjinnCoordinator'):
    """Set global coordinator (called by __init__)."""
    global _coordinator_instance
    with _coordinator_lock:
        _coordinator_instance = coordinator

@dataclass
class CoordinatorConfig:
    """Configuration for Djinn coordinator."""
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

        # ✅ FIX: Always prefer explicit config values over defaults
        network_config = {
            'control_port': self.control_port or config.network.control_port,
            'data_port': self.data_port or config.network.data_port,
            'prefer_dpdk': self.prefer_dpdk if self.prefer_dpdk is not None else config.network.prefer_dpdk,
            'require_dpdk': self.require_dpdk if self.require_dpdk is not None else config.network.require_dpdk,
            'tcp_fallback': self.tcp_fallback if self.tcp_fallback is not None else config.network.tcp_fallback,
        }

        logger.debug(f"CoordinatorConfig network config: control_port={self.control_port}, data_port={self.data_port}")
        logger.debug(f"Final network config: {network_config}")
        return network_config

    def get_performance_config(self):
        """Get performance configuration."""
        from ..config import get_config
        config = get_config()

        return {
            'enable_profiling': self.enable_profiling if self.enable_profiling is not None else config.performance.enable_profiling,
            'operation_timeout': config.performance.operation_timeout,
            'transfer_timeout': config.performance.transfer_timeout,
        }

class DjinnCoordinator:
    """
    The ONE coordinator for all tensor transfers.
    
    Usage:
        coordinator = DjinnCoordinator(config)
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

        # ✅ ADD: Result management (refactored to separate module)
        from .coordinator_result import ResultManager
        self._result_manager = ResultManager()

        # ✅ ADD: Scheduler integration (CRITICAL for semantic awareness)
        self.scheduler = Scheduler()

        # ✅ ADD: Profiling integration (CRITICAL for performance analysis)
        network_config = self._central_config.get_network_config()
        perf_config = self._central_config.get_performance_config()
        self.enable_profiling = perf_config.enable_profiling

        # Profiling is handled via ProfilingContext (djinn/server/profiling_context.py)
        # No need for separate profiler instance here
        if self.enable_profiling:
            logger.info("✓ Profiling enabled (via ProfilingContext)")

        # ✅ ADD: Global Fleet Coordinator integration (Phase 3)
        self.global_coordinator = None
        self._init_global_coordinator()

        # ✅ Register as global instance
        set_coordinator(self)

    def _init_global_coordinator(self):
        """Initialize global fleet coordinator if enabled."""
        fleet_config = self._central_config.fleet
        
        if not fleet_config.enable_global_coordinator:
            logger.debug("Global fleet coordinator disabled")
            return
        
        if not fleet_config.global_coordinator_address:
            logger.debug("Global fleet coordinator address not configured, using direct server mode")
            return
        
        try:
            from ..fleet import GlobalFleetCoordinator
            from ..fleet.model_registry import GlobalModelRegistry
            
            # Initialize global coordinator with Redis backend if available
            model_registry = GlobalModelRegistry(
                backend=None,  # Auto-detect
                redis_url=fleet_config.redis_url
            )
            
            self.global_coordinator = GlobalFleetCoordinator(
                model_registry=model_registry,
                redis_url=fleet_config.redis_url
            )
            
            logger.info(
                f"✅ Global fleet coordinator initialized "
                f"(address={fleet_config.global_coordinator_address})"
            )
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize global fleet coordinator: {e}")
            logger.warning("Falling back to direct server mode")
            self.global_coordinator = None
    
    def start_sync(self):
        """Synchronous wrapper for start()."""
        try:
            # Try to get existing event loop
            loop = asyncio.get_running_loop()
            logger.warning("Cannot start coordinator from running event loop")
            return False
        except RuntimeError:
            # No running loop, create new one
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.start())
                return True
            except Exception as e:
                logger.error(f"Failed to start coordinator: {e}")
                return False

    async def start(self):
        """Initialize coordinator and all transports."""
        logger.info(f"Starting DjinnCoordinator: {self.node_id}")

        # 1. Initialize control plane (TCP, always available)
        try:
            from djinn.backend.runtime.control_plane import ControlPlane
        except ImportError:
            from ...backend.runtime.control_plane import ControlPlane
        network_config = self.config.get_network_config()
        logger.debug(f"Coordinator network config: {network_config}")
        self.control_plane = ControlPlane(
            self.node_id,
            network_config.get('control_port', 5555)
        )
        await self.control_plane.start()
        logger.info("✓ Control plane started")

        # 2. Initialize memory manager
        try:
            from djinn.backend.memory import GPUMemoryManager
        except ImportError:
            from ...backend.memory import GPUMemoryManager
        self.memory_manager = GPUMemoryManager()
        logger.info("✓ Memory manager initialized")
        
        # 3. Try to initialize DPDK transport
        if self.config.prefer_dpdk:
            try:
                try:
                    from djinn.server.transport.dpdk_transport import DPDKTransport
                except ImportError:
                    from ...server.transport.dpdk_transport import DPDKTransport
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
            try:
                from djinn.server.transport.tcp_transport import TCPTransport
            except ImportError:
                from ...server.transport.tcp_transport import TCPTransport
            self.transports['tcp'] = TCPTransport(self.config)
            # Only initialize TCP server for client coordinators (not servers)
            # Servers will use their own TCP server
            is_server_coordinator = (hasattr(self.config, 'is_server') and self.config.is_server)
            if not is_server_coordinator:
                # ✅ FIX: Register callback BEFORE initialize() to avoid race condition
                self.transports['tcp']._result_callback = self._result_manager.handle_result_received
                logger.info("✓ Result callback registered")

                # Client coordinators need to initialize TCP to receive results from servers
                success = await self.transports['tcp'].initialize()
                if success:
                    logger.info("✓ TCP fallback available")
                    # ✅ FIX: Add operation callback to handle incoming operation requests
                    # (Client might also act as server in some scenarios)
                    self.transports['tcp']._operation_callback = self._handle_operation_request
                    logger.info("✓ Operation callback registered")
                else:
                    logger.warning("✗ TCP transport initialization failed, removing transport")
                    del self.transports['tcp']
        
        if not self.transports:
            raise RuntimeError("No transports available!")
            
        logger.info(f"DjinnCoordinator ready: {list(self.transports.keys())}")
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Get server capabilities from remote server.
        
        Returns:
            Dictionary with server capabilities (gpu_count, gpus, etc.)
            
        NOTE: This queries the control plane for capabilities. The control plane
        should have received server capabilities during connection establishment.
        If not available, returns a minimal dict to avoid breaking initialization.
        """
        if self.control_plane is None:
            raise RuntimeError("Control plane not initialized")
        
        # Get capabilities from control plane
        # The control plane stores server capabilities after capability exchange
        try:
            capabilities = self.control_plane.get_capabilities()
            
            # Handle None case (capabilities not yet exchanged)
            if capabilities is None:
                logger.debug("Control plane capabilities not yet available (capability exchange may not have completed)")
                return {
                    'gpu_count': 0,  # Unknown until exchange completes
                    'gpus': [],
                    'networks': [],
                    'total_memory_gb': 0,
                    'supported_transports': ['tcp'],
                    'hostname': 'unknown',
                    'node_id': 'unknown',
                }
            
            # Convert NodeCapabilities dataclass to dict
            from dataclasses import asdict
            try:
                caps_dict = asdict(capabilities)
                # Ensure gpu_count is present (NodeCapabilities has it)
                if 'gpu_count' not in caps_dict:
                    caps_dict['gpu_count'] = getattr(capabilities, 'gpu_count', 0)
                return caps_dict
            except Exception:
                # Fallback: manual conversion
                return {
                    'gpu_count': getattr(capabilities, 'gpu_count', 0),
                    'gpus': [],
                    'networks': [],
                    'total_memory_gb': 0,
                    'supported_transports': ['tcp'],
                    'hostname': getattr(capabilities, 'node_id', 'unknown'),
                    'node_id': getattr(capabilities, 'node_id', 'unknown'),
                }
            
        except Exception as e:
            logger.warning(f"Failed to get capabilities from control plane: {e}")
            # Return minimal dict to avoid breaking initialization
            return {
                'gpu_count': 0,
                'gpus': [],
                'networks': [],
                'total_memory_gb': 0,
                'supported_transports': ['tcp'],
                'hostname': 'unknown',
                'node_id': 'unknown',
            }
    
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
        from .coordinator_transport import MetadataExtractor, TransportSelector
        metadata_extractor = MetadataExtractor(self.scheduler)
        transport_selector = TransportSelector(self.transports)
        
        metadata = metadata_extractor.extract_metadata(tensor, semantic_metadata)
        
        print(f"\n=== Transfer {transfer_id} ===")
        print(f"  Tensor: {tensor.shape} {tensor.dtype} on {tensor.device}")
        print(f"  Target: {target}")
        print(f"  Size: {metadata['size_bytes'] / 1024**2:.2f} MB")
        
        # Select best transport
        transport = transport_selector.select_transport(tensor, metadata)
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
        from .coordinator_transport import TransportSelector
        transport_selector = TransportSelector(self.transports)
        transport = transport_selector.select_transport_for_metadata(metadata)
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
        # ✅ FIX: Use the transport's actual listening port
        client_port = self.transports['tcp'].data_port if 'tcp' in self.transports else self._central_config.network.data_port
        operation_metadata = create_operation_metadata(
            operation=operation,
            result_id=result_id,
            inputs=[tensor],
            client_port=client_port  # ✅ Actual listening port
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
        
        # Create result queue for receiving the result
        result_queue = self._create_result_queue(result_id)
        result = await asyncio.wait_for(
            result_queue.get(),
            timeout=self._central_config.performance.operation_timeout
        )

        logger.info(f"✅ Remote execution complete: {result.shape}")
        return result

    async def execute_remote_operation_async(
        self,
        operation: str,
        inputs: list,
        target: str,
        timeout: Optional[float] = None
    ) -> asyncio.Future:
        """
        Execute operation remotely asynchronously (non-blocking).

        Returns a Future that resolves to the result tensor.

        This enables parallel execution of independent operations.
        """
        # Use centralized timeout configuration
        if timeout is None:
            timeout = self._central_config.performance.operation_timeout

        # Generate IDs
        transfer_id = str(uuid.uuid4())
        result_id = f"{transfer_id}_result"

        logger.info(f"🚀 Async remote execution: {operation} with {len(inputs)} inputs → {target}")

        # Create result queue BEFORE sending
        result_queue = self._create_result_queue(result_id)

        try:
            # Prepare metadata
            primary_tensor = inputs[0]
            client_port = self.transports['tcp'].data_port if 'tcp' in self.transports else self._central_config.network.data_port
            operation_metadata = create_operation_metadata(
                operation=operation,
                result_id=result_id,
                inputs=inputs,
                client_port=client_port
            )
            metadata = self._extract_metadata(primary_tensor, operation_metadata.to_dict())

            # Register target server with scheduler
            if hasattr(self.scheduler, 'register_server'):
                self.scheduler.register_server(target)

            # Consult scheduler
            scheduling_decision = self.scheduler.schedule(
                operation=operation,
                inputs=inputs,
                metadata=metadata
            )

            # Use scheduler's device choice
            scheduled_target = scheduling_decision['device']
            logger.info(f"Scheduler placed {operation} on {scheduled_target}")

            # Select transport
            transport = self._select_transport(primary_tensor, metadata)

            # Send tensors using scheduler's target
            success = await transport.send_multi_tensor(
                inputs, scheduled_target, transfer_id, metadata
            )

            if not success:
                raise RuntimeError(f"Failed to send tensors for {operation}")

            # Return future that will resolve when result arrives
            async def wait_for_result():
                try:
                    result = await asyncio.wait_for(
                        result_queue.get(),
                        timeout=timeout
                    )

                    if isinstance(result, Exception):
                        raise result

                    return result

                except asyncio.TimeoutError:
                    raise RuntimeError(f"Async operation timeout after {timeout}s")

            return asyncio.create_task(wait_for_result())

        except Exception as e:
            logger.error(f"Async remote execution failed: {e}")
            # Return a future that raises the exception
            async def failing_future():
                raise e
            return asyncio.create_task(failing_future())
        finally:
            # Cleanup queue
            self._result_queues.pop(result_id, None)

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

        # Profiling is handled via ProfilingContext (djinn/server/profiling_context.py)
        # No need for separate profiler instance here
        
        # Generate IDs
        transfer_id = str(uuid.uuid4())
        result_id = f"{transfer_id}_result"

        logger.info(f"🚀 Remote execution: {operation} with {len(inputs)} inputs → {target}")

        # Time scheduler decision
        t0 = time.perf_counter()
        primary_tensor = inputs[0]
        # ✅ FIX: Use the transport's actual listening port, not the config default
        client_port = self.transports['tcp'].data_port if 'tcp' in self.transports else self._central_config.network.data_port
        logger.debug(f"Using client port for operation: {client_port}")
        operation_metadata = create_operation_metadata(
            operation=operation,
            result_id=result_id,
            inputs=inputs,
            client_port=client_port  # ✅ Actual listening port
        )
        metadata = self._extract_metadata(primary_tensor, operation_metadata.to_dict())

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
        from .coordinator_transport import TransportSelector
        transport_selector = TransportSelector(self.transports)
        transport = transport_selector.select_transport(primary_tensor, metadata)

        # Step 7: Send multiple tensors using SCHEDULER'S TARGET
        logger.debug(f"  Sending {len(inputs)} tensors to {scheduled_target}")

        # Time network transfer
        t0 = time.perf_counter()
        success = await transport.send_multi_tensor(
            inputs, scheduled_target, transfer_id, metadata
        )

        if not success:
            raise RuntimeError(f"Failed to send tensors for {operation}")

        # Step 8: Wait for result (with timeout)
        logger.debug(f"  Waiting for result (timeout={timeout}s)...")

        # Time result waiting
        t0 = time.perf_counter()
        result_queue = self._result_manager.create_result_queue(result_id)
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

    async def execute_remote_subgraph(self,
                                      subgraph: Dict[str, Any],
                                      input_data: Dict[str, torch.Tensor],
                                      target: str,
                                      timeout: float = 300,  # Increased default timeout for large subgraphs
                                      graph_id: Optional[str] = None,
                                      enable_differential: bool = True,
                                      model: Optional[torch.nn.Module] = None) -> torch.Tensor:
        """
        Execute subgraph remotely (O(1) network transfer).

        This is the KEY optimization - send entire graph once!
        Reduces O(n) individual operations to O(1) subgraph execution.

        Args:
            subgraph: Serialized RemoteSubgraph from SmartSubgraphBuilder
            input_data: Dict of tensor_id -> input tensor for external inputs
            target: Server address ('hostname:port')
            timeout: Max wait time in seconds
            graph_id: Optional graph identifier for differential updates
            enable_differential: Whether to enable differential protocol
            model: Optional model instance (for model cache integration)

        Returns:
            Result tensor (on CPU)

        Raises:
            RuntimeError: If execution fails or times out
        """
        import pickle
        import time

        # ✅ NEW: Try model cache first if model is provided
        if model is not None:
            try:
                from .enhanced_model_manager import EnhancedModelManager
                from .model_tracker import get_model_tracker
                
                manager = EnhancedModelManager(coordinator=self)
                tracker = get_model_tracker()
                
                # Compute fingerprint
                from .model_fingerprint import ModelFingerprint
                fingerprint = ModelFingerprint.compute(model)
                
                # Track model if not already tracked
                if not tracker.is_registered(fingerprint):
                    tracker.track_model(model, fingerprint)
                
                # Try model cache execution
                if manager.use_model_cache:
                    try:
                        # Convert input_data to dict format expected by execute_model
                        # Assume first input is the main input
                        inputs_dict = {}
                        if len(input_data) == 1:
                            # Single input - use 'x' as key
                            inputs_dict['x'] = list(input_data.values())[0]
                        else:
                            # Multiple inputs - use tensor_id as key
                            inputs_dict = {k: v for k, v in input_data.items()}
                        
                        logger.info(f"🎯 Attempting model cache execution for {fingerprint}")
                        result = await manager.execute_model(model, inputs_dict)
                        logger.info(f"✅ Model cache execution successful: {result.shape}")
                        return result
                    except Exception as e:
                        logger.debug(f"Model cache execution failed: {e}, falling back to graph execution")
                        # Fall through to graph execution
            except Exception as e:
                logger.debug(f"Model cache integration error: {e}, using graph execution")
                # Fall through to graph execution
        
        # ✅ NEW: Also try to detect model from input tensors (automatic detection)
        try:
            from .model_tracker import get_model_tracker
            tracker = get_model_tracker()
            
            # Check if any input tensor belongs to a registered model
            detected_fingerprint = None
            for tensor_id_str, tensor in input_data.items():
                if isinstance(tensor, torch.Tensor):
                    fingerprint = tracker.get_model_fingerprint(tensor)
                    if fingerprint and tracker.is_registered(fingerprint):
                        detected_fingerprint = fingerprint
                        logger.debug(f"🎯 Auto-detected registered model {fingerprint} from input tensor")
                        break
            
            # If we detected a model but don't have the instance, we can't use model cache
            # (we need the model instance to call execute_model)
            # This is a limitation - we'd need to store model instances in the tracker
            # For now, we'll just log and use graph execution
            if detected_fingerprint:
                logger.debug(f"⚠️  Detected model {detected_fingerprint} but no model instance available, using graph execution")
        except Exception as e:
            logger.debug(f"Auto model detection failed: {e}")
            # Fall through to graph execution

        logger.info(f"🚀 Subgraph execution: {len(subgraph.get('operations', []))} ops, "
                   f"{len(input_data)} inputs → {target}")

        start_time = time.perf_counter()

        # ✅ PHASE 1: Cache query optimization (NEW)
        # Convert tensor IDs to stable identifiers and query cache
        from .cache_query import get_cache_query_client
        from .model_registry import get_model_registry
        
        cache_client = get_cache_query_client()
        registry = get_model_registry()
        
        # Map tensors to stable identifiers
        tensor_identifiers = {}  # tensor_id_str → identifier
        identifier_to_tensor_id = {}  # identifier → tensor_id_str (for server mapping)
        
        for tensor_id_str, tensor in input_data.items():
            if isinstance(tensor, torch.Tensor):
                identity = registry.get_identity(tensor)
                tensor_identifiers[tensor_id_str] = identity.identifier
                identifier_to_tensor_id[identity.identifier] = tensor_id_str
        
        # Query cache for which identifiers are cached
        query_start = time.perf_counter()
        cache_query_result = await cache_client.query_cached_identifiers(
            target,
            set(tensor_identifiers.values()),
            use_local_cache=True
        )
        query_time = (time.perf_counter() - query_start) * 1000
        
        logger.info(
            f"📊 Cache query: {len(cache_query_result.cached_identifiers)}/{len(tensor_identifiers)} cached "
            f"({cache_query_result.cache_hit_rate:.1f}% hit rate, {query_time:.1f}ms)"
        )
        
        # Filter tensors: only send uncached ones
        # Also filter out meta tensors (they have no data and can't be sent)
        tensors_to_send = {}
        meta_tensors_filtered = 0
        for tensor_id_str, identifier in tensor_identifiers.items():
            if identifier in cache_query_result.missing_identifiers:
                tensor = input_data[tensor_id_str]
                # ✅ FIX: Filter meta tensors before sending (defense in depth)
                if isinstance(tensor, torch.Tensor) and tensor.device.type == 'meta':
                    logger.warning(
                        f"⚠️  Filtering meta tensor[{tensor_id_str}] before sending to server. "
                        f"Meta tensors have no data and cannot be transferred. "
                        f"This should have been filtered earlier in subgraph building."
                    )
                    meta_tensors_filtered += 1
                    continue
                tensors_to_send[tensor_id_str] = tensor
        
        if meta_tensors_filtered > 0:
            logger.warning(
                f"⚠️  Filtered {meta_tensors_filtered} meta tensor(s) from input_data. "
                f"This may cause execution errors if these tensors are actually needed."
            )
        
        # Calculate transfer savings
        total_size = sum(t.element_size() * t.numel() for t in input_data.values())
        transfer_size = sum(t.element_size() * t.numel() for t in tensors_to_send.values())
        savings = 100 * (1 - transfer_size / total_size) if total_size > 0 else 0
        
        logger.info(
            f"💾 Transfer optimization: {len(tensors_to_send)}/{len(input_data)} tensors to send "
            f"({transfer_size / 1e6:.1f}MB / {total_size / 1e6:.1f}MB, {savings:.1f}% savings)"
        )

        try:
            # ✅ Week 3: Use DifferentialGraphProtocol for iterative workloads
            # 
            # STATUS: DISABLED (2024)
            # Reason: Disabled for compatibility with current model cache system.
            # The new model cache system (memory_aware_model_cache.py) handles
            # model caching more efficiently, making differential graph updates
            # less critical. This feature may be re-enabled in the future if
            # profiling shows it provides benefits for specific workloads.
            # 
            # To re-enable: Set condition to `if enable_differential and graph_id:`
            # and ensure server-side delta reconstruction is implemented.
            if False:  # enable_differential and graph_id:
                try:
                    from djinn.server.differential_graph import DifferentialGraphProtocol
                except ImportError:
                    from ...server.differential_graph import DifferentialGraphProtocol

                # Initialize protocol if not exists
                if not hasattr(self, '_differential_protocol'):
                    self._differential_protocol = DifferentialGraphProtocol()

                # Send graph using differential protocol
                message_data = self._differential_protocol.send_graph(
                    graph_id=graph_id,
                    graph=subgraph,
                    is_update=self._differential_protocol.client_versions.get(graph_id, 0) > 0
                )

                logger.debug(f"Differential protocol: {message_data.get('type', 'unknown')}")
            else:
                # Standard full subgraph transmission
                message_data = {
                    'type': 'full_subgraph',
                    'subgraph': subgraph
                }

            # Establish connection
            host, port_str = target.split(':')
            port = int(port_str)
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=5.0
            )

            # Prepare message with cached identifiers and mapping
            # Map cached identifiers to their original tensor_id_str for server resolution
            cached_identifier_map = {
                identifier: identifier_to_tensor_id[identifier]
                for identifier in cache_query_result.cached_identifiers
                if identifier in identifier_to_tensor_id
            }
            
            # Map uncached tensors to their identifiers (so server can cache them)
            uncached_identifier_map = {
                tensor_id_str: tensor_identifiers[tensor_id_str]
                for tensor_id_str in tensors_to_send.keys()
                if tensor_id_str in tensor_identifiers
            }
            
            # ✅ PHASE 2: Negotiate protocol version and serialize
            try:
                from djinn.server.optimizations.fast_serialization import (
                    FastSerializer,
                    get_protocol_negotiator,
                    SerializationVersion
                )
                
                negotiator = get_protocol_negotiator()
                version = await negotiator.negotiate_version(target)
                
                # Serialize with negotiated version
                if version == SerializationVersion.V2_BINARY:
                    logger.debug("Using fast binary serialization (V2)")
                    # Extract subgraph from message_data
                    subgraph_dict = message_data.get('subgraph', {}) if isinstance(message_data, dict) else message_data
                    
                    try:
                        # Serialize with FastSerializer
                        data = FastSerializer.serialize_subgraph(
                            subgraph=subgraph_dict,
                            input_data=tensors_to_send,
                            version=version,
                            graph_id=graph_id,
                            cached_identifiers=list(cache_query_result.cached_identifiers),
                            cached_identifier_map=cached_identifier_map,
                            uncached_identifier_map=uncached_identifier_map,
                        )
                        # Message type for V2_BINARY format
                        message_type = (0x03).to_bytes(1, 'big')  # EXECUTE_SUBGRAPH coordinator
                    except (TypeError, ValueError, AttributeError) as e:
                        # FastSerializer failed (e.g., non-serializable objects), fall back to pickle
                        logger.warning(f"FastSerializer failed ({type(e).__name__}: {str(e)[:100]}), falling back to pickle")
                        version = SerializationVersion.V1_PICKLE  # Force V1 for fallback
                        raise  # Re-raise to trigger V1_PICKLE path below
                else:
                    # Fallback to pickle (V1, compatibility mode)
                    logger.debug("Using pickle serialization (V1, compatibility mode)")
                    message = {
                        'type': 0x03,  # EXECUTE_SUBGRAPH (from tcp_server.py)
                        'subgraph_data': message_data,  # Use processed graph data
                        'input_data': {
                            k: pickle.dumps(v) for k, v in tensors_to_send.items()  # Only uncached tensors
                        },
                        'cached_identifiers': list(cache_query_result.cached_identifiers),
                        'cached_identifier_map': cached_identifier_map,
                        'uncached_identifier_map': uncached_identifier_map,
                        'timeout': timeout
                    }
                    data = pickle.dumps(message)
                    message_type = (0x03).to_bytes(1, 'big')  # EXECUTE_SUBGRAPH coordinator
            except (ImportError, TypeError, ValueError, AttributeError):
                # Fallback if fast_serialization not available or failed
                logger.warning("FastSerializer unavailable or failed, using pickle")
                message = {
                    'type': 0x03,
                    'subgraph_data': message_data,
                    'input_data': {
                        k: pickle.dumps(v) for k, v in tensors_to_send.items()
                    },
                    'cached_identifiers': list(cache_query_result.cached_identifiers),
                    'cached_identifier_map': cached_identifier_map,
                    'uncached_identifier_map': uncached_identifier_map,
                    'timeout': timeout
                }
                data = pickle.dumps(message)
                message_type = (0x03).to_bytes(1, 'big')  # EXECUTE_SUBGRAPH coordinator
            data_size = len(data)
            # Check if message is too large
            # TODO: Implement proper model weight caching via tensor registry to avoid sending weights every time
            # For now, we use 8-byte size field to accommodate large models (GPT2-XL has ~5.8GB of weights)
            MAX_MESSAGE_SIZE = 16 * (1024**3)  # 16GB limit (8-byte size field supports up to 2^64)
            if data_size > MAX_MESSAGE_SIZE:
                raise RuntimeError(
                    f"Message too large: {data_size / (1024**3):.2f}GB "
                    f"(max {MAX_MESSAGE_SIZE / (1024**3):.2f}GB). "
                    f"Input tensors: {len(input_data)}, "
                    f"Subgraph operations: {len(subgraph.get('operations', []))}. "
                    f"Consider using model weight caching to reduce message size."
                )
            # Use 8-byte size field to support messages > 4GB
            size_bytes = data_size.to_bytes(8, 'big')

            # Protocol: message_type (1 byte) + length (8 bytes) + data
            # Updated to 8 bytes to support large messages > 4GB
            message_type = (0x03).to_bytes(1, 'big')  # EXECUTE_SUBGRAPH coordinator
            
            # ✅ PROFILING: Measure network transfer time (client→server)
            try:
                from djinn.server.profiling_context import get_profiler, record_phase
                profiler = get_profiler()
                if profiler and profiler.enabled:
                    network_start = time.perf_counter()
                    writer.write(message_type)
                    writer.write(size_bytes)
                    writer.write(data)
                    await writer.drain()
                    network_duration_ms = (time.perf_counter() - network_start) * 1000
                    profiler.record_phase('network_c2s', network_duration_ms, 
                                        metadata={'data_size_bytes': data_size})
                else:
                    writer.write(message_type)
                    writer.write(size_bytes)
                    writer.write(data)
                    await writer.drain()
            except ImportError:
                # Fallback if profiling not available
                writer.write(message_type)
                writer.write(size_bytes)
                writer.write(data)
                await writer.drain()

            # Receive result - read message type first (with timeout)
            try:
                msg_type_bytes = await asyncio.wait_for(
                    reader.readexactly(1),
                    timeout=timeout
                )
                msg_type = msg_type_bytes[0]
            except asyncio.TimeoutError:
                writer.close()
                await writer.wait_closed()
                raise RuntimeError(
                    f"Timeout waiting for server response after {timeout}s. "
                    f"Server may be processing a large subgraph ({len(subgraph.get('operations', []))} operations, "
                    f"{len(input_data)} inputs). Consider increasing timeout or reducing subgraph size."
                )

            if msg_type == 0x04:  # RESULT
                # ✅ PROFILING: Extract profiling data from response
                # Format: [profiling_size (4)] [profiling_json] [tensor_size (4)] [tensor_bytes]
                # Wrap all reads in timeout to prevent hanging
                remaining_timeout = timeout - (time.perf_counter() - start_time)
                if remaining_timeout <= 0:
                    raise RuntimeError(f"Timeout exceeded while reading response headers")
                
                profiling_size_bytes = await asyncio.wait_for(
                    reader.readexactly(4),
                    timeout=remaining_timeout
                )
                profiling_size = int.from_bytes(profiling_size_bytes, 'big')
                profiling_json_bytes = await asyncio.wait_for(
                    reader.readexactly(profiling_size) if profiling_size > 0 else asyncio.sleep(0),
                    timeout=remaining_timeout
                ) if profiling_size > 0 else b'{}'
                
                # Parse profiling data
                server_profiling_data = {}
                if profiling_size > 0:
                    try:
                        server_profiling_data = json.loads(profiling_json_bytes.decode('utf-8'))
                        logger.debug(f"📊 Received server-side profiling data: {server_profiling_data}")
                    except Exception as e:
                        logger.debug(f"Could not parse profiling data: {e}")
                
                # Store profiling data for later retrieval
                # Attach to response or store in a global context
                try:
                    from djinn.server.profiling_context import get_profiler
                    profiler = get_profiler()
                    if profiler and server_profiling_data:
                        # Merge server-side profiling data into client profiler
                        for phase_name, duration_ms in server_profiling_data.items():
                            profiler.record_phase(f"server_{phase_name}", duration_ms)
                except Exception as e:
                    logger.debug(f"Could not merge server profiling data: {e}")
                
                # Read tensor data (this may be large, so allow more time)
                remaining_timeout = timeout - (time.perf_counter() - start_time)
                if remaining_timeout <= 0:
                    raise RuntimeError(f"Timeout exceeded while reading tensor size")
                
                tensor_size_bytes = await asyncio.wait_for(
                    reader.readexactly(4),
                    timeout=remaining_timeout
                )
                tensor_size = int.from_bytes(tensor_size_bytes, 'big')
                
                remaining_timeout = timeout - (time.perf_counter() - start_time)
                if remaining_timeout <= 0:
                    raise RuntimeError(f"Timeout exceeded while reading tensor data (size: {tensor_size} bytes)")
                
                # ✅ PROFILING: Measure network transfer time (server→client)
                try:
                    from djinn.server.profiling_context import get_profiler
                    profiler = get_profiler()
                    if profiler and profiler.enabled:
                        network_start = time.perf_counter()
                        result_data = await asyncio.wait_for(
                            reader.readexactly(tensor_size),
                            timeout=remaining_timeout
                        )
                        network_duration_ms = (time.perf_counter() - network_start) * 1000
                        profiler.record_phase('network_s2c', network_duration_ms,
                                            metadata={'data_size_bytes': tensor_size})
                    else:
                        result_data = await asyncio.wait_for(
                            reader.readexactly(tensor_size),
                            timeout=remaining_timeout
                        )
                except ImportError:
                    # Fallback if profiling not available
                    result_data = await asyncio.wait_for(
                        reader.readexactly(tensor_size),
                        timeout=remaining_timeout
                    )
                
                # Result is tensor bytes - deserialize with optimized method
                try:
                    try:
                        from djinn.server.serialization import deserialize_tensor
                    except ImportError:
                        from ...server.serialization import deserialize_tensor
                    response = deserialize_tensor(result_data)
                except Exception as e:
                    logger.warning(f"Optimized deserialization failed: {e}, falling back to torch.load")
                    # Fallback to torch.load for tensor bytes
                    import io
                    result_buffer = io.BytesIO(result_data)
                    response = torch.load(result_buffer)
            elif msg_type == 0x05:  # ERROR
                # Error is string bytes - read size first (with timeout)
                remaining_timeout = timeout - (time.perf_counter() - start_time)
                if remaining_timeout <= 0:
                    raise RuntimeError(f"Timeout exceeded while reading error message")
                
                size_bytes = await asyncio.wait_for(
                    reader.readexactly(4),
                    timeout=remaining_timeout
                )
                size = int.from_bytes(size_bytes, 'big')
                error_data = await asyncio.wait_for(
                    reader.readexactly(size),
                    timeout=remaining_timeout
                )
                error_message = error_data.decode()
                raise RuntimeError(f"Server error: {error_message}")
            else:
                raise RuntimeError(f"Unknown message type: {msg_type}")

            writer.close()
            await writer.wait_closed()

            execution_time = time.perf_counter() - start_time
            logger.info(
                f"✅ Subgraph execution complete: {response.shape if hasattr(response, 'shape') else 'unknown'} in {execution_time:.3f}s "
                f"(cache query: {query_time:.1f}ms, transfer savings: {savings:.1f}%)"
            )

            return response

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"❌ Subgraph execution failed after {execution_time:.3f}s: {e}")
            raise RuntimeError(f"Subgraph execution failed: {e}") from e

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
        result_queue = self._result_manager.create_result_queue(result_id)

        try:
            # Step 2: Prepare metadata (WITHOUT tensor data - tensors sent separately)
            primary_tensor = inputs[0]
            # ✅ FIX: Use the transport's actual listening port
            client_port = self.transports['tcp'].data_port if 'tcp' in self.transports else self._central_config.network.data_port
            operation_metadata = create_operation_metadata(
                operation=operation,
                result_id=result_id,
                inputs=inputs,
                client_port=client_port  # ✅ Actual listening port
            )
            from .coordinator_transport import MetadataExtractor
            metadata_extractor = MetadataExtractor(self.scheduler)
            metadata = metadata_extractor.extract_metadata(primary_tensor, operation_metadata.to_dict())
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
            from .coordinator_transport import TransportSelector
            transport_selector = TransportSelector(self.transports)
            transport = transport_selector.select_transport(primary_tensor, metadata)

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
            if result_id in self._result_manager._result_queues:
                # Result arrived via callback during timeout handling
                try:
                    result = await asyncio.wait_for(
                        self._result_manager._result_queues[result_id].get(),
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
            logger.error(f"  Queue status: {self._result_manager.get_active_queue_count()} active queues")

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
            logger.error(f"  Queue cleanup: {self._result_manager.get_active_queue_count()} queues remain")

            # Re-raise with enhanced context
            raise RuntimeError(
                f"Remote execution failed for {operation}: {e}. "
                f"Target: {scheduled_target}, Inputs: {len(inputs)} tensors. "
                f"Check server status and network connectivity."
            ) from e
        finally:
            # Step 10: Cleanup queue
            self._result_manager.cleanup_queue(result_id)

    # Result management methods moved to coordinator_result.py
    # Kept as aliases for backward compatibility
    def _create_result_queue(self, result_id: str) -> asyncio.Queue:
        """Create result queue for operation (delegates to ResultManager)."""
        return self._result_manager.create_result_queue(result_id)
    
    async def _handle_result_received(self, result_id: str, result):
        """Handle result from transport (delegates to ResultManager)."""
        return await self._result_manager.handle_result_received(result_id, result)

    async def _handle_operation_request(self, transfer_id: str, tensor_or_tensors, metadata: Dict):
        """
        Handle incoming operation request (for client acting as server).

        This allows the client coordinator to also act as a server in peer-to-peer scenarios.
        """
        import logging
        logger = logging.getLogger(__name__)

        operation = metadata.get('operation')
        if not operation:
            logger.warning(f"Operation request missing operation field: {transfer_id}")
            return

        logger.info(f"Received operation request: {operation} for {transfer_id}")

        # For now, delegate to a simple execution (could be enhanced)
        try:
            # This is a basic implementation - in practice, might want to use the executor
            if isinstance(tensor_or_tensors, list):
                inputs = tensor_or_tensors
            else:
                inputs = [tensor_or_tensors]

            # Execute using universal dispatcher
            from ..frontend.core.universal_dispatcher import get_universal_dispatcher
            dispatcher = get_universal_dispatcher()
            result = dispatcher.dispatch(operation, inputs, {})

            # Send result back (using the metadata's client_port if available)
            result_id = metadata.get('result_id', f"{transfer_id}_result")
            client_port = metadata.get('client_port')

            if client_port:
                # This is a simplified response - in practice would need proper transport setup
                logger.info(f"Would send result back to client on port {client_port}")
            else:
                logger.warning(f"No client_port in operation metadata for {transfer_id}")

        except Exception as e:
            logger.error(f"Failed to execute operation {operation}: {e}")
            # In practice, would send error response back

    # Transport and metadata methods moved to coordinator_transport.py
    # Kept as aliases for backward compatibility
    def _select_transport(self, tensor, metadata):
        """Select best transport (delegates to TransportSelector)."""
        from .coordinator_transport import TransportSelector
        selector = TransportSelector(self.transports)
        return selector.select_transport(tensor, metadata)
    
    def _select_transport_for_metadata(self, metadata):
        """Select transport based on metadata (delegates to TransportSelector)."""
        from .coordinator_transport import TransportSelector
        selector = TransportSelector(self.transports)
        return selector.select_transport_for_metadata(metadata)
    
    def _extract_metadata(self, tensor, user_metadata):
        """Extract semantic metadata (delegates to MetadataExtractor)."""
        from .coordinator_transport import MetadataExtractor
        extractor = MetadataExtractor(self.scheduler)
        return extractor.extract_metadata(tensor, user_metadata)
    
    def _infer_phase(self, tensor):
        """Infer execution phase (delegates to MetadataExtractor)."""
        from .coordinator_transport import MetadataExtractor
        extractor = MetadataExtractor(self.scheduler)
        return extractor._infer_phase(tensor)
    
    async def stop(self):
        """Shutdown coordinator."""
        logger.info("Stopping DjinnCoordinator...")

        # Stop control plane
        if self.control_plane:
            await self.control_plane.stop()

        # Stop transports
        for name, transport in self.transports.items():
            if hasattr(transport, 'stop'):
                await transport.stop()

        logger.info("DjinnCoordinator stopped")