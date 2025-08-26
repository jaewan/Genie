"""
AsyncIO Bridge for Zero-Copy Transport

Provides Python async interface to the C++ zero-copy transport layer.
Handles coordination between Python control plane and C++ data plane.
"""

import asyncio
import ctypes
import uuid
import time
import logging
from typing import Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class TransferState(Enum):
    """Transfer state enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TransferRequest:
    """Transfer request descriptor"""
    transfer_id: str
    tensor: torch.Tensor
    target_node: str
    target_gpu: int
    priority: int = 0
    timeout_s: float = 30.0
    
    # Internal state
    state: TransferState = TransferState.PENDING
    start_time: float = 0
    end_time: float = 0
    bytes_transferred: int = 0
    error_message: str = ""
    
    # Async coordination
    future: Optional[asyncio.Future] = None
    
    @property
    def size(self) -> int:
        """Get tensor size in bytes"""
        return self.tensor.numel() * self.tensor.element_size()
    
    @property
    def duration(self) -> float:
        """Get transfer duration"""
        if self.end_time > 0 and self.start_time > 0:
            return self.end_time - self.start_time
        return 0
    
    @property
    def throughput_gbps(self) -> float:
        """Calculate throughput in Gbps"""
        if self.duration > 0:
            return (self.size * 8) / (self.duration * 1e9)
        return 0

class AsyncZeroCopyBridge:
    """
    Async bridge between Python control plane and C++ data plane.
    
    Provides high-level async interface for zero-copy tensor transfers
    while handling low-level coordination with DPDK threads.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize async bridge.
        
        Args:
            config: Configuration dictionary with:
                - lib_path: Path to C++ shared library
                - port_id: DPDK port ID
                - gpu_id: GPU device ID
                - use_gpu_direct: Enable GPU Direct
                - mtu: Maximum transmission unit
                - num_workers: Number of worker threads
        """
        self.config = config
        self.lib_path = config.get('lib_path', './libgenie_data_plane.so')
        self.port_id = config.get('port_id', 0)
        self.gpu_id = config.get('gpu_id', 0)
        self.use_gpu_direct = config.get('use_gpu_direct', True)
        self.mtu = config.get('mtu', 8192)
        self.num_workers = config.get('num_workers', 4)
        
        # Load C++ library
        self.lib = None
        self.transport = None
        
        # Transfer tracking
        self.active_transfers: Dict[str, TransferRequest] = {}
        self.transfer_lock = asyncio.Lock()
        
        # Completion handling
        self.completion_queue = asyncio.Queue()
        self.completion_task = None
        
        # Worker pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        
        # Statistics
        self.stats = {
            'transfers_sent': 0,
            'transfers_received': 0,
            'transfers_failed': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'avg_throughput_gbps': 0,
            'avg_latency_ms': 0
        }
        
        # Callbacks
        self.transfer_callbacks: Dict[str, Callable] = {}
        
        self.running = False
    
    async def initialize(self) -> bool:
        """Initialize the bridge and underlying transport"""
        try:
            # Load C++ library
            self.lib = ctypes.CDLL(self.lib_path)
            
            # Define C function signatures
            self._setup_c_bindings()
            
            # Create transport instance
            self.transport = self.lib.create_zero_copy_transport(
                self.port_id,
                self.gpu_id,
                self.use_gpu_direct,
                self.mtu
            )
            
            if not self.transport:
                logger.error("Failed to create zero-copy transport")
                return False
            
            # Initialize transport
            if not self.lib.transport_initialize(self.transport):
                logger.error("Failed to initialize transport")
                return False
            
            # Start completion handler
            self.completion_task = asyncio.create_task(self._completion_handler())
            
            self.running = True
            logger.info("AsyncZeroCopyBridge initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize bridge: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the bridge"""
        self.running = False
        
        # Cancel all active transfers
        async with self.transfer_lock:
            for transfer in self.active_transfers.values():
                if transfer.future and not transfer.future.done():
                    transfer.future.cancel()
            self.active_transfers.clear()
        
        # Stop completion handler
        if self.completion_task:
            self.completion_task.cancel()
            try:
                await self.completion_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown transport
        if self.lib and self.transport:
            self.lib.transport_shutdown(self.transport)
            self.lib.destroy_transport(self.transport)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("AsyncZeroCopyBridge shutdown complete")
    
    async def send_tensor(self, 
                         tensor: torch.Tensor,
                         target_node: str,
                         target_gpu: int = 0,
                         priority: int = 0,
                         timeout: float = 30.0) -> TransferRequest:
        """
        Send tensor using zero-copy transport.
        
        Args:
            tensor: PyTorch tensor to send (must be on GPU)
            target_node: Target node address (IP:port)
            target_gpu: Target GPU ID
            priority: Transfer priority
            timeout: Timeout in seconds
            
        Returns:
            TransferRequest with completion future
        """
        # Validate tensor
        if not tensor.is_cuda:
            raise ValueError("Tensor must be on GPU for zero-copy transfer")
        
        # Create transfer request
        transfer_id = str(uuid.uuid4())
        request = TransferRequest(
            transfer_id=transfer_id,
            tensor=tensor,
            target_node=target_node,
            target_gpu=target_gpu,
            priority=priority,
            timeout_s=timeout
        )
        
        # Create future for async completion
        request.future = asyncio.Future()
        
        # Register transfer
        async with self.transfer_lock:
            self.active_transfers[transfer_id] = request
        
        # Submit to C++ transport
        try:
            # Parse target address
            ip, port = self._parse_address(target_node)
            
            # Get tensor data pointer and size
            data_ptr = tensor.data_ptr()
            size = tensor.numel() * tensor.element_size()
            
            # Start transfer in executor (non-blocking)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._submit_transfer,
                transfer_id, data_ptr, size, ip, port
            )
            
            request.state = TransferState.IN_PROGRESS
            request.start_time = time.time()
            
            # Update stats
            self.stats['transfers_sent'] += 1
            self.stats['bytes_sent'] += size
            
            # Set timeout
            asyncio.create_task(self._handle_timeout(request))
            
        except Exception as e:
            request.state = TransferState.FAILED
            request.error_message = str(e)
            request.future.set_exception(e)
            
            async with self.transfer_lock:
                del self.active_transfers[transfer_id]
            
            self.stats['transfers_failed'] += 1
            
        return request
    
    async def receive_tensor(self,
                            transfer_id: str,
                            expected_size: int,
                            target_gpu: int = 0,
                            timeout: float = 30.0) -> torch.Tensor:
        """
        Prepare to receive tensor using zero-copy.
        
        Args:
            transfer_id: Transfer ID from control plane negotiation
            expected_size: Expected size in bytes
            target_gpu: GPU to receive tensor on
            timeout: Timeout in seconds
            
        Returns:
            Received tensor
        """
        # Create receive buffer on GPU
        torch.cuda.set_device(target_gpu)
        
        # Calculate tensor shape (assuming float32 for now)
        num_elements = expected_size // 4
        tensor = torch.empty(num_elements, dtype=torch.float32, device='cuda')
        
        # Prepare receive in C++ transport
        data_ptr = tensor.data_ptr()
        
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            self.executor,
            self._prepare_receive,
            transfer_id, data_ptr, expected_size
        )
        
        if not success:
            raise RuntimeError(f"Failed to prepare receive for {transfer_id}")
        
        # Create future for completion
        future = asyncio.Future()
        
        # Register for completion
        async with self.transfer_lock:
            request = TransferRequest(
                transfer_id=transfer_id,
                tensor=tensor,
                target_node="",  # Receiving
                target_gpu=target_gpu,
                timeout_s=timeout
            )
            request.future = future
            request.state = TransferState.IN_PROGRESS
            request.start_time = time.time()
            
            self.active_transfers[transfer_id] = request
        
        # Wait for completion
        try:
            await asyncio.wait_for(future, timeout=timeout)
            
            self.stats['transfers_received'] += 1
            self.stats['bytes_received'] += expected_size
            
            return tensor
            
        except asyncio.TimeoutError:
            request.state = TransferState.FAILED
            request.error_message = "Receive timeout"
            self.stats['transfers_failed'] += 1
            raise
    
    async def _completion_handler(self):
        """Handle transfer completions from C++ layer"""
        while self.running:
            try:
                # Poll for completions (would be callback from C++)
                await asyncio.sleep(0.001)  # 1ms polling
                
                # Check C++ completion queue
                completions = await self._poll_completions()
                
                for transfer_id, success, error_msg in completions:
                    await self._handle_completion(transfer_id, success, error_msg)
                    
            except Exception as e:
                logger.error(f"Error in completion handler: {e}")
    
    async def _poll_completions(self):
        """Poll C++ layer for completed transfers"""
        # This would call into C++ to check completion ring
        # For now, return empty list
        return []
    
    async def _handle_completion(self, transfer_id: str, success: bool, error_msg: str):
        """Handle transfer completion"""
        async with self.transfer_lock:
            request = self.active_transfers.get(transfer_id)
            if not request:
                return
            
            request.end_time = time.time()
            
            if success:
                request.state = TransferState.COMPLETED
                request.bytes_transferred = request.size
                
                if request.future and not request.future.done():
                    request.future.set_result(request.tensor)
                
                # Update statistics
                duration = request.duration
                if duration > 0:
                    throughput = request.throughput_gbps
                    
                    # Update rolling average
                    n = self.stats['transfers_sent'] + self.stats['transfers_received']
                    self.stats['avg_throughput_gbps'] = (
                        (self.stats['avg_throughput_gbps'] * (n - 1) + throughput) / n
                    )
                    self.stats['avg_latency_ms'] = (
                        (self.stats['avg_latency_ms'] * (n - 1) + duration * 1000) / n
                    )
            else:
                request.state = TransferState.FAILED
                request.error_message = error_msg
                
                if request.future and not request.future.done():
                    request.future.set_exception(RuntimeError(error_msg))
                
                self.stats['transfers_failed'] += 1
            
            # Call registered callbacks
            if transfer_id in self.transfer_callbacks:
                callback = self.transfer_callbacks[transfer_id]
                asyncio.create_task(callback(request))
            
            # Clean up
            del self.active_transfers[transfer_id]
    
    async def _handle_timeout(self, request: TransferRequest):
        """Handle transfer timeout"""
        await asyncio.sleep(request.timeout_s)
        
        async with self.transfer_lock:
            if request.transfer_id in self.active_transfers:
                if request.state == TransferState.IN_PROGRESS:
                    request.state = TransferState.FAILED
                    request.error_message = "Transfer timeout"
                    
                    if request.future and not request.future.done():
                        request.future.set_exception(asyncio.TimeoutError())
                    
                    # Cancel in C++ layer
                    self._cancel_transfer(request.transfer_id)
                    
                    del self.active_transfers[request.transfer_id]
                    self.stats['transfers_failed'] += 1
    
    def register_callback(self, transfer_id: str, callback: Callable):
        """Register completion callback for transfer"""
        self.transfer_callbacks[transfer_id] = callback
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        # Also get stats from C++ layer
        if self.lib and self.transport:
            c_stats = self._get_c_stats()
            self.stats.update(c_stats)
        
        return self.stats.copy()
    
    def print_stats(self):
        """Print statistics"""
        stats = self.get_stats()
        
        print("\n=== Async Zero-Copy Bridge Statistics ===")
        print(f"Transfers sent:     {stats['transfers_sent']}")
        print(f"Transfers received: {stats['transfers_received']}")
        print(f"Transfers failed:   {stats['transfers_failed']}")
        print(f"Bytes sent:         {stats['bytes_sent'] / (1024**3):.2f} GB")
        print(f"Bytes received:     {stats['bytes_received'] / (1024**3):.2f} GB")
        print(f"Avg throughput:     {stats['avg_throughput_gbps']:.2f} Gbps")
        print(f"Avg latency:        {stats['avg_latency_ms']:.2f} ms")
    
    # C++ binding helpers
    
    def _setup_c_bindings(self):
        """Setup ctypes bindings for C++ functions"""
        # Transport creation
        self.lib.create_zero_copy_transport.argtypes = [
            ctypes.c_uint16,  # port_id
            ctypes.c_int,      # gpu_id
            ctypes.c_bool,     # use_gpu_direct
            ctypes.c_size_t    # mtu
        ]
        self.lib.create_zero_copy_transport.restype = ctypes.c_void_p
        
        # Initialize
        self.lib.transport_initialize.argtypes = [ctypes.c_void_p]
        self.lib.transport_initialize.restype = ctypes.c_bool
        
        # Send
        self.lib.transport_send.argtypes = [
            ctypes.c_void_p,    # transport
            ctypes.c_char_p,    # transfer_id
            ctypes.c_void_p,    # gpu_ptr
            ctypes.c_size_t,    # size
            ctypes.c_uint32,    # dest_ip
            ctypes.c_uint16     # dest_port
        ]
        self.lib.transport_send.restype = ctypes.c_bool
        
        # Receive
        self.lib.transport_prepare_receive.argtypes = [
            ctypes.c_void_p,    # transport
            ctypes.c_char_p,    # transfer_id
            ctypes.c_void_p,    # gpu_ptr
            ctypes.c_size_t     # size
        ]
        self.lib.transport_prepare_receive.restype = ctypes.c_bool
    
    def _submit_transfer(self, transfer_id: str, data_ptr: int, 
                        size: int, ip: int, port: int) -> bool:
        """Submit transfer to C++ transport"""
        return self.lib.transport_send(
            self.transport,
            transfer_id.encode(),
            ctypes.c_void_p(data_ptr),
            size,
            ip,
            port
        )
    
    def _prepare_receive(self, transfer_id: str, data_ptr: int, size: int) -> bool:
        """Prepare receive in C++ transport"""
        return self.lib.transport_prepare_receive(
            self.transport,
            transfer_id.encode(),
            ctypes.c_void_p(data_ptr),
            size
        )
    
    def _cancel_transfer(self, transfer_id: str):
        """Cancel transfer in C++ layer"""
        if self.lib and self.transport:
            self.lib.transport_cancel(
                self.transport,
                transfer_id.encode()
            )
    
    def _get_c_stats(self) -> Dict[str, Any]:
        """Get statistics from C++ layer"""
        # Would call C++ stats function
        return {}
    
    def _parse_address(self, address: str) -> Tuple[int, int]:
        """Parse IP:port string to integers"""
        parts = address.split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid address format: {address}")
        
        # Convert IP to integer
        ip_parts = parts[0].split('.')
        if len(ip_parts) != 4:
            raise ValueError(f"Invalid IP address: {parts[0]}")
        
        ip_int = 0
        for part in ip_parts:
            ip_int = (ip_int << 8) | int(part)
        
        port = int(parts[1])
        
        return ip_int, port


class ZeroCopyTransferManager:
    """
    High-level manager for zero-copy transfers.
    
    Coordinates with control plane for transfer negotiation
    and uses AsyncZeroCopyBridge for actual data transfer.
    """
    
    def __init__(self, control_client, bridge_config: Dict[str, Any]):
        """
        Initialize transfer manager.
        
        Args:
            control_client: Control plane client for negotiation
            bridge_config: Configuration for AsyncZeroCopyBridge
        """
        self.control_client = control_client
        self.bridge = AsyncZeroCopyBridge(bridge_config)
        
        # Transfer tracking
        self.pending_transfers = {}
        
    async def initialize(self) -> bool:
        """Initialize the transfer manager"""
        return await self.bridge.initialize()
    
    async def shutdown(self):
        """Shutdown the transfer manager"""
        await self.bridge.shutdown()
    
    async def transfer_tensor(self,
                             tensor: torch.Tensor,
                             target_node: str,
                             target_gpu: int = 0) -> torch.Tensor:
        """
        Complete tensor transfer with control plane negotiation.
        
        Args:
            tensor: Tensor to transfer
            target_node: Target node address
            target_gpu: Target GPU ID
            
        Returns:
            Transferred tensor (for receives)
        """
        # Step 1: Negotiate transfer via control plane
        transfer_id = str(uuid.uuid4())
        
        negotiation = await self.control_client.request_transfer(
            transfer_id=transfer_id,
            size=tensor.numel() * tensor.element_size(),
            dtype=str(tensor.dtype),
            shape=list(tensor.shape),
            target_node=target_node,
            target_gpu=target_gpu
        )
        
        if not negotiation['accepted']:
            raise RuntimeError(f"Transfer rejected: {negotiation.get('reason')}")
        
        # Step 2: Perform zero-copy transfer
        request = await self.bridge.send_tensor(
            tensor=tensor,
            target_node=target_node,
            target_gpu=target_gpu
        )
        
        # Step 3: Wait for completion
        try:
            await request.future
            
            # Step 4: Confirm completion via control plane
            await self.control_client.confirm_transfer(transfer_id)
            
            logger.info(f"Transfer {transfer_id} completed: "
                       f"{request.throughput_gbps:.2f} Gbps")
            
            return tensor
            
        except Exception as e:
            # Report failure to control plane
            await self.control_client.report_failure(transfer_id, str(e))
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transfer statistics"""
        return self.bridge.get_stats()
    
    def print_stats(self):
        """Print transfer statistics"""
        self.bridge.print_stats()

