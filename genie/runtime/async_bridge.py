"""
AsyncIO Bridge for Zero-Copy Transport

This module provides an async-friendly interface to the high-performance
C++ data plane, bridging the gap between Python's async/await paradigm
and the low-level DPDK threads.

Key features:
- Async/await interface for tensor transfers
- Future-based completion tracking
- Progress callbacks with backpressure
- Automatic cleanup and resource management
"""

from __future__ import annotations

import asyncio
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any, List
import logging
import time
import uuid

logger = logging.getLogger(__name__)

@dataclass
class TransferFuture:
    """Future-like object for tracking tensor transfers"""
    transfer_id: str
    tensor_id: str
    size: int
    start_time: float = field(default_factory=time.time)
    
    # Async completion
    _future: asyncio.Future = field(default_factory=asyncio.Future)
    _progress_callbacks: List[Callable] = field(default_factory=list)
    _cleanup_callbacks: List[Callable] = field(default_factory=list)
    
    # Statistics
    bytes_transferred: int = 0
    packets_sent: int = 0
    packets_received: int = 0
    
    async def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for transfer completion"""
        try:
            await asyncio.wait_for(self._future, timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
    
    def done(self) -> bool:
        """Check if transfer is complete"""
        return self._future.done()
    
    def result(self) -> Any:
        """Get transfer result (blocks if not complete)"""
        return self._future.result()
    
    def cancel(self) -> bool:
        """Cancel the transfer"""
        return self._future.cancel()
    
    def add_progress_callback(self, callback: Callable):
        """Add callback for progress updates"""
        self._progress_callbacks.append(callback)
    
    def add_cleanup_callback(self, callback: Callable):
        """Add callback for cleanup on completion"""
        self._cleanup_callbacks.append(callback)
    
    def _complete(self, result: Any = None):
        """Mark transfer as complete"""
        if not self._future.done():
            self._future.set_result(result)
            
        # Run cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Cleanup callback error: {e}")
    
    def _fail(self, error: Exception):
        """Mark transfer as failed"""
        if not self._future.done():
            self._future.set_exception(error)
            
        # Run cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Cleanup callback error: {e}")
    
    def _update_progress(self, bytes_transferred: int, packets: int):
        """Update transfer progress"""
        self.bytes_transferred = bytes_transferred
        
        # Calculate progress percentage
        progress = min(100, int(bytes_transferred * 100 / self.size))
        
        # Run progress callbacks
        for callback in self._progress_callbacks:
            try:
                callback(self, progress)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

class AsyncBridge:
    """
    Bridges async Python code with the C++ data plane threads
    
    This class manages the interaction between Python's asyncio event loop
    and the high-performance C++ DPDK threads, providing a clean async
    interface while maintaining zero-copy performance.
    """
    
    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        self.loop = loop or asyncio.get_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="genie-bridge")
        
        # Transfer tracking
        self.active_transfers: Dict[str, TransferFuture] = {}
        self.transfer_lock = threading.RLock()
        
        # Completion queue for inter-thread communication
        self.completion_queue: asyncio.Queue = asyncio.Queue()
        self.completion_task: Optional[asyncio.Task] = None
        
        # Weak references for automatic cleanup
        self.tensor_refs: Dict[str, weakref.ReferenceType] = {}
        
        # Statistics
        self.total_transfers = 0
        self.completed_transfers = 0
        self.failed_transfers = 0
    
    async def start(self):
        """Start the async bridge"""
        # Start completion handler
        self.completion_task = asyncio.create_task(self._process_completions())
        logger.info("AsyncBridge started")
    
    async def stop(self):
        """Stop the async bridge"""
        # Cancel all active transfers
        with self.transfer_lock:
            for transfer in self.active_transfers.values():
                transfer.cancel()
            self.active_transfers.clear()
        
        # Stop completion handler
        if self.completion_task:
            self.completion_task.cancel()
            try:
                await self.completion_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        logger.info("AsyncBridge stopped")
    
    async def transfer_tensor(self, 
                             tensor: Any,
                             target_node: str,
                             priority: int = 1,
                             timeout: Optional[float] = None,
                             progress_callback: Optional[Callable] = None) -> TransferFuture:
        """
        Initiate an async tensor transfer
        
        Args:
            tensor: PyTorch tensor to transfer
            target_node: Target node identifier
            priority: Transfer priority (higher = more important)
            timeout: Optional timeout in seconds
            progress_callback: Optional callback for progress updates
            
        Returns:
            TransferFuture for tracking the transfer
        """
        # Generate IDs
        transfer_id = str(uuid.uuid4())
        tensor_id = str(uuid.uuid4())
        
        # Get tensor info
        size = tensor.numel() * tensor.element_size()
        
        # Create transfer future
        future = TransferFuture(
            transfer_id=transfer_id,
            tensor_id=tensor_id,
            size=size
        )
        
        if progress_callback:
            future.add_progress_callback(progress_callback)
        
        # Keep weak reference to tensor
        self.tensor_refs[transfer_id] = weakref.ref(tensor, 
            lambda ref: self._tensor_deallocated(transfer_id))
        
        # Track transfer
        with self.transfer_lock:
            self.active_transfers[transfer_id] = future
            self.total_transfers += 1
        
        # Submit to C++ data plane (via executor to avoid blocking)
        asyncio.create_task(
            self._submit_transfer(future, tensor, target_node, priority, timeout)
        )
        
        return future
    
    async def _submit_transfer(self,
                              future: TransferFuture,
                              tensor: Any,
                              target_node: str,
                              priority: int,
                              timeout: Optional[float]):
        """Submit transfer to C++ data plane"""
        try:
            # This would call into the C++ data plane via ctypes
            # For now, simulate the submission
            from .transport_coordinator import get_transport_coordinator
            
            coordinator = get_transport_coordinator()
            if coordinator and coordinator.data_plane:
                # Submit to C++ data plane
                gpu_ptr = tensor.data_ptr()
                success = await self.loop.run_in_executor(
                    self.executor,
                    coordinator.data_plane.send_tensor,
                    future.transfer_id,
                    future.tensor_id,
                    gpu_ptr,
                    future.size,
                    target_node
                )
                
                if not success:
                    raise RuntimeError("Failed to submit transfer to data plane")
                
                # Set timeout if specified
                if timeout:
                    asyncio.create_task(self._enforce_timeout(future, timeout))
                
            else:
                # Fallback simulation
                await self._simulate_transfer(future)
                
        except Exception as e:
            logger.error(f"Failed to submit transfer: {e}")
            future._fail(e)
            self._cleanup_transfer(future.transfer_id)
    
    async def _simulate_transfer(self, future: TransferFuture):
        """Simulate transfer for testing"""
        try:
            # Simulate progress updates
            for progress in [25, 50, 75, 100]:
                await asyncio.sleep(0.1)
                future._update_progress(
                    int(future.size * progress / 100),
                    progress
                )
            
            # Complete transfer
            await self.completion_queue.put((future.transfer_id, True, None))
            
        except Exception as e:
            await self.completion_queue.put((future.transfer_id, False, str(e)))
    
    async def _process_completions(self):
        """Process completion notifications from C++ threads"""
        while True:
            try:
                # Get completion from queue
                transfer_id, success, error = await self.completion_queue.get()
                
                # Find transfer future
                with self.transfer_lock:
                    future = self.active_transfers.get(transfer_id)
                
                if future:
                    if success:
                        future._complete()
                        self.completed_transfers += 1
                    else:
                        future._fail(RuntimeError(error or "Transfer failed"))
                        self.failed_transfers += 1
                    
                    # Cleanup
                    self._cleanup_transfer(transfer_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing completion: {e}")
    
    async def _enforce_timeout(self, future: TransferFuture, timeout: float):
        """Enforce timeout on transfer"""
        try:
            await asyncio.sleep(timeout)
            
            if not future.done():
                future._fail(asyncio.TimeoutError(f"Transfer timed out after {timeout}s"))
                self._cleanup_transfer(future.transfer_id)
                
        except asyncio.CancelledError:
            pass
    
    def _cleanup_transfer(self, transfer_id: str):
        """Clean up completed transfer"""
        with self.transfer_lock:
            if transfer_id in self.active_transfers:
                del self.active_transfers[transfer_id]
            
            if transfer_id in self.tensor_refs:
                del self.tensor_refs[transfer_id]
    
    def _tensor_deallocated(self, transfer_id: str):
        """Handle tensor deallocation during transfer"""
        with self.transfer_lock:
            future = self.active_transfers.get(transfer_id)
            if future and not future.done():
                logger.warning(f"Tensor deallocated during transfer {transfer_id}")
                future._fail(RuntimeError("Tensor deallocated during transfer"))
                self._cleanup_transfer(transfer_id)
    
    def notify_completion(self, transfer_id: str, success: bool, error: Optional[str] = None):
        """Called by C++ threads to notify completion"""
        # Schedule on event loop
        asyncio.run_coroutine_threadsafe(
            self.completion_queue.put((transfer_id, success, error)),
            self.loop
        )
    
    def update_progress(self, transfer_id: str, bytes_transferred: int, packets: int):
        """Called by C++ threads to update progress"""
        with self.transfer_lock:
            future = self.active_transfers.get(transfer_id)
            if future:
                # Update in event loop to avoid threading issues
                self.loop.call_soon_threadsafe(
                    future._update_progress,
                    bytes_transferred,
                    packets
                )
    
    async def batch_transfer(self,
                            tensors: List[Any],
                            target_node: str,
                            max_concurrent: int = 10) -> List[TransferFuture]:
        """
        Transfer multiple tensors with concurrency control
        
        Args:
            tensors: List of tensors to transfer
            target_node: Target node identifier
            max_concurrent: Maximum concurrent transfers
            
        Returns:
            List of TransferFutures
        """
        futures = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def transfer_with_limit(tensor):
            async with semaphore:
                return await self.transfer_tensor(tensor, target_node)
        
        # Start all transfers
        tasks = [transfer_with_limit(tensor) for tensor in tensors]
        futures = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_futures = []
        for i, future in enumerate(futures):
            if isinstance(future, Exception):
                logger.error(f"Failed to start transfer for tensor {i}: {future}")
            else:
                valid_futures.append(future)
        
        return valid_futures
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics"""
        with self.transfer_lock:
            return {
                'active_transfers': len(self.active_transfers),
                'total_transfers': self.total_transfers,
                'completed_transfers': self.completed_transfers,
                'failed_transfers': self.failed_transfers,
                'success_rate': self.completed_transfers / max(1, self.total_transfers)
            }

# Global async bridge instance
_async_bridge: Optional[AsyncBridge] = None

async def get_async_bridge() -> AsyncBridge:
    """Get or create global async bridge"""
    global _async_bridge
    if _async_bridge is None:
        _async_bridge = AsyncBridge()
        await _async_bridge.start()
    return _async_bridge

async def shutdown_async_bridge():
    """Shutdown global async bridge"""
    global _async_bridge
    if _async_bridge:
        await _async_bridge.stop()
        _async_bridge = None
