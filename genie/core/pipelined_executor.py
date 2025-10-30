"""
Network Pipelining Executor for Genie Framework

Overlaps network I/O with GPU computation by queueing multiple operations
and sending them while previous operations execute on the remote GPU.

Expected speedup: 2-5x for small, sequential operations (e.g., decode)
"""

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Deque, Any
from concurrent.futures import Future

import torch

logger = logging.getLogger(__name__)


@dataclass
class PipelineOperation:
    """Represents a single operation in the pipeline queue"""
    op_id: str
    operation: str
    input_tensor: torch.Tensor
    target: str
    
    # Timing information
    submitted_at: float = field(default_factory=time.perf_counter)
    sent_at: Optional[float] = None
    result_received_at: Optional[float] = None
    
    # Result handling
    result_future: asyncio.Future = field(default_factory=asyncio.Future)
    retry_count: int = 0
    max_retries: int = 3
    
    def time_to_send(self) -> float:
        """Time since operation was submitted"""
        return time.perf_counter() - self.submitted_at
    
    def time_in_flight(self) -> float:
        """Time since operation was sent"""
        if self.sent_at is None:
            return 0
        return time.perf_counter() - self.sent_at


@dataclass
class PipelineState:
    """Manages overall pipeline execution state"""
    queue: Deque[PipelineOperation] = field(default_factory=deque)
    pending: Dict[str, PipelineOperation] = field(default_factory=dict)
    completed: Dict[str, PipelineOperation] = field(default_factory=dict)
    max_concurrent: int = 5
    
    def can_send_next(self) -> bool:
        """Check if we can send another operation (not at max concurrent)"""
        return len(self.pending) < self.max_concurrent and len(self.queue) > 0
    
    def is_full(self) -> bool:
        """Check if queue is at capacity (backpressure point)"""
        return len(self.queue) >= self.max_concurrent * 2
    
    def mark_sent(self, op: PipelineOperation) -> None:
        """Move operation from queue to pending"""
        if op.op_id in [o.op_id for o in self.queue]:
            self.queue.popleft()
        self.pending[op.op_id] = op
        op.sent_at = time.perf_counter()
        logger.debug(f"Marked sent: {op.op_id}, pending now: {len(self.pending)}")
    
    def mark_result_received(self, op_id: str, result: torch.Tensor) -> None:
        """Move operation from pending to completed"""
        if op_id not in self.pending:
            logger.warning(f"Result received for unknown op: {op_id}")
            return
        
        op = self.pending.pop(op_id)
        op.result_received_at = time.perf_counter()
        self.completed[op_id] = op
        
        # Resolve the future with the result
        if not op.result_future.done():
            op.result_future.set_result(result)
        
        logger.debug(f"Completed: {op_id}, latency: {op.result_received_at - op.sent_at:.2f}ms")
    
    def mark_failed(self, op_id: str, error: Exception) -> None:
        """Mark operation as failed"""
        if op_id not in self.pending:
            return
        
        op = self.pending[op_id]
        op.retry_count += 1
        
        if op.retry_count < op.max_retries:
            # Re-queue for retry
            self.pending.pop(op_id)
            self.queue.append(op)
            logger.info(f"Retry {op.retry_count}/{op.max_retries}: {op_id}")
        else:
            # Give up after max retries
            self.pending.pop(op_id)
            if not op.result_future.done():
                op.result_future.set_exception(error)
            logger.error(f"Failed after {op.retry_count} retries: {op_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics"""
        return {
            "queued": len(self.queue),
            "pending": len(self.pending),
            "completed": len(self.completed),
            "capacity_used": len(self.pending) / self.max_concurrent,
        }


class PipelinedExecutor:
    """
    Executes operations with network pipelining.
    
    Key features:
    - Maintains queue of operations
    - Sends next operation while previous executes
    - Async result handling with callbacks
    - Automatic backpressure (max N concurrent)
    - Health monitoring and timeout detection
    """
    
    def __init__(self, coordinator, max_concurrent: int = 5):
        """
        Initialize pipelined executor.
        
        Args:
            coordinator: GenieCoordinator instance for remote execution
            max_concurrent: Maximum operations in flight simultaneously (default: 5)
        """
        self.coordinator = coordinator
        self.state = PipelineState(max_concurrent=max_concurrent)
        
        # Async event for signaling sender to wake up
        self._sender_event = asyncio.Event()
        self._sender_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Timeout settings
        self.operation_timeout = 30.0  # seconds
        self.monitor_interval = 5.0     # seconds
        
        logger.info(f"PipelinedExecutor initialized with max_concurrent={max_concurrent}")
    
    async def start(self) -> None:
        """Start background pipeline tasks"""
        if self._running:
            logger.warning("Pipeline already running")
            return
        
        self._running = True
        self._sender_task = asyncio.create_task(self._send_loop())
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Pipeline tasks started")
    
    async def stop(self) -> None:
        """Stop background pipeline tasks"""
        self._running = False
        
        if self._sender_task and not self._sender_task.done():
            self._sender_task.cancel()
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
        
        logger.info("Pipeline tasks stopped")
    
    async def execute_pipelined(
        self,
        lazy_tensor: Any,
        target: str = "localhost:5556"
    ) -> torch.Tensor:
        """
        Execute operation with pipelining.
        
        Queues the operation and returns a future that will be resolved
        when the result arrives from the remote server.
        
        Args:
            lazy_tensor: The lazy tensor operation to execute
            target: Remote server address
            
        Returns:
            Future that resolves to the result tensor
        """
        # Create pipeline operation
        op = PipelineOperation(
            op_id=str(uuid.uuid4()),
            operation=lazy_tensor.operation if hasattr(lazy_tensor, 'operation') else 'forward',
            input_tensor=self._materialize_inputs(lazy_tensor),
            target=target
        )
        
        # Apply backpressure if queue is full
        while self.state.is_full():
            await asyncio.sleep(0.001)
        
        # Queue the operation
        self.state.queue.append(op)
        logger.debug(f"Queued operation: {op.op_id}")
        
        # Signal sender to wake up if idle
        self._sender_event.set()
        
        # Wait for result
        try:
            result = await asyncio.wait_for(op.result_future, timeout=self.operation_timeout)
            return result
        except asyncio.TimeoutError:
            logger.error(f"Operation timeout: {op.op_id}")
            self.state.mark_failed(op.op_id, TimeoutError(f"Operation {op.op_id} timed out"))
            raise
    
    async def _send_loop(self) -> None:
        """
        Background task: send operations when pipeline ready.
        
        This loop continuously checks if we can send the next operation
        and sends it non-blocking when ready.
        """
        logger.info("Send loop started")
        
        while self._running:
            try:
                # Try to send next operation
                if self.state.can_send_next():
                    op = self.state.queue[0]
                    
                    try:
                        # Send operation to remote server (non-blocking)
                        await self._send_operation(op)
                        self.state.mark_sent(op)
                        
                    except Exception as e:
                        logger.error(f"Send failed for {op.op_id}: {e}")
                        self.state.mark_failed(op.op_id, e)
                
                # Wait for signal or sleep briefly
                try:
                    self._sender_event.clear()
                    await asyncio.wait_for(self._sender_event.wait(), timeout=0.01)
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue
                    
            except Exception as e:
                logger.error(f"Send loop error: {e}")
                await asyncio.sleep(0.01)
        
        logger.info("Send loop stopped")
    
    async def _monitor_loop(self) -> None:
        """
        Background task: monitor pipeline health and detect timeouts.
        
        Periodically checks for:
        - Operations that have been in flight too long
        - Pipeline utilization
        - Queue depth
        """
        logger.info("Monitor loop started")
        
        while self._running:
            try:
                # Check for timeouts
                now = time.perf_counter()
                for op_id, op in list(self.state.pending.items()):
                    if op.sent_at is not None:
                        elapsed = now - op.sent_at
                        if elapsed > self.operation_timeout:
                            logger.warning(f"Timeout detected for {op_id}: {elapsed:.2f}s")
                            self.state.mark_failed(op_id, TimeoutError(f"Operation timed out after {elapsed:.2f}s"))
                
                # Log pipeline stats
                stats = self.state.get_stats()
                if stats["queued"] > 0 or stats["pending"] > 0:
                    logger.debug(f"Pipeline stats: {stats}")
                
                await asyncio.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(self.monitor_interval)
        
        logger.info("Monitor loop stopped")
    
    async def _send_operation(self, op: PipelineOperation) -> None:
        """Send a single operation to the remote server (non-blocking)"""
        # Schedule the remote execution as a background task
        asyncio.create_task(
            self._execute_remote_and_callback(op)
        )
    
    async def _execute_remote_and_callback(self, op: PipelineOperation) -> None:
        """Execute operation remotely and handle result via callback"""
        try:
            # Call coordinator's async remote execution method
            result = await self.coordinator.execute_remote_operation_async(
                operation=op.operation,
                input_tensor=op.input_tensor,
                target=op.target,
                timeout=self.operation_timeout
            )
            
            # Mark result received
            self.state.mark_result_received(op.op_id, result)
            
            # Signal sender in case more operations can be sent
            self._sender_event.set()
            
        except Exception as e:
            logger.error(f"Remote execution failed for {op.op_id}: {e}")
            self.state.mark_failed(op.op_id, e)
            self._sender_event.set()
    
    def _materialize_inputs(self, lazy_tensor: Any) -> torch.Tensor:
        """Convert lazy tensor to concrete tensor for transmission"""
        if isinstance(lazy_tensor, torch.Tensor):
            return lazy_tensor
        
        # If it's a lazy tensor object, materialize it
        if hasattr(lazy_tensor, 'materialize'):
            return lazy_tensor.materialize()
        
        # Otherwise assume it's already materialized
        return lazy_tensor
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics for monitoring"""
        stats = self.state.get_stats()
        stats["running"] = self._running
        stats["timeout"] = self.operation_timeout
        return stats
    
    async def flush(self) -> None:
        """Wait for all pending operations to complete"""
        while len(self.state.queue) > 0 or len(self.state.pending) > 0:
            await asyncio.sleep(0.01)
        logger.info("Pipeline flushed - all operations completed")
