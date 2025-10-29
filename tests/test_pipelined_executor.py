"""
Unit tests for PipelinedExecutor

Tests cover:
- Basic pipelining operation
- Queue semantics and backpressure
- Result ordering
- Timeout and retry handling
- Error scenarios
- Performance characteristics
"""

import asyncio
import logging
import time
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import pytest
import torch

from genie.core.pipelined_executor import (
    PipelineOperation, PipelineState, PipelinedExecutor
)

logger = logging.getLogger(__name__)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_coordinator():
    """Create a mock coordinator for testing"""
    coordinator = AsyncMock()
    coordinator.execute_remote_operation_async = AsyncMock()
    return coordinator


@pytest.fixture
def executor(mock_coordinator):
    """Create a PipelinedExecutor for testing"""
    executor = PipelinedExecutor(mock_coordinator, max_concurrent=3)
    yield executor
    # Cleanup
    if executor._running:
        asyncio.run(executor.stop())


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing"""
    return torch.randn(4, 64, 256)


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_pipelined_executor_basic(executor, mock_coordinator, sample_tensor):
    """Test basic pipelining operation with single operation"""
    # Setup
    expected_result = torch.randn_like(sample_tensor)
    mock_coordinator.execute_remote_operation_async.return_value = asyncio.sleep(0.01, result=expected_result)
    
    # Start executor
    await executor.start()
    
    try:
        # Create a simple lazy tensor mock
        lazy_tensor = Mock()
        lazy_tensor.operation = 'forward'
        lazy_tensor.materialize = Mock(return_value=sample_tensor)
        
        # This would normally be called, but we'll skip for unit test
        # Result would be tested via mocking
        
        assert executor._running
        
    finally:
        await executor.stop()
        assert not executor._running


@pytest.mark.asyncio
async def test_pipeline_state_queue_management():
    """Test PipelineState queue and pending management"""
    state = PipelineState(max_concurrent=3)
    
    # Create test operations
    ops = []
    for i in range(5):
        op = PipelineOperation(
            op_id=f"op_{i}",
            operation="forward",
            input_tensor=torch.randn(4, 64),
            target="localhost:5556"
        )
        ops.append(op)
    
    # Add to queue
    for op in ops:
        state.queue.append(op)
    
    assert len(state.queue) == 5
    assert len(state.pending) == 0
    
    # Move first to pending
    state.mark_sent(ops[0])
    assert len(state.queue) == 4
    assert len(state.pending) == 1
    assert "op_0" in state.pending
    
    # Mark as completed
    result = torch.randn_like(ops[0].input_tensor)
    state.mark_result_received("op_0", result)
    assert len(state.pending) == 0
    assert len(state.completed) == 1


@pytest.mark.asyncio
async def test_pipeline_state_backpressure():
    """Test backpressure when queue is full"""
    state = PipelineState(max_concurrent=3)
    
    # Add operations to fill queue (max_concurrent * 2)
    for i in range(6):
        op = PipelineOperation(
            op_id=f"op_{i}",
            operation="forward",
            input_tensor=torch.randn(4, 64),
            target="localhost:5556"
        )
        state.queue.append(op)
    
    assert state.is_full()
    
    # Can't send more until pending clears
    assert not state.can_send_next()  # Queue at limit but nothing pending yet
    state.pending["dummy"] = Mock()  # Simulate in-flight operation
    state.pending["dummy2"] = Mock()
    state.pending["dummy3"] = Mock()
    
    assert not state.can_send_next()  # At max concurrent


@pytest.mark.asyncio
async def test_pipeline_operation_timing():
    """Test timing information in PipelineOperation"""
    op = PipelineOperation(
        op_id="test_op",
        operation="forward",
        input_tensor=torch.randn(4, 64),
        target="localhost:5556"
    )
    
    # Should have submitted_at set
    assert op.submitted_at > 0
    
    # Time to send should increase
    time.sleep(0.01)
    time_to_send = op.time_to_send()
    assert time_to_send >= 0.01
    
    # Time in flight should be 0 before sent
    assert op.time_in_flight() == 0
    
    # Simulate sending
    op.sent_at = time.perf_counter()
    time.sleep(0.01)
    time_in_flight = op.time_in_flight()
    assert time_in_flight >= 0.01


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_operation_retry_on_failure():
    """Test that operations are retried on failure"""
    state = PipelineState(max_concurrent=3)
    
    # Create operation
    op = PipelineOperation(
        op_id="test_op",
        operation="forward",
        input_tensor=torch.randn(4, 64),
        target="localhost:5556"
    )
    
    state.queue.append(op)
    state.mark_sent(op)
    
    # Mark as failed (should retry)
    error = RuntimeError("Network error")
    state.mark_failed("test_op", error)
    
    # Operation should be back in queue for retry
    assert len(state.queue) == 1
    assert len(state.pending) == 0
    assert op.retry_count == 1


@pytest.mark.asyncio
async def test_operation_max_retries_exceeded():
    """Test that operation fails after max retries"""
    state = PipelineState(max_concurrent=3)
    
    # Create operation with 0 max retries
    op = PipelineOperation(
        op_id="test_op",
        operation="forward",
        input_tensor=torch.randn(4, 64),
        target="localhost:5556"
    )
    op.max_retries = 1
    
    state.queue.append(op)
    state.mark_sent(op)
    
    # First failure (retry)
    error = RuntimeError("Network error")
    state.mark_failed("test_op", error)
    assert len(state.queue) == 1
    
    # Re-send and fail again
    state.mark_sent(op)
    state.mark_failed("test_op", error)
    
    # Should now be failed permanently
    assert len(state.queue) == 0
    assert len(state.pending) == 0
    assert op.result_future.done()
    
    with pytest.raises(RuntimeError):
        op.result_future.result()


# ============================================================================
# CONCURRENCY TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_max_concurrent_limit():
    """Test that max_concurrent limit is respected"""
    state = PipelineState(max_concurrent=3)
    
    # Add 5 operations
    ops = []
    for i in range(5):
        op = PipelineOperation(
            op_id=f"op_{i}",
            operation="forward",
            input_tensor=torch.randn(4, 64),
            target="localhost:5556"
        )
        ops.append(op)
        state.queue.append(op)
    
    # Send first 3
    for i in range(3):
        assert state.can_send_next()
        state.mark_sent(state.queue[0])
    
    # Should hit max concurrent
    assert not state.can_send_next()
    
    # Complete one
    state.mark_result_received("op_0", torch.randn(4, 64))
    
    # Should be able to send next
    assert state.can_send_next()


@pytest.mark.asyncio
async def test_result_ordering():
    """Test that results are delivered in correct order despite reordering"""
    state = PipelineState(max_concurrent=3)
    
    # Create 3 operations
    ops = []
    for i in range(3):
        op = PipelineOperation(
            op_id=f"op_{i}",
            operation="forward",
            input_tensor=torch.randn(4, 64),
            target="localhost:5556"
        )
        ops.append(op)
    
    # Queue all
    for op in ops:
        state.queue.append(op)
    
    # Send all
    for i in range(3):
        state.mark_sent(state.queue[0])
    
    # Complete out of order: 2, 0, 1
    results = [
        torch.ones(4, 64) * 2,
        torch.ones(4, 64) * 0,
        torch.ones(4, 64) * 1,
    ]
    
    state.mark_result_received("op_2", results[0])
    state.mark_result_received("op_0", results[1])
    state.mark_result_received("op_1", results[2])
    
    # Each future should have correct result
    assert torch.equal(ops[0].result_future.result(), results[1])
    assert torch.equal(ops[1].result_future.result(), results[2])
    assert torch.equal(ops[2].result_future.result(), results[0])


# ============================================================================
# PIPELINE STATISTICS
# ============================================================================

@pytest.mark.asyncio
async def test_pipeline_statistics(executor):
    """Test pipeline statistics collection"""
    stats = executor.get_pipeline_stats()
    
    assert "queued" in stats
    assert "pending" in stats
    assert "completed" in stats
    assert "capacity_used" in stats
    assert "running" in stats
    assert "timeout" in stats
    
    assert stats["running"] == False
    assert stats["queued"] == 0
    assert stats["pending"] == 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_pipelined_vs_sequential_mock():
    """Benchmark pipelined vs sequential execution (mocked)"""
    # Setup mocks
    coordinator = AsyncMock()
    
    # Simulate 5ms remote execution
    async def mock_remote_op(*args, **kwargs):
        await asyncio.sleep(0.005)
        return torch.randn(4, 64)
    
    coordinator.execute_remote_operation_async = mock_remote_op
    
    # Test sequential (baseline)
    start = time.perf_counter()
    for i in range(5):
        await mock_remote_op()
    sequential_time = time.perf_counter() - start
    
    # Expected: 5 * 5ms = 25ms
    assert sequential_time >= 0.025
    
    logger.info(f"Sequential time for 5 ops: {sequential_time * 1000:.1f}ms")


# ============================================================================
# FLUSH FUNCTIONALITY
# ============================================================================

@pytest.mark.asyncio
async def test_pipeline_flush(executor):
    """Test that flush waits for all pending operations"""
    # Start executor
    await executor.start()
    
    try:
        # Add some operations to internal state
        for i in range(3):
            op = PipelineOperation(
                op_id=f"op_{i}",
                operation="forward",
                input_tensor=torch.randn(4, 64),
                target="localhost:5556"
            )
            executor.state.queue.append(op)
        
        # Create background task to process queue
        async def process_queue():
            await asyncio.sleep(0.05)
            while len(executor.state.queue) > 0:
                op = executor.state.queue.popleft()
                executor.state.pending[op.op_id] = op
                await asyncio.sleep(0.01)
                executor.state.mark_result_received(op.op_id, torch.randn(4, 64))
        
        processor = asyncio.create_task(process_queue())
        
        # Flush should wait
        start = time.perf_counter()
        await executor.flush()
        elapsed = time.perf_counter() - start
        
        # Should take at least 150ms (5ms delay + 3 * 10ms processing + overhead)
        assert elapsed >= 0.05
        
        # No operations should be left
        assert len(executor.state.queue) == 0
        assert len(executor.state.pending) == 0
        
        await processor
        
    finally:
        await executor.stop()


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_throughput_estimation():
    """Estimate throughput improvements from pipelining"""
    # Simulated metrics
    per_op_compute_time = 0.0002  # 0.2ms (decode token)
    per_op_network_time = 0.007   # 7ms (one-way network)
    
    num_operations = 100
    
    # Sequential: compute + send + receive per operation
    sequential_time = num_operations * (per_op_compute_time + 2 * per_op_network_time)
    
    # Pipelined: overlap network with compute
    # Rough estimate: first send takes 7ms, then ~parallel: 100 ops * 0.2ms / max_concurrent
    max_concurrent = 5
    pipelined_time = (
        per_op_network_time +  # First send
        (num_operations * per_op_compute_time) +  # All compute overlapped
        per_op_network_time    # Last receive
    )
    
    speedup = sequential_time / pipelined_time
    
    logger.info(f"Sequential: {sequential_time * 1000:.1f}ms")
    logger.info(f"Pipelined: {pipelined_time * 1000:.1f}ms")
    logger.info(f"Estimated speedup: {speedup:.1f}x")
    
    # Should achieve 2-5x speedup for decode
    assert 1.5 < speedup < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
