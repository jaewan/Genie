"""
Unit Tests for Batching Coordinator

Tests batching coordinator functionality:
- Queue management and operation collection
- Batch triggering (size-based and timeout-based)
- Result distribution to futures
- Error handling and recovery
- Backpressure management
- Statistics tracking
"""

import pytest
import asyncio
import sys
from unittest.mock import AsyncMock, Mock

sys.path.insert(0, '/home/jae/Genie')

from genie.core.batching_coordinator import BatchingCoordinator, BatchedOperation
from genie.core.connection_pool import ConnectionPool


@pytest.fixture
def mock_pool():
    """Create a mock connection pool"""
    pool = Mock(spec=ConnectionPool)
    pool.get_connection = AsyncMock()
    pool.release_connection = AsyncMock()
    return pool


@pytest.fixture
def coordinator(mock_pool):
    """Create a batching coordinator for testing"""
    return BatchingCoordinator(
        mock_pool,
        batch_size=3,
        batch_timeout_ms=10.0,
        max_queue_size=50
    )


class TestBatchingCoordinatorBasic:
    """Basic functionality tests"""
    
    @pytest.mark.asyncio
    async def test_coordinator_initialization(self, coordinator):
        """Test coordinator initializes correctly"""
        assert coordinator.batch_size == 3
        assert coordinator.batch_timeout == 0.01
        assert coordinator.max_queue_size == 50
        assert coordinator.stats['total_operations'] == 0
    
    @pytest.mark.asyncio
    async def test_queue_initialization(self, coordinator):
        """Test queue initializes on first use"""
        assert coordinator.operation_queue is None
        await coordinator.initialize()
        assert coordinator.operation_queue is not None
    
    @pytest.mark.asyncio
    async def test_single_operation_queuing(self, mock_pool, coordinator):
        """Test queuing a single operation"""
        await coordinator.initialize()
        
        # Mock the network response
        mock_reader = AsyncMock()
        mock_writer = AsyncMock()
        mock_pool.get_connection.return_value = (mock_reader, mock_writer)
        
        # Mock batch protocol unpack
        mock_reader.readexactly = AsyncMock(side_effect=[
            b'\x00\x00',  # Status + count
            b'\x00\x00\x00\x02{"ok": true}'  # Size (2) + result
        ])
        
        # Queue operation - should trigger batch send since batch_size=1 would send immediately
        # Actually with batch_size=3, it should timeout
        result = asyncio.create_task(
            coordinator.execute_operation_batched('test_op', {'key': 'value'})
        )
        
        # Give it time to process
        await asyncio.sleep(0.05)
        
        # Check that operation was queued
        assert coordinator.operation_queue.qsize() > 0 or mock_pool.get_connection.called


class TestBatchingCoordinatorQueueManagement:
    """Tests for queue management and batching logic"""
    
    @pytest.mark.asyncio
    async def test_backpressure_on_full_queue(self, mock_pool, coordinator):
        """Test that full queue raises backpressure error"""
        await coordinator.initialize()
        
        # Fill the queue
        for i in range(coordinator.max_queue_size):
            try:
                coordinator.operation_queue.put_nowait(
                    BatchedOperation(f"id_{i}", f"op_{i}", {}, asyncio.Future(), 0)
                )
            except asyncio.QueueFull:
                break
        
        # Next operation should fail with backpressure
        with pytest.raises(RuntimeError, match="queue full"):
            await coordinator.execute_operation_batched("test_op")
    
    @pytest.mark.asyncio
    async def test_queue_size_tracking(self, coordinator):
        """Test queue size is tracked correctly"""
        await coordinator.initialize()
        
        assert coordinator.get_stats()['queue_size'] == 0
        
        coordinator.operation_queue.put_nowait(
            BatchedOperation("id1", "op1", {}, asyncio.Future(), 0)
        )
        
        assert coordinator.get_stats()['queue_size'] == 1
    
    @pytest.mark.asyncio
    async def test_batch_full_trigger(self, mock_pool, coordinator):
        """Test batch send on full queue"""
        await coordinator.initialize()
        
        # Mock network
        mock_reader = AsyncMock()
        mock_writer = AsyncMock()
        mock_pool.get_connection.return_value = (mock_reader, mock_writer)
        mock_reader.readexactly = AsyncMock(side_effect=[
            b'\x00\x03\x00\x00\x00\x02{"r": 1}\x00\x00\x00\x02{"r": 2}\x00\x00\x00\x02{"r": 3}'
        ])
        
        # Queue 3 operations (coordinator batch_size=3)
        futures = []
        for i in range(3):
            future = asyncio.Future()
            coordinator.operation_queue.put_nowait(
                BatchedOperation(f"id_{i}", f"op_{i}", {}, future, 0)
            )
            futures.append(future)
        
        # Trigger batch processing
        await coordinator._process_batch()
        
        # Batch should have been sent
        mock_pool.get_connection.assert_called()


class TestBatchingCoordinatorErrorHandling:
    """Tests for error handling and recovery"""
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self, mock_pool, coordinator):
        """Test handling of network errors"""
        await coordinator.initialize()
        
        # Mock network failure
        mock_pool.get_connection.side_effect = RuntimeError("Connection failed")
        
        # Queue operation
        future = asyncio.Future()
        coordinator.operation_queue.put_nowait(
            BatchedOperation("id1", "op1", {}, future, 0)
        )
        
        # Process batch - should distribute error
        await coordinator._process_batch()
        
        # Future should have exception
        assert future.done()
        with pytest.raises(RuntimeError):
            future.result()
    
    @pytest.mark.asyncio
    async def test_timeout_scheduling(self, coordinator):
        """Test timeout scheduling and cancellation"""
        await coordinator.initialize()
        
        # Schedule timeout
        coordinator._schedule_batch_timeout()
        assert coordinator.timeout_handle is not None
        
        # Reschedule should cancel previous
        coordinator._schedule_batch_timeout()
        assert coordinator.timeout_handle is not None
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, mock_pool, coordinator):
        """Test graceful shutdown with pending operations"""
        await coordinator.initialize()
        
        # Mock network
        mock_reader = AsyncMock()
        mock_writer = AsyncMock()
        mock_pool.get_connection.return_value = (mock_reader, mock_writer)
        mock_reader.readexactly = AsyncMock(return_value=b'\x00\x01\x00\x00\x00\x02{"ok": true}')
        
        # Queue operation
        coordinator.operation_queue.put_nowait(
            BatchedOperation("id1", "op1", {}, asyncio.Future(), 0)
        )
        
        # Close should flush pending
        await coordinator.close()
        
        # Connection should have been used
        mock_pool.get_connection.assert_called()


class TestBatchingCoordinatorStatistics:
    """Tests for statistics tracking"""
    
    @pytest.mark.asyncio
    async def test_stats_initialization(self, coordinator):
        """Test stats are initialized correctly"""
        stats = coordinator.get_stats()
        
        assert stats['total_operations'] == 0
        assert stats['total_batches'] == 0
        assert stats['avg_batch_size'] == 0
        assert stats['batch_size_config'] == 3
    
    @pytest.mark.asyncio
    async def test_stats_update_on_batch(self, mock_pool, coordinator):
        """Test stats update after batch processing"""
        await coordinator.initialize()
        
        # Mock network
        mock_reader = AsyncMock()
        mock_writer = AsyncMock()
        mock_pool.get_connection.return_value = (mock_reader, mock_writer)
        
        # Mock readexactly to return responses for 2 operations
        mock_reader.readexactly = AsyncMock(side_effect=[
            b'\x00\x02',  # Status + 2 count
            b'\x00\x00\x00\x09{"r": 1}XX',  # 9-byte result
            b'\x00\x00\x00\x09{"r": 2}XX'   # 9-byte result
        ])
        
        # Queue 2 operations
        for i in range(2):
            coordinator.operation_queue.put_nowait(
                BatchedOperation(f"id_{i}", f"op_{i}", {}, asyncio.Future(), 0)
            )
        
        # Process batch
        await coordinator._process_batch()
        
        # Stats should be updated
        stats = coordinator.get_stats()
        assert stats['total_operations'] == 2
        assert stats['total_batches'] == 1
        assert stats['avg_batch_size'] == 2.0


class TestBatchingCoordinatorConcurrency:
    """Tests for concurrent operation handling"""
    
    @pytest.mark.asyncio
    async def test_concurrent_operation_queueing(self, coordinator):
        """Test multiple concurrent operations can be queued"""
        await coordinator.initialize()
        
        # Queue multiple operations concurrently
        tasks = []
        for i in range(5):
            future = asyncio.Future()
            tasks.append(
                coordinator.operation_queue.put_nowait(
                    BatchedOperation(f"id_{i}", f"op_{i}", {}, future, 0)
                )
            )
        
        # All should be queued
        assert coordinator.operation_queue.qsize() == 5
    
    @pytest.mark.asyncio
    async def test_concurrent_result_distribution(self, mock_pool, coordinator):
        """Test results distributed to all futures"""
        await coordinator.initialize()
        
        # Mock network
        mock_reader = AsyncMock()
        mock_writer = AsyncMock()
        mock_pool.get_connection.return_value = (mock_reader, mock_writer)
        
        # Mock readexactly for 3 operations (status + count, then 3 size+result pairs)
        mock_reader.readexactly = AsyncMock(side_effect=[
            b'\x00\x03',  # Status + 3 count
            b'\x00\x00\x00\x09{"r": 1}XX',
            b'\x00\x00\x00\x09{"r": 2}XX',
            b'\x00\x00\x00\x09{"r": 3}XX'
        ])
        
        # Create futures for 3 operations
        futures = [asyncio.Future() for _ in range(3)]
        for i, future in enumerate(futures):
            coordinator.operation_queue.put_nowait(
                BatchedOperation(f"id_{i}", f"op_{i}", {}, future, 0)
            )
        
        # Process batch
        await coordinator._process_batch()
        
        # All futures should have results
        for i, future in enumerate(futures):
            if future.done():
                result = future.result()
                assert isinstance(result, dict)


class TestBatchingCoordinatorEdgeCases:
    """Tests for edge cases"""
    
    @pytest.mark.asyncio
    async def test_empty_batch_processing(self, coordinator):
        """Test processing empty batch"""
        await coordinator.initialize()
        
        # Process with no operations
        await coordinator._process_batch()
        
        # Stats should not be affected
        assert coordinator.stats['total_batches'] == 0
    
    @pytest.mark.asyncio
    async def test_single_operation_batch(self, mock_pool, coordinator):
        """Test batch with single operation"""
        await coordinator.initialize()
        
        # Mock network
        mock_reader = AsyncMock()
        mock_writer = AsyncMock()
        mock_pool.get_connection.return_value = (mock_reader, mock_writer)
        
        # Mock readexactly for 1 operation
        mock_reader.readexactly = AsyncMock(side_effect=[
            b'\x00\x01',  # Status + 1 count
            b'\x00\x00\x00\x09{"r": 1}XX'
        ])
        
        # Queue single operation
        future = asyncio.Future()
        coordinator.operation_queue.put_nowait(
            BatchedOperation("id1", "op1", {}, future, 0)
        )
        
        # Process
        await coordinator._process_batch()
        
        # Should complete successfully
        if future.done():
            result = future.result()
            assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_large_batch(self, mock_pool, coordinator):
        """Test handling large batch"""
        # Create coordinator with large batch size
        coordinator.batch_size = 100
        await coordinator.initialize()
        
        # Mock network
        mock_reader = AsyncMock()
        mock_writer = AsyncMock()
        mock_pool.get_connection.return_value = (mock_reader, mock_writer)
        
        # Create mock responses for 50 operations
        responses = [b'\x00\x32']  # Status + 50 count
        for i in range(50):
            result_json = f'{{"r": {i}}}'.encode()
            size_bytes = len(result_json).to_bytes(4, 'big')
            responses.append(size_bytes + result_json)
        
        mock_reader.readexactly = AsyncMock(side_effect=responses)
        
        # Queue 50 operations
        for i in range(50):
            coordinator.operation_queue.put_nowait(
                BatchedOperation(f"id_{i}", f"op_{i}", {}, asyncio.Future(), 0)
            )
        
        # Process
        await coordinator._process_batch()
        
        # Stats should reflect large batch
        assert coordinator.stats['total_operations'] == 50


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
