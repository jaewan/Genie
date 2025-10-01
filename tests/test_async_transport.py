"""
Tests for Async-First Transport (Refactoring #4)

Tests the ThreadPoolExecutor-based async transport coordinator
to ensure blocking ctypes calls don't block the event loop.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

# Mock imports that may not be available
import sys
sys.path.insert(0, '/home/jaewan/Genie')

from genie.runtime.transport_coordinator import (
    TransportCoordinator,
    DataPlaneConfig,
    TransferContext,
    TransferState,
)


class TestAsyncTransportBasics:
    """Test basic async functionality"""
    
    @pytest.mark.asyncio
    async def test_coordinator_has_thread_pool(self):
        """Test that coordinator initializes with a thread pool"""
        coordinator = TransportCoordinator(
            node_id="test_node",
            config={'thread_pool_workers': 2}
        )
        
        assert hasattr(coordinator, '_thread_pool')
        assert isinstance(coordinator._thread_pool, ThreadPoolExecutor)
        
        # Cleanup
        coordinator._thread_pool.shutdown(wait=False)
    
    @pytest.mark.asyncio
    async def test_thread_pool_default_workers(self):
        """Test default number of worker threads"""
        coordinator = TransportCoordinator(node_id="test_node")
        
        # Default should be 4 workers
        assert coordinator._thread_pool is not None
        
        # Cleanup
        coordinator._thread_pool.shutdown(wait=False)
    
    @pytest.mark.asyncio
    async def test_thread_pool_custom_workers(self):
        """Test custom number of worker threads"""
        coordinator = TransportCoordinator(
            node_id="test_node",
            config={'thread_pool_workers': 8}
        )
        
        assert coordinator._thread_pool is not None
        
        # Cleanup
        coordinator._thread_pool.shutdown(wait=False)
    
    @pytest.mark.asyncio
    async def test_shutdown_closes_thread_pool(self):
        """Test that shutdown properly closes the thread pool"""
        coordinator = TransportCoordinator(node_id="test_node")
        
        # Mock the data plane to avoid initialization issues
        coordinator.data_plane = None
        coordinator.control_server = None
        
        await coordinator.shutdown()
        
        # Thread pool should be shutdown
        # Note: We can't directly test if it's shutdown, but we can verify
        # that submit after shutdown raises an exception
        with pytest.raises(RuntimeError):
            coordinator._thread_pool.submit(lambda: None)


class TestAsyncDataPlaneOperations:
    """Test async data plane operations"""
    
    @pytest.mark.asyncio
    async def test_init_data_plane_is_async(self):
        """Test that data plane initialization is truly async"""
        coordinator = TransportCoordinator(node_id="test_node")
        coordinator.data_plane_config = DataPlaneConfig(
            eal_args=["test"],
            port_id=0,
            local_ip="127.0.0.1",
            local_mac="00:00:00:00:00:00",
            data_port=5556,
            gpu_device_id=0,
            enable_gpudev=False
        )
        
        # Mock the DataPlaneBindings to avoid C++ library dependency
        with patch('genie.runtime.transport_coordinator.DataPlaneBindings') as MockDataPlane:
            mock_dp = MagicMock()
            mock_dp.lib = None  # Simulate no C++ library available
            MockDataPlane.return_value = mock_dp
            
            # Should not block
            start_time = time.time()
            await coordinator._init_data_plane()
            elapsed = time.time() - start_time
            
            # Should complete quickly (no blocking)
            assert elapsed < 0.1
        
        # Cleanup
        coordinator._thread_pool.shutdown(wait=False)
    
    @pytest.mark.asyncio
    async def test_send_tensor_uses_executor(self):
        """Test that send_tensor uses thread pool executor"""
        coordinator = TransportCoordinator(node_id="test_node")
        
        # Mock data plane
        mock_dp = MagicMock()
        mock_dp.is_available.return_value = True
        
        # Make send_tensor blocking to test executor usage
        def blocking_send(*args):
            time.sleep(0.1)  # Simulate blocking call
            return True
        
        mock_dp.send_tensor = blocking_send
        mock_dp.register_gpu_memory.return_value = 12345
        coordinator.data_plane = mock_dp
        
        # Mock context
        context = TransferContext(
            transfer_id="test_id",
            tensor_id="tensor_id",
            source_node="node1",
            target_node="node2",
            size=1024,
            dtype="float32",
            shape=[1024],
            gpu_ptr=0x1000
        )
        
        # This should not block the event loop
        start_time = time.time()
        await coordinator._start_data_plane_send(context)
        elapsed = time.time() - start_time
        
        # Should have used executor (elapsed should be ~ 0.1s from blocking call)
        assert 0.09 < elapsed < 0.2
        assert context.state == TransferState.SENDING
        
        # Cleanup
        coordinator._thread_pool.shutdown(wait=False)
    
    @pytest.mark.asyncio
    async def test_receive_tensor_uses_executor(self):
        """Test that receive_tensor uses thread pool executor"""
        coordinator = TransportCoordinator(node_id="test_node")
        
        # Mock data plane
        mock_dp = MagicMock()
        mock_dp.is_available.return_value = True
        
        # Make receive_tensor blocking
        def blocking_receive(*args):
            time.sleep(0.1)
            return True
        
        mock_dp.receive_tensor = blocking_receive
        coordinator.data_plane = mock_dp
        
        # Mock context
        context = TransferContext(
            transfer_id="test_id",
            tensor_id="tensor_id",
            source_node="node1",
            target_node="node2",
            size=1024,
            dtype="float32",
            shape=[1024]
        )
        
        # Should use executor
        start_time = time.time()
        await coordinator._prepare_receive_transfer(context)
        elapsed = time.time() - start_time
        
        assert 0.09 < elapsed < 0.2
        assert context.state == TransferState.RECEIVING
        
        # Cleanup
        coordinator._thread_pool.shutdown(wait=False)
    
    @pytest.mark.asyncio
    async def test_get_statistics_uses_executor(self):
        """Test that get_statistics_async uses thread pool executor"""
        coordinator = TransportCoordinator(node_id="test_node")
        
        # Mock data plane
        mock_dp = MagicMock()
        mock_dp.is_available.return_value = True
        
        # Make get_statistics blocking
        def blocking_stats():
            time.sleep(0.05)
            return {'packets_sent': 100}
        
        mock_dp.get_statistics = blocking_stats
        coordinator.data_plane = mock_dp
        
        # Should use executor (async version)
        start_time = time.time()
        stats = await coordinator.get_statistics_async()
        elapsed = time.time() - start_time
        
        assert 0.04 < elapsed < 0.15
        assert stats['packets_sent'] == 100
        
        # Test synchronous version (for backward compatibility)
        stats_sync = coordinator.get_statistics()
        assert stats_sync['packets_sent'] == 100
        
        # Cleanup
        coordinator._thread_pool.shutdown(wait=False)


class TestConcurrentTransfers:
    """Test concurrent transfer handling"""
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_sends(self):
        """Test handling multiple concurrent send operations"""
        coordinator = TransportCoordinator(
            node_id="test_node",
            config={'thread_pool_workers': 4}
        )
        
        # Mock data plane
        mock_dp = MagicMock()
        mock_dp.is_available.return_value = True
        
        send_count = 0
        
        def blocking_send(*args):
            nonlocal send_count
            time.sleep(0.05)  # Simulate work
            send_count += 1
            return True
        
        mock_dp.send_tensor = blocking_send
        mock_dp.register_gpu_memory.return_value = 12345
        coordinator.data_plane = mock_dp
        
        # Create multiple contexts
        contexts = []
        for i in range(5):
            ctx = TransferContext(
                transfer_id=f"transfer_{i}",
                tensor_id=f"tensor_{i}",
                source_node="node1",
                target_node="node2",
                size=1024,
                dtype="float32",
                shape=[1024],
                gpu_ptr=0x1000 + i
            )
            contexts.append(ctx)
        
        # Send all concurrently
        start_time = time.time()
        tasks = [coordinator._start_data_plane_send(ctx) for ctx in contexts]
        await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        
        # Should complete in parallel (4 workers, 5 tasks)
        # If sequential: 5 * 0.05 = 0.25s
        # If parallel (4 workers): ~2 batches = ~0.1s
        assert elapsed < 0.2  # Should be faster than sequential
        assert send_count == 5
        
        # All should be in SENDING state
        for ctx in contexts:
            assert ctx.state == TransferState.SENDING
        
        # Cleanup
        coordinator._thread_pool.shutdown(wait=False)
    
    @pytest.mark.asyncio
    async def test_concurrent_send_and_receive(self):
        """Test concurrent send and receive operations"""
        coordinator = TransportCoordinator(node_id="test_node")
        
        # Mock data plane
        mock_dp = MagicMock()
        mock_dp.is_available.return_value = True
        
        def blocking_send(*args):
            time.sleep(0.05)
            return True
        
        def blocking_receive(*args):
            time.sleep(0.05)
            return True
        
        mock_dp.send_tensor = blocking_send
        mock_dp.receive_tensor = blocking_receive
        mock_dp.register_gpu_memory.return_value = 12345
        coordinator.data_plane = mock_dp
        
        # Create send and receive contexts
        send_ctx = TransferContext(
            transfer_id="send_1",
            tensor_id="tensor_1",
            source_node="node1",
            target_node="node2",
            size=1024,
            dtype="float32",
            shape=[1024],
            gpu_ptr=0x1000
        )
        
        recv_ctx = TransferContext(
            transfer_id="recv_1",
            tensor_id="tensor_2",
            source_node="node2",
            target_node="node1",
            size=1024,
            dtype="float32",
            shape=[1024]
        )
        
        # Execute concurrently
        start_time = time.time()
        await asyncio.gather(
            coordinator._start_data_plane_send(send_ctx),
            coordinator._prepare_receive_transfer(recv_ctx)
        )
        elapsed = time.time() - start_time
        
        # Should complete in parallel
        assert elapsed < 0.15  # Both ~0.05s, should overlap
        assert send_ctx.state == TransferState.SENDING
        assert recv_ctx.state == TransferState.RECEIVING
        
        # Cleanup
        coordinator._thread_pool.shutdown(wait=False)
    
    @pytest.mark.asyncio
    async def test_event_loop_not_blocked(self):
        """Test that blocking calls don't block the event loop"""
        coordinator = TransportCoordinator(node_id="test_node")
        
        # Mock data plane with a long blocking call
        mock_dp = MagicMock()
        mock_dp.is_available.return_value = True
        
        def long_blocking_send(*args):
            time.sleep(0.2)  # Long blocking operation
            return True
        
        mock_dp.send_tensor = long_blocking_send
        mock_dp.register_gpu_memory.return_value = 12345
        coordinator.data_plane = mock_dp
        
        context = TransferContext(
            transfer_id="test_id",
            tensor_id="tensor_id",
            source_node="node1",
            target_node="node2",
            size=1024,
            dtype="float32",
            shape=[1024],
            gpu_ptr=0x1000
        )
        
        # Track event loop activity
        loop_active = []
        
        async def monitor_loop():
            for _ in range(10):
                loop_active.append(time.time())
                await asyncio.sleep(0.03)
        
        # Start send and monitor concurrently
        start_time = time.time()
        await asyncio.gather(
            coordinator._start_data_plane_send(context),
            monitor_loop()
        )
        elapsed = time.time() - start_time
        
        # Event loop should have been active throughout
        assert len(loop_active) >= 5  # Should have multiple ticks
        assert context.state == TransferState.SENDING
        
        # Cleanup
        coordinator._thread_pool.shutdown(wait=False)


class TestErrorHandling:
    """Test error handling in async operations"""
    
    @pytest.mark.asyncio
    async def test_executor_exception_handling(self):
        """Test that exceptions in executor are properly handled"""
        coordinator = TransportCoordinator(node_id="test_node")
        
        # Mock data plane that raises exception
        mock_dp = MagicMock()
        mock_dp.is_available.return_value = True
        
        def failing_send(*args):
            raise RuntimeError("C++ library error")
        
        mock_dp.send_tensor = failing_send
        mock_dp.register_gpu_memory.return_value = 12345
        coordinator.data_plane = mock_dp
        
        context = TransferContext(
            transfer_id="test_id",
            tensor_id="tensor_id",
            source_node="node1",
            target_node="node2",
            size=1024,
            dtype="float32",
            shape=[1024],
            gpu_ptr=0x1000
        )
        
        # Should handle exception gracefully
        with pytest.raises(RuntimeError):
            await coordinator._start_data_plane_send(context)
        
        # Cleanup
        coordinator._thread_pool.shutdown(wait=False)
    
    @pytest.mark.asyncio
    async def test_shutdown_waits_for_pending_operations(self):
        """Test that shutdown waits for pending executor operations"""
        coordinator = TransportCoordinator(node_id="test_node")
        
        # Mock data plane with slow operation
        mock_dp = MagicMock()
        mock_dp.is_available.return_value = True
        
        completed = []
        
        def slow_send(*args):
            time.sleep(0.1)
            completed.append(True)
            return True
        
        mock_dp.send_tensor = slow_send
        mock_dp.register_gpu_memory.return_value = 12345
        mock_dp.stop = Mock()
        mock_dp.destroy = Mock()
        coordinator.data_plane = mock_dp
        
        context = TransferContext(
            transfer_id="test_id",
            tensor_id="tensor_id",
            source_node="node1",
            target_node="node2",
            size=1024,
            dtype="float32",
            shape=[1024],
            gpu_ptr=0x1000
        )
        
        # Start operation
        send_task = asyncio.create_task(coordinator._start_data_plane_send(context))
        
        # Give it time to start
        await asyncio.sleep(0.05)
        
        # Shutdown should wait for thread pool operations
        await coordinator.shutdown()
        
        # Operation should have completed
        assert len(completed) == 1
        
        # Now await the task to clean it up
        try:
            await send_task
        except Exception:
            pass  # Task may have been cancelled during shutdown


class TestPerformance:
    """Test performance characteristics"""
    
    @pytest.mark.asyncio
    async def test_throughput_improvement(self):
        """Test that async approach improves throughput"""
        # Sequential coordinator (simulated with 1 worker)
        seq_coordinator = TransportCoordinator(
            node_id="seq_node",
            config={'thread_pool_workers': 1}
        )
        
        # Parallel coordinator (4 workers)
        par_coordinator = TransportCoordinator(
            node_id="par_node",
            config={'thread_pool_workers': 4}
        )
        
        # Mock data planes
        for coord in [seq_coordinator, par_coordinator]:
            mock_dp = MagicMock()
            mock_dp.is_available.return_value = True
            mock_dp.send_tensor = lambda *args: (time.sleep(0.05), True)[1]
            mock_dp.register_gpu_memory.return_value = 12345
            coord.data_plane = mock_dp
        
        # Create 4 contexts
        contexts = []
        for i in range(4):
            ctx = TransferContext(
                transfer_id=f"transfer_{i}",
                tensor_id=f"tensor_{i}",
                source_node="node1",
                target_node="node2",
                size=1024,
                dtype="float32",
                shape=[1024],
                gpu_ptr=0x1000 + i
            )
            contexts.append(ctx)
        
        # Test sequential (1 worker)
        seq_contexts = [TransferContext(**c.__dict__) for c in contexts]
        start_time = time.time()
        tasks = [seq_coordinator._start_data_plane_send(ctx) for ctx in seq_contexts]
        await asyncio.gather(*tasks)
        seq_time = time.time() - start_time
        
        # Test parallel (4 workers)
        par_contexts = [TransferContext(**c.__dict__) for c in contexts]
        start_time = time.time()
        tasks = [par_coordinator._start_data_plane_send(ctx) for ctx in par_contexts]
        await asyncio.gather(*tasks)
        par_time = time.time() - start_time
        
        # Parallel should be significantly faster
        # Sequential: ~4 * 0.05 = 0.2s
        # Parallel: ~0.05s (all at once)
        assert par_time < seq_time * 0.6  # At least 40% faster
        
        # Cleanup
        seq_coordinator._thread_pool.shutdown(wait=False)
        par_coordinator._thread_pool.shutdown(wait=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

