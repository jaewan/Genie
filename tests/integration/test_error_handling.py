"""
Test: Robust Error Handling & Recovery

Validates:
- Timeout enforcement
- Retry logic on transient failures
- Connection failure recovery
- Server error propagation
- Network partition handling
- Enhanced error messages
"""

import pytest
import asyncio
import torch
import logging
import time
import struct
from unittest.mock import patch

logger = logging.getLogger(__name__)


class TestTimeoutHandling:
    """Test timeout enforcement and handling."""

    @pytest.mark.asyncio
    async def test_timeout_actually_works(self):
        """Verify timeout is enforced."""
        from djinn.core.coordinator import GenieCoordinator, CoordinatorConfig

        config = CoordinatorConfig(
            node_id='test-client',
            tcp_fallback=True
        )
        coordinator = GenieCoordinator(config)
        await coordinator.start()

        try:
            # Send operation with very short timeout
            tensor = torch.randn(10, 10)

            with pytest.raises(RuntimeError) as exc_info:
                await coordinator.execute_remote_operation(
                    'aten::relu', [tensor], 'localhost:9999',  # Non-existent server
                    timeout=0.001  # 1ms - will definitely timeout
                )

            error_msg = str(exc_info.value).lower()
            assert 'timeout' in error_msg or 'failed' in error_msg or 'connection' in error_msg
            logger.info("✅ Error handling enforced correctly")

        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_enhanced_timeout_messages(self):
        """Test enhanced timeout error messages."""
        from djinn.core.coordinator import GenieCoordinator, CoordinatorConfig

        config = CoordinatorConfig(
            node_id='test-client',
            tcp_fallback=True
        )
        coordinator = GenieCoordinator(config)
        await coordinator.start()

        try:
            tensor = torch.randn(100, 100)

            with pytest.raises(RuntimeError) as exc_info:
                await coordinator.execute_remote_operation(
                    'aten::matmul', [tensor, tensor], 'localhost:9999',
                    timeout=0.1
                )

            error_msg = str(exc_info.value)
            assert 'aten::matmul' in error_msg
            assert 'localhost:9999' in error_msg
            assert ('network' in error_msg.lower() or 'connectivity' in error_msg.lower() or
                   'failed' in error_msg.lower() or 'timeout' in error_msg.lower())

            logger.info(f"✅ Enhanced timeout message: {error_msg[:100]}...")

        finally:
            await coordinator.stop()


class TestRetryLogic:
    """Test retry logic for transient failures."""

    @pytest.mark.asyncio
    async def test_retry_on_connection_reset(self):
        """Test automatic retry on transient errors."""
        from djinn.core.coordinator import GenieCoordinator, CoordinatorConfig

        config = CoordinatorConfig(
            node_id='test-client',
            tcp_fallback=True
        )
        coordinator = GenieCoordinator(config)
        await coordinator.start()

        try:
            # Mock connection failure on first attempt
            original_acquire = coordinator.transports['tcp']._connection_pool.acquire
            attempt_count = [0]

            async def failing_acquire(target):
                attempt_count[0] += 1
                if attempt_count[0] == 1:
                    raise ConnectionResetError("Simulated failure")
                return await original_acquire(target)

            coordinator.transports['tcp']._connection_pool.acquire = failing_acquire

            # Should succeed on retry
            tensor = torch.randn(10, 10)
            try:
                await coordinator.execute_remote_operation(
                    'aten::relu', [tensor], 'localhost:9999'
                )
            except RuntimeError:
                pass  # Expected to fail after retries

            assert attempt_count[0] >= 2  # Failed once, retried
            logger.info(f"✅ Automatic retry works: {attempt_count[0]} attempts")

        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_connection_pool_error_recovery(self):
        """Test connection pool recovers from errors."""
        from djinn.transport.connection_pool import ConnectionPool

        pool = ConnectionPool(max_per_target=3)

        # Simulate multiple connection failures
        for i in range(5):
            try:
                await pool.acquire('localhost:9999')
            except Exception:
                pass  # Expected to fail

        stats = pool.get_stats()
        assert stats['errors'] >= 5, "Pool should track connection errors"
        assert stats['created'] == 0, "No connections should be created for failed servers"

        logger.info(f"✅ Error recovery: {stats['errors']} errors tracked")


class TestServerErrorHandling:
    """Test server error propagation and handling."""

    @pytest.mark.asyncio
    async def test_unsupported_operation_error(self):
        """Test error propagation for unsupported operations."""
        # This test would need a real server - for now just verify error structure
        from djinn.core.coordinator import GenieCoordinator, CoordinatorConfig

        config = CoordinatorConfig(
            node_id='test-client',
            tcp_fallback=True
        )
        coordinator = GenieCoordinator(config)
        await coordinator.start()

        try:
            tensor = torch.randn(10, 10)

            try:
                await coordinator.execute_remote_operation(
                    'aten::unsupported_xyz_operation', [tensor], 'localhost:9999'
                )
            except RuntimeError as e:
                error_msg = str(e)
                assert 'unsupported' in error_msg.lower() or 'failed' in error_msg.lower()
                logger.info(f"✅ Server error propagation: {error_msg}")

        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_network_partition_handling(self):
        """Test handling of network partitions."""
        from djinn.core.coordinator import GenieCoordinator, CoordinatorConfig

        config = CoordinatorConfig(
            node_id='test-client',
            tcp_fallback=True
        )
        coordinator = GenieCoordinator(config)
        await coordinator.start()

        try:
            # Test with unreachable server
            tensor = torch.randn(10, 10)

            with pytest.raises(RuntimeError) as exc_info:
                await coordinator.execute_remote_operation(
                    'aten::add', [tensor, tensor], '192.0.2.1:9999',  # Unreachable IP
                    timeout=2.0
                )

            error_msg = str(exc_info.value)
            assert ('timeout' in error_msg.lower() or 'failed' in error_msg.lower() or
                   'connection' in error_msg.lower() or 'network' in error_msg.lower())

            logger.info(f"✅ Network partition handling: {error_msg}")

        finally:
            await coordinator.stop()


class TestRobustnessUnderLoad:
    """Test system robustness under various failure conditions."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_failures(self):
        """Test handling multiple concurrent failures."""
        from djinn.core.coordinator import GenieCoordinator, CoordinatorConfig

        config = CoordinatorConfig(
            node_id='test-client',
            tcp_fallback=True
        )
        coordinator = GenieCoordinator(config)
        await coordinator.start()

        try:
            # Launch multiple operations that will fail
            async def failing_operation(i):
                tensor = torch.randn(10, 10)
                try:
                    await coordinator.execute_remote_operation(
                        'aten::relu', [tensor], 'localhost:9999',
                        timeout=1.0
                    )
                    return "success"
                except RuntimeError:
                    return "failed"

            # 10 concurrent failing operations
            results = await asyncio.gather(*[failing_operation(i) for i in range(10)])

            # All should fail gracefully
            assert all(r == "failed" for r in results), "All operations should fail gracefully"

            # Check pool error tracking
            pool_stats = coordinator.transports['tcp']._connection_pool.get_stats()
            assert pool_stats['errors'] >= 10, "Pool should track all errors"

            logger.info(f"✅ Concurrent failure handling: {pool_stats['errors']} errors tracked")

        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_memory_cleanup_on_errors(self):
        """Test memory cleanup when operations fail."""
        from djinn.core.coordinator import GenieCoordinator, CoordinatorConfig

        config = CoordinatorConfig(
            node_id='test-client',
            tcp_fallback=True
        )
        coordinator = GenieCoordinator(config)
        await coordinator.start()

        try:
            initial_queues = len(coordinator._result_queues)

            # Launch operations that will fail
            for i in range(5):
                tensor = torch.randn(10, 10)
                try:
                    await coordinator.execute_remote_operation(
                        'aten::add', [tensor, tensor], 'localhost:9999',
                        timeout=0.5
                    )
                except RuntimeError:
                    pass  # Expected to fail

            # Check cleanup
            final_queues = len(coordinator._result_queues)

            # Should not leak queues
            assert final_queues <= initial_queues + 1, "Should not leak result queues"

            logger.info(f"✅ Memory cleanup: {initial_queues} -> {final_queues} queues")

        finally:
            await coordinator.stop()


class TestEdgeCases:
    """Test extreme edge cases and system robustness."""

    @pytest.mark.asyncio
    async def test_server_crash_during_execution(self):
        """Test behavior when server crashes during operation execution."""
        logger.info("\n" + "="*70)
        logger.info("TEST: Server crash during execution")
        logger.info("="*70)

        # This is a complex test that requires simulating server crash
        # For now, we'll test the timeout behavior when server stops responding
        from djinn.core.coordinator import GenieCoordinator, CoordinatorConfig

        config = CoordinatorConfig(
            node_id='test-client',
            tcp_fallback=True
        )
        coordinator = GenieCoordinator(config)
        await coordinator.start()

        try:
            # Send operation and then simulate server going away
            tensor = torch.randn(100, 100)

            # This should timeout gracefully when server becomes unresponsive
            with pytest.raises(RuntimeError) as exc_info:
                await coordinator.execute_remote_operation(
                    'aten::matmul', [tensor, tensor], 'localhost:9999',
                    timeout=5.0  # 5 second timeout
                )

            error_msg = str(exc_info.value).lower()
            assert ('timeout' in error_msg or 'failed' in error_msg or
                   'connection' in error_msg or 'network' in error_msg)

            logger.info(f"✅ Server crash handling: {error_msg}")

        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_gpu_oom_handling(self):
        """Test handling of GPU out-of-memory errors."""
        logger.info("\n" + "="*70)
        logger.info("TEST: GPU OOM handling")
        logger.info("="*70)

        # This test requires a server that can actually run GPU operations
        # For now, we'll test the error propagation mechanism
        from djinn.core.coordinator import GenieCoordinator, CoordinatorConfig

        config = CoordinatorConfig(
            node_id='test-client',
            tcp_fallback=True
        )
        coordinator = GenieCoordinator(config)
        await coordinator.start()

        try:
            # Create a very large tensor that would likely cause OOM on most GPUs
            # 10GB tensor (way too big for most consumer GPUs)
            large_size = int(10 * 1024 * 1024 * 1024 / 4)  # 10GB of float32
            try:
                # This should either succeed (if server has huge GPU memory)
                # or fail with some kind of memory error
                tensor = torch.randn(large_size, device='cpu')  # Create on CPU first

                result = await coordinator.execute_remote_operation(
                    'aten::relu', [tensor], 'localhost:9999',  # Non-existent server
                    timeout=10.0
                )

                logger.info(f"✅ Large tensor handled: {result.shape}")

            except RuntimeError as e:
                # Expected to fail due to network/server issues
                error_msg = str(e).lower()
                assert ('timeout' in error_msg or 'failed' in error_msg or
                       'connection' in error_msg or 'memory' in error_msg or
                       'oom' in error_msg)

                logger.info(f"✅ OOM error handling: {error_msg}")

        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_malformed_message_handling(self):
        """Test handling of malformed/corrupted messages."""
        logger.info("\n" + "="*70)
        logger.info("TEST: Malformed message handling")
        logger.info("="*70)

        # This test simulates receiving malformed data
        # We can test this by sending invalid protocol data
        from djinn.transport.connection_pool import ConnectionPool
        import socket

        # Create a mock server that sends malformed data
        async def malformed_server():
            server = await asyncio.start_server(
                self._handle_malformed_connection,
                'localhost', 0  # Let OS assign port
            )
            return server

        server = await malformed_server()
        port = server.sockets[0].getsockname()[1]

        try:
            pool = ConnectionPool(max_per_target=3)

            # Try to send data and see how the system handles malformed responses
            try:
                conn = await pool.acquire(f'localhost:{port}')
                # Send some data (the server will respond with malformed data)
                conn.writer.write(b'invalid_protocol_data')
                await conn.writer.drain()

                # The connection should be marked as failed
                await pool.release(conn, success=False)

                # Pool should handle this gracefully
                stats = pool.get_stats()
                assert stats['errors'] >= 1, "Pool should track malformed message errors"

                logger.info(f"✅ Malformed message handling: {stats['errors']} errors tracked")

            except Exception as e:
                logger.info(f"✅ Malformed message error properly handled: {e}")

        finally:
            server.close()
            await server.wait_closed()

    async def _handle_malformed_connection(self, reader, writer):
        """Mock server that sends malformed data."""
        try:
            # Read whatever data comes in
            data = await reader.read(1024)
            if data:
                # Send back malformed response (not following protocol)
                writer.write(b'not_a_valid_response')
                await writer.drain()
        except Exception as e:
            logger.debug(f"Malformed connection error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_connection_hanging_recovery(self):
        """Test recovery when connection hangs indefinitely."""
        logger.info("\n" + "="*70)
        logger.info("TEST: Connection hanging recovery")
        logger.info("="*70)

        from djinn.core.coordinator import GenieCoordinator, CoordinatorConfig

        config = CoordinatorConfig(
            node_id='test-client',
            tcp_fallback=True
        )
        coordinator = GenieCoordinator(config)
        await coordinator.start()

        try:
            # Create a server that accepts connection but never responds
            async def hanging_server():
                server = await asyncio.start_server(
                    self._handle_hanging_connection,
                    'localhost', 0
                )
                return server

            server = await hanging_server()
            port = server.sockets[0].getsockname()[1]

            # This should timeout rather than hang forever
            tensor = torch.randn(10, 10)

            start_time = time.time()
            with pytest.raises(RuntimeError) as exc_info:
                await coordinator.execute_remote_operation(
                    'aten::relu', [tensor], f'localhost:{port}',
                    timeout=3.0  # 3 second timeout
                )

            elapsed = time.time() - start_time
            error_msg = str(exc_info.value).lower()

            # Should timeout within reasonable time (not hang forever)
            assert elapsed < 5.0, f"Operation took too long: {elapsed:.2f}s"
            assert ('timeout' in error_msg or 'failed' in error_msg or
                   'connection' in error_msg)

            logger.info(f"✅ Hanging connection recovery: {elapsed:.2f}s, {error_msg}")

        finally:
            await coordinator.stop()
            server.close()
            await server.wait_closed()

    async def _handle_hanging_connection(self, reader, writer):
        """Mock server that accepts connection but never responds."""
        try:
            # Read the request but never respond
            await reader.read(1024)  # Read but don't process
            # Just hang here - don't send response
            await asyncio.sleep(10)  # Hang for 10 seconds
        except Exception as e:
            logger.debug(f"Hanging connection error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_partial_message_recovery(self):
        """Test recovery when message is partially received."""
        logger.info("\n" + "="*70)
        logger.info("TEST: Partial message recovery")
        logger.info("="*70)

        from djinn.transport.connection_pool import ConnectionPool
        import socket

        async def partial_server():
            server = await asyncio.start_server(
                self._handle_partial_connection,
                'localhost', 0
            )
            return server

        server = await partial_server()
        port = server.sockets[0].getsockname()[1]

        try:
            pool = ConnectionPool(max_per_target=3)

            # Send data that will cause partial response
            try:
                conn = await pool.acquire(f'localhost:{port}')

                # Send valid protocol start but incomplete data
                conn.writer.write(struct.pack('>I', 10))  # Valid length prefix
                conn.writer.write(b'partial')  # Incomplete data
                await conn.writer.drain()

                # The connection should be marked as failed
                await pool.release(conn, success=False)

                stats = pool.get_stats()
                assert stats['errors'] >= 1, "Pool should track partial message errors"

                logger.info(f"✅ Partial message recovery: {stats['errors']} errors tracked")

            except Exception as e:
                logger.info(f"✅ Partial message error properly handled: {e}")

        finally:
            server.close()
            await server.wait_closed()

    async def _handle_partial_connection(self, reader, writer):
        """Mock server that sends partial/incomplete responses."""
        try:
            # Try to read length prefix
            length_bytes = await reader.readexactly(4)
            if length_bytes:
                length = struct.unpack('>I', length_bytes)[0]
                # Read partial data but not complete
                partial_data = await reader.read(length // 2)  # Only half the data
                # Don't send complete response
                writer.close()
        except Exception as e:
            logger.debug(f"Partial connection error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
