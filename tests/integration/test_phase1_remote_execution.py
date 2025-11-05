"""
Phase 1 Integration Tests: End-to-End Remote Execution

These tests define success criteria for Phase 1 implementation.
They will initially FAIL (expected) and serve as targets for the
implementation work.

Timeline:
- Days 1-5: Make test_remote_add pass (Week 1 goal)
- Days 6-9: Pass all 7 tests (Week 2)
- Days 10-12: Performance baseline and documentation (Week 3)

To run all tests:
    pytest tests/integration/test_phase1_remote_execution.py -v -s

To run specific test:
    pytest tests/integration/test_phase1_remote_execution.py::TestPhase1RemoteExecution::test_remote_add -v -s

To run with debug logging:
    pytest tests/integration/test_phase1_remote_execution.py -v -s --log-cli-level=DEBUG
"""

import pytest
import asyncio
import torch
import logging
import time
from typing import Optional
import pytest_asyncio
import socket

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)

logger = logging.getLogger(__name__)

# Test configuration
def get_free_port():
    """Get a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

TEST_SERVER_PORT = get_free_port()  # Dynamic
TEST_DATA_PORT = get_free_port()    # Dynamic
TEST_TIMEOUT = 30.0

logger.info(f"Using ports: SERVER={TEST_SERVER_PORT}, DATA={TEST_DATA_PORT}")


# ============================================================================
# Fixtures
# ============================================================================

@pytest_asyncio.fixture
async def test_server():
    """Start test server on localhost."""
    try:
        from djinn.server.server import GenieServer, ServerConfig
    except ImportError:
        pytest.skip("GenieServer not available")
    
    config = ServerConfig(
        node_id='test-server',
        control_port=TEST_SERVER_PORT,
        data_port=TEST_DATA_PORT,
        gpu_indices=[0] if torch.cuda.is_available() else None,
        prefer_dpdk=False,
        tcp_fallback=True
    )
    
    server = GenieServer(config)
    success = await server.start()
    
    if not success:
        pytest.fail("Server failed to start")
    
    # Wait for server to be ready
    await asyncio.sleep(2)
    
    logger.info("✓ Test server started")
    
    yield server
    
    # Cleanup
    try:
        await server.stop()
        logger.info("✓ Test server stopped")
    except Exception as e:
        logger.error(f"Error stopping server: {e}")


@pytest_asyncio.fixture
async def test_coordinator():
    """Create test coordinator (client)."""
    try:
        from djinn.core.coordinator import GenieCoordinator, CoordinatorConfig
    except ImportError:
        pytest.skip("GenieCoordinator not available")
    
    config = CoordinatorConfig(
        node_id='test-client',
        control_port=16555,
        data_port=16556,
        prefer_dpdk=False,
        tcp_fallback=True
    )
    
    coordinator = GenieCoordinator(config)
    await coordinator.start()
    
    logger.info("✓ Test coordinator started")
    
    yield coordinator
    
    # Cleanup
    try:
        await coordinator.stop()
        logger.info("✓ Test coordinator stopped")
    except Exception as e:
        logger.error(f"Error stopping coordinator: {e}")


# ============================================================================
# Test Suite 1: Basic Operations
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.integration
class TestPhase1RemoteExecution:
    """Phase 1: Core integration tests."""
    
    async def test_remote_add(self, test_server, test_coordinator):
        """
        TEST 1: Remote addition (simplest operation).
        
        Success Criteria:
        - Operation executes on server GPU
        - Result returns to client
        - Result is correct
        - Latency < 1 second
        
        This is the primary Week 1 goal.
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 1: Remote Add (PRIMARY WEEK 1 GOAL)")
        logger.info("="*70)
        
        # Create inputs
        a = torch.randn(100, 100)
        b = torch.randn(100, 100)
        
        logger.info(f"Inputs: a={a.shape}, b={b.shape}")
        
        # Execute remotely
        start_time = time.time()
        
        result = await test_coordinator.execute_remote_operation(
            operation='aten::add',
            inputs=[a, b],
            target=f'localhost:{TEST_DATA_PORT}',
            timeout=TEST_TIMEOUT
        )
        
        elapsed = time.time() - start_time
        
        # Verify correctness
        expected = a + b
        assert result.shape == (100, 100), f"Wrong shape: {result.shape}"
        # Use more lenient tolerance for GPU vs CPU differences
        assert torch.allclose(result, expected, rtol=1e-2, atol=1e-4), "Result mismatch"
        assert result.device.type == 'cpu', "Result should be on CPU"
        
        # Verify performance
        assert elapsed < 1.0, f"Too slow: {elapsed:.3f}s"
        
        logger.info(f"✅ Remote add successful in {elapsed:.3f}s")
    
    async def test_remote_matmul(self, test_server, test_coordinator):
        """
        TEST 2: Remote matrix multiplication.
        
        Success Criteria:
        - Larger data transfer works
        - Compute-intensive operation works
        - Result is correct
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 2: Remote Matmul")
        logger.info("="*70)
        
        a = torch.randn(500, 300)
        b = torch.randn(300, 500)
        
        result = await test_coordinator.execute_remote_operation(
            operation='aten::matmul',
            inputs=[a, b],
            target=f'localhost:{TEST_DATA_PORT}',
            timeout=TEST_TIMEOUT
        )
        
        expected = torch.matmul(a, b)
        assert result.shape == (500, 500)
        # Use more lenient tolerance for GPU vs CPU differences
        assert torch.allclose(result, expected, rtol=1e-2, atol=1e-4)
        
        logger.info("✅ Remote matmul successful")
    
    async def test_remote_relu(self, test_server, test_coordinator):
        """
        TEST 3: Remote activation (single-input operation).
        
        Success Criteria:
        - Single-input operations work
        - Element-wise operations work
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 3: Remote ReLU")
        logger.info("="*70)
        
        x = torch.randn(200, 200)
        
        result = await test_coordinator.execute_remote_operation(
            operation='aten::relu',
            inputs=[x],
            target=f'localhost:{TEST_DATA_PORT}',
            timeout=TEST_TIMEOUT
        )
        
        expected = torch.relu(x)
        assert torch.allclose(result, expected)
        
        logger.info("✅ Remote relu successful")


# ============================================================================
# Test Suite 2: Error Handling
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.integration
class TestErrorHandling:
    """Error handling and edge cases."""
    
    async def test_timeout_handling(self, test_coordinator):
        """
        TEST 4: Timeout when server unreachable.
        
        Success Criteria:
        - Timeout error within 2 seconds (not 30)
        - Error message is clear
        - No hanging
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 4: Timeout Handling")
        logger.info("="*70)
        
        x = torch.randn(10, 10)
        
        with pytest.raises(RuntimeError, match="timeout|connection|failed"):
            await test_coordinator.execute_remote_operation(
                operation='aten::add',
                inputs=[x, x],
                target='localhost:19999',  # Wrong port
                timeout=2.0
            )
        
        logger.info("✅ Timeout handling works")
    
    async def test_server_error_propagation(self, test_server, test_coordinator):
        """
        TEST 5: Server errors propagate to client.
        
        Success Criteria:
        - Unsupported operation raises error
        - Error message includes server context
        - No client-side hang
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 5: Server Error Propagation")
        logger.info("="*70)
        
        x = torch.randn(10, 10)
        
        with pytest.raises(RuntimeError, match="not supported|failed|NotImplementedError"):
            await test_coordinator.execute_remote_operation(
                operation='aten::unsupported_xyz_operation',
                inputs=[x],
                target=f'localhost:{TEST_DATA_PORT}',
                timeout=TEST_TIMEOUT
            )
        
        logger.info("✅ Error propagation works")


# ============================================================================
# Test Suite 3: Concurrency
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.integration
class TestConcurrency:
    """Concurrent execution tests."""
    
    async def test_sequential_operations(self, test_server, test_coordinator):
        """
        TEST 6: Multiple operations in sequence.
        
        Success Criteria:
        - No state pollution between operations
        - Consistent results
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 6: Sequential Operations")
        logger.info("="*70)
        
        for i in range(5):
            a = torch.randn(50, 50)
            b = torch.randn(50, 50)
            
            result = await test_coordinator.execute_remote_operation(
                operation='aten::add',
                inputs=[a, b],
                target=f'localhost:{TEST_DATA_PORT}'
            )
            
            expected = a + b
            assert torch.allclose(result, expected)
            logger.info(f"  ✓ Operation {i+1}/5 passed")
        
        logger.info("✅ Sequential operations work")
    
    async def test_concurrent_operations(self, test_server, test_coordinator):
        """
        TEST 7: Truly concurrent operations.
        
        Success Criteria:
        - Multiple operations can be in flight
        - Result routing is correct
        - No race conditions
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 7: Concurrent Operations")
        logger.info("="*70)
        
        async def execute_op(size):
            """Execute single operation concurrently."""
            a = torch.randn(size, size)
            b = torch.randn(size, size)
            
            result = await test_coordinator.execute_remote_operation(
                operation='aten::matmul',
                inputs=[a, b],
                target=f'localhost:{TEST_DATA_PORT}'
            )
            
            expected = torch.matmul(a, b)
            # Use more lenient tolerance for GPU vs CPU differences
            assert torch.allclose(result, expected, rtol=1e-2, atol=1e-4)
            logger.info(f"  ✓ Concurrent op ({size}x{size}) completed")
            return result.shape
        
        # Execute 5 operations concurrently
        results = await asyncio.gather(
            execute_op(20),
            execute_op(30),
            execute_op(40),
            execute_op(50),
            execute_op(60)
        )
        
        expected_shapes = [(20, 20), (30, 30), (40, 40), (50, 50), (60, 60)]
        assert results == expected_shapes
        
        logger.info("✅ Concurrent operations work")


# ============================================================================
# Test Execution
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '--log-cli-level=DEBUG'])
