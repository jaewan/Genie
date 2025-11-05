"""
Test: Network stack basic functionality

Validates:
- Server starts/stops cleanly
- Basic tensor transfer works
- Multiple sequential transfers work
- Concurrent transfers handled
"""

import pytest
import pytest_asyncio
import asyncio
import torch
from djinn.server.server import GenieServer, ServerConfig
from djinn.core.coordinator import GenieCoordinator, CoordinatorConfig


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def test_cluster():
    """Fixture: Start/stop server and coordinator."""

    # Start server
    server_config = ServerConfig(
        node_id='test_server',
        control_port=8001,
        data_port=8002,
        gpu_indices=[0] if torch.cuda.is_available() else []
    )
    server = GenieServer(server_config)
    await server.start()

    # Start coordinator
    client_config = CoordinatorConfig(
        node_id='test_client',
        control_port=8003,
        data_port=8004
    )
    coordinator = GenieCoordinator(client_config)
    await coordinator.start()

    # Wait for connection with timeout
    await asyncio.sleep(0.1)
    for _ in range(50):  # 5 second timeout
        # Check if connected (implementation-specific)
        # For now, just wait a bit
        await asyncio.sleep(0.1)
        break

    yield (server, coordinator)

    # Cleanup
    await coordinator.stop()
    await server.stop()


@pytest.mark.asyncio
class TestNetworkBasic:
    """Test basic network stack functionality."""

    @pytest.mark.asyncio
    async def test_server_lifecycle(self):
        """Test server starts and stops cleanly."""

        config = ServerConfig(
            node_id='lifecycle_test',
            control_port=9001,
            data_port=9002
        )

        server = GenieServer(config)

        # Start
        try:
            await server.start()
        except Exception as e:
            pytest.fail(f"Server failed to start: {e}")

        await asyncio.sleep(0.5)

        # Stop
        try:
            await server.stop()
        except Exception as e:
            pytest.fail(f"Server failed to stop: {e}")

        print("✅ Server lifecycle works")

    @pytest.mark.asyncio
    async def test_basic_transfer(self, test_cluster):
        """Test basic tensor transfer."""

        server, coordinator = test_cluster

        test_tensor = torch.randn(100, 100)

        try:
            transfer_id = await coordinator.send_tensor(
                tensor=test_tensor,
                target='test_server',
                semantic_metadata={'test': 'basic'}
            )
        except Exception as e:
            pytest.fail(f"Transfer failed: {e}")

        try:
            result = await coordinator.receive_tensor(
                transfer_id,
                {}
            )
        except Exception as e:
            pytest.fail(f"Receive failed: {e}")

        assert result is not None, "Received None!"
        assert result.shape == test_tensor.shape, \
            f"Shape mismatch: {result.shape} vs {test_tensor.shape}"

        print("✅ Basic transfer works")

    @pytest.mark.asyncio
    async def test_multiple_sequential_transfers(self, test_cluster):
        """Test multiple sequential transfers."""

        server, coordinator = test_cluster

        for i in range(5):
            test_tensor = torch.randn(50, 50)

            transfer_id = await coordinator.send_tensor(
                tensor=test_tensor,
                target='test_server',
                semantic_metadata={'transfer_num': i}
            )

            result = await coordinator.receive_tensor(
                transfer_id,
                {}
            )

            assert result.shape == test_tensor.shape

        print(f"✅ Multiple sequential transfers work")

    @pytest.mark.asyncio
    async def test_concurrent_transfers(self, test_cluster):
        """Test concurrent transfers."""

        server, coordinator = test_cluster

        async def transfer_task(i):
            test_tensor = torch.randn(50, 50)

            transfer_id = await coordinator.send_tensor(
                tensor=test_tensor,
                target='test_server',
                semantic_metadata={'concurrent_id': i}
            )

            result = await coordinator.receive_tensor(
                transfer_id,
                {}
            )

            return result.shape == test_tensor.shape

        # Run 3 concurrent transfers
        tasks = [transfer_task(i) for i in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent transfer {i} failed: {result}")
            assert result is True, f"Transfer {i} validation failed"

        print("✅ Concurrent transfers work")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
