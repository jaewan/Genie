import asyncio

from genie.runtime.dpdk_backend import DPDKBackend


def test_register_with_control_plane(monkeypatch):
    backend = DPDKBackend({"control_plane": {"host": "127.0.0.1", "port": 0}})

    class FakeServer:
        def __init__(self):
            class Caps:
                def __init__(self):
                    self.node_id = "fake-node"
                    self.gpu_count = 1
                    self.max_transfer_size = 1024
            self.capabilities = Caps()

    class FakeCoordinator:
        def __init__(self):
            self.control_server = None

        async def initialize(self):
            return True

        async def _init_control_plane(self):
            self.control_server = FakeServer()

    # Monkeypatch initialize_transport to return fake coordinator
    from genie.runtime import transport_coordinator as tc

    async def fake_initialize_transport(*args, **kwargs):
        return FakeCoordinator()

    monkeypatch.setattr(tc, "initialize_transport", fake_initialize_transport)

    ok = backend.register_with_control_plane({"gpu_count": 4, "max_transfer_size": 4096})
    assert ok is True

