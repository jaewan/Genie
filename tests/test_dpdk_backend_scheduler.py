from genie.runtime.dpdk_backend import DPDKBackend


def test_scheduler_noop_methods(monkeypatch):
    backend = DPDKBackend()

    class FakeCoordinator:
        def __init__(self):
            self.node_id = "node-1"

        async def initialize(self):
            return True

    # Monkeypatch coordinator init
    from genie.runtime import transport_coordinator as tc

    async def fake_initialize_transport(*args, **kwargs):
        return FakeCoordinator()

    monkeypatch.setattr(tc, "initialize_transport", fake_initialize_transport)

    # No scheduler set -> should be benign true
    assert backend.register_with_scheduler(capabilities={"gpu_count": 2}) is True
    assert backend.send_scheduler_heartbeat() is True

    # Set endpoint but aiohttp may not be installed; still should return True (no-op fallback)
    backend.set_scheduler_endpoint("http://localhost:9999")
    assert backend.register_with_scheduler(capabilities={"gpu_count": 2}) is True
    assert backend.send_scheduler_heartbeat() is True

