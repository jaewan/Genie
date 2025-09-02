import types

from genie.runtime.transport_coordinator import DataPlaneBindings, DataPlaneConfig


class FakeLib:
    def __init__(self):
        self._calls = []

    def genie_data_plane_create(self, cfg):
        return 1234

    def genie_set_reliability_mode(self, ptr, mode):
        self._calls.append((ptr, mode))
        return 0

    def genie_get_reliability_mode(self, ptr):
        if not self._calls:
            return 0
        return self._calls[-1][1]


def test_env_flag_sets_kcp_mode(monkeypatch):
    b = DataPlaneBindings()
    # Inject fake lib and data_plane_ptr readiness
    b.lib = FakeLib()
    # Monkeypatch loader to skip signature setup for this test
    b.data_plane_ptr = None

    # Force env
    monkeypatch.setenv("GENIE_RELIABILITY_MODE", "kcp")

    cfg = DataPlaneConfig()
    assert b.create(cfg) is True
    # Verify mode call recorded
    assert isinstance(b.lib._calls, list)
    assert b.lib._calls and b.lib._calls[0][1] == 1


def test_config_flag_sets_custom(monkeypatch):
    b = DataPlaneBindings()
    b.lib = FakeLib()
    b.data_plane_ptr = None
    # Ensure env not set to override
    monkeypatch.delenv("GENIE_RELIABILITY_MODE", raising=False)
    cfg = DataPlaneConfig(reliability_mode="custom")
    assert b.create(cfg) is True
    assert b.lib._calls and b.lib._calls[-1][1] == 0


def test_query_reliability_mode(monkeypatch):
    b = DataPlaneBindings()
    b.lib = FakeLib()
    b.data_plane_ptr = None
    cfg = DataPlaneConfig(reliability_mode="kcp")
    assert b.create(cfg) is True
    mode = b.get_reliability_mode()
    assert mode == "kcp"

