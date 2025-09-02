import os
import pytest


@pytest.mark.skipif(os.environ.get("GENIE_WITH_KCP_TEST") != "1", reason="KCP smoke test disabled")
def test_kcp_mode_toggle(monkeypatch):
    # Only checks the Python binding toggling; does not require real DPDK
    from genie.runtime.transport_coordinator import DataPlaneBindings, DataPlaneConfig

    class FakeLib:
        def __init__(self):
            self.set_calls = []

        def genie_data_plane_create(self, cfg):
            return 1234

        def genie_set_reliability_mode(self, ptr, mode):
            self.set_calls.append((ptr, mode))
            return 0

    b = DataPlaneBindings()
    b.lib = FakeLib()
    b.data_plane_ptr = None
    monkeypatch.setenv("GENIE_RELIABILITY_MODE", "kcp")
    cfg = DataPlaneConfig(reliability_mode=None)
    assert b.create(cfg) is True
    assert b.lib.set_calls and b.lib.set_calls[-1][1] == 1

