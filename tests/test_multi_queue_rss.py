import os
import pytest


def test_multi_queue_rss_config(monkeypatch):
    # Simulate availability of C++ library but don't actually call it
    os.environ["GENIE_DISABLE_CPP_DATAPLANE"] = "1"

    from genie.runtime.transport_coordinator import DataPlaneConfig

    cfg = DataPlaneConfig()
    cfg.rx_queues = 4
    cfg.tx_queues = 4
    cfg.enable_rss = True
    cfg.rx_offload_checksum = True
    cfg.tx_offload_udp_cksum = True
    cfg.enable_cuda_graphs = True

    js = cfg.to_json()
    assert '"rx_queues": 4' in js
    assert '"tx_queues": 4' in js
    assert '"enable_rss": true' in js
    assert '"rx_offload_checksum": true' in js
    assert '"tx_offload_udp_cksum": true' in js
    assert '"enable_cuda_graphs": true' in js


