from genie.runtime.metrics import MetricsExporter


def test_metrics_exporter_noop():
    m = MetricsExporter(port=9200)
    m.start()
    m.record_packet("tx", 128)
    m.set_active_transfers(3)
    m.observe_transfer_latency(0.002)
    snap = m.dump_snapshot()
    assert "enabled" in snap and "server_started" in snap

