import types

from genie.runtime.latency_profiler import LatencyProfiler


class _DummyTensor:
    def __init__(self, n=1024, elem=4):
        self._n = n
        self._e = elem
    def numel(self):
        return self._n
    def element_size(self):
        return self._e


def _make_plan():
    plan = types.SimpleNamespace()
    plan.fragments = ["f0"]
    plan.placement = {"f0": {"node": "local"}}
    plan.transfers = [{"fragment_id": "f0", "tensor": _DummyTensor(), "target": "local"}]
    return plan


def test_latency_profiler_smoke(monkeypatch):
    # Monkeypatch backend.execute_plan to avoid underlying transport
    from genie.runtime import dpdk_backend as dp
    original_execute = dp.DPDKBackend.execute_plan
    dp.DPDKBackend.execute_plan = lambda self, plan: {"ok": True}
    try:
        profiler = LatencyProfiler({"data_plane": {"enable_gpudev": False}})
        result = profiler.profile_execution_plan(_make_plan())
        timings = result["timings"]
        assert timings.total_ms >= 0.0
        assert timings.parsing_ms >= 0.0
        assert timings.transfer_ms >= 0.0
    finally:
        dp.DPDKBackend.execute_plan = original_execute


