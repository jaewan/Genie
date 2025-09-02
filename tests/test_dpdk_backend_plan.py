import asyncio
import types

import pytest

from genie.semantic.workload import ExecutionPlan, PlanFragment
from genie.runtime.dpdk_backend import DPDKBackend


class DummyTensor:
    def __init__(self, numel: int, elem_size: int):
        self._numel = numel
        self._elem_size = elem_size

    def numel(self):
        return self._numel

    def element_size(self):
        return self._elem_size

    def data_ptr(self):  # Not used in this test path
        return 0

    @property
    def dtype(self):
        return "float32"

    @property
    def shape(self):
        return [self._numel]


def create_plan_with_transfer():
    frag = PlanFragment(fragment_id="frag_1", subgraph=None, inputs=[], outputs=[])
    plan = ExecutionPlan(
        plan_id="plan_1",
        fragments=[frag],
        placement={"frag_1": {"node": "local", "device": "cpu:0"}},
        transfers=[{"fragment_id": "frag_1", "tensor": DummyTensor(16, 4), "target": "remote-node"}],
        feature_flags={}
    )
    return plan


def test_execute_plan_smoke(monkeypatch):
    backend = DPDKBackend({"data_plane": {"enable_gpudev": False}})

    # Monkeypatch transport coordinator to avoid real DPDK/async server
    from genie.runtime import transport_coordinator as tc

    class FakeCoordinator:
        async def initialize(self):
            return True

        async def send_tensor(self, tensor, target):
            # Return a deterministic transfer id
            return "tx_1234"

    async def fake_initialize_transport(*args, **kwargs):
        return FakeCoordinator()

    monkeypatch.setattr(tc, "initialize_transport", fake_initialize_transport)

    plan = create_plan_with_transfer()
    result = backend.execute_plan(plan)

    assert isinstance(result, dict)
    assert "tx_1234" in result
    assert result["tx_1234"]["started"] is True



