from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .dpdk_backend import DPDKBackend


@dataclass
class TimingBreakdown:
    parsing_ms: float
    registration_ms: float
    transfer_ms: float
    completion_ms: float
    total_ms: float


@contextmanager
def timer() -> Any:
    start = time.perf_counter()
    result = type("_Timer", (), {})()
    try:
        yield result
    finally:
        end = time.perf_counter()
        result.elapsed_ms = (end - start) * 1000.0


class LatencyProfiler:
    """Profile end-to-end latency of executing an ExecutionPlan via DPDKBackend.

    This intentionally avoids hard dependencies on GPU/DPDK by delegating to the
    existing Python wrapper. In environments without the full data plane, callers
    can monkeypatch the backend's transport methods to produce deterministic timing.
    """

    def __init__(self, backend_config: Optional[Dict[str, Any]] = None) -> None:
        self._backend_config = backend_config or {"data_plane": {"enable_gpudev": False}}

    def profile_execution_plan(self, plan: Any) -> Dict[str, Any]:
        backend = DPDKBackend(self._backend_config)

        parsing_ms = 0.0
        registration_ms = 0.0
        transfer_ms = 0.0
        completion_ms = 0.0

        with timer() as t_total:
            # Parsing stage: lightweight walk of the plan object
            with timer() as t_parse:
                _ = getattr(plan, "fragments", [])
                _ = getattr(plan, "placement", {})
                _ = getattr(plan, "transfers", [])
            parsing_ms = t_parse.elapsed_ms

            # Memory registration stage (placeholder at Python layer)
            with timer() as t_reg:
                # In real runs this would pin/register memory; in Python wrapper we noop
                pass
            registration_ms = t_reg.elapsed_ms

            # Transfer execution stage
            with timer() as t_xfer:
                _ = backend.execute_plan(plan)
            transfer_ms = t_xfer.elapsed_ms

            # Completion wait stage (no explicit wait path in wrapper yet)
            with timer() as t_comp:
                # If a completion API is added, call it here. For now this is a noop.
                pass
            completion_ms = t_comp.elapsed_ms

        total_ms = t_total.elapsed_ms

        return {
            "timings": TimingBreakdown(
                parsing_ms=parsing_ms,
                registration_ms=registration_ms,
                transfer_ms=transfer_ms,
                completion_ms=completion_ms,
                total_ms=total_ms,
            ),
            "backend": backend,
        }


