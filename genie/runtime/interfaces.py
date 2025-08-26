from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple


@dataclass(frozen=True)
class RemoteAccelerator:
    hostname: str
    device_index: int


@dataclass(frozen=True)
class TensorHandle:
    tensor_id: str
    shape: Tuple[int, ...]
    dtype: str
    device: str


@dataclass(frozen=True)
class MemoryView:
    tensor_id: str
    size_bytes: int
    # When using zero-copy, this may represent a DMA handle instead of raw bytes
    data: Optional[memoryview] = None
    dma_handle: Optional[Dict[str, Any]] = None


class HealthStatus(enum.Enum):
    OK = "ok"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True)
class PlanFragment:
    plan_id: str
    nodes: List[Dict[str, Any]]
    inputs: List[str]
    outputs: List[str]


@dataclass(frozen=True)
class TransferRequest:
    tensor_id: str
    num_bytes: int
    destination: RemoteAccelerator
    # Optional payload for data-plane transfers; may carry a memoryview over
    # CPU (pinned) storage to enable zero-copy framing on TCP/RDMA paths.
    payload: Optional[memoryview] = None
    # Keep-alive owner to ensure the underlying storage isn't GC'd while in-flight
    payload_owner: Optional[object] = None

    @staticmethod
    def from_tensor(tensor: Any, destination: RemoteAccelerator) -> "TransferRequest":  # noqa: ANN401
        import torch

        if isinstance(tensor, torch.Tensor):
            # Stage to CPU pinned memory when coming from CUDA
            try:
                device_type = tensor.device.type  # type: ignore[attr-defined]
            except Exception:
                device_type = "cpu"

            owner: Optional[object] = None
            cpu_tensor = tensor
            if device_type == "cuda":
                # Create a pinned CPU buffer and copy synchronously
                try:
                    pinned = torch.empty_like(tensor, device=torch.device("cpu"), pin_memory=True)
                except Exception:
                    pinned = torch.empty_like(tensor, device=torch.device("cpu"))
                try:
                    pinned.copy_(tensor, non_blocking=False)
                except Exception:
                    pinned.copy_(tensor)
                cpu_tensor = pinned
                owner = pinned
            else:
                # Ensure contiguous CPU storage; attempt to pin for faster I/O when possible
                try:
                    cpu_tensor = tensor.contiguous()
                    if hasattr(cpu_tensor, "pin_memory"):
                        cpu_tensor = cpu_tensor.pin_memory()
                except Exception:
                    cpu_tensor = tensor.contiguous()

            # Build a memoryview over the underlying storage via numpy bridge
            try:
                np_arr = cpu_tensor.detach().cpu().contiguous().numpy()
                payload_view = memoryview(np_arr)
                num_bytes = np_arr.nbytes
                tensor_id = str(id(cpu_tensor))
                return TransferRequest(
                    tensor_id=tensor_id,
                    num_bytes=num_bytes,
                    destination=destination,
                    payload=payload_view,
                    payload_owner=owner or cpu_tensor,
                )
            except Exception:
                # Fallback: no payload; header-only
                num_bytes = cpu_tensor.element_size() * cpu_tensor.nelement()
                tensor_id = str(id(cpu_tensor))
                return TransferRequest(tensor_id=tensor_id, num_bytes=num_bytes, destination=destination)
        else:
            # Fallback for unknown data; treat as bytes-like
            buf = memoryview(tensor)
            num_bytes = buf.nbytes
            tensor_id = str(id(buf))
            return TransferRequest(tensor_id=tensor_id, num_bytes=num_bytes, destination=destination, payload=buf, payload_owner=tensor)


@dataclass
class TransferResult:
    ok: bool
    tensor_id: str
    error: Optional[str] = None

    @staticmethod
    def success(tensor_id: str) -> "TransferResult":
        return TransferResult(ok=True, tensor_id=tensor_id)

    @staticmethod
    def failure(tensor_id: str, error: str) -> "TransferResult":
        return TransferResult(ok=False, tensor_id=tensor_id, error=error)


class ExecutionService(Protocol):
    def run_subgraph(self, plan_fragment: PlanFragment) -> List[TensorHandle]:
        ...

    def fetch_result(self, tensor_id: str) -> MemoryView:
        ...

    def cancel(self, plan_id: str) -> None:
        ...

    def health(self) -> HealthStatus:
        ...


