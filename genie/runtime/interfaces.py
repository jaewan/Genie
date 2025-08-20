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

    @staticmethod
    def from_tensor(tensor: Any, destination: RemoteAccelerator) -> "TransferRequest":  # noqa: ANN401
        import torch

        if isinstance(tensor, torch.Tensor):
            num_bytes = tensor.element_size() * tensor.nelement()
            tensor_id = str(id(tensor))
        else:
            # Fallback for unknown data; treat as bytes-like
            buf = memoryview(tensor)
            num_bytes = buf.nbytes
            tensor_id = str(id(buf))
        return TransferRequest(tensor_id=tensor_id, num_bytes=num_bytes, destination=destination)


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


