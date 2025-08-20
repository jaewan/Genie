"""Genie Zero-Copy Runtime Interfaces and Implementations.

This package contains the transport abstractions, transfer manager, and
allocator shims to evolve from a developer-friendly pinned-memory TCP path
to a zero-copy RDMA path.
"""

from .interfaces import (
    ExecutionService,
    HealthStatus,
    MemoryView,
    PlanFragment,
    RemoteAccelerator,
    TensorHandle,
    TransferRequest,
    TransferResult,
)
from .transfer_manager import TransferManager, TransferFuture

__all__ = [
    "ExecutionService",
    "HealthStatus",
    "MemoryView",
    "PlanFragment",
    "RemoteAccelerator",
    "TensorHandle",
    "TransferRequest",
    "TransferResult",
    "TransferManager",
    "TransferFuture",
]


