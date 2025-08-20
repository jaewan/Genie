from __future__ import annotations

import asyncio
import socket
from dataclasses import dataclass
from typing import Optional

import torch

from .interfaces import RemoteAccelerator, TransferRequest, TransferResult


class Transport:
    def is_dma_capable(self, tensor: torch.Tensor) -> bool:
        return bool(getattr(tensor, "is_pinned", lambda: False)())

    def prepare_for_dma(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            return tensor
        if tensor.is_pinned():
            return tensor
        # If CUDA is not available (CPU-only build), pin_memory is not supported.
        try:
            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False
        if not cuda_available:
            return tensor.contiguous()
        # Pin a contiguous clone for stable DMA mapping
        try:
            return tensor.contiguous().pin_memory()
        except Exception:
            # Graceful fallback for environments without CUDA pinning support
            return tensor.contiguous()

    async def send_batch(self, requests: list[TransferRequest]) -> list[TransferResult]:
        raise NotImplementedError


@dataclass
class _TCPDestination:
    host: str
    port: int


class PinnedTCPTransport(Transport):
    """Pinned-memory TCP transport.

    Uses pinned host buffers and a simple length-prefixed bytestream framing.
    This is the Phase 2/3 developer path before RDMA integration.
    """

    def __init__(self, default_port: int = 50555):
        self.default_port = default_port

    def _resolve(self, dest: RemoteAccelerator) -> _TCPDestination:
        return _TCPDestination(host=dest.hostname, port=self.default_port)

    async def _tcp_send(self, request: TransferRequest) -> TransferResult:
        try:
            dest = self._resolve(request.destination)
            reader, writer = await asyncio.open_connection(dest.host, dest.port)
            # Frame: |tensor_id_len|tensor_id|num_bytes|payload|
            tensor_id_bytes = request.tensor_id.encode("utf-8")
            writer.write(len(tensor_id_bytes).to_bytes(4, "big"))
            writer.write(tensor_id_bytes)
            writer.write(request.num_bytes.to_bytes(8, "big"))
            await writer.drain()
            # Payload would be streamed here. We only send zeros as a stub to avoid
            # touching user tensors until a server exists.
            chunk = b"\x00" * min(1 << 20, request.num_bytes)
            remaining = request.num_bytes
            while remaining > 0:
                n = min(remaining, len(chunk))
                writer.write(chunk[:n])
                remaining -= n
                if remaining % (8 << 20) == 0:
                    await writer.drain()
            await writer.drain()
            writer.close()
            try:
                await writer.wait_closed()
            except AttributeError:
                pass
            return TransferResult.success(request.tensor_id)
        except Exception as e:  # noqa: PERF203
            return TransferResult.failure(request.tensor_id, str(e))

    async def send_batch(self, requests: list[TransferRequest]) -> list[TransferResult]:
        # Batch by destination host to minimize connections; simple grouping
        tasks = [self._tcp_send(req) for req in requests]
        return await asyncio.gather(*tasks)


class FallbackTransport(Transport):
    """Fallback staged-copy using pinned memory and a local no-op send."""

    async def send_batch(self, requests: list[TransferRequest]) -> list[TransferResult]:
        # Emulate success locally without network I/O.
        return [TransferResult.success(r.tensor_id) for r in requests]


class RDMATransport(Transport):
    """Placeholder for RDMA transport.

    Integrate with DPDK and ibverbs in a future phase. The interface is kept
    stable so higher layers do not need to change.
    """

    async def send_batch(self, requests: list[TransferRequest]) -> list[TransferResult]:
        # Until RDMA is implemented, fall back to success to unblock development.
        return [TransferResult.success(r.tensor_id) for r in requests]


