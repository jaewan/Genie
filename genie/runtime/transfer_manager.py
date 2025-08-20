from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .interfaces import RemoteAccelerator, TransferRequest
from .transports import Transport


@dataclass
class TransferFuture:
    tensor_id: str
    _event: asyncio.Event
    _result_ok: Optional[bool] = None
    _error: Optional[str] = None

    def done(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()

    def result(self) -> None:
        if not self.done():
            raise RuntimeError("Transfer not complete")
        if self._result_ok:
            return None
        raise RuntimeError(self._error or "transfer failed")


class TransferManager:
    def __init__(self, transport: Transport):
        self.transport = transport
        self.queue: "asyncio.Queue[TransferRequest]" = asyncio.Queue()
        self._active: Dict[str, TransferFuture] = {}
        self._batch_bytes = 0
        self._max_batch_bytes = 1 << 20  # Target >= 1MB
        self._flush_lock = asyncio.Lock()

    def should_flush_batch(self) -> bool:
        return self._batch_bytes >= self._max_batch_bytes

    async def transfer_tensor(self, tensor: torch.Tensor, target: RemoteAccelerator) -> TransferFuture:
        if not self.transport.is_dma_capable(tensor):
            tensor = self.transport.prepare_for_dma(tensor)
        request = TransferRequest.from_tensor(tensor, target)
        fut = TransferFuture(tensor_id=request.tensor_id, _event=asyncio.Event())
        self._active[request.tensor_id] = fut
        await self.queue.put(request)
        self._batch_bytes += request.num_bytes
        if self.should_flush_batch():
            await self.execute_batch()
        return fut

    async def execute_batch(self) -> None:
        async with self._flush_lock:
            requests = []
            while not self.queue.empty():
                try:
                    requests.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if not requests:
                return
            self._batch_bytes = 0
            results = await self.transport.send_batch(requests)
            for res in results:
                fut = self._active.get(res.tensor_id)
                if fut is None:
                    continue
                fut._result_ok = res.ok
                fut._error = res.error
                fut._event.set()


