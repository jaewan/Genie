#!/usr/bin/env python3

import asyncio
import torch
from genie.runtime.async_zero_copy_bridge import AsyncZeroCopyBridge


async def main():
    cfg = {'lib_path': './build/libgenie_data_plane.so'}
    bridge = AsyncZeroCopyBridge(cfg)
    ok = await bridge.initialize()
    assert ok

    try:
        tensors = [torch.randn(1024) for _ in range(200)]
        futures = []
        for t in tensors:
            req = await bridge.send_tensor(t, '127.0.0.1:12345', timeout=2.0)
            futures.append(req.future)
        results = await asyncio.gather(*futures, return_exceptions=True)
        assert all((not isinstance(r, Exception)) for r in results)
    finally:
        await bridge.shutdown()


if __name__ == "__main__":
    asyncio.run(main())


