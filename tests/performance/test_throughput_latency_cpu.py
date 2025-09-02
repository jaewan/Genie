#!/usr/bin/env python3

import time
import numpy as np
from genie.runtime.async_zero_copy_bridge import AsyncZeroCopyBridge
import asyncio
import torch


async def main():
    cfg = {'lib_path': './build/libgenie_data_plane.so'}
    bridge = AsyncZeroCopyBridge(cfg)
    ok = await bridge.initialize()
    assert ok

    try:
        sizes = [1024, 1024*1024, 10*1024*1024]
        results = {}
        for size in sizes:
            t = torch.randn(size // 4)  # ~size bytes, cpu tensor fallback
            start = time.perf_counter()
            for _ in range(20):
                req = await bridge.send_tensor(t, '127.0.0.1:12345', timeout=1.0)
                await req.future
            elapsed = time.perf_counter() - start
            throughput = (20 * size * 8) / (elapsed * 1e9)
            results[size] = throughput

        lats = []
        t = torch.randn(256)  # 1KB
        for _ in range(100):
            s = time.perf_counter_ns()
            req = await bridge.send_tensor(t, '127.0.0.1:12345', timeout=1.0)
            await req.future
            lats.append((time.perf_counter_ns() - s) / 1000)  # us
        p50 = float(np.percentile(lats, 50))
        p99 = float(np.percentile(lats, 99))

        print({'throughput_gbps': results, 'latency_us': {'p50': p50, 'p99': p99}})
    finally:
        await bridge.shutdown()


if __name__ == "__main__":
    asyncio.run(main())


