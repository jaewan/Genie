#!/usr/bin/env python3

import asyncio

from genie.runtime.async_zero_copy_bridge import AsyncZeroCopyBridge
import torch


async def main():
    cfg = {
        'lib_path': './build/libgenie_data_plane.so',
        'simulate_packet_loss': 1.0,  # Force loss to guarantee retransmission
        'max_retransmissions': 1,
    }
    bridge = AsyncZeroCopyBridge(cfg)
    ok = await bridge.initialize()
    assert ok

    try:
        t = torch.randn(1024, 1024)  # CPU tensor for fallback
        req = await bridge.send_tensor(t, '127.0.0.1:12345', timeout=1.0)
        try:
            await req.future
        except Exception:
            # Even with retries, allow failure but require that retransmissions happened
            pass

        stats = bridge.get_stats()
        # Verify retransmissions occurred under loss
        assert stats.get('retransmissions', 0) >= 1
    finally:
        await bridge.shutdown()


if __name__ == "__main__":
    asyncio.run(main())


