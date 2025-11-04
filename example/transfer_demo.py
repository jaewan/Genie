from __future__ import annotations

import asyncio
import os
import sys

import torch

# Ensure imports work when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genie.runtime.interfaces import RemoteAccelerator
from genie.runtime.transfer_manager import TransferManager
from genie.runtime.transports import PinnedTCPTransport, FallbackTransport


async def run_transfer(use_tcp: bool = True) -> None:
    transport = PinnedTCPTransport() if use_tcp else FallbackTransport()
    tm = TransferManager(transport)

    # Create a sample tensor (~4MB)
    t = torch.randn(1024, 1024)
    dest = RemoteAccelerator("127.0.0.1", 0)

    fut = await tm.transfer_tensor(t, dest)
    await tm.execute_batch()
    await fut.wait()
    print(f"transfer complete: tensor_id={fut.tensor_id}")


def main() -> None:
    use_tcp = os.environ.get("GENIE_USE_TCP", "1") != "0"
    asyncio.run(run_transfer(use_tcp=use_tcp))


if __name__ == "__main__":
    main()


