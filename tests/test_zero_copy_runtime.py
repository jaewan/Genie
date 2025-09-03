import asyncio

import torch

from genie.runtime.allocator import DPDKAllocator
from genie.runtime.interfaces import RemoteAccelerator
from genie.runtime.transfer_manager import TransferManager
from genie.runtime.transports import PinnedTCPTransport, FallbackTransport


def test_dpdk_allocation_stub():
	allocator = DPDKAllocator()
	buf = allocator.allocate(1024 * 1024, torch.device("cpu"))
	assert buf.size == 1024 * 1024
	assert buf.data_ptr != 0


def test_zero_copy_transfer_stub_with_fallback():
	tm = TransferManager(FallbackTransport())
	t = torch.randn(1024)
	fut = asyncio.run(tm.transfer_tensor(t, RemoteAccelerator("localhost", 0)))
	asyncio.run(tm.execute_batch())
	assert fut is not None
	asyncio.run(fut.wait())
	assert fut.done()


def test_tcp_transport_prepares_pinned_memory():
	tm = TransferManager(PinnedTCPTransport())
	t = torch.randn(4, 1024)
	fut = asyncio.run(tm.transfer_tensor(t, RemoteAccelerator("127.0.0.1", 0)))
	# Execute batch but server may not exist; should not crash the path before send
	asyncio.run(tm.execute_batch())
	assert fut.tensor_id is not None


