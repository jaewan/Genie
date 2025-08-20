# Component: Zero-Copy Runtime

## Purpose
Implements efficient, zero-copy data movement between client and remote accelerators using DPDK and GPU-direct technologies.

## Context
- **Upstream**: Optimization/Execution Engine (transfer + execution requests)
- **Downstream**: Remote GPU nodes
- **Interactions**: Memory pools, network stack, GPU drivers

## Key Requirements
- Network bandwidth utilization >90% (Phase 4)
- Zero-copy from application to GPU when gpudev available
- <50µs DMA setup overhead
- Graceful fallback when GPU-direct unavailable

## Phased Transport & Execution
- Phase 2 (dev): TCP + pinned memory; execution service collocated or remote
- Phase 3: Remote runtime over TCP; same protocol; placement + scheduling
- Phase 4: DPDK mempools + gpudev; GPUDirect RDMA; overlap comm/compute

## Execution Service Interface (Control Plane)
```python
class ExecutionService(ABC):
    @abstractmethod
    def run_subgraph(self, plan_fragment: PlanFragment) -> List[TensorHandle]:
        """Execute a PlanFragment and return result handles."""

    @abstractmethod
    def fetch_result(self, tensor_id: str) -> MemoryView:
        """Return a read-only view (or DMA handle) for result tensor."""

    @abstractmethod
    def cancel(self, plan_id: str) -> None:
        pass

    @abstractmethod
    def health(self) -> HealthStatus:
        pass
```

## Data Plane (Transport)
- TCP (pinned memory): structured bytestreams for tensors
- RDMA (DPDK/ibverbs): DMAHandles with {iova,lkey,rkey,pool_id}
- Batching: group transfers by node/device; target ≥1MB average
- Overlap: double-buffering and async completion callbacks

## 1. DPDK Memory Allocator
```python
class DPDKAllocator(torch.Allocator):
    def __init__(self):
        self.eal_initialized = False
        self.memory_pools = {}
        self.huge_page_size = "2MB"
        self.initialize_dpdk()
    
    def initialize_dpdk(self):
        # Initialize DPDK EAL (Environment Abstraction Layer)
        # ... (omitted for brevity)
        self.eal_initialized = True
        self.create_memory_pools()
    
    def create_memory_pools(self):
        # Create pools for different size classes (small/medium/large/huge)
        # ... (omitted)
    
    def allocate(self, size: int, device: torch.device) -> DMABuffer:
        pool = self.select_pool(size)
        mbuf = dpdk.rte_pktmbuf_alloc(pool)
        if not mbuf:
            raise MemoryError(f"Failed to allocate {size} bytes from DPDK")
        dma_handle = self.register_for_dma(mbuf, size)
        return DMABuffer(
            data=mbuf.buf_addr,
            size=size,
            dma_handle=dma_handle,
            pool_name=pool.name,
        )
```

## 2. GPU-Direct Integration
```python
class GPUDevInterface:
    def __init__(self):
        self.gpudev_available = self.check_gpudev_support()
        self.gpu_handles = {}
    # ... (probing, registration, memory management)
```

## 3. Transfer Manager
```python
class TransferManager:
    def __init__(self, transport: Transport):
        self.transport = transport
        self.queue = asyncio.Queue()
        self.active = {}
    
    async def transfer_tensor(self, tensor: torch.Tensor, target: RemoteAccelerator) -> TransferFuture:
        if not self.transport.is_dma_capable(tensor):
            tensor = self.transport.prepare_for_dma(tensor)
        request = TransferRequest.from_tensor(tensor, target)
        await self.queue.put(request)
        fut = TransferFuture(request.tensor_id)
        self.active[request.tensor_id] = fut
        if self.should_flush_batch():
            await self.execute_batch()
        return fut
```

## 4. Transport Implementations
- `PinnedTCPTransport`: pinned host buffers + TCP bytestreams
- `RDMATransport`: DPDK mempools + ibverbs; RDMA WRITE/READ; gpudev path when available

## 5. Fallback Mechanisms
```python
class FallbackTransport:
    async def staged_copy(self, tensor: torch.Tensor, target: RemoteAccelerator):
        pinned = torch.empty_like(tensor, pin_memory=True)
        pinned.copy_(tensor)
        await self._tcp_send(pinned)
        return TransferResult.ok()
```

## Testing Requirements
```python
def test_dpdk_allocation():
    allocator = DPDKAllocator()
    buffer = allocator.allocate(1024 * 1024, torch.device('cpu'))
    assert buffer.size == 1024 * 1024
    assert buffer.dma_handle is not None

def test_zero_copy_transfer_stub():
    tm = TransferManager(PinnedTCPTransport())
    t = torch.randn(1024)
    fut = asyncio.run(tm.transfer_tensor(t, RemoteAccelerator("node1", 0)))
    assert fut is not None
```

## Performance Targets
- Memory allocation: <100µs for typical sizes
- DMA setup: <50µs per transfer
- Network utilization: >90% of link speed
- Fallback penalty: <20% performance loss

## Integration Points
- **Optimization Engine**: Receives transfer/execution requests
- **Remote Runtime**: Destination execution service
- **PyTorch Allocator**: Memory management integration
- **Network Stack**: DPDK/RDMA libraries; TCP fallback
