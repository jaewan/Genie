# Component: Zero-Copy Runtime

## Purpose
Implements efficient, zero-copy data movement between client and remote accelerators using DPDK and GPU-direct technologies.

## Context
- **Upstream**: Optimization Engine (transfer requests)
- **Downstream**: Remote GPU nodes
- **Interactions**: Memory pools, network stack, GPU drivers

## Key Requirements
- Network bandwidth utilization >90%
- Zero-copy from application to GPU
- <50μs DMA setup overhead
- Graceful fallback when GPU-direct unavailable

## Core Implementation

### 1. DPDK Memory Allocator
```python
class DPDKAllocator(torch.Allocator):
    def __init__(self):
        self.eal_initialized = False
        self.memory_pools = {}
        self.huge_page_size = "2MB"
        self.initialize_dpdk()
        
    def initialize_dpdk(self):
        # Initialize DPDK EAL (Environment Abstraction Layer)
        eal_args = [
            'genie',
            '-l', '0-1',  # Use 2 CPU cores
            '--socket-mem', '4096,4096',  # 4GB per socket
            '--huge-dir', '/dev/hugepages',
            '--file-prefix', 'genie'
        ]
        
        ret = dpdk.rte_eal_init(len(eal_args), eal_args)
        if ret < 0:
            raise RuntimeError(f"DPDK initialization failed: {ret}")
        
        self.eal_initialized = True
        self.create_memory_pools()
    
    def create_memory_pools(self):
        # Create pools for different size classes
        pool_configs = [
            ('small', 64 * 1024, 4096),      # 64KB buffers, 4096 count
            ('medium', 1 * 1024 * 1024, 1024), # 1MB buffers, 1024 count
            ('large', 16 * 1024 * 1024, 256),  # 16MB buffers, 256 count
            ('huge', 256 * 1024 * 1024, 64)    # 256MB buffers, 64 count
        ]
        
        for name, size, count in pool_configs:
            pool = dpdk.rte_mempool_create(
                name=f"genie_{name}",
                n=count,
                elt_size=size,
                cache_size=32,
                private_data_size=0,
                mp_init=None,
                mp_init_arg=None,
                obj_init=None,
                obj_init_arg=None,
                socket_id=dpdk.SOCKET_ID_ANY,
                flags=0
            )
            self.memory_pools[name] = pool
    
    def allocate(self, size: int, device: torch.device) -> DMABuffer:
        # Select appropriate pool
        pool = self.select_pool(size)
        
        # Allocate from DPDK pool
        mbuf = dpdk.rte_pktmbuf_alloc(pool)
        if not mbuf:
            raise MemoryError(f"Failed to allocate {size} bytes from DPDK")
        
        # Register for DMA
        dma_handle = self.register_for_dma(mbuf, size)
        
        return DMABuffer(
            data=mbuf.buf_addr,
            size=size,
            dma_handle=dma_handle,
            pool_name=pool.name
        )
```

### 2. GPU-Direct Integration
```python
class GPUDevInterface:
    def __init__(self):
        self.gpudev_available = self.check_gpudev_support()
        self.gpu_handles = {}
        
    def check_gpudev_support(self):
        # Check for DPDK gpudev library
        try:
            import dpdk.gpudev as gpudev
            return gpudev.rte_gpu_count() > 0
        except ImportError:
            return False
    
    def setup_gpu_direct(self, gpu_id: int) -> bool:
        if not self.gpudev_available:
            return False
            
        # Initialize GPU device in DPDK
        gpu_info = gpudev.rte_gpu_info_get(gpu_id)
        
        # Register GPU memory regions
        gpu_mem = gpudev.rte_gpu_mem_alloc(
            gpu_id,
            size=1024 * 1024 * 1024,  # 1GB initial allocation
            flags=gpudev.RTE_GPU_MEM_GDRCOPY
        )
        
        self.gpu_handles[gpu_id] = {
            'info': gpu_info,
            'memory': gpu_mem,
            'gdrcopy_available': self.check_gdrcopy(gpu_id)
        }
        
        return True
    
    def register_gpu_memory(self, gpu_ptr: int, size: int, gpu_id: int) -> DMAHandle:
        if gpu_id not in self.gpu_handles:
            self.setup_gpu_direct(gpu_id)
            
        # Register with DPDK for DMA
        handle = gpudev.rte_gpu_mem_register(
            gpu_id,
            size,
            gpu_ptr
        )
        
        return DMAHandle(
            iova=handle.iova,  # IO virtual address
            lkey=handle.lkey,  # Local key for RDMA
            rkey=handle.rkey,  # Remote key for RDMA
            gpu_id=gpu_id
        )
```

### 3. Transfer Manager
```python
class TransferManager:
    def __init__(self):
        self.dpdk_allocator = DPDKAllocator()
        self.gpudev = GPUDevInterface()
        self.transfer_queue = asyncio.Queue()
        self.active_transfers = {}
        
    async def transfer_tensor(self, 
                             tensor: torch.Tensor,
                             target: RemoteAccelerator) -> TransferFuture:
        # Check if tensor is already in DMA-capable memory
        if not self.is_dma_capable(tensor):
            tensor = self.prepare_for_dma(tensor)
        
        # Create transfer request
        request = TransferRequest(
            tensor_id=generate_uuid(),
            source=self.get_tensor_location(tensor),
            target=target,
            size=tensor.numel() * tensor.element_size(),
            priority=0
        )
        
        # Queue for batching
        await self.transfer_queue.put(request)
        
        # Create future for async completion
        future = TransferFuture(request.tensor_id)
        self.active_transfers[request.tensor_id] = future
        
        # Start transfer if batch is ready
        if self.should_flush_batch():
            await self.execute_batch()
        
        return future
    
    def prepare_for_dma(self, tensor: torch.Tensor) -> torch.Tensor:
        # Allocate DMA-capable buffer
        dma_buffer = self.dpdk_allocator.allocate(
            tensor.numel() * tensor.element_size(),
            tensor.device
        )
        
        # Copy tensor to DMA buffer
        if tensor.is_cuda:
            # GPU to DMA buffer
            cuda.memcpy_dtod(dma_buffer.data, tensor.data_ptr(), dma_buffer.size)
        else:
            # CPU to DMA buffer
            ctypes.memmove(dma_buffer.data, tensor.data_ptr(), dma_buffer.size)
        
        # Create tensor view of DMA buffer
        return torch.from_dlpack(dma_buffer.to_dlpack())
    
    async def execute_batch(self):
        batch = []
        while not self.transfer_queue.empty() and len(batch) < 16:
            batch.append(await self.transfer_queue.get())
        
        if not batch:
            return
        
        # Sort by target for locality
        batch.sort(key=lambda x: (x.target.node_id, x.target.gpu_id))
        
        # Execute transfers
        for request in batch:
            await self.execute_single_transfer(request)
    
    async def execute_single_transfer(self, request: TransferRequest):
        try:
            # Setup DMA descriptor
            descriptor = self.setup_dma_descriptor(request)
            
            # Initiate RDMA transfer
            if self.gpudev.gdrcopy_available:
                # Direct GPU-to-GPU copy
                await self.rdma_gpu_direct(descriptor)
            else:
                # Fallback to staged copy
                await self.rdma_staged_copy(descriptor)
            
            # Mark completion
            future = self.active_transfers[request.tensor_id]
            future.set_result(TransferResult(
                tensor_id=request.tensor_id,
                bytes_transferred=request.size,
                transfer_time_ms=descriptor.elapsed_ms
            ))
            
        except Exception as e:
            future = self.active_transfers[request.tensor_id]
            future.set_exception(TransferError(str(e)))
```

### 4. Network Protocol Handler
```python
class RDMAProtocol:
    def __init__(self):
        self.ibv_context = None
        self.pd = None  # Protection domain
        self.cq = None  # Completion queue
        self.qp = None  # Queue pair
        self.setup_rdma()
        
    def setup_rdma(self):
        # Get RDMA device
        dev_list = ibv.get_device_list()
        if not dev_list:
            raise RuntimeError("No RDMA devices found")
        
        # Open device context
        self.ibv_context = ibv.open_device(dev_list[0])
        
        # Allocate protection domain
        self.pd = ibv.alloc_pd(self.ibv_context)
        
        # Create completion queue
        self.cq = ibv.create_cq(self.ibv_context, 1024, None, None, 0)
        
        # Create queue pair
        qp_init_attr = ibv.QPInitAttr(
            send_cq=self.cq,
            recv_cq=self.cq,
            cap=ibv.QPCap(
                max_send_wr=1024,
                max_recv_wr=1024,
                max_send_sge=1,
                max_recv_sge=1
            ),
            qp_type=ibv.QPType.RC  # Reliable connection
        )
        self.qp = ibv.create_qp(self.pd, qp_init_attr)
    
    async def rdma_write(self, 
                        local_addr: int,
                        remote_addr: int, 
                        size: int,
                        lkey: int,
                        rkey: int):
        # Create scatter-gather entry
        sge = ibv.SGE(addr=local_addr, length=size, lkey=lkey)
        
        # Create send work request
        send_wr = ibv.SendWR(
            wr_id=generate_wr_id(),
            sg_list=[sge],
            num_sge=1,
            opcode=ibv.WROpcode.RDMA_WRITE,
            send_flags=ibv.SendFlags.SIGNALED,
            wr=ibv.RdmaWR(
                remote_addr=remote_addr,
                rkey=rkey
            )
        )
        
        # Post send request
        ibv.post_send(self.qp, send_wr, None)
        
        # Wait for completion
        wc = ibv.WC()
        while ibv.poll_cq(self.cq, 1, wc) == 0:
            await asyncio.sleep(0.001)
        
        if wc.status != ibv.WCStatus.SUCCESS:
            raise RuntimeError(f"RDMA write failed: {wc.status}")
```

### 5. Fallback Mechanisms
```python
class FallbackTransport:
    def __init__(self):
        self.tcp_sockets = {}
        
    async def staged_copy(self, tensor: torch.Tensor, target: RemoteAccelerator):
        # Stage 1: Copy to pinned host memory
        pinned = torch.empty_like(tensor, pin_memory=True)
        pinned.copy_(tensor)
        
        # Stage 2: Transfer over network
        socket = await self.get_or_create_socket(target)
        
        # Send header
        header = {
            'tensor_id': generate_uuid(),
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'size_bytes': pinned.numel() * pinned.element_size()
        }
        await self.send_json(socket, header)
        
        # Send data
        await self.send_tensor_data(socket, pinned)
        
        # Wait for acknowledgment
        ack = await self.recv_json(socket)
        if not ack['success']:
            raise RuntimeError(f"Transfer failed: {ack.get('error')}")
        
        return ack['tensor_id']
```

## Testing Requirements
```python
def test_dpdk_allocation():
    allocator = DPDKAllocator()
    buffer = allocator.allocate(1024 * 1024, torch.device('cpu'))
    assert buffer.size == 1024 * 1024
    assert buffer.dma_handle is not None

def test_gpu_direct_setup():
    gpudev = GPUDevInterface()
    if torch.cuda.is_available():
        success = gpudev.setup_gpu_direct(0)
        assert success or not gpudev.gpudev_available

def test_zero_copy_transfer():
    manager = TransferManager()
    tensor = torch.randn(1000, 1000)
    target = RemoteAccelerator(node_id="node1", gpu_id=0)
    
    future = asyncio.run(manager.transfer_tensor(tensor, target))
    result = future.result(timeout=5.0)
    
    assert result.bytes_transferred == tensor.numel() * tensor.element_size()
```

## Performance Targets
- Memory allocation: <100μs for typical sizes
- DMA setup: <50μs per transfer
- Network utilization: >90% of link speed
- Fallback penalty: <20% performance loss

## Integration Points
- **Optimization Engine**: Receives transfer requests
- **Remote Runtime**: Destination for transfers
- **PyTorch Allocator**: Memory management integration
- **Network Stack**: DPDK/RDMA libraries
