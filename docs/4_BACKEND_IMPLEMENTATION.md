# Djinn Backend Implementation

**Status**: ✅ Production Ready
**Last Updated**: November 6, 2025
**Focus**: Network transport, remote execution, GPU cache, result handling

---

## Table of Contents

1. [Overview](#1-overview)
2. [Server Architecture](#2-server-architecture)
3. [Network Transport Layer](#3-network-transport-layer)
4. [Serialization Protocol](#4-serialization-protocol)
5. [Remote Execution](#5-remote-execution)
6. [GPU Memory Management](#6-gpu-memory-management)
7. [Result Handling](#7-result-handling)
8. [Error Recovery and Fault Tolerance](#8-error-recovery-and-fault-tolerance)
9. [Production Best Practices](#9-production-best-practices)
10. [Performance Optimization](#10-performance-optimization)

---

## §1. Overview

### §1.1 Backend Responsibilities

The backend translates the scheduler's execution plan into **concrete execution** on remote GPUs:

```
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Server Orchestration: Coordinate execution across GPUs       │
│  2. Network Transport: Move data between client and server      │
│  3. Serialization: Convert tensors to wire format               │
│  4. Remote Execution: Execute operations on remote GPU          │
│  5. GPU Cache: Persistent weight storage (LRU)                  │
│  6. Result Handling: Return results to client                   │
│  7. Fault Tolerance: Recover from failures                      │
└─────────────────────────────────────────────────────────────────┘
```

### §1.2 Key Components

||| Component | File | Purpose |
|||-----------|------|---------|
||| **Main Server** | `djinn/server/server.py` | Server orchestration and lifecycle management |
||| **TCP Server** | `djinn/server/tcp_server.py` | Asyncio TCP server with binary protocol |
||| **TCP Transport** | `djinn/server/transport/tcp_transport.py` | Production transport with connection pooling |
||| **DPDK Transport** | `djinn/server/transport/dpdk_transport.py` | Zero-copy GPU transport (optional) |
||| **Optimization Executor** | `djinn/server/optimization_executor.py` | Integrated registry+fusion+execution |
||| **Phase Executor** | `djinn/server/phase_executor.py` | Phase-aware execution strategies |
||| **Phase-Aware Memory Manager** | `djinn/server/semantic_memory_manager.py` | Lifetime-based memory management |
||| **Adaptive Budget Tuner** | `djinn/server/adaptive_budget_tuner.py` | Dynamic memory budget tuning |
||| **GPU Cache** | `djinn/server/gpu_cache.py` | Weight caching (LRU) |
||| **Graph Cache** | `djinn/server/graph_cache.py` | Parsed graph caching |
||| **Tensor Registry** | `djinn/server/tensor_registry.py` | Smart caching with version-aware keys |
||| **Fusion Compiler** | `djinn/server/fusion_compiler.py` | Pattern grouping and optimization |

---

## §2. Server Architecture

### §2.1 Architecture Overview

Djinn's backend uses an **asyncio-based TCP architecture** with length-prefixed binary framing, not HTTP/FastAPI. The server consists of two main components:

**Files**: `djinn/server/server.py`, `djinn/server/tcp_server.py`

### §2.2 Main Server (`djinn/server/server.py`)

The main server orchestrates the entire backend lifecycle:

```python
class DjinnServer:
    """Main server for Djinn disaggregated GPU cluster."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.capabilities = None
        self.coordinator = None
        self.executor = None
        self.tcp_server = None  # Raw asyncio TCP server
        self.transport = None    # Transport for sending results

    async def start(self) -> bool:
        """Start the Djinn server with all components."""
        # 1. Discover GPU capabilities
        # 2. Start TCP server for operation requests
        # 3. Initialize transport with operation callbacks
        # 4. Initialize optimization executor
        # 5. Start background heartbeat/transfer loops
```

**Key Features**:
- **GPU Discovery**: Automatically detects available GPUs and their capabilities
- **Transport Management**: Handles both incoming operation requests and outgoing results
- **Lifecycle Management**: Proper startup/shutdown of all components
- **Callback Integration**: Wires operation handlers to transport layer

### §2.3 TCP Server (`djinn/server/tcp_server.py`)

Raw asyncio TCP server with binary length-prefixed protocol:

```python
# Global state for the TCP server
DEVICE = None
OPTIMIZATION_EXECUTOR = None
GPU_CACHE = None
GRAPH_CACHE = None

async def _handle_connection(reader, writer):
    """Handle incoming TCP connections with binary protocol."""

    # 1. Read transfer_id (length-prefixed)
    transfer_id_len_bytes = await reader.readexactly(4)
    transfer_id_len = struct.unpack('>I', transfer_id_len_bytes)[0]
    transfer_id_bytes = await reader.readexactly(transfer_id_len)
    transfer_id = transfer_id_bytes.decode('utf-8')

    # 2. Read metadata (length-prefixed JSON)
    metadata_len_bytes = await reader.readexactly(4)
    metadata_len = struct.unpack('>I', metadata_len_bytes)[0]
    metadata_bytes = await reader.readexactly(metadata_len)
    metadata = json.loads(metadata_bytes.decode('utf-8'))

    # 3. Read tensor data (length-prefixed binary)
    # 4. Execute operation via optimization executor
    # 5. Send result back via transport
```

**Protocol**: Length-prefixed binary framing for efficient tensor transfer
**Performance**: Optimized for low-latency GPU operations
**Scalability**: Handles multiple concurrent connections

---

## §3. Network Transport Layer

### §3.1 Transport Overview

Djinn supports **two transport mechanisms** with different design tradeoffs:

||| Transport | Characteristics | Use Case | Status |
|||-----------|-----------------|----------|--------|
||| **TCP** | Low-latency, connection pooling, async/await | Production | ✅ Deployed |
||| **DPDK** | Zero-copy, ultra-low latency, hardware requirements | High-performance | 🚧 Optional |

Choose based on your infrastructure constraints and requirements.

### §3.2 TCP Transport (Production Primary)

**File**: `djinn/server/transport/tcp_transport.py`

**Architecture**: Async/await-based with connection pooling, length-prefixed protocol, and zero-copy optimizations

#### Core Design Features

**1. Connection Pooling**

```python
class ConnectionPool:
    """Intelligent connection reuse and health checking."""

    def __init__(self, max_per_target: int = 5, enable_warming: bool = True):
        self._pools: Dict[str, asyncio.Queue] = {}
        self._frequent_targets: Dict[str, int] = {}

    async def acquire(self, target: str) -> Connection:
        """Get or create connection with intelligent reuse."""
        # Track frequency for pre-warming
        # Reuse healthy connections from pool
        # Auto-health-check on reuse
        # Pre-warm connections for hot targets
```

**Benefits**:
- Avoids TCP handshake overhead (50ms per connection)
- Automatic health checking (detects stale connections)
- Connection warming for frequently-used targets
- Resource limits (max 5 per target)

**2. Length-Prefixed Protocol**

```
Frame Structure:
┌─────────────────────────────────────────────┐
│ Transfer ID:                                │
│   [4 bytes: length]                         │
│   [N bytes: transfer_id (UTF-8)]            │
├─────────────────────────────────────────────┤
│ Metadata:                                   │
│   [4 bytes: length]                         │
│   [N bytes: metadata JSON]                  │
├─────────────────────────────────────────────┤
│ Tensor Data:                                │
│   [8 bytes: total size]                     │
│   [N bytes: tensor buffer]                  │
└─────────────────────────────────────────────┘
```

**Advantages**:
- No delimiters needed (efficient parsing)
- Fixed-size headers (fast framing)
- Zero-copy capable (memoryview access)
- Supports multi-tensor batching

**3. Zero-Copy Serialization**

```python
async def send(self, tensor, target, transfer_id, metadata):
    # Move to CPU if needed (non_blocking=True for overlap)
    cpu_tensor = tensor.cpu(non_blocking=True).contiguous()

    # Direct storage access (no copy)
    storage = cpu_tensor.storage()
    tensor_bytes = memoryview(storage).cast('B')

    # Send via memoryview (zero-copy at Python level)
    await self._send_tensor(conn, cpu_tensor)
```

#### Connection Lifecycle

```
1. ACQUIRE
   └─ Check pool for healthy connections
      ├─ Found: Return existing (stats.reused += 1)
      └─ Not found: Create new (stats.created += 1)

2. USE
   └─ Send frames via async I/O
      ├─ Success: Mark successful
      └─ Failure: Mark unhealthy

3. RELEASE
   ├─ Success: Return to pool
   └─ Failure: Close connection
```

### §3.3 DPDK Transport (Optional)

**File**: `djinn/server/transport/dpdk_transport.py`

**Vision**: GPU-to-GPU zero-copy via DPDK + GPUDirect RDMA

```
Current path (TCP):
  GPU → CPU → NIC → Network → NIC → CPU → GPU (4 copies)

DPDK path (future):
  GPU → NIC → Network → NIC → GPU (0 copies)
```

**Requirements**:
- DPDK-compatible NIC (Mellanox ConnectX-5+)
- GPUDirect RDMA driver support
- IOMMU configuration
- Kernel modules (gpudev)

**Expected Impact**: <1ms latency vs 10ms with TCP

### §3.4 Transport Selection Strategy

**Implementation**: TCP is the primary transport, DPDK is optional enhancement

```python
# In server.py initialization
self.transport = TCPTransport(server_config)
await self.transport.initialize()

# DPDK only if explicitly configured and hardware available
if self.config.prefer_dpdk:
    try:
        from .transport.dpdk_transport import DPDKTransport
        # Initialize DPDK if available
    except Exception as e:
        logger.warning(f"DPDK unavailable: {e}")
```

---

## §4. Serialization Protocol

### §4.1 Serialization Requirements

**Challenges**:
1. Tensors (large, binary data)
2. Operations (ATen ops, not JSON-serializable)
3. Special types (dtype, slice, tuple)
4. Model parameters (nn.Parameter vs torch.Tensor)

### §4.2 Optimized Serialization

**File**: `djinn/server/serialization.py`

**Two approaches with automatic format detection**:

1. **torch.save** (standard, backward compatible):
```python
buffer = io.BytesIO()
torch.save(tensor_dict, buffer)
bytes_data = buffer.getvalue()
```

2. **Numpy buffer** (optimized, 23% faster):
```python
# Format header for version detection
buffer.write(FORMAT_NUMPY)

cpu_tensor = tensor.cpu().contiguous()
numpy_array = cpu_tensor.numpy()

# Use numpy.save for efficient serialization
np.save(buffer, numpy_array, allow_pickle=False)
bytes_data = buffer.getvalue()
```

**Automatic Format Detection**:
```python
def deserialize_tensor(data: bytes) -> torch.Tensor:
    buffer = io.BytesIO(data)

    # Read format header (4 bytes)
    header = buffer.read(HEADER_SIZE)

    if header == FORMAT_NUMPY:
        # Fast path: numpy.load
        result_np = np.load(buffer, allow_pickle=False)
        tensor = torch.from_numpy(result_np)
    elif header == FORMAT_TORCH:
        # Compatible path: torch.load
        tensor = torch.load(buffer)
    else:
        # Legacy path: assume torch.save format
        buffer.seek(0)
        tensor = torch.load(buffer)

    return tensor
```

**Performance Characteristics**:
- `torch.save`: ~185ms for 196MB tensor (GPT-2 output)
- `numpy.save`: ~143ms for 196MB tensor (23% faster, zero-copy when possible)
- **Automatic fallback**: Old data formats remain compatible

---

## §5. Remote Execution

### §5.1 Execution Architecture

The backend uses a **modular execution pipeline** with integrated optimization components:

1. **Optimization Executor** (`djinn/server/optimization_executor.py`)
   - Integrates tensor registry, fusion compiler, phase-aware memory manager, and execution
   - Handles operation dispatch and result routing
   - Includes proactive memory pressure handling and TensorRT compilation

2. **Phase Executor** (`djinn/server/phase_executor.py`)
   - Phase-aware execution strategies (Prefill/Decode/Vision)
   - Automatic phase detection and optimization switching
   - Supports different memory allocation strategies per phase

3. **Subgraph Executor** (`djinn/server/subgraph_executor.py`)
   - Core GPU execution engine
   - Topological execution order with memory tracking
   - Batch execution optimization with CUDA streams

4. **Materialization Optimizer** (integrated into LazyTensor)
   - Topological sort for efficient batch execution
   - CUDA streams for computation and transfer overlap
   - Pinned memory allocation for faster CPU↔GPU transfers

### §5.2 Execution Flow

```python
# In tcp_server.py operation handler
async def _handle_operation_request(self, transfer_id, tensors, metadata):
    # 1. Extract operation and parameters
    operation = metadata.get('operation')
    result_id = metadata.get('result_id')

    # 2. Execute via optimization executor
    result = await self.executor.execute(operation, *tensors)

    # 3. Send result back to client
    await self.result_transport.send(result, target=client_target, ...)
```

---

## §6. GPU Memory Management

### §6.1 Three-Phase Memory Management

**Phase 1: Reactive Memory Management** ✅
- Enhanced GPU cache with memory-aware eviction
- Per-model memory tracking

**Phase 2: Semantic-Aware Memory Management** ✅
- **Lifetime-based eviction** (`djinn/server/semantic_memory_manager.py`)
- **Phase-aware budgets** (different strategies for prefill vs decode)
- **Cost-based recomputation** (should we recompute or cache?)

**Phase 3: Production Hardening** ✅
- **Phase-aware memory manager** integrated into optimization executor
- **Proactive memory pressure handler** with configurable thresholds
- **TensorRT compilation** for repeated execution optimization
- **Prometheus metrics integration** (50+ metrics)
- **Adaptive budget tuner** (`djinn/server/adaptive_budget_tuner.py`)

### §6.2 Advanced Memory Features

#### Phase-Aware Memory Manager
```python
class PhaseAwareMemoryManager:
    """Leverages SRG structure for intelligent memory decisions."""

    def __init__(self, total_gpu_memory_mb: float):
        self.total_memory_mb = total_gpu_memory_mb
        self.current_phase = ExecutionPhase.UNKNOWN
        # Different budget allocations per phase
        self.budgets: Dict[str, float] = {}

    def adjust_for_phase(self, phase: ExecutionPhase) -> None:
        """Adjust memory budgets based on execution phase."""
        if phase == ExecutionPhase.LLM_PREFILL:
            # Prefill: Parallel attention, needs activation memory
            self.budgets = {
                'weights': 0.3 * self.total_memory_mb,      # 30%
                'activations': 0.6 * self.total_memory_mb,  # 60%
                'kv_cache': 0.1 * self.total_memory_mb      # 10%
            }
        elif phase == ExecutionPhase.LLM_DECODE:
            # Decode: Sequential, memory-bound, growing KV cache
            self.budgets = {
                'weights': 0.3 * self.total_memory_mb,      # 30%
                'activations': 0.1 * self.total_memory_mb,  # 10%
                'kv_cache': 0.6 * self.total_memory_mb      # 60%
            }
```

#### Memory Pressure Handler
```python
class MemoryPressureHandler:
    """Proactive memory management with configurable thresholds."""

    def __init__(self, total_gpu_memory_mb: float,
                 warning_threshold_percent: float = 80.0,
                 critical_threshold_percent: float = 95.0):
        self.total_memory_mb = total_gpu_memory_mb
        self.warning_threshold = warning_threshold_percent / 100.0
        self.critical_threshold = critical_threshold_percent / 100.0
        # Eviction callbacks for different memory pressure levels
        self.eviction_callbacks: Dict[str, Callable] = {}

    async def monitor_memory_pressure(self):
        """Continuous monitoring with automatic eviction triggers."""
        while True:
            utilization = self._get_memory_utilization()
            if utilization > self.critical_threshold:
                await self._trigger_critical_eviction()
            elif utilization > self.warning_threshold:
                await self._trigger_warning_eviction()
            await asyncio.sleep(self.monitoring_interval)
```

#### Adaptive Budget Tuning
```python
class AdaptiveBudgetTuner:
    """Learns optimal phase-specific memory budgets."""

    def record_observation(self, phase: str, utilization: float):
        """Track memory utilization per phase."""
        # Collect observations across workloads
        # Calculate efficiency scores
        # Gradually adjust budgets toward optimal allocation
```

### §6.3 Memory Distribution (32GB GPU)

| Phase | Weights | Activations | KV Cache |
|-------|---------|-------------|----------|
| **Prefill** | 30% | 60% | 10% |
| **Decode** | 30% | 10% | 60% |

*Budgets adapt based on observed utilization patterns.*

---

## §7. Result Handling

### §7.1 Result Serialization

**Server-side**: Automatic format selection with optimized numpy serialization

```python
# tcp_server.py result handling
try:
    if USE_OPTIMIZED_SERIALIZATION and OPTIMIZED_SERIALIZATION_AVAILABLE:
        logger.info("✅ Using optimized numpy.save serialization")
        result_bytes = serialize_tensor(result, use_numpy=True)
    else:
        logger.warning("⚠️ Using fallback torch.save serialization")
        result_buffer = io.BytesIO()
        torch.save(result, result_buffer)
        result_bytes = result_buffer.getvalue()
except Exception as e:
    logger.error(f"Failed to serialize result: {e}")
    # Error handling with fallback to local execution
```

**Client-side**: Format-aware deserialization with backward compatibility

```python
# coordinator.py result handling
def execute_remote_subgraph(self, ...):
    # Receive message with type and payload
    msg_type_bytes = await reader.readexactly(1)
    msg_type = msg_type_bytes[0]

    size_bytes = await reader.readexactly(4)
    size = int.from_bytes(size_bytes, 'big')
    result_data = await reader.readexactly(size)

    if msg_type == 0x04:  # RESULT
        # Try optimized deserialization first
        try:
            from ...server.serialization import deserialize_tensor
            response = deserialize_tensor(result_data)
        except Exception as e:
            # Fallback to torch.load for compatibility
            import io
            result_buffer = io.BytesIO(result_data)
            response = torch.load(result_buffer)
    # Handle other message types...
```

### §7.2 Result Routing

**Client Port Resolution**: Results are routed back to the correct client port

```python
# Extract client port from operation metadata
client_port = metadata.get('client_port')
if client_port:
    result_target = f"{client_ip}:{client_port}"
    await self.result_transport.send(result, target=result_target, ...)
```

---

## §8. Error Recovery and Fault Tolerance

### §8.1 Error Types

| Error Type | Cause | Recovery Strategy |
|------------|-------|-------------------|
| **Network Timeout** | Slow network, server overload | Retry with exponential backoff |
| **Connection Error** | Server down, network partition | Failover to alternative server |
| **GPU OOM** | Insufficient GPU memory | Evict cached models, retry |
| **Execution Error** | Invalid operation, shape mismatch | Log error, return to client |
| **Serialization Error** | Unsupported type | Fallback to local execution |

### §8.2 Recovery Implementation

**GPU OOM Recovery**:
```python
try:
    result = executor.execute(operation, *tensors)
except torch.cuda.OutOfMemoryError:
    # 1. Evict cached models (LRU)
    GPU_CACHE.evict_oldest()

    # 2. Clear CUDA cache
    torch.cuda.empty_cache()

    # 3. Retry execution
    result = executor.execute(operation, *tensors)
```

---

## §9. Production Best Practices

### §9.1 Configuration

**TCP Transport Configuration**:
```python
from djinn.server.transport.tcp_transport import TCPTransport

config = CoordinatorConfig(
    data_port=5556,
    connection_pool_size=4,
    connection_timeout=30.0
)

transport = TCPTransport(config)
```

### §9.2 Monitoring

**Key Metrics**:
- GPU cache hit rate (target: >90%)
- Graph cache hit rate (target: >95%)
- Network latency and throughput
- GPU memory utilization
- Backend request latency

### §9.3 Health Checks

```python
@app.get("/health")
async def health_check():
    """Production health check endpoint."""

    # Check GPU availability
    if not torch.cuda.is_available():
        raise HTTPException(503, "GPU not available")

    # Check GPU memory
    free_memory = torch.cuda.mem_get_info()[0]
    if free_memory < 2_000_000_000:  # 2GB threshold
        raise HTTPException(503, "Insufficient GPU memory")

    return {"status": "healthy", "gpu_memory_free_gb": free_memory / 1024**3}
```

---

## §10. Performance Optimization

### §10.1 Optimization Checklist

- ✅ **TCP transport** (production default)
- ✅ **Enable GPU cache** (recommended for repeated workloads)
- ✅ **Enable graph cache** (recommended for repeated graphs)
- ✅ **Connection pooling** (automatic with TCP)
- ✅ **Optimized numpy serialization** (23% faster for large tensors)
- ✅ **Phase-aware memory management** (automatic budget adjustment)
- ✅ **Proactive memory pressure handling** (OOM prevention)
- ✅ **TensorRT compilation** (2-3x speedup for repeated executions)
- ✅ **Materialization optimization** (batch execution with CUDA streams)
- ✅ **Differential graph protocol** (10x network reduction for iterative workloads)
- 🔧 **Zero-copy serialization** (optional, requires specific tensor layouts)
- 🔧 **Operation batching** (automatic for small operations)
- 🔧 **DPDK transport** (optional, requires specialized hardware)

### §10.2 Performance Targets

**Latency**:
- Local execution: <5ms
- Network round-trip: <10ms (TCP), <1ms (DPDK)
- Serialization: 23% faster for large tensors (196MB: 185ms → 143ms)

**Throughput**:
- Single GPU: 1000+ operations/second
- Multi-GPU: 5000+ operations/second
- Iterative workloads: 10x network reduction with differential protocol

**Memory Efficiency**:
- Cache hit rate: >90%
- Memory utilization: Phase-aware adaptive (70% of GPU memory)
- OOM prevention: Proactive monitoring with configurable thresholds

**Optimization Impact**:
- Large tensor serialization: 23% reduction in result handling time
- Iterative workloads: Up to 10x reduction in network transfers
- Memory-constrained workloads: Phase-specific budget optimization
- Repeated executions: TensorRT compilation for 2-3x speedup

---

## §11. Conclusion

The Djinn backend provides **production-ready remote execution** with integrated performance optimizations:

✅ **Asyncio TCP architecture** (not HTTP/FastAPI)
✅ **Binary length-prefixed protocol** for efficient tensor transfer
✅ **Optimized numpy serialization** (23% faster for large tensors)
✅ **Phase-aware memory management** with automatic budget adjustment
✅ **Proactive memory pressure handling** with configurable thresholds
✅ **TensorRT compilation** for repeated execution optimization
✅ **Differential graph protocol** for 10x network reduction in iterative workloads
✅ **Materialization optimization** with batch execution and CUDA streams
✅ **Comprehensive error recovery** and fault tolerance
✅ **Production monitoring** with health checks and metrics

**Design Highlights**:
- Modular architecture supports multiple transports
- Configurable caching layers with semantic awareness
- Integrated optimization pipeline (serialization → memory → execution → network)
- Built-in monitoring and diagnostics
- Automatic performance optimization based on workload patterns

**Architecture Evolution**:
- Phase 1: Basic reactive memory management
- Phase 2: Semantic-aware memory and execution
- Phase 3: Production hardening with integrated optimizations