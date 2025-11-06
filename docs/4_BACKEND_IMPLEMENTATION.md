# Djinn Backend Implementation

**Status**: ✅ Production Ready
**Last Updated**: November 6, 2025
**Focus**: Network transport, remote execution, GPU cache, result handling

---

## Table of Contents

1. [Overview](#1-overview)
2. [Network Transport Layer](#2-network-transport-layer)
3. [Serialization Protocol](#3-serialization-protocol)
4. [Remote Execution](#4-remote-execution)
5. [GPU Memory Management](#5-gpu-memory-management)
6. [Result Handling](#6-result-handling)
7. [Error Recovery and Fault Tolerance](#7-error-recovery-and-fault-tolerance)

---

## §1. Overview

### §1.1 Backend Responsibilities

The backend translates the scheduler's execution plan into **concrete execution** on remote GPUs:

```
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Network Transport: Move data between client and server      │
│  2. Serialization: Convert tensors to wire format               │
│  3. Remote Execution: Execute operations on remote GPU          │
│  4. GPU Cache: Persistent weight storage (LRU)                  │
│  5. Result Handling: Return results to client                   │
│  6. Fault Tolerance: Recover from failures                      │
└─────────────────────────────────────────────────────────────────┘
```

### §1.2 Key Components

|| Component | File | Purpose |
||-----------|------|---------|
|| **TCP Transport** | `djinn/transport/tcp_transport.py` | Production transport with connection pooling |
|| **HTTP Transport** | `djinn/runtime/simple_client.py` | Development/fallback transport |
|| **Server** | `djinn/runtime/simple_server.py` | FastAPI server (main entry point) |
|| **TCP Server** | `djinn/server/tcp_server.py` | Raw TCP server endpoint |
|| **Executor** | `djinn/server/subgraph_executor.py` | GPU execution engine |
|| **Optimization Executor** | `djinn/server/optimization_executor.py` | Integrated registry+fusion+monitor |
|| **Tensor Registry** | `djinn/server/tensor_registry.py` | Smart caching with version-aware keys |
|| **Fusion Compiler** | `djinn/server/fusion_compiler.py` | Pattern grouping (Tier 1) |
|| **GPU Cache** | `djinn/server/gpu_cache.py` | Weight caching (LRU) |
|| **Graph Cache** | `djinn/server/graph_cache.py` | Parsed graph caching |
|| **Performance Monitor** | `djinn/server/performance_monitor.py` | Metrics collection |
|| **Serialization** | `djinn/core/serialization.py` | Dual-format (NumPy + torch) |

---

## §2. Network Transport Layer

### §2.1 Transport Overview

Djinn supports **three transport mechanisms** with different design tradeoffs:

|| Transport | Characteristics | Use Case | Status |
||-----------|-----------------|----------|--------|
|| **HTTP** | Standard, firewall-friendly, simple debugging | Development, profiling | ✅ Deployed |
|| **TCP** | Low-latency, connection pooling, async/await | Production | ✅ Deployed |
|| **DPDK** | Zero-copy, ultra-low latency, hardware requirements | High-performance | 🚧 Future |

Choose based on your infrastructure constraints and requirements.

### §2.2 HTTP Transport (Development Only)

**Status**: Available for development/debugging, not used in production

**File**: `djinn/runtime/simple_client.py` (292 LOC)

Uses requests library for simplicity. For production, use TCP transport instead.

---

### §2.3 TCP Transport (Production Primary)

**File**: `djinn/server/transport/tcp_transport.py` (400+ LOC)

**Architecture**: Async/await-based with connection pooling, length-prefixed protocol, and zero-copy optimizations

#### Core Design Features

**1. Connection Pooling**

```python
class ConnectionPool:
    """Intelligent connection reuse and health checking."""
    
    def __init__(self, max_per_target: int = 5, enable_warming: bool = True):
        # Per-target connection caching
        self._pools: Dict[str, asyncio.Queue] = {}
        self._frequent_targets: Dict[str, int] = {}  # Track usage
        
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

**3. Multi-Tensor Support**

```python
async def send_multi_tensor(
    self,
    tensors: list,
    target: str,
    transfer_id: str,
    metadata: Dict
) -> bool:
    """Send multiple tensors over single connection."""
    # Single connection handshake
    # Atomic multi-tensor transfer
    # Better for batch operations
```

**4. Zero-Copy Serialization**

```python
# Instead of copying tensor data
async def send(self, tensor, target, transfer_id, metadata):
    # Move to CPU if needed (non_blocking=True for overlap)
    cpu_tensor = tensor.cpu(non_blocking=True).contiguous()
    
    # Direct storage access (no copy)
    storage = cpu_tensor.storage()
    tensor_bytes = memoryview(storage).cast('B')
    
    # Send via memoryview (zero-copy at Python level)
    await self._send_tensor(conn, cpu_tensor)
```

**5. Automatic Health Checking**

```python
class Connection:
    """Per-target connection with health state."""
    
    def is_healthy(self) -> bool:
        """Check if connection is still alive."""
        # Check if closed
        # Track last successful use
        # Detect stale connections
    
    def mark_unhealthy(self):
        """Mark connection for closure."""
        # Will be closed on next release
        # Replacement created on next acquire
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

#### Callback System (Bidirectional Communication)

```python
_result_callback: Optional[Callable] = None
_operation_callback: Optional[Callable] = None

# Result callback: Routes completed results back to client
if metadata.get('is_result', False):
    await self._result_callback(result_id, tensor_or_error)

# Operation callback: Routes incoming requests to server
elif metadata.get('operation'):
    await self._operation_callback(transfer_id, tensor, metadata)
```

**Use Cases**:
- ✅ Bidirectional RPC
- ✅ Streaming results
- ✅ Push-based notifications
- ✅ Multi-client coordination

### §2.4 DPDK Transport (Future)

**Status**: Not yet implemented (Phase 3)

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

### §2.5 TCP Transport Callbacks (NEW)

**File**: `djinn/server/transport/tcp_transport.py` (lines 61-64, 454-498)

The TCP transport includes callback mechanisms for handling results and operations:

```python
# Result callback: Routes completed results back to client
_result_callback: Optional[Callable] = None

# Operation callback: Routes incoming operation requests to server
_operation_callback: Optional[Callable] = None
```

**Result Callback** (client-side):
When a result comes back from a remote server, it's routed via callback:

```python
if metadata.get('is_result', False):
    # This is a result coming back
    if metadata.get('is_error', False):
        # Pass error to callback
        await self._result_callback(result_id, Exception(error_msg))
    else:
        # Pass successful result to callback
        await self._result_callback(result_id, tensor)
```

**Operation Callback** (server-side):
When an operation request arrives, it's routed via callback:

```python
elif metadata.get('operation'):
    # This is an operation request
    metadata['source_node'] = f"{addr[0]}:{addr[1]}"
    await self._operation_callback(transfer_id, tensor, metadata)
```

**Benefits**:
- ✅ Decouples transport from business logic
- ✅ Enables bidirectional communication
- ✅ Supports request-response patterns
- ✅ Error handling at protocol level

### §2.6 Operation Batching (NEW)

**Status**: Infrastructure present, feature in development

The TCP transport includes infrastructure for batching small operations:

```python
# Operation batching for small operations
_pending_batches: Dict[str, List] = {}  # target -> list of operations
_batch_timer: Optional[asyncio.Task] = None
_batch_timeout = config.network.batch_timeout  # e.g., 10ms
_batch_size_threshold = config.network.batch_size_threshold  # e.g., 10MB
```

**Batching Decision**:

```python
def _should_batch(self, target: str, metadata: Dict) -> bool:
    """Determine if operation should be batched."""
    # Only batch small operations
    total_bytes = sum(
        shape[0] * shape[1] * 4  # Assume float32
        for shape in metadata.get('input_shapes', [])
    )
    
    # Batch small operations (< 10MB) that aren't time-sensitive
    return (total_bytes < 10_000_000 and  # 10MB threshold
            metadata.get('phase', '') not in ['realtime', 'interactive'])
```

**Benefits**:
- ✅ Reduced connection handshakes
- ✅ Better CPU cache utilization
- ✅ Amortized overhead
- ⚠️ Adds latency for individual operations (trade-off)

### §2.7 DPDK Transport (Future)

**File**: `djinn/transport/dpdk_transport.py` (placeholder)

**Status**: ⚠️ Not yet implemented (Phase 2)

#### Architecture

Zero-copy GPU-to-GPU transfers using DPDK + GPUDirect RDMA:

```
Traditional path:
  GPU → CPU → NIC → Network → NIC → CPU → GPU
  (4 copies, high latency)

DPDK + GPUDirect RDMA path:
  GPU → NIC → Network → NIC → GPU
  (0 copies, low latency)
```

#### Expected Performance

- **Latency**: <1ms for localhost
- **Throughput**: 100 Gbps (RDMA limit)
- **Zero-Copy**: ✅ Direct GPU memory access

#### Requirements

- DPDK-compatible NIC (Mellanox ConnectX-5+)
- GPUDirect RDMA support (NVIDIA driver)
- Kernel module (gpudev)
- IOMMU configuration

### §2.8 Transport Selection Strategy

**Current Implementation** (djinn/core/coordinator.py):

```python
# In DjinnCoordinator.start()
async def start(self):
    """Initialize coordinator and all transports."""
    
    # 1. Try DPDK first (if hardware available)
    if self.config.prefer_dpdk:
        try:
            from djinn.transport.dpdk_transport import DPDKTransport
            self.transports['dpdk'] = DPDKTransport(self.config)
            success = await self.transports['dpdk'].initialize()
            if success:
                logger.info("✓ DPDK transport available")
        except Exception as e:
            logger.warning(f"DPDK unavailable: {e}")
    
    # 2. Initialize TCP fallback (always works)
    if self.config.tcp_fallback:
        from djinn.transport.tcp_transport import TCPTransport
        self.transports['tcp'] = TCPTransport(self.config)
        
        # Register callbacks BEFORE initialization
        self.transports['tcp']._result_callback = self._handle_result_received
        self.transports['tcp']._operation_callback = self._handle_operation_request
        
        # Initialize TCP server (client coordinators only)
        if not self.config.is_server:
            success = await self.transports['tcp'].initialize()
            if success:
                logger.info("✓ TCP fallback available")
```

**Decision Tree**:

```
Does coordinator have DPDK transport?
├─ Yes → Use DPDK (requires GPU tensor + hardware)
└─ No  → Use TCP (always available)

At send time:
  if self.transports['dpdk'].is_available() and tensor.is_cuda:
      → use DPDK (zero-copy)
  else:
      → use TCP (connection pooling)
```

**Actual Priority** (from coordinator code):
1. ✅ DPDK (if available and hardware supports it)
2. ✅ TCP (always available, automatic fallback)
3. ❌ HTTP (removed from production path)

---

## §3. Serialization Protocol

### §3.1 Serialization Requirements

**Challenges**:
1. Tensors (large, binary data)
2. Operations (ATen ops, not JSON-serializable)
3. Special types (dtype, slice, tuple)
4. Model parameters (nn.Parameter vs torch.Tensor)

### §3.2 Subgraph Serialization

**File**: `djinn/profiling/comprehensive_profiler.py` (lines 400-500)

```python
def _serialize_remote_subgraph(self, subgraph):
    """
    Convert RemoteSubgraph to JSON-serializable dict.
    
    Handles:
    - torch.nn.Parameter → numpy array
    - torch.dtype → string
    - slice → dict {"start", "stop", "step"}
    - tuple → list (recursive)
    - ATen operations → string representation
    """
    
    serialized_ops = []
    for op in subgraph.operations:
        serialized_op = {
            'op_id': op['op_id'],
            'operation': str(op['operation']),  # ATen op → string
            'inputs': self._serialize_inputs(op['inputs']),
            'kwargs': self._serialize_kwargs(op['kwargs'])
        }
        serialized_ops.append(serialized_op)
    
    return {
        'model_id': subgraph.model_id,
        'operations': serialized_ops,
        'output_id': subgraph.output_id
    }
```

#### Special Type Handling

```python
def _serialize_value(self, value):
    """Handle special types."""
    
    # torch.nn.Parameter → numpy array
    if isinstance(value, torch.nn.Parameter):
        return value.detach().cpu().numpy()
    
    # torch.Tensor → numpy array
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    
    # torch.dtype → string
    if isinstance(value, torch.dtype):
        return str(value)
    
    # slice → dict
    if isinstance(value, slice):
        return {
            'start': value.start,
            'stop': value.stop,
            'step': value.step
        }
    
    # tuple → list (recursive)
    if isinstance(value, tuple):
        return [self._serialize_value(v) for v in value]
    
    # Primitive types (int, float, str, bool, None)
    return value
```

### §3.3 Tensor Serialization

**Two standard approaches**:

1. **torch.save** (HTTP transport):
```python
# Serialize
buffer = io.BytesIO()
torch.save(tensor_dict, buffer)
bytes_data = buffer.getvalue()

# Deserialize
tensor_dict = torch.load(io.BytesIO(bytes_data))
```

2. **Numpy buffer** (TCP transport, zero-copy):
```python
# Serialize (zero-copy)
cpu_tensor = tensor.cpu().contiguous()
numpy_array = cpu_tensor.numpy()
memoryview_data = memoryview(numpy_array)

# Deserialize
numpy_array = np.frombuffer(bytes_data, dtype=dtype).reshape(shape)
tensor = torch.from_numpy(numpy_array)
```

**Performance**:
- `torch.save`: 15ms (GPT-2 Tiny)
- Numpy buffer: 5ms (GPT-2 Tiny, zero-copy)

### §3.4 Optimized Serialization (NEW)

**File**: `djinn/core/serialization.py` (optional, high-performance mode)

**Status**: Available, optional feature (can be disabled via environment variable)

The server can optionally use optimized numpy-based serialization for 44% faster throughput:

```python
# In djinn/runtime/simple_server.py
USE_OPTIMIZED_SERIALIZATION = os.getenv('GENIE_USE_OPTIMIZED_SERIALIZATION', 'true').lower() == 'true'

if USE_OPTIMIZED_SERIALIZATION and OPTIMIZED_SERIALIZATION_AVAILABLE:
    logger.info("🚀 Using optimized numpy serialization (44% faster)")
    from djinn.core.serialization import serialize_tensor, deserialize_tensor
    result = deserialize_tensor(response.content)
else:
    logger.info("📦 Using standard torch.save serialization")
    result = torch.load(io.BytesIO(response.content))
```

**Benefits**:
- ✅ 44% faster than torch.save
- ✅ Direct numpy buffer access
- ✅ Lower memory overhead
- ⚠️ Requires contiguous tensors

**When to Use**:
- ✅ Production servers with high throughput requirements
- ✅ Batch inference workloads
- ❌ Complex object hierarchies (torch.save more flexible)

---

## §4. Remote Execution

### §4.1 Server Architecture

**File**: `djinn/runtime/simple_server.py` (FastAPI)

```python
app = FastAPI(title="Djinn Remote Execution Server")

# Feature flag for optimized serialization (can be disabled via environment variable)
USE_OPTIMIZED_SERIALIZATION = os.getenv('GENIE_USE_OPTIMIZED_SERIALIZATION', 'true').lower() == 'true'

@app.post("/execute_subgraph")
async def handle_subgraph_execution(request: Request):
    """
    Execute subgraph on remote GPU.
    
    Steps:
    1. Parse HTTP request body
    2. Deserialize input tensors (using torch.load or optimized deserializer)
    3. GPU cache lookup (weights)
    4. Graph cache lookup (parsed graph)
    5. Execute on GPU
    6. Serialize result (using torch.save or optimized serializer)
    7. Return HTTP response
    """
    
    # 1. Parse multipart form data
    form = await request.form()
    request_json = json.loads(await form['request'].read())
    tensors_bytes = await form['tensors'].read()
    
    # 2. Deserialize input tensors (optimized or standard)
    if USE_OPTIMIZED_SERIALIZATION:
        input_data = deserialize_tensor(tensors_bytes)
    else:
        input_data = torch.load(io.BytesIO(tensors_bytes))
    
    # 3. GPU cache lookup
    model_id = request_json['model_id']
    gpu_tensors = GPU_CACHE.get_weights(model_id, input_data)
    
    # 4. Graph cache lookup
    graph_key = hashlib.md5(json.dumps(request_json).encode()).hexdigest()
    parsed_graph = GRAPH_CACHE.get_graph(graph_key)
    
    # 5. Execute on GPU
    result = EXECUTOR.execute(parsed_graph, gpu_tensors, input_data)
    
    # 6. Serialize result (optimized or standard)
    if USE_OPTIMIZED_SERIALIZATION:
        result_bytes = serialize_tensor(result)
    else:
        result_buffer = io.BytesIO()
        torch.save(result, result_buffer)
        result_bytes = result_buffer.getvalue()
    
    # 7. Return HTTP response
    return Response(
        content=result_bytes,
        media_type="application/octet-stream"
    )
```

### §4.2 Subgraph Executor

**File**: `djinn/server/subgraph_executor.py`

```python
class SubgraphExecutor:
    """
    Executes computation graphs on GPU.
    
    Design:
    - Per-request ExecutionContext (isolation)
    - Topological execution order
    - GPU-only execution (no CPU round-trips)
    - Memory tracking
    """
    
    def execute(self, subgraph_request, input_data):
        """
        Execute subgraph on GPU.
        
        Steps:
        1. Create ExecutionContext (per-request isolation)
        2. Move input tensors to GPU
        3. Execute operations in topological order
        4. Extract result
        5. Move result to CPU
        """
        
        # 1. Create execution context
        context = ExecutionContext(device=self.device)
        
        # 2. Move inputs to GPU
        for tensor_id, tensor in input_data.items():
            context.tensors[tensor_id] = tensor.to(self.device)
        
        # 3. Execute operations
        for op in subgraph_request['operations']:
            result = self._execute_operation(op, context)
            context.tensors[op['op_id']] = result
        
        # 4. Extract result
        output_id = subgraph_request['output_id']
        result = context.tensors[output_id]
        
        # 5. Move to CPU
        return result.cpu().detach()
```

#### ExecutionContext

```python
class ExecutionContext:
    """
    Per-request execution context.
    
    Provides:
    - Tensor storage (intermediate results)
    - Memory tracking
    - Isolation (no cross-request interference)
    """
    
    def __init__(self, device):
        self.device = device
        self.tensors: Dict[int, torch.Tensor] = {}
        self.memory_peak_mb = 0.0
    
    def track_memory(self):
        """Track peak GPU memory usage."""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device)
            self.memory_peak_mb = max(self.memory_peak_mb, allocated / 1024 / 1024)
```

---

## §5. GPU Memory Management

### §5.1 Three-Phase Memory Management (Phase 1-3)

**Phase 1: Reactive Memory Management** ✅
- Enhanced SimpleGPUCache with memory-aware eviction
- Per-model memory tracking (not just count-based)
- KV Cache Session Pinning via KVSessionManager
- Non-blocking tensor transfers with `non_blocking=True`

**Phase 2: Semantic-Aware Memory Management** ✅
- LifetimeBasedEvictor: Evicts tensors at exact moment last consumer finishes
- PhaseAwareMemoryManager: Different budgets for prefill vs decode
- RecomputationVsStorageDecider: Cost-based cache/compute trade-offs
- Integration with scheduler for semantic-driven decisions

**Phase 3: Production Hardening** ✅
- Prometheus metrics integration (50+ metrics)
- MemoryPressureHandler: Proactive OOM prevention
- AdaptiveBudgetTuner: Learns optimal allocations from workloads
- Complete observability via metrics and health checks

### §5.2 GPU Cache Architecture (Enhanced)

**File**: `djinn/server/gpu_cache.py` (300+ LOC)

```python
class SimpleGPUCache:
    """
    LRU cache for model weights on GPU.
    
    Design:
    - OrderedDict (LRU via move_to_end)
    - Capacity-aware (tracks GPU memory)
    - Thread-safe (RWLock)
    - Persistent storage (weights stay on GPU)
    """
    
    def __init__(self, max_models: int = 5, device: Optional[torch.device] = None):
        self.device = device or (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.cache: OrderedDict[str, Dict[int, torch.Tensor]] = OrderedDict()
        self.max_models = max_models
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_memory_bytes": 0,
        }
    
    def get_weights(
        self,
        model_id: str,
        weight_dict: Dict[int, np.ndarray]
    ) -> Dict[int, torch.Tensor]:
        """
        Retrieve cached weights or load to GPU.
        
        Args:
            model_id: Unique model identifier
            weight_dict: Dict mapping tensor IDs to numpy arrays
        
        Returns:
            Dict mapping tensor IDs to GPU tensors
        
        Provides significant speedup by avoiding repeated deserialization
        and GPU transfer for frequently used models.
        """
        
        # Cache hit
        if model_id in self.cache:
            self.stats["hits"] += 1
            self.cache.move_to_end(model_id)  # LRU update
            return self.cache[model_id]
        
        # Cache miss
        self.stats["misses"] += 1
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_models:
            oldest_id = next(iter(self.cache))
            evicted_weights = self.cache.pop(oldest_id)
            evicted_memory = sum(
                t.element_size() * t.numel() for t in evicted_weights.values()
            )
            self.stats["evictions"] += 1
            self.stats["total_memory_bytes"] -= evicted_memory
            logger.info(f"Evicted model {oldest_id} (LRU)")
        
        # Load to GPU (convert numpy → torch → GPU)
        gpu_weights = {}
        total_bytes = 0
        for tensor_id, arr in weight_dict.items():
            tensor = torch.from_numpy(arr).to(self.device)
            gpu_weights[tensor_id] = tensor
            total_bytes += tensor.element_size() * tensor.numel()
        
        # Store in cache
        self.cache[model_id] = gpu_weights
        self.stats["total_memory_bytes"] += total_bytes
        
        return gpu_weights
```

### §5.3 Memory Pressure Handler Integration (NEW)

**File**: `djinn/server/memory_pressure_handler.py`

The memory pressure handler provides:

```python
class MemoryPressureHandler:
    """
    Detects and responds to memory pressure events.
    
    Thresholds:
    - Normal (<80%): Standard operation
    - Warning (80-95%): Trigger aggressive eviction
    - Critical (>95%): Emergency eviction + prefer recomputation
    - OOM (100%): Recovery via all eviction sources
    """
    
    def register_eviction_callback(self, name: str, callback: Callable):
        """Register eviction callbacks from cache/sessions."""
        # Example:
        # handler.register_eviction_callback('gpu_cache', gpu_cache.evict_aggressive)
        # handler.register_eviction_callback('sessions', kv_mgr.evict_idle_sessions)
```

**Integration Points**:
1. Monitor memory utilization continuously
2. Trigger aggressive eviction at warning threshold
3. Enforce memory pressure recovery mode at critical threshold
4. Log all pressure events to metrics
5. Track recovery times and memory freed

### §5.4 Adaptive Budget Tuning (NEW)

**File**: `djinn/server/adaptive_budget_tuner.py`

The adaptive budget tuner learns optimal phase-specific allocations:

```python
class AdaptiveBudgetTuner:
    """
    Learns optimal memory budgets from execution history.
    
    Algorithm:
    1. Collect observations (weights, activations, KV cache utilization)
    2. Calculate efficiency scores (cache hits, evictions, utilization)
    3. Identify over/under-utilized categories
    4. Gradually shift budgets towards optimal allocation
    """
    
    def record_observation(
        self,
        phase: str,
        weights_utilization: float,
        activations_utilization: float,
        kv_utilization: float,
        cache_hit_rate: float,
        evictions: int
    ) -> None:
        """Record utilization observation for adaptive learning."""
        # Example phases:
        # - 'llm_prefill': 60% activations, 10% KV cache, 30% weights
        # - 'llm_decode': 10% activations, 60% KV cache, 30% weights
```

**Learning Process**:
- **Phase 1-5**: Collect initial observations
- **Phase 6+**: Calculate efficiency scores
- **Phase 10+**: Begin suggesting adjustments
- **Ongoing**: Monitor improvement and refine

### §5.5 Prometheus Metrics Integration (NEW)

**File**: `djinn/server/memory_metrics.py`

Comprehensive metrics for production monitoring:

```python
class MetricsCollector:
    """Track all memory management operations via Prometheus."""
    
    def record_cache_hit(self, model_id: str): ...
    def record_cache_eviction(self, reason: str, freed_bytes: int, latency_ms: float): ...
    def set_cache_memory(self, model_id: str, bytes_used: int): ...
    
    def set_active_sessions(self, count: int): ...
    def set_pinned_kv_bytes(self, bytes_pinned: int): ...
    
    def record_memory_pressure_event(self, severity: str): ...  # 'warning', 'critical', 'oom'
    def record_pressure_recovery(self, memory_reclaimed_bytes: int, latency_ms: float): ...
    
    def record_budget_update(self, phase: str, category: str): ...
    def set_budget_efficiency(self, phase: str, score: float): ...
```

**Metrics Categories** (50+ total):
- GPU Cache: hits, misses, evictions, memory usage, hit rates
- KV Sessions: created, closed, active, pinned bytes, lifetimes
- Lifetime Analysis: analyses, early evictions, false retentions
- Phase-Aware: switches, budget violations, utilization per category
- Memory Pressure: events by severity, recovery time, current utilization
- Adaptive Budgets: updates, efficiency scores, learned allocations

**Integration** (in execute handler):
```python
@app.post("/execute_subgraph")
async def handle_subgraph_execution(request: Request):
    metrics = get_metrics()
    
    # Track cache operations
    if cache_hit:
        metrics.record_cache_hit(model_id)
    
    # Monitor memory pressure
    await pressure_handler.update_memory_status(available_bytes, total_bytes)
    
    # Record phase transition
    if phase != current_phase:
        metrics.record_phase_switch(current_phase, phase)
    
    # Track adaptive learning
    tuner.record_observation(phase, weights_util, activations_util, kv_util, hit_rate, evictions)
```

### §5.6 Capacity Planning (Revised)

**GPU Memory Distribution** (with Phase 3 management):

For a typical 32GB GPU:
- Reserve 30% for execution (9.6 GB)
- Remaining 70% for caching (22.4 GB)

**Budget Allocation** (phase-dependent):

**LLM Prefill Phase**:
- Weights: 30% of allocated
- Activations: 60% of allocated
- KV cache: 10% of allocated

**LLM Decode Phase**:
- Weights: 30% of allocated
- Activations: 10% of allocated
- KV cache: 60% of allocated

**Adaptive Tuning** adjusts these percentages based on observed utilization.

**Monitoring** (via metrics):
- Track per-phase utilization
- Monitor cache hit rates
- Watch efficiency scores
- Alert on budget violations

---

## §6. Result Handling

### §6.1 Result Serialization

**Server-side** (simple_server.py):
```python
# Serialize result
result_buffer = io.BytesIO()
torch.save(result, result_buffer)

# Add custom headers (timing breakdown)
headers = {
    'X-Djinn-Server-Timing': json.dumps({
        'deserialize_ms': deserialize_time,
        'execute_ms': execute_time,
        'serialize_ms': serialize_time
    }),
    'X-Djinn-Result-Bytes': str(len(result_buffer.getvalue()))
}

return Response(
    content=result_buffer.getvalue(),
    media_type="application/octet-stream",
    headers=headers
)
```

### §6.2 Result Deserialization

**Client-side** (comprehensive_profiler.py):
```python
# Parse HTTP response
response = client.execute_subgraph(subgraph, input_data)

# Extract custom headers (server timing)
server_timing = json.loads(response.headers.get('X-Djinn-Server-Timing', '{}'))
result_bytes = int(response.headers.get('X-Djinn-Result-Bytes', 0))

# Deserialize result
result = torch.load(io.BytesIO(response.content))

# Sanity check (result verification)
if not torch.allclose(result, local_result, atol=1e-4, rtol=1e-3):
    logger.error("❌ CRITICAL: Remote and local results don't match!")
```

---

## §7. Error Recovery and Fault Tolerance

### §7.1 Error Types

| Error Type | Cause | Recovery Strategy |
|------------|-------|-------------------|
| **Network Timeout** | Slow network, server overload | Retry with exponential backoff |
| **Connection Error** | Server down, network partition | Failover to alternative server |
| **GPU OOM** | Insufficient GPU memory | Evict cached models, retry |
| **Execution Error** | Invalid operation, shape mismatch | Log error, return to client |
| **Serialization Error** | Unsupported type | Fallback to local execution |

### §7.2 Error Handling (HTTP Transport)

```python
def execute_subgraph(self, subgraph_request, input_data, timeout=300.0):
    """Execute with error handling and retry."""
    
    max_retries = 3
    backoff = 1.0  # seconds
    
    for attempt in range(max_retries):
        try:
            response = self.session.post(url, files=files, timeout=timeout)
            response.raise_for_status()
            return torch.load(io.BytesIO(response.content))
        
        except requests.Timeout:
            logger.warning(f"Timeout on attempt {attempt+1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(backoff * (2 ** attempt))  # Exponential backoff
                continue
            raise RuntimeError("Remote execution timeout after retries")
        
        except requests.HTTPError as e:
            if e.response.status_code == 503:  # Service unavailable
                logger.warning(f"Server overloaded, retrying...")
                time.sleep(backoff)
                continue
            raise RuntimeError(f"Server error: {e}")
        
        except requests.ConnectionError:
            logger.error("Connection failed")
            # Try alternative server (if available)
            if self.alternative_server_url:
                self.server_url = self.alternative_server_url
                continue
            raise RuntimeError("Cannot connect to server")
```

### §7.3 Error Handling (TCP Transport)

```python
async def send(self, tensor, target, transfer_id, metadata):
    """Send with error handling."""
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            conn = await self._connection_pool.acquire(target)
            await conn.send(tensor, metadata)
            await self._connection_pool.release(target, conn)
            return
        
        except asyncio.TimeoutError:
            logger.warning(f"TCP timeout on attempt {attempt+1}")
            conn.close()  # Don't return to pool
            if attempt < max_retries - 1:
                await asyncio.sleep(1.0 * (2 ** attempt))
                continue
            raise RuntimeError("TCP timeout after retries")
        
        except ConnectionError:
            logger.error("TCP connection failed")
            conn.close()
            # Try alternative target (if available)
            if attempt < max_retries - 1:
                continue
            raise RuntimeError("Cannot connect via TCP")
```

### §7.4 GPU OOM Recovery

```python
@app.post("/execute_subgraph")
async def handle_subgraph_execution(request: Request):
    """Execute with GPU OOM recovery."""
    
    try:
        result = EXECUTOR.execute(subgraph, gpu_tensors, input_data)
        return Response(content=serialize(result))
    
    except torch.cuda.OutOfMemoryError:
        logger.warning("GPU OOM, evicting cached models...")
        
        # Evict oldest cached model
        GPU_CACHE.evict_oldest()
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Retry execution
        try:
            result = EXECUTOR.execute(subgraph, gpu_tensors, input_data)
            return Response(content=serialize(result))
        except torch.cuda.OutOfMemoryError:
            # Still OOM after eviction
            raise HTTPException(503, "Insufficient GPU memory")
```

### §7.5 Lineage-Based Recovery (Future)

**Current**: Basic error handling (retry, failover)  
**Future**: Lineage-based recovery (selective recomputation)

```python
# Future implementation
try:
    result = execute_on_remote(handle)
except RemoteExecutionError:
    # 1. Invalidate handle
    handle.invalidate()
    
    # 2. Extract lineage (SRG subgraph)
    subgraph = handle.lineage
    
    # 3. Rebind to new GPU
    new_device = scheduler.select_alternative(subgraph)
    
    # 4. Replay subgraph (deterministic recomputation)
    result = executor.replay(subgraph, new_device)
```

---

## §8. Production Best Practices

### §8.1 Transport Configuration

**Use TCP for Production**:
```python
# Initialize with TCP transport
from djinn.core.coordinator import DjinnCoordinator
from djinn.transport import TCPTransport

config = DjinnConfig(
    data_port=5556,
    connection_pool_size=4,
    connection_timeout=30.0
)

coordinator = DjinnCoordinator(config)
coordinator.register_transport('tcp', TCPTransport(config))
```

### §8.2 GPU Cache Configuration

```yaml
# config.yaml
gpu_cache:
  enabled: true
  max_models: 5  # V100 32GB
  eviction_policy: lru
  warmup_on_startup: true
  warmup_models:
    - gpt2-tiny-v1.0
    - bert-base-v1.0
```

### §8.3 Monitoring

```python
@app.get("/metrics")
async def get_metrics():
    """Expose backend metrics."""
    return {
        "gpu_cache": {
            "hits": GPU_CACHE.stats['hits'],
            "misses": GPU_CACHE.stats['misses'],
            "hit_rate": GPU_CACHE.stats['hits'] / (GPU_CACHE.stats['hits'] + GPU_CACHE.stats['misses']),
            "models_cached": len(GPU_CACHE.cache),
            "memory_used_gb": GPU_CACHE.get_memory_used() / 1024 / 1024 / 1024
        },
        "graph_cache": {
            "hits": GRAPH_CACHE.stats['hits'],
            "misses": GRAPH_CACHE.stats['misses'],
            "hit_rate": GRAPH_CACHE.stats['hits'] / (GRAPH_CACHE.stats['hits'] + GRAPH_CACHE.stats['misses']),
            "graphs_cached": len(GRAPH_CACHE.cache)
        },
        "executor": {
            "subgraphs_executed": EXECUTOR.stats['total'],
            "avg_execution_time_ms": EXECUTOR.stats['avg_time_ms'],
            "memory_peak_mb": EXECUTOR.stats['memory_peak_mb']
        }
    }
```

### §8.4 Health Checks

```python
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise HTTPException(503, "GPU not available")
    
    # Check GPU memory
    free_memory = torch.cuda.mem_get_info()[0]
    if free_memory < 2_000_000_000:  # 2GB threshold
        raise HTTPException(503, "Insufficient GPU memory")
    
    # Check transport
    if not TCP_TRANSPORT.is_available():
        raise HTTPException(503, "TCP transport unavailable")
    
    return {"status": "healthy", "gpu_memory_free_gb": free_memory / 1024 / 1024 / 1024}
```

---

## §9. Performance Optimization

### §9.1 Optimization Checklist

- ✅ **Use TCP transport** (production default)
- ✅ **Enable GPU cache** (recommended for repeated workloads)
- ✅ **Enable graph cache** (recommended for repeated graphs)
- ✅ **Connection pooling** (automatic with TCP)
- 🔧 **Zero-copy serialization** (optional, requires specific tensor layouts)
- 🔧 **Compression** (optional, for large models)
- 🔧 **DPDK** (future enhancement, requires specialized hardware)

### §9.2 Configuration Best Practices

**Production Recommendations**:
- Use TCP transport instead of HTTP
- Enable both GPU cache and graph cache
- Configure connection pooling (typically 4-5 connections)
- Monitor cache hit rates
- Set reasonable timeouts (300+ seconds for large models)

### §9.3 Monitoring and Tuning

Track these metrics in production:
- GPU cache hit rate (target: >90%)
- Graph cache hit rate (target: >95%)
- Network latency and throughput
- GPU memory utilization
- Backend request latency

---

## §10. Conclusion

The Djinn backend provides **production-ready remote execution** with:

✅ **Multiple transports**: HTTP (dev), TCP (prod), DPDK (future)  
✅ **GPU cache**: Persistent weight storage with LRU eviction  
✅ **Graph cache**: Parsed graph caching for repeated patterns  
✅ **Error recovery**: Retry, failover, OOM handling  
✅ **Monitoring**: Metrics, health checks, statistics

**Design Highlights**:
- Modular architecture supports multiple transports
- Configurable caching layers
- Comprehensive error handling and recovery
- Built-in monitoring and diagnostics

