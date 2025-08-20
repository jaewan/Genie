# Genie Architecture Document

## Executive Summary

Genie is a semantic-driven framework-level disaggregation system for AI accelerators that operates at the PyTorch level to bridge the "semantic translation gap" in current disaggregation approaches. By transforming PyTorch's eager execution into a semantically-aware lazy execution model, Genie captures rich application context and orchestrates efficient execution across disaggregated GPU pools with minimal CPU requirements at remote nodes.

## System Architecture Overview

### Core Design Principles

1. **Framework-Level Integration**: Operate at PyTorch's "narrow waist" to access both semantic richness and generality
2. **Deferred Execution**: Transform eager operations into lazy tensors to enable global optimization
3. **Zero-Copy Data Path**: Proactive integration with DPDK for true end-to-end zero-copy transfers
4. **CPU-Minimal Remote Nodes**: Minimize CPU requirements at accelerator nodes (2 cores, 8GB RAM)
5. **Semantic-Aware Optimization**: Leverage workload understanding for intelligent resource orchestration

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Application                         │
│                    (Unchanged PyTorch Code)                  │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                      PyTorch Framework                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Genie Device Extension                    │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │   │
│  │  │  Dispatcher  │──▶│  LazyTensor │──▶│  Graph    │ │   │
│  │  │  Integration │  │   Engine     │  │  Builder  │ │   │
│  │  └──────────────┘  └──────────────┘  └───────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    Semantic Analysis Layer                   │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐   │
│  │  FX Static   │  │    Pattern    │  │     Hook       │   │
│  │   Analysis   │  │    Library    │  │    Manager     │   │
│  └──────────────┘  └──────────────┘  └────────────────┘   │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                   Optimization & Execution                   │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐   │
│  │ Optimization │  │  Execution   │  │   Resource     │   │
│  │    Engine    │  │  Scheduler   │  │    Planner     │   │
│  └──────────────┘  └──────────────┘  └────────────────┘   │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    Zero-Copy Runtime Layer                   │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐   │
│  │    DPDK      │  │   Transfer   │  │    GPUDev      │   │
│  │  Allocator   │  │   Manager    │  │   Interface    │   │
│  └──────────────┘  └──────────────┘  └────────────────┘   │
└─────────────────────────────┬───────────────────────────────┘
                              │ Network (RDMA/RoCE)
                              │
┌─────────────────────────────▼───────────────────────────────┐
│              Remote Accelerator Nodes (CPU-Minimal)          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Node 1: [2 CPU cores] [8GB RAM] [8x H100 GPUs]     │   │
│  │  ┌────────────┐  ┌────────────┐  ┌──────────────┐  │   │
│  │  │   Thin     │  │  SmartNIC/ │  │     GPU      │  │   │
│  │  │  Runtime   │  │    DPU     │  │   Cluster    │  │   │
│  │  └────────────┘  └────────────┘  └──────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. PyTorch Integration Layer

#### 1.1 Device Extension
```python
class RemoteAcceleratorDevice(torch._C.Device):
	"""Custom PyTorch device for disaggregated accelerators"""
	
	def __init__(self, device_id: int):
		self.device_type = "remote_accelerator"
		self.index = device_id
		self.dispatcher_key = torch._C.DispatchKey.PrivateUse1
```

#### 1.2 Dispatcher Integration
- **Primary Capture Mechanism**: PyTorch `__torch_function__` protocol for comprehensive interception (>95% of ops) involving `LazyTensor`
- **Factory Interception**: Lightweight hooks for tensor creation (e.g., `randn/zeros/ones/empty`) targeting `remote_accelerator`
- **Dispatcher Notes**: Torch dispatcher (`aten::`) integration remains optional for targeted cases; not required for broad coverage in Phase 1
- **Return**: `LazyTensor` proxies instead of eager execution
- **Metadata Collection**: Operation type, tensor shapes, dtypes, device placement

### 2. LazyTensor Subsystem

#### 2.1 LazyTensor Data Structure
```python
class LazyTensor:
	# Core execution tracking
	operation: str                    # e.g., 'aten::matmul'
	inputs: List[Union["LazyTensor", torch.Tensor]]
	kwargs: Dict[str, Any]
	# Inferred properties (may be meta until materialization)
	shape: Tuple[int, ...]
	dtype: torch.dtype
	device: Optional[torch.device]   # device hint only; materialization coerces as needed
```
Materialization triggers (e.g., `.cpu()`, `.item()`, tensor-to-numpy) schedule execution through the executor.

### 3. Semantic Analysis Engine

Three-tier capture:
- Tier 1 (Dispatcher capture): dynamic ops/shapes/control flow via `LazyTensor` metadata
- Tier 2 (FX): static module graph and boundaries
- Tier 3 (Hooks): module-level annotations, phase detection (prefill/decode)

Pattern Library: plugin system for LLM, Vision, RecSys, Multi-modal; supports rule- and ML-based classifiers.

### 4. Optimization & Execution

Pipeline: Graph → Pattern Match → Classify → Select Strategies → Generate Plan.

Workload strategies:
- LLM: co-locate decode with KV cache, adaptive batching, parallel prefill
- Vision: pipeline stages, fuse transfers, optimize checkpointing
- Multi-modal: parallelize modalities, just-in-time fusion, heterogeneous placement

Execution Scheduler: overlap compute and communication; handle retries and partial failures.

### 5. Zero-Copy Data Path

Memory architecture:
- Pre-registered, DMA-capable pools (DPDK huge pages)
- gpudev integration for GPUDirect RDMA when available; pinned-memory fallback

Transfer Manager: batches small transfers (target ≥1MB avg), aligns with graph plan.

### 6. Remote Execution Runtime (CPU-Minimal)

Node profile: 2 CPU cores, 8GB RAM, 200Gbps NIC/DPU, 4–8 GPUs.

Thin runtime:
- Executes plan fragments (C++ preferred, minimal Python dependency)
- Single control/telemetry channel
- Ephemeral caches only; safe restart within 2s

### 7. Execution Layer Architecture (Phased)

Phased plan to evolve from developer-friendly local execution to zero-copy remote execution:

- Phase A: Local materialization (current)
	- Capture via `__torch_function__`; build `LazyTensor` graph and metadata
	- Materialize on CPU using eager PyTorch as fallback; coerce any `device` in kwargs to CPU
	- Purpose: correctness, CI stability, and rapid iteration

- Phase B: Local-remote dev backend (single-node)
	- Add a subprocess or in-process C++ runtime using LibTorch for execution
	- Control-plane over loopback TCP; data-plane via shared/pinned memory
	- Same protocol types as remote (ExecutionPlan, TensorHandle) to ease swap-in

- Phase C: Remote runtime over TCP
	- Reuse Phase B runtime on a remote node; transport tensors via TCP pinned memory
	- Introduce scheduling of plan fragments with explicit placement

- Phase D: Zero-copy and GPUDirect RDMA
	- Integrate DPDK mempools and gpudev registration; transport via RDMA
	- Overlap communication and compute; enable batching and double-buffering

Responsibilities by language:
- Python:
	- Capture, graph building, semantic analysis (FX, hooks), plan generation
	- Orchestration, observability, user-facing API
- C++:
	- Execution runtime (LibTorch kernels), memory manager (pinned/DPDK), transport (TCP→RDMA)
	- Low-latency RPC server, batching, flow control

Development modes:
- `GENIE_EXECUTION_MODE=local|local_remote|remote` selects Phase A/B/C paths at runtime

## Data Flow

1) Application targets `remote_accelerator` → 2) Capture via `__torch_function__` builds LazyTensor graph → 3) Semantic analysis and pattern matching → 4) Plan generation (placement, comms) → 5) Execute via selected mode (local / local-remote / remote) → 6) Results materialized and returned.

## Observability & Reliability

- Metrics: per-op/node latency, bytes moved, overlap %, GPU/CPU util
- Tracing: correlate graph regions with network transfers and kernels
- Resilience: automatic retries; idempotent operations; DR hooks for control-plane state

## Communication Patterns

1. **Control Plane**: TCP/IP for metadata and coordination
2. **Data Plane**: RDMA for tensor transfers (fallback: TCP with pinned memory)
3. **Synchronization**: Custom RDMA-based barriers

## Deployment Architecture

### Development Setup

```yaml
# docker-compose.yml
version: '3.8'
services:
  client:
    image: genie:latest
    devices:
      - /dev/infiniband:/dev/infiniband
    volumes:
      - /dev/hugepages:/dev/hugepages
    environment:
      - GENIE_MODE=client
      
  remote_gpu:
    image: genie:thin-runtime
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 8
    environment:
      - GENIE_MODE=executor
```

### Production Deployment

```
┌────────────────┐     ┌─────────────────┐
│  Load Balancer │────▶│  Genie Clients  │
└────────────────┘     │   (Full CPU)    │
                      └────────┬────────┘
                              │ RDMA
                   ┌──────────▼──────────┐
                   │   Network Fabric    │
                   │  (200Gbps RDMA)     │
                   └──────────┬──────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       ↓                      ↓                      ↓
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  GPU Pool 1   │    │  GPU Pool 2   │    │  GPU Pool N   │
│ (Minimal CPU) │    │ (Minimal CPU) │    │ (Minimal CPU) │
└───────────────┘    └───────────────┘    └───────────────┘
```

## Security Architecture

### Authentication & Authorization
- mTLS for control plane communication
- RDMA protection domains for data isolation
- Token-based access to remote accelerators

### Data Protection
- Encryption at rest for cached tensors
- Optional encryption in transit (with performance tradeoff)
- Secure key management via HSM integration

## Monitoring & Observability

### Metrics Collection
```python
class MetricsCollector:
	latency_histogram: Histogram  # End-to-end operation latency
	throughput_counter: Counter   # Operations per second
	network_traffic: Gauge        # Bytes transferred
	gpu_utilization: Gauge        # Remote GPU usage
	semantic_cache_hits: Counter  # Pattern recognition cache
```

### Telemetry Pipeline

1. **Application Metrics**: PyTorch profiler integration
2. **Network Metrics**: DPDK statistics
3. **GPU Metrics**: NVML/DCGM integration
4. **System Metrics**: CPU, memory, disk usage

## Failure Handling

### Failure Modes

1. **Network Partition**: Fallback to local execution
2. **Remote GPU Failure**: Automatic failover to backup pool
3. **Memory Exhaustion**: Spill to host memory with degraded performance
4. **Pattern Mismatch**: Conservative execution with warning

### Recovery Mechanisms
```python
class FailureHandler:
	def handle_remote_failure(self, op: Operation):
		if self.can_execute_locally(op):
			return self.local_fallback(op)
		elif backup := self.find_backup_accelerator():
			return self.retry_on_backup(op, backup)
		else:
			raise AcceleratorUnavailableError()
```

## Performance Optimizations

### Caching Strategies

1. **Semantic Pattern Cache**: Reuse pattern matching results
2. **Compilation Cache**: Cache optimized execution plans
3. **Remote Tensor Cache**: Keep frequently accessed tensors remote

### Communication Optimizations

1. **Operation Batching**: Combine small transfers
2. **Pipeline Parallelism**: Overlap compute and communication
3. **Compression**: Selective tensor compression for bandwidth-limited scenarios

## Extension Points

### Plugin Architecture
```python
class GeniePlugin(ABC):
	@abstractmethod
	def register_patterns(self, registry: PatternRegistry):
		"""Add custom workload patterns"""
		
	@abstractmethod
	def register_optimizations(self, engine: OptimizationEngine):
		"""Add custom optimization strategies"""
		
	@abstractmethod
	def register_backends(self, runtime: Runtime):
		"""Add custom accelerator backends"""
```

### Custom Accelerator Support

1. Implement accelerator-specific runtime
2. Define memory management strategy
3. Create operation mapping from PyTorch ops
4. Register with device extension system

## Future Architecture Evolution

### Phase 1: Current Implementation
- Capture via `__torch_function__` and factory interception
- Local CPU materialization fallback; force CPU device during execution to avoid PrivateUse1 allocations
- Single-user, single-application focus; static pattern library
- Example suites for correctness and performance on single node

### Phase 2: Multi-Tenancy
- Workload isolation
- Fair resource scheduling
- QoS guarantees

### Phase 3: Global Scheduler Integration
- Semantic graph export
- Dynamic resource negotiation
- Autonomous scaling

### Phase 4: Compiler Co-Design
- Ahead-of-time optimization
- Custom kernel generation
- Hardware-specific tuning

## Conclusion

This architecture enables Genie to bridge the semantic translation gap by operating at the ML framework level, capturing rich application context through lazy execution, and leveraging this understanding to orchestrate efficient execution across disaggregated accelerators. The combination of semantic awareness and zero-copy data paths makes practical what was previously impossible: general-purpose, performant AI accelerator disaggregation.

## Interface alignment summary

- Planner→Scheduler: single artifact named ExecutionPlan with feature_flags; no separate "ExecutionSchedule" term.
- IDs: UUIDv4 strings for TensorID/NodeId/op_id; dtype is torch.dtype in-process and canonical string on-wire.
- Transfer path: DPDKAllocator.register()->DMAHandle {iova,lkey,rkey?,pool_id}; TransferManager returns TransferFuture with progress and callbacks.
- Optional paths: CompressionManager and checksum validation; enable via feature_flags.
- Remote runtime lifecycle: single control channel with heartbeat(interval_ms), bounded CPU p95 <2ms; telemetry plane is separate but multiplexed over control channel where feasible.