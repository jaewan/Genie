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
- **Hook Point**: PyTorch's unified dispatcher (`aten::` operations)
- **Interception**: All tensor operations on `remote_accelerator` device
- **Return**: LazyTensor proxies instead of eager execution
- **Metadata Collection**: Operation type, tensor shapes, dtypes, device placement

### 2. LazyTensor Subsystem

#### 2.1 LazyTensor Data Structure
```python
class LazyTensor:
    # Core execution tracking
    operation: str                    # e.g., 'aten::matmul'
    inputs: List[Union[LazyTensor, torch.Tensor]]
    kwargs: Dict[str, Any]
    # Genie Architecture Document

    ## Executive Summary

    Genie operates at the ML framework “narrow waist” (PyTorch) to bridge the semantic translation gap and make accelerator disaggregation practical. It converts eager operations into LazyTensor graphs enriched with semantic context, plans execution with workload-aware optimizations, and executes plans over a zero-copy user-space data path to CPU-minimal remote nodes.

    ## System Architecture Overview

    ### Core Design Principles

    1. Framework-level integration for semantics and generality
    2. Deferred execution (LazyTensor) for global, cross-op optimization
    3. Zero-copy data path (DPDK + gpudev) for near-native throughput
    4. CPU-minimal remote runtime (2 cores, 8GB RAM) for cost efficiency
    5. Semantic-guided planning for workload diversity (LLM/Vision/RecSys/Multi-modal)

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
    │  │  Node: [2 CPU cores] [8GB RAM] [4–8x GPUs]           │   │
    │  │  ┌────────────┐  ┌────────────┐  ┌──────────────┐  │   │
    │  │  │   Thin     │  │  SmartNIC/ │  │     GPU      │  │   │
    │  │  │  Runtime   │  │    DPU     │  │   Cluster    │  │   │
    │  │  └────────────┘  └────────────┘  └──────────────┘  │   │
    │  └─────────────────────────────────────────────────────┘   │
    └───────────────────────────────────────────────────────────┘
    ```

    ## Component Architecture

    ### 1. PyTorch Integration Layer

    Device and dispatcher hooks intercept ops on a `remote_accelerator` device and return LazyTensors instead of executing eagerly.

    Key responsibilities:
    - Capture op type, tensor shapes/dtypes/devices
    - Maintain graph edges (data dependencies)
    - Detect materialization triggers

    ### 2. LazyTensor Subsystem

    Data structures:
    - LazyTensor: symbolic node with metadata and references to inputs
    - ComputationGraph: DAG with regions (e.g., attention blocks, fusion points)

    Materialization: explicit (.item, .cpu) and implicit (control-flow) triggers schedule execution plans.

    ### 3. Semantic Analysis Engine

    Three-tier capture:
    - Tier 1 (Dispatcher): dynamic ops/shapes/control flow
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

    ## Data Flow

    1) Application targets `remote_accelerator` → 2) Dispatcher creates LazyTensor graph → 3) Semantic analysis and pattern matching → 4) Plan generation (placement, comms) → 5) Zero-copy transfers and remote execution → 6) Results materialized and returned.

    ## Observability & Reliability

    - Metrics: per-op/node latency, bytes moved, overlap %, GPU/CPU util
    - Tracing: correlate graph regions with network transfers and kernels
    - Resilience: automatic retries; idempotent operations; DR hooks for control-plane state
  RDMA/RoCE Network
```

### Communication Patterns

1. **Control Plane**: TCP/IP for metadata and coordination
2. **Data Plane**: RDMA for tensor transfers
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
- Single-user, single-application focus
- Manual remote device specification
- Static pattern library

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