# Genie Interface Contracts

## Overview
This document defines the precise interfaces and data contracts between Genie components. All interactions must conform to these specifications to ensure correct system operation.

## Component Interaction Map
```
LazyTensor → Semantic Analyzer → Optimization Engine → Execution Scheduler
     ↓                                    ↓                    ↓
Graph Builder              Pattern Library      Transfer Manager
                                                       ↓
                                              Remote Runtime
```

## 1. LazyTensor → Semantic Analyzer

### GraphHandoff Protocol
```python
@dataclass
class GraphHandoff:
    """Data passed from LazyTensor to Semantic Analyzer"""
    graph: ComputationGraph
    lazy_tensors: Dict[NodeId, LazyTensor]  # Node → tensor mapping
    materialization_frontier: Set[NodeId]   # Current materialization points
    metadata_version: str = "1.0"
    
    def validate(self) -> bool:
        # All nodes have corresponding tensors
        for node_id in self.graph.nodes:
            if node_id not in self.lazy_tensors:
                return False
        # No dangling references
        for edge in self.graph.edges:
            if edge.source not in self.graph.nodes or edge.target not in self.graph.nodes:
                return False
        return True
```

### Trigger Conditions
- Graph size reaches threshold (1000 nodes)
- Explicit materialization requested
- Control flow boundary reached
- Memory pressure detected

## 2. Semantic Analyzer → Optimization Engine

### WorkloadProfile Contract
```python
@dataclass
class WorkloadProfile:
    """Semantic analysis result passed to optimizer"""
    workload_type: WorkloadType  # LLM, VISION, MULTIMODAL, RECSYS, UNKNOWN
    patterns: List[MatchedPattern]
    confidence: float  # 0.0 to 1.0
    
    # Semantic metadata
    phases: Dict[str, List[NodeId]]  # e.g., {'prefill': [...], 'decode': [...]} (optional Phase 2+)
    modalities: Dict[str, Subgraph]
    dependencies: List[DataDependency]
    
    # Resource hints
    compute_intensity: float  # FLOPs per byte
    memory_bandwidth: float   # GB/s required
    latency_sensitivity: str  # 'high', 'medium', 'low'
    
    # Optimization hints from patterns
    optimization_hints: Dict[str, Any]
    
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.85
```

### Pattern Match Format
```python
@dataclass
class MatchedPattern:
    pattern_name: str  # e.g., "transformer_attention"
    confidence: float
    subgraph: Subgraph  # Nodes matching this pattern
    optimization_hints: Dict[str, Any]
    metadata: Dict[str, Any]  # Pattern-specific metadata
```

## 3. Optimization Engine → Execution Scheduler

### ExecutionPlan Contract
```python
@dataclass
class ExecutionPlan:
    """Optimized execution plan (single unified term)"""
    plan_id: str  # UUIDv4
    feature_flags: Dict[str, bool]  # Negotiated capabilities
    fragments: List[PlanFragment]   # Executable subgraphs
    placement: Dict[str, DevicePlacement]  # fragment_id -> device/node
    communication_plan: CommunicationPlan
    
    # Execution metadata
    critical_path: List[NodeId]
    parallel_groups: List[List[NodeId]]
    estimated_makespan: float  # ms
    
    def validate_dependencies(self) -> bool:
        # Ensure no cycles and all dependencies satisfied
        return validate_dag(self.fragments)

@dataclass
class PlanFragment:
    fragment_id: str
    subgraph: ComputationGraph
    inputs: List[TensorHandle]
    outputs: List[TensorHandle]
```

### ScheduledOperation (Optional, if not using fragments)
```python
@dataclass
class ScheduledOperation:
    op_id: str
    op_type: str  # aten operation or composite
    inputs: List[TensorID]
    outputs: List[TensorID]
    device: RemoteAccelerator
    scheduled_time: float
    dependencies: List[str]
    estimated_duration: float
    priority: int
    metadata: SemanticMetadata
```

## 4. Transfer Manager → Remote Runtime

### RemoteExecutionRequest Contract
```python
@dataclass
class RemoteExecutionRequest:
    """Request to execute fragment on remote accelerator"""
    request_id: str
    plan_id: str
    fragment_id: str
    fragment: PlanFragment
    
    # Execution hints
    execution_hints: Dict[str, Any]
    timeout_ms: float
    checkpoint: Optional[bytes] = None
```

### DMA Handle Contract
```python
@dataclass
class DMAHandle:
    """Handle for direct memory access"""
    iova: int
    lkey: int
    rkey: Optional[int]
    pool_id: str
    
    def is_valid(self) -> bool:
        return self.iova != 0 and self.lkey != 0
```

## 5. Transfer Protocol

### TransferRequest Contract
```python
@dataclass
class TransferRequest:
    src: Union[TensorID, DMAHandle]
    dst: Union[RemoteAccelerator, DMAHandle]
    size_bytes: int
    allow_compression: bool = False
    checksum: Optional[str] = None
    priority: int = 0
    tensor_metadata: TensorMetadata
    transfer_id: str = field(default_factory=lambda: str(uuid4()))
```

### TransferFuture Contract
```python
class TransferFuture:
    def result(self, timeout: Optional[float] = None) -> TransferResult: ...
    def done(self) -> bool: ...
    def cancel(self) -> bool: ...
    def add_done_callback(self, fn: Callable) -> None: ...
    def progress(self) -> float: ...
```

## 6. Execution Service API (Runtime)
```python
class ExecutionService(ABC):
    @abstractmethod
    def run_subgraph(self, plan_fragment: PlanFragment) -> List[TensorHandle]:
        """Execute a PlanFragment and return result handles."""

    @abstractmethod
    def fetch_result(self, tensor_id: str) -> MemoryView: ...

    @abstractmethod
    def cancel(self, plan_id: str) -> None: ...

    @abstractmethod
    def health(self) -> HealthStatus: ...
```

## 7. Type Definitions

### Core Types
```python
TensorID = str  # UUIDv4 string
NodeId = str    # UUIDv4 string
OpId = str      # UUIDv4 string
```

### Data Type Representation
```python
# In-process dtype representation
dtype: torch.dtype  # Native PyTorch dtype

# On-wire dtype representation (JSON/RPC)
dtype_string: str  # Canonical string: "float16", "float32", "int64", etc.

# Conversion functions (unchanged)
```

## 8. Error Handling

### Error Context Contract
```python
@dataclass
class ErrorContext:
    """Context for error reporting"""
    plan_id: str
    op_id: Optional[str]
    attempt: int
    node: Optional[str]
    timestamp: float
    
@dataclass
class TransferError(Exception):
    """Transfer-specific error"""
    code: str  # Error code
    message: str
    op_id: Optional[str]
    context: ErrorContext
    retriable: bool = True
```

## 9. Session & Feature Negotiation

### Session Setup Contract
```python
@dataclass
class SessionCapabilities:
    """Capabilities negotiation during setup"""
    client_version: str
    supported_ops: Set[str]
    features: Dict[str, bool]  # Feature flags
    
    # Hardware capabilities
    has_gpudirect: bool
    has_rdma: bool
    dpdk_version: Optional[str]
    
    def negotiate(self, remote: 'SessionCapabilities') -> Dict[str, bool]:
        """Negotiate common feature set"""
        return {
            feature: self.features[feature] and remote.features.get(feature, False)
            for feature in self.features
        }
```

## 10. Heartbeat Protocol

### Heartbeat Contract
```python
@dataclass
class HeartbeatRequest:
    interval_ms: int
    include_metrics: bool = False
    
@dataclass
class HeartbeatResponse:
    timestamp: float
    health: str  # 'healthy', 'degraded', 'unhealthy'
    metrics: Optional[Dict[str, float]] = None
    
    # Resource status
    gpu_utilization: List[float]  # Per GPU
    memory_available: int  # Bytes
    active_operations: int
```

## Usage Guidelines

1. **Immutability**: Once created, contract objects should not be modified
2. **Validation**: Always validate at component boundaries
3. **Versioning**: Include version in serialized data
4. **Error Handling**: Use specific error types with context
5. **Async Operations**: Use futures for long-running operations
6. **Feature Flags**: Check capabilities before using optional features
7. **Monitoring**: Include operation IDs in all requests for tracing

## Testing Requirements

Each interface must have:
- Unit tests for contract validation
- Integration tests for component interaction
- Performance tests for overhead measurement
- Fault injection tests for error handling
- Compatibility tests for version migration
