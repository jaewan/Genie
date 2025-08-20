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
    phases: Dict[str, List[NodeId]]  # e.g., {'prefill': [...], 'decode': [...]}
    modalities: Dict[str, Subgraph]  # e.g., {'vision': ..., 'text': ...}
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
    operations: List[ScheduledOperation]
    resource_allocation: ResourceAllocation
    communication_plan: CommunicationPlan
    
    # Execution metadata
    critical_path: List[NodeId]
    parallel_groups: List[List[NodeId]]
    estimated_makespan: float  # ms
    
    def validate_dependencies(self) -> bool:
        # Ensure no cycles and all dependencies satisfied
        return validate_dag(self.operations)
```

### ScheduledOperation Format
```python
@dataclass
class ScheduledOperation:
    op_id: str  # UUIDv4, must match plan operation
    op_type: str  # aten operation or composite
    inputs: List[TensorID]
    outputs: List[TensorID]
    device: RemoteAccelerator
    
    # Scheduling metadata
    scheduled_time: float  # Relative time in ms
    dependencies: List[str]  # op_ids that must complete first
    estimated_duration: float  # ms
    priority: int  # 0-9, higher = more important
    
    # Semantic metadata preserved
    metadata: SemanticMetadata
```

## 4. Transfer Manager → Remote Runtime

### RemoteExecutionRequest Contract
```python
@dataclass
class RemoteExecutionRequest:
    """Request to execute operation on remote accelerator"""
    request_id: str  # UUIDv4
    plan_id: str  # References ExecutionPlan
    op_id: str  # Must equal ScheduledOperation.op_id
    
    # Operation details
    operation: str  # Serialized operation or IR
    input_tensors: List[TensorDescriptor]
    output_specs: List[TensorSpec]
    
    # Execution hints
    execution_hints: Dict[str, Any]
    timeout_ms: float
    checkpoint: Optional[bytes] = None

@dataclass
class TensorDescriptor:
    """Descriptor for tensor location and access"""
    tensor_id: TensorID
    location: TensorLocation
    dma_handle: Optional[DMAHandle]
    
@dataclass
class TensorLocation:
    device: str  # e.g., "cuda:0", "cpu"
    memory_address: int
    size_bytes: int
    is_ready: bool
    transfer_handle: Optional[str]
```

### DMA Handle Contract
```python
@dataclass
class DMAHandle:
    """Handle for direct memory access"""
    iova: int  # IO virtual address
    lkey: int  # Local key for RDMA
    rkey: Optional[int]  # Remote key (if registered)
    pool_id: str  # Memory pool identifier
    
    def is_valid(self) -> bool:
        return self.iova != 0 and self.lkey != 0
```

## 5. Transfer Protocol

### TransferRequest Contract
```python
@dataclass
class TransferRequest:
    """Request to transfer tensor between devices"""
    src: Union[TensorID, DMAHandle]
    dst: Union[RemoteAccelerator, DMAHandle]
    size_bytes: int
    
    # Optional features
    allow_compression: bool = False
    checksum: Optional[str] = None  # SHA256 hex
    priority: int = 0  # 0-9
    
    # Metadata
    tensor_metadata: TensorMetadata
    transfer_id: str = field(default_factory=lambda: str(uuid4()))
```

### TransferFuture Contract
```python
class TransferFuture:
    """Async handle for transfer completion"""
    
    def result(self, timeout: Optional[float] = None) -> TransferResult:
        """Block until transfer completes or timeout"""
        
    def done(self) -> bool:
        """Check if transfer is complete"""
        
    def cancel(self) -> bool:
        """Attempt to cancel transfer"""
        
    def add_done_callback(self, fn: Callable) -> None:
        """Add completion callback"""
        
    def progress(self) -> float:
        """Return progress [0.0, 1.0]"""
```

## 6. Error Handling

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

## 7. Feature Negotiation

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

## 8. Heartbeat Protocol

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

## 9. Type Definitions

### Core Types
```python
# Type aliases
TensorID = str  # UUIDv4 string
NodeId = str    # UUIDv4 string
OpId = str      # UUIDv4 string

# Enums
class WorkloadType(Enum):
    LLM = "llm"
    VISION = "vision"
    MULTIMODAL = "multimodal"
    RECSYS = "recsys"
    UNKNOWN = "unknown"

class OperationState(Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    TRANSFERRING = "transferring"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

# Device specification
@dataclass
class RemoteAccelerator:
    cluster: str  # Cluster identifier
    node: str     # Node identifier
    gpu_id: int   # GPU index on node
    
    def __hash__(self):
        return hash((self.cluster, self.node, self.gpu_id))
```

### Data Type Representation
```python
# In-process dtype representation
dtype: torch.dtype  # Native PyTorch dtype

# On-wire dtype representation (JSON/RPC)
dtype_string: str  # Canonical string: "float16", "float32", "int64", etc.

# Conversion functions
def dtype_to_string(dt: torch.dtype) -> str:
    mapping = {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.float64: "float64",
        torch.int8: "int8",
        torch.int16: "int16",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.bool: "bool",
    }
    return mapping[dt]

def string_to_dtype(s: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }
    return mapping[s]
```

## 10. Validation Requirements

### Contract Validation
All components must validate contracts at boundaries:

```python
def validate_contract(data: Any, contract_type: Type) -> bool:
    """Validate data conforms to contract"""
    if not isinstance(data, contract_type):
        return False
    
    # Type-specific validation
    if hasattr(data, 'validate'):
        return data.validate()
    
    # Field validation
    for field in dataclasses.fields(contract_type):
        if not hasattr(data, field.name):
            return False
        # Validate field types recursively
        
    return True
```

### Version Compatibility
```python
def check_version_compatibility(client_version: str, server_version: str) -> bool:
    """Check if versions are compatible"""
    client_major = int(client_version.split('.')[0])
    server_major = int(server_version.split('.')[0])
    return client_major == server_major  # Major version must match
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
