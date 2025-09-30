# Semantic Layer

## Overview

The semantic layer implements Genie's core innovation: capturing and exploiting rich application semantics for workload-specific optimizations. This layer analyzes computation graphs to identify execution phases, memory patterns, and optimization opportunities that low-level systems miss.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Graph Analysis Pipeline                        │
│                                                                 │
│  LazyTensor Graph                                              │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Tier 1: Dynamic Dispatcher Interception               │  │
│  │  - Fine-grained operation stream                        │  │
│  │  - Runtime shapes and control flow                      │  │
│  └─────────────────────────────────────────────────────────┘  │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Tier 2: Static FX Analysis                            │  │
│  │  - Module hierarchy (nn.Module boundaries)              │  │
│  │  - Architectural blueprint (ResNet, ViT, etc.)         │  │
│  └─────────────────────────────────────────────────────────┘  │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Tier 3: Hook-Based Semantic Enhancement               │  │
│  │  - Execution phase detection (prefill/decode)          │  │
│  │  - Modality tracking (vision/text)                     │  │
│  │  - Semantic role inference                             │  │
│  └─────────────────────────────────────────────────────────┘  │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Pattern Matching & Classification                      │  │
│  │  - LLM, Vision, RecSys, Multi-Modal detection          │  │
│  └─────────────────────────────────────────────────────────┘  │
│         │                                                       │
│         ▼                                                       │
│  WorkloadProfile → ExecutionPlan                               │
└─────────────────────────────────────────────────────────────────┘
```

## Module Organization

### Analysis Modules

1. **`analyzer.py`** - Base semantic analyzer with three-tier capture
2. **`enhanced_analyzer.py`** - Enhanced analyzer with TorchDynamo patterns
3. **`fx_analyzer.py`** - FX-based structural analysis
4. **`pattern_registry.py`** - Pattern plugin registry

### Semantic Enrichment

5. **`hooks.py`** - Module-level semantic hooks
6. **`phase_detector.py`** - Execution phase detection (prefill/decode/fusion)
7. **`module_context.py`** - Module execution context tracking
8. **`graph_utils.py`** - Graph analysis utilities with NetworkX

### Optimization & Planning

9. **`optimizer.py`** - Semantic-aware optimizations
10. **`scheduler.py`** - Scheduling strategies (pipeline, parallel, priority)
11. **`placement.py`** - Device placement decisions
12. **`planner.py`** - Execution plan generation

### Data Structures

13. **`workload.py`** - Workload types, profiles, and execution plans
14. **`graph_handoff.py`** - Interface between LazyTensor and analyzer

## Core Concepts

### 1. Three-Tier Semantic Capture (HotNets'25 §3.1)

#### Tier 1: Dispatcher-Level Interception

**Purpose**: Capture exact dynamic operation stream

**Implementation** (in `analyzer.py`):
```python
class SemanticAnalyzer:
    def analyze_graph(self, graph: ComputationGraph) -> WorkloadProfile:
        # Tier 1: Analyze operations from dispatcher
        ops_metadata = analyze_operations_advanced(graph)
        
        # Extract operation histogram, entropy, graph topology
        {
            'op_histogram': Counter({'aten::matmul': 24, ...}),
            'num_nodes': 156,
            'num_edges': 243,
            'entropy': 2.34,
            'diversity': 18,  # Unique op types
            'is_dag': True
        }
```

**Example Capture**:
```python
# User code
x = torch.randn(10, 10, device="remote_accelerator:0")
y = torch.matmul(x, x)

# Tier 1 captures:
Node1: {op: "aten::randn", args: [10, 10]}
Node2: {op: "aten::matmul", args: [Node1, Node1]}
```

#### Tier 2: Static FX Analysis

**Purpose**: Identify high-level module structure

**Implementation** (in `fx_analyzer.py`):
```python
class FXAnalyzer:
    def analyze_structure(self, graph: ComputationGraph) -> StructuralInfo:
        # Get FX graph from GraphBuilder
        fx_graph = GraphBuilder.current().get_fx_graph()
        
        # Extract module hierarchy
        modules = self.extract_module_hierarchy(fx_graph)
        # {'vit.encoder.layer_0': {...}, 'vit.encoder.layer_1': {...}}
        
        # Identify architecture
        architecture = self.identify_architecture(fx_graph, modules)
        # Returns: 'vit', 'resnet', 'cnn', 'mlp', or None
        
        return StructuralInfo(
            modules=modules,
            architecture=architecture,
            depth=24,  # Number of layers
            width=768,  # Hidden dimension
            parameters=86_000_000  # 86M params
        )
```

**Architecture Detection Heuristics**:
```python
# ViT: attention + layer norm + frequent matmul
if matmul >= 2 and softmax >= 1 and layer_norm >= 1:
    return "vit"

# ResNet: many convs + relu + adds (residual)
if conv2d >= 6 and relu >= 2 and add >= 1:
    return "resnet"

# CNN: convs and activations without residuals
if conv2d >= 2 and relu >= 1:
    return "cnn"
```

#### Tier 3: Hook-Based Enhancement

**Purpose**: Recover lost high-level intent

**Implementation** (in `hooks.py`):
```python
class HookManager:
    def inject_hooks(self, model: nn.Module):
        for name, module in model.named_modules():
            # Capture context
            module.register_forward_hook(
                lambda m, i, o: self.capture_context(name, m, i, o)
            )
            
            # Phase detection
            if self.enable_phase_detection:
                phase_hook = PhaseAwareHook(module_name=name)
                module.register_forward_hook(phase_hook)
```

**Example VQA Model** (HotNets'25 §3.1):
```python
# Tier 3 captures:
{
    'module_path': 'VQA.fusion_block.attention',
    'module_type': 'MultiheadAttention',
    'execution_phase': 'multimodal_fusion',
    'semantic_role': 'cross_attention_projection',
    'input_shapes': [[1, 196, 768], [1, 128, 768]],
    'data_lineage': {
        'source_vision': 'ViT.encoder.layer_11.output',
        'source_text': 'BERT.encoder.layer_5.output'
    }
}
```

### 2. Execution Phase Detection

**File**: `genie/runtime/phase_detector.py`

**Purpose**: Runtime identification of execution phases for phase-aware scheduling.

**Supported Phases**:
```python
class ExecutionPhase(Enum):
    UNKNOWN = "unknown"
    PREFILL = "prefill"           # LLM: initial context processing
    DECODE = "decode"             # LLM: token generation
    EMBEDDING = "embedding"       # Initial embedding lookup
    VISION_BACKBONE = "vision_backbone"  # Feature extraction
    VISION_HEAD = "vision_head"   # Classification/detection
    MULTIMODAL_FUSION = "multimodal_fusion"  # Cross-modal attention
```

**Detection Algorithm**:
```python
class PhaseDetector:
    def detect_phase(self, operation: str, inputs: List, 
                     metadata: Dict) -> ExecutionPhase:
        # 1. Check for explicit hints
        if 'execution_phase' in metadata:
            return metadata['execution_phase']
        
        # 2. LLM-specific detection
        phase = self._detect_llm_phase(operation, inputs, metadata)
        if phase != UNKNOWN:
            return phase
        
        # 3. Vision-specific detection
        phase = self._detect_vision_phase(operation, inputs, metadata)
        if phase != UNKNOWN:
            return phase
        
        # 4. Multi-modal detection
        phase = self._detect_multimodal_phase(operation, inputs, metadata)
        return phase
```

**LLM Phase Detection**:
```python
def _detect_llm_phase(self, operation, inputs, metadata):
    # KV cache detection
    if self._is_kv_cache_operation(operation, metadata):
        seq_len = self._get_sequence_length(inputs)
        if seq_len > 1:
            return ExecutionPhase.PREFILL  # Multiple tokens
        else:
            return ExecutionPhase.DECODE   # Single token
    
    # Attention analysis
    if operation in ['matmul', 'softmax', 'attention']:
        seq_len = self._get_sequence_length(inputs)
        return PREFILL if seq_len > 1 else DECODE
```

**Phase Transition Tracking**:
```python
# Automatic transition detection
detector = get_phase_detector()

# First forward pass
phase = detector.detect_phase("embedding", [...])
# Returns: PREFILL

# Subsequent passes
phase = detector.detect_phase("matmul", [seq_len=1, ...])
# Returns: DECODE

# Get statistics
stats = detector.get_phase_statistics()
{
    'total_phases': 5,
    'current_phase': 'decode',
    'transitions': [
        (PhaseTransition.INIT_TO_PREFILL, timestamp),
        (PhaseTransition.PREFILL_TO_DECODE, timestamp)
    ],
    'token_position': 42
}
```

### 3. Pattern Recognition

**File**: `genie/semantic/pattern_registry.py`

**Purpose**: Identify workload patterns for optimization.

**Pattern Plugins**:
```python
# Builtin patterns (in genie/patterns/advanced_patterns.py)
- AdvancedLLMPattern       # Transformer attention patterns
- AdvancedVisionPattern    # CNN/ViT patterns
- RecSysPattern            # Embedding + MLP patterns
- MultiModalPattern        # Cross-modal fusion
- ResidualBlockPattern     # ResNet blocks
```

**Pattern Matching**:
```python
registry = PatternRegistry()

# Match patterns against graph
matches = registry.match_patterns(graph)

for match in matches:
    print(f"{match.pattern_name}: {match.confidence:.2f}")
    # AdvancedLLMPattern: 0.92
    # ResidualBlockPattern: 0.15
```

**Pattern Structure**:
```python
@dataclass
class MatchedPattern:
    pattern_name: str
    confidence: float  # 0.0 to 1.0
    subgraph: Optional[ComputationGraph]
    optimization_hints: Dict[str, Any]
    metadata: Dict[str, Any]
```

**Workload Classification**:
```python
class WorkloadClassifier:
    def classify(self, patterns: List[MatchedPattern]) -> WorkloadType:
        scores = {p.pattern_name: p.confidence for p in patterns}
        
        # Multi-modal: high confidence in both LLM and Vision
        if scores.get("llm", 0) >= 0.85 and scores.get("vision", 0) >= 0.85:
            return WorkloadType.MULTIMODAL
        
        # Single-modality based on highest confidence
        if scores.get("llm", 0) > 0.8:
            return WorkloadType.LLM
        if scores.get("vision", 0) > 0.8:
            return WorkloadType.VISION
        
        return WorkloadType.UNKNOWN
```

### 4. Workload Profiles

**File**: `genie/semantic/workload.py`

**Purpose**: Unified representation of workload characteristics.

**WorkloadProfile Structure**:
```python
@dataclass
class WorkloadProfile:
    workload_type: WorkloadType  # LLM, VISION, MULTIMODAL, RECSYS
    patterns: List[MatchedPattern]  # Matched patterns
    metadata: Dict[str, Any]  # Operation statistics
    structure: StructuralInfo  # FX analysis results
    context: Dict[str, Any]  # Hook-captured context
    
    # Optional optimization hints
    confidence: float
    phases: Dict[str, List[str]]  # Phase → operations
    compute_intensity: float
    memory_bandwidth: float
    latency_sensitivity: str
```

**Example Profile**:
```python
# For a ViT-BERT multi-modal model
profile = WorkloadProfile(
    workload_type=WorkloadType.MULTIMODAL,
    patterns=[
        MatchedPattern("vision", confidence=0.88),
        MatchedPattern("llm", confidence=0.91),
        MatchedPattern("multimodal", confidence=0.95)
    ],
    metadata={
        'op_histogram': {
            'aten::matmul': 48,
            'aten::conv2d': 16,
            'aten::softmax': 24,
            'aten::layer_norm': 36
        },
        'compute_intensity': 0.72,
        'memory_bandwidth': 0.28,
        'parallelism': 0.35
    },
    structure=StructuralInfo(
        modules={'ViT': {...}, 'BERT': {...}, 'Fusion': {...}},
        architecture='vit',
        parameters=150_000_000
    )
)
```

### 5. Graph Analysis Utilities

**File**: `genie/semantic/graph_utils.py`

**Purpose**: Efficient graph analysis using NetworkX and statistical methods.

**Key Functions**:

**Graph Conversion**:
```python
def graph_to_networkx(graph: ComputationGraph) -> nx.DiGraph:
    """Convert to NetworkX with caching."""
    # LRU cache for repeated conversions
    # Hash-based on (nodes, edges) for stability
```

**Pattern Finding**:
```python
def find_attention_pattern(G: nx.DiGraph) -> List[Dict]:
    """Find: matmul → softmax → matmul"""
    pattern_nodes = ["aten::matmul", "aten::softmax", "aten::matmul"]
    pattern_edges = [(0, 1), (1, 2)]
    return find_subgraph_patterns(G, pattern_nodes, pattern_edges)

def find_conv_activation_pattern(G: nx.DiGraph) -> List[Dict]:
    """Find: conv → relu/sigmoid/tanh/gelu"""
    for activation in ["relu", "sigmoid", "tanh", "gelu"]:
        matches = find_subgraph_patterns(
            G, 
            ["aten::conv2d", f"aten::{activation}"],
            [(0, 1)]
        )
```

**Complexity Analysis**:
```python
def analyze_graph_complexity(G: nx.DiGraph) -> Dict:
    """Analyze computational characteristics."""
    compute_ops = {"matmul", "mm", "conv2d", "linear"}
    memory_ops = {"embedding", "gather", "index_select"}
    
    compute_intensity = count(compute_ops) / total_nodes
    memory_bandwidth = count(memory_ops) / total_nodes
    
    # Parallelism from graph width
    levels = list(nx.topological_generations(G))
    max_width = max(len(level) for level in levels)
    parallelism = max_width / total_nodes
    
    return {
        'compute_intensity': 0.72,
        'memory_bandwidth': 0.08,
        'parallelism': 0.45
    }
```

**Performance**:
```python
# Caching for efficiency
@lru_cache(maxsize=128)
def convert_to_networkx(graph_hash, nodes_tuple, edges_tuple):
    # Cached conversion avoids repeated work
    
# Performance tracking
@track_performance
def analyze_operations_advanced(graph):
    # Warns if >100ms (configurable via GENIE_ANALYZER_SLOW_MS)
```

## Semantic Analysis Flow

### Complete Analysis Pipeline

```python
# 1. Create analyzer
analyzer = SemanticAnalyzer()

# 2. Build computation graph (from LazyTensors)
from genie.semantic.graph_handoff import build_graph_handoff
handoff = build_graph_handoff()

# 3. Analyze graph
profile = analyzer.analyze_graph(handoff.graph)

# 4. Generate execution plan
from genie.semantic.planner import Planner
planner = Planner()
plan = planner.generate_plan(handoff.graph, profile)

# 5. Execute plan
from genie.runtime.dpdk_backend import DPDKBackend
backend = DPDKBackend()
results = backend.execute_plan(plan)
```

### Analysis Results

**For LLM Workload**:
```python
profile = WorkloadProfile(
    workload_type=WorkloadType.LLM,
    patterns=[
        MatchedPattern("llm", confidence=0.94)
    ],
    metadata={
        'execution_phases': {
            'prefill': 12,  # operations
            'decode': 156
        },
        'kv_cache_operations': 24,
        'attention_blocks': 12
    }
)
```

**For Vision Workload**:
```python
profile = WorkloadProfile(
    workload_type=WorkloadType.VISION,
    patterns=[
        MatchedPattern("vision", confidence=0.89),
        MatchedPattern("residual_block", confidence=0.76)
    ],
    metadata={
        'conv_layers': 16,
        'residual_connections': 8,
        'pooling_operations': 5
    },
    structure=StructuralInfo(architecture='resnet', depth=50)
)
```

**For Multi-Modal Workload**:
```python
profile = WorkloadProfile(
    workload_type=WorkloadType.MULTIMODAL,
    patterns=[
        MatchedPattern("vision", confidence=0.88),
        MatchedPattern("llm", confidence=0.91),
        MatchedPattern("multimodal", confidence=0.95)
    ],
    metadata={
        'modalities': ['vision', 'text'],
        'fusion_points': ['VQA.fusion_block.cross_attn'],
        'parallel_branches': 2
    }
)
```

## Optimization Strategies (HotNets'25 §3.2)

### Semantic-Aware Optimizations

**File**: `genie/semantic/optimizer.py`

**Implementation**:
```python
class SemanticOptimizer:
    def optimize(self, graph: fx.GraphModule, 
                 profile: WorkloadProfile) -> Tuple[fx.GraphModule, OptimizationPlan]:
        # Apply workload-specific optimizations
        if profile.workload_type == WorkloadType.LLM:
            graph = self._apply_llm_optimizations(graph, plan)
        elif profile.workload_type == WorkloadType.VISION:
            graph = self._apply_vision_optimizations(graph, plan)
        elif profile.workload_type == WorkloadType.MULTIMODAL:
            graph = self._apply_multimodal_optimizations(graph, plan)
        
        return graph, plan
```

### LLM Optimizations

**1. Decode Co-Location** (HotNets'25 Example):

**Problem**: KV cache repeatedly transferred
```
Request 1: Decode on GPU 0 → Transfer KV cache (5 GB)
Request 2: Decode on GPU 1 → Transfer KV cache (5 GB)  # Wasteful!
```

**Solution**: Pin decoder + cache to same GPU
```python
def _apply_llm_optimizations(self, graph, plan):
    # Find KV cache operations
    kv_cache_nodes = self._find_kv_cache_nodes(graph)
    
    # Co-locate on same device
    for node in kv_cache_nodes:
        node.meta['placement_hint'] = 'kv_cache_device'
        node.meta['colocation_group'] = 'kv_cache'
        node.meta['priority'] = 10  # High priority
```

**2. Prefill Parallelization**:

**Opportunity**: Prefill is compute-bound and parallelizable
```python
# Create parallel branches
mid = len(attention_nodes) // 2
plan.parallel_branches.append((
    attention_nodes[:mid],   # Branch 1
    attention_nodes[mid:]    # Branch 2
))

# Mark for parallel execution
for node in branch1:
    node.meta['parallel_group'] = 'prefill_branch_1'
    node.meta['can_parallelize'] = True
```

### Vision Optimizations

**1. CNN Pipeline Scheduling**:

**Concept**: Layer-wise pipelining
```python
def _create_cnn_pipeline_stages(self, conv_nodes):
    stages = []
    stage_size = len(conv_nodes) // 3  # 3 stages
    
    for i in range(0, len(conv_nodes), stage_size):
        stages.append(conv_nodes[i:i+stage_size])
    
    return stages
    # Stage 0: conv1, conv2, conv3 → GPU 0
    # Stage 1: conv4, conv5, conv6 → GPU 1
    # Stage 2: conv7, conv8, conv9 → GPU 2
```

**2. Conv-BN-ReLU Fusion**:
```python
def _fuse_conv_bn_relu(self, graph):
    for conv_node in self._find_conv_nodes(graph):
        users = list(conv_node.users)
        if users and 'norm' in str(users[0].target):
            bn_node = users[0]
            bn_users = list(bn_node.users)
            if bn_users and 'relu' in str(bn_users[0].target):
                # Mark for fusion
                conv_node.meta['fusion_group'] = 'conv_bn_relu'
                bn_node.meta['fusion_group'] = 'conv_bn_relu'
                bn_users[0].meta['fusion_group'] = 'conv_bn_relu'
```

### Multi-Modal Optimizations

**1. Parallel Modality Processing**:

**VQA Example** (HotNets'25 §3):
```python
# Identify separate branches
vision_branch = [nodes processing image]
text_branch = [nodes processing text]

# Mark for parallel execution
for node in vision_branch:
    node.meta['modality'] = 'vision'
    node.meta['parallel_group'] = 'vision_branch'
    node.meta['placement_hint'] = 'memory_bandwidth_gpu'

for node in text_branch:
    node.meta['modality'] = 'text'
    node.meta['parallel_group'] = 'text_branch'
    node.meta['placement_hint'] = 'compute_gpu'
```

**2. JIT Fusion at Fusion Points**:
```python
# Identify fusion point
fusion_nodes = self._find_fusion_candidates(graph)

for fusion_node in fusion_nodes:
    fusion_node.meta['is_fusion_point'] = True
    fusion_node.meta['jit_transfer'] = True
    
    # Schedule predecessors for early computation
    for pred in fusion_node.all_input_nodes:
        pred.meta['feeds_fusion'] = True
        pred.meta['early_compute'] = True
```

## Scheduling and Placement

### Scheduler

**File**: `genie/semantic/scheduling.py`

**Purpose**: Create execution schedules respecting dependencies.

**Scheduling Strategies**:
```python
class SchedulingStrategy(Enum):
    SEQUENTIAL = "sequential"  # One after another
    PARALLEL = "parallel"      # Concurrent execution
    PIPELINE = "pipeline"      # Staged execution
    PRIORITY = "priority"      # Priority-based
    DYNAMIC = "dynamic"        # Runtime-adaptive
```

**Schedule Creation**:
```python
scheduler = Scheduler()
schedule = scheduler.create_schedule(graph, optimization_plan)

# Result: ExecutionSchedule
{
    'stages': [
        [group1, group2],  # Stage 0: parallel
        [group3],          # Stage 1: sequential
        [group4, group5]   # Stage 2: parallel
    ],
    'node_to_stage': {'node1': 0, 'node2': 0, 'node3': 1, ...},
    'strategy': SchedulingStrategy.PIPELINE
}
```

**Pipeline Scheduler** (for CNNs):
```python
pipeline_scheduler = PipelineScheduler(num_stages=3)
schedule = pipeline_scheduler.create_pipeline_schedule(graph)

# Creates pipeline stages automatically
# Stage 0: Early layers → GPU 0
# Stage 1: Middle layers → GPU 1
# Stage 2: Late layers → GPU 2
```

### Placement Engine

**File**: `genie/semantic/placement.py`

**Purpose**: Decide which device executes each operation.

**Device Types**:
```python
class DeviceType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    REMOTE_GPU = "remote_gpu"
    SPECIALIZED = "specialized"  # TPU, NPU
```

**Placement Decisions**:
```python
engine = PlacementEngine()
plan = engine.create_placement_plan(graph, optimization_plan)

# Example decisions
{
    'attention_layer_0': PlacementDecision(
        device_id='cuda:0',
        reasoning='LLM decode phase - co-locate with KV cache',
        priority=8
    ),
    'vision_backbone': PlacementDecision(
        device_id='remote_gpu:0',
        reasoning='Vision backbone - GPU optimized',
        priority=6
    ),
    'fusion_block': PlacementDecision(
        device_id='cuda:0',
        reasoning='Multi-modal fusion point',
        priority=7
    )
}
```

**Placement Heuristics**:
```python
# Based on execution phase
if phase == ExecutionPhase.DECODE:
    device = find_kv_cache_device()  # Co-locate with cache

# Based on memory pattern
if memory_pattern == MemoryPattern.PERSISTENT:
    device = select_local_device()  # Avoid remote

# Based on compute intensity
if compute_intensity > 8.0:
    device = select_best_gpu()  # Need powerful GPU
```

## Execution Planning

### Planner

**File**: `genie/semantic/planner.py`

**Purpose**: Generate ExecutionPlan from analyzed graph.

**ExecutionPlan Structure**:
```python
@dataclass
class ExecutionPlan:
    plan_id: str
    fragments: List[PlanFragment]  # Subgraphs to execute
    placement: Dict[str, Any]      # fragment_id → device
    transfers: List[Dict]          # Required data transfers
    feature_flags: Dict[str, bool] # overlap_io, micro_batching
```

**Plan Generation**:
```python
planner = Planner()
plan = planner.generate_plan(graph, profile)

# Phase 1: Simple whole-graph fragment
plan = ExecutionPlan(
    plan_id="plan_abc123",
    fragments=[
        PlanFragment(
            fragment_id="frag_0",
            subgraph=graph,
            inputs=[],
            outputs=["output_0"]
        )
    ],
    placement={
        "frag_0": {
            "device": "remote_accelerator:0",
            "node": "node_b",
            "workload": "llm"
        }
    },
    transfers=[],
    feature_flags={'overlap_io': False}
)
```

**Future Enhancements**:
```python
# Phase 2+: Multi-fragment plans
plan = ExecutionPlan(
    fragments=[
        PlanFragment(id="vision_backbone", ...),
        PlanFragment(id="text_encoder", ...),
        PlanFragment(id="fusion", ...)
    ],
    placement={
        "vision_backbone": {"device": "remote_gpu:0"},
        "text_encoder": {"device": "remote_gpu:1"},
        "fusion": {"device": "cuda:0"}  # Local for fusion
    },
    transfers=[
        {"from": "vision_backbone", "to": "fusion", "tensor": "vision_features"},
        {"from": "text_encoder", "to": "fusion", "tensor": "text_features"}
    ],
    feature_flags={'overlap_io': True, 'micro_batching': True}
)
```

## Enhanced Analyzer with TorchDynamo

### Enhanced Analyzer

**File**: `genie/semantic/enhanced_analyzer.py`

**Purpose**: Advanced analyzer using TorchDynamo declarative patterns.

**Key Features**:
- TorchDynamo pattern matching integration
- Phase detection with statistics
- FX graph analysis
- Custom pattern registration

**Usage**:
```python
analyzer = EnhancedSemanticAnalyzer(
    use_dynamo=True,
    enable_phase_detection=True
)

# Analyze FX graph directly
fx_graph_module = torch.fx.symbolic_trace(model)
profile = analyzer.analyze_fx_graph(fx_graph_module)

# Register custom patterns
def my_pattern(x, y):
    return torch.matmul(x, y.transpose(-1, -2))

analyzer.register_custom_pattern(
    pattern_name="custom_attention",
    pattern_fn=my_pattern,
    metadata={'role': 'attention'}
)

# Get statistics
stats = analyzer.get_pattern_statistics()
```

**Pattern Analysis Integration**:
```python
def analyze_fx_graph(self, fx_graph_module):
    # Use TorchDynamo if available
    if self.pattern_matcher:
        analysis = self.pattern_matcher.analyze_graph(fx_graph_module)
    else:
        analysis = self._fallback_pattern_analysis(fx_graph_module)
    
    # Classify workload
    workload_type = self._classify_workload_from_patterns(analysis)
    
    # Add phase detection
    phase_info = self.phase_detector.get_phase_statistics()
    
    return WorkloadProfile(
        workload_type=workload_type,
        metadata={
            'pattern_analysis': analysis,
            'phase_info': phase_info
        }
    )
```

## Module Context Tracking

### Module Context

**File**: `genie/semantic/module_context.py`

**Purpose**: Track nn.Module execution context for rich metadata.

**Context Stack**:
```python
tracker = ModuleContextTracker.get_instance()
tracker.activate(model)

# During forward pass, maintains stack:
[
    ModuleContext(
        module_path='vit.encoder.layer_0',
        module_type='TransformerBlock',
        layer_depth=2,
        parent_modules=['vit', 'vit.encoder'],
        attributes={'num_heads': 12, 'hidden_size': 768}
    ),
    ModuleContext(
        module_path='vit.encoder.layer_0.attention',
        module_type='MultiheadAttention',
        layer_depth=3,
        ...
    )
]

# Get current context
context = tracker.get_current_context()
phase = tracker.detect_execution_phase(context)
role = tracker.infer_semantic_role(operation, context)
```

**Semantic Role Inference**:
```python
def infer_semantic_role(self, operation, context):
    if context.attributes.get('is_attention'):
        if 'q_proj' in context.module_path:
            return "attention_query_projection"
        elif 'k_proj' in context.module_path:
            return "attention_key_projection"
        elif 'v_proj' in context.module_path:
            return "attention_value_projection"
    
    if 'mlp' in context.module_path:
        if context.layer_depth % 2 == 0:
            return "ffn_up_projection"
        else:
            return "ffn_down_projection"
```

## Performance Considerations

### Analysis Latency Target: <100ms

**From Requirements**:
- Graph analysis must complete in <100ms for interactive workloads
- Configurable via `GENIE_ANALYZER_SLOW_MS` environment variable

**Optimizations**:
```python
# 1. Result caching
@lru_cache(maxsize=128)
def convert_to_networkx(graph_hash, ...):
    # Cache NetworkX conversion

# 2. Graph ID for cache keys
def compute_graph_id(graph: ComputationGraph) -> str:
    # Stable SHA1 hash over nodes and edges
    # Enables cache hits for repeated graphs

# 3. Performance tracking
@track_performance
def analyze_operations_advanced(graph):
    # Automatically warns if >100ms

# 4. Early exit for simple graphs
if node_count <= 32:
    # Use lightweight patterns only
    patterns = [AdvancedLLMPattern(), ...]
```

**Performance Report**:
```python
report = analyzer.get_performance_report()
{
    'analyzer_stats': {
        'last_analysis_time': 0.045  # 45ms
    },
    'pattern_stats': {
        'AdvancedLLMPattern': {
            'avg_latency_ms': 12.3,
            'max_latency_ms': 48.7,
            'call_count': 156
        }
    }
}
```

### Memory Efficiency

**Weak References**:
```python
# Prevent memory leaks in TransferContext
context = TransferContext(
    tensor_ref=weakref.ref(tensor),  # Don't hold strong ref
    gpu_ptr=tensor.data_ptr()
)
```

**LRU Eviction**:
```python
# GPU memory registration cache
manager = GPUDevMemoryManager(cache_size=500)

# Auto-evict on size limit
while len(cache) >= 500:
    evict_lru()

# Auto-cleanup on age
manager.cleanup_expired_registrations(max_age_seconds=300)
```

## Graph Handoff Interface

**File**: `genie/semantic/graph_handoff.py`

**Purpose**: Well-defined interface between LazyTensor and Semantic layers.

**Contract**:
```python
@dataclass
class GraphHandoff:
    graph: ComputationGraph           # The computation DAG
    lazy_tensors: Dict[str, object]   # node_id → LazyTensor
    materialization_frontier: Set[str] # Nodes ready to execute
    metadata_version: str = "1.0"
    
    def validate(self) -> bool:
        # All nodes have corresponding tensors
        # No dangling edge references
```

**Usage**:
```python
# LazyTensor layer builds graph
from genie.core.graph import GraphBuilder
builder = GraphBuilder.current()

# Create handoff
handoff = build_graph_handoff()

# Validate before passing to analyzer
assert handoff.validate()

# Analyzer consumes
profile = analyzer.analyze_graph(handoff.graph)
```

## Testing

### Unit Tests

**Semantic Analyzer**:
```bash
pytest tests/test_analyzer_confidence.py -v
pytest tests/test_analyzer_smoke.py -v
```

**Pattern Recognition**:
```bash
pytest tests/test_dynamo_patterns.py -v
# Tests LLM, Vision, Multi-modal pattern detection
```

**Phase Detection**:
```bash
pytest tests/test_phase_detection.py -v
# Tests prefill/decode, vision stages, modality tracking
```

**Enhanced Metadata**:
```bash
pytest tests/test_enhanced_metadata.py -v
# Tests semantic metadata capture and enrichment
```

### Integration Tests

**End-to-End Analysis**:
```python
def test_complete_analysis_pipeline():
    # 1. Create model
    model = create_vqa_model()
    
    # 2. Capture with LazyTensors
    x_img = torch.randn(1, 3, 224, 224, device="remote_accelerator:0")
    x_text = torch.randn(1, 128, device="remote_accelerator:0")
    output = model(x_img, x_text)
    
    # 3. Build handoff
    handoff = build_graph_handoff()
    
    # 4. Analyze
    analyzer = SemanticAnalyzer()
    profile = analyzer.analyze_graph(handoff.graph)
    
    # 5. Verify
    assert profile.workload_type == WorkloadType.MULTIMODAL
    assert any('vision' in p.pattern_name for p in profile.patterns)
    assert any('llm' in p.pattern_name for p in profile.patterns)
```

## Advanced Features

### Adaptive Optimization

**File**: `genie/semantic/optimizer.py` - `AdaptiveOptimizer`

**Concept**: Learn from execution feedback
```python
optimizer = AdaptiveOptimizer()

# Initial optimization
graph, plan = optimizer.optimize(graph, profile)

# Execute and collect metrics
perf = execute_and_measure(graph)

# Adapt based on feedback
graph, plan = optimizer.optimize_with_feedback(
    graph, profile, previous_perf=perf
)

# Optimizer adjusts enabled optimizations
# based on effectiveness scores
```

### Dynamic Scheduling

**Runtime Adaptation**:
```python
scheduler = DynamicScheduler()

# Adapt to constraints
schedule = scheduler.create_adaptive_schedule(
    graph,
    runtime_constraints={
        'memory_limit': 16.0,  # GB
        'latency_target': 0.050  # 50ms
    }
)

# Automatically adjusts:
# - More stages if memory constrained
# - Fewer stages if latency constrained
```

### Custom Pattern Plugins

**External Pattern Loading**:
```python
# Via environment variable
export GENIE_PATTERN_PLUGINS="my_module:MyPattern,other:get_patterns"

# Via entry points (setup.py)
entry_points={
    'genie.patterns': [
        'custom_pattern = my_package:CustomPattern',
    ]
}

# Patterns auto-loaded on registry creation
registry = PatternRegistry()
# Includes builtin + custom patterns
```

## Configuration

### Environment Variables

```bash
# Analysis configuration
export GENIE_ANALYZER_CACHE=1           # Enable graph caching
export GENIE_ANALYZER_SLOW_MS=100       # Warning threshold
export GENIE_ANALYZER_DEBUG=1           # Debug logging

# Pattern matching
export GENIE_FAST_PATTERNS=1            # Use lightweight patterns
export GENIE_PATTERN_PLUGINS="..."     # Custom patterns

# Phase detection
export GENIE_PHASE_DETECTION=1          # Enable phase tracking
```

### Programmatic Configuration

```python
# Analyzer with custom registry
custom_registry = PatternRegistry()
custom_registry.register_pattern(MyCustomPattern())

analyzer = SemanticAnalyzer(pattern_registry=custom_registry)

# Enhanced analyzer with options
analyzer = EnhancedSemanticAnalyzer(
    use_dynamo=True,
    enable_phase_detection=True
)

# Optimizer with selective optimizations
optimizer = SemanticOptimizer(enable_all=False)
optimizer.enabled_optimizations[OptimizationType.KV_CACHE_COLOCATION] = True
optimizer.enabled_optimizations[OptimizationType.CNN_PIPELINING] = False
```

## Debugging

### Analyze Single Graph

```python
from genie.semantic.analyzer import SemanticAnalyzer
from genie.core.graph import GraphBuilder

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Create analyzer
analyzer = SemanticAnalyzer()

# Get current graph
graph = GraphBuilder.current().get_graph()

# Analyze with timing
import time
start = time.perf_counter()
profile = analyzer.analyze_graph(graph)
elapsed = time.perf_counter() - start

print(f"Analysis time: {elapsed*1000:.2f} ms")
print(f"Workload type: {profile.workload_type}")
print(f"Patterns: {[p.pattern_name for p in profile.patterns]}")
print(f"Metadata: {profile.metadata}")
```

### Inspect Pattern Matches

```python
registry = PatternRegistry()
matches = registry.match_patterns(graph)

for match in matches:
    print(f"\nPattern: {match.pattern_name}")
    print(f"Confidence: {match.confidence:.3f}")
    print(f"Hints: {match.optimization_hints}")
    if match.metadata:
        print(f"Metadata: {match.metadata}")
```

### Track Phase Transitions

```python
detector = get_phase_detector()

# Reset before analysis
detector.reset()

# Run model
model(inputs)

# Get history
history = detector.get_phase_history()
print(f"Total phases: {len(history.states)}")
print(f"Transitions: {len(history.transitions)}")

for state in history.states:
    print(f"Phase: {state.phase.value}")
    print(f"Operations: {len(state.operations)}")
    print(f"Duration: {state.start_time}")
```

## See Also

- [Pattern Recognition Deep Dive](07-pattern-recognition.md)
- [Scheduler and Optimizer Details](08-scheduler-optimizer.md)
- [Runtime Transport Layer](05-runtime-transport.md)
- [HotNets'25 Paper](../../.kiro/HotNets25.tex) - Sections 2 & 3

## References

1. PyTorch FX: https://pytorch.org/docs/stable/fx.html
2. TorchDynamo: https://pytorch.org/docs/stable/dynamo/
3. NetworkX: https://networkx.org/

---

**Last Updated**: 2025-09-30  
**Status**: Complete after refactoring  
**Maintainers**: Genie Core Team
