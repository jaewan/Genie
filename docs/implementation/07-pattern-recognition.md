# Pattern Recognition System

## Overview

Genie's pattern recognition system identifies workload characteristics (LLM, Vision, RecSys, Multi-Modal) from computation graphs to enable workload-specific optimizations. This system bridges the semantic translation gap by recovering high-level intent from low-level operations.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│              Pattern Recognition Pipeline                │
│                                                          │
│  ComputationGraph                                        │
│         │                                                │
│         ▼                                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Pattern Registry                                  │ │
│  │  - AdvancedLLMPattern                             │ │
│  │  - AdvancedVisionPattern                          │ │
│  │  - RecSysPattern                                  │ │
│  │  - MultiModalPattern                              │ │
│  │  - ResidualBlockPattern                           │ │
│  │  + Custom patterns (plugins)                      │ │
│  └────────────────────────────────────────────────────┘ │
│         │                                                │
│         ▼                                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Pattern Matching (per pattern)                   │ │
│  │  - Subgraph isomorphism (NetworkX)                │ │
│  │  - Heuristic scoring                              │ │
│  │  - Confidence calculation                         │ │
│  └────────────────────────────────────────────────────┘ │
│         │                                                │
│         ▼                                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Workload Classification                          │ │
│  │  - Aggregate pattern confidences                  │ │
│  │  - Multi-modal detection                          │ │
│  │  - Fallback to UNKNOWN                            │ │
│  └────────────────────────────────────────────────────┘ │
│         │                                                │
│         ▼                                                │
│  List[MatchedPattern] → WorkloadType                    │
└──────────────────────────────────────────────────────────┘
```

## Pattern Plugin System

### Base Pattern Interface

**File**: `genie/patterns/base.py`

```python
from abc import ABC, abstractmethod

class PatternPlugin(ABC):
    """Base class for pattern plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique pattern name."""
        pass
    
    @abstractmethod
    def match(self, graph: ComputationGraph) -> Optional[PatternMatch]:
        """Match pattern against graph."""
        pass
    
    def get_hints(self) -> Dict[str, Any]:
        """Optimization hints if pattern matches."""
        return {}

@dataclass
class PatternMatch:
    """Result of pattern matching."""
    confidence: float  # 0.0 to 1.0
    matched_nodes: List[str]
    metadata: Optional[Dict[str, Any]] = None
```

### Creating Custom Patterns

**Simple Pattern**:
```python
class MyAttentionPattern(PatternPlugin):
    @property
    def name(self) -> str:
        return "my_attention"
    
    def match(self, graph: ComputationGraph) -> Optional[PatternMatch]:
        # Look for matmul → softmax → matmul sequence
        G = graph_to_networkx(graph)
        matches = find_attention_pattern(G)
        
        if not matches:
            return None
        
        # Calculate confidence
        confidence = min(1.0, len(matches) / 10.0)
        
        return PatternMatch(
            confidence=confidence,
            matched_nodes=[node for match in matches for node in match.values()],
            metadata={'attention_blocks': len(matches)}
        )
    
    def get_hints(self) -> Dict:
        return {
            'can_parallelize_heads': True,
            'memory_pattern': 'reused_weights'
        }
```

**Register Pattern**:
```python
# Programmatic registration
registry = PatternRegistry()
registry.register_pattern(MyAttentionPattern())

# Via environment variable
export GENIE_PATTERN_PLUGINS="my_module:MyAttentionPattern"

# Via entry point (setup.py)
entry_points={
    'genie.patterns': [
        'my_attention = my_module:MyAttentionPattern',
    ]
}
```

## Builtin Patterns

### 1. AdvancedLLMPattern

**File**: `genie/patterns/advanced_patterns.py`

**Detection Signals**:
- Multiple attention blocks (matmul → softmax → matmul)
- Layer normalization
- Embedding operations
- MLP/FFN blocks
- Residual connections

**Scoring**:
```python
def match(self, graph: ComputationGraph) -> Optional[PatternMatch]:
    G = graph_to_networkx(graph)
    
    # Count key indicators
    attention_patterns = find_attention_pattern(G)
    mlp_patterns = find_mlp_pattern(G)
    layer_norms = count_operations(G, 'layer_norm')
    embeddings = count_operations(G, 'embedding')
    
    # Calculate confidence
    attention_score = min(1.0, len(attention_patterns) / 12.0)
    mlp_score = min(1.0, len(mlp_patterns) / 12.0)
    norm_score = min(1.0, layer_norms / 24.0)
    
    confidence = (attention_score * 0.5 + 
                  mlp_score * 0.3 + 
                  norm_score * 0.2)
    
    if confidence < 0.5:
        return None
    
    return PatternMatch(
        confidence=confidence,
        matched_nodes=...,
        metadata={
            'attention_blocks': len(attention_patterns),
            'mlp_blocks': len(mlp_patterns),
            'transformer_layers': len(attention_patterns)
        }
    )
```

**Optimization Hints**:
```python
def get_hints(self) -> Dict:
    return {
        'kv_cache_colocation': True,
        'prefill_parallelization': True,
        'decode_batching': True,
        'memory_pattern': 'persistent_cache',
        'latency_sensitive': 'decode_phase'
    }
```

### 2. AdvancedVisionPattern

**Detection Signals**:
- Convolution operations
- Pooling layers
- Batch normalization
- ReLU/GELU activations
- Residual connections (for ResNet)

**Scoring**:
```python
def match(self, graph: ComputationGraph) -> Optional[PatternMatch]:
    G = graph_to_networkx(graph)
    
    conv_patterns = find_conv_activation_pattern(G)
    residual_blocks = find_residual_block_pattern(G)
    pooling_ops = count_operations(G, 'pool')
    
    conv_score = min(1.0, len(conv_patterns) / 16.0)
    residual_score = min(1.0, len(residual_blocks) / 8.0)
    pooling_score = min(1.0, pooling_ops / 5.0)
    
    confidence = (conv_score * 0.6 + 
                  residual_score * 0.3 + 
                  pooling_score * 0.1)
    
    if confidence < 0.5:
        return None
    
    return PatternMatch(
        confidence=confidence,
        metadata={
            'conv_layers': len(conv_patterns),
            'residual_blocks': len(residual_blocks),
            'architecture_hint': 'resnet' if residual_blocks else 'cnn'
        }
    )
```

**Optimization Hints**:
```python
def get_hints(self) -> Dict:
    return {
        'pipeline_scheduling': True,
        'conv_fusion': ['conv', 'batch_norm', 'relu'],
        'memory_pattern': 'streaming',
        'parallelism': 'layer_wise'
    }
```

### 3. RecSysPattern

**Detection Signals**:
- Sparse embedding lookups
- Large embedding tables
- Dense MLP layers
- Categorical feature handling

**Scoring**:
```python
def match(self, graph: ComputationGraph) -> Optional[PatternMatch]:
    G = graph_to_networkx(graph)
    
    embeddings = find_embedding_pattern(G)
    mlp_patterns = find_mlp_pattern(G)
    
    # RecSys typically has many embeddings + dense MLP
    embedding_score = min(1.0, len(embeddings) / 20.0)
    mlp_score = min(1.0, len(mlp_patterns) / 3.0)
    
    # Must have embeddings for RecSys
    if len(embeddings) == 0:
        return None
    
    confidence = embedding_score * 0.7 + mlp_score * 0.3
    
    return PatternMatch(
        confidence=confidence,
        metadata={
            'embedding_tables': len(embeddings),
            'mlp_layers': len(mlp_patterns)
        }
    )
```

**Optimization Hints**:
```python
def get_hints(self) -> Dict:
    return {
        'embedding_caching': True,
        'hot_cold_tiering': True,
        'memory_pattern': 'sparse_access',
        'placement': 'memory_bandwidth_optimized'
    }
```

### 4. MultiModalPattern

**Detection Signals**:
- Both vision and language patterns present
- Cross-modal fusion operations (cat, add with different modalities)
- Multiple input branches that merge

**Scoring**:
```python
def match(self, graph: ComputationGraph) -> Optional[PatternMatch]:
    G = graph_to_networkx(graph)
    
    # Look for both modalities
    conv_patterns = find_conv_activation_pattern(G)
    attention_patterns = find_attention_pattern(G)
    
    has_vision = len(conv_patterns) >= 3
    has_language = len(attention_patterns) >= 2
    
    if not (has_vision and has_language):
        return None
    
    # Look for fusion points
    fusion_nodes = []
    for node_id, data in G.nodes(data=True):
        op = data.get('operation', '')
        if op in ['aten::cat', 'aten::add']:
            # Check if inputs come from different branches
            predecessors = list(G.predecessors(node_id))
            if len(predecessors) >= 2:
                fusion_nodes.append(node_id)
    
    fusion_score = min(1.0, len(fusion_nodes) / 3.0)
    
    confidence = 0.5 + fusion_score * 0.5
    
    return PatternMatch(
        confidence=confidence,
        metadata={
            'vision_operations': len(conv_patterns),
            'language_operations': len(attention_patterns),
            'fusion_points': fusion_nodes
        }
    )
```

**Optimization Hints**:
```python
def get_hints(self) -> Dict:
    return {
        'parallel_modalities': True,
        'jit_fusion_transfer': True,
        'heterogeneous_placement': {
            'vision': 'bandwidth_optimized_gpu',
            'language': 'compute_optimized_gpu'
        }
    }
```

### 5. ResidualBlockPattern

**Detection**: ResNet-style residual connections

**Pattern Structure**:
```
Input
  ├──→ conv → bn → relu → conv → bn ──┐
  │                                    │
  └────────────────────────────────────┴→ add → relu → Output
```

**Implementation**:
```python
def match(self, graph: ComputationGraph) -> Optional[PatternMatch]:
    G = graph_to_networkx(graph)
    
    residual_blocks = []
    
    # Look for add operations (potential residual)
    for node_id, data in G.nodes(data=True):
        if data.get('operation') == 'aten::add':
            predecessors = list(G.predecessors(node_id))
            if len(predecessors) == 2:
                # Check if one path is longer (conv path)
                path1_len = self._path_length_to_input(G, predecessors[0])
                path2_len = self._path_length_to_input(G, predecessors[1])
                
                if abs(path1_len - path2_len) >= 3:
                    residual_blocks.append(node_id)
    
    if not residual_blocks:
        return None
    
    confidence = min(0.95, len(residual_blocks) / 8.0)
    
    return PatternMatch(
        confidence=confidence,
        matched_nodes=residual_blocks,
        metadata={'residual_blocks': len(residual_blocks)}
    )
```

## Pattern Matching Algorithms

### Subgraph Isomorphism

**Function**: `find_subgraph_patterns` (in `graph_utils.py`)

**Algorithm**: NetworkX DiGraphMatcher
```python
def find_subgraph_patterns(G: nx.DiGraph, 
                          pattern_nodes: List[str],
                          pattern_edges: List[Tuple[int, int]]) -> List[Dict]:
    """Find all instances of a pattern in the graph."""
    
    # Create pattern graph
    pattern_G = nx.DiGraph()
    for i, op in enumerate(pattern_nodes):
        pattern_G.add_node(f"p{i}", operation=op)
    for src_idx, dst_idx in pattern_edges:
        pattern_G.add_edge(f"p{src_idx}", f"p{dst_idx}")
    
    # Match using NetworkX
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        G, pattern_G,
        node_match=lambda n1, n2: n1['operation'] == n2['operation']
    )
    
    # Find all matches (limit to 10 for performance)
    matches = []
    for mapping in matcher.subgraph_isomorphisms_iter():
        reverse_mapping = {v: k for k, v in mapping.items()}
        matches.append(reverse_mapping)
        if len(matches) >= 10:
            break
    
    return matches
```

**Example Usage**:
```python
# Find attention pattern: matmul → softmax → matmul
G = graph_to_networkx(computation_graph)

matches = find_subgraph_patterns(
    G,
    pattern_nodes=["aten::matmul", "aten::softmax", "aten::matmul"],
    pattern_edges=[(0, 1), (1, 2)]
)

# Result: [{'p0': 'node_5', 'p1': 'node_6', 'p2': 'node_7'}, ...]
```

### Specialized Pattern Finders

**Attention Pattern**:
```python
def find_attention_pattern(G: nx.DiGraph) -> List[Dict[str, str]]:
    """Find self-attention patterns."""
    # QK^T → softmax → softmax*V
    return find_subgraph_patterns(
        G,
        ["aten::matmul", "aten::softmax", "aten::matmul"],
        [(0, 1), (1, 2)]
    )
```

**Conv-Activation Pattern**:
```python
def find_conv_activation_pattern(G: nx.DiGraph) -> List[Dict]:
    """Find conv followed by activation."""
    all_matches = []
    for activation in ["relu", "sigmoid", "tanh", "gelu"]:
        matches = find_subgraph_patterns(
            G,
            ["aten::conv2d", f"aten::{activation}"],
            [(0, 1)]
        )
        all_matches.extend(matches)
    return all_matches
```

**MLP Pattern**:
```python
def find_mlp_pattern(G: nx.DiGraph) -> List[Dict]:
    """Find: linear → activation → linear"""
    all_matches = []
    for activation in ["relu", "gelu", "sigmoid"]:
        matches = find_subgraph_patterns(
            G,
            ["aten::linear", f"aten::{activation}", "aten::linear"],
            [(0, 1), (1, 2)]
        )
        all_matches.extend(matches)
    return all_matches
```

## FX-Based Pattern Matching

### FX Graph Patterns

**File**: `genie/patterns/fx_patterns.py`

**Purpose**: Pattern matching on PyTorch FX graphs (higher-level than aten ops).

**Example Patterns**:
```python
def find_matmul_chain(fx_graph: fx.Graph) -> List[List[fx.Node]]:
    """Find chains of consecutive matmul operations."""
    chains = []
    current_chain = []
    
    for node in fx_graph.nodes:
        if node.op == 'call_function':
            if 'matmul' in str(node.target):
                current_chain.append(node)
            elif current_chain:
                if len(current_chain) >= 2:
                    chains.append(current_chain)
                current_chain = []
    
    return chains

def find_attention_pattern_fx(fx_graph: fx.Graph) -> List[Tuple]:
    """Find attention patterns in FX graph."""
    # Look for Q @ K.T → softmax → result @ V pattern
    attention_blocks = []
    
    for node in fx_graph.nodes:
        if node.op == 'call_function' and 'softmax' in str(node.target):
            # Check predecessors and successors
            preds = list(node.all_input_nodes)
            succs = list(node.users)
            
            has_matmul_before = any('matmul' in str(p.target) 
                                    for p in preds if p.op == 'call_function')
            has_matmul_after = any('matmul' in str(s.target) 
                                   for s in succs if s.op == 'call_function')
            
            if has_matmul_before and has_matmul_after:
                attention_blocks.append((preds[0], node, succs[0]))
    
    return attention_blocks
```

### TorchDynamo Integration

**File**: `genie/patterns/dynamo_patterns.py`

**Purpose**: Declarative pattern matching using TorchDynamo.

**Pattern DSL**:
```python
from genie.patterns.pattern_dsl import PatternBuilder, PatternType

# Define attention pattern declaratively
attention_pattern = (
    PatternBuilder("attention")
    .with_type(PatternType.ATTENTION)
    .with_operations(["matmul", "softmax", "matmul"])
    .with_metadata(role="self_attention")
    .build()
)

# Register with matcher
matcher = DynamoPatternMatcher()
matcher.add_custom_pattern(attention_pattern)

# Match against FX graph
results = matcher.analyze_graph(fx_graph_module)
```

**Analysis Results**:
```python
{
    'total_patterns_matched': 15,
    'pattern_types': {
        'attention': 12,
        'convolution': 0,
        'fusion': 3
    },
    'execution_phases': {
        'prefill': 8,
        'decode': 4,
        'multimodal_fusion': 3
    },
    'optimization_hints': [
        'enable_kv_cache_colocation',
        'parallelize_prefill',
        'jit_fusion_transfer'
    ],
    'high_priority_ops': ['fusion_block_cross_attn'],
    'fusion_opportunities': [
        ['qkv_proj_0', 'qkv_proj_1', 'qkv_proj_2']
    ]
}
```

## Workload Classification

### Classification Algorithm

**File**: `genie/semantic/workload.py` - `WorkloadClassifier`

**Logic**:
```python
class WorkloadClassifier:
    def classify(self, patterns: List[MatchedPattern]) -> WorkloadType:
        scores = {p.pattern_name: p.confidence for p in patterns}
        
        # Multi-modal requires BOTH modalities
        llm_conf = scores.get("llm", 0.0)
        vision_conf = max(
            scores.get("vision", 0.0),
            scores.get("conv_pattern", 0.0)
        )
        
        if llm_conf >= 0.85 and vision_conf >= 0.85:
            return WorkloadType.MULTIMODAL
        
        # Single modality: highest confidence wins
        if llm_conf > 0.8:
            return WorkloadType.LLM
        if vision_conf > 0.8:
            return WorkloadType.VISION
        if scores.get("recsys", 0.0) > 0.7:
            return WorkloadType.RECSYS
        
        return WorkloadType.UNKNOWN
```

**Confidence Thresholds**:
```python
# Aggressive thresholds for clear signals
MULTIMODAL_THRESHOLD = 0.85  # Both must be high
SINGLE_MODALITY_THRESHOLD = 0.80  # High confidence
RECSYS_THRESHOLD = 0.70  # Slightly lower (sparser signals)
```

## Pattern-Driven Optimizations

### Optimization Mapping

**From Patterns to Optimizations**:
```python
# LLM Pattern → LLM Optimizations
if profile.workload_type == WorkloadType.LLM:
    optimizations = [
        OptimizationType.KV_CACHE_COLOCATION,
        OptimizationType.PREFILL_PARALLELIZATION
    ]

# Vision Pattern → Vision Optimizations
if profile.workload_type == WorkloadType.VISION:
    optimizations = [
        OptimizationType.CNN_PIPELINING,
        OptimizationType.CONV_FUSION
    ]

# Multi-Modal Pattern → Multi-Modal Optimizations
if profile.workload_type == WorkloadType.MULTIMODAL:
    optimizations = [
        OptimizationType.MULTIMODAL_FUSION,
        OptimizationType.PARALLEL_MODALITIES
    ]
```

### Optimization Hints Usage

**From Pattern Hints**:
```python
# Pattern provides hints
hints = pattern.get_hints()
{
    'kv_cache_colocation': True,
    'memory_pattern': 'persistent_cache',
    'latency_sensitive': 'decode_phase'
}

# Optimizer uses hints
if hints.get('kv_cache_colocation'):
    # Enable co-location optimization
    plan.optimizations.append(OptimizationType.KV_CACHE_COLOCATION)
    
if hints.get('memory_pattern') == 'persistent_cache':
    # Prefer local devices for persistent data
    placement_hints['prefer_local'] = True
```

## Pattern Performance

### Performance Tracking

**File**: `genie/semantic/pattern_registry.py`

**Automatic Tracking**:
```python
def match_patterns(self, graph):
    matches = []
    
    for pattern in self._patterns.values():
        start_time = time.perf_counter()
        match = pattern.match(graph)
        latency = time.perf_counter() - start_time
        
        # Track performance
        self._performance_stats[pattern.name].append(latency)
        
        # Warn if slow
        if latency > 0.05:  # 50ms default
            logger.debug(f"{pattern.name} took {latency*1000:.1f}ms")
```

**Performance Report**:
```python
report = registry.get_performance_report()
{
    'AdvancedLLMPattern': {
        'avg_latency_ms': 12.3,
        'max_latency_ms': 48.7,
        'min_latency_ms': 2.1,
        'call_count': 156,
        'total_time_ms': 1918.8
    },
    'AdvancedVisionPattern': {
        'avg_latency_ms': 8.4,
        'max_latency_ms': 32.1,
        'min_latency_ms': 1.5,
        'call_count': 98,
        'total_time_ms': 823.2
    }
}
```

### Optimization for Performance

**1. Early Exit**:
```python
# Stop after first confident match for simple graphs
if node_count <= 32:
    for pattern in lightweight_patterns:
        match = pattern.match(graph)
        if match and match.confidence >= 0.5:
            break  # Early exit
```

**2. Pattern Ordering**:
```python
# Try most likely patterns first
patterns_to_try = [
    AdvancedLLMPattern(),      # Common
    AdvancedVisionPattern(),   # Common
    MultiModalPattern(),       # Less common
    RecSysPattern()            # Specialized
]
```

**3. Caching**:
```python
# Graph ID-based caching
graph_id = compute_graph_id(graph)  # Stable hash
cached_profile = self._cache.get(graph_id)
if cached_profile:
    return cached_profile  # Skip analysis
```

## Integration with Semantic Metadata

### Metadata Enrichment

**Pattern matches enrich nodes**:
```python
# After pattern matching
for match in matches:
    for node_id in match.matched_nodes:
        node = graph.nodes[node_id]
        
        # Add pattern-derived metadata
        node.metadata['matched_pattern'] = match.pattern_name
        node.metadata['pattern_confidence'] = match.confidence
        node.metadata['optimization_hints'] = match.optimization_hints
        
        # Infer semantic role
        if match.pattern_name == 'llm':
            node.metadata['semantic_role'] = 'transformer_attention'
        elif match.pattern_name == 'vision':
            node.metadata['semantic_role'] = 'convolutional_feature_extraction'
```

### Execution Phase Assignment

**From Patterns to Phases**:
```python
# LLM pattern → Assign phases
if workload_type == WorkloadType.LLM:
    # First attention blocks = prefill
    for node in attention_nodes[:len(attention_nodes)//2]:
        node.metadata['execution_phase'] = ExecutionPhase.PREFILL
    
    # Later attention blocks = decode
    for node in attention_nodes[len(attention_nodes)//2:]:
        node.metadata['execution_phase'] = ExecutionPhase.DECODE

# Vision pattern → Assign stages
if workload_type == WorkloadType.VISION:
    for i, node in enumerate(conv_nodes):
        if i < len(conv_nodes) * 0.7:
            node.metadata['execution_phase'] = ExecutionPhase.VISION_BACKBONE
        else:
            node.metadata['execution_phase'] = ExecutionPhase.VISION_HEAD
```

## Example: VQA Model Analysis

### Input Model

**VQA (Visual Question Answering)**:
```python
class VQAModel(nn.Module):
    def __init__(self):
        self.vision = ViTEncoder()      # Vision backbone
        self.language = BERTEncoder()   # Language backbone
        self.fusion = CrossAttention()  # Multi-modal fusion
        self.classifier = nn.Linear(768, 1000)
    
    def forward(self, image, text):
        vision_features = self.vision(image)      # [1, 196, 768]
        language_features = self.language(text)  # [1, 128, 768]
        fused = self.fusion(vision_features, language_features)
        return self.classifier(fused)
```

### Pattern Matching Results

**Step 1: Pattern Detection**
```python
matches = registry.match_patterns(graph)

# AdvancedVisionPattern
PatternMatch(
    confidence=0.88,
    matched_nodes=['vision.conv_0', 'vision.conv_1', ...],
    metadata={'conv_layers': 16, 'architecture_hint': 'vit'}
)

# AdvancedLLMPattern  
PatternMatch(
    confidence=0.91,
    matched_nodes=['language.attn_0', 'language.attn_1', ...],
    metadata={'attention_blocks': 12, 'transformer_layers': 12}
)

# MultiModalPattern
PatternMatch(
    confidence=0.95,
    matched_nodes=['fusion.cross_attn', ...],
    metadata={'fusion_points': ['fusion.cross_attn'], 'modalities': 2}
)
```

**Step 2: Classification**
```python
classifier = WorkloadClassifier()
workload_type = classifier.classify(matches)
# Result: WorkloadType.MULTIMODAL (both LLM and Vision > 0.85)
```

**Step 3: Optimization Hints**
```python
hints = {}
for match in matches:
    pattern = registry._patterns[match.pattern_name]
    hints.update(pattern.get_hints())

# Aggregated hints:
{
    'parallel_modalities': True,          # From MultiModalPattern
    'jit_fusion_transfer': True,          # From MultiModalPattern
    'pipeline_scheduling': True,          # From VisionPattern
    'prefill_parallelization': True,      # From LLMPattern
    'heterogeneous_placement': {...}      # Device preferences
}
```

**Step 4: Execution Plan**
```python
plan = ExecutionPlan(
    fragments=[
        PlanFragment(id="vision", subgraph=vision_subgraph),
        PlanFragment(id="language", subgraph=language_subgraph),
        PlanFragment(id="fusion", subgraph=fusion_subgraph)
    ],
    placement={
        "vision": "remote_gpu:0",      # Bandwidth-optimized
        "language": "remote_gpu:1",    # Compute-optimized
        "fusion": "cuda:0"             # Local for low-latency fusion
    },
    transfers=[
        {"from": "vision", "to": "fusion", "tensor": "vision_features"},
        {"from": "language", "to": "fusion", "tensor": "language_features"}
    ]
)
```

## Pattern DSL

### Declarative Pattern Definition

**File**: `genie/patterns/pattern_dsl.py`

**Builder API**:
```python
from genie.patterns.pattern_dsl import PatternBuilder, PatternType

# Define pattern
pattern = (
    PatternBuilder("custom_attention")
    .with_type(PatternType.ATTENTION)
    .with_operations(["matmul", "transpose", "softmax", "matmul"])
    .with_constraints({
        'min_matmul_count': 2,
        'requires_softmax': True
    })
    .with_metadata(
        role="scaled_dot_product_attention",
        phase=ExecutionPhase.DECODE
    )
    .with_optimization_hints({
        'can_fuse_qkv': True,
        'supports_flash_attention': True
    })
    .build()
)

# Use pattern
matcher = DynamoPatternMatcher()
matcher.add_custom_pattern(pattern)
```

**Pattern Types**:
```python
class PatternType(Enum):
    ATTENTION = "attention"
    CONVOLUTION = "convolution"
    MLP = "mlp"
    FUSION = "fusion"
    RESIDUAL = "residual"
    EMBEDDING = "embedding"
    CUSTOM = "custom"
```

### Pattern Composition

**Combine Multiple Patterns**:
```python
# Transformer block = Attention + MLP
transformer_block = (
    PatternBuilder("transformer_block")
    .with_type(PatternType.CUSTOM)
    .with_subpatterns([
        attention_pattern,
        layer_norm_pattern,
        mlp_pattern,
        layer_norm_pattern
    ])
    .with_constraints({'subpattern_order': 'sequential'})
    .build()
)
```

## Testing

### Pattern Matching Tests

```python
def test_llm_pattern_detection():
    # Create graph with attention patterns
    graph = create_transformer_graph(num_layers=12)
    
    # Match patterns
    pattern = AdvancedLLMPattern()
    match = pattern.match(graph)
    
    # Verify
    assert match is not None
    assert match.confidence >= 0.8
    assert match.metadata['attention_blocks'] == 12

def test_multimodal_detection():
    # Create VQA-style graph
    graph = create_vqa_graph()
    
    # Classify
    analyzer = SemanticAnalyzer()
    profile = analyzer.analyze_graph(graph)
    
    # Should detect multi-modal
    assert profile.workload_type == WorkloadType.MULTIMODAL
    
    # Should have both modality patterns
    pattern_names = [p.pattern_name for p in profile.patterns]
    assert 'vision' in pattern_names or 'conv_pattern' in pattern_names
    assert 'llm' in pattern_names
```

### Performance Tests

```python
def test_pattern_matching_performance():
    graph = create_large_graph(num_nodes=500)
    
    registry = PatternRegistry()
    
    start = time.perf_counter()
    matches = registry.match_patterns(graph)
    elapsed = time.perf_counter() - start
    
    # Must complete in <100ms
    assert elapsed < 0.1, f"Pattern matching took {elapsed*1000:.1f}ms"
    
    # Get detailed stats
    report = registry.get_performance_report()
    for pattern_name, stats in report.items():
        print(f"{pattern_name}: {stats['avg_latency_ms']:.2f}ms avg")
```

## Debugging

### Visualize Pattern Matches

```python
def debug_pattern_matches(graph: ComputationGraph):
    registry = PatternRegistry()
    matches = registry.match_patterns(graph)
    
    print(f"Found {len(matches)} pattern matches:\n")
    
    for match in matches:
        print(f"Pattern: {match.pattern_name}")
        print(f"  Confidence: {match.confidence:.3f}")
        print(f"  Matched nodes: {len(match.matched_nodes) if match.matched_nodes else 0}")
        
        if match.optimization_hints:
            print(f"  Optimization hints:")
            for key, value in match.optimization_hints.items():
                print(f"    - {key}: {value}")
        
        if match.metadata:
            print(f"  Metadata:")
            for key, value in match.metadata.items():
                print(f"    - {key}: {value}")
        print()
```

### Trace Pattern Matching

```python
# Enable debug logging for pattern matching
logging.getLogger('genie.semantic.pattern_registry').setLevel(logging.DEBUG)
logging.getLogger('genie.patterns').setLevel(logging.DEBUG)

# Set slow threshold to see all timings
os.environ['GENIE_ANALYZER_DEBUG'] = '1'
os.environ['GENIE_ANALYZER_SLOW_MS'] = '0'

# Run analysis
profile = analyzer.analyze_graph(graph)

# Output shows:
# DEBUG:genie.patterns.advanced_patterns:AdvancedLLMPattern matching...
# DEBUG:genie.semantic.pattern_registry:Pattern AdvancedLLMPattern took 12.3ms
# DEBUG:genie.patterns.advanced_patterns:Found 12 attention blocks
```

## Best Practices

### Writing Pattern Plugins

**1. Clear Scoring Logic**:
```python
# BAD: Magic numbers
confidence = 0.5 + (len(matches) * 0.1)

# GOOD: Explicit logic with comments
def calculate_confidence(self, matches):
    # Normalize to 0-1 range based on expected count
    # Typical transformer: 12 attention blocks
    normalized = min(1.0, len(matches) / 12.0)
    
    # Boost if architecture confirmed
    if self.architecture == 'transformer':
        normalized *= 1.2
    
    return min(1.0, normalized)
```

**2. Defensive Pattern Matching**:
```python
def match(self, graph):
    try:
        G = graph_to_networkx(graph)
    except Exception as e:
        logger.error(f"Failed to convert graph: {e}")
        return None
    
    try:
        matches = self._find_patterns(G)
    except Exception as e:
        logger.error(f"Pattern matching failed: {e}")
        return None
    
    if not matches:
        return None  # Explicit no-match
    
    return PatternMatch(...)
```

**3. Meaningful Metadata**:
```python
# Provide actionable metadata
metadata = {
    'attention_blocks': 12,
    'kv_cache_size_gb': 4.5,
    'sequence_length': 2048,
    'batch_size': 16,
    'estimated_prefill_ms': 150,
    'estimated_decode_ms': 8
}
```

### Pattern Testing

```python
def test_my_pattern():
    # Create minimal graph
    graph = ComputationGraph()
    graph.add_node('n1', operation='aten::matmul')
    graph.add_node('n2', operation='aten::softmax')
    graph.add_node('n3', operation='aten::matmul')
    graph.add_edge('n1', 'n2')
    graph.add_edge('n2', 'n3')
    
    # Test pattern
    pattern = MyAttentionPattern()
    match = pattern.match(graph)
    
    assert match is not None
    assert match.confidence > 0.7
    assert 'n1' in match.matched_nodes
    assert 'n2' in match.matched_nodes
    assert 'n3' in match.matched_nodes
```

## See Also

- [Semantic Layer Overview](06-semantic-layer.md)
- [Scheduler and Optimizer](08-scheduler-optimizer.md)
- [HotNets'25 Paper](../../.kiro/HotNets25.tex) - Table 1, Section 3

---

**Last Updated**: 2025-09-30  
**Status**: Complete  
**Maintainers**: Genie Core Team
