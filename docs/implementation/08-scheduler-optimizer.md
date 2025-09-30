# Scheduler and Optimizer

## Overview

The scheduler and optimizer implement Genie's workload-specific optimization strategies from HotNets'25 §3.2. By leveraging semantic information from pattern recognition and phase detection, these components apply targeted optimizations that semantically-blind systems cannot achieve.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│           Semantic-Aware Optimization Pipeline           │
│                                                          │
│  WorkloadProfile + FX Graph                             │
│         │                                                │
│         ▼                                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │  SemanticOptimizer                                 │ │
│  │  - Analyze opportunities                           │ │
│  │  - Create OptimizationPlan                         │ │
│  │  - Apply transformations                           │ │
│  └────────────────────────────────────────────────────┘ │
│         │                                                │
│         ▼                                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Scheduler                                         │ │
│  │  - Analyze dependencies                            │ │
│  │  - Create scheduling groups                        │ │
│  │  - Build execution stages                          │ │
│  └────────────────────────────────────────────────────┘ │
│         │                                                │
│         ▼                                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │  PlacementEngine                                   │ │
│  │  - Device capability matching                      │ │
│  │  - Co-location enforcement                         │ │
│  │  - Load balancing                                  │ │
│  └────────────────────────────────────────────────────┘ │
│         │                                                │
│         ▼                                                │
│  Optimized Graph + Schedule + Placement Plan            │
└──────────────────────────────────────────────────────────┘
```

## Semantic Optimizer

### Overview

**File**: `genie/semantic/optimizer.py`

**Purpose**: Apply workload-specific optimizations based on semantic analysis.

**Key Classes**:
- `OptimizationType`: Enum of available optimizations
- `OptimizationPlan`: Plan for applying optimizations
- `SemanticOptimizer`: Main optimizer class
- `AdaptiveOptimizer`: Learning-based adaptive optimizer

### Optimization Types

```python
class OptimizationType(Enum):
    # LLM optimizations
    KV_CACHE_COLOCATION = "kv_cache_colocation"
    PREFILL_PARALLELIZATION = "prefill_parallelization"
    
    # Vision optimizations
    CNN_PIPELINING = "cnn_pipelining"
    
    # Multi-modal optimizations
    MULTIMODAL_FUSION = "multimodal_fusion"
    
    # General optimizations
    RECOMPUTATION = "recomputation"
    MEMORY_OPTIMIZATION = "memory_optimization"
```

### Optimization Plan

```python
@dataclass
class OptimizationPlan:
    optimizations: List[OptimizationType]
    node_placement: Dict[str, str]              # node → device
    colocation_groups: Dict[str, List[str]]     # group → nodes
    pipeline_stages: List[List[str]]            # stages of nodes
    parallel_branches: List[Tuple[List, List]]  # branch pairs
    recompute_nodes: Set[str]                   # Nodes to recompute
    fusion_points: List[str]                    # Fusion nodes
    metadata: Dict[str, Any]
```

### Main Optimization Flow

```python
optimizer = SemanticOptimizer(enable_all=True)

# Optimize based on workload profile
graph, plan = optimizer.optimize(fx_graph, profile)

# Result: Annotated graph + execution plan
# Nodes have metadata: placement_hint, parallel_group, priority, etc.
```

## LLM Optimizations (HotNets'25 §3.2)

### 1. KV Cache Co-Location

**Problem** (from paper):
```
Without co-location:
  Request 1: Decode on GPU 0 → Transfer 5 GB KV cache
  Request 2: Decode on GPU 1 → Transfer 5 GB KV cache again!
  
Total overhead: 10 GB transferred
```

**Solution**:
```python
def _apply_llm_optimizations(self, graph, plan):
    # Find KV cache operations
    kv_cache_nodes = self._find_kv_cache_nodes(graph)
    
    if kv_cache_nodes:
        plan.optimizations.append(OptimizationType.KV_CACHE_COLOCATION)
        
        # Group for co-location
        plan.colocation_groups["kv_cache"] = [n.name for n in kv_cache_nodes]
        
        # Mark nodes with placement hints
        for node in kv_cache_nodes:
            node.meta['placement_hint'] = 'kv_cache_device'
            node.meta['colocation_group'] = 'kv_cache'
            node.meta['priority'] = 10  # Highest priority
```

**KV Cache Detection**:
```python
def _find_kv_cache_nodes(self, graph):
    kv_nodes = []
    
    for node in graph.graph.nodes:
        # Check semantic metadata
        if 'semantic' in node.meta:
            if node.meta['semantic'].kv_cache_related:
                kv_nodes.append(node)
        
        # Check operation patterns
        op_name = str(node.target).lower()
        if any(kv in op_name for kv in ['cache', 'past_key', 'past_value']):
            kv_nodes.append(node)
        
        # Check for cache concatenation (cat with zeros)
        if 'cat' in op_name:
            for inp in node.all_input_nodes:
                if 'zeros' in str(inp.target):
                    kv_nodes.append(node)
                    break
    
    return kv_nodes
```

**Result**:
```
With co-location:
  All decode requests on same GPU with KV cache
  Total overhead: 0 GB transferred (cache stays local)
```

### 2. Prefill Parallelization

**Concept**: Prefill is compute-bound and parallelizable

**Implementation**:
```python
if OptimizationType.PREFILL_PARALLELIZATION in plan.optimizations:
    # Find attention operations in prefill phase
    attention_nodes = [n for n in self._find_attention_nodes(graph)
                       if self._is_prefill_phase(n)]
    
    # Split into parallel branches
    mid = len(attention_nodes) // 2
    plan.parallel_branches.append((
        [n.name for n in attention_nodes[:mid]],   # Branch 1
        [n.name for n in attention_nodes[mid:]]    # Branch 2
    ))
    
    # Mark for parallel execution
    for node in attention_nodes[:mid]:
        node.meta['parallel_group'] = 'prefill_branch_1'
        node.meta['can_parallelize'] = True
    
    for node in attention_nodes[mid:]:
        node.meta['parallel_group'] = 'prefill_branch_2'
        node.meta['can_parallelize'] = True
```

**Phase Detection**:
```python
def _is_prefill_phase(self, node):
    if 'semantic' in node.meta:
        metadata = node.meta['semantic']
        if hasattr(metadata, 'execution_phase'):
            return metadata.execution_phase == ExecutionPhase.PREFILL
    return False
```

### 3. Attention Block Optimization

**QKV Projection Fusion**:
```python
def _optimize_attention_blocks(self, graph):
    """Fuse Q, K, V projections for efficiency."""
    for node in graph.graph.nodes:
        if node.op == 'call_function' and 'linear' in str(node.target):
            # Check semantic role
            if 'semantic' in node.meta:
                role = node.meta['semantic'].semantic_role
                if role in ['query_proj', 'key_proj', 'value_proj']:
                    node.meta['can_fuse'] = True
                    node.meta['fusion_group'] = 'qkv_projection'
```

## Vision Optimizations

### 1. CNN Pipeline Scheduling

**Concept**: Layer-wise pipelining across GPUs

**Implementation**:
```python
def _apply_vision_optimizations(self, graph, plan):
    if OptimizationType.CNN_PIPELINING in plan.optimizations:
        # Create pipeline stages
        conv_nodes = self._find_conv_nodes(graph)
        stages = self._create_cnn_pipeline_stages(conv_nodes)
        
        plan.pipeline_stages = stages
        
        # Annotate nodes with stage information
        for stage_idx, stage in enumerate(stages):
            for node in graph.graph.nodes:
                if node.name in stage:
                    node.meta['pipeline_stage'] = stage_idx
                    node.meta['pipeline_enabled'] = True
                    # Earlier stages get higher priority
                    node.meta['priority'] = len(stages) - stage_idx

def _create_cnn_pipeline_stages(self, conv_nodes):
    """Divide CNN layers into pipeline stages."""
    num_stages = 3  # Default: 3-stage pipeline
    stage_size = max(1, len(conv_nodes) // num_stages)
    
    stages = []
    for i in range(0, len(conv_nodes), stage_size):
        stage = [node.name for node in conv_nodes[i:i+stage_size]]
        if stage:
            stages.append(stage)
    
    return stages
    # Example result:
    # Stage 0: [conv1, conv2, conv3, conv4]  → GPU 0
    # Stage 1: [conv5, conv6, conv7, conv8]  → GPU 1
    # Stage 2: [conv9, conv10, fc]           → GPU 2
```

**Pipeline Execution** (conceptual):
```
Time:  t0    t1    t2    t3    t4
GPU 0: [B1]  [B2]  [B3]  [B4]  [B5]
GPU 1:  ---  [B1]  [B2]  [B3]  [B4]
GPU 2:  ---   ---  [B1]  [B2]  [B3]

Throughput: 3 batches / 3 time units = 1 batch/time
(vs sequential: 3 batches / 9 time units = 0.33 batch/time)
```

### 2. Conv-BN-ReLU Fusion

**Pattern**: `conv2d → batch_norm → relu`

**Optimization**: Fuse into single kernel
```python
def _fuse_conv_bn_relu(self, graph):
    conv_nodes = self._find_conv_nodes(graph)
    
    for conv_node in conv_nodes:
        users = list(conv_node.users)
        if users and 'norm' in str(users[0].target):
            bn_node = users[0]
            bn_users = list(bn_node.users)
            
            if bn_users and 'relu' in str(bn_users[0].target):
                relu_node = bn_users[0]
                
                # Mark all three for fusion
                fusion_id = f"conv_bn_relu_{conv_node.name}"
                conv_node.meta['fusion_group'] = fusion_id
                bn_node.meta['fusion_group'] = fusion_id
                relu_node.meta['fusion_group'] = fusion_id
                
                # Mark as fused for executor
                conv_node.meta['fused_ops'] = ['conv', 'bn', 'relu']
```

### 3. Pooling Optimization

```python
def _optimize_pooling(self, graph):
    for node in graph.graph.nodes:
        if 'pool' in str(node.target).lower():
            # Adaptive pooling can be optimized
            if 'adaptive' in str(node.target):
                node.meta['can_optimize'] = True
                node.meta['optimization_type'] = 'adaptive_pooling'
            
            # Global pooling marks end of backbone
            if 'global' in str(node.target) or 'adaptive' in str(node.target):
                node.meta['execution_phase'] = ExecutionPhase.VISION_HEAD
```

## Multi-Modal Optimizations

### 1. Parallel Modality Processing

**VQA Example** (HotNets'25 §3):

**Concept**: Process vision and text branches in parallel
```
Sequential:
  Vision (200ms) → Text (150ms) → Fusion (50ms) = 400ms total

Parallel:
  Vision (200ms) ┐
                 ├─→ Fusion (50ms) = 250ms total
  Text (150ms)  ┘
```

**Implementation**:
```python
def _apply_multimodal_optimizations(self, graph, plan):
    # Identify separate modality branches
    vision_branch, text_branch = self._identify_modality_branches(graph)
    
    if vision_branch and text_branch:
        # Mark for parallel execution
        for node in vision_branch:
            node.meta['modality'] = 'vision'
            node.meta['parallel_group'] = 'vision_branch'
            node.meta['can_parallelize'] = True
        
        for node in text_branch:
            node.meta['modality'] = 'text'
            node.meta['parallel_group'] = 'text_branch'
            node.meta['can_parallelize'] = True
        
        logger.info(f"Parallelized {len(vision_branch)} vision ops "
                    f"and {len(text_branch)} text ops")

def _identify_modality_branches(self, graph):
    """Identify vision and text branches."""
    vision_nodes = []
    text_nodes = []
    
    for node in graph.graph.nodes:
        # Check semantic metadata
        if 'semantic' in node.meta:
            modality = node.meta['semantic'].data_lineage.modality
            if modality == 'vision':
                vision_nodes.append(node)
            elif modality == 'text':
                text_nodes.append(node)
        
        # Heuristic based on operation
        op = str(node.target).lower()
        if 'conv' in op or 'pool' in op:
            vision_nodes.append(node)
        elif 'embedding' in op or 'lstm' in op:
            text_nodes.append(node)
    
    return vision_nodes, text_nodes
```

### 2. JIT Fusion Transfer

**Concept**: Transfer fusion inputs just-in-time

**Implementation**:
```python
if OptimizationType.MULTIMODAL_FUSION in plan.optimizations:
    # Find fusion points
    fusion_nodes = self._find_fusion_candidates(graph)
    
    for fusion_node in fusion_nodes:
        # Mark fusion point
        fusion_node.meta['is_fusion_point'] = True
        fusion_node.meta['jit_transfer'] = True
        fusion_node.meta['priority'] = 8
        
        # Mark predecessors for early scheduling
        for pred in fusion_node.all_input_nodes:
            pred.meta['feeds_fusion'] = True
            pred.meta['early_compute'] = True
            pred.meta['transfer_priority'] = 7

def _find_fusion_candidates(self, graph):
    """Find multi-modal fusion points."""
    fusion_nodes = []
    
    for node in graph.graph.nodes:
        # Check for cross-modal operations
        if 'semantic' in node.meta:
            if node.meta['semantic'].execution_phase == ExecutionPhase.MULTIMODAL_FUSION:
                fusion_nodes.append(node)
        
        # Check for cat/add with different modalities
        op = str(node.target).lower()
        if op in ['cat', 'concat', 'add']:
            input_modalities = {inp.meta.get('modality') 
                               for inp in node.all_input_nodes}
            if len(input_modalities) > 1:
                fusion_nodes.append(node)
    
    return fusion_nodes
```

## General Optimizations

### 1. Dynamic Recomputation

**Concept** (HotNets'25 §3.2): Recompute vs. transfer trade-off

**Decision Logic**:
```python
def _should_recompute(self, node, network_latency_ms):
    """Decide whether to recompute or transfer."""
    recompute_cost_ms = self._estimate_recompute_cost(node)
    transfer_cost_ms = self._estimate_transfer_cost(node, network_latency_ms)
    
    # Recompute if cheaper than transfer
    return recompute_cost_ms < transfer_cost_ms

def _estimate_recompute_cost(self, node):
    """Estimate recomputation cost."""
    if 'semantic' in node.meta:
        # Use semantic metadata if available
        return node.meta['semantic'].compute_intensity * 10.0
    
    # Heuristic based on operation
    op = str(node.target).lower()
    if op in ['relu', 'dropout', 'reshape']:
        return 0.1  # Very cheap
    elif op in ['norm', 'softmax']:
        return 5.0  # Medium cost
    else:
        return 50.0  # Expensive (matmul, conv)

def _estimate_transfer_cost(self, node, network_latency_ms):
    """Estimate transfer cost."""
    # Get tensor size from metadata
    if 'tensor_size_bytes' in node.meta:
        size_bytes = node.meta['tensor_size_bytes']
    else:
        # Estimate from shape
        shape = node.meta.get('tensor_shape', [1])
        size_bytes = 4 * product(shape)  # Assume float32
    
    # Transfer time = latency + size/bandwidth
    bandwidth_gbps = 100.0  # 100 Gbps network
    transfer_time_ms = network_latency_ms + (size_bytes * 8) / (bandwidth_gbps * 1e6)
    
    return transfer_time_ms
```

**Application**:
```python
# During optimization
for node in graph.graph.nodes:
    if self._should_recompute(node, network_latency=5.0):
        node.meta['can_recompute'] = True
        node.meta['prefer_recompute'] = True
        plan.recompute_nodes.add(node.name)
```

### 2. Memory Optimization

**Identify Recomputable Nodes**:
```python
def _identify_recomputable_nodes(self, graph):
    """Find nodes that can be recomputed cheaply."""
    recomputable = []
    
    for node in graph.graph.nodes:
        op = str(node.target).lower()
        
        # Cheap operations
        if any(cheap in op for cheap in ['relu', 'dropout', 'reshape', 
                                         'view', 'transpose']):
            recomputable.append(node)
    
    return recomputable
```

**Annotate with Cost**:
```python
for node in recomputable:
    node.meta['can_recompute'] = True
    node.meta['recompute_cost'] = self._estimate_recompute_cost(node)
    
    # Add to plan if very cheap
    if node.meta['recompute_cost'] < 0.5:
        plan.recompute_nodes.add(node.name)
```

### 3. Dead Code Elimination

```python
def _eliminate_dead_code(self, graph):
    """Remove operations with no users."""
    # Find output nodes
    output_nodes = set()
    for node in graph.graph.nodes:
        if node.op == 'output':
            def collect_deps(n):
                if isinstance(n, fx.Node):
                    output_nodes.add(n)
                    for inp in n.all_input_nodes:
                        collect_deps(inp)
            
            for arg in node.args:
                collect_deps(arg)
    
    # Mark dead nodes
    dead_nodes = []
    for node in graph.graph.nodes:
        if node.op in ['call_function', 'call_method', 'call_module']:
            if not node.users and node not in output_nodes:
                dead_nodes.append(node)
                node.meta['dead_code'] = True
    
    if dead_nodes:
        logger.info(f"Identified {len(dead_nodes)} dead code nodes")
```

### 4. Common Subexpression Elimination

```python
def _eliminate_common_subexpressions(self, graph):
    """Eliminate duplicate computations."""
    expr_map = {}  # signature → node
    
    for node in graph.graph.nodes:
        if node.op == 'call_function':
            # Create signature
            sig = (node.target, tuple(str(arg) for arg in node.args))
            
            if sig in expr_map:
                # Found duplicate
                original = expr_map[sig]
                node.meta['duplicate_of'] = original.name
                node.meta['can_eliminate'] = True
            else:
                expr_map[sig] = node
```

## Scheduler

### Overview

**File**: `genie/semantic/scheduling.py`

**Purpose**: Create execution schedules that respect dependencies while maximizing parallelism.

**Key Classes**:
- `SchedulingStrategy`: Execution strategies (sequential, parallel, pipeline, priority, dynamic)
- `SchedulingGroup`: Group of nodes to schedule together
- `ExecutionSchedule`: Complete execution schedule
- `Scheduler`: Base scheduler
- `PipelineScheduler`: Specialized for CNNs
- `DynamicScheduler`: Runtime-adaptive scheduler

### Scheduling Strategies

```python
class SchedulingStrategy(Enum):
    SEQUENTIAL = "sequential"  # Execute one after another
    PARALLEL = "parallel"      # Execute concurrently
    PIPELINE = "pipeline"      # Staged pipeline execution
    PRIORITY = "priority"      # Priority-based scheduling
    DYNAMIC = "dynamic"        # Adapt at runtime
```

### Schedule Creation

**Main Algorithm**:
```python
def create_schedule(self, graph, optimization_plan):
    # 1. Analyze dependencies
    dependencies = self._analyze_dependencies(graph)
    # {'node2': {'node1'}, 'node3': {'node1', 'node2'}, ...}
    
    # 2. Identify scheduling groups
    groups = self._identify_scheduling_groups(graph, optimization_plan)
    
    # 3. Create execution stages
    stages = self._create_execution_stages(groups, dependencies)
    
    # 4. Build schedule
    schedule = ExecutionSchedule(
        stages=stages,
        total_stages=len(stages),
        strategy=self._determine_strategy(groups)
    )
    
    return schedule
```

**Dependency Analysis**:
```python
def _analyze_dependencies(self, graph):
    dependencies = defaultdict(set)
    
    for node in graph.graph.nodes:
        if node.op in ['call_function', 'call_method', 'call_module']:
            # Add all input nodes as dependencies
            for inp in node.all_input_nodes:
                dependencies[node.name].add(inp.name)
    
    return dependencies
```

### Scheduling Groups

**From Optimization Plan**:
```python
def _identify_scheduling_groups(self, graph, optimization_plan):
    groups = []
    processed_nodes = set()
    
    if optimization_plan:
        # 1. Co-location groups
        for group_id, node_names in optimization_plan.colocation_groups.items():
            group = SchedulingGroup(
                group_id=f"coloc_{group_id}",
                nodes=node_names,
                strategy=SchedulingStrategy.SEQUENTIAL,
                priority=10  # High priority
            )
            groups.append(group)
            processed_nodes.update(node_names)
        
        # 2. Pipeline stages
        for stage_idx, stage_nodes in enumerate(optimization_plan.pipeline_stages):
            group = SchedulingGroup(
                group_id=f"pipeline_stage_{stage_idx}",
                nodes=stage_nodes,
                strategy=SchedulingStrategy.PIPELINE,
                priority=5
            )
            groups.append(group)
            processed_nodes.update(stage_nodes)
        
        # 3. Parallel branches
        for branch_idx, (branch1, branch2) in enumerate(optimization_plan.parallel_branches):
            # Create two groups that can run in parallel
            groups.append(SchedulingGroup(
                group_id=f"parallel_branch_{branch_idx}_a",
                nodes=branch1,
                strategy=SchedulingStrategy.PARALLEL,
                priority=7
            ))
            groups.append(SchedulingGroup(
                group_id=f"parallel_branch_{branch_idx}_b",
                nodes=branch2,
                strategy=SchedulingStrategy.PARALLEL,
                priority=7
            ))
            processed_nodes.update(branch1 + branch2)
    
    # 4. Default groups for remaining nodes
    for node in graph.graph.nodes:
        if node.op in ['call_function', 'call_method', 'call_module']:
            if node.name not in processed_nodes:
                groups.append(SchedulingGroup(
                    group_id=f"default_{node.name}",
                    nodes=[node.name],
                    strategy=SchedulingStrategy.SEQUENTIAL,
                    priority=0
                ))
    
    return groups
```

### Stage Creation

**Topological Sort with Priorities**:
```python
def _create_execution_stages(self, groups, dependencies):
    """Create stages respecting dependencies and priorities."""
    # Build group dependencies
    group_deps = defaultdict(set)
    node_to_group = {n: g.group_id for g in groups for n in g.nodes}
    
    for group in groups:
        for node in group.nodes:
            for dep in dependencies.get(node, set()):
                dep_group = node_to_group.get(dep)
                if dep_group and dep_group != group.group_id:
                    group_deps[group.group_id].add(dep_group)
    
    # Sort groups by priority
    sorted_groups = sorted(groups, key=lambda g: -g.priority)
    
    # Topological scheduling
    stages = []
    scheduled = set()
    remaining = {g.group_id: g for g in sorted_groups}
    
    while remaining:
        # Find ready groups (all deps satisfied)
        ready = [g for gid, g in remaining.items()
                 if group_deps[gid].issubset(scheduled)]
        
        if not ready:
            # Break cycle (shouldn't happen for DAGs)
            ready = [next(iter(remaining.values()))]
            logger.warning(f"Breaking cycle at {ready[0].group_id}")
        
        # Add to stage
        stages.append(ready)
        for group in ready:
            scheduled.add(group.group_id)
            del remaining[group.group_id]
    
    return stages
```

**Example Schedule**:
```python
schedule = ExecutionSchedule(
    stages=[
        # Stage 0: Parallel modality branches
        [
            SchedulingGroup(id='parallel_vision', nodes=['v1','v2','v3']),
            SchedulingGroup(id='parallel_text', nodes=['t1','t2','t3'])
        ],
        # Stage 1: Fusion (depends on stage 0)
        [
            SchedulingGroup(id='fusion', nodes=['f1'])
        ],
        # Stage 2: Classifier
        [
            SchedulingGroup(id='classifier', nodes=['c1'])
        ]
    ],
    node_to_stage={'v1': 0, 'v2': 0, ..., 'f1': 1, 'c1': 2},
    strategy=SchedulingStrategy.PARALLEL
)
```

### Pipeline Scheduler

**For CNN Workloads**:
```python
class PipelineScheduler(Scheduler):
    def __init__(self, num_stages: int = 3):
        super().__init__()
        self.num_stages = num_stages
    
    def create_pipeline_schedule(self, graph):
        # Find CNN operations
        conv_ops = [n for n in graph.graph.nodes
                    if any(op in str(n.target).lower() 
                          for op in ['conv', 'pool', 'norm', 'relu'])]
        
        # Divide into pipeline stages
        stage_size = max(1, len(conv_ops) // self.num_stages)
        pipeline_stages = []
        
        for i in range(0, len(conv_ops), stage_size):
            stage_nodes = conv_ops[i:i+stage_size]
            group = SchedulingGroup(
                group_id=f"pipeline_stage_{len(pipeline_stages)}",
                nodes=[n.name for n in stage_nodes],
                strategy=SchedulingStrategy.PIPELINE,
                priority=self.num_stages - len(pipeline_stages)
            )
            pipeline_stages.append([group])
        
        return ExecutionSchedule(
            stages=pipeline_stages,
            total_stages=len(pipeline_stages),
            strategy=SchedulingStrategy.PIPELINE
        )
```

### Dynamic Scheduler

**Runtime Adaptation**:
```python
class DynamicScheduler(Scheduler):
    def create_adaptive_schedule(self, graph, runtime_constraints):
        """Adapt schedule to runtime constraints."""
        base_schedule = self.create_schedule(graph)
        
        # Adapt for memory constraints
        if 'memory_limit' in runtime_constraints:
            base_schedule = self._adapt_for_memory(
                base_schedule, 
                runtime_constraints['memory_limit']
            )
        
        # Adapt for latency targets
        if 'latency_target' in runtime_constraints:
            base_schedule = self._adapt_for_latency(
                base_schedule,
                runtime_constraints['latency_target']
            )
        
        return base_schedule
    
    def _adapt_for_memory(self, schedule, limit_gb):
        """Increase stages to reduce peak memory."""
        if schedule.total_stages < 10:
            # Split large stages
            new_stages = []
            for stage in schedule.stages:
                if len(stage) > 2:
                    mid = len(stage) // 2
                    new_stages.append(stage[:mid])
                    new_stages.append(stage[mid:])
                else:
                    new_stages.append(stage)
            
            schedule.stages = new_stages
            schedule.total_stages = len(new_stages)
        
        return schedule
    
    def _adapt_for_latency(self, schedule, target_ms):
        """Merge stages to reduce overhead."""
        if schedule.total_stages > 3:
            # Merge adjacent low-priority stages
            new_stages = []
            i = 0
            while i < len(schedule.stages):
                stage = schedule.stages[i]
                
                # Try to merge with next
                if i + 1 < len(schedule.stages):
                    next_stage = schedule.stages[i + 1]
                    if all(g.priority < 5 for g in stage + next_stage):
                        new_stages.append(stage + next_stage)
                        i += 2
                        continue
                
                new_stages.append(stage)
                i += 1
            
            schedule.stages = new_stages
            schedule.total_stages = len(new_stages)
        
        return schedule
```

## Placement Engine

### Overview

**File**: `genie/semantic/placement.py`

**Purpose**: Decide which device executes each operation.

**Device Model**:
```python
@dataclass
class DeviceCapabilities:
    device_id: str              # "cuda:0", "remote_gpu:0"
    device_type: DeviceType     # CPU, GPU, REMOTE_GPU
    memory_gb: float            # 24.0
    compute_tflops: float       # 15.0
    bandwidth_gbps: float       # 100.0
    supports_fp16: bool         # True
    is_available: bool          # True
    current_memory_used: float  # 2.3 GB
```

### Placement Decisions

**Decision Factors**:
1. Execution phase (prefill/decode)
2. Memory pattern (persistent/streaming)
3. Compute intensity
4. Co-location requirements
5. Device availability

**Implementation**:
```python
def _make_placement_decision(self, node, hints, current_plan):
    # 1. Check explicit hint
    if 'placement_hint' in node.meta:
        device = self._resolve_device_hint(node.meta['placement_hint'])
        return PlacementDecision(
            node_name=node.name,
            device_id=device,
            reasoning="Explicit placement hint"
        )
    
    # 2. Semantic-based placement
    if 'semantic' in node.meta:
        return self._semantic_placement(node, hints)
    
    # 3. Default placement
    return self._default_placement(node)
```

**Semantic Placement**:
```python
def _semantic_placement(self, node, hints):
    metadata = node.meta['semantic']
    
    # Phase-based placement
    if metadata.execution_phase == ExecutionPhase.DECODE:
        # Co-locate with KV cache
        device = self._find_kv_cache_device()
        return PlacementDecision(
            node_name=node.name,
            device_id=device,
            reasoning="LLM decode - co-locate with KV cache",
            priority=8
        )
    
    elif metadata.execution_phase == ExecutionPhase.VISION_BACKBONE:
        # Use bandwidth-optimized GPU
        device = self._select_best_gpu()
        return PlacementDecision(
            node_name=node.name,
            device_id=device,
            reasoning="Vision backbone - GPU optimized",
            priority=6
        )
    
    elif metadata.execution_phase == ExecutionPhase.MULTIMODAL_FUSION:
        # Local device for low-latency fusion
        device = self._select_fusion_device()
        return PlacementDecision(
            node_name=node.name,
            device_id=device,
            reasoning="Multi-modal fusion - low latency needed",
            priority=7
        )
    
    # Memory pattern-based
    if metadata.memory_pattern == MemoryPattern.PERSISTENT:
        # Avoid remote for persistent data
        device = self._select_local_device()
        return PlacementDecision(
            device_id=device,
            reasoning="Persistent memory - avoid remote"
        )
```

### Device Selection Algorithms

**Best GPU Selection**:
```python
def _select_best_gpu(self):
    """Select GPU with best compute/memory."""
    best_device = None
    best_score = -1
    
    for device_id, device in self.device_map.items():
        if device.device_type in [DeviceType.GPU, DeviceType.REMOTE_GPU]:
            if device.is_available:
                # Score = compute + memory (normalized)
                score = device.compute_tflops + (device.memory_gb / 10)
                
                # Penalty for remote devices (network latency)
                if device.device_type == DeviceType.REMOTE_GPU:
                    score *= 0.8
                
                if score > best_score:
                    best_score = score
                    best_device = device_id
    
    return best_device or "cpu:0"
```

**Fusion Device Selection**:
```python
def _select_fusion_device(self):
    """Select device for fusion (prioritize bandwidth)."""
    best_device = None
    best_bandwidth = -1
    
    for device_id, device in self.device_map.items():
        if device.device_type in [DeviceType.GPU, DeviceType.REMOTE_GPU]:
            if device.is_available and device.bandwidth_gbps > best_bandwidth:
                best_bandwidth = device.bandwidth_gbps
                best_device = device_id
    
    return best_device or "cuda:0"
```

**Least Loaded Device**:
```python
def _select_least_loaded_device(self):
    """Select device with lowest memory usage."""
    best_device = None
    min_usage = float('inf')
    
    for device_id, device in self.device_map.items():
        if device.is_available:
            usage = device.current_memory_used / device.memory_gb
            if usage < min_usage:
                min_usage = usage
                best_device = device_id
    
    return best_device or "cpu:0"
```

### Placement Plan

```python
@dataclass
class PlacementPlan:
    decisions: Dict[str, PlacementDecision]
    device_assignments: Dict[str, List[str]]  # device → nodes
    colocation_groups: Dict[str, str]         # group → device
    total_devices_used: int
    strategy: str
```

**Example Plan**:
```python
plan = PlacementPlan(
    decisions={
        'vision_conv1': PlacementDecision(
            device_id='remote_gpu:0',
            reasoning='Vision backbone - bandwidth optimized',
            priority=6
        ),
        'language_attn1': PlacementDecision(
            device_id='remote_gpu:1',
            reasoning='LLM attention - compute optimized',
            priority=6
        ),
        'fusion_cross_attn': PlacementDecision(
            device_id='cuda:0',
            reasoning='Fusion point - local for low latency',
            priority=8
        )
    },
    device_assignments={
        'remote_gpu:0': ['vision_conv1', 'vision_conv2', ...],
        'remote_gpu:1': ['language_attn1', 'language_attn2', ...],
        'cuda:0': ['fusion_cross_attn', 'classifier']
    },
    colocation_groups={
        'kv_cache': 'remote_gpu:1'
    },
    total_devices_used=3
)
```

## Adaptive Optimization

### Concept

**Learning from Execution Feedback**:
```python
class AdaptiveOptimizer(SemanticOptimizer):
    def __init__(self):
        super().__init__(enable_all=True)
        self.performance_history = []
        self.optimization_effectiveness = defaultdict(float)
    
    def optimize_with_feedback(self, graph, profile, previous_perf):
        # Update effectiveness
        if previous_perf:
            self._update_effectiveness(previous_perf)
        
        # Adjust enabled optimizations
        self._adjust_optimizations()
        
        # Apply optimizations
        return self.optimize(graph, profile)
```

**Effectiveness Tracking**:
```python
def _update_effectiveness(self, perf_metrics):
    """Learn from previous execution."""
    latency = perf_metrics.get('latency', 1.0)
    throughput = perf_metrics.get('throughput', 1.0)
    
    # Score = throughput / latency (higher is better)
    score = throughput / max(latency, 0.001)
    
    # Store in history
    self.performance_history.append({
        'score': score,
        'metrics': perf_metrics,
        'optimizations': list(self._optimization_stats.keys())
    })
    
    # Update per-optimization effectiveness
    for opt in self._optimization_stats:
        self.optimization_effectiveness[opt] = score

def _adjust_optimizations(self):
    """Enable/disable based on effectiveness."""
    threshold = 0.5
    
    for opt_type in OptimizationType:
        effectiveness = self.optimization_effectiveness.get(opt_type.value, 1.0)
        self.enabled_optimizations[opt_type] = effectiveness > threshold
```

## Integration Example

### End-to-End Optimization Pipeline

```python
# 1. Analyze graph
from genie.semantic.analyzer import SemanticAnalyzer
analyzer = SemanticAnalyzer()
profile = analyzer.analyze_graph(graph)

# 2. Optimize
from genie.semantic.optimizer import SemanticOptimizer
optimizer = SemanticOptimizer()
optimized_graph, opt_plan = optimizer.optimize(fx_graph, profile)

# 3. Schedule
from genie.semantic.scheduling import Scheduler
scheduler = Scheduler()
schedule = scheduler.create_schedule(optimized_graph, opt_plan)

# 4. Place
from genie.semantic.placement import PlacementEngine
placement_engine = PlacementEngine()
placement_plan = placement_engine.create_placement_plan(
    optimized_graph, 
    opt_plan, 
    schedule
)

# 5. Execute
from genie.semantic.workload import ExecutionPlan
execution_plan = ExecutionPlan(
    plan_id="exec_1",
    fragments=...,
    placement={frag_id: placement_plan.decisions[frag_id].device_id},
    transfers=...,
    feature_flags={'overlap_io': True}
)

from genie.runtime.dpdk_backend import DPDKBackend
backend = DPDKBackend()
results = backend.execute_plan(execution_plan)
```

## Testing

### Optimizer Tests

```python
def test_llm_optimization():
    # Create LLM graph
    graph = create_transformer_graph(num_layers=12)
    profile = WorkloadProfile(workload_type=WorkloadType.LLM)
    
    # Optimize
    optimizer = SemanticOptimizer()
    opt_graph, plan = optimizer.optimize(graph, profile)
    
    # Verify KV cache co-location
    assert OptimizationType.KV_CACHE_COLOCATION in plan.optimizations
    assert 'kv_cache' in plan.colocation_groups
    
    # Verify prefill parallelization
    assert OptimizationType.PREFILL_PARALLELIZATION in plan.optimizations
    assert len(plan.parallel_branches) > 0

def test_vision_optimization():
    graph = create_resnet_graph(depth=50)
    profile = WorkloadProfile(workload_type=WorkloadType.VISION)
    
    optimizer = SemanticOptimizer()
    opt_graph, plan = optimizer.optimize(graph, profile)
    
    # Verify pipelining
    assert OptimizationType.CNN_PIPELINING in plan.optimizations
    assert len(plan.pipeline_stages) >= 3
```

### Scheduler Tests

```python
def test_schedule_creation():
    scheduler = Scheduler()
    schedule = scheduler.create_schedule(graph, opt_plan)
    
    # Verify stages respect dependencies
    for stage in schedule.stages:
        for group in stage:
            for node in group.nodes:
                # All dependencies in earlier stages
                deps = dependencies[node]
                for dep in deps:
                    dep_stage = schedule.node_to_stage[dep]
                    assert dep_stage < schedule.node_to_stage[node]
```

### Placement Tests

```python
def test_placement_engine():
    engine = PlacementEngine()
    plan = engine.create_placement_plan(graph, opt_plan)
    
    # Verify co-location groups on same device
    for group_id, device_id in plan.colocation_groups.items():
        nodes = opt_plan.colocation_groups[group_id]
        for node in nodes:
            assert plan.decisions[node].device_id == device_id
```

## Performance Considerations

### Optimization Overhead

**Target**: Optimization should complete quickly (<50ms)

**Profiling**:
```python
import time

start = time.perf_counter()
opt_graph, plan = optimizer.optimize(graph, profile)
elapsed = time.perf_counter() - start

print(f"Optimization time: {elapsed*1000:.2f} ms")
# Should be < 50ms for most graphs
```

**Optimization**:
- Cache optimization plans for repeated graphs
- Use early exit when optimization opportunities not found
- Parallelize independent optimizations

### Memory Overhead

**Metadata Storage**:
```python
# Each node gets additional metadata
node.meta['placement_hint'] = 'kv_cache_device'
node.meta['parallel_group'] = 'prefill_branch_1'
node.meta['priority'] = 8
node.meta['can_recompute'] = True
node.meta['fusion_group'] = 'qkv_projection'

# Typical overhead: ~200 bytes per node
# For 1000-node graph: ~200 KB metadata
```

## Configuration

### Optimizer Configuration

```python
# Enable specific optimizations
optimizer = SemanticOptimizer(enable_all=False)
optimizer.enabled_optimizations[OptimizationType.KV_CACHE_COLOCATION] = True
optimizer.enabled_optimizations[OptimizationType.CNN_PIPELINING] = True

# Get statistics
stats = optimizer.get_optimization_stats()
{
    'kv_cache_colocation': 15,  # Applied 15 times
    'cnn_pipelining': 8
}
```

### Scheduler Configuration

```python
# Pipeline scheduler with custom stages
pipeline_scheduler = PipelineScheduler(num_stages=4)

# Dynamic scheduler with constraints
dynamic_scheduler = DynamicScheduler()
schedule = dynamic_scheduler.create_adaptive_schedule(
    graph,
    runtime_constraints={
        'memory_limit': 16.0,  # GB
        'latency_target': 50.0  # ms
    }
)
```

### Placement Configuration

```python
# Custom device list
devices = [
    DeviceCapabilities(
        device_id="cuda:0",
        device_type=DeviceType.GPU,
        memory_gb=24.0,
        compute_tflops=20.0,
        bandwidth_gbps=900.0
    ),
    DeviceCapabilities(
        device_id="remote_gpu:0",
        device_type=DeviceType.REMOTE_GPU,
        memory_gb=80.0,
        compute_tflops=50.0,
        bandwidth_gbps=100.0,  # Network limited
        metadata={'latency_ms': 5}
    )
]

engine = PlacementEngine(devices=devices)
```

## Advanced Features

### Priority-Based Scheduling

**High-Priority Operations**:
```python
# Mark critical operations
fusion_node.meta['priority'] = 10  # Highest
attention_node.meta['priority'] = 8
conv_node.meta['priority'] = 5
reshape_node.meta['priority'] = 2  # Lowest

# Scheduler respects priorities
sorted_groups = sorted(groups, key=lambda g: -g.priority)
```

### Heterogeneous Placement

**Different Devices for Different Modalities**:
```python
# Vision → Bandwidth-optimized GPU
for node in vision_branch:
    node.meta['placement_hint'] = 'bandwidth_gpu'

# Language → Compute-optimized GPU  
for node in language_branch:
    node.meta['placement_hint'] = 'compute_gpu'

# Fusion → Local GPU
for node in fusion_nodes:
    node.meta['placement_hint'] = 'local_gpu'
```

### Cost Model Integration (Future)

**Concept**: Use learned cost model for decisions
```python
class CostModel:
    def estimate_compute_cost(self, node, device):
        """Estimate execution cost on device."""
        # Machine learning model predicting latency
        
    def estimate_transfer_cost(self, size_bytes, src_device, dst_device):
        """Estimate transfer cost between devices."""
        
    def estimate_total_cost(self, plan):
        """Estimate total execution cost."""
        compute_cost = sum(self.estimate_compute_cost(n, d) 
                          for n, d in plan.placement.items())
        transfer_cost = sum(self.estimate_transfer_cost(...) 
                           for t in plan.transfers)
        return compute_cost + transfer_cost
```

## Debugging

### Visualize Optimization Plan

```python
def print_optimization_plan(plan: OptimizationPlan):
    print("=== Optimization Plan ===")
    print(f"Optimizations: {[o.value for o in plan.optimizations]}")
    
    if plan.colocation_groups:
        print("\nCo-location Groups:")
        for group_id, nodes in plan.colocation_groups.items():
            print(f"  {group_id}: {len(nodes)} nodes")
    
    if plan.pipeline_stages:
        print("\nPipeline Stages:")
        for i, stage in enumerate(plan.pipeline_stages):
            print(f"  Stage {i}: {len(stage)} nodes")
    
    if plan.parallel_branches:
        print("\nParallel Branches:")
        for i, (b1, b2) in enumerate(plan.parallel_branches):
            print(f"  Branch {i}: {len(b1)} nodes || {len(b2)} nodes")
    
    if plan.recompute_nodes:
        print(f"\nRecompute: {len(plan.recompute_nodes)} nodes")
```

### Trace Optimization Application

```python
# Enable debug logging
logging.getLogger('genie.semantic.optimizer').setLevel(logging.DEBUG)

# Optimize with tracing
optimizer = SemanticOptimizer()
graph, plan = optimizer.optimize(fx_graph, profile)

# Output shows:
# DEBUG: Applying LLM optimizations
# INFO: Co-located 12 KV cache operations
# INFO: Parallelized prefill with 6 + 6 operations
# DEBUG: Applying general optimizations
# INFO: Identified 8 dead code nodes
```

## See Also

- [Semantic Layer](06-semantic-layer.md) - Overall semantic architecture
- [Pattern Recognition](07-pattern-recognition.md) - Pattern detection
- [Runtime Transport](05-runtime-transport.md) - Zero-copy execution
- [HotNets'25 Paper](../../.kiro/HotNets25.tex) - Section 3.2

---

**Last Updated**: 2025-09-30  
**Status**: Complete  
**Maintainers**: Genie Core Team
