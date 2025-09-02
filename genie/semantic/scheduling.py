"""Scheduling strategies for semantic-aware execution.

This module implements scheduling strategies for different workload types,
including pipeline scheduling for CNNs and parallel scheduling for multi-modal.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import torch.fx as fx
from collections import defaultdict, deque

from ..core.semantic_metadata import ExecutionPhase

logger = logging.getLogger(__name__)


class SchedulingStrategy(Enum):
    """Types of scheduling strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    DYNAMIC = "dynamic"
    PRIORITY = "priority"


@dataclass
class SchedulingGroup:
    """Group of nodes to be scheduled together."""
    group_id: str
    nodes: List[str]
    strategy: SchedulingStrategy
    priority: int = 0
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ExecutionSchedule:
    """Execution schedule for a computation graph."""
    stages: List[List[SchedulingGroup]] = field(default_factory=list)
    node_to_stage: Dict[str, int] = field(default_factory=dict)
    node_to_group: Dict[str, str] = field(default_factory=dict)
    total_stages: int = 0
    strategy: SchedulingStrategy = SchedulingStrategy.SEQUENTIAL
    metadata: Dict = field(default_factory=dict)


class Scheduler:
    """Scheduler for semantic-aware execution planning."""
    
    def __init__(self):
        self._scheduling_cache = {}
        self._stats = defaultdict(int)
    
    def create_schedule(self, graph: fx.GraphModule, 
                        optimization_plan: Optional[Dict] = None) -> ExecutionSchedule:
        """Create execution schedule for a graph.
        
        Args:
            graph: FX GraphModule to schedule
            optimization_plan: Optional optimization plan with placement hints
            
        Returns:
            ExecutionSchedule with stages and groups
        """
        # Analyze graph dependencies
        dependencies = self._analyze_dependencies(graph)
        
        # Identify scheduling groups based on metadata
        groups = self._identify_scheduling_groups(graph, optimization_plan)
        
        # Create execution stages
        stages = self._create_execution_stages(groups, dependencies)
        
        # Build schedule
        schedule = ExecutionSchedule(
            stages=stages,
            total_stages=len(stages)
        )
        
        # Fill node mappings
        for stage_idx, stage_groups in enumerate(stages):
            for group in stage_groups:
                for node_name in group.nodes:
                    schedule.node_to_stage[node_name] = stage_idx
                    schedule.node_to_group[node_name] = group.group_id
        
        # Determine overall strategy
        schedule.strategy = self._determine_strategy(groups)
        
        # Add metadata
        schedule.metadata['total_nodes'] = len(list(graph.graph.nodes))
        schedule.metadata['total_groups'] = len(groups)
        
        logger.info(f"Created schedule with {schedule.total_stages} stages and {len(groups)} groups")
        
        return schedule
    
    def _analyze_dependencies(self, graph: fx.GraphModule) -> Dict[str, Set[str]]:
        """Analyze node dependencies in the graph.
        
        Args:
            graph: FX GraphModule
            
        Returns:
            Dictionary mapping node names to their dependencies
        """
        dependencies = defaultdict(set)
        
        for node in graph.graph.nodes:
            if node.op in ['call_function', 'call_method', 'call_module']:
                for inp in node.all_input_nodes:
                    dependencies[node.name].add(inp.name)
        
        return dependencies
    
    def _identify_scheduling_groups(self, graph: fx.GraphModule, 
                                   optimization_plan: Optional[Dict]) -> List[SchedulingGroup]:
        """Identify scheduling groups based on node metadata.
        
        Args:
            graph: FX GraphModule
            optimization_plan: Optional optimization plan
            
        Returns:
            List of SchedulingGroups
        """
        groups = []
        processed_nodes = set()
        
        # 1. Groups from optimization plan
        if optimization_plan:
            # Co-location groups
            if hasattr(optimization_plan, 'colocation_groups'):
                for group_id, node_names in optimization_plan.colocation_groups.items():
                    group = SchedulingGroup(
                        group_id=f"coloc_{group_id}",
                        nodes=node_names,
                        strategy=SchedulingStrategy.SEQUENTIAL,
                        priority=10
                    )
                    groups.append(group)
                    processed_nodes.update(node_names)
            
            # Pipeline stages
            if hasattr(optimization_plan, 'pipeline_stages'):
                for stage_idx, stage_nodes in enumerate(optimization_plan.pipeline_stages):
                    group = SchedulingGroup(
                        group_id=f"pipeline_stage_{stage_idx}",
                        nodes=stage_nodes,
                        strategy=SchedulingStrategy.PIPELINE,
                        priority=5
                    )
                    groups.append(group)
                    processed_nodes.update(stage_nodes)
            
            # Parallel branches
            if hasattr(optimization_plan, 'parallel_branches'):
                for branch_idx, (branch1, branch2) in enumerate(optimization_plan.parallel_branches):
                    # First branch
                    group1 = SchedulingGroup(
                        group_id=f"parallel_branch_{branch_idx}_a",
                        nodes=branch1,
                        strategy=SchedulingStrategy.PARALLEL,
                        priority=7
                    )
                    groups.append(group1)
                    processed_nodes.update(branch1)
                    
                    # Second branch
                    group2 = SchedulingGroup(
                        group_id=f"parallel_branch_{branch_idx}_b",
                        nodes=branch2,
                        strategy=SchedulingStrategy.PARALLEL,
                        priority=7
                    )
                    groups.append(group2)
                    processed_nodes.update(branch2)
        
        # 2. Groups from node metadata
        parallel_groups = defaultdict(list)
        fusion_groups = defaultdict(list)
        
        for node in graph.graph.nodes:
            if node.name in processed_nodes:
                continue
            
            # Check for parallel group metadata
            if 'parallel_group' in node.meta:
                group_name = node.meta['parallel_group']
                parallel_groups[group_name].append(node.name)
            
            # Check for fusion group metadata
            elif 'fusion_group' in node.meta:
                group_name = node.meta['fusion_group']
                fusion_groups[group_name].append(node.name)
            
            # Check for priority operations
            elif node.meta.get('priority', 0) >= 8:
                # High priority operations get their own group
                group = SchedulingGroup(
                    group_id=f"priority_{node.name}",
                    nodes=[node.name],
                    strategy=SchedulingStrategy.PRIORITY,
                    priority=node.meta['priority']
                )
                groups.append(group)
                processed_nodes.add(node.name)
        
        # Create groups for parallel operations
        for group_name, nodes in parallel_groups.items():
            if nodes:
                group = SchedulingGroup(
                    group_id=f"parallel_{group_name}",
                    nodes=nodes,
                    strategy=SchedulingStrategy.PARALLEL,
                    priority=6
                )
                groups.append(group)
                processed_nodes.update(nodes)
        
        # Create groups for fusion operations
        for group_name, nodes in fusion_groups.items():
            if nodes:
                group = SchedulingGroup(
                    group_id=f"fusion_{group_name}",
                    nodes=nodes,
                    strategy=SchedulingStrategy.SEQUENTIAL,
                    priority=5
                )
                groups.append(group)
                processed_nodes.update(nodes)
        
        # 3. Default groups for remaining nodes
        for node in graph.graph.nodes:
            if node.op in ['call_function', 'call_method', 'call_module']:
                if node.name not in processed_nodes:
                    # Create individual group
                    group = SchedulingGroup(
                        group_id=f"default_{node.name}",
                        nodes=[node.name],
                        strategy=SchedulingStrategy.SEQUENTIAL,
                        priority=0
                    )
                    groups.append(group)
        
        return groups
    
    def _create_execution_stages(self, groups: List[SchedulingGroup], 
                                dependencies: Dict[str, Set[str]]) -> List[List[SchedulingGroup]]:
        """Create execution stages respecting dependencies.
        
        Args:
            groups: List of scheduling groups
            dependencies: Node dependency map
            
        Returns:
            List of stages, each containing groups that can execute in parallel
        """
        # Build group dependencies
        group_deps = defaultdict(set)
        node_to_group = {}
        
        for group in groups:
            for node in group.nodes:
                node_to_group[node] = group.group_id
        
        for group in groups:
            for node in group.nodes:
                for dep in dependencies.get(node, set()):
                    dep_group = node_to_group.get(dep)
                    if dep_group and dep_group != group.group_id:
                        group_deps[group.group_id].add(dep_group)
        
        # Sort groups by priority
        sorted_groups = sorted(groups, key=lambda g: -g.priority)
        
        # Create stages using topological sort with priority
        stages = []
        scheduled = set()
        remaining = {g.group_id: g for g in sorted_groups}
        
        while remaining:
            # Find groups that can be scheduled
            ready = []
            for group_id, group in remaining.items():
                deps = group_deps[group_id]
                if deps.issubset(scheduled):
                    ready.append(group)
            
            if not ready:
                # Break cycle by scheduling highest priority remaining
                ready = [next(iter(remaining.values()))]
                logger.warning(f"Breaking dependency cycle with group {ready[0].group_id}")
            
            # Group ready nodes by strategy
            stage_groups = []
            parallel_groups = []
            
            for group in ready:
                if group.strategy == SchedulingStrategy.PARALLEL:
                    parallel_groups.append(group)
                else:
                    stage_groups.append(group)
            
            # Add parallel groups to same stage if possible
            if parallel_groups:
                stage_groups.extend(parallel_groups)
            
            if stage_groups:
                stages.append(stage_groups)
                for group in stage_groups:
                    scheduled.add(group.group_id)
                    del remaining[group.group_id]
        
        return stages
    
    def _determine_strategy(self, groups: List[SchedulingGroup]) -> SchedulingStrategy:
        """Determine overall scheduling strategy.
        
        Args:
            groups: List of scheduling groups
            
        Returns:
            Overall SchedulingStrategy
        """
        strategy_counts = defaultdict(int)
        
        for group in groups:
            strategy_counts[group.strategy] += len(group.nodes)
        
        if not strategy_counts:
            return SchedulingStrategy.SEQUENTIAL
        
        # Return most common strategy
        return max(strategy_counts.items(), key=lambda x: x[1])[0]


class PipelineScheduler(Scheduler):
    """Specialized scheduler for pipeline execution."""
    
    def __init__(self, num_stages: int = 3):
        super().__init__()
        self.num_stages = num_stages
    
    def create_pipeline_schedule(self, graph: fx.GraphModule) -> ExecutionSchedule:
        """Create pipeline schedule for CNN-like workloads.
        
        Args:
            graph: FX GraphModule
            
        Returns:
            Pipeline ExecutionSchedule
        """
        # Find all convolution and related operations
        conv_ops = []
        for node in graph.graph.nodes:
            if node.op == 'call_function':
                op_name = str(node.target).lower()
                if any(op in op_name for op in ['conv', 'pool', 'norm', 'relu']):
                    conv_ops.append(node)
        
        if not conv_ops:
            # Fallback to regular scheduling
            return self.create_schedule(graph)
        
        # Divide into pipeline stages
        stage_size = max(1, len(conv_ops) // self.num_stages)
        pipeline_stages = []
        
        for i in range(0, len(conv_ops), stage_size):
            stage_nodes = conv_ops[i:i+stage_size]
            if stage_nodes:
                group = SchedulingGroup(
                    group_id=f"pipeline_stage_{len(pipeline_stages)}",
                    nodes=[n.name for n in stage_nodes],
                    strategy=SchedulingStrategy.PIPELINE,
                    priority=self.num_stages - len(pipeline_stages)
                )
                pipeline_stages.append([group])
        
        # Create schedule
        schedule = ExecutionSchedule(
            stages=pipeline_stages,
            total_stages=len(pipeline_stages),
            strategy=SchedulingStrategy.PIPELINE
        )
        
        # Fill mappings
        for stage_idx, stage_groups in enumerate(pipeline_stages):
            for group in stage_groups:
                for node_name in group.nodes:
                    schedule.node_to_stage[node_name] = stage_idx
                    schedule.node_to_group[node_name] = group.group_id
        
        schedule.metadata['pipeline_depth'] = self.num_stages
        
        return schedule


class DynamicScheduler(Scheduler):
    """Dynamic scheduler that adapts based on runtime conditions."""
    
    def __init__(self):
        super().__init__()
        self.runtime_stats = defaultdict(lambda: {'latency': 0, 'memory': 0})
    
    def create_adaptive_schedule(self, graph: fx.GraphModule, 
                                runtime_constraints: Optional[Dict] = None) -> ExecutionSchedule:
        """Create adaptive schedule based on runtime constraints.
        
        Args:
            graph: FX GraphModule
            runtime_constraints: Optional runtime constraints (memory, latency targets)
            
        Returns:
            Adaptive ExecutionSchedule
        """
        base_schedule = self.create_schedule(graph)
        
        if not runtime_constraints:
            return base_schedule
        
        # Adapt based on constraints
        memory_limit = runtime_constraints.get('memory_limit')
        latency_target = runtime_constraints.get('latency_target')
        
        if memory_limit:
            base_schedule = self._adapt_for_memory(base_schedule, memory_limit)
        
        if latency_target:
            base_schedule = self._adapt_for_latency(base_schedule, latency_target)
        
        base_schedule.strategy = SchedulingStrategy.DYNAMIC
        
        return base_schedule
    
    def _adapt_for_memory(self, schedule: ExecutionSchedule, limit: float) -> ExecutionSchedule:
        """Adapt schedule for memory constraints.
        
        Args:
            schedule: Base schedule
            limit: Memory limit
            
        Returns:
            Adapted schedule
        """
        # Increase number of stages to reduce peak memory
        # This is simplified - real implementation would estimate memory usage
        
        if schedule.total_stages < 10:  # Arbitrary limit
            # Split large stages
            new_stages = []
            for stage in schedule.stages:
                if len(stage) > 2:
                    # Split into smaller stages
                    mid = len(stage) // 2
                    new_stages.append(stage[:mid])
                    new_stages.append(stage[mid:])
                else:
                    new_stages.append(stage)
            
            schedule.stages = new_stages
            schedule.total_stages = len(new_stages)
        
        schedule.metadata['memory_optimized'] = True
        
        return schedule
    
    def _adapt_for_latency(self, schedule: ExecutionSchedule, target: float) -> ExecutionSchedule:
        """Adapt schedule for latency target.
        
        Args:
            schedule: Base schedule
            target: Latency target
            
        Returns:
            Adapted schedule
        """
        # Merge stages to reduce overhead
        # This is simplified - real implementation would estimate latency
        
        if schedule.total_stages > 3:
            # Merge adjacent stages with low priority
            new_stages = []
            i = 0
            while i < len(schedule.stages):
                stage = schedule.stages[i]
                
                # Check if can merge with next stage
                if i + 1 < len(schedule.stages):
                    next_stage = schedule.stages[i + 1]
                    # Merge if both have low priority
                    if all(g.priority < 5 for g in stage + next_stage):
                        merged = stage + next_stage
                        new_stages.append(merged)
                        i += 2
                        continue
                
                new_stages.append(stage)
                i += 1
            
            schedule.stages = new_stages
            schedule.total_stages = len(new_stages)
        
        schedule.metadata['latency_optimized'] = True
        
        return schedule
    
    def update_runtime_stats(self, node_name: str, latency: float, memory: float):
        """Update runtime statistics for adaptive scheduling.
        
        Args:
            node_name: Name of the node
            latency: Measured latency
            memory: Measured memory usage
        """
        self.runtime_stats[node_name]['latency'] = latency
        self.runtime_stats[node_name]['memory'] = memory
