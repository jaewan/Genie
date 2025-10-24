"""
Basic scheduler implementing greedy placement with co-location.

This is intentionally simple - proves the concept, not optimal.
Phase 4 improvements will add sophisticated algorithms.
"""

from dataclasses import dataclass
from typing import Dict, List, Set
import logging

logger = logging.getLogger(__name__)


@dataclass
class Device:
    """Simple device representation."""
    id: str
    memory_gb: float
    location: str  # For co-location (e.g., "rack1", "node2")


@dataclass
class SchedulingDecision:
    """Placement decision for a single node."""
    node_id: str
    device_id: str
    reason: str  # Why this placement? (for debugging)


class BasicScheduler:
    """
    Basic greedy scheduler with semantic awareness.

    Strategy:
    1. Identify co-location groups (e.g., KV cache + decoder)
    2. Place co-location groups on same device
    3. Balance remaining operations across devices

    This is NOT optimal, but demonstrates semantic-aware placement.
    """

    def __init__(self, devices: List[Device]):
        self.devices = {dev.id: dev for dev in devices}

    def schedule(self, annotated_graph) -> Dict[str, SchedulingDecision]:
        """
        Create placement schedule for graph.

        Returns:
            Dict mapping node_id → SchedulingDecision
        """
        decisions = {}
        device_loads = {dev_id: 0.0 for dev_id in self.devices}

        # Step 1: Identify co-location groups
        colocation_groups = self._find_colocation_groups(annotated_graph)
        logger.info(f"Found {len(colocation_groups)} co-location groups")

        # Step 2: Place co-location groups
        for group in colocation_groups:
            device = self._select_device_for_group(group, device_loads)

            for node_id in group['nodes']:
                decisions[node_id] = SchedulingDecision(
                    node_id=node_id,
                    device_id=device.id,
                    reason=f"co-location:{group['reason']}"
                )
                device_loads[device.id] += 1.0  # Simple load metric

        # Step 3: Place remaining nodes (greedy load balancing)
        placed_nodes = set(decisions.keys())

        for node in annotated_graph.nodes():
            if node.id in placed_nodes:
                continue

            # Simple greedy: pick least loaded device
            device_id = min(device_loads, key=device_loads.get)

            decisions[node.id] = SchedulingDecision(
                node_id=node.id,
                device_id=device_id,
                reason="load_balancing"
            )
            device_loads[device_id] += 1.0

        # Log scheduling summary
        self._log_schedule_summary(decisions, device_loads)

        return decisions

    def _find_colocation_groups(self, annotated_graph) -> List[Dict]:
        """
        Identify nodes that should be co-located.

        Semantic rules:
        1. KV cache + decode operations (LLM decode phase)
        2. Attention Q, K, V projections
        3. Consecutive conv layers (fusion opportunity)
        """
        groups = []

        # Rule 1: KV cache co-location (LLM decode phase)
        kv_cache_nodes = []
        decode_nodes = []

        for node in annotated_graph.nodes():
            metadata = annotated_graph.get_metadata(node.id)
            if metadata and isinstance(metadata, dict):
                # Check for KV cache pattern
                if 'kv_cache' in annotated_graph.patterns:
                    for pattern in annotated_graph.patterns['kv_cache']:
                        if node in pattern.nodes:
                            kv_cache_nodes.append(node.id)

                # Check for decode phase
                if metadata.get('phase') == 'llm_decode':
                    decode_nodes.append(node.id)

        if kv_cache_nodes and decode_nodes:
            # Co-locate KV cache with decode operations
            groups.append({
                'nodes': kv_cache_nodes + decode_nodes,
                'reason': 'kv_cache_decode_colocation'
            })

        # Rule 2: Attention Q, K, V co-location
        if 'attention' in annotated_graph.patterns:
            for attn_pattern in annotated_graph.patterns['attention']:
                # All attention nodes should be on same device
                attn_nodes = [n.id for n in attn_pattern.nodes]
                if len(attn_nodes) > 1:
                    groups.append({
                        'nodes': attn_nodes,
                        'reason': 'attention_colocation'
                    })

        return groups

    def _select_device_for_group(self, group: Dict, device_loads: Dict) -> Device:
        """Select best device for co-location group."""
        # Simple strategy: pick least loaded device
        device_id = min(device_loads, key=device_loads.get)
        return self.devices[device_id]

    def _log_schedule_summary(self, decisions: Dict, device_loads: Dict):
        """Log scheduling decisions for debugging."""
        logger.info("=== Scheduling Summary ===")

        # Count decisions by device
        device_counts = {}
        reason_counts = {}

        for decision in decisions.values():
            device_counts[decision.device_id] = device_counts.get(decision.device_id, 0) + 1
            reason_counts[decision.reason] = reason_counts.get(decision.reason, 0) + 1

        logger.info("Placement by device:")
        for device_id, count in device_counts.items():
            logger.info(f"  {device_id}: {count} nodes")

        logger.info("Placement reasons:")
        for reason, count in reason_counts.items():
            logger.info(f"  {reason}: {count} nodes")

    # ✅ PHASE 1 SEMANTIC OPTIMIZATIONS
    
    def _optimize_llm_phases(self, annotated_graph) -> List[Dict]:
        """
        Optimize placement for LLM workloads (prefill vs decode).
        
        Key insight: Prefill and decode have different scaling characteristics:
        - Prefill: compute-bound, parallelizable across sequence positions
        - Decode: memory-bound, sequential, small batch
        
        Strategy:
        1. Prefill: spread across devices for parallelism
        2. Decode: co-locate with KV cache on single device
        """
        groups = []
        
        # Detect LLM phases in graph
        prefill_nodes = []
        decode_nodes = []
        
        for node in annotated_graph.nodes():
            metadata = annotated_graph.get_metadata(node.id)
            if metadata and isinstance(metadata, dict):
                phase = metadata.get('phase', '')
                if 'prefill' in phase.lower():
                    prefill_nodes.append(node.id)
                elif 'decode' in phase.lower():
                    decode_nodes.append(node.id)
        
        if decode_nodes:
            groups.append({
                'nodes': decode_nodes,
                'reason': 'llm_decode_sequential',
                'optimization': 'single_device'  # Keep on one device for KV cache locality
            })
        
        if prefill_nodes:
            groups.append({
                'nodes': prefill_nodes,
                'reason': 'llm_prefill_parallel',
                'optimization': 'spread_devices'  # Spread for parallelism
            })
        
        return groups

    def _optimize_multimodal_placement(self, annotated_graph) -> List[Dict]:
        """
        Optimize placement for multi-modal workloads (vision + text fusion).
        
        Strategy:
        1. Vision backbone: GPU with good FLOPS (data parallel)
        2. Text encoder: GPU with good memory bandwidth
        3. Fusion: any GPU (small data)
        """
        groups = []
        
        vision_nodes = []
        text_nodes = []
        fusion_nodes = []
        
        for node in annotated_graph.nodes():
            metadata = annotated_graph.get_metadata(node.id)
            if metadata and isinstance(metadata, dict):
                modality = metadata.get('modality', '')
                if modality == 'vision':
                    vision_nodes.append(node.id)
                elif modality == 'text':
                    text_nodes.append(node.id)
                elif modality == 'fusion':
                    fusion_nodes.append(node.id)
        
        if vision_nodes:
            groups.append({
                'nodes': vision_nodes,
                'reason': 'vision_backbone',
                'optimization': 'high_flops'
            })
        
        if text_nodes:
            groups.append({
                'nodes': text_nodes,
                'reason': 'text_encoder',
                'optimization': 'high_bandwidth'
            })
        
        if fusion_nodes:
            groups.append({
                'nodes': fusion_nodes,
                'reason': 'multimodal_fusion',
                'optimization': 'flexible'
            })
        
        return groups

    def estimate_cost(self, node, metadata: Dict) -> float:
        """
        Estimate computational cost for a node.

        Used for load-aware placement decisions.

        Returns: relative cost (higher = more expensive)
        """
        # Extract FLOPs estimate from metadata
        flops = metadata.get('flops', 0.0)

        # Normalize to 0-1 range (assuming max 1B FLOPs per operation)
        max_flops = 1e9
        normalized_cost = min(flops / max_flops, 1.0)

        # Apply phase-based scaling
        phase = metadata.get('phase', '')
        if 'prefill' in phase.lower():
            # Prefill is more expensive, scale up
            normalized_cost *= 1.5
        elif 'decode' in phase.lower():
            # Decode is cheaper, scale down
            normalized_cost *= 0.5

        return normalized_cost
