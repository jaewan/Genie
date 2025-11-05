"""
Unit tests for Phase 2 Semantic Memory Management.

Tests lifetime-based eviction, phase-aware budgets, and recomputation decisions.
"""

import pytest

from djinn.server.semantic_memory_manager import (
    ExecutionPhase,
    DataResidency,
    EvictionPriority,
    TensorLifetime,
    LifetimeBasedEvictor,
    PhaseAwareMemoryManager,
    RecomputationVsStorageDecider,
)


class TestLifetimeBasedEvictor:
    """Test lifetime-based eviction feature."""
    
    def test_evictor_initialization(self):
        """Test evictor initializes correctly."""
        evictor = LifetimeBasedEvictor()
        assert len(evictor.tensor_lifetimes) == 0
        assert len(evictor.node_execution_order) == 0
    
    def test_simple_graph_analysis(self):
        """Test analyzing a simple linear graph."""
        evictor = LifetimeBasedEvictor()
        
        # Simple graph: node_0 → node_1 → node_2
        srg_nodes = [
            {'id': 'node_0', 'operation': 'input', 'metadata': {'residency': 'ephemeral_activation', 'phase': 'unknown'}},
            {'id': 'node_1', 'operation': 'linear', 'metadata': {'residency': 'ephemeral_activation', 'phase': 'unknown'}},
            {'id': 'node_2', 'operation': 'output', 'metadata': {'residency': 'ephemeral_activation', 'phase': 'unknown'}},
        ]
        
        srg_edges = [
            {'source_id': 'node_0', 'target_id': 'node_1', 'tensor_id': 'tensor_01'},
            {'source_id': 'node_1', 'target_id': 'node_2', 'tensor_id': 'tensor_12'},
        ]
        
        evictor.analyze_graph_lifetimes(srg_nodes, srg_edges)
        
        # Should have 2 tensors
        assert len(evictor.tensor_lifetimes) == 2
        assert 'tensor_01' in evictor.tensor_lifetimes
        assert 'tensor_12' in evictor.tensor_lifetimes
    
    def test_execution_order_tracking(self):
        """Test that execution order is tracked correctly."""
        evictor = LifetimeBasedEvictor()
        
        srg_nodes = [
            {'id': 'node_0', 'operation': 'input', 'metadata': {}},
            {'id': 'node_1', 'operation': 'op1', 'metadata': {}},
            {'id': 'node_2', 'operation': 'op2', 'metadata': {}},
        ]
        
        srg_edges = [
            {'source_id': 'node_0', 'target_id': 'node_1', 'tensor_id': 't1'},
            {'source_id': 'node_1', 'target_id': 'node_2', 'tensor_id': 't2'},
        ]
        
        evictor.analyze_graph_lifetimes(srg_nodes, srg_edges)
        
        # Check execution order
        assert evictor.node_execution_order['node_0'] == 0
        assert evictor.node_execution_order['node_1'] == 1
        assert evictor.node_execution_order['node_2'] == 2
    
    def test_last_consumer_identification(self):
        """Test identifying last consumer of a tensor."""
        evictor = LifetimeBasedEvictor()
        
        # Graph: node_0 → node_1, node_2, node_3 (fan-out)
        srg_nodes = [
            {'id': 'node_0', 'operation': 'input', 'metadata': {}},
            {'id': 'node_1', 'operation': 'op1', 'metadata': {}},
            {'id': 'node_2', 'operation': 'op2', 'metadata': {}},
            {'id': 'node_3', 'operation': 'op3', 'metadata': {}},
        ]
        
        srg_edges = [
            {'source_id': 'node_0', 'target_id': 'node_1', 'tensor_id': 't_out'},
            {'source_id': 'node_0', 'target_id': 'node_2', 'tensor_id': 't_out'},
            {'source_id': 'node_0', 'target_id': 'node_3', 'tensor_id': 't_out'},
        ]
        
        evictor.analyze_graph_lifetimes(srg_nodes, srg_edges)
        
        # Last consumer should be node_3
        lifetime = evictor.tensor_lifetimes['t_out']
        assert lifetime.last_consumer_node_id == 'node_3'
        assert len(lifetime.consumer_node_ids) == 3
    
    def test_tensors_to_evict_after_node(self):
        """Test getting tensors to evict after a node."""
        evictor = LifetimeBasedEvictor()
        
        srg_nodes = [
            {'id': 'n0', 'operation': 'input', 'metadata': {}},
            {'id': 'n1', 'operation': 'op1', 'metadata': {}},
            {'id': 'n2', 'operation': 'op2', 'metadata': {}},
        ]
        
        srg_edges = [
            {'source_id': 'n0', 'target_id': 'n1', 'tensor_id': 't1'},
            {'source_id': 'n1', 'target_id': 'n2', 'tensor_id': 't2'},
        ]
        
        evictor.analyze_graph_lifetimes(srg_nodes, srg_edges)
        
        # After n1 completes, tensor 't1' should be evicted
        to_evict = evictor.get_tensors_to_evict_after_node('n1')
        assert 't1' in to_evict
    
    def test_eviction_priority_computation(self):
        """Test eviction priority computation from metadata."""
        evictor = LifetimeBasedEvictor()
        
        # KV cache during decode should be CRITICAL
        priority = evictor._compute_priority({
            'residency': 'stateful_kv_cache',
            'phase': 'llm_decode'
        })
        assert priority == EvictionPriority.CRITICAL
        
        # Weights should be HIGH
        priority = evictor._compute_priority({
            'residency': 'persistent_weight'
        })
        assert priority == EvictionPriority.HIGH
        
        # Cheap activation should be EPHEMERAL
        priority = evictor._compute_priority({
            'residency': 'ephemeral_activation',
            'flop_cost': 100
        })
        assert priority == EvictionPriority.EPHEMERAL
    
    def test_evictor_statistics(self):
        """Test statistics tracking."""
        evictor = LifetimeBasedEvictor()
        
        srg_nodes = [{'id': 'n0', 'operation': 'input', 'metadata': {}}]
        srg_edges = []
        
        evictor.analyze_graph_lifetimes(srg_nodes, srg_edges)
        
        stats = evictor.get_stats()
        assert stats['lifetime_analyses'] == 1
        assert 'total_tensors_analyzed' in stats


class TestPhaseAwareMemoryManager:
    """Test phase-aware memory management."""
    
    def test_initialization(self):
        """Test manager initializes correctly."""
        mgr = PhaseAwareMemoryManager(32000.0)  # 32 GB
        assert mgr.total_memory_mb == 32000.0
        assert mgr.current_phase == ExecutionPhase.UNKNOWN
    
    def test_llm_prefill_budgets(self):
        """Test budget allocation for LLM prefill."""
        mgr = PhaseAwareMemoryManager(1000.0)
        mgr.adjust_for_phase(ExecutionPhase.LLM_PREFILL)
        
        # Prefill: 30% weights, 60% activations, 10% KV
        assert mgr.budgets['weights'] == 300.0
        assert mgr.budgets['activations'] == 600.0
        assert mgr.budgets['kv_cache'] == 100.0
    
    def test_llm_decode_budgets(self):
        """Test budget allocation for LLM decode."""
        mgr = PhaseAwareMemoryManager(1000.0)
        mgr.adjust_for_phase(ExecutionPhase.LLM_DECODE)
        
        # Decode: 30% weights, 10% activations, 60% KV
        assert mgr.budgets['weights'] == 300.0
        assert mgr.budgets['activations'] == 100.0
        assert mgr.budgets['kv_cache'] == 600.0
    
    def test_vision_encoding_budgets(self):
        """Test budget allocation for vision encoding."""
        mgr = PhaseAwareMemoryManager(1000.0)
        mgr.adjust_for_phase(ExecutionPhase.VISION_ENCODING)
        
        # Vision: 40% weights, 60% activations, 0% KV
        assert mgr.budgets['weights'] == 400.0
        assert mgr.budgets['activations'] == 600.0
        assert mgr.budgets['kv_cache'] == 0.0
    
    def test_check_allocation(self):
        """Test allocation checking."""
        mgr = PhaseAwareMemoryManager(1000.0)
        mgr.adjust_for_phase(ExecutionPhase.LLM_DECODE)
        
        # Should fit in budget
        assert mgr.check_allocation('weights', 300.0) is True
        
        # Should exceed budget
        assert mgr.check_allocation('weights', 400.0) is False
    
    def test_eviction_priority_order_decode(self):
        """Test eviction priority order for decode."""
        mgr = PhaseAwareMemoryManager(1000.0)
        mgr.adjust_for_phase(ExecutionPhase.LLM_DECODE)
        
        priority = mgr.get_eviction_priority_order()
        
        # Decode: evict activations first, KV cache last
        assert priority[0] == 'activations'
        assert priority[-1] == 'kv_cache'
    
    def test_eviction_priority_order_prefill(self):
        """Test eviction priority order for prefill."""
        mgr = PhaseAwareMemoryManager(1000.0)
        mgr.adjust_for_phase(ExecutionPhase.LLM_PREFILL)
        
        priority = mgr.get_eviction_priority_order()
        
        # Prefill: evict KV cache first, activations last
        assert priority[0] == 'kv_cache'
        assert priority[-1] == 'activations'
    
    def test_phase_switching(self):
        """Test phase switching and statistics."""
        mgr = PhaseAwareMemoryManager(1000.0)
        
        mgr.adjust_for_phase(ExecutionPhase.LLM_PREFILL)
        mgr.adjust_for_phase(ExecutionPhase.LLM_DECODE)
        mgr.adjust_for_phase(ExecutionPhase.LLM_DECODE)  # No change
        
        stats = mgr.get_stats()
        # Should have 2 switches (UNKNOWN→PREFILL, PREFILL→DECODE)
        assert stats['phase_switches'] == 2


class TestRecomputationVsStorageDecider:
    """Test recomputation vs storage decisions."""
    
    def test_initialization(self):
        """Test decider initializes correctly."""
        decider = RecomputationVsStorageDecider(10.0)
        assert decider.network_bandwidth_gbps == 10.0
    
    def test_multi_consumer_always_cache(self):
        """Test that multi-consumer tensors are always cached."""
        decider = RecomputationVsStorageDecider(10.0)
        
        # This should definitely cache (multi-consumer)
        should_cache = decider.should_cache_tensor(
            tensor_size_mb=100.0,
            flop_cost=1000,
            num_consumers=2  # Multi-consumer key
        )
        assert should_cache is True
    
    def test_expensive_recomputation_cache(self):
        """Test that multi-consumer tensors are always cached."""
        decider = RecomputationVsStorageDecider(10.0)
        
        # This should definitely cache (multi-consumer)
        should_cache = decider.should_cache_tensor(
            tensor_size_mb=100.0,
            flop_cost=1000,
            num_consumers=2  # Multi-consumer key
        )
        assert should_cache is True
    
    def test_cheap_recomputation_discard(self):
        """Test discarding cheap-to-recompute tensors."""
        decider = RecomputationVsStorageDecider(10.0)
        
        # Cheap recomputation: should discard
        should_cache = decider.should_cache_tensor(
            tensor_size_mb=100.0,
            flop_cost=1000,  # 1000 FLOPs (very cheap)
            num_consumers=1,
            gpu_tflops=10.0
        )
        assert should_cache is False
    
    def test_statistics_tracking(self):
        """Test statistics tracking."""
        decider = RecomputationVsStorageDecider(10.0)
        
        # Multi-consumer should cache
        decider.should_cache_tensor(100.0, 1000, 2, 10.0)  # Cache (multi-consumer)
        
        # Cheap recomputation with single consumer should recompute
        decider.should_cache_tensor(100.0, 1000, 1, 10.0)  # Recompute
        
        stats = decider.get_stats()
        assert stats['store_decisions'] == 1  # From multi-consumer
        assert stats['recompute_decisions'] == 1  # From cheap single-use


class TestSemanticMemoryIntegration:
    """Integration tests combining Phase 2 components."""
    
    def test_lifetime_and_phase_awareness(self):
        """Test using lifetime analysis with phase awareness."""
        evictor = LifetimeBasedEvictor()
        mgr = PhaseAwareMemoryManager(1000.0)
        
        # Analyze graph
        srg_nodes = [
            {'id': 'n0', 'operation': 'input', 'metadata': {'phase': 'llm_prefill'}},
            {'id': 'n1', 'operation': 'attn', 'metadata': {'phase': 'llm_prefill'}},
            {'id': 'n2', 'operation': 'decode', 'metadata': {'phase': 'llm_decode'}},
        ]
        
        srg_edges = [
            {'source_id': 'n0', 'target_id': 'n1', 'tensor_id': 't1'},
            {'source_id': 'n1', 'target_id': 'n2', 'tensor_id': 't2'},
        ]
        
        evictor.analyze_graph_lifetimes(srg_nodes, srg_edges)
        
        # Use phase-aware budgets
        mgr.adjust_for_phase(ExecutionPhase.LLM_PREFILL)
        assert mgr.budgets['activations'] == 600.0
        
        mgr.adjust_for_phase(ExecutionPhase.LLM_DECODE)
        assert mgr.budgets['activations'] == 100.0
    
    def test_recomputation_with_budget_constraints(self):
        """Test recomputation decisions under budget constraints."""
        decider = RecomputationVsStorageDecider(10.0)
        mgr = PhaseAwareMemoryManager(1000.0)
        
        mgr.adjust_for_phase(ExecutionPhase.LLM_DECODE)
        
        # Check if large activation fits in decode activation budget (100 MB)
        fits = mgr.check_allocation('activations', 50.0)
        assert fits is True
        
        # Multi-consumer tensors should always be cached
        # regardless of memory budget constraints
        cached = decider.should_cache_tensor(
            tensor_size_mb=60.0,
            flop_cost=1000,
            num_consumers=3  # Multi-consumer → always cache
        )
        assert cached is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
