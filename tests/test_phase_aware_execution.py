"""
Test suite for Phase-Aware Execution.

Tests cover:
1. Phase detection based on input characteristics
2. Strategy selection per phase
3. Execution with phase-specific optimizations
4. Memory management integration
5. Statistics tracking
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from djinn.server.optimizations.phase_executor import (
    PhaseAwareExecutor,
    PrefillExecutionStrategy,
    DecodeExecutionStrategy,
    VisionExecutionStrategy,
    ExecutionPhase,
)
from djinn.server.semantic_memory_manager import (
    PhaseAwareMemoryManager,
    LifetimeBasedEvictor,
)


class SimpleTransformer(nn.Module):
    """Simple transformer model for testing."""
    
    def __init__(self, vocab_size=50257, hidden_size=768, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_size, nhead=8, dim_feedforward=3072)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids, past_key_values=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


class SimpleVisionModel(nn.Module):
    """Simple vision model for testing."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.output = nn.Linear(128, 10)
    
    def forward(self, pixel_values):
        x = self.conv1(pixel_values)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        return x


class TestPhaseDetection:
    """Tests for phase detection logic."""
    
    def test_detect_decode_phase_with_kv_cache(self):
        """Test that decode phase is detected when KV cache is present."""
        executor = PhaseAwareExecutor()
        graph = SimpleTransformer()
        
        inputs = {
            'input_ids': torch.randint(0, 50257, (4, 1)),
            'past_key_values': [torch.randn(4, 8, 10, 64) for _ in range(4)]
        }
        
        phase = executor._detect_phase(graph, inputs)
        assert phase == ExecutionPhase.LLM_DECODE, "Should detect decode phase with KV cache"
    
    def test_detect_prefill_phase_with_long_sequence(self):
        """Test that prefill phase is detected with long sequences."""
        executor = PhaseAwareExecutor()
        graph = SimpleTransformer()
        
        inputs = {
            'input_ids': torch.randint(0, 50257, (8, 512))  # Long sequence
        }
        
        phase = executor._detect_phase(graph, inputs)
        assert phase == ExecutionPhase.LLM_PREFILL, "Should detect prefill phase with long sequence"
    
    def test_detect_vision_phase_with_pixel_values(self):
        """Test that vision phase is detected with pixel values."""
        executor = PhaseAwareExecutor()
        graph = SimpleVisionModel()
        
        inputs = {
            'pixel_values': torch.randn(4, 3, 224, 224)
        }
        
        phase = executor._detect_phase(graph, inputs)
        assert phase == ExecutionPhase.VISION_ENCODING, "Should detect vision phase with pixel values"
    
    def test_detect_forward_phase_default(self):
        """Test that forward phase is default when no specific signals."""
        executor = PhaseAwareExecutor()
        graph = SimpleTransformer()
        
        inputs = {
            'input_ids': torch.randint(0, 50257, (8, 5))  # Short sequence
        }
        
        phase = executor._detect_phase(graph, inputs)
        assert phase == ExecutionPhase.FORWARD, "Should default to forward phase"


class TestStrategySelection:
    """Tests for phase strategy selection and configuration."""
    
    def test_prefill_strategy_configuration(self):
        """Test prefill strategy configuration."""
        strategy = PrefillExecutionStrategy()
        exec_strategy = strategy.get_strategy()
        
        assert exec_strategy.phase == ExecutionPhase.LLM_PREFILL
        assert exec_strategy.batch_size == 32, "Prefill should use large batch"
        assert exec_strategy.use_mixed_precision is True, "Prefill should use mixed precision"
        assert exec_strategy.enable_fusion is True, "Prefill should enable fusion"
        assert exec_strategy.pin_kv_cache is False, "Prefill should not pin KV cache"
        assert exec_strategy.target_gpu_utilization == 0.90, "Prefill should target 90% GPU utilization"
    
    def test_decode_strategy_configuration(self):
        """Test decode strategy configuration."""
        strategy = DecodeExecutionStrategy()
        exec_strategy = strategy.get_strategy()
        
        assert exec_strategy.phase == ExecutionPhase.LLM_DECODE
        assert exec_strategy.batch_size == 4, "Decode should use small batch"
        assert exec_strategy.use_mixed_precision is False, "Decode should use full precision"
        assert exec_strategy.enable_fusion is False, "Decode should disable fusion"
        assert exec_strategy.pin_kv_cache is True, "Decode should pin KV cache"
        assert exec_strategy.incremental_compute is True, "Decode should use incremental compute"
        assert exec_strategy.target_gpu_utilization == 0.45, "Decode should target 45% GPU utilization"
    
    def test_vision_strategy_configuration(self):
        """Test vision strategy configuration."""
        strategy = VisionExecutionStrategy()
        exec_strategy = strategy.get_strategy()
        
        assert exec_strategy.phase == ExecutionPhase.VISION_ENCODING
        assert exec_strategy.batch_size == 16, "Vision should use medium batch"
        assert exec_strategy.use_mixed_precision is True, "Vision should use mixed precision"
        assert exec_strategy.enable_fusion is True, "Vision should enable fusion"
        assert exec_strategy.target_gpu_utilization == 0.85, "Vision should target 85% GPU utilization"


class TestPrefillExecution:
    """Tests for prefill-specific execution."""
    
    def test_prefill_batch_padding(self):
        """Test that prefill pads batch to target size."""
        strategy = PrefillExecutionStrategy()
        
        inputs = {
            'input_ids': torch.randint(0, 50257, (8, 512))  # Batch size 8
        }
        
        prepared = strategy.prepare_for_execution(inputs)
        
        # Should pad to batch size 32
        assert prepared['input_ids'].shape[0] == 32, f"Expected batch size 32, got {prepared['input_ids'].shape[0]}"
        assert prepared['input_ids'].shape[1] == 512, "Sequence length should be preserved"
    
    def test_prefill_batch_truncation(self):
        """Test that prefill truncates batch if too large."""
        strategy = PrefillExecutionStrategy()
        
        inputs = {
            'input_ids': torch.randint(0, 50257, (64, 512))  # Batch size 64 > target 32
        }
        
        prepared = strategy.prepare_for_execution(inputs)
        
        # Should truncate to batch size 32
        assert prepared['input_ids'].shape[0] == 32, f"Expected batch size 32, got {prepared['input_ids'].shape[0]}"


class TestDecodeExecution:
    """Tests for decode-specific execution."""
    
    def test_decode_incremental_computation(self):
        """Test that decode uses only last token."""
        strategy = DecodeExecutionStrategy()
        
        inputs = {
            'input_ids': torch.randint(0, 50257, (4, 512))  # Full sequence
        }
        
        prepared = strategy.prepare_for_execution(inputs)
        
        # Should take only last token
        assert prepared['input_ids'].shape == (4, 1), f"Expected (4, 1), got {prepared['input_ids'].shape}"
    
    def test_decode_kv_cache_pinning(self):
        """Test that decode pins KV cache."""
        strategy = DecodeExecutionStrategy()
        
        kv_cache = torch.randn(4, 8, 10, 64)
        strategy.pin_kv_cache(kv_cache)
        
        assert 'kv_cache' in strategy.pinned_tensors, "KV cache should be pinned"
        assert torch.equal(strategy.pinned_tensors['kv_cache'], kv_cache), "Pinned tensor should match input"
    
    def test_decode_kv_cache_unpinning(self):
        """Test that decode unpins KV cache."""
        strategy = DecodeExecutionStrategy()
        
        kv_cache = torch.randn(4, 8, 10, 64)
        strategy.pin_kv_cache(kv_cache)
        strategy.unpin_kv_cache()
        
        assert len(strategy.pinned_tensors) == 0, "Pinned tensors should be cleared"


class TestPhaseAwareExecutor:
    """Tests for PhaseAwareExecutor integration."""
    
    def test_executor_initialization(self):
        """Test executor initialization."""
        executor = PhaseAwareExecutor(total_gpu_memory_mb=16000)
        
        assert executor.total_gpu_memory_mb == 16000
        assert len(executor.phase_strategies) == 3
        assert ExecutionPhase.LLM_PREFILL in executor.phase_strategies
        assert ExecutionPhase.LLM_DECODE in executor.phase_strategies
        assert ExecutionPhase.VISION_ENCODING in executor.phase_strategies
    
    def test_executor_phase_switching(self):
        """Test executor switches strategies on phase change."""
        executor = PhaseAwareExecutor()
        graph = SimpleTransformer()
        
        # First: Prefill
        prefill_inputs = {
            'input_ids': torch.randint(0, 50257, (8, 512))
        }
        
        result1 = executor.execute_with_phase_optimization(graph, prefill_inputs)
        assert executor.stats['current_phase'] == ExecutionPhase.LLM_PREFILL
        assert executor.stats['phase_switches'] == 1
        
        # Second: Decode (phase change)
        decode_inputs = {
            'input_ids': torch.randint(0, 50257, (4, 1)),
            'past_key_values': [torch.randn(4, 8, 10, 64) for _ in range(4)]
        }
        
        result2 = executor.execute_with_phase_optimization(graph, decode_inputs)
        assert executor.stats['current_phase'] == ExecutionPhase.LLM_DECODE
        assert executor.stats['phase_switches'] == 2
    
    def test_executor_statistics_tracking(self):
        """Test executor tracks execution statistics."""
        executor = PhaseAwareExecutor()
        graph = SimpleTransformer()
        
        inputs = {
            'input_ids': torch.randint(0, 50257, (8, 512))
        }
        
        # Execute multiple times
        for _ in range(3):
            executor.execute_with_phase_optimization(graph, inputs)
        
        stats = executor.get_stats()
        assert stats['total_executions'] == 3
        assert stats['total_time_ms'] > 0
        assert stats['avg_time_ms'] == stats['total_time_ms'] / 3
    
    def test_executor_memory_management_integration(self):
        """Test executor integrates with memory manager."""
        executor = PhaseAwareExecutor(total_gpu_memory_mb=24000)
        graph = SimpleTransformer()
        
        # Prefill phase
        prefill_inputs = {
            'input_ids': torch.randint(0, 50257, (8, 512))
        }
        
        executor.execute_with_phase_optimization(graph, prefill_inputs)
        memory_stats = executor.memory_manager.get_stats()
        
        # Should have switched to prefill budget
        assert memory_stats['current_phase'] == ExecutionPhase.LLM_PREFILL.value
        assert 'weights' in memory_stats['budgets_mb']
        assert 'activations' in memory_stats['budgets_mb']
        assert 'kv_cache' in memory_stats['budgets_mb']


class TestMemoryManagementIntegration:
    """Tests for memory management integration."""
    
    def test_phase_aware_budgets_differ(self):
        """Test that phase-aware budgets differ by phase."""
        memory_mgr = PhaseAwareMemoryManager(total_gpu_memory_mb=24000)
        
        # Prefill phase
        memory_mgr.adjust_for_phase(ExecutionPhase.LLM_PREFILL)
        prefill_budgets = dict(memory_mgr.budgets)
        
        # Decode phase
        memory_mgr.adjust_for_phase(ExecutionPhase.LLM_DECODE)
        decode_budgets = dict(memory_mgr.budgets)
        
        # Activation budget should differ (60% vs 10%)
        assert prefill_budgets['activations'] > decode_budgets['activations'], \
            "Prefill should allocate more to activations"
        
        # KV cache budget should differ (10% vs 60%)
        assert prefill_budgets['kv_cache'] < decode_budgets['kv_cache'], \
            "Decode should allocate more to KV cache"
    
    def test_lifetime_based_eviction(self):
        """Test lifetime-based eviction analysis."""
        evictor = LifetimeBasedEvictor()
        
        # Create simple SRG
        srg_nodes = [
            {'id': 'node_0', 'operation': 'input', 'metadata': {}},
            {'id': 'node_1', 'operation': 'matmul', 'metadata': {'flop_cost': 1000}},
            {'id': 'node_2', 'operation': 'output', 'metadata': {}},
        ]
        
        srg_edges = [
            {'source_id': 'node_0', 'target_id': 'node_1', 'tensor_id': 'tensor_0'},
            {'source_id': 'node_1', 'target_id': 'node_2', 'tensor_id': 'tensor_1'},
        ]
        
        evictor.analyze_graph_lifetimes(srg_nodes, srg_edges)
        
        # tensor_0 should end at node_1
        tensors_to_evict_at_1 = evictor.get_tensors_to_evict_after_node('node_1')
        assert 'tensor_0' in tensors_to_evict_at_1, "tensor_0 lifetime should end at node_1"
        
        stats = evictor.get_stats()
        assert stats['total_tensors_analyzed'] >= 2


class TestExecutionWithDifferentGraphTypes:
    """Tests for execution with different graph representations."""
    
    def test_execution_with_callable_graph(self):
        """Test execution with callable graph."""
        executor = PhaseAwareExecutor()
        
        # Simple callable
        def simple_graph(input_ids, **kwargs):
            return torch.randn(input_ids.shape)
        
        inputs = {
            'input_ids': torch.randint(0, 50257, (8, 512))
        }
        
        result = executor.execute_with_phase_optimization(simple_graph, inputs)
        # Note: Prefill phase pads batch to 32, so result will have batch_size=32
        assert result.shape[1] == inputs['input_ids'].shape[1], "Sequence length should match"
        assert result.shape[0] >= inputs['input_ids'].shape[0], "Batch might be padded in prefill"
    
    def test_execution_with_nn_module(self):
        """Test execution with nn.Module."""
        executor = PhaseAwareExecutor()
        graph = SimpleTransformer()
        
        inputs = {
            'input_ids': torch.randint(0, 50257, (8, 512))
        }
        
        result = executor.execute_with_phase_optimization(graph, inputs)
        # Note: Prefill phase pads batch to 32
        assert result.shape[1] == 512, "Sequence length should be preserved"
        assert result.shape[0] >= 8, "Batch might be padded in prefill"


@pytest.mark.parametrize("batch_size,seq_len,expected_phase", [
    (4, 1, ExecutionPhase.LLM_DECODE),  # Single token → decode
    (8, 512, ExecutionPhase.LLM_PREFILL),  # Long sequence → prefill
    (1, 5, ExecutionPhase.FORWARD),  # Short sequence → forward
])
def test_phase_detection_parametrized(batch_size, seq_len, expected_phase):
    """Parametrized test for phase detection."""
    executor = PhaseAwareExecutor()
    graph = SimpleTransformer()
    
    inputs = {
        'input_ids': torch.randint(0, 50257, (batch_size, seq_len))
    }
    
    # Add KV cache only for decode case
    if expected_phase == ExecutionPhase.LLM_DECODE:
        inputs['past_key_values'] = [torch.randn(batch_size, 8, seq_len, 64) for _ in range(4)]
    
    phase = executor._detect_phase(graph, inputs)
    assert phase == expected_phase, f"Expected {expected_phase}, got {phase}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
