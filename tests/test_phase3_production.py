"""
Comprehensive Tests for Phase 3 Production Hardening.

Tests for:
- Prometheus metrics collection
- Memory pressure handling
- Adaptive budget tuning
- Health checks and diagnostics
"""

import asyncio
import pytest

from genie.server.memory_metrics import MetricsCollector, get_metrics
from genie.server.memory_pressure_handler import MemoryPressureHandler, MemoryPressureEvent
from genie.server.adaptive_budget_tuner import AdaptiveBudgetTuner


# =============================================================================
# PROMETHEUS METRICS TESTS
# =============================================================================

class TestMetricsCollector:
    """Test Prometheus metrics collection."""
    
    def test_metrics_initialization(self):
        """Test metrics collector initializes."""
        metrics = MetricsCollector()
        assert metrics.prometheus_enabled is not None
    
    def test_record_cache_operations(self):
        """Test recording cache operations."""
        metrics = MetricsCollector()
        
        # Record hits and misses
        metrics.record_cache_hit("model_1")
        metrics.record_cache_miss("model_1")
        metrics.record_cache_hit("model_2")
        
        # Should not raise
        assert True
    
    def test_record_cache_eviction(self):
        """Test recording cache evictions."""
        metrics = MetricsCollector()
        
        metrics.record_cache_eviction(
            reason='lru_count',
            freed_bytes=1024 * 1024,  # 1 MB
            latency_ms=5.0
        )
        
        # Should not raise
        assert True
    
    def test_set_cache_memory(self):
        """Test setting cache memory."""
        metrics = MetricsCollector()
        
        metrics.set_cache_memory("model_1", 512 * 1024)
        metrics.set_models_loaded(3)
        metrics.set_cache_hit_rate(85.5)
        
        # Should not raise
        assert True
    
    def test_record_kv_session_operations(self):
        """Test recording KV session operations."""
        metrics = MetricsCollector()
        
        metrics.record_session_created()
        metrics.set_active_sessions(5)
        metrics.set_pinned_kv_bytes(100 * 1024 * 1024)
        metrics.record_session_update()
        metrics.record_session_closed(lifetime_seconds=30.5)
        
        # Should not raise
        assert True
    
    def test_record_lifetime_evictions(self):
        """Test recording lifetime-based evictions."""
        metrics = MetricsCollector()
        
        metrics.record_lifetime_analysis()
        metrics.record_early_eviction(memory_saved_bytes=50 * 1024 * 1024)
        
        # Should not raise
        assert True
    
    def test_record_phase_switches(self):
        """Test recording phase transitions."""
        metrics = MetricsCollector()
        
        metrics.record_phase_switch('llm_prefill', 'llm_decode')
        metrics.set_phase_memory_allocated('llm_decode', 'kv_cache', 600 * 1024 * 1024)
        metrics.set_phase_utilization('llm_decode', 'kv_cache', 45.0)
        
        # Should not raise
        assert True
    
    def test_record_recomputation_decisions(self):
        """Test recording recomputation decisions."""
        metrics = MetricsCollector()
        
        metrics.record_recompute_decision(latency_us=123.5)
        metrics.record_storage_decision(latency_us=45.2)
        
        # Should not raise
        assert True
    
    def test_record_memory_pressure(self):
        """Test recording memory pressure events."""
        metrics = MetricsCollector()
        
        metrics.record_memory_pressure_event('warning')
        metrics.record_pressure_recovery(
            memory_reclaimed_bytes=200 * 1024 * 1024,
            latency_ms=50.0
        )
        metrics.set_memory_utilization(percent=85.0, available_bytes=512 * 1024 * 1024)
        
        # Should not raise
        assert True
    
    def test_record_budget_updates(self):
        """Test recording budget adaptations."""
        metrics = MetricsCollector()
        
        metrics.record_budget_update('llm_prefill', 'activations')
        metrics.set_budget_efficiency('llm_prefill', 85.5)
        metrics.set_learned_budget('llm_prefill', 'activations', 55.0)
        
        # Should not raise
        assert True


# =============================================================================
# MEMORY PRESSURE HANDLER TESTS
# =============================================================================

class TestMemoryPressureHandler:
    """Test memory pressure detection and handling."""
    
    def test_handler_initialization(self):
        """Test handler initializes correctly."""
        handler = MemoryPressureHandler(
            total_gpu_memory_mb=32000,
            warning_threshold_percent=80.0,
            critical_threshold_percent=95.0
        )
        
        assert handler.total_memory_mb == 32000
        assert handler.warning_threshold == 0.8
        assert handler.critical_threshold == 0.95
        assert handler.is_under_pressure is False
    
    def test_register_eviction_callback(self):
        """Test registering eviction callbacks."""
        handler = MemoryPressureHandler(32000)
        
        def dummy_eviction(severity):
            return 100 * 1024 * 1024  # Return 100 MB
        
        handler.register_eviction_callback('cache', dummy_eviction)
        assert 'cache' in handler.eviction_callbacks
    
    @pytest.mark.asyncio
    async def test_update_memory_status(self):
        """Test updating memory status."""
        handler = MemoryPressureHandler(32000)
        
        total_bytes = 32000 * 1024 * 1024
        available_bytes = int(0.5 * total_bytes)  # 50% available
        
        await handler.update_memory_status(available_bytes, total_bytes)
        
        # Utilization should be 50%
        assert 0.49 < handler.current_utilization < 0.51
    
    @pytest.mark.asyncio
    async def test_oom_event_handling(self):
        """Test OOM event handling."""
        handler = MemoryPressureHandler(32000)
        
        freed_bytes = 100 * 1024 * 1024
        def oom_eviction(severity):
            return freed_bytes
        
        handler.register_eviction_callback('cache', oom_eviction)
        
        success = await handler.handle_oom_event()
        
        assert success is True
        assert handler.stats['oom_events'] == 1
        assert handler.stats['total_memory_freed'] == freed_bytes
    
    @pytest.mark.asyncio
    async def test_pressure_recovery(self):
        """Test recovery from memory pressure."""
        handler = MemoryPressureHandler(32000)
        
        # Start under pressure
        handler.is_under_pressure = True
        
        # Update to normal memory
        total_bytes = 32000 * 1024 * 1024
        available_bytes = int(0.95 * total_bytes)  # 95% available = 5% utilization
        
        await handler.update_memory_status(available_bytes, total_bytes)
        
        # Should recover
        assert handler.is_under_pressure is False
    
    def test_prefer_recomputation_under_pressure(self):
        """Test recomputation preference under pressure."""
        handler = MemoryPressureHandler(32000)
        
        # Normal operation
        assert handler.should_prefer_recomputation() is False
        assert handler.get_adaptive_recomputation_threshold() == 1.0
        
        # Critical pressure
        handler.pressure_recovery_mode = True
        assert handler.should_prefer_recomputation() is True
        assert handler.get_adaptive_recomputation_threshold() == 0.5
        
        # Warning pressure
        handler.pressure_recovery_mode = False
        handler.is_under_pressure = True
        assert handler.get_adaptive_recomputation_threshold() == 0.8
    
    def test_pressure_statistics(self):
        """Test pressure statistics."""
        handler = MemoryPressureHandler(32000)
        
        stats = handler.get_pressure_statistics()
        
        assert 'warning_events' in stats
        assert 'critical_events' in stats
        assert 'oom_events' in stats
        assert stats['warning_events'] == 0


# =============================================================================
# ADAPTIVE BUDGET TUNER TESTS
# =============================================================================

class TestAdaptiveBudgetTuner:
    """Test adaptive budget learning."""
    
    def test_tuner_initialization(self):
        """Test tuner initializes with budgets."""
        initial_budgets = {
            'llm_prefill': {'weights': 30, 'activations': 60, 'kv_cache': 10},
            'llm_decode': {'weights': 30, 'activations': 10, 'kv_cache': 60},
        }
        
        tuner = AdaptiveBudgetTuner(initial_budgets, learning_rate=0.1)
        
        assert tuner.current_budgets == initial_budgets
        assert tuner.learning_rate == 0.1
    
    def test_record_observation(self):
        """Test recording utilization observations."""
        initial_budgets = {
            'llm_prefill': {'weights': 30, 'activations': 60, 'kv_cache': 10},
        }
        
        tuner = AdaptiveBudgetTuner(initial_budgets)
        
        # Record observation
        tuner.record_observation(
            phase='llm_prefill',
            weights_utilization=25.0,
            activations_utilization=55.0,
            kv_utilization=8.0,
            cache_hit_rate=75.0,
            evictions=2
        )
        
        stats = tuner.phase_stats['llm_prefill']
        assert stats.total_observations == 1
        assert stats.avg_weights_utilization == 25.0
    
    def test_efficiency_scoring(self):
        """Test efficiency score calculation."""
        initial_budgets = {
            'llm_prefill': {'weights': 30, 'activations': 60, 'kv_cache': 10},
        }
        
        tuner = AdaptiveBudgetTuner(initial_budgets)
        
        # Perfect utilization (70% average) and high hit rate
        for _ in range(15):
            tuner.record_observation(
                phase='llm_prefill',
                weights_utilization=30.0,
                activations_utilization=40.0,
                kv_utilization=0.0,
                cache_hit_rate=95.0,
                evictions=0
            )
        
        stats = tuner.phase_stats['llm_prefill']
        # With 70% avg utilization and 95% hit rate, expect ~77.5% efficiency
        assert stats.efficiency_score > 75
    
    def test_budget_adaptation(self):
        """Test budget gets adapted based on utilization."""
        initial_budgets = {
            'llm_prefill': {'weights': 30, 'activations': 60, 'kv_cache': 10},
        }
        
        tuner = AdaptiveBudgetTuner(initial_budgets, learning_rate=0.2)
        
        # Record observations showing high activation utilization
        for _ in range(15):
            tuner.record_observation(
                phase='llm_prefill',
                weights_utilization=10.0,
                activations_utilization=80.0,
                kv_utilization=5.0,
                cache_hit_rate=50.0,
                evictions=5
            )
        
        # Budget should have shifted towards activations
        budgets = tuner.get_current_budgets()['llm_prefill']
        assert budgets['activations'] > 60  # Should increase
        assert budgets['weights'] < 30  # Should decrease
    
    def test_get_phase_statistics(self):
        """Test retrieving phase statistics."""
        initial_budgets = {
            'llm_prefill': {'weights': 30, 'activations': 60, 'kv_cache': 10},
            'llm_decode': {'weights': 30, 'activations': 10, 'kv_cache': 60},
        }
        
        tuner = AdaptiveBudgetTuner(initial_budgets)
        
        tuner.record_observation(
            phase='llm_prefill',
            weights_utilization=25.0,
            activations_utilization=55.0,
            kv_utilization=8.0,
            cache_hit_rate=75.0,
            evictions=1
        )
        
        stats = tuner.get_phase_statistics()
        
        assert 'llm_prefill' in stats
        assert 'llm_decode' in stats
        assert stats['llm_prefill']['observations'] == 1
        assert stats['llm_prefill']['avg_cache_hit_rate'] == 75.0
    
    def test_reset_observations(self):
        """Test resetting observations."""
        initial_budgets = {
            'llm_prefill': {'weights': 30, 'activations': 60, 'kv_cache': 10},
        }
        
        tuner = AdaptiveBudgetTuner(initial_budgets)
        
        # Record some observations
        tuner.record_observation(
            phase='llm_prefill',
            weights_utilization=25.0,
            activations_utilization=55.0,
            kv_utilization=8.0,
            cache_hit_rate=75.0,
            evictions=1
        )
        
        # Reset
        tuner.reset_phase_observations('llm_prefill')
        
        stats = tuner.phase_stats['llm_prefill']
        assert stats.total_observations == 0
        assert stats.avg_cache_hit_rate == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
