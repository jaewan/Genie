"""
Adaptive Budget Tuner for Phase 3.

Learns optimal phase-specific memory budgets from execution history.

Key insight: Static budget allocations (e.g., 60% activations during prefill)
work well for the "typical" workload, but workloads vary:
- Some models need more activation memory
- Others are KV-heavy even during prefill
- Budget needs may change with model size/batch size

Solution: Track memory utilization and cache hit rates per phase, then
adaptively adjust budgets to maximize efficiency.

Algorithm:
1. Track observed utilization per category per phase
2. Calculate efficiency scores (cache hits, evictions avoided)
3. Suggest budget adjustments that improve efficiency
4. Gradually shift budgets towards optimal allocation
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BudgetObservation:
    """Single observation of memory utilization."""
    phase: str
    weights_utilization_percent: float
    activations_utilization_percent: float
    kv_cache_utilization_percent: float
    cache_hit_rate_percent: float
    evictions_this_phase: int


@dataclass
class PhaseStatistics:
    """Statistics for a single execution phase."""
    phase_name: str
    observations: List[BudgetObservation] = field(default_factory=list)
    total_observations: int = 0
    avg_weights_utilization: float = 0.0
    avg_activations_utilization: float = 0.0
    avg_kv_utilization: float = 0.0
    avg_cache_hit_rate: float = 0.0
    total_evictions: int = 0
    efficiency_score: float = 0.0  # 0-100, higher is better


class AdaptiveBudgetTuner:
    """
    Learns optimal phase-specific memory budgets from observations.
    
    Gradually adjusts budgets towards measured utilization patterns.
    """
    
    def __init__(
        self,
        initial_budgets: Dict[str, Dict[str, float]],
        learning_rate: float = 0.1,
        min_observations_per_phase: int = 10
    ):
        """
        Initialize adaptive budget tuner.
        
        Args:
            initial_budgets: Dict[phase] -> Dict[category] -> percentage
                Example:
                {
                    'llm_prefill': {'weights': 30, 'activations': 60, 'kv_cache': 10},
                    'llm_decode': {'weights': 30, 'activations': 10, 'kv_cache': 60},
                }
            learning_rate: How quickly to adjust budgets (0.1 = 10% per cycle)
            min_observations_per_phase: Collect this many before making adjustments
        """
        self.initial_budgets = initial_budgets
        self.current_budgets = {phase: dict(alloc) for phase, alloc in initial_budgets.items()}
        self.learning_rate = learning_rate
        self.min_observations = min_observations_per_phase
        
        # Track observations per phase
        self.phase_stats: Dict[str, PhaseStatistics] = {}
        for phase in initial_budgets.keys():
            self.phase_stats[phase] = PhaseStatistics(phase_name=phase)
        
        logger.info(
            "AdaptiveBudgetTuner initialized with %d phases (learning_rate=%.1f%%)",
            len(initial_budgets), learning_rate * 100
        )
    
    def record_observation(
        self,
        phase: str,
        weights_utilization: float,
        activations_utilization: float,
        kv_utilization: float,
        cache_hit_rate: float,
        evictions: int
    ) -> None:
        """
        Record a memory utilization observation.
        
        Args:
            phase: Current execution phase
            weights_utilization: Weight memory utilization (0-100)
            activations_utilization: Activation memory utilization (0-100)
            kv_utilization: KV cache memory utilization (0-100)
            cache_hit_rate: Current cache hit rate (0-100)
            evictions: Number of evictions in this phase
        """
        if phase not in self.phase_stats:
            logger.warning("Unknown phase: %s", phase)
            return
        
        # Create observation
        obs = BudgetObservation(
            phase=phase,
            weights_utilization_percent=weights_utilization,
            activations_utilization_percent=activations_utilization,
            kv_cache_utilization_percent=kv_utilization,
            cache_hit_rate_percent=cache_hit_rate,
            evictions_this_phase=evictions
        )
        
        # Record observation
        stats = self.phase_stats[phase]
        stats.observations.append(obs)
        stats.total_observations += 1
        
        # Update running averages
        n = stats.total_observations
        stats.avg_weights_utilization = (
            (stats.avg_weights_utilization * (n - 1) + weights_utilization) / n
        )
        stats.avg_activations_utilization = (
            (stats.avg_activations_utilization * (n - 1) + activations_utilization) / n
        )
        stats.avg_kv_utilization = (
            (stats.avg_kv_utilization * (n - 1) + kv_utilization) / n
        )
        stats.avg_cache_hit_rate = (
            (stats.avg_cache_hit_rate * (n - 1) + cache_hit_rate) / n
        )
        stats.total_evictions += evictions
        
        # Recalculate efficiency score
        self._update_efficiency_score(phase)
        
        # Check if we should adapt budgets
        if stats.total_observations >= self.min_observations:
            self._suggest_budget_adjustments(phase)
    
    def _update_efficiency_score(self, phase: str) -> None:
        """Calculate efficiency score for a phase."""
        stats = self.phase_stats[phase]
        
        # Efficiency based on:
        # 1. High cache hit rate (good)
        # 2. Low memory utilization (good - not bumping against limits)
        # 3. Few evictions (good)
        
        hit_rate_score = stats.avg_cache_hit_rate  # 0-100
        
        # Utilization score: penalize both under and over utilization
        # Target: 70% utilization (good balance)
        avg_utilization = (
            stats.avg_weights_utilization +
            stats.avg_activations_utilization +
            stats.avg_kv_utilization
        ) / 3
        
        # Ideal is 70%, penalize deviation
        utilization_score = 100 - abs(avg_utilization - 70) / 0.7
        utilization_score = max(0, min(100, utilization_score))
        
        # Eviction penalty: each eviction reduces score
        eviction_penalty = min(50, stats.total_evictions * 5)
        
        # Combined score (weighted)
        efficiency = (
            hit_rate_score * 0.5 +      # Cache hit rate is most important
            utilization_score * 0.3 +   # Memory utilization is important
            (100 - eviction_penalty) * 0.2  # Evictions matter less
        )
        
        stats.efficiency_score = max(0, min(100, efficiency))
        logger.debug(
            "Phase %s efficiency: %.1f%% (hit_rate=%.1f%%, util=%.1f%%, evictions=%d)",
            phase, stats.efficiency_score, stats.avg_cache_hit_rate,
            avg_utilization, stats.total_evictions
        )
    
    def _suggest_budget_adjustments(self, phase: str) -> None:
        """Suggest and apply budget adjustments for a phase."""
        stats = self.phase_stats[phase]
        
        # If efficiency is already good (>80%), don't change much
        if stats.efficiency_score > 80:
            logger.debug("Phase %s already efficient (%.1f%%)", phase, stats.efficiency_score)
            return
        
        # Identify over/under-utilized categories
        weights_util = stats.avg_weights_utilization
        activations_util = stats.avg_activations_utilization
        kv_util = stats.avg_kv_utilization
        
        # Calculate adjustment: move budget towards high utilization categories
        total_util = weights_util + activations_util + kv_util
        if total_util == 0:
            return
        
        # Target allocation based on observed utilization
        target_weights_pct = (weights_util / total_util) * 100
        target_activations_pct = (activations_util / total_util) * 100
        target_kv_pct = (kv_util / total_util) * 100
        
        # Current budgets
        current = self.current_budgets[phase]
        
        # Adjust towards target with learning rate
        lr = self.learning_rate
        new_budgets = {
            'weights': current.get('weights', 0) * (1 - lr) + target_weights_pct * lr,
            'activations': current.get('activations', 0) * (1 - lr) + target_activations_pct * lr,
            'kv_cache': current.get('kv_cache', 0) * (1 - lr) + target_kv_pct * lr,
        }
        
        # Normalize to sum to 100
        total = sum(new_budgets.values())
        if total > 0:
            new_budgets = {k: (v / total) * 100 for k, v in new_budgets.items()}
        
        # Check if change is significant
        old_budgets = current
        weights_change = abs(new_budgets['weights'] - old_budgets.get('weights', 0))
        if weights_change > 1.0:  # Only update if change > 1%
            logger.info(
                "Adapting budget for %s: weights %.1f%%→%.1f%%, activations %.1f%%→%.1f%%, kv %.1f%%→%.1f%%",
                phase,
                old_budgets.get('weights', 0), new_budgets['weights'],
                old_budgets.get('activations', 0), new_budgets['activations'],
                old_budgets.get('kv_cache', 0), new_budgets['kv_cache']
            )
            self.current_budgets[phase] = new_budgets
    
    def get_current_budgets(self) -> Dict[str, Dict[str, float]]:
        """Get current (possibly adapted) budgets."""
        return {phase: dict(alloc) for phase, alloc in self.current_budgets.items()}
    
    def get_phase_statistics(self) -> Dict[str, dict]:
        """Get detailed statistics per phase."""
        result = {}
        for phase, stats in self.phase_stats.items():
            result[phase] = {
                'observations': stats.total_observations,
                'avg_weights_utilization': stats.avg_weights_utilization,
                'avg_activations_utilization': stats.avg_activations_utilization,
                'avg_kv_utilization': stats.avg_kv_utilization,
                'avg_cache_hit_rate': stats.avg_cache_hit_rate,
                'total_evictions': stats.total_evictions,
                'efficiency_score': stats.efficiency_score,
                'current_budgets': self.current_budgets[phase],
            }
        return result
    
    def reset_phase_observations(self, phase: str) -> None:
        """Reset observations for a phase (e.g., at start of new workload)."""
        if phase in self.phase_stats:
            self.phase_stats[phase] = PhaseStatistics(phase_name=phase)
            logger.info("Reset observations for phase: %s", phase)
    
    def reset_all_observations(self) -> None:
        """Reset all observations and revert to initial budgets."""
        for phase in self.phase_stats:
            self.phase_stats[phase] = PhaseStatistics(phase_name=phase)
            self.current_budgets[phase] = dict(self.initial_budgets[phase])
        logger.info("Reset all observations and reverted to initial budgets")
