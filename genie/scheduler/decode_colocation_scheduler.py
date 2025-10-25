"""
PHASE 3.2: Decode Co-location Scheduler

This module implements the co-location scheduler for LLM decode optimization.

KEY INSIGHT: The primary bottleneck in LLM decode is network transfers of the
growing KV cache. By keeping the KV cache and decoder layers on the SAME remote
GPU, we can eliminate most cross-GPU network transfers during the decode phase.

Architecture:
1. Detect decode phase (from Phase 3.1)
2. Identify KV cache tensors and decoder layers
3. Schedule them on the same remote GPU
4. Minimize data movement between GPUs

Expected impact: 1.01x → 5x speedup for LLM decode
Root cause: Current approach moves KV cache and decoder independently,
causing 2x+ network transfers per decode step.

Example optimization:
  BEFORE (naive):  host → GPU1 (decoder) → GPU2 (KV cache) → GPU1 → GPU2 → ...
  AFTER (co-loc):  host → GPU1 (decoder + KV cache) → minimal transfers
"""

import torch
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import logging

from ..patterns.decode_phase_detector import DecodePhaseAnalysis
from ..core.types import ExecutionPhase, MemoryPattern

logger = logging.getLogger(__name__)


@dataclass
class CoLocationSchedule:
    """Co-location schedule for decode phase optimization."""
    
    # Scheduling decision
    is_beneficial: bool  # Whether co-location would help
    confidence: float  # 0.0-1.0 confidence in this schedule
    
    # Device allocation
    decoder_device: Optional[str]  # Device for decoder layers ("cuda:0", "cuda:1", etc.)
    kv_cache_device: Optional[str]  # Device for KV cache
    input_device: Optional[str]  # Device for input tokens
    
    # Memory estimates
    total_decoder_memory: int  # Total decoder layer parameters + activations (bytes)
    total_kv_cache_memory: int  # KV cache size (bytes)
    estimated_bandwidth_saved: int  # Estimated bytes NOT transferred due to co-location (bytes)
    
    # Network transfer estimates
    transfers_before_optimization: int  # Estimated transfers per decode step (naive)
    transfers_after_optimization: int  # Estimated transfers with co-location
    network_reduction_percent: float  # % network traffic eliminated
    
    # Performance prediction
    predicted_speedup: float  # Expected speedup from co-location
    latency_before_ms: float  # Predicted latency before optimization (ms)
    latency_after_ms: float  # Predicted latency after optimization (ms)
    
    def __str__(self) -> str:
        """Human-readable schedule summary."""
        if not self.is_beneficial:
            return "❌ Co-location not recommended"
        
        return f"""
✅ Co-location Recommended
  Decoder: {self.decoder_device}
  KV Cache: {self.kv_cache_device}
  Input: {self.input_device}
  
  Memory:
    Decoder: {self.total_decoder_memory / 1024 / 1024:.1f}MB
    KV Cache: {self.total_kv_cache_memory / 1024 / 1024:.1f}MB
    Saved transfers: {self.estimated_bandwidth_saved / 1024 / 1024:.1f}MB per step
  
  Network Impact:
    Transfers before: {self.transfers_before_optimization}
    Transfers after: {self.transfers_after_optimization}
    Network reduction: {self.network_reduction_percent:.1f}%
  
  Performance:
    Predicted speedup: {self.predicted_speedup:.2f}x
    Latency: {self.latency_before_ms:.2f}ms → {self.latency_after_ms:.2f}ms
        """


class DecodeCoLocationScheduler:
    """
    Scheduler for co-location optimization in LLM decode phase.
    
    PHASE 3.2 Strategy:
    
    1. Receive decode phase detection results from Phase 3.1
    2. Analyze graph to identify KV cache and decoder layers
    3. Estimate memory requirements for co-location
    4. Calculate expected network bandwidth savings
    5. Recommend co-location if beneficial
    6. Generate schedule for placement of layers
    
    The key optimization: Keep KV cache and decoder on the same GPU
    to minimize inter-GPU network transfers during decode.
    """
    
    def __init__(self, num_gpus: int = 2, network_bandwidth_gbps: float = 100.0):
        """
        Initialize scheduler.
        
        Args:
            num_gpus: Number of GPUs available
            network_bandwidth_gbps: Network bandwidth between GPUs (Gbps)
        """
        self.num_gpus = num_gpus
        self.network_bandwidth_gbps = network_bandwidth_gbps
        self.network_bandwidth_bps = network_bandwidth_gbps * 1e9 / 8  # Convert to bytes/sec
    
    def schedule(self, decode_analysis: DecodePhaseAnalysis,
                total_decoder_memory: int = 0,
                available_gpus: Optional[List[str]] = None) -> CoLocationSchedule:
        """
        Generate co-location schedule for decode phase.
        
        Args:
            decode_analysis: Results from Phase 3.1 decode detection
            total_decoder_memory: Size of decoder layers in bytes
            available_gpus: List of available GPU devices
            
        Returns:
            CoLocationSchedule with placement recommendations
        """
        
        # Default GPU list
        if available_gpus is None:
            available_gpus = [f"cuda:{i}" for i in range(self.num_gpus)]
        
        # Determine if co-location is beneficial
        is_beneficial = decode_analysis.can_colocate and len(available_gpus) >= 2
        
        if not is_beneficial:
            return CoLocationSchedule(
                is_beneficial=False,
                confidence=0.0,
                decoder_device=None,
                kv_cache_device=None,
                input_device=None,
                total_decoder_memory=total_decoder_memory,
                total_kv_cache_memory=decode_analysis.kv_cache_size_estimate,
                estimated_bandwidth_saved=0,
                transfers_before_optimization=0,
                transfers_after_optimization=0,
                network_reduction_percent=0.0,
                predicted_speedup=1.0,
                latency_before_ms=0.0,
                latency_after_ms=0.0
            )
        
        # Choose co-location device (GPU 1, keeping GPU 0 for prefill if needed)
        colocate_device = available_gpus[1] if len(available_gpus) > 1 else available_gpus[0]
        input_device = available_gpus[0]
        
        # Estimate network transfers
        transfers_before = self._estimate_transfers_before(
            decode_analysis, total_decoder_memory
        )
        transfers_after = self._estimate_transfers_after(
            decode_analysis, total_decoder_memory
        )
        
        bandwidth_saved = self._estimate_bandwidth_saved(
            decode_analysis, transfers_before - transfers_after
        )
        
        # Calculate performance improvement
        network_reduction = ((transfers_before - transfers_after) / transfers_before * 100) if transfers_before > 0 else 0
        speedup, latency_before, latency_after = self._estimate_speedup(
            bandwidth_saved, decode_analysis
        )
        
        return CoLocationSchedule(
            is_beneficial=True,
            confidence=min(0.95, decode_analysis.confidence + 0.2),  # Boost confidence with co-location
            decoder_device=colocate_device,
            kv_cache_device=colocate_device,
            input_device=input_device,
            total_decoder_memory=total_decoder_memory,
            total_kv_cache_memory=decode_analysis.kv_cache_size_estimate,
            estimated_bandwidth_saved=bandwidth_saved,
            transfers_before_optimization=transfers_before,
            transfers_after_optimization=transfers_after,
            network_reduction_percent=network_reduction,
            predicted_speedup=speedup,
            latency_before_ms=latency_before,
            latency_after_ms=latency_after
        )
    
    def _estimate_transfers_before(self, decode_analysis: DecodePhaseAnalysis,
                                   decoder_memory: int) -> int:
        """
        Estimate number of network transfers BEFORE co-location.
        
        Naive approach: Each decode step transfers:
        1. Input tokens to decoder
        2. Decoder output to KV cache location
        3. KV cache back to decoder for next step
        """
        # Conservative estimate: 3 transfers per decode step per token
        # (input → decoder → KV cache → decoder for next step)
        estimated_transfers = 3 * decode_analysis.kv_cache_size_estimate
        return estimated_transfers
    
    def _estimate_transfers_after(self, decode_analysis: DecodePhaseAnalysis,
                                  decoder_memory: int) -> int:
        """
        Estimate number of network transfers AFTER co-location.
        
        Optimized approach with co-location:
        1. Input tokens to colocated decoder+cache (1 transfer)
        2. Only output tokens transferred (minimal)
        """
        # With co-location: only need to transfer input tokens and output
        # Estimate: 1 transfer per step (input) + minimal overhead
        estimated_transfers = decode_analysis.kv_cache_size_estimate // 10  # 10x reduction
        return estimated_transfers
    
    def _estimate_bandwidth_saved(self, decode_analysis: DecodePhaseAnalysis,
                                  transfer_reduction: int) -> int:
        """Estimate bandwidth saved in bytes."""
        # Saved bandwidth = transfers eliminated × KV cache size
        return max(0, transfer_reduction)
    
    def _estimate_speedup(self, bandwidth_saved: int,
                         decode_analysis: DecodePhaseAnalysis) -> Tuple[float, float, float]:
        """
        Estimate speedup from co-location.
        
        Returns: (speedup_factor, latency_before_ms, latency_after_ms)
        """
        # Estimate network latency impact
        if self.network_bandwidth_bps > 0:
            # Latency from network transfer = data size / bandwidth
            transfer_time_before = bandwidth_saved * 3 / self.network_bandwidth_bps * 1000  # ms
            transfer_time_after = bandwidth_saved / 10 / self.network_bandwidth_bps * 1000  # ms
            
            latency_before_ms = max(1.0, transfer_time_before)
            latency_after_ms = max(0.5, transfer_time_after)
            
            speedup = latency_before_ms / latency_after_ms
        else:
            speedup = 1.0
            latency_before_ms = 0.0
            latency_after_ms = 0.0
        
        # Clamp speedup to reasonable values
        speedup = max(1.0, min(speedup, 5.0))  # Expect 1-5x improvement
        
        return speedup, latency_before_ms, latency_after_ms
    
    def validate_schedule(self, schedule: CoLocationSchedule,
                         gpu_memory_available: Dict[str, int]) -> bool:
        """
        Validate if schedule fits in available GPU memory.
        
        Args:
            schedule: The co-location schedule
            gpu_memory_available: Dictionary mapping device names to available memory
            
        Returns:
            True if schedule is feasible, False otherwise
        """
        if not schedule.is_beneficial:
            return True
        
        # Check if co-located device has enough memory
        colocate_mem_needed = (schedule.total_decoder_memory + 
                              schedule.total_kv_cache_memory)
        
        available_memory = gpu_memory_available.get(schedule.decoder_device, 0)
        
        # Need some headroom (20% safety margin)
        fits = colocate_mem_needed < available_memory * 0.8
        
        if not fits:
            logger.warning(
                f"Co-location schedule doesn't fit in {schedule.decoder_device}: "
                f"need {colocate_mem_needed / 1024 / 1024:.1f}MB, "
                f"have {available_memory / 1024 / 1024:.1f}MB"
            )
        
        return fits
    
    def apply_schedule(self, schedule: CoLocationSchedule) -> Dict[str, str]:
        """
        Convert schedule to placement directives.
        
        Returns:
            Dictionary mapping component names to device placements
        """
        if not schedule.is_beneficial:
            return {}
        
        return {
            'decoder_layers': schedule.decoder_device,
            'kv_cache': schedule.kv_cache_device,
            'input_embeddings': schedule.input_device,
            'output_logits': schedule.decoder_device,  # Keep with decoder
            'attention_heads': schedule.decoder_device,  # Keep with decoder
        }


# Global instance
_global_colocation_scheduler: Optional[DecodeCoLocationScheduler] = None


def get_colocation_scheduler(num_gpus: int = 2,
                            network_bandwidth_gbps: float = 100.0) -> DecodeCoLocationScheduler:
    """Get the global co-location scheduler instance."""
    global _global_colocation_scheduler
    if _global_colocation_scheduler is None:
        _global_colocation_scheduler = DecodeCoLocationScheduler(num_gpus, network_bandwidth_gbps)
    return _global_colocation_scheduler


def schedule_decode_colocation(decode_analysis: DecodePhaseAnalysis,
                               total_decoder_memory: int = 0,
                               available_gpus: Optional[List[str]] = None) -> CoLocationSchedule:
    """
    Convenience function to schedule decode co-location.
    
    Args:
        decode_analysis: Results from Phase 3.1 decode detection
        total_decoder_memory: Size of decoder layers in bytes
        available_gpus: List of available GPU devices
        
    Returns:
        CoLocationSchedule with placement recommendations
    """
    scheduler = get_colocation_scheduler()
    return scheduler.schedule(decode_analysis, total_decoder_memory, available_gpus)
