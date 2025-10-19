"""
Phase detection for execution phases.

Identifies phases like:
- LLM prefill (parallel attention over sequence)
- LLM decode (sequential generation with KV cache)
- Forward propagation
- Backward propagation (future)
"""

from enum import Enum
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class ExecutionPhase(Enum):
    """Execution phase classification."""
    LLM_PREFILL = "llm_prefill"
    LLM_DECODE = "llm_decode"
    FORWARD = "forward"
    BACKWARD = "backward"
    INIT = "initialization"
    UNKNOWN = "unknown"


class PhaseDetector:
    """
    Detects execution phases in computation graphs.
    
    Uses pattern matches and heuristics to classify operations into phases.
    """
    
    def __init__(self, pattern_registry):
        self.pattern_registry = pattern_registry
    
    def detect_phases(self, graph) -> Dict[str, ExecutionPhase]:
        """
        Detect phases for all nodes in graph.
            
        Returns:
            Dict mapping node_id → ExecutionPhase
        """
        phases = {}
        patterns = self.pattern_registry.match_all(graph)
        
        # Phase 1: Detect KV cache (indicates decode)
        if "kv_cache" in patterns and patterns["kv_cache"]:
            decode_nodes = self._mark_decode_phase(patterns["kv_cache"], graph)
            phases.update(decode_nodes)
        
        # Phase 2: Detect parallel attention (indicates prefill)
        prefill_nodes = self._mark_prefill_phase(patterns.get("attention", []), graph)
        phases.update(prefill_nodes)
        
        # Phase 3: Mark remaining as forward
        for node in graph.nodes():
            if node.id not in phases:
                phases[node.id] = ExecutionPhase.FORWARD
        
        return phases
    
    def _mark_decode_phase(self, kv_cache_patterns, graph) -> Dict[str, ExecutionPhase]:
        """Mark nodes in decode phase."""
        phases = {}
        
        for pattern in kv_cache_patterns:
            for node in pattern.nodes:
                phases[node.id] = ExecutionPhase.LLM_DECODE
                
                # Also mark attention nodes connected to KV cache as decode
                # (they're part of the decode loop)
        
        return phases
    
    def _mark_prefill_phase(self, attention_patterns, graph) -> Dict[str, ExecutionPhase]:
        """Mark nodes in prefill phase."""
        phases = {}
        
        for pattern in attention_patterns:
            # Check if this is parallel attention (prefill)
            if self._is_parallel_attention(pattern):
                for node in pattern.nodes:
                    phases[node.id] = ExecutionPhase.LLM_PREFILL
        
        return phases
    
    def _is_parallel_attention(self, pattern) -> bool:
        """Check if attention processes multiple positions in parallel."""
        # Heuristic: If batch size in sequence dimension > 1
        return True  # Simplified


class PhaseAnnotator:
    """
    Annotates graph nodes with phase information.
    
    Stores phase annotations in node metadata.
    """
    
    def __init__(self, phase_detector):
        self.phase_detector = phase_detector
    
    def annotate(self, graph):
        """
        Annotate graph nodes with phase information.
        
        Modifies node metadata in-place.
        """
        phases = self.phase_detector.detect_phases(graph)
        
        for node in graph.nodes():
            phase = phases.get(node.id, ExecutionPhase.UNKNOWN)
            if not hasattr(node, 'metadata'):
                node.metadata = {}
            if 'phase' not in node.metadata:
                node.metadata['phase'] = phase


# Backward compatibility shims for existing code
def get_phase_detector():
    """Singleton phase detector (backward compatibility)."""
    # Return a minimal dummy implementation for backward compatibility
    class DummyPhaseDetector:
        def detect_phase(self, *args, **kwargs):
            return ExecutionPhase.FORWARD
    
    return DummyPhaseDetector()


class PhaseAwareHook:
    """Hook for phase detection (backward compatibility)."""
    def __init__(self, module_name=""):
        self.module_name = module_name
    
    def __call__(self, module, inputs, outputs):
        return outputs
