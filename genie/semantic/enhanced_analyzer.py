"""Enhanced semantic analyzer with TorchDynamo pattern matching.

This module integrates the TorchDynamo-based pattern matching system
with the semantic analyzer for improved pattern recognition.
"""

import time
import logging
from typing import Dict, Any, Optional
import torch.fx as fx

from ..core.fx_graph_builder import FXGraphBuilder
from ..patterns.dynamo_patterns import DynamoPatternMatcher, get_pattern_matcher
from .workload import WorkloadProfile, WorkloadType, WorkloadClassifier
from .analyzer import SemanticAnalyzer
from ..core.types import ExecutionPhase
from .phase_detector import get_phase_detector, PhaseHistory

logger = logging.getLogger(__name__)


class EnhancedSemanticAnalyzer(SemanticAnalyzer):
    """Enhanced semantic analyzer using TorchDynamo pattern matching.
    
    This analyzer extends the base SemanticAnalyzer with declarative
    pattern matching capabilities using TorchDynamo.
    """
    
    def __init__(self, use_dynamo: bool = True, enable_phase_detection: bool = True):
        """Initialize enhanced analyzer.
        
        Args:
            use_dynamo: Whether to use TorchDynamo pattern matching
            enable_phase_detection: Whether to enable phase detection
        """
        super().__init__()
        self.use_dynamo = use_dynamo
        self.enable_phase_detection = enable_phase_detection
        self.pattern_matcher = get_pattern_matcher() if use_dynamo else None
        self.phase_detector = get_phase_detector() if enable_phase_detection else None
        self._pattern_cache = {}
    
    def analyze_fx_graph(self, fx_graph_module: fx.GraphModule) -> WorkloadProfile:
        """Analyze an FX graph using TorchDynamo patterns.
        
        Args:
            fx_graph_module: The FX GraphModule to analyze
            
        Returns:
            WorkloadProfile with semantic analysis results
        """
        start_time = time.perf_counter()
        
        # Use TorchDynamo pattern matching if enabled
        if self.use_dynamo and self.pattern_matcher:
            pattern_analysis = self.pattern_matcher.analyze_graph(fx_graph_module)
        else:
            pattern_analysis = self._fallback_pattern_analysis(fx_graph_module)
        
        # Classify workload based on patterns
        workload_type = self._classify_workload_from_patterns(pattern_analysis)
        
        # Add phase detection results if enabled
        phase_info = {}
        if self.enable_phase_detection and self.phase_detector:
            phase_info = {
                'phase_statistics': self.phase_detector.get_phase_statistics(),
                'current_phase': self.phase_detector.get_current_phase().value if self.phase_detector.get_current_phase() else 'unknown',
                'phase_history_length': len(self.phase_detector.get_phase_history().states)
            }
        
        # Build workload profile with empty patterns list
        profile = WorkloadProfile(
            workload_type=workload_type,
            patterns=[],  # Empty list of matched patterns
            metadata={
                'pattern_analysis': pattern_analysis,
                'total_patterns': pattern_analysis.get('total_patterns_matched', 0),
                'pattern_types': pattern_analysis.get('pattern_types', {}),
                'execution_phases': pattern_analysis.get('execution_phases', {}),
                'optimization_hints': pattern_analysis.get('optimization_hints', []),
                'high_priority_ops': pattern_analysis.get('high_priority_ops', []),
                'fusion_opportunities': pattern_analysis.get('fusion_opportunities', []),
                'phase_info': phase_info
            }
        )
        
        # Track performance
        analysis_time = time.perf_counter() - start_time
        self._analysis_stats['fx_analysis_time'] = analysis_time
        
        if analysis_time > 0.1:  # Log if analysis takes >100ms
            logger.warning(f"FX graph analysis took {analysis_time:.3f}s")
        
        return profile
    
    def _classify_workload_from_patterns(self, pattern_analysis: Dict[str, Any]) -> WorkloadType:
        """Classify workload type based on pattern analysis.
        
        Args:
            pattern_analysis: Pattern analysis results
            
        Returns:
            Detected WorkloadType
        """
        pattern_types = pattern_analysis.get('pattern_types', {})
        execution_phases = pattern_analysis.get('execution_phases', {})
        
        # Count pattern types
        attention_count = pattern_types.get('attention', 0)
        conv_count = pattern_types.get('convolution', 0)
        fusion_count = pattern_types.get('fusion', 0)
        
        # Check execution phases
        has_decode = execution_phases.get(ExecutionPhase.DECODE.value, 0) > 0
        has_prefill = execution_phases.get(ExecutionPhase.PREFILL.value, 0) > 0
        has_vision = execution_phases.get(ExecutionPhase.VISION_BACKBONE.value, 0) > 0
        has_fusion = execution_phases.get(ExecutionPhase.MULTIMODAL_FUSION.value, 0) > 0
        
        # Classification logic
        if has_fusion or fusion_count > 0:
            return WorkloadType.MULTIMODAL
        elif (attention_count > 2 and (has_decode or has_prefill)) or attention_count > 5:
            return WorkloadType.LLM
        elif conv_count > 3 or has_vision:
            return WorkloadType.VISION
        elif attention_count > 0 and conv_count > 0:
            return WorkloadType.MULTIMODAL
        else:
            return WorkloadType.UNKNOWN
    
    def _fallback_pattern_analysis(self, fx_graph_module: fx.GraphModule) -> Dict[str, Any]:
        """Fallback pattern analysis without TorchDynamo.
        
        Args:
            fx_graph_module: The FX GraphModule to analyze
            
        Returns:
            Basic pattern analysis results
        """
        analysis = {
            'total_patterns_matched': 0,
            'pattern_counts': {},
            'pattern_types': {},
            'execution_phases': {},
            'high_priority_ops': [],
            'fusion_opportunities': [],
            'optimization_hints': []
        }
        
        # Basic node counting
        for node in fx_graph_module.graph.nodes:
            if node.op == 'call_function':
                target_name = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)
                
                # Simple pattern detection
                if 'matmul' in target_name:
                    analysis['pattern_types']['attention'] = analysis['pattern_types'].get('attention', 0) + 1
                elif 'conv' in target_name:
                    analysis['pattern_types']['convolution'] = analysis['pattern_types'].get('convolution', 0) + 1
                elif 'cat' in target_name or 'concat' in target_name:
                    analysis['pattern_types']['fusion'] = analysis['pattern_types'].get('fusion', 0) + 1
                
                # Check metadata for execution phase
                if 'semantic' in node.meta:
                    metadata = node.meta['semantic']
                    if metadata.execution_phase:
                        phase = str(metadata.execution_phase.value if hasattr(metadata.execution_phase, 'value') else metadata.execution_phase)
                        analysis['execution_phases'][phase] = analysis['execution_phases'].get(phase, 0) + 1
        
        analysis['total_patterns_matched'] = sum(analysis['pattern_types'].values())
        
        return analysis
    
    def analyze_from_lazy_tensors(self) -> WorkloadProfile:
        """Analyze the current LazyTensor graph using FX and TorchDynamo.
        
        Returns:
            WorkloadProfile with analysis results
        """
        # Get current FX graph
        fx_builder = FXGraphBuilder.current()
        fx_graph_module = fx_builder.to_graph_module()
        
        # Analyze with TorchDynamo patterns
        profile = self.analyze_fx_graph(fx_graph_module)
        
        # Add graph statistics
        profile.metadata['graph_stats'] = fx_builder.get_semantic_summary()
        
        return profile
    
    def register_custom_pattern(self, pattern_name: str, 
                              pattern_fn: callable,
                              replacement_fn: Optional[callable] = None,
                              **metadata):
        """Register a custom pattern with the matcher.
        
        Args:
            pattern_name: Name for the pattern
            pattern_fn: Function defining the pattern
            replacement_fn: Optional replacement function
            **metadata: Additional metadata for the pattern
        """
        if not self.pattern_matcher:
            logger.warning("Pattern matcher not initialized")
            return
        
        from ..patterns.pattern_dsl import PatternBuilder, PatternType
        
        pattern = PatternBuilder(pattern_name) \
            .with_type(PatternType.CUSTOM) \
            .with_pattern(pattern_fn) \
            .with_replacement(replacement_fn) \
            .with_metadata(**metadata) \
            .build()
        
        self.pattern_matcher.add_custom_pattern(pattern)
        logger.info(f"Registered custom pattern: {pattern_name}")
    
    def get_pattern_statistics(self) -> Dict[str, int]:
        """Get pattern matching statistics.
        
        Returns:
            Dictionary of pattern names to match counts
        """
        if self.pattern_matcher:
            return self.pattern_matcher.get_match_statistics()
        return {}
    
    def reset_statistics(self):
        """Reset pattern matching statistics."""
        if self.pattern_matcher:
            self.pattern_matcher.reset_statistics()
        self._pattern_cache.clear()
        logger.info("Statistics reset")


# Convenience function for backward compatibility
def analyze_with_dynamo(fx_graph_module: fx.GraphModule) -> WorkloadProfile:
    """Analyze an FX graph using TorchDynamo patterns.
    
    Args:
        fx_graph_module: The FX GraphModule to analyze
        
    Returns:
        WorkloadProfile with analysis results
    """
    analyzer = EnhancedSemanticAnalyzer(use_dynamo=True)
    return analyzer.analyze_fx_graph(fx_graph_module)
