"""Execution phase detection for semantic-aware execution.

This module implements sophisticated phase detection for different workload types,
enabling runtime identification of execution phases like prefill, decode, and fusion.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import torch
import threading
from collections import deque, defaultdict

from ..core.semantic_metadata import ExecutionPhase, MemoryPattern

logger = logging.getLogger(__name__)


class PhaseTransition(Enum):
    """Types of phase transitions."""
    INIT_TO_PREFILL = "init_to_prefill"
    PREFILL_TO_DECODE = "prefill_to_decode"
    DECODE_TO_DECODE = "decode_to_decode"
    VISION_STAGE_TRANSITION = "vision_stage_transition"
    MODAL_TO_FUSION = "modal_to_fusion"
    FUSION_TO_OUTPUT = "fusion_to_output"
    UNKNOWN = "unknown"


@dataclass
class PhaseState:
    """State information for a phase."""
    phase: ExecutionPhase
    start_time: float
    operations: List[str] = field(default_factory=list)
    tensor_ids: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    sequence_length: int = 0
    batch_size: int = 0


@dataclass
class PhaseHistory:
    """History of phase transitions."""
    states: List[PhaseState] = field(default_factory=list)
    transitions: List[Tuple[PhaseTransition, float]] = field(default_factory=list)
    current_state: Optional[PhaseState] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PhaseDetector:
    """Sophisticated phase detector for execution tracking.
    
    This detector analyzes operations, tensor shapes, and metadata to determine
    the current execution phase of the workload.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for phase detector."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize phase detector."""
        if not hasattr(self, '_initialized'):
            self.phase_stack = deque(maxlen=10)
            self.phase_history = PhaseHistory()
            self.operation_counts = defaultdict(int)
            self.kv_cache_size = {}
            self.token_position = 0
            self.is_first_forward = True
            self.vision_stage_counter = 0
            self.detected_modalities = set()
            self._initialized = True
            
            # Phase detection patterns
            self.attention_ops = {'matmul', 'softmax', 'attention', 'scaled_dot_product'}
            self.conv_ops = {'conv2d', 'conv1d', 'conv3d', 'convolution', 'conv'}
            self.fusion_ops = {'cat', 'concat', 'add', 'mul', 'fusion', 'cross_attention'}
            self.kv_cache_ops = {'update_cache', 'concat_cache', 'cat'}
    
    def detect_phase(self, operation: str, inputs: List[Any], 
                    metadata: Optional[Dict] = None) -> ExecutionPhase:
        """Detect the current execution phase.
        
        Args:
            operation: The operation being performed
            inputs: Input tensors/values
            metadata: Optional metadata about the operation
            
        Returns:
            Detected ExecutionPhase
        """
        metadata = metadata or {}
        
        # Clean operation name
        op_name = operation.split('::')[-1].lower() if '::' in operation else operation.lower()
        # Remove underscore numbers (conv2d_0 -> conv2d)
        if '_' in op_name and op_name.split('_')[-1].isdigit():
            op_name = '_'.join(op_name.split('_')[:-1])
        self.operation_counts[op_name] += 1
        
        # Check for explicit phase hint in metadata
        if 'execution_phase' in metadata:
            phase = metadata['execution_phase']
            if isinstance(phase, ExecutionPhase):
                self._update_phase_state(phase, op_name, metadata)
                return phase
        
        # LLM phase detection
        phase = self._detect_llm_phase(op_name, inputs, metadata)
        if phase != ExecutionPhase.UNKNOWN:
            self._update_phase_state(phase, op_name, metadata)
            return phase
        
        # Vision phase detection
        phase = self._detect_vision_phase(op_name, inputs, metadata)
        if phase != ExecutionPhase.UNKNOWN:
            self._update_phase_state(phase, op_name, metadata)
            return phase
        
        # Multi-modal phase detection
        phase = self._detect_multimodal_phase(op_name, inputs, metadata)
        if phase != ExecutionPhase.UNKNOWN:
            self._update_phase_state(phase, op_name, metadata)
            return phase
        
        # Default based on operation type
        phase = self._detect_by_operation_type(op_name, inputs, metadata)
        self._update_phase_state(phase, op_name, metadata)
        
        return phase
    
    def _detect_llm_phase(self, operation: str, inputs: List[Any], 
                         metadata: Dict) -> ExecutionPhase:
        """Detect LLM-specific phases (prefill/decode).
        
        Args:
            operation: Operation name
            inputs: Input tensors
            metadata: Operation metadata
            
        Returns:
            Detected phase or UNKNOWN
        """
        # Check for KV cache operations
        if self._is_kv_cache_operation(operation, metadata):
            # Check cache size to determine phase
            if self._is_prefill_phase(inputs, metadata):
                return ExecutionPhase.PREFILL
            else:
                return ExecutionPhase.DECODE
        
        # Check for attention operations
        if operation in self.attention_ops:
            # Analyze sequence length
            seq_len = self._get_sequence_length(inputs)
            
            if seq_len is not None:
                if seq_len > 1:
                    # Processing multiple tokens - likely prefill
                    return ExecutionPhase.PREFILL
                elif seq_len == 1:
                    # Single token - likely decode
                    return ExecutionPhase.DECODE
        
        # Check for embedding operations (often start of prefill)
        if 'embedding' in operation:
            # First embedding is usually prefill
            if self.is_first_forward:
                self.is_first_forward = False
                return ExecutionPhase.PREFILL
            else:
                # Subsequent embeddings might be decode
                return ExecutionPhase.EMBEDDING
        
        return ExecutionPhase.UNKNOWN
    
    def _detect_vision_phase(self, operation: str, inputs: List[Any], 
                            metadata: Dict) -> ExecutionPhase:
        """Detect vision-specific phases.
        
        Args:
            operation: Operation name
            inputs: Input tensors
            metadata: Operation metadata
            
        Returns:
            Detected phase or UNKNOWN
        """
        if operation in self.conv_ops or 'conv' in operation:
            # Track vision modality
            self.detected_modalities.add('vision')
            
            # Track convolution stages
            self.vision_stage_counter += 1
            
            # Early layers are backbone
            if self.vision_stage_counter <= 5:
                return ExecutionPhase.VISION_BACKBONE
            else:
                # Later layers are head
                return ExecutionPhase.VISION_HEAD
        
        # Pooling operations often indicate transition
        if 'pool' in operation:
            # Global pooling usually at end of backbone
            if 'global' in operation or 'adaptive' in operation:
                return ExecutionPhase.VISION_HEAD
            else:
                return ExecutionPhase.VISION_BACKBONE
        
        # Vision-specific normalizations
        if 'batch_norm' in operation or 'layer_norm' in operation:
            # Check if we're in vision context
            if self.vision_stage_counter > 0:
                return ExecutionPhase.VISION_BACKBONE
        
        return ExecutionPhase.UNKNOWN
    
    def _detect_multimodal_phase(self, operation: str, inputs: List[Any], 
                                metadata: Dict) -> ExecutionPhase:
        """Detect multi-modal phases.
        
        Args:
            operation: Operation name
            inputs: Input tensors
            metadata: Operation metadata
            
        Returns:
            Detected phase or UNKNOWN
        """
        # Track modalities
        if 'modality' in metadata:
            self.detected_modalities.add(metadata['modality'])
        
        # Check for fusion operations
        if operation in self.fusion_ops:
            # Check if inputs come from different modalities
            if self._has_multiple_modalities(inputs, metadata):
                return ExecutionPhase.MULTIMODAL_FUSION
        
        # Cross-attention is a strong signal of fusion
        if 'cross' in operation and 'attention' in operation:
            return ExecutionPhase.MULTIMODAL_FUSION
        
        # Check if we're combining features
        if len(self.detected_modalities) > 1:
            if operation in ['cat', 'concat', 'add', 'mul']:
                # Could be fusion if we have multiple modalities
                return ExecutionPhase.MULTIMODAL_FUSION
        
        return ExecutionPhase.UNKNOWN
    
    def _detect_by_operation_type(self, operation: str, inputs: List[Any], 
                                 metadata: Dict) -> ExecutionPhase:
        """Detect phase based on general operation type.
        
        Args:
            operation: Operation name
            inputs: Input tensors
            metadata: Operation metadata
            
        Returns:
            Best guess ExecutionPhase
        """
        # Look at recent phase history
        if self.phase_history.current_state:
            # Tend to stay in same phase unless clear transition
            return self.phase_history.current_state.phase
        
        # Default heuristics
        if operation in self.attention_ops:
            return ExecutionPhase.DECODE  # Default for attention
        elif operation in self.conv_ops:
            return ExecutionPhase.VISION_BACKBONE
        elif operation in self.fusion_ops:
            return ExecutionPhase.MULTIMODAL_FUSION
        elif 'embedding' in operation:
            return ExecutionPhase.EMBEDDING
        else:
            return ExecutionPhase.UNKNOWN
    
    def _is_kv_cache_operation(self, operation: str, metadata: Dict) -> bool:
        """Check if operation involves KV cache.
        
        Args:
            operation: Operation name
            metadata: Operation metadata
            
        Returns:
            True if KV cache operation
        """
        # Check operation name
        if any(kv_op in operation for kv_op in self.kv_cache_ops):
            return True
        
        # Check metadata
        if metadata.get('kv_cache_related', False):
            return True
        
        # Check for cache-like concatenation
        if operation == 'cat' and metadata.get('dim') == 1:
            # Could be sequence dimension concatenation for cache
            return True
        
        return False
    
    def _is_prefill_phase(self, inputs: List[Any], metadata: Dict) -> bool:
        """Determine if we're in prefill phase based on inputs.
        
        Args:
            inputs: Input tensors
            metadata: Operation metadata
            
        Returns:
            True if prefill phase
        """
        # Check token position
        if self.token_position == 0:
            # First forward is prefill
            return True
        
        # Check sequence length
        seq_len = self._get_sequence_length(inputs)
        if seq_len and seq_len > 1:
            # Processing multiple tokens
            return True
        
        # Check if KV cache is empty or small
        for inp in inputs:
            if hasattr(inp, 'shape'):
                shape = inp.shape
                if len(shape) >= 2:
                    # Check if this looks like an empty or small cache
                    if shape[1] <= 5:  # Small sequence dimension
                        return True
        
        return False
    
    def _get_sequence_length(self, inputs: List[Any]) -> Optional[int]:
        """Extract sequence length from inputs.
        
        Args:
            inputs: Input tensors
            
        Returns:
            Sequence length or None
        """
        for inp in inputs:
            if hasattr(inp, 'shape'):
                shape = inp.shape
                if len(shape) >= 2:
                    # Assume dimension 1 is sequence for [batch, seq, ...]
                    return shape[1] if len(shape) >= 3 else shape[0]
        
        return None
    
    def _has_multiple_modalities(self, inputs: List[Any], metadata: Dict) -> bool:
        """Check if inputs come from multiple modalities.
        
        Args:
            inputs: Input tensors
            metadata: Operation metadata
            
        Returns:
            True if multiple modalities detected
        """
        # Simple heuristic - if we've seen multiple modalities recently
        return len(self.detected_modalities) > 1
    
    def _update_phase_state(self, phase: ExecutionPhase, operation: str, 
                          metadata: Dict):
        """Update phase state tracking.
        
        Args:
            phase: Detected phase
            operation: Operation name
            metadata: Operation metadata
        """
        import time
        
        # Check for phase transition
        if self.phase_history.current_state:
            if self.phase_history.current_state.phase != phase:
                # Phase transition detected
                transition = self._classify_transition(
                    self.phase_history.current_state.phase, phase
                )
                self.phase_history.transitions.append((transition, time.time()))
                
                # Save current state to history
                self.phase_history.states.append(self.phase_history.current_state)
        
        # Update or create current state
        if self.phase_history.current_state and self.phase_history.current_state.phase == phase:
            # Same phase - update existing state
            self.phase_history.current_state.operations.append(operation)
            if 'tensor_id' in metadata:
                self.phase_history.current_state.tensor_ids.add(metadata['tensor_id'])
        else:
            # New phase
            self.phase_history.current_state = PhaseState(
                phase=phase,
                start_time=time.time(),
                operations=[operation],
                metadata=metadata.copy()
            )
        
        # Update phase stack
        self.phase_stack.append(phase)
        
        # Update token position for LLM phases
        if phase == ExecutionPhase.DECODE:
            self.token_position += 1
        elif phase == ExecutionPhase.PREFILL:
            # Prefill processes multiple tokens
            seq_len = metadata.get('sequence_length', 1)
            self.token_position += seq_len
    
    def _classify_transition(self, from_phase: ExecutionPhase, 
                           to_phase: ExecutionPhase) -> PhaseTransition:
        """Classify the type of phase transition.
        
        Args:
            from_phase: Previous phase
            to_phase: New phase
            
        Returns:
            PhaseTransition type
        """
        if from_phase == ExecutionPhase.UNKNOWN:
            if to_phase == ExecutionPhase.PREFILL:
                return PhaseTransition.INIT_TO_PREFILL
        elif from_phase == ExecutionPhase.PREFILL:
            if to_phase == ExecutionPhase.DECODE:
                return PhaseTransition.PREFILL_TO_DECODE
        elif from_phase == ExecutionPhase.DECODE:
            if to_phase == ExecutionPhase.DECODE:
                return PhaseTransition.DECODE_TO_DECODE
        elif from_phase in [ExecutionPhase.VISION_BACKBONE, ExecutionPhase.VISION_HEAD]:
            if to_phase in [ExecutionPhase.VISION_BACKBONE, ExecutionPhase.VISION_HEAD]:
                return PhaseTransition.VISION_STAGE_TRANSITION
        elif to_phase == ExecutionPhase.MULTIMODAL_FUSION:
            return PhaseTransition.MODAL_TO_FUSION
        elif from_phase == ExecutionPhase.MULTIMODAL_FUSION:
            return PhaseTransition.FUSION_TO_OUTPUT
        
        return PhaseTransition.UNKNOWN
    
    def get_current_phase(self) -> ExecutionPhase:
        """Get the current execution phase.
        
        Returns:
            Current ExecutionPhase
        """
        if self.phase_history.current_state:
            return self.phase_history.current_state.phase
        return ExecutionPhase.UNKNOWN
    
    def get_phase_history(self) -> PhaseHistory:
        """Get the complete phase history.
        
        Returns:
            PhaseHistory object
        """
        return self.phase_history
    
    def get_phase_statistics(self) -> Dict[str, Any]:
        """Get statistics about phase execution.
        
        Returns:
            Dictionary of phase statistics
        """
        stats = {
            'total_phases': len(self.phase_history.states),
            'current_phase': self.get_current_phase().value,
            'transitions': len(self.phase_history.transitions),
            'operation_counts': dict(self.operation_counts),
            'token_position': self.token_position,
            'detected_modalities': list(self.detected_modalities),
            'vision_stages': self.vision_stage_counter
        }
        
        # Count time in each phase
        phase_times = defaultdict(float)
        for state in self.phase_history.states:
            if hasattr(state, 'start_time'):
                # Simplified - would need end time for accurate measurement
                phase_times[state.phase.value] += 1.0
        
        stats['phase_times'] = dict(phase_times)
        
        return stats
    
    def reset(self):
        """Reset the phase detector state."""
        self.phase_stack.clear()
        self.phase_history = PhaseHistory()
        self.operation_counts.clear()
        self.kv_cache_size.clear()
        self.token_position = 0
        self.is_first_forward = True
        self.vision_stage_counter = 0
        self.detected_modalities.clear()
    
    def mark_batch_boundary(self):
        """Mark the boundary between batches."""
        # Reset per-batch state
        self.is_first_forward = True
        self.vision_stage_counter = 0
        # Keep token position and cache for continuous generation


def get_phase_detector() -> PhaseDetector:
    """Get the singleton phase detector instance.
    
    Returns:
        PhaseDetector instance
    """
    return PhaseDetector()


class PhaseAwareHook:
    """Hook that integrates with phase detection.
    
    This hook can be attached to modules to automatically detect phases
    during forward passes.
    """
    
    def __init__(self, module_name: str = ""):
        """Initialize phase-aware hook.
        
        Args:
            module_name: Name of the module this hook is attached to
        """
        self.module_name = module_name
        self.detector = get_phase_detector()
    
    def __call__(self, module, inputs, outputs):
        """Hook function called during forward pass.
        
        Args:
            module: The module being executed
            inputs: Input tensors
            outputs: Output tensors
            
        Returns:
            outputs (unchanged)
        """
        # Detect phase based on module type and inputs
        operation = self._get_operation_name(module)
        metadata = self._extract_metadata(module, inputs, outputs)
        
        phase = self.detector.detect_phase(operation, inputs, metadata)
        
        # Attach phase to output tensors if possible
        if hasattr(outputs, 'meta'):
            outputs.meta['execution_phase'] = phase
        elif isinstance(outputs, (list, tuple)):
            for out in outputs:
                if hasattr(out, 'meta'):
                    out.meta['execution_phase'] = phase
        
        return outputs
    
    def _get_operation_name(self, module) -> str:
        """Extract operation name from module.
        
        Args:
            module: PyTorch module
            
        Returns:
            Operation name string
        """
        module_type = module.__class__.__name__
        
        # Map module types to operations
        if 'Linear' in module_type:
            return 'linear'
        elif 'Conv' in module_type:
            return module_type.lower()
        elif 'Attention' in module_type:
            return 'attention'
        elif 'LayerNorm' in module_type or 'BatchNorm' in module_type:
            return 'normalization'
        else:
            return module_type.lower()
    
    def _extract_metadata(self, module, inputs, outputs) -> Dict[str, Any]:
        """Extract metadata from module execution.
        
        Args:
            module: PyTorch module
            inputs: Input tensors
            outputs: Output tensors
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'module_name': self.module_name,
            'module_type': module.__class__.__name__
        }
        
        # Extract shapes
        if isinstance(inputs, (list, tuple)) and len(inputs) > 0:
            first_input = inputs[0]
            if hasattr(first_input, 'shape'):
                metadata['input_shape'] = list(first_input.shape)
                if len(first_input.shape) >= 2:
                    metadata['batch_size'] = first_input.shape[0]
                    metadata['sequence_length'] = first_input.shape[1] if len(first_input.shape) >= 3 else 1
        
        # Check for specific attributes
        if hasattr(module, 'num_heads'):
            metadata['num_heads'] = module.num_heads
        if hasattr(module, 'hidden_size'):
            metadata['hidden_size'] = module.hidden_size
        
        return metadata
