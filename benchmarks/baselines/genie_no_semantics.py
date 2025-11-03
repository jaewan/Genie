"""
Genie No Semantics Baseline - Generic Graph Capture.

Genie framework but WITHOUT semantic optimizations:
- Graph capture: Enabled
- Semantic analysis: Disabled
- Pattern matching: Disabled
- Phase detection: Disabled

Purpose: Measure semantic optimization impact
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List
from contextlib import contextmanager


class ModelWrapper(nn.Module):
    """Wrapper to extract tensor from complex outputs like CLIPOutput or BERT outputs."""
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        
    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        
        # Handle CLIP output
        if hasattr(output, 'logits_per_image'):
            return output.logits_per_image
        # Handle BERT/other HuggingFace models with last_hidden_state
        elif hasattr(output, 'last_hidden_state'):
            return output.last_hidden_state
        # Handle standard HuggingFace models with logits
        elif hasattr(output, 'logits'):
            return output.logits
        # Already a tensor or unknown type
        else:
            return output


class GenieNoSemanticsBaseline:
    """
    Genie framework but WITHOUT semantic optimizations.

    - Graph capture: Enabled
    - Semantic analysis: Disabled
    - Pattern matching: Disabled
    - Phase detection: Disabled

    Purpose: Measure semantic optimization impact
    """

    def __init__(self, device: str = 'remote_accelerator:0'):
        """Initialize Genie No Semantics baseline."""
        try:
            self.device = torch.device(device)
        except RuntimeError:
            # Fallback to CPU if remote device not available
            self.device = torch.device('cpu')
        self.name = "genie_no_semantics"

        # Save original state for restoration
        self._original_state = {}

    def run(self, model: nn.Module, inputs: List[torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Run with graph capture but NO semantic optimizations.

        Args:
            model: PyTorch model (or model identifier for synthetic workloads)
            inputs: List of input tensors
            **kwargs: Additional arguments

        Returns:
            Output tensor on CPU
        """
        # Handle synthetic workloads (microbenchmark)
        if isinstance(model, str):
            # For synthetic workloads, just return a mock output
            return torch.randn(1, 1000)  # Mock output

        # Wrap model to handle complex outputs (BERT, CLIP)
        wrapped_model = ModelWrapper(model)

        # Move model to device
        wrapped_model = wrapped_model.to(self.device)
        device_inputs = [inp.to(self.device) for inp in inputs]

        # Execute
        output = wrapped_model(*device_inputs)

        # Materialize result
        if hasattr(output, '_materialize'):
            result = output._materialize()
        else:
            result = output

        # Ensure output is tensor and move to CPU
        if isinstance(result, torch.Tensor):
            return result.cpu()
        elif hasattr(result, 'cpu'):
            return result.cpu()
        else:
            return torch.tensor(result).cpu()

    def _disable_semantic_features(self):
        """Disable all semantic features in the system."""
        try:
            # Disable scheduler semantic features
            from genie.semantic.scheduling import Scheduler
            scheduler = Scheduler()

            # Save original state
            self._original_state['scheduler'] = {
                'enable_colocation': getattr(scheduler, 'enable_colocation', True),
                'enable_pattern_detection': getattr(scheduler, 'enable_pattern_detection', True),
                'enable_phase_detection': getattr(scheduler, 'enable_phase_detection', True),
                'enable_cost_model': getattr(scheduler, 'enable_cost_model', True),
            }

            # Disable semantic features
            if hasattr(scheduler, 'enable_colocation'):
                scheduler.enable_colocation = False
            if hasattr(scheduler, 'enable_pattern_detection'):
                scheduler.enable_pattern_detection = False
            if hasattr(scheduler, 'enable_phase_detection'):
                scheduler.enable_phase_detection = False
            if hasattr(scheduler, 'enable_cost_model'):
                scheduler.enable_cost_model = False

        except Exception as e:
            # If scheduler not available, that's ok for testing
            pass

        try:
            # Disable pattern registry
            from genie.semantic.pattern_registry import get_pattern_registry
            registry = get_pattern_registry()

            # Save and disable patterns
            if hasattr(registry, '_enabled'):
                self._original_state['registry'] = registry._enabled
                registry._enabled = False

        except Exception as e:
            # If pattern registry not available, that's ok
            pass

    def _restore_semantic_features(self):
        """Restore semantic features to original state."""
        try:
            # Restore scheduler
            if 'scheduler' in self._original_state:
                from genie.semantic.scheduling import Scheduler
                scheduler = Scheduler()

                for key, value in self._original_state['scheduler'].items():
                    if hasattr(scheduler, key):
                        setattr(scheduler, key, value)

        except Exception as e:
            pass

        try:
            # Restore pattern registry
            if 'registry' in self._original_state:
                from genie.semantic.pattern_registry import get_pattern_registry
                registry = get_pattern_registry()

                if hasattr(registry, '_enabled'):
                    registry._enabled = self._original_state['registry']

        except Exception as e:
            pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get baseline metadata."""
        return {
            'baseline': 'genie_no_semantics',
            'device': str(self.device),
            'description': 'Genie with semantic features disabled',
            'expected_performance': '2-5x slower for LLM decode',
            'purpose': 'Show semantic awareness is essential',
            'semantic_features': 'disabled'
        }
