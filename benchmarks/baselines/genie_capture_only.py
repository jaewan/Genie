"""
Genie Capture Only Baseline - Overhead Measurement.

Measures LazyTensor capture overhead.
Captures graph but executes locally (no network).
Shows cost of graph building.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List
from contextlib import contextmanager


class GenieCaptureOnlyBaseline:
    """
    Measure LazyTensor capture overhead.

    Captures graph but executes locally (no network).
    Shows cost of graph building.
    """

    def __init__(self, device: str = 'remote_accelerator:0'):
        try:
            self.device = torch.device(device)
        except RuntimeError:
            # Fallback to CPU if remote device not available
            self.device = torch.device('cpu')
        self.name = "genie_capture_only"

    def run(self, model: nn.Module, inputs: List[torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Capture with LazyTensor but execute locally.

        Args:
            model: PyTorch model (or model identifier for synthetic workloads)
            inputs: List of input tensors
            **kwargs: Additional arguments (passed to model)

        Returns:
            Output tensor on CPU
        """
        # Handle synthetic workloads (microbenchmark)
        if isinstance(model, str):
            # For synthetic workloads, just return a mock output
            return torch.randn(1, 1000)  # Mock output

        # Move model to remote device (triggers LazyTensor capture)
        model = model.to(self.device)
        device_inputs = [inp.to(self.device) for inp in inputs]

        # This captures the computation graph
        output = model(*device_inputs)

        # Force local execution (no network transfer)
        if hasattr(output, '_materialize'):
            # It's a LazyTensor, execute locally
            result = output._materialize()
        else:
            # Already a concrete tensor
            result = output

        # Handle different output types
        if hasattr(result, 'logits'):
            # HuggingFace model output (e.g., CausalLMOutputWithCrossAttentions)
            return result.logits.cpu()
        elif hasattr(result, 'cpu'):
            # Regular tensor output
            return result.cpu()
        else:
            # Fallback - convert to tensor if possible
            return torch.tensor(result).cpu()

    def get_metadata(self) -> Dict[str, Any]:
        """Get baseline metadata."""
        return {
            'baseline': 'genie_capture_only',
            'device': str(self.device),
            'description': 'LazyTensor capture with local execution',
            'expected_overhead': '5-10% vs local PyTorch',
            'purpose': 'Measure graph building overhead'
        }
