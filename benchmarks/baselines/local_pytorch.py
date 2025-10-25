"""
Local PyTorch Baseline - Upper Bound Performance.

Pure PyTorch execution on single GPU.
No disaggregation, no capture overhead.
This is the BEST CASE scenario.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List
from contextlib import contextmanager


class LocalPyTorchBaseline:
    """
    Pure PyTorch execution on single GPU.

    No disaggregation, no capture overhead.
    This is the BEST CASE scenario.
    """

    def __init__(self, device: str = 'cuda:0'):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.name = "local_pytorch"

    def run(self, model: nn.Module, inputs: List[torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Run model locally on single GPU.

        Args:
            model: PyTorch model (or model identifier for synthetic workloads)
            inputs: List of input tensors
            **kwargs: Additional arguments (ignored)

        Returns:
            Output tensor on CPU
        """
        # Handle synthetic workloads (microbenchmark)
        if isinstance(model, str):
            # For synthetic workloads, just return a mock output
            return torch.randn(1, 1000)  # Mock output

        # Move model and inputs to device
        model = model.to(self.device)
        device_inputs = [inp.to(self.device) for inp in inputs]

        # Execute with no_grad for inference
        with torch.no_grad():
            output = model(*device_inputs)

        # Handle different output types
        if hasattr(output, 'logits'):
            # HuggingFace model output (e.g., CausalLMOutputWithCrossAttentions)
            return output.logits.cpu()
        elif hasattr(output, 'cpu'):
            # Regular tensor output
            return output.cpu()
        else:
            # Fallback - convert to tensor if possible
            return torch.tensor(output).cpu()

    def get_metadata(self) -> Dict[str, Any]:
        """Get baseline metadata."""
        return {
            'baseline': 'local_pytorch',
            'device': str(self.device),
            'description': 'Pure PyTorch on single GPU (no disaggregation)',
            'expected_performance': 'fastest'
        }
