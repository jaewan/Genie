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


class LocalPyTorchBaseline:
    """
    Pure PyTorch execution on single GPU.

    No disaggregation, no capture overhead.
    This is the BEST CASE scenario.
    """

    def __init__(self, device: str = 'cuda:0'):
        self.device = torch.device(device)
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

        # Wrap model to handle complex outputs (BERT, CLIP)
        wrapped_model = ModelWrapper(model)
        
        # Move model and inputs to GPU
        wrapped_model = wrapped_model.to(self.device)
        device_inputs = [inp.to(self.device) for inp in inputs]

        # Execute with no_grad for inference
        with torch.no_grad():
            output = wrapped_model(*device_inputs)

        # Ensure output is tensor and move to CPU
        if isinstance(output, torch.Tensor):
            return output.cpu()
        elif hasattr(output, 'cpu'):
            return output.cpu()
        else:
            return torch.tensor(output).cpu()

    def get_metadata(self) -> Dict[str, Any]:
        """Get baseline metadata."""
        return {
            'baseline': 'local_pytorch',
            'device': str(self.device),
            'description': 'Pure PyTorch on single GPU (no disaggregation)',
            'expected_performance': 'fastest'
        }
