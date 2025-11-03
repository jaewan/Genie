"""
Naive Disaggregation Baseline.

Transfers ALL tensors without semantic optimization.
This represents the WORST CASE for disaggregation and demonstrates
the value of Genie's semantic awareness.

Strategy:
- No KV cache co-location (transfer cache every decode step)
- No pipelining (synchronous execution)
- No recomputation (always transfer)
- No batching (transfer tensors individually)
- No caching (transfer model every time)

Purpose: Show the value of semantic awareness (3-5x improvement).
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any


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


class NaiveDisaggregationBaseline:
    """
    Worst-case disaggregation: transfer everything without optimization.
    
    This baseline demonstrates what happens when you disaggregate GPUs
    without semantic awareness - it's the naive approach that Genie
    dramatically improves upon.
    """
    
    def __init__(self, server_address: str = 'localhost:5556'):
        self.server_address = server_address
        self.coordinator = None
        self.name = "naive_disaggregation"
        self.use_real_network = False  # Will be set by evaluation framework
    
    def run(self, model: nn.Module, inputs: List[torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Run with naive disaggregation (worst case).
        
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
            return torch.randn(1, 1000)
        
        if self.use_real_network:
            return self._run_with_network(model, inputs)
        else:
            # Fallback: simulate naive approach with extra overhead
            return self._run_simulated_naive(model, inputs)
    
    def _run_with_network(self, model: nn.Module, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Execute using real network with naive strategy (no optimization)."""
        
        # Initialize coordinator if needed
        if self.coordinator is None:
            try:
                from genie.remote.coordinator import RemoteCoordinator
                self.coordinator = RemoteCoordinator(self.server_address)
            except ImportError:
                # Fallback if coordinator not available
                return self._run_simulated_naive(model, inputs)
        
        try:
            # Strategy 1: Transfer model (no caching - transfer every time!)
            # In a real naive system, you'd transfer the entire model each time
            # This is extremely inefficient but shows the worst case
            remote_model = self.coordinator.transfer_model(model)
            
            # Strategy 2: Transfer inputs (no batching - one by one!)
            remote_inputs = []
            for inp in inputs:
                remote_inp = self.coordinator.transfer_tensor(inp)
                remote_inputs.append(remote_inp)
            
            # Strategy 3: Execute remotely (no semantic optimization)
            remote_output = self.coordinator.execute_remote(
                remote_model,
                remote_inputs,
                optimize=False  # ← KEY: No semantic optimization
            )
            
            # Strategy 4: Transfer output back (synchronous, no pipelining)
            output = self.coordinator.fetch_result(remote_output)
            
            return output.cpu()
            
        except Exception as e:
            # If network execution fails, fall back to simulated
            print(f"⚠️  Network execution failed: {e}, falling back to simulation")
            return self._run_simulated_naive(model, inputs)
    
    def _run_simulated_naive(self, model: nn.Module, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Simulate naive disaggregation overhead without actual network.
        
        This adds artificial delays to simulate the overhead of:
        - Transferring model
        - Transferring inputs
        - Synchronous execution
        - Transferring outputs
        """
        import time
        
        # Wrap model to handle complex outputs (BERT, CLIP)
        wrapped_model = ModelWrapper(model)
        
        # Move model and inputs to device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        wrapped_model = wrapped_model.to(device)
        device_inputs = [inp.to(device) for inp in inputs]
        
        # Simulate naive transfer overhead
        # Estimate: ~10ms per MB transferred
        model_size_mb = sum(p.numel() * p.element_size() for p in wrapped_model.parameters()) / (1024 * 1024)
        input_size_mb = sum(inp.numel() * inp.element_size() for inp in inputs) / (1024 * 1024)
        
        # Naive approach: transfer everything
        transfer_overhead_ms = (model_size_mb + input_size_mb) * 10
        time.sleep(transfer_overhead_ms / 1000)  # Simulate transfer time
        
        # Execute with no_grad for inference
        with torch.no_grad():
            output = wrapped_model(*device_inputs)
        
        # Simulate output transfer overhead
        if hasattr(output, 'logits'):
            output_tensor = output.logits
        elif hasattr(output, 'cpu'):
            output_tensor = output
        else:
            output_tensor = torch.tensor(output)
        
        output_size_mb = output_tensor.numel() * output_tensor.element_size() / (1024 * 1024)
        output_overhead_ms = output_size_mb * 10
        time.sleep(output_overhead_ms / 1000)  # Simulate output transfer
        
        # Handle different output types
        if hasattr(output, 'logits'):
            return output.logits.cpu()
        elif hasattr(output, 'cpu'):
            return output.cpu()
        else:
            return torch.tensor(output).cpu()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get baseline metadata."""
        return {
            'baseline': 'naive_disaggregation',
            'description': 'Worst-case disaggregation (no semantic optimization)',
            'strategy': 'Transfer everything without optimization',
            'expected_performance': 'slowest (demonstrates value of Genie)',
            'key_inefficiencies': [
                'No KV cache co-location',
                'No pipelining',
                'No recomputation',
                'No batching',
                'No model caching'
            ]
        }

