"""
Phase 3: Remote Block Execution

Server-side execution of TorchScript blocks with pipelining support.
"""

import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of a single block execution."""
    block_id: int
    block_name: str
    outputs: Dict[str, torch.Tensor]
    execution_time_ms: float
    status: str = 'success'
    error: Optional[str] = None


class RemoteBlockExecutor:
    """
    Server-side executor for TorchScript blocks.
    
    Features:
    - Execute compiled blocks on remote GPU
    - Pipelined execution across blocks
    - Result aggregation
    - Error handling and recovery
    """
    
    def __init__(self, device: str = 'cuda:0'):
        self.device = torch.device(device)
        self.block_cache: Dict[int, torch.jit.ScriptModule] = {}
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_execution_time_ms': 0.0,
        }
    
    def register_block(self, block_id: int, torchscript_module: torch.jit.ScriptModule):
        """
        Register a compiled block on server.
        
        Args:
            block_id: Unique block ID
            torchscript_module: Compiled TorchScript module
        """
        self.block_cache[block_id] = torchscript_module.to(self.device)
        logger.info(f"Registered block {block_id} on device {self.device}")
    
    def execute_block(
        self,
        block_id: int,
        inputs: Dict[str, torch.Tensor],
        block_name: str = "unknown"
    ) -> ExecutionResult:
        """
        Execute a single block.
        
        Args:
            block_id: ID of block to execute
            inputs: Dictionary of input tensors
            block_name: Name of block for logging
            
        Returns:
            ExecutionResult with outputs and timing
        """
        try:
            # Get cached block
            if block_id not in self.block_cache:
                raise ValueError(f"Block {block_id} not registered")
            
            block_module = self.block_cache[block_id]
            
            # Move inputs to device
            device_inputs = {
                key: tensor.to(self.device) if isinstance(tensor, torch.Tensor) else tensor
                for key, tensor in inputs.items()
            }
            
            # Execute with timing
            start_time = time.perf_counter()
            
            with torch.no_grad():
                if len(device_inputs) == 1:
                    # Single input
                    input_tensor = list(device_inputs.values())[0]
                    output = block_module(input_tensor)
                else:
                    # Multiple inputs - call with dict
                    output = block_module(**device_inputs)
            
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Package outputs
            if isinstance(output, torch.Tensor):
                outputs = {'output': output}
            elif isinstance(output, dict):
                outputs = output
            elif isinstance(output, (tuple, list)):
                outputs = {f'output_{i}': o for i, o in enumerate(output)}
            else:
                outputs = {'output': output}
            
            # Update stats
            self.stats['total_executions'] += 1
            self.stats['successful_executions'] += 1
            self.stats['total_execution_time_ms'] += execution_time_ms
            
            return ExecutionResult(
                block_id=block_id,
                block_name=block_name,
                outputs=outputs,
                execution_time_ms=execution_time_ms,
                status='success'
            )
        
        except Exception as e:
            self.stats['total_executions'] += 1
            self.stats['failed_executions'] += 1
            
            logger.error(f"Block execution failed: {e}")
            
            return ExecutionResult(
                block_id=block_id,
                block_name=block_name,
                outputs={},
                execution_time_ms=0.0,
                status='error',
                error=str(e)
            )
    
    def execute_pipeline(
        self,
        blocks: List[tuple],  # List of (block_id, block_name, inputs)
    ) -> List[ExecutionResult]:
        """
        Execute blocks in pipeline (sequentially with output chaining).
        
        Args:
            blocks: List of (block_id, block_name, inputs) tuples
            
        Returns:
            List of ExecutionResult for each block
        """
        results = []
        current_outputs = {}
        
        for block_id, block_name, initial_inputs in blocks:
            # Chain outputs from previous block
            if current_outputs and not initial_inputs:
                inputs = current_outputs
            else:
                inputs = initial_inputs or {}
            
            # Execute block
            result = self.execute_block(block_id, inputs, block_name)
            results.append(result)
            
            # Store outputs for next block
            current_outputs = result.outputs
            
            # Check for errors
            if result.status != 'success':
                logger.warning(f"Block {block_name} failed, stopping pipeline")
                break
        
        return results
    
    def execute_batch(
        self,
        batch_requests: List[Dict[str, Any]]
    ) -> List[ExecutionResult]:
        """
        Execute a batch of requests.
        
        Args:
            batch_requests: List of execution requests
            
        Returns:
            List of ExecutionResults
        """
        results = []
        
        for request in batch_requests:
            block_id = request['block_id']
            inputs = request['inputs']
            block_name = request.get('block_name', f'block_{block_id}')
            
            result = self.execute_block(block_id, inputs, block_name)
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        total = self.stats['total_executions']
        
        return {
            'total_executions': total,
            'successful_executions': self.stats['successful_executions'],
            'failed_executions': self.stats['failed_executions'],
            'success_rate': self.stats['successful_executions'] / total if total > 0 else 0,
            'avg_execution_time_ms': self.stats['total_execution_time_ms'] / total if total > 0 else 0,
            'total_execution_time_ms': self.stats['total_execution_time_ms'],
        }


class PipelinedBlockExecutor(RemoteBlockExecutor):
    """
    Pipelined block executor for overlapping communication and computation.
    
    Extends RemoteBlockExecutor with pipelining support.
    """
    
    def __init__(self, device: str = 'cuda:0'):
        super().__init__(device)
        self.pipeline_queue: List[Dict[str, Any]] = []
    
    def queue_for_pipeline(self, block_id: int, inputs: Dict[str, torch.Tensor]):
        """Queue a block for pipelined execution."""
        self.pipeline_queue.append({
            'block_id': block_id,
            'inputs': inputs,
        })
    
    def flush_pipeline(self) -> List[ExecutionResult]:
        """Execute all queued blocks in pipeline."""
        if not self.pipeline_queue:
            return []
        
        results = []
        current_outputs = {}
        
        for request in self.pipeline_queue:
            block_id = request['block_id']
            inputs = request.get('inputs', {})
            
            # Chain outputs
            if not inputs and current_outputs:
                inputs = current_outputs
            
            result = self.execute_block(block_id, inputs)
            results.append(result)
            
            current_outputs = result.outputs
        
        self.pipeline_queue.clear()
        return results


# Global executor instance
_global_executor = None
_executor_lock = None


def get_remote_executor(device: str = 'cuda:0') -> RemoteBlockExecutor:
    """Get or create global remote executor."""
    global _global_executor, _executor_lock
    
    if _executor_lock is None:
        import threading
        _executor_lock = threading.Lock()
    
    if _global_executor is None:
        with _executor_lock:
            if _global_executor is None:
                _global_executor = RemoteBlockExecutor(device)
    
    return _global_executor
