"""
Phase 3: Block Serialization

Handles serialization of TorchScript blocks and tensors for network transfer.
"""

import torch
import pickle
import json
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import io
import logging

logger = logging.getLogger(__name__)


@dataclass
class SerializedTensor:
    """Serialized tensor metadata and data."""
    shape: Tuple[int, ...]
    dtype: str
    device: str
    data: bytes
    
    def deserialize(self) -> torch.Tensor:
        """Reconstruct tensor from serialized form."""
        # Deserialize data
        buffer = io.BytesIO(self.data)
        array = pickle.load(buffer)
        
        # Reconstruct tensor
        dtype = getattr(torch, self.dtype.split('.')[-1])
        tensor = torch.tensor(array, dtype=dtype)
        
        # Move to device if needed
        if self.device != 'cpu':
            tensor = tensor.to(self.device)
        
        return tensor


@dataclass  
class SerializedBlock:
    """Serialized block with metadata."""
    block_id: int
    name: str
    input_names: list
    output_names: list
    torchscript_bytes: bytes
    metadata: Dict[str, Any]


class TensorSerializer:
    """Serializes and deserializes tensors for network transfer."""
    
    @staticmethod
    def serialize(tensor: torch.Tensor) -> SerializedTensor:
        """
        Serialize tensor to bytes with metadata.
        
        Args:
            tensor: Tensor to serialize
            
        Returns:
            SerializedTensor with shape, dtype, and bytes
        """
        # Convert to CPU for serialization
        cpu_tensor = tensor.cpu().detach()
        
        # Serialize to bytes
        buffer = io.BytesIO()
        pickle.dump(cpu_tensor.numpy(), buffer)
        data = buffer.getvalue()
        
        return SerializedTensor(
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            data=data
        )
    
    @staticmethod
    def deserialize(serialized: SerializedTensor) -> torch.Tensor:
        """Deserialize bytes back to tensor."""
        return serialized.deserialize()
    
    @staticmethod
    def serialize_dict(tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, SerializedTensor]:
        """Serialize dictionary of tensors."""
        return {
            key: TensorSerializer.serialize(tensor)
            for key, tensor in tensor_dict.items()
        }
    
    @staticmethod
    def deserialize_dict(serialized_dict: Dict[str, SerializedTensor]) -> Dict[str, torch.Tensor]:
        """Deserialize dictionary of tensors."""
        return {
            key: TensorSerializer.deserialize(serialized)
            for key, serialized in serialized_dict.items()
        }


class BlockSerializer:
    """Serializes and deserializes executable blocks."""
    
    @staticmethod
    def serialize_block(block) -> SerializedBlock:
        """
        Serialize an ExecutableBlock for network transfer.
        
        Args:
            block: ExecutableBlock to serialize
            
        Returns:
            SerializedBlock with all necessary data
        """
        # Serialize TorchScript module
        ts_bytes = block.serialize() if block.torchscript_module else b''
        
        # Collect metadata
        metadata = {
            'operation_count': block.operation_count,
            'memory_bytes': block.memory_bytes,
            'compute_flops': block.compute_flops,
            'dependencies': block.dependencies,
        }
        
        return SerializedBlock(
            block_id=block.block_id,
            name=block.name,
            input_names=block.input_names,
            output_names=block.output_names,
            torchscript_bytes=ts_bytes,
            metadata=metadata
        )
    
    @staticmethod
    def deserialize_block(serialized: SerializedBlock, BlockClass):
        """
        Reconstruct ExecutableBlock from serialized form.
        
        Args:
            serialized: SerializedBlock
            BlockClass: ExecutableBlock class
            
        Returns:
            Reconstructed ExecutableBlock
        """
        # Deserialize TorchScript module
        if serialized.torchscript_bytes:
            buffer = io.BytesIO(serialized.torchscript_bytes)
            ts_module = torch.jit.load(buffer)
        else:
            ts_module = None
        
        # Reconstruct block
        block = BlockClass(
            block_id=serialized.block_id,
            name=serialized.name,
            torchscript_module=ts_module,
            input_names=serialized.input_names,
            output_names=serialized.output_names,
            operation_count=serialized.metadata['operation_count'],
            memory_bytes=serialized.metadata['memory_bytes'],
            compute_flops=serialized.metadata['compute_flops'],
            dependencies=serialized.metadata['dependencies'],
        )
        
        return block


class RequestSerializer:
    """Serializes execution requests for transmission."""
    
    @staticmethod
    def serialize_execution_request(
        block_id: int,
        inputs: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Serialize a block execution request.
        
        Args:
            block_id: ID of block to execute
            inputs: Input tensors
            metadata: Optional metadata (request_id, timestamp, etc.)
            
        Returns:
            Serialized request dict
        """
        serialized_inputs = TensorSerializer.serialize_dict(inputs)
        
        # Convert SerializedTensor to dict for JSON serialization
        inputs_dict = {
            key: {
                'shape': tensor.shape,
                'dtype': tensor.dtype,
                'device': tensor.device,
                # data will be sent separately (binary)
            }
            for key, tensor in serialized_inputs.items()
        }
        
        request = {
            'block_id': block_id,
            'inputs_metadata': inputs_dict,
            'metadata': metadata or {},
        }
        
        return request
    
    @staticmethod
    def serialize_execution_response(
        block_id: int,
        outputs: Dict[str, torch.Tensor],
        status: str = 'success',
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Serialize execution response.
        
        Args:
            block_id: ID of executed block
            outputs: Output tensors
            status: 'success' or 'error'
            error: Error message if failed
            
        Returns:
            Serialized response dict
        """
        serialized_outputs = TensorSerializer.serialize_dict(outputs)
        
        # Convert to dict for JSON
        outputs_dict = {
            key: {
                'shape': tensor.shape,
                'dtype': tensor.dtype,
                'device': tensor.device,
            }
            for key, tensor in serialized_outputs.items()
        }
        
        response = {
            'block_id': block_id,
            'status': status,
            'outputs_metadata': outputs_dict if status == 'success' else None,
            'error': error,
        }
        
        return response


class BatchSerializer:
    """Serializes batches of requests and responses."""
    
    @staticmethod
    def serialize_batch(requests: list) -> bytes:
        """Serialize a batch of requests."""
        return pickle.dumps(requests)
    
    @staticmethod
    def deserialize_batch(data: bytes) -> list:
        """Deserialize a batch of requests."""
        return pickle.loads(data)
