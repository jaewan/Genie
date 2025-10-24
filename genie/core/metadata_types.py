"""
Typed metadata classes for the Genie framework.

Replaces untyped Dict[str, Any] usage with strongly-typed dataclasses.
Provides validation, better IDE support, and prevents common errors like typos in key names.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch
from enum import Enum
from .types import ExecutionPhase, MemoryPattern, Modality


class MessageType(Enum):
    """Types of messages in the transport protocol."""
    OPERATION_REQUEST = "operation_request"
    RESULT = "result"
    ERROR = "error"
    TENSOR_TRANSFER = "tensor_transfer"
    MULTI_TENSOR = "multi_tensor"


@dataclass
class TensorMetadata:
    """Metadata for tensor transfers."""
    shape: List[int]
    dtype: str  # torch.dtype string representation
    device: str  # torch.device string representation
    size_bytes: int

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'TensorMetadata':
        """Create TensorMetadata from a PyTorch tensor."""
        return cls(
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            size_bytes=tensor.numel() * tensor.element_size()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'device': self.device,
            'size_bytes': self.size_bytes
        }


@dataclass
class OperationMetadata:
    """Metadata for remote operation requests."""
    operation: str
    result_id: str
    expects_result: bool = True
    num_inputs: int = 1
    input_shapes: List[List[int]] = field(default_factory=list)
    input_dtypes: List[str] = field(default_factory=list)
    client_port: Optional[int] = None

    # Semantic information
    phase: Optional[str] = None
    modality: Optional[Modality] = None

    # Additional context
    source_node: Optional[str] = None
    original_transfer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            'operation': self.operation,
            'result_id': self.result_id,
            'expects_result': self.expects_result,
            'num_inputs': self.num_inputs,
            'input_shapes': self.input_shapes,
            'input_dtypes': self.input_dtypes,
        }

        # Always include client_port if it's set (even if 0 or None, for debugging)
        if hasattr(self, 'client_port'):
            data['client_port'] = self.client_port

        if self.phase is not None:
            data['phase'] = self.phase
        if self.modality is not None:
            data['modality'] = self.modality.value
        if self.source_node is not None:
            data['source_node'] = self.source_node
        if self.original_transfer is not None:
            data['original_transfer'] = self.original_transfer

        return data


@dataclass
class ResultMetadata:
    """Metadata for successful operation results."""
    result_id: str
    original_transfer: str
    dtype: str
    shape: List[int]
    size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'is_result': True,
            'result_id': self.result_id,
            'original_transfer': self.original_transfer,
            'dtype': self.dtype,
            'shape': self.shape,
            'size_bytes': self.size_bytes
        }


@dataclass
class ErrorMetadata:
    """Metadata for error responses."""
    result_id: str
    original_transfer: str
    error_message: str
    error_type: str
    shape: List[int] = field(default_factory=lambda: [0])
    dtype: str = "torch.float32"
    size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'is_error': True,
            'is_result': True,  # Still considered a "result" (error result)
            'result_id': self.result_id,
            'error_message': self.error_message,
            'error_type': self.error_type,
            'original_transfer': self.original_transfer,
            'shape': self.shape,
            'dtype': self.dtype,
            'size_bytes': self.size_bytes
        }


@dataclass
class TransferMetadata:
    """Base metadata for all transfers."""
    transfer_id: str
    message_type: MessageType

    # Common fields
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransferMetadata':
        """Create TransferMetadata from dictionary."""
        transfer_id = data.get('transfer_id', '')
        message_type_str = data.get('message_type', 'tensor_transfer')
        message_type = MessageType(message_type_str)

        # Extract metadata, excluding transfer-specific fields
        metadata = {k: v for k, v in data.items()
                   if k not in ['transfer_id', 'message_type']}

        return cls(
            transfer_id=transfer_id,
            message_type=message_type,
            metadata=metadata
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = self.metadata.copy()
        result['transfer_id'] = self.transfer_id
        result['message_type'] = self.message_type.value
        return result


# Utility functions for creating metadata
def create_operation_metadata(
    operation: str,
    result_id: str,
    inputs: List[torch.Tensor],
    **kwargs
) -> OperationMetadata:
    """Create OperationMetadata from operation and inputs."""
    return OperationMetadata(
        operation=operation,
        result_id=result_id,
        expects_result=True,
        num_inputs=len(inputs),
        input_shapes=[list(t.shape) for t in inputs],
        input_dtypes=[str(t.dtype) for t in inputs],
        **kwargs
    )


def create_result_metadata(
    result_id: str,
    original_transfer: str,
    result: torch.Tensor
) -> ResultMetadata:
    """Create ResultMetadata from result tensor."""
    return ResultMetadata(
        result_id=result_id,
        original_transfer=original_transfer,
        dtype=str(result.dtype),
        shape=list(result.shape),
        size_bytes=result.numel() * result.element_size()
    )


def create_error_metadata(
    result_id: str,
    original_transfer: str,
    error: Exception
) -> ErrorMetadata:
    """Create ErrorMetadata from exception."""
    return ErrorMetadata(
        result_id=result_id,
        original_transfer=original_transfer,
        error_message=str(error),
        error_type=type(error).__name__
    )
