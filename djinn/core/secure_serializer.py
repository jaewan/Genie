"""
Secure request/response serializer for TCP protocol.

Replaces unsafe pickle with JSON + binary protocol:
- JSON for metadata (safe, human-readable)
- Binary for tensor data (efficient)
- No code execution risk

Protocol format:
[1 byte: version] [4 bytes: metadata_len] [N bytes: JSON metadata] [tensor data...]
"""

import json
import struct
import logging
from typing import Dict, Any, Optional, Tuple
import torch
import numpy as np

logger = logging.getLogger(__name__)

# Protocol version
PROTOCOL_VERSION = 1  # JSON + binary format

# Maximum metadata size (prevent DoS)
MAX_METADATA_SIZE = 10 * 1024 * 1024  # 10MB


class SecureSerializer:
    """Secure serializer for request/response messages."""
    
    @staticmethod
    def serialize_request(request: Dict[str, Any]) -> bytes:
        """
        Serialize request to secure binary format.
        
        Format:
        [1 byte: version] [4 bytes: metadata_len] [N bytes: JSON metadata] [tensor data...]
        
        ✅ OPTIMIZATION: If request has 'weights_binary', use direct binary protocol:
        [1 byte: version] [4 bytes: metadata_len] [N bytes: JSON metadata] [weights_binary]
        
        Args:
            request: Request dictionary (JSON-serializable metadata + optional tensors)
            
        Returns:
            Serialized bytes
        """
        # ✅ OPTIMIZATION: Check for direct binary weights (new protocol)
        # Supports both 'weights_binary' (full model) and 'chunk_data_binary' (chunks)
        has_binary_weights = ('weights_binary' in request and isinstance(request['weights_binary'], bytes))
        has_binary_chunk = ('chunk_data_binary' in request and isinstance(request['chunk_data_binary'], bytes))
        
        if has_binary_weights or has_binary_chunk:
            # Direct binary protocol - skip JSON for weights/chunks
            # Handle architecture_data specially (it's bytes, can't be in JSON)
            architecture_data = request.get('architecture_data')
            # Determine which binary field to use
            binary_field = 'weights_binary' if has_binary_weights else 'chunk_data_binary'
            binary_data = request[binary_field]
            
            metadata = {k: v for k, v in request.items() 
                       if k not in ('weights_binary', 'chunk_data_binary', 'architecture_data')}
            # ✅ TD-2 FIX: Ensure binary protocol flag is always set and preserved
            # Use multiple indicators to ensure detection works even if flag is lost
            metadata['_binary_protocol'] = True
            metadata['_has_binary_weights'] = has_binary_weights  # Additional indicator
            metadata['_has_binary_chunk'] = has_binary_chunk    # Additional indicator
            # Store architecture_data size in metadata (actual data sent separately)
            if architecture_data:
                metadata['_architecture_data_size'] = len(architecture_data)
            
            # ✅ TD-2 FIX: Validate flag will be preserved in JSON
            metadata_json = json.dumps(metadata, default=str).encode('utf-8')
            # Verify flag is in JSON (it should be, but check to be sure)
            metadata_decoded = json.loads(metadata_json.decode('utf-8'))
            if not metadata_decoded.get('_binary_protocol'):
                logger.warning("⚠️  Binary protocol flag lost during JSON encoding, using fallback detection")
            
            if len(metadata_json) > MAX_METADATA_SIZE:
                raise ValueError(f"Metadata too large: {len(metadata_json)} bytes (max: {MAX_METADATA_SIZE})")
            
            # Protocol: [version] [metadata_len] [metadata_json] [architecture_data_len] [architecture_data] [binary_data]
            # ✅ IMPORTANT: For binary protocol, we DON'T include tensor_count/binary_data_count headers
            # The server will detect this by checking _binary_protocol flag in metadata
            msg = bytearray()
            msg.append(PROTOCOL_VERSION)
            msg.extend(struct.pack('>I', len(metadata_json)))
            msg.extend(metadata_json)
            
            # Send architecture_data if present (before binary_data)
            if architecture_data:
                msg.extend(struct.pack('>Q', len(architecture_data)))  # 8 bytes: size
                msg.extend(architecture_data)  # N bytes: architecture data
            else:
                msg.extend(struct.pack('>Q', 0))  # 8 bytes: size = 0 (no architecture data)
            
            msg.extend(binary_data)  # Direct binary, no JSON, no tensor count headers
            
            return bytes(msg)
        
        # Legacy path: Separate metadata from tensor data
        metadata = {}
        tensor_data = {}
        
        for key, value in request.items():
            if isinstance(value, torch.Tensor):
                # Store tensor metadata, data will be serialized separately
                tensor_data[key] = value
                metadata[key] = {
                    '_is_tensor': True,
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                    'device': str(value.device)
                }
            elif isinstance(value, bytes):
                # ✅ IMPROVEMENT: For large binary data, send separately (like tensors)
                # Base64 adds 33% overhead, so avoid for large data
                BINARY_SIZE_THRESHOLD = 1024  # 1KB threshold
                
                if len(value) > BINARY_SIZE_THRESHOLD:
                    # Large binary data - send separately (like tensors)
                    tensor_data[key] = value  # Store as bytes
                    metadata[key] = {
                        '_is_binary': True,
                        'size': len(value)
                    }
                else:
                    # Small binary data - base64 is fine (overhead acceptable)
                    import base64
                    metadata[key] = {
                        '_is_bytes': True,
                        'data': base64.b64encode(value).decode('utf-8')
                    }
            elif isinstance(value, dict):
                # Check if it's a dict of tensors (like uncached_weights)
                has_tensors = any(isinstance(v, torch.Tensor) for v in value.values())
                if has_tensors:
                    # Dict of tensors - serialize each tensor separately
                    tensor_data[key] = value  # Store as special dict
                    # Store minimal metadata (just keys, not full tensor metadata)
                    # Full tensor metadata will be stored per-tensor in binary section
                    metadata[key] = {
                        '_is_tensor_dict': True,
                        'keys': list(value.keys())
                        # Don't store tensor_metadata here - it's too large for JSON
                        # Tensor metadata will be stored per-tensor in binary section
                    }
                # ✅ BUG FIX: Check for dict of serialized tensor dicts (like execution inputs)
                # Pattern: {'input_ids': {'data': bytes, 'shape': [...], 'format': '...'}, ...}
                elif len(value) > 0 and all(
                    isinstance(v, dict) and 
                    'data' in v and 
                    'format' in v and 
                    isinstance(v.get('data'), bytes)
                    for v in value.values()
                ):
                    # Dict of serialized tensor dicts (like execution inputs from _serialize_inputs)
                    # Send each as binary data, not JSON
                    tensor_data[key] = value  # Store as special dict
                    metadata[key] = {
                        '_is_serialized_dict': True,
                        'keys': list(value.keys()),
                        # Store metadata for each tensor (shape, dtype) for reconstruction
                        'tensors': {
                            k: {
                                'shape': v.get('shape', []),
                                'dtype': v.get('dtype', 'float32')
                            }
                            for k, v in value.items()
                        }
                    }
                elif isinstance(value, dict) and 'data' in value and 'format' in value and isinstance(value.get('data'), bytes):
                    # ✅ FIX: Single serialized tensor dict (from _serialize_tensor)
                    # Pattern: {'data': bytes, 'shape': [...], 'dtype': '...', 'format': '...'}
                    # Extract bytes directly and send as binary data
                    tensor_data[key] = value['data']  # Store as bytes
                    metadata[key] = {
                        '_is_binary': True,
                        'size': len(value['data']),
                        'shape': value.get('shape', []),
                        'dtype': value.get('dtype', 'float32'),
                        'format': value.get('format', 'numpy_binary')
                    }
                # Legacy dict-based serialization removed - use binary protocol instead
                # If you see this code path, it means something is still using old dict format
                # This should not happen - all weights should use _serialize_weights_binary()
                else:
                    # Regular dict - JSON-serializable
                    metadata[key] = value
            elif isinstance(value, (list, str, int, float, bool, type(None))):
                # JSON-serializable
                metadata[key] = value
            else:
                # Try to convert to JSON-serializable
                try:
                    json.dumps(value)  # Test if serializable
                    metadata[key] = value
                except (TypeError, ValueError):
                    logger.warning(f"Value for key '{key}' is not JSON-serializable, converting to string")
                    metadata[key] = str(value)
        
        # Serialize metadata to JSON
        try:
            metadata_json = json.dumps(metadata, default=str).encode('utf-8')
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to serialize metadata to JSON: {e}")
        
        if len(metadata_json) > MAX_METADATA_SIZE:
            raise ValueError(f"Metadata too large: {len(metadata_json)} bytes (max: {MAX_METADATA_SIZE})")
        
        # Build message
        msg = bytearray()
        msg.append(PROTOCOL_VERSION)  # 1 byte: version
        msg.extend(struct.pack('>I', len(metadata_json)))  # 4 bytes: metadata length
        msg.extend(metadata_json)  # N bytes: JSON metadata
        
        # Serialize tensors and binary data
        if tensor_data:
            # Count total tensors and binary data (including dicts of tensors)
            total_tensors = 0
            binary_data_count = 0
            for value in tensor_data.values():
                if isinstance(value, dict):
                    total_tensors += len(value)
                elif isinstance(value, bytes):
                    binary_data_count += 1
                else:
                    total_tensors += 1
            
            # Add tensor count (4 bytes) + binary data count (4 bytes)
            msg.extend(struct.pack('>I', total_tensors))
            msg.extend(struct.pack('>I', binary_data_count))
            
            # First, serialize binary data
            for key, value in tensor_data.items():
                if isinstance(value, bytes):
                    # Binary data - serialize like tensor but without shape/dtype
                    key_bytes = key.encode('utf-8')
                    msg.extend(struct.pack('>I', len(key_bytes)))
                    msg.extend(key_bytes)
                    msg.extend(struct.pack('>Q', len(value)))  # 8 bytes: size
                    msg.extend(value)  # N bytes: binary data
            
            # Then, serialize tensors
            for key, value in tensor_data.items():
                if isinstance(value, bytes):
                    continue  # Already handled above
                elif isinstance(value, dict):
                    # Check if this is a dict of already-serialized weights
                    is_serialized = any(isinstance(v, dict) and 'data' in v and 'format' in v for v in value.values())
                    
                    if is_serialized:
                        # Dict of serialized weights - serialize each as binary data
                        for sub_key, serialized_weight in value.items():
                            full_key = f"{key}.{sub_key}"  # e.g., "chunk_data.layer1.weight"
                            key_bytes = full_key.encode('utf-8')
                            msg.extend(struct.pack('>I', len(key_bytes)))
                            msg.extend(key_bytes)
                            
                            # Serialized weight is a dict with 'data', 'shape', 'dtype', 'format'
                            # Send the binary data directly
                            weight_data = serialized_weight['data']  # Already bytes
                            msg.extend(struct.pack('>Q', len(weight_data)))  # 8 bytes: data size
                            msg.extend(weight_data)  # N bytes: binary data
                    else:
                        # Dict of tensors - serialize each tensor with key prefix
                        for sub_key, tensor in value.items():
                            full_key = f"{key}.{sub_key}"  # e.g., "uncached_weights.layer1.weight"
                            key_bytes = full_key.encode('utf-8')
                            msg.extend(struct.pack('>I', len(key_bytes)))
                            msg.extend(key_bytes)
                            
                            # Convert tensor to numpy and serialize
                            if tensor.is_cuda:
                                tensor_cpu = tensor.cpu()
                            else:
                                tensor_cpu = tensor
                            
                            np_array = tensor_cpu.detach().numpy()
                            tensor_bytes = np_array.tobytes()
                            
                            msg.extend(struct.pack('>Q', len(tensor_bytes)))  # 8 bytes: tensor size
                            msg.extend(tensor_bytes)  # N bytes: tensor data
                else:
                    # Single tensor
                    key_bytes = key.encode('utf-8')
                    msg.extend(struct.pack('>I', len(key_bytes)))
                    msg.extend(key_bytes)
                    
                    # Convert tensor to numpy and serialize
                    if value.is_cuda:
                        tensor_cpu = value.cpu()
                    else:
                        tensor_cpu = value
                    
                    np_array = tensor_cpu.detach().numpy()
                    tensor_bytes = np_array.tobytes()
                    
                    msg.extend(struct.pack('>Q', len(tensor_bytes)))  # 8 bytes: tensor size
                    msg.extend(tensor_bytes)  # N bytes: tensor data
        else:
            # No tensors or binary data: [4 bytes: 0] [4 bytes: 0]
            msg.extend(struct.pack('>I', 0))  # tensor count
            msg.extend(struct.pack('>I', 0))  # binary data count
        
        return bytes(msg)
    
    @staticmethod
    def deserialize_request(data: bytes) -> Dict[str, Any]:
        """
        Deserialize request from secure binary format.
        
        Args:
            data: Serialized request bytes
            
        Returns:
            Deserialized request dictionary
        """
        if len(data) < 5:
            raise ValueError(f"Invalid request data: too short ({len(data)} bytes)")
        
        offset = 0
        
        # Read version
        version = data[offset]
        offset += 1
        
        if version != PROTOCOL_VERSION:
            raise ValueError(f"Unsupported protocol version: {version} (expected: {PROTOCOL_VERSION})")
        
        # Read metadata length
        metadata_len = struct.unpack('>I', data[offset:offset+4])[0]
        offset += 4
        
        if metadata_len > MAX_METADATA_SIZE:
            raise ValueError(f"Metadata too large: {metadata_len} bytes (max: {MAX_METADATA_SIZE})")
        
        # Read metadata
        metadata_json = data[offset:offset+metadata_len].decode('utf-8')
        offset += metadata_len
        
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse metadata JSON: {e}")
        
        # ✅ OPTIMIZATION: Check if this is direct binary protocol (new format)
        # Check for explicit flag first (most reliable)
        request_type = metadata.get('type', '')
        is_register_model = request_type == 'REGISTER_MODEL'
        is_register_chunk = request_type == 'REGISTER_MODEL_CHUNK'
        
        # ✅ BUG FIX: More robust binary protocol detection
        # Check multiple indicators, not just flag (in case flag is lost during JSON encoding)
        is_binary_protocol = (
            metadata.get('_binary_protocol') or  # Explicit flag (most reliable)
            'weights_binary' in metadata or      # Field presence (fallback)
            'chunk_data_binary' in metadata      # Field presence (fallback)
        ) and (is_register_model or is_register_chunk)
        
        # Log protocol detection for debugging
        logger.debug(
            f"Protocol detection: _binary_protocol={metadata.get('_binary_protocol')}, "
            f"type={request_type}, has_weights_binary={'weights_binary' in metadata}, "
            f"has_chunk_data_binary={'chunk_data_binary' in metadata}, "
            f"is_binary_protocol={is_binary_protocol}, data_len={len(data)}, offset={offset}"
        )
        
        if is_binary_protocol:
            # Direct binary protocol - read architecture_data first, then binary_data
            if len(data) < offset + 8:
                raise ValueError("Invalid binary protocol data: missing architecture_data size")
            
            arch_data_size = struct.unpack('>Q', data[offset:offset+8])[0]
            offset += 8
            
            # ✅ BUG FIX: Validate arch_data_size to catch protocol mismatches
            # If arch_data_size is suspiciously large, it might be message length being read incorrectly
            MAX_REASONABLE_ARCH_SIZE = 100 * 1024 * 1024  # 100MB max for architecture data
            if arch_data_size > MAX_REASONABLE_ARCH_SIZE:
                raise ValueError(
                    f"⚠️  Suspicious arch_data_size: {arch_data_size} bytes ({arch_data_size / (1024*1024):.1f} MB). "
                    f"This is likely a protocol mismatch - arch_data_size should be < 100MB. "
                    f"Check if binary protocol detection is working correctly."
                )
            
            architecture_data = None
            if arch_data_size > 0:
                if len(data) < offset + arch_data_size:
                    raise ValueError(f"Invalid binary protocol data: missing architecture_data (expected {arch_data_size} bytes)")
                architecture_data = data[offset:offset+arch_data_size]
                offset += arch_data_size
            
            # Remaining data is binary (weights_binary or chunk_data_binary)
            remaining_data = data[offset:]
            request = metadata.copy()
            
            # Determine which field to use based on request type
            if is_register_model:
                request['weights_binary'] = remaining_data
            elif is_register_chunk:
                request['chunk_data_binary'] = remaining_data
            
            if architecture_data:
                request['architecture_data'] = architecture_data
            
            # ✅ TD-6 FIX: Validate that binary protocol detection worked correctly
            # Verify we have the expected fields
            if is_register_model and 'weights_binary' not in request:
                raise ValueError(
                    f"Binary protocol detection failed: expected 'weights_binary' but got keys: {list(request.keys())}"
                )
            if is_register_chunk and 'chunk_data_binary' not in request:
                raise ValueError(
                    f"Binary protocol detection failed: expected 'chunk_data_binary' but got keys: {list(request.keys())}"
                )
            
            logger.debug(f"✅ Detected direct binary protocol (explicit flag, type={request_type}, arch_data={len(architecture_data) if architecture_data else 0} bytes, validated)")
            return request
        
        # Fallback: Try to detect by pattern (for backward compatibility)
        remaining_data = data[offset:]
        if len(remaining_data) > 0 and 'type' in metadata and metadata.get('type') == 'REGISTER_MODEL':
            # Check if remaining data looks like binary weights (starts with num_weights uint32)
            if len(remaining_data) >= 4:
                # Try to detect: first 4 bytes should be reasonable num_weights (< 10000)
                try:
                    num_weights = struct.unpack('>I', remaining_data[0:4])[0]
                    if 0 < num_weights < 10000:  # Reasonable range for model weights
                        # Additional check: second 4 bytes should be small (name length, not tensor count)
                        if len(remaining_data) >= 8:
                            second_word = struct.unpack('>I', remaining_data[4:8])[0]
                            if second_word < 1000:  # Small value = likely name length in binary format
                                # This is likely direct binary format
                                request = metadata.copy()
                                request['weights_binary'] = remaining_data
                                logger.debug(f"✅ Detected direct binary protocol (pattern): {num_weights} weights")
                                return request
                except:
                    pass  # Not binary format, continue with legacy path
        
        # Legacy path: Read tensor count and binary data count
        if len(data) < offset + 8:
            raise ValueError("Invalid request data: missing tensor/binary count")
        
        tensor_count = struct.unpack('>I', data[offset:offset+4])[0]
        offset += 4
        binary_data_count = struct.unpack('>I', data[offset:offset+4])[0]
        offset += 4
        
        # Deserialize tensors and bytes
        request = metadata.copy()
        
        # First, handle small bytes data (base64 encoded)
        for key, value in request.items():
            if isinstance(value, dict) and value.get('_is_bytes'):
                # Restore bytes from base64
                import base64
                request[key] = base64.b64decode(value['data'])
            elif isinstance(value, dict) and value.get('_is_tensor_dict'):
                # Initialize dict for tensor dict
                request[key] = {}
            elif isinstance(value, dict) and value.get('_is_serialized_dict'):
                # Initialize dict for serialized weight dict
                request[key] = {}
            elif isinstance(value, dict) and value.get('_is_binary'):
                # Large binary data - will be deserialized below
                # Store metadata for reconstruction (shape, dtype, format)
                # Don't overwrite - keep the metadata dict
                pass
        
        # Deserialize binary data first
        for _ in range(binary_data_count):
            if len(data) < offset + 4:
                raise ValueError("Invalid request data: missing binary key length")
            
            # Read binary key
            key_len = struct.unpack('>I', data[offset:offset+4])[0]
            offset += 4
            
            if len(data) < offset + key_len:
                raise ValueError("Invalid request data: missing binary key")
            
            key = data[offset:offset+key_len].decode('utf-8')
            offset += key_len
            
            # Read binary size
            if len(data) < offset + 8:
                raise ValueError("Invalid request data: missing binary size")
            
            binary_size = struct.unpack('>Q', data[offset:offset+8])[0]
            offset += 8
            
            # Read binary data
            if len(data) < offset + binary_size:
                raise ValueError(f"Invalid request data: missing binary data (expected {binary_size} bytes)")
            
            binary_data = data[offset:offset+binary_size]
            offset += binary_size
            
            # Store binary data
            # ✅ FIX: If this is a serialized tensor (has metadata), reconstruct dict
            if key in request and isinstance(request[key], dict) and request[key].get('_is_binary'):
                # Reconstruct serialized tensor dict (from _serialize_tensor)
                metadata = request[key]
                request[key] = {
                    'data': binary_data,
                    'shape': metadata.get('shape', []),
                    'dtype': metadata.get('dtype', 'float32'),
                    'format': metadata.get('format', 'numpy_binary')
                }
            else:
                # Plain binary data
                request[key] = binary_data
        
        # Then, deserialize tensors
        for _ in range(tensor_count):
            if len(data) < offset + 4:
                raise ValueError("Invalid request data: missing tensor key length")
            
            # Read tensor key
            key_len = struct.unpack('>I', data[offset:offset+4])[0]
            offset += 4
            
            if len(data) < offset + key_len:
                raise ValueError("Invalid request data: missing tensor key")
            
            full_key = data[offset:offset+key_len].decode('utf-8')
            offset += key_len
            
            # Check if this is a nested key (e.g., "uncached_weights.layer1.weight")
            if '.' in full_key:
                # Split into parent key and sub key
                parts = full_key.split('.', 1)
                parent_key = parts[0]
                sub_key = parts[1]
                
                # Ensure parent dict exists
                if parent_key not in request:
                    request[parent_key] = {}
                elif not isinstance(request[parent_key], dict):
                    request[parent_key] = {}
            else:
                parent_key = full_key
                sub_key = None
            
            # Read tensor size
            if len(data) < offset + 8:
                raise ValueError("Invalid request data: missing tensor size")
            
            tensor_size = struct.unpack('>Q', data[offset:offset+8])[0]
            offset += 8
            
            # Read tensor data
            if len(data) < offset + tensor_size:
                raise ValueError(f"Invalid request data: missing tensor data (expected {tensor_size} bytes)")
            
            tensor_bytes = data[offset:offset+tensor_size]
            offset += tensor_size
            
            # ✅ BUG FIX: Handle nested keys (e.g., "inputs.input_ids" or "chunk_data.layer1.weight")
            # For nested keys, look up metadata from parent key's tensor metadata
            if sub_key:
                # Nested tensor - get metadata from parent's tensor dict
                parent_metadata = metadata.get(parent_key, {})
                if isinstance(parent_metadata, dict) and parent_metadata.get('_is_serialized_dict'):
                    # ✅ BUG FIX: Handle _is_serialized_dict (execution inputs or chunked weights)
                    tensor_metadata = parent_metadata.get('tensors', {})
                    tensor_info = tensor_metadata.get(sub_key, {})
                elif isinstance(parent_metadata, dict) and parent_metadata.get('_is_tensor_dict'):
                    # Legacy _is_tensor_dict handling
                    tensor_metadata = parent_metadata.get('tensors', {})
                    tensor_info = tensor_metadata.get(sub_key, {})
                else:
                    tensor_info = {}
            else:
                # Simple tensor - get metadata directly
                tensor_info = metadata.get(full_key, {})
            
            shape = tuple(tensor_info.get('shape', []))
            dtype_str = tensor_info.get('dtype', 'torch.float32')
            
            # If shape is empty, try to infer from tensor size
            if not shape and tensor_size > 0:
                # This shouldn't happen, but handle gracefully
                logger.warning(f"No shape metadata for tensor {full_key}, inferring from size")
                # Can't infer shape without dtype, so use default
                shape = (tensor_size // 4,)  # Assume float32 as fallback
            
            # Parse dtype
            if dtype_str.startswith('torch.'):
                dtype_str = dtype_str[6:]
            
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'float64': torch.float64,
                'int64': torch.int64,
                'int32': torch.int32,
                'int8': torch.int8,
                'uint8': torch.uint8,
                'bool': torch.bool,
            }
            torch_dtype = dtype_map.get(dtype_str, torch.float32)
            
            # Convert bytes to numpy array
            numpy_dtype_map = {
                torch.float32: np.float32,
                torch.float16: np.float16,
                torch.float64: np.float64,
                torch.int64: np.int64,
                torch.int32: np.int32,
                torch.int8: np.int8,
                torch.uint8: np.uint8,
                torch.bool: np.bool_,
            }
            numpy_dtype = numpy_dtype_map.get(torch_dtype, np.float32)
            
            np_array = np.frombuffer(tensor_bytes, dtype=numpy_dtype)
            np_array = np_array.reshape(shape)
            
            # ✅ BUG FIX: Store as serialized tensor dict for _is_serialized_dict, raw tensor otherwise
            # For execution inputs (from _serialize_inputs), we need to keep the dict format
            # so deserialize_tensor_from_dict can handle it correctly
            if sub_key:
                # Nested key - store in parent dict
                parent_metadata = metadata.get(parent_key, {})
                if isinstance(parent_metadata, dict) and parent_metadata.get('_is_serialized_dict'):
                    # ✅ BUG FIX: Store as serialized tensor dict (execution inputs or chunked weights)
                    # This matches the format expected by deserialize_tensor_from_dict
                    request[parent_key][sub_key] = {
                        'data': tensor_bytes,  # ✅ Keep as bytes, not string!
                        'shape': shape,
                        'dtype': dtype_str,
                        'format': 'numpy_binary'
                    }
                else:
                    # Legacy path - store as raw tensor
                    tensor = torch.from_numpy(np_array.copy())
                request[parent_key][sub_key] = tensor
            else:
                # Simple key - store directly
                # Remove tensor metadata marker
                if full_key in request and isinstance(request[full_key], dict) and request[full_key].get('_is_tensor'):
                    del request[full_key]
                tensor = torch.from_numpy(np_array.copy())
                request[full_key] = tensor
        
        # Clean up metadata markers
        for key in list(request.keys()):
            if isinstance(request[key], dict):
                if request[key].get('_is_tensor') or request[key].get('_is_tensor_dict'):
                    # Already handled, but clean up if needed
                    pass
        
        return request
    
    @staticmethod
    def serialize_response(response: Dict[str, Any]) -> bytes:
        """Serialize response (same format as request)."""
        return SecureSerializer.serialize_request(response)
    
    @staticmethod
    def deserialize_response(data: bytes) -> Dict[str, Any]:
        """Deserialize response (same format as request)."""
        return SecureSerializer.deserialize_request(data)

