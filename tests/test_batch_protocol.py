"""
Unit Tests for Batch Protocol

Tests batch protocol pack/unpack functionality:
- Packing operations into binary requests
- Unpacking results from binary responses
- Error handling and edge cases
- Batch size estimation
- Round-trip encoding/decoding
"""

import pytest
import json
import sys

sys.path.insert(0, '/home/jae/Genie')

from djinn.server.optimizations.batch_protocol import (
    BatchProtocol,
    BatchProtocolError,
    BatchProtocolPackError,
    BatchProtocolUnpackError,
    REQUEST_TYPE_BATCH,
    STATUS_SUCCESS,
    STATUS_ERROR
)


class TestBatchProtocolPack:
    """Tests for packing operations into batch requests"""
    
    def test_pack_single_operation(self):
        """Test packing a single operation"""
        operations = [{'operation': 'test_op'}]
        data = BatchProtocol.pack_batch_request(operations)
        
        assert data[0] == REQUEST_TYPE_BATCH
        assert data[1] == 1  # count
        assert len(data) > 2
    
    def test_pack_multiple_operations(self):
        """Test packing multiple operations"""
        operations = [
            {'operation': 'op1'},
            {'operation': 'op2'},
            {'operation': 'op3'},
        ]
        data = BatchProtocol.pack_batch_request(operations)
        
        assert data[0] == REQUEST_TYPE_BATCH
        assert data[1] == 3  # count
        assert len(data) > 10
    
    def test_pack_with_metadata(self):
        """Test packing operations with metadata"""
        operations = [
            {'operation': 'gpt2_decode', 'metadata': {'batch': 1}},
            {'operation': 'gpt2_decode', 'metadata': {'batch': 2}},
        ]
        data = BatchProtocol.pack_batch_request(operations)
        
        assert data[0] == REQUEST_TYPE_BATCH
        assert data[1] == 2
        assert len(data) > 20
    
    def test_pack_empty_operations_raises(self):
        """Test packing empty list raises error"""
        with pytest.raises(ValueError, match="empty"):
            BatchProtocol.pack_batch_request([])
    
    def test_pack_missing_operation_field_raises(self):
        """Test missing 'operation' field raises error"""
        with pytest.raises(ValueError, match="operation"):
            BatchProtocol.pack_batch_request([{'metadata': {}}])
    
    def test_pack_large_operation_names(self):
        """Test packing with large operation names"""
        long_name = 'a' * 1000
        operations = [{'operation': long_name}]
        data = BatchProtocol.pack_batch_request(operations)
        
        # Should handle large names
        assert len(data) > 1000
    
    def test_pack_complex_metadata(self):
        """Test packing with complex nested metadata"""
        operations = [
            {
                'operation': 'complex_op',
                'metadata': {
                    'nested': {'deep': {'value': 123}},
                    'list': [1, 2, 3],
                    'string': 'test'
                }
            }
        ]
        data = BatchProtocol.pack_batch_request(operations)
        
        assert data[0] == REQUEST_TYPE_BATCH
        assert len(data) > 50


class TestBatchProtocolUnpack:
    """Tests for unpacking results from batch responses"""
    
    def test_unpack_single_result(self):
        """Test unpacking a single result"""
        results = [{'output': 'token1'}]
        data = BatchProtocol.pack_batch_response(results)
        
        unpacked = BatchProtocol.unpack_batch_response(data)
        assert len(unpacked) == 1
        assert unpacked[0] == {'output': 'token1'}
    
    def test_unpack_multiple_results(self):
        """Test unpacking multiple results"""
        results = [
            {'output': 'token1'},
            {'output': 'token2'},
            {'output': 'token3'},
        ]
        data = BatchProtocol.pack_batch_response(results)
        
        unpacked = BatchProtocol.unpack_batch_response(data)
        assert len(unpacked) == 3
        assert unpacked[0] == {'output': 'token1'}
        assert unpacked[2] == {'output': 'token3'}
    
    def test_unpack_complex_results(self):
        """Test unpacking complex result objects"""
        results = [
            {'output': 'token1', 'logits': [0.1, 0.2, 0.7], 'hidden': [1, 2, 3]},
            {'output': 'token2', 'logits': [0.2, 0.3, 0.5], 'hidden': [4, 5, 6]},
        ]
        data = BatchProtocol.pack_batch_response(results)
        
        unpacked = BatchProtocol.unpack_batch_response(data)
        assert len(unpacked) == 2
        assert unpacked[0]['logits'] == [0.1, 0.2, 0.7]
        assert unpacked[1]['hidden'] == [4, 5, 6]
    
    def test_unpack_too_short_data_raises(self):
        """Test unpacking data too short raises error"""
        with pytest.raises(ValueError, match="too short"):
            BatchProtocol.unpack_batch_response(b'')
        
        with pytest.raises(ValueError, match="too short"):
            BatchProtocol.unpack_batch_response(b'\x00')
    
    def test_unpack_error_status_raises(self):
        """Test unpacking error status raises error"""
        # Create response with error status
        error_data = bytes([STATUS_ERROR, 0])
        
        with pytest.raises(RuntimeError, match="failed"):
            BatchProtocol.unpack_batch_response(error_data)
    
    def test_unpack_truncated_result_raises(self):
        """Test unpacking truncated result raises error"""
        # Create malformed response
        bad_data = bytes([STATUS_SUCCESS, 1, 0xFF, 0xFF, 0xFF, 0xFF])  # large size but no data
        
        with pytest.raises(ValueError, match="truncated"):
            BatchProtocol.unpack_batch_response(bad_data)
    
    def test_unpack_invalid_json_raises(self):
        """Test unpacking invalid JSON raises error"""
        # Create response with invalid JSON
        bad_json = b'{invalid json}'
        response = bytearray([STATUS_SUCCESS, 1])
        response.extend(len(bad_json).to_bytes(4, byteorder='big'))
        response.extend(bad_json)
        
        with pytest.raises(ValueError, match="Invalid JSON"):
            BatchProtocol.unpack_batch_response(bytes(response))


class TestBatchProtocolRoundTrip:
    """Tests for round-trip encode/decode"""
    
    def test_roundtrip_pack_unpack(self):
        """Test packing operations and unpacking results round-trip"""
        operations = [
            {'operation': 'op1', 'metadata': {'id': 1}},
            {'operation': 'op2', 'metadata': {'id': 2}},
        ]
        
        # Pack request
        request_data = BatchProtocol.pack_batch_request(operations)
        
        # Verify it's a valid batch request
        assert request_data[0] == REQUEST_TYPE_BATCH
        assert request_data[1] == 2
        
        # Create matching results
        results = [
            {'output': 'result1'},
            {'output': 'result2'},
        ]
        
        # Pack response
        response_data = BatchProtocol.pack_batch_response(results)
        
        # Unpack and verify
        unpacked = BatchProtocol.unpack_batch_response(response_data)
        assert unpacked == results
    
    def test_roundtrip_preserves_data(self):
        """Test round-trip preserves all data accurately"""
        original_results = [
            {'token': 'hello', 'score': 0.95, 'position': 0},
            {'token': 'world', 'score': 0.92, 'position': 1},
            {'token': '!', 'score': 0.88, 'position': 2},
        ]
        
        # Pack and unpack
        packed = BatchProtocol.pack_batch_response(original_results)
        unpacked = BatchProtocol.unpack_batch_response(packed)
        
        # Verify exact match
        assert unpacked == original_results
        assert len(unpacked) == 3
    
    def test_roundtrip_large_batch(self):
        """Test round-trip with large batch"""
        # Create large batch
        results = [
            {'id': i, 'data': f'result_{i}', 'value': i * 1.5}
            for i in range(100)
        ]
        
        packed = BatchProtocol.pack_batch_response(results)
        unpacked = BatchProtocol.unpack_batch_response(packed)
        
        assert len(unpacked) == 100
        assert unpacked[0] == results[0]
        assert unpacked[99] == results[99]


class TestBatchProtocolEdgeCases:
    """Tests for edge cases and boundary conditions"""
    
    def test_pack_special_characters(self):
        """Test packing operations with special characters"""
        operations = [
            {'operation': 'op_with_unicode_✓', 'metadata': {'msg': '你好世界'}},
        ]
        data = BatchProtocol.pack_batch_request(operations)
        
        # Should successfully pack and be valid
        assert data[0] == REQUEST_TYPE_BATCH
        assert len(data) > 10
    
    def test_pack_empty_metadata(self):
        """Test packing with empty metadata"""
        operations = [
            {'operation': 'op1'},  # No metadata key
            {'operation': 'op2', 'metadata': {}},  # Empty metadata
        ]
        data = BatchProtocol.pack_batch_request(operations)
        
        assert data[0] == REQUEST_TYPE_BATCH
        assert data[1] == 2
    
    def test_unpack_max_size_result(self):
        """Test unpacking with maximum size result"""
        # Create result with large data
        large_data = 'x' * 10000
        results = [{'data': large_data}]
        
        packed = BatchProtocol.pack_batch_response(results)
        unpacked = BatchProtocol.unpack_batch_response(packed)
        
        assert unpacked[0]['data'] == large_data
    
    def test_pack_numeric_operation_names(self):
        """Test that numeric-like operation names work"""
        operations = [
            {'operation': '12345'},
            {'operation': '0.5'},
            {'operation': 'op_123'},
        ]
        data = BatchProtocol.pack_batch_request(operations)
        
        assert data[0] == REQUEST_TYPE_BATCH
        assert data[1] == 3


class TestBatchSizeEstimation:
    """Tests for batch size estimation"""
    
    def test_estimate_single_operation(self):
        """Test estimating size of single operation"""
        operations = [{'operation': 'test'}]
        estimate = BatchProtocol.get_batch_size_estimate(operations)
        
        # Should be reasonable estimate
        assert estimate > 0
        assert estimate < 100
    
    def test_estimate_multiple_operations(self):
        """Test estimating size of multiple operations"""
        operations = [
            {'operation': f'op_{i}', 'metadata': {'id': i}}
            for i in range(10)
        ]
        estimate = BatchProtocol.get_batch_size_estimate(operations)
        
        # Estimate should grow with number of operations
        assert estimate > 100
    
    def test_estimate_matches_actual_size(self):
        """Test that estimate is close to actual packed size"""
        operations = [
            {'operation': 'gpt2_decode', 'metadata': {'batch': 1}},
            {'operation': 'gpt2_decode', 'metadata': {'batch': 2}},
        ]
        
        estimate = BatchProtocol.get_batch_size_estimate(operations)
        actual = len(BatchProtocol.pack_batch_request(operations))
        
        # Estimate should be close (within 10%)
        assert estimate > 0
        assert abs(estimate - actual) < actual * 0.1


class TestBatchProtocolConsistency:
    """Tests for protocol consistency and format"""
    
    def test_request_format_structure(self):
        """Test that request format follows protocol specification"""
        operations = [{'operation': 'test_op'}]
        data = BatchProtocol.pack_batch_request(operations)
        
        # Check header
        assert data[0] == REQUEST_TYPE_BATCH  # Request type
        assert data[1] == 1  # Batch count
        
        # Should have more data after header
        assert len(data) > 2
    
    def test_response_format_structure(self):
        """Test that response format follows protocol specification"""
        results = [{'output': 'test'}]
        data = BatchProtocol.pack_batch_response(results)
        
        # Check header
        assert data[0] == STATUS_SUCCESS  # Status
        assert data[1] == 1  # Result count
        
        # Should have size prefix for result (4 bytes)
        assert len(data) > 6
        
        # Extract size
        size = int.from_bytes(data[2:6], byteorder='big')
        assert size > 0
    
    def test_multiple_packs_same_data(self):
        """Test that packing same data produces same result"""
        operations = [
            {'operation': 'op1', 'metadata': {'x': 1}},
            {'operation': 'op2', 'metadata': {'x': 2}},
        ]
        
        pack1 = BatchProtocol.pack_batch_request(operations)
        pack2 = BatchProtocol.pack_batch_request(operations)
        
        # Should produce identical results
        assert pack1 == pack2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
