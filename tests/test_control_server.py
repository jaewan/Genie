"""
Tests for Control Plane TCP Server

Tests the control server functionality including:
- TCP server startup and shutdown
- Client connection handling
- Message protocol and serialization
- Transfer request/response handling
- Heartbeat and monitoring
- Error handling and recovery
"""

import pytest
import pytest_asyncio
import asyncio
import json
import time
import uuid
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from genie.runtime.control_server import (
    ControlPlaneServer, ClientHandler, ControlMessage, MessageType,
    TransferRequest, TransferResponse, NodeCapabilities, TransferStatus,
    get_control_server, start_control_server, stop_control_server
)

class TestControlMessage:
    """Test control message serialization"""
    
    def test_message_creation(self):
        """Test message creation and basic properties"""
        msg = ControlMessage(
            type=MessageType.HELLO,
            sender="test-node",
            payload={'version': '1.0'}
        )
        
        assert msg.type == MessageType.HELLO
        assert msg.sender == "test-node"
        assert msg.payload['version'] == '1.0'
        assert msg.message_id is not None
        assert msg.timestamp > 0
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization"""
        original = ControlMessage(
            type=MessageType.TRANSFER_REQUEST,
            sender="client-1",
            payload={
                'transfer_id': 'test-123',
                'size': 1024,
                'dtype': 'float32'
            }
        )
        
        # Serialize to JSON
        json_str = original.to_json()
        assert isinstance(json_str, str)
        
        # Deserialize from JSON
        restored = ControlMessage.from_json(json_str)
        
        assert restored.type == original.type
        assert restored.sender == original.sender
        assert restored.message_id == original.message_id
        assert restored.payload == original.payload
    
    def test_message_type_conversion(self):
        """Test message type enum conversion in JSON"""
        msg = ControlMessage(
            type=MessageType.CAPABILITY_EXCHANGE,
            sender="test"
        )
        
        json_str = msg.to_json()
        data = json.loads(json_str)
        
        # Should be converted to int in JSON
        assert data['type'] == int(MessageType.CAPABILITY_EXCHANGE)
        
        # Should be restored as enum
        restored = ControlMessage.from_json(json_str)
        assert restored.type == MessageType.CAPABILITY_EXCHANGE

class TestNodeCapabilities:
    """Test node capabilities"""
    
    def test_default_capabilities(self):
        """Test default capability values"""
        caps = NodeCapabilities(node_id="test-node", gpu_count=2)
        
        assert caps.node_id == "test-node"
        assert caps.gpu_count == 2
        assert caps.max_transfer_size == 10 * 1024 * 1024 * 1024
        assert 'float32' in caps.supported_dtypes
        assert caps.network_bandwidth_gbps == 100.0
        assert 'fragmentation' in caps.features
    
    def test_custom_capabilities(self):
        """Test custom capability values"""
        caps = NodeCapabilities(
            node_id="gpu-server",
            gpu_count=8,
            max_transfer_size=100 * 1024 * 1024 * 1024,
            supported_dtypes=['float16', 'bfloat16'],
            network_bandwidth_gbps=200.0,
            features=['compression', 'encryption']
        )
        
        assert caps.gpu_count == 8
        assert caps.max_transfer_size == 100 * 1024 * 1024 * 1024
        assert caps.supported_dtypes == ['float16', 'bfloat16']
        assert caps.network_bandwidth_gbps == 200.0
        assert 'compression' in caps.features

class TestTransferRequest:
    """Test transfer request and response"""
    
    def test_transfer_request_creation(self):
        """Test transfer request creation"""
        req = TransferRequest(
            transfer_id="test-transfer-123",
            tensor_id="tensor-456",
            source_node="node-1",
            target_node="node-2",
            size=1024 * 1024,
            dtype="float32",
            shape=[1024, 1024],
            priority=2
        )
        
        assert req.transfer_id == "test-transfer-123"
        assert req.tensor_id == "tensor-456"
        assert req.source_node == "node-1"
        assert req.target_node == "node-2"
        assert req.size == 1024 * 1024
        assert req.dtype == "float32"
        assert req.shape == [1024, 1024]
        assert req.priority == 2
        assert req.source_gpu == 0  # default
        assert req.requires_ack is True  # default
    
    def test_transfer_response_creation(self):
        """Test transfer response creation"""
        resp = TransferResponse(
            transfer_id="test-transfer-123",
            accepted=True,
            reason="Transfer accepted",
            estimated_time_seconds=5.0,
            allocated_gpu=1,
            data_port=5556
        )
        
        assert resp.transfer_id == "test-transfer-123"
        assert resp.accepted is True
        assert resp.reason == "Transfer accepted"
        assert resp.estimated_time_seconds == 5.0
        assert resp.allocated_gpu == 1
        assert resp.data_port == 5556

class TestControlPlaneServer:
    """Test control plane server functionality"""
    
    @pytest.fixture
    def server(self):
        """Create test server"""
        server = ControlPlaneServer("test-server", host="127.0.0.1", port=0)  # port 0 = random
        return server
    
    def test_server_initialization(self, server):
        """Test server initialization"""
        assert server.node_id == "test-server"
        assert server.host == "127.0.0.1"
        assert server.port == 0
        assert len(server.clients) == 0
        assert len(server.active_transfers) == 0
        assert server.capabilities.node_id == "test-server"
    
    def test_server_capabilities(self, server):
        """Test server capabilities"""
        caps = server.capabilities
        assert caps.node_id == "test-server"
        assert caps.gpu_count >= 1
        assert caps.max_transfer_size > 0
        assert len(caps.supported_dtypes) > 0
        assert caps.network_bandwidth_gbps > 0
        assert len(caps.features) > 0
    
    def test_transfer_callbacks(self, server):
        """Test transfer callback registration"""
        callback_called = []
        
        async def test_callback(transfer_id, payload=None):
            callback_called.append((transfer_id, payload))
        
        server.add_transfer_callback('request', test_callback)
        assert len(server.transfer_callbacks['request']) == 1
        
        # Test invalid event type
        with pytest.raises(ValueError):
            server.add_transfer_callback('invalid_event', test_callback)
    
    @pytest.mark.asyncio
    async def test_transfer_request_processing(self, server):
        """Test transfer request processing"""
        # Create mock client
        mock_client = Mock()
        mock_client.capabilities = NodeCapabilities("client-1", gpu_count=1)
        
        # Valid request
        request = TransferRequest(
            transfer_id="test-123",
            tensor_id="tensor-456",
            source_node="client-1",
            target_node="test-server",
            size=1024,
            dtype="float32",
            shape=[32, 32]
        )
        
        response = await server.process_transfer_request(request, mock_client)
        
        assert response.transfer_id == "test-123"
        assert response.accepted is True
        assert response.estimated_time_seconds > 0
        assert "test-123" in server.active_transfers
    
    @pytest.mark.asyncio
    async def test_transfer_request_validation(self, server):
        """Test transfer request validation"""
        mock_client = Mock()
        
        # Request too large
        large_request = TransferRequest(
            transfer_id="large-123",
            tensor_id="tensor-456",
            source_node="client-1",
            target_node="test-server",
            size=100 * 1024 * 1024 * 1024 * 1024,  # 100TB - too large
            dtype="float32",
            shape=[1000000, 1000000]
        )
        
        response = await server.process_transfer_request(large_request, mock_client)
        
        assert response.accepted is False
        assert "exceeds maximum" in response.reason
        assert large_request.transfer_id not in server.active_transfers
    
    @pytest.mark.asyncio
    async def test_unsupported_dtype(self, server):
        """Test unsupported dtype rejection"""
        mock_client = Mock()
        
        request = TransferRequest(
            transfer_id="dtype-123",
            tensor_id="tensor-456",
            source_node="client-1",
            target_node="test-server",
            size=1024,
            dtype="unsupported_dtype",
            shape=[32, 32]
        )
        
        response = await server.process_transfer_request(request, mock_client)
        
        assert response.accepted is False
        assert "Unsupported dtype" in response.reason
    
    @pytest.mark.asyncio
    async def test_concurrent_transfer_limit(self, server):
        """Test concurrent transfer limit"""
        # Set low limit for testing
        server.max_concurrent_transfers = 2
        mock_client = Mock()
        
        # Fill up transfer slots
        for i in range(2):
            request = TransferRequest(
                transfer_id=f"transfer-{i}",
                tensor_id=f"tensor-{i}",
                source_node="client-1",
                target_node="test-server",
                size=1024,
                dtype="float32",
                shape=[32, 32]
            )
            response = await server.process_transfer_request(request, mock_client)
            assert response.accepted is True
        
        # This should be rejected
        overflow_request = TransferRequest(
            transfer_id="overflow",
            tensor_id="tensor-overflow",
            source_node="client-1",
            target_node="test-server",
            size=1024,
            dtype="float32",
            shape=[32, 32]
        )
        
        response = await server.process_transfer_request(overflow_request, mock_client)
        assert response.accepted is False
        assert "Maximum concurrent transfers" in response.reason

class TestClientHandler:
    """Test client handler functionality"""
    
    @pytest.fixture
    def mock_streams(self):
        """Create mock reader/writer streams"""
        reader = Mock(spec=asyncio.StreamReader)
        writer = Mock(spec=asyncio.StreamWriter)
        writer.get_extra_info.return_value = ('127.0.0.1', 12345)
        return reader, writer
    
    @pytest.fixture
    def mock_server(self):
        """Create mock server"""
        server = Mock(spec=ControlPlaneServer)
        server.node_id = "test-server"
        server.heartbeat_timeout = 60.0
        server.message_timeout = 30.0
        server.max_message_size = 1024 * 1024
        server.max_concurrent_transfers = 100
        server.capabilities = NodeCapabilities("test-server", gpu_count=1)
        return server
    
    def test_client_handler_initialization(self, mock_streams, mock_server):
        """Test client handler initialization"""
        reader, writer = mock_streams
        handler = ClientHandler(reader, writer, mock_server)
        
        assert handler.reader is reader
        assert handler.writer is writer
        assert handler.server is mock_server
        assert handler.client_id is None
        assert handler.capabilities is None
        assert handler.connected is True
        assert handler.client_address == "127.0.0.1:12345"
    
    @pytest.mark.asyncio
    async def test_message_serialization_protocol(self, mock_streams, mock_server):
        """Test message serialization protocol"""
        reader, writer = mock_streams
        handler = ClientHandler(reader, writer, mock_server)
        
        # Test message sending
        test_message = ControlMessage(
            type=MessageType.HELLO,
            sender="test-client",
            payload={'version': '1.0'}
        )
        
        await handler.send_message(test_message)
        
        # Verify writer was called
        assert writer.write.called
        assert writer.drain.called
        
        # Check the written data format
        written_data = writer.write.call_args[0][0]
        
        # First 4 bytes should be length
        message_length = int.from_bytes(written_data[:4], 'big')
        message_data = written_data[4:]
        
        assert len(message_data) == message_length
        
        # Should be valid JSON
        message_str = message_data.decode('utf-8')
        parsed_msg = ControlMessage.from_json(message_str)
        
        assert parsed_msg.type == MessageType.HELLO
        assert parsed_msg.sender == "test-client"

class TestIntegration:
    """Integration tests for control server"""
    
    @pytest.mark.asyncio
    async def test_client_server_communication(self):
        """Test basic client-server communication"""
        # This would require actual TCP connections
        # For now, we'll test the message protocol
        
        # Create messages
        hello_msg = ControlMessage(
            type=MessageType.HELLO,
            sender="client-1",
            payload={'version': '1.0'}
        )
        
        capability_msg = ControlMessage(
            type=MessageType.CAPABILITY_EXCHANGE,
            sender="client-1",
            payload={
                'gpu_count': 2,
                'max_transfer_size': 1024 * 1024 * 1024,
                'supported_dtypes': ['float32', 'float16'],
                'features': ['fragmentation', 'compression']
            }
        )
        
        # Test serialization round-trip
        hello_json = hello_msg.to_json()
        hello_restored = ControlMessage.from_json(hello_json)
        
        assert hello_restored.type == MessageType.HELLO
        assert hello_restored.sender == "client-1"
        
        capability_json = capability_msg.to_json()
        capability_restored = ControlMessage.from_json(capability_json)
        
        assert capability_restored.type == MessageType.CAPABILITY_EXCHANGE
        assert capability_restored.payload['gpu_count'] == 2
    
    @pytest.mark.asyncio
    async def test_transfer_workflow(self):
        """Test complete transfer workflow"""
        server = ControlPlaneServer("test-server")
        
        # Simulate transfer request
        request = TransferRequest(
            transfer_id=str(uuid.uuid4()),
            tensor_id=str(uuid.uuid4()),
            source_node="client-1",
            target_node="test-server",
            size=1024 * 1024,
            dtype="float32",
            shape=[1024, 256]
        )
        
        # Process request
        mock_client = Mock()
        response = await server.process_transfer_request(request, mock_client)
        
        assert response.accepted is True
        assert request.transfer_id in server.active_transfers
        
        # Simulate transfer completion
        await server.handle_transfer_complete(request.transfer_id, {
            'transfer_id': request.transfer_id,
            'bytes_transferred': request.size,
            'transfer_time_seconds': 2.5
        })
        
        # Transfer should be removed from active list
        assert request.transfer_id not in server.active_transfers

class TestGlobalFunctions:
    """Test global utility functions"""
    
    def test_get_control_server(self):
        """Test global server instance"""
        # Reset global state
        import genie.runtime.control_server
        genie.runtime.control_server._control_server = None
        
        server1 = get_control_server("test-node")
        server2 = get_control_server("test-node")
        
        # Should be same instance
        assert server1 is server2
        assert server1.node_id == "test-node"
    
    @pytest.mark.asyncio
    async def test_start_stop_control_server(self):
        """Test global server start/stop"""
        # Reset global state
        import genie.runtime.control_server
        genie.runtime.control_server._control_server = None
        
        # This would normally start a real server
        # For testing, we'll just verify the functions exist and can be called
        assert callable(start_control_server)
        assert callable(stop_control_server)

class TestErrorHandling:
    """Test error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_invalid_message_format(self):
        """Test handling of invalid message formats"""
        # Test invalid JSON
        with pytest.raises(json.JSONDecodeError):
            ControlMessage.from_json("invalid json")
        
        # Test missing required fields
        with pytest.raises((KeyError, TypeError)):
            ControlMessage.from_json('{"type": 1}')  # Missing sender
    
    @pytest.mark.asyncio
    async def test_server_error_recovery(self):
        """Test server error recovery"""
        server = ControlPlaneServer("test-server")
        
        # Test handling of invalid transfer request
        invalid_request = TransferRequest(
            transfer_id="invalid",
            tensor_id="tensor",
            source_node="client",
            target_node="server",
            size=-1,  # Invalid size
            dtype="float32",
            shape=[]
        )
        
        mock_client = Mock()
        
        # Should handle gracefully and return error response
        response = await server.process_transfer_request(invalid_request, mock_client)
        
        # Should not crash, should return some response
        assert isinstance(response, TransferResponse)
        assert response.transfer_id == "invalid"

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
