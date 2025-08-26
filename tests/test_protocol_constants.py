"""
Tests for Protocol Constants and Types

Tests the Python protocol constants and types that are shared between
the Python control plane and C++ data plane.

NOTE: Actual packet processing is implemented in C++ (src/data_plane/genie_data_plane.cpp).
This only tests the constants and utility functions.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from genie.runtime.protocol import (
    GENIE_MAGIC, GENIE_VERSION, DEFAULT_MTU, DEFAULT_UDP_PORT,
    PacketType, PacketFlags, NetworkNode, PacketInfo,
    ETHERNET_HEADER_SIZE, IPV4_HEADER_SIZE, UDP_HEADER_SIZE, GENIE_HEADER_SIZE,
    TOTAL_HEADER_SIZE, MAX_PAYLOAD_SIZE,
    calculate_fragments, validate_network_node
)

class TestProtocolConstants:
    """Test protocol constants"""
    
    def test_magic_number(self):
        """Test magic number constant"""
        assert GENIE_MAGIC == 0x47454E49  # "GENI" in ASCII
        
        # Verify it's the correct ASCII representation
        magic_bytes = GENIE_MAGIC.to_bytes(4, 'big')
        assert magic_bytes == b'GENI'
    
    def test_version(self):
        """Test protocol version"""
        assert GENIE_VERSION == 1
        assert isinstance(GENIE_VERSION, int)
    
    def test_network_constants(self):
        """Test network-related constants"""
        assert DEFAULT_MTU == 1500
        assert DEFAULT_UDP_PORT == 5556
        assert isinstance(DEFAULT_UDP_PORT, int)
        assert 1 <= DEFAULT_UDP_PORT <= 65535

class TestPacketEnums:
    """Test packet type and flag enums"""
    
    def test_packet_types(self):
        """Test PacketType enum values"""
        assert PacketType.DATA == 0
        assert PacketType.ACK == 1
        assert PacketType.NACK == 2
        assert PacketType.HEARTBEAT == 3
        assert PacketType.CONTROL == 4
        
        # Test enum properties
        assert len(PacketType) == 5
        assert all(isinstance(pt.value, int) for pt in PacketType)
    
    def test_packet_flags(self):
        """Test PacketFlags enum values"""
        assert PacketFlags.NONE == 0x00
        assert PacketFlags.FRAGMENTED == 0x01
        assert PacketFlags.LAST_FRAGMENT == 0x02
        assert PacketFlags.RETRANSMIT == 0x04
        assert PacketFlags.COMPRESSED == 0x08
        assert PacketFlags.ENCRYPTED == 0x10
        
        # Test bitfield properties
        assert all(isinstance(pf.value, int) for pf in PacketFlags)
        
        # Test bitfield combinations
        combined = PacketFlags.FRAGMENTED | PacketFlags.LAST_FRAGMENT
        assert combined == 0x03

class TestHeaderSizes:
    """Test header size constants"""
    
    def test_individual_header_sizes(self):
        """Test individual header sizes"""
        assert ETHERNET_HEADER_SIZE == 14
        assert IPV4_HEADER_SIZE == 20
        assert UDP_HEADER_SIZE == 8
        assert GENIE_HEADER_SIZE == 64
    
    def test_total_header_size(self):
        """Test total header size calculation"""
        expected = ETHERNET_HEADER_SIZE + IPV4_HEADER_SIZE + UDP_HEADER_SIZE + GENIE_HEADER_SIZE
        assert TOTAL_HEADER_SIZE == expected
        assert TOTAL_HEADER_SIZE == 106  # 14 + 20 + 8 + 64
    
    def test_max_payload_size(self):
        """Test maximum payload size"""
        expected = DEFAULT_MTU - TOTAL_HEADER_SIZE
        assert MAX_PAYLOAD_SIZE == expected
        assert MAX_PAYLOAD_SIZE == 1394  # 1500 - 106

class TestNetworkNode:
    """Test NetworkNode named tuple"""
    
    def test_network_node_creation(self):
        """Test NetworkNode creation"""
        node = NetworkNode(
            id="test-node",
            ip="192.168.1.100",
            mac="aa:bb:cc:dd:ee:ff"
        )
        
        assert node.id == "test-node"
        assert node.ip == "192.168.1.100"
        assert node.mac == "aa:bb:cc:dd:ee:ff"
        assert node.port == DEFAULT_UDP_PORT  # Default value
    
    def test_network_node_with_port(self):
        """Test NetworkNode with custom port"""
        node = NetworkNode(
            id="test-node",
            ip="192.168.1.100",
            mac="aa:bb:cc:dd:ee:ff",
            port=8080
        )
        
        assert node.port == 8080

class TestPacketInfo:
    """Test PacketInfo named tuple"""
    
    def test_packet_info_creation(self):
        """Test PacketInfo creation"""
        info = PacketInfo(
            packet_type=PacketType.DATA,
            flags=PacketFlags.FRAGMENTED,
            tensor_id="test-tensor-123",
            seq_num=42,
            frag_id=1,
            frag_count=3,
            offset=1024,
            length=512,
            total_size=2048,
            timestamp_ns=1234567890
        )
        
        assert info.packet_type == PacketType.DATA
        assert info.flags == PacketFlags.FRAGMENTED
        assert info.tensor_id == "test-tensor-123"
        assert info.seq_num == 42
        assert info.frag_id == 1
        assert info.frag_count == 3
        assert info.offset == 1024
        assert info.length == 512
        assert info.total_size == 2048
        assert info.timestamp_ns == 1234567890

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_calculate_fragments(self):
        """Test fragment calculation"""
        # Single fragment
        assert calculate_fragments(100) == 1
        assert calculate_fragments(MAX_PAYLOAD_SIZE) == 1
        
        # Multiple fragments
        assert calculate_fragments(MAX_PAYLOAD_SIZE + 1) == 2
        assert calculate_fragments(MAX_PAYLOAD_SIZE * 2) == 2
        assert calculate_fragments(MAX_PAYLOAD_SIZE * 2 + 1) == 3
        
        # Large data
        large_size = 10 * 1024 * 1024  # 10MB
        expected_fragments = (large_size + MAX_PAYLOAD_SIZE - 1) // MAX_PAYLOAD_SIZE
        assert calculate_fragments(large_size) == expected_fragments
    
    def test_calculate_fragments_custom_mtu(self):
        """Test fragment calculation with custom MTU"""
        custom_mtu = 9000  # Jumbo frames
        max_payload = custom_mtu - TOTAL_HEADER_SIZE
        
        assert calculate_fragments(100, custom_mtu) == 1
        assert calculate_fragments(max_payload, custom_mtu) == 1
        assert calculate_fragments(max_payload + 1, custom_mtu) == 2
    
    def test_validate_network_node_valid(self):
        """Test network node validation with valid nodes"""
        valid_nodes = [
            NetworkNode("node1", "192.168.1.100", "aa:bb:cc:dd:ee:ff"),
            NetworkNode("node2", "10.0.0.1", "00:11:22:33:44:55", 8080),
            NetworkNode("node3", "172.16.0.1", "ff:ee:dd:cc:bb:aa", 1234),
        ]
        
        for node in valid_nodes:
            assert validate_network_node(node), f"Node should be valid: {node}"
    
    def test_validate_network_node_invalid_ip(self):
        """Test network node validation with invalid IPs"""
        invalid_nodes = [
            NetworkNode("node1", "256.1.1.1", "aa:bb:cc:dd:ee:ff"),  # Invalid IP octet
            NetworkNode("node2", "192.168.1", "aa:bb:cc:dd:ee:ff"),   # Incomplete IP
            NetworkNode("node3", "not.an.ip", "aa:bb:cc:dd:ee:ff"),   # Non-numeric
            NetworkNode("node4", "", "aa:bb:cc:dd:ee:ff"),            # Empty IP
        ]
        
        for node in invalid_nodes:
            assert not validate_network_node(node), f"Node should be invalid: {node}"
    
    def test_validate_network_node_invalid_mac(self):
        """Test network node validation with invalid MACs"""
        invalid_nodes = [
            NetworkNode("node1", "192.168.1.1", "aa:bb:cc:dd:ee"),     # Too short
            NetworkNode("node2", "192.168.1.1", "aa:bb:cc:dd:ee:ff:gg"), # Too long
            NetworkNode("node3", "192.168.1.1", "not:a:valid:mac:addr"), # Invalid hex
            NetworkNode("node4", "192.168.1.1", ""),                   # Empty MAC
        ]
        
        for node in invalid_nodes:
            assert not validate_network_node(node), f"Node should be invalid: {node}"
    
    def test_validate_network_node_invalid_port(self):
        """Test network node validation with invalid ports"""
        invalid_nodes = [
            NetworkNode("node1", "192.168.1.1", "aa:bb:cc:dd:ee:ff", 0),      # Port 0
            NetworkNode("node2", "192.168.1.1", "aa:bb:cc:dd:ee:ff", 65536),  # Port too high
            NetworkNode("node3", "192.168.1.1", "aa:bb:cc:dd:ee:ff", -1),     # Negative port
        ]
        
        for node in invalid_nodes:
            assert not validate_network_node(node), f"Node should be invalid: {node}"
    
    def test_validate_network_node_empty_fields(self):
        """Test network node validation with empty fields"""
        invalid_nodes = [
            NetworkNode("", "192.168.1.1", "aa:bb:cc:dd:ee:ff"),      # Empty ID
            NetworkNode("node1", "", "aa:bb:cc:dd:ee:ff"),            # Empty IP
            NetworkNode("node1", "192.168.1.1", ""),                 # Empty MAC
        ]
        
        for node in invalid_nodes:
            assert not validate_network_node(node), f"Node should be invalid: {node}"

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
