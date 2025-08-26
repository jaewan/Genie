"""
Protocol Constants and Types for Genie Zero-Copy Transport

This module defines protocol constants and types that are shared between
the Python control plane and C++ data plane. The actual packet processing
implementation is in C++ (src/data_plane/genie_data_plane.cpp).

This file only contains:
- Protocol constants (magic numbers, versions, etc.)
- Enum definitions (packet types, flags)
- Type definitions for Python-C++ interface

All actual packet building, parsing, fragmentation, and network processing
is handled by the C++ data plane for maximum performance.
"""

from enum import IntEnum
from typing import NamedTuple

# Protocol Constants (must match C++ implementation)
GENIE_MAGIC = 0x47454E49  # "GENI" in ASCII
GENIE_VERSION = 1
DEFAULT_MTU = 1500
DEFAULT_UDP_PORT = 5556

class PacketType(IntEnum):
    """Genie packet types (must match C++ enum)"""
    DATA = 0
    ACK = 1
    NACK = 2
    HEARTBEAT = 3
    CONTROL = 4

class PacketFlags(IntEnum):
    """Genie packet flags bitfield (must match C++ enum)"""
    NONE = 0x00
    FRAGMENTED = 0x01
    LAST_FRAGMENT = 0x02
    RETRANSMIT = 0x04
    COMPRESSED = 0x08
    ENCRYPTED = 0x10

class NetworkNode(NamedTuple):
    """Network node information"""
    id: str
    ip: str
    mac: str
    port: int = DEFAULT_UDP_PORT

class PacketInfo(NamedTuple):
    """Packet information for Python-C++ interface"""
    packet_type: PacketType
    flags: PacketFlags
    tensor_id: str
    seq_num: int
    frag_id: int
    frag_count: int
    offset: int
    length: int
    total_size: int
    timestamp_ns: int

# Header sizes (for reference, actual structures are in C++)
ETHERNET_HEADER_SIZE = 14  # dst_mac(6) + src_mac(6) + ethertype(2)
IPV4_HEADER_SIZE = 20      # Standard IPv4 header without options
UDP_HEADER_SIZE = 8        # src_port(2) + dst_port(2) + length(2) + checksum(2)
GENIE_HEADER_SIZE = 64     # Custom application header (see C++ GeniePacketHeader)

# Total header overhead
TOTAL_HEADER_SIZE = ETHERNET_HEADER_SIZE + IPV4_HEADER_SIZE + UDP_HEADER_SIZE + GENIE_HEADER_SIZE  # 106 bytes

# Maximum payload size per packet
MAX_PAYLOAD_SIZE = DEFAULT_MTU - TOTAL_HEADER_SIZE  # 1394 bytes

def calculate_fragments(data_size: int, mtu: int = DEFAULT_MTU) -> int:
    """Calculate number of fragments needed for data size"""
    max_payload = mtu - TOTAL_HEADER_SIZE
    return (data_size + max_payload - 1) // max_payload

def validate_network_node(node: NetworkNode) -> bool:
    """Validate network node configuration"""
    if not node.id or not node.ip or not node.mac:
        return False
    
    # Basic IP validation (IPv4)
    try:
        parts = node.ip.split('.')
        if len(parts) != 4:
            return False
        for part in parts:
            if not (0 <= int(part) <= 255):
                return False
    except (ValueError, AttributeError):
        return False
    
    # Basic MAC validation
    try:
        mac_parts = node.mac.split(':')
        if len(mac_parts) != 6:
            return False
        for part in mac_parts:
            if len(part) != 2:
                return False
            int(part, 16)  # Validate hex
    except (ValueError, AttributeError):
        return False
    
    # Port validation
    if not (1 <= node.port <= 65535):
        return False
    
    return True

# Note: All actual packet processing functions have been moved to C++
# The C++ data plane (src/data_plane/genie_data_plane.cpp) handles:
# - PacketBuilder: Building packets with all headers
# - PacketParser: Parsing incoming packets
# - FragmentManager: Fragmentation and reassembly
# - AddressResolver: IP/MAC address resolution
# - Checksum calculation and validation
# - Network byte order conversion
# - DPDK mbuf management
# - Zero-copy packet processing

# For reference, the original Python implementation is available in:
# legacy/protocol_python_reference.py
