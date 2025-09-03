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

# Legacy ctypes-compatible structures and helpers for tests
try:
    import ctypes  # type: ignore
except Exception:  # pragma: no cover
    ctypes = None  # type: ignore

if ctypes is not None:
    class EthernetHeader(ctypes.Structure):
        _pack_ = 1
        _fields_ = [
            ("dst_mac", ctypes.c_uint8 * 6),
            ("src_mac", ctypes.c_uint8 * 6),
            ("ethertype", ctypes.c_uint16),
        ]
        def __init__(self):  # type: ignore[no-redef]
            super().__init__()
            self.ethertype = 0x0800  # IPv4

    class IPv4Header(ctypes.Structure):
        _pack_ = 1
        _fields_ = [
            ("version_ihl", ctypes.c_uint8),
            ("tos", ctypes.c_uint8),
            ("total_length", ctypes.c_uint16),
            ("identification", ctypes.c_uint16),
            ("flags_fragment", ctypes.c_uint16),
            ("ttl", ctypes.c_uint8),
            ("protocol", ctypes.c_uint8),
            ("checksum", ctypes.c_uint16),
            ("src_ip", ctypes.c_uint32),
            ("dst_ip", ctypes.c_uint32),
        ]
        def __init__(self):  # type: ignore[no-redef]
            super().__init__()
            self.version_ihl = 0x45
            self.ttl = 64
            self.protocol = 17  # UDP

    class UDPHeader(ctypes.Structure):
        _pack_ = 1
        _fields_ = [
            ("src_port", ctypes.c_uint16),
            ("dst_port", ctypes.c_uint16),
            ("length", ctypes.c_uint16),
            ("checksum", ctypes.c_uint16),
        ]

    class GeniePacketHeader(ctypes.Structure):
        _pack_ = 1
        _fields_ = [
            ("magic", ctypes.c_uint32),
            ("version", ctypes.c_uint8),
            ("flags", ctypes.c_uint8),
            ("type", ctypes.c_uint8),
            ("reserved", ctypes.c_uint8),
            ("tensor_id", ctypes.c_uint8 * 16),
            ("seq_num", ctypes.c_uint32),
            ("frag_id", ctypes.c_uint16),
            ("frag_count", ctypes.c_uint16),
            ("offset", ctypes.c_uint32),
            ("length", ctypes.c_uint32),
            ("total_size", ctypes.c_uint32),
            ("checksum", ctypes.c_uint32),
            ("timestamp_ns", ctypes.c_uint64),
            ("padding", ctypes.c_uint8 * 8),
        ]

        def __init__(self):  # type: ignore[no-redef]
            super().__init__()
            self.magic = GENIE_MAGIC
            self.version = GENIE_VERSION
            try:
                import time as _time
                self.timestamp_ns = _time.time_ns()
            except Exception:
                self.timestamp_ns = 1

    class GeniePacket(ctypes.Structure):
        _pack_ = 1
        _fields_ = [
            ("eth", EthernetHeader),
            ("ip", IPv4Header),
            ("udp", UDPHeader),
            ("app", GeniePacketHeader),
        ]
        # No __init__ override; size is sum of sub-headers (14 + 20 + 8 + 64)

    # Legacy dataclasses and helpers for tests
    try:
        from dataclasses import dataclass, field  # type: ignore
        import time as _time
        from typing import List, Optional, Dict

        @dataclass
        class NetworkNode:
            node_id: str
            ip_address: str
            mac_address: str
            port: int = DEFAULT_UDP_PORT
            gpu_count: int = 1
            capabilities: List[str] = field(default_factory=list)

        @dataclass
        class TensorFragment:
            tensor_id: str
            fragment_id: int
            total_fragments: int
            offset: int
            length: int
            data: Optional[bytes] = None
            timestamp: float = field(default_factory=_time.time)

        class AddressResolver:
            def __init__(self):
                self.nodes: Dict[str, NetworkNode] = {}
                self.ip_to_node: Dict[str, str] = {}
                self.mac_cache: Dict[str, str] = {}
            def add_node(self, node: NetworkNode) -> None:
                self.nodes[node.node_id] = node
                self.ip_to_node[node.ip_address] = node.node_id
                self.mac_cache[node.ip_address] = node.mac_address
            def get_node(self, node_id: str):
                return self.nodes.get(node_id)
            def get_node_by_ip(self, ip: str):
                node_id = self.ip_to_node.get(ip)
                return self.nodes.get(node_id) if node_id else None
            def resolve_mac(self, ip: str):
                return self.mac_cache.get(ip)

        _global_resolver: Optional[AddressResolver] = None
        def get_address_resolver() -> AddressResolver:
            global _global_resolver
            if _global_resolver is None:
                _global_resolver = AddressResolver()
            return _global_resolver

        class FragmentManager:
            def __init__(self, mtu: int = DEFAULT_MTU):
                self.mtu = mtu
                self.max_payload = mtu - 106
                self.reassembly_buffers: Dict[str, Dict[int, TensorFragment]] = {}
                self.fragment_timeouts: Dict[str, float] = {}
            def fragment_tensor(self, tensor_id: str, tensor_data: bytes):
                frags = []
                total_size = len(tensor_data)
                if total_size <= self.max_payload:
                    frags.append(TensorFragment(tensor_id, 0, 1, 0, total_size, tensor_data))
                else:
                    count = (total_size + self.max_payload - 1) // self.max_payload
                    for i in range(count):
                        off = i * self.max_payload
                        ln = min(self.max_payload, total_size - off)
                        frags.append(TensorFragment(tensor_id, i, count, off, ln, tensor_data[off:off+ln]))
                return frags
            def add_fragment(self, fragment: TensorFragment):
                tid = fragment.tensor_id
                buf = self.reassembly_buffers.setdefault(tid, {})
                buf[fragment.fragment_id] = fragment
                if len(buf) == fragment.total_fragments:
                    data = b''.join(buf[i].data for i in range(fragment.total_fragments))
                    del self.reassembly_buffers[tid]
                    self.fragment_timeouts.pop(tid, None)
                    return data
                return None

        def get_fragment_manager() -> FragmentManager:
            return FragmentManager()
    except Exception:
        pass

    # Prefer re-exporting the full legacy implementations if available
    try:
        import importlib.util, os as _os
        _legacy_path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "legacy", "protocol_python_reference.py"))
        _spec = importlib.util.spec_from_file_location("_genie_legacy_protocol", _legacy_path)
        if _spec and _spec.loader:
            _legacy = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_legacy)  # type: ignore[attr-defined]
            # Re-export richer legacy classes for test compatibility
            PacketBuilder = _legacy.PacketBuilder  # type: ignore
            PacketParser = _legacy.PacketParser    # type: ignore
            FragmentManager = _legacy.FragmentManager  # type: ignore
            AddressResolver = _legacy.AddressResolver  # type: ignore
            NetworkNode = _legacy.NetworkNode  # type: ignore
            get_address_resolver = _legacy.get_address_resolver  # type: ignore
            get_fragment_manager = _legacy.get_fragment_manager  # type: ignore
    except Exception:
        # Fall back to minimal shims above
        pass

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
    """Network node information (modern API)."""
    id: str
    ip: str
    mac: str
    port: int = DEFAULT_UDP_PORT

class LegacyNetworkNode(NamedTuple):
    """Legacy network node compatible with legacy tests."""
    node_id: str
    ip_address: str
    mac_address: str
    port: int = DEFAULT_UDP_PORT
    gpu_count: int = 1
    capabilities: tuple = ()

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

# Prefer re-exporting legacy protocol reference for test compatibility
try:
    from legacy.protocol_python_reference import (
        PacketBuilder as _LegacyPacketBuilder,
        PacketParser as _LegacyPacketParser,
        FragmentManager as _LegacyFragmentManager,
        AddressResolver as _LegacyAddressResolver,
        NetworkNode as _LegacyNetworkNode,
        TensorFragment as _LegacyTensorFragment,
        get_address_resolver as _legacy_get_address_resolver,
        get_fragment_manager as _legacy_get_fragment_manager,
    )
    PacketBuilder = _LegacyPacketBuilder  # type: ignore
    PacketParser = _LegacyPacketParser    # type: ignore
    FragmentManager = _LegacyFragmentManager  # type: ignore
    AddressResolver = _LegacyAddressResolver  # type: ignore
    TensorFragment = _LegacyTensorFragment  # type: ignore
    get_address_resolver = _legacy_get_address_resolver  # type: ignore
    get_fragment_manager = _legacy_get_fragment_manager  # type: ignore
    # Re-export legacy NetworkNode under a different name; tests import both modern and legacy suites
    LegacyNetworkNode = _LegacyNetworkNode  # type: ignore
except Exception:
    # If legacy module isn't importable, the minimal shims above will remain
    pass
