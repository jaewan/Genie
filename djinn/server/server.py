"""
Djinn Server - Main server component for disaggregated GPU cluster.

Manages:
- GPU resource discovery and allocation
- Network transport initialization
- Control plane for coordination
- Remote computation execution
"""

import asyncio
import logging
import os
import struct
import json
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch

from ..core.coordinator import DjinnCoordinator, CoordinatorConfig
from .optimizations.optimization_executor import OptimizationExecutor
from .capability_provider import CapabilityProvider
from ..core.metadata_types import ResultMetadata, ErrorMetadata, create_result_metadata, create_error_metadata

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for Djinn server."""
    node_id: Optional[str] = None  # Use centralized config if None
    control_port: Optional[int] = None  # Use centralized config if None
    data_port: Optional[int] = None     # Use centralized config if None
    gpu_indices: Optional[List[int]] = None  # Which GPUs to use (None = all available)
    prefer_dpdk: Optional[bool] = None  # Use centralized config if None
    tcp_fallback: Optional[bool] = None  # Use centralized config if None
    max_concurrent_transfers: Optional[int] = None  # Use centralized config if None

    def get_config(self):
        """Get configuration using centralized config as fallback."""
        from ..config import get_config
        config = get_config()

        return {
            'node_id': self.node_id or config.server.node_id,
            'control_port': self.control_port or config.network.control_port,
            'data_port': self.data_port or config.network.data_port,
            'gpu_indices': self.gpu_indices or [],
            'prefer_dpdk': self.prefer_dpdk if self.prefer_dpdk is not None else config.network.prefer_dpdk,
            'tcp_fallback': self.tcp_fallback if self.tcp_fallback is not None else config.network.tcp_fallback,
            'max_concurrent_transfers': self.max_concurrent_transfers or config.server.max_concurrent_transfers,
        }


class DjinnServer:
    """
    Main server for Djinn disaggregated GPU cluster.

    Handles remote tensor execution and data transfers.
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        # Get centralized configuration
        from ..config import get_config
        self._central_config = get_config()

        # Resolve configuration values
        resolved_config = config.get_config()
        self.node_id = resolved_config['node_id']
        self.control_port = resolved_config['control_port']
        self.data_port = resolved_config['data_port']
        self.max_concurrent_transfers = resolved_config['max_concurrent_transfers']

        self.capabilities = None
        self.coordinator = None
        self.control_plane = None
        self.executor = None  # Remote executor for operations
        self.is_running = False
        self.tcp_server = None  # Server's own TCP server
        self.tcp_transport = None  # TCP transport for sending results
        self._model_handler = None  # Shared model handler instance
        
        from .flow_control import get_flow_controller
        self.flow_controller = get_flow_controller(
            max_credits=100,  # Max 100 concurrent requests/data transfers
            credit_recovery_rate=1.0  # 1 credit per second recovery
        )
        
        # Task registry for concurrent request handling
        self._pending_tasks: Dict[str, asyncio.Task] = {}  # task_id -> Task
        self._task_results: Dict[str, Any] = {}  # task_id -> result
        self._task_lock = asyncio.Lock()
        
        # : Server health reporter for fleet coordination
        self._health_reporter = None

    async def start(self) -> bool:
        """Start the Djinn server."""
        try:
            logger.info(f"Starting Djinn server: {self.node_id}")

            # 1. Discover capabilities
            logger.info("Discovering capabilities...")
            self.capabilities = CapabilityProvider.discover()
            logger.info(f"✓ Found {self.capabilities.gpu_count} GPUs")

            # 2. Start TCP server for listening to incoming operation requests
            logger.info("Starting TCP server for operation requests...")
            import asyncio
            # Configure socket options for high-performance network transfer
            async def optimize_connection(reader, writer):
                """Optimize incoming connection with Phase 3 TCP optimizations."""
                import socket
                sock = writer.get_extra_info('socket')
                if sock:
                    try:
                        # 1. Disable Nagle's algorithm (reduce latency)
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        
                        # 2. Increase send buffer to 16MB (better TCP window utilization)
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)
                        
                        # 3. Increase receive buffer to 16MB (better TCP window utilization)
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
                        
                        # 4. Enable TCP window scaling (Linux-specific)
                        try:
                            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_WINDOW_CLAMP, 64 * 1024 * 1024)  # 64MB window
                        except (AttributeError, OSError):
                            pass  # Not available on this system
                        
                        # Get actual buffer sizes (may be adjusted by OS)
                        actual_sndbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
                        actual_rcvbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                        logger.debug(
                            f"✅ Server TCP optimized: NODELAY=1, SNDBUF={actual_sndbuf/(1024*1024):.1f}MB, "
                            f"RCVBUF={actual_rcvbuf/(1024*1024):.1f}MB"
                        )
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to optimize server TCP socket: {e}")
                await self._handle_connection(reader, writer)
            
            # Try to start TCP server, with port fallback if needed
            ports_to_try = [self.data_port, 5557, 5558, 5559, 5560]
            self.tcp_server = None
            actual_port = None

            for port in ports_to_try:
                try:
                    self.tcp_server = await asyncio.start_server(
                        optimize_connection,
                        '0.0.0.0',
                        port
                    )
                    actual_port = port
                    break
                except OSError as e:
                    if e.errno == 98:  # Address already in use
                        logger.warning(f"Port {port} in use, trying next port...")
                        continue
                    else:
                        raise

            if self.tcp_server is None:
                raise RuntimeError(f"Could not bind to any port in {ports_to_try}")

            # Update data_port to the actual port we're using
            self.data_port = actual_port
            logger.info(f"✓ TCP server listening on port {self.data_port} (Phase 3: TCP_NODELAY, 16MB buffers)")

            # Set up transport for handling operation requests and sending results
            from .transport.tcp_transport import TCPTransport
            from ..core.coordinator import CoordinatorConfig

            # Create transport config for server operations
            server_config = CoordinatorConfig(
                node_id=f"{self.node_id}-server",
                control_port=self.control_port,
                data_port=self.data_port,
                prefer_dpdk=False,
                tcp_fallback=True,
                is_server=True
            )

            # Initialize transport that can handle both incoming requests and outgoing results
            self.transport = TCPTransport(server_config)
            await self.transport.initialize()

            # ✅ WIRE: Connect operation callback to transport
            self.transport._operation_callback = self._handle_operation_request

            # Also set up result transport for sending responses back to clients
            result_config = CoordinatorConfig(
                node_id=f"{self.node_id}-result-sender",
                control_port=self.control_port + self._central_config.network.result_port_offset,
                data_port=self.data_port + self._central_config.network.result_port_offset,
                prefer_dpdk=False,
                tcp_fallback=True,
                is_server=False
            )

            self.result_transport = TCPTransport(result_config)
            # Don't initialize as server - this is for sending results only
            logger.info("✓ Server transport with operation callback wired")
            logger.info("✓ Result transport configured")

            # Server doesn't need control plane for basic operation
            # Control plane is handled by the coordinator if needed

            # 4. Initialize optimization executor (with tensor registry and fusion compiler)
            logger.info("Initializing optimization executor...")
            self.executor = OptimizationExecutor(gpu_id=0)  # Use first GPU
            logger.info(f"✓ Optimization executor ready (GPU {self.executor.gpu_id}, Registry: enabled, Fusion: enabled)")

            self.is_running = True
            logger.info(f"\n🎉 Djinn server ready on {self.node_id}")
            logger.info(f"   Control plane: {self.control_port}")
            logger.info(f"   Data plane: {self.data_port}")
            logger.info(f"   GPUs: {len(self.capabilities.gpu_indices)}")
            logger.info(f"   Memory: {self.capabilities.total_memory_gb}GB")

            # Start background tasks
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._transfer_handler_loop())
            
            # ✅ Phase 3: Start health reporting to global coordinator
            await self._start_health_reporting()

            return True

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            await self.stop()
            return False

    async def stop(self):
        """Stop the Djinn server."""
        logger.info(f"Stopping Djinn server: {self.node_id}")

        self.is_running = False

        # Server doesn't use control plane for basic operation
        # if self.control_plane:
        #     await self.control_plane.stop()

        # Server doesn't use coordinator for basic operation
        # if self.coordinator:
        #     await self.coordinator.stop()

        if hasattr(self, 'tcp_server') and self.tcp_server:
            self.tcp_server.close()
            await self.tcp_server.wait_closed()

        if hasattr(self, 'transport') and self.transport:
            if hasattr(self.transport, 'stop'):
                await self.transport.stop()

        if hasattr(self, 'result_transport') and self.result_transport:
            if self.result_transport.server:
                self.result_transport.server.close()
                await self.result_transport.server.wait_closed()
        
        # ✅ Phase 3: Stop health reporting
        if self._health_reporter:
            await self._health_reporter.stop()

        logger.info("✓ Server stopped")

    async def _handle_connection(self, reader, writer):
        """Handle incoming TCP connections for operation requests."""
        addr = writer.get_extra_info('peername')
        logger.info(f"TCP CONNECTION: New connection from {addr}")

        try:
            # Import protocol constants early for validation
            from .transport.protocol import MessageType
            
            # Read the first byte (message type)
            first_byte = await reader.readexactly(1)
            msg_type_value = struct.unpack('B', first_byte)[0]
            
            # ✅ CRITICAL: Log first byte received for ALL connections
            logger.info(
                f"📥 SERVER: Received first byte from {addr}: "
                f"0x{msg_type_value:02x} ({msg_type_value})"
            )
            
            try:
                msg_type = MessageType(msg_type_value)
            except ValueError:
                logger.error(
                    f"🚨 PROTOCOL ERROR: Unrecognized message type 0x{msg_type_value:02x} "
                    f"from {addr}. Only message-type-based protocol (0x05-0x0B, 0xFF) is supported."
                )
                raise ValueError(
                    f"Invalid protocol: 0x{msg_type_value:02x} is not a supported message type."
                )

            logger.info(
                f"🔍 Detected message type: {msg_type.name} (0x{msg_type_value:02x}) from {addr}"
            )

            # Store msg_type for finally block to use appropriate delay
            self._last_msg_type = msg_type
            try:
                # Special handling for CACHE_QUERY (uses raw JSON protocol, not secure protocol)
                if msg_type == MessageType.CACHE_QUERY:
                    await self._handle_cache_query_raw(reader, writer, addr)
                else:
                    await self._handle_message_type_protocol(reader, writer, msg_type.value, addr)
            except Exception as proto_error:
                logger.error(f"Error in message type protocol handler: {proto_error}")
                import traceback
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                raise
            return

        except Exception as e:
            logger.error(f"Error handling connection from {addr}: {e}")
        finally:
            # ✅ IMPROVEMENT: Support keep-alive for message type protocol
            from .transport.protocol import MessageType
            
            # Check if this was a keep-alive request (for chunked transfers or frequent requests)
            should_keep_alive = False
            
            # Determine if we should keep connection alive
            if hasattr(self, '_last_msg_type'):
                msg_type = self._last_msg_type
                # Keep-alive for chunked protocol messages (client will reuse connection)
                # Also check if request had keep_alive flag
                if MessageType.is_chunked_protocol(msg_type):
                    should_keep_alive = True
                elif hasattr(self, '_last_request') and isinstance(self._last_request, dict):
                    should_keep_alive = self._last_request.get('_keep_alive', False)
            
            if should_keep_alive:
                # ✅ FIX: DISABLE KEEP-ALIVE - Always close connections to prevent buffer pollution
                # When server keeps connections alive, chunks arriving on different connections
                # can read leftover response data from previous messages, causing protocol errors!
                logger.debug(f"⚠️  Keep-alive DISABLED to prevent buffer pollution (was requested for {addr})")
                # Fall through to close connection
                should_keep_alive = False
            
            # Always close connection after response
            try:
                # Variable delay based on message type
                delay = 5.0 if (hasattr(self, '_last_msg_type') and 
                               self._last_msg_type == MessageType.REGISTER_MODEL_FINALIZE) else 0.1  # Reduced delay
                await asyncio.sleep(delay)  # Delay to allow client to read
                if not writer.is_closing():
                    writer.close()
                    await writer.wait_closed()
            except Exception:
                pass  # Ignore errors during cleanup
    
    async def _handle_message_type_protocol(self, reader, writer, msg_type, addr):
        """Handle new message type protocol (REGISTER_MODEL, EXECUTE_MODEL).
        
        Uses secure JSON + binary serialization (no pickle for security).
        """
        from djinn.core.secure_serializer import SecureSerializer
        
        try:
            logger.info(f"📥 Handling message type 0x{msg_type:02x} from {addr}")
            
            # Read length (8 bytes, big-endian)
            # Use timeout for large reads to prevent hanging
            import asyncio
            try:
                logger.info(f"Reading message length header (msg_type=0x{msg_type:02x}) from {addr}...")
                length_bytes = await asyncio.wait_for(
                    reader.readexactly(8),
                    timeout=10.0  # 10s timeout for length header
                )
                length = int.from_bytes(length_bytes, 'big')
                
                # ✅ BUG FIX: Log the raw bytes for debugging protocol mismatches
                logger.debug(f"Length header bytes: {length_bytes.hex()} = {length} bytes ({length / (1024*1024):.2f} MB)")
                
                # ✅ BUG FIX: Validate length header is not a suspicious round number BEFORE reading data
                # This catches protocol mismatches early (before wasting time reading wrong amount of data)
                SUSPICIOUS_SIZES = [
                    64 * 1024 * 1024,      # 64MB (2^26) - common in errors
                    128 * 1024 * 1024,     # 128MB (2^27)
                    256 * 1024 * 1024,     # 256MB (2^28)
                ]
                if length in SUSPICIOUS_SIZES:
                    # This is almost certainly a protocol mismatch - reject early
                    raise ValueError(
                        f"⚠️  PROTOCOL MISMATCH: Suspicious length header {length} bytes ({length / (1024*1024):.1f} MB) "
                        f"for message type 0x{msg_type:02x}. "
                        f"Length header bytes: {length_bytes.hex()}. "
                        f"This indicates leftover data from previous message or connection state corruption. "
                        f"Closing connection and requesting client to reconnect."
                    )
                
                # ✅ BUG FIX: Validate message length to catch protocol mismatches
                MAX_REASONABLE_MESSAGE_SIZE = 10 * 1024 * 1024 * 1024  # 10GB (safety limit)
                if length > MAX_REASONABLE_MESSAGE_SIZE:
                    raise ValueError(
                        f"⚠️  Suspicious message length: {length} bytes ({length / (1024*1024):.1f} MB). "
                        f"This is likely a protocol mismatch or leftover data from previous message. "
                        f"Expected reasonable size (< 10GB)."
                    )
                if length == 0:
                    raise ValueError("Invalid message length: 0 bytes (possible protocol mismatch)")
                
                # Import protocol constants for validation
                from .transport.protocol import MessageType
                
                # ✅ BUG FIX: Check for suspicious round numbers (like 64MB = 2^26)
                # These often indicate protocol mismatches or leftover data
                # Apply to ALL message types, not just registration
                SUSPICIOUS_SIZES = [
                    64 * 1024 * 1024,      # 64MB (2^26) - common in errors
                    128 * 1024 * 1024,     # 128MB (2^27)
                    256 * 1024 * 1024,     # 256MB (2^28)
                    512 * 1024 * 1024,     # 512MB (2^29)
                    1024 * 1024 * 1024,    # 1GB (2^30)
                ]
                if length in SUSPICIOUS_SIZES:
                    # ✅ BUG FIX: Raise error for suspicious round numbers - they indicate protocol mismatch
                    # This applies to ALL message types, not just registration
                    raise ValueError(
                        f"⚠️  Suspicious message length: {length} bytes ({length / (1024*1024):.1f} MB) "
                        f"is a round number (power of 2) for message type 0x{msg_type:02x}. "
                        f"This indicates protocol mismatch or leftover data from previous message. "
                        f"Expected actual message size, not a round number. Check connection state and protocol detection."
                    )
                
                # ✅ BUG FIX: For registration messages, check if length is reasonable
                # Small models should be < 100MB, large models might be up to 1GB
                # But 64MB exactly is suspicious (round number)
                if msg_type == MessageType.REGISTER_MODEL and length > 50 * 1024 * 1024:  # > 50MB
                    logger.warning(
                        f"⚠️  Large registration message: {length} bytes ({length / (1024*1024):.1f} MB). "
                        f"If this is a small model, this might indicate protocol mismatch."
                    )
                
                logger.info(
                    f"📊 Message length: {length} bytes ({length / (1024*1024):.1f} MB, "
                    f"msg_type=0x{msg_type:02x}) from {addr}"
                )
                
                # For very large payloads, use chunked reading with timeout
                MAX_SINGLE_READ = 100 * 1024 * 1024  # 100MB
                if length > MAX_SINGLE_READ:
                    # Large payload - read in chunks with timeout
                    logger.info(
                        f"📦 Large payload detected ({length / (1024*1024):.1f} MB, msg_type=0x{msg_type:02x}), "
                        f"reading in chunks from {addr}..."
                    )
                    request_bytes = bytearray()
                    remaining = length
                    # Use larger chunks for faster reading (50MB instead of 10MB)
                    # This reduces the number of read operations for large payloads
                    chunk_size = 50 * 1024 * 1024  # 50MB chunks (increased from 10MB)
                    
                    chunk_num = 0
                    while remaining > 0:
                        read_size = min(chunk_size, remaining)
                        chunk_num += 1
                        logger.info(
                            f"📖 Reading chunk {chunk_num}: {read_size / (1024*1024):.1f} MB, "
                            f"{remaining / (1024*1024):.1f} MB remaining..."
                        )
                        try:
                            # Increase timeout for larger chunks (2s per MB, min 30s)
                            chunk_timeout = max(30.0, (read_size / (1024 * 1024)) * 2)
                            chunk = await asyncio.wait_for(
                                reader.readexactly(read_size),
                                timeout=chunk_timeout
                            )
                            request_bytes.extend(chunk)
                            remaining -= len(chunk)
                            logger.info(
                                f"✅ Read chunk {chunk_num}: {len(chunk) / (1024*1024):.1f} MB, "
                                f"{remaining / (1024*1024):.1f} MB remaining"
                            )
                        except asyncio.TimeoutError:
                            logger.error(
                                f"⏱️  Timeout reading chunk {chunk_num}: {read_size / (1024*1024):.1f} MB "
                                f"after 60s from {addr}"
                            )
                            raise
                    
                    request_bytes = bytes(request_bytes)
                    logger.info(f"✅ Finished reading large payload: {len(request_bytes)} bytes")
                else:
                    # Small payload - read all at once with timeout
                    logger.info(f"Reading {length} bytes (small payload)...")
                    timeout = max(30.0, length / (1024 * 1024) * 2)  # 2s per MB, min 30s
                    logger.debug(f"Using timeout: {timeout:.1f}s")
                    request_bytes = await asyncio.wait_for(
                        reader.readexactly(length),
                        timeout=timeout
                    )
                    logger.info(f"✅ Finished reading payload: {len(request_bytes)} bytes")
                
                logger.info(f"Deserializing request (msg_type=0x{msg_type:02x})...")
                try:
                    # ✅ OPTIMIZATION: Deserialize in thread pool for large payloads (non-blocking)
                    # For chunks, deserialization can be CPU-intensive (30-40MB of data)
                    # Import MessageType here to avoid scope issues
                    from .transport.protocol import MessageType
                    # Use pickle (FastSerializer needs more work for tensor handling)
                    import pickle
                    request = pickle.loads(request_bytes)
                    logger.info(f"✅ Request deserialized successfully (pickle format)")
                    # Store request for keep-alive decision
                    self._last_request = request
                except Exception as deserialize_error:
                    logger.error(f"Failed to deserialize request: {deserialize_error}")
                    # ✅ IMPROVEMENT: Check if pickle fallback is allowed
                    allow_pickle_fallback = getattr(self._central_config, 'security', None)
                    if allow_pickle_fallback and hasattr(allow_pickle_fallback, 'allow_pickle_fallback'):
                        allow_pickle = allow_pickle_fallback.allow_pickle_fallback
                    else:
                        # Default: disable pickle fallback for security
                        allow_pickle = False
                    
                    if allow_pickle:
                        # Try pickle for backward compatibility (legacy clients) - SECURITY RISK
                        try:
                            import pickle
                            logger.warning("⚠️  Falling back to pickle deserialization (legacy client) - SECURITY RISK")
                            request = pickle.loads(request_bytes)
                            logger.info(f"✅ Request deserialized successfully (pickle fallback)")
                            # Store request for keep-alive decision
                            self._last_request = request
                        except Exception as pickle_error:
                            logger.error(f"Pickle fallback also failed: {pickle_error}")
                            raise ValueError(f"Failed to deserialize request: {deserialize_error}")
                    else:
                        # Pickle fallback disabled - reject legacy clients
                        logger.error("Legacy protocol (pickle) not supported. Client must upgrade to secure protocol.")
                        raise ValueError(
                            f"Failed to deserialize request with secure protocol: {deserialize_error}. "
                            "Legacy pickle protocol is disabled for security. Please upgrade client."
                        )
                
            except asyncio.TimeoutError:
                logger.error(f"Timeout reading message from {addr} (type={msg_type:02x})")
                raise RuntimeError("Request timeout - payload too large or connection too slow")
            except asyncio.IncompleteReadError as e:
                # ✅ BUG FIX: Better error message for protocol mismatch detection
                partial_len = len(e.partial) if e.partial else 0
                expected_len = e.expected if e.expected else 0
                
                # Check if this is a suspicious round number (protocol mismatch indicator)
                SUSPICIOUS_SIZES = [
                    64 * 1024 * 1024,      # 64MB (2^26)
                    128 * 1024 * 1024,     # 128MB (2^27)
                    256 * 1024 * 1024,     # 256MB (2^28)
                ]
                
                if expected_len in SUSPICIOUS_SIZES and partial_len < 1000:
                    # This is almost certainly a protocol mismatch
                    logger.error(
                        f"⚠️  PROTOCOL MISMATCH detected from {addr}: "
                        f"Expected {expected_len} bytes ({expected_len / (1024*1024):.1f} MB, round number), "
                        f"but only got {partial_len} bytes. "
                        f"This indicates the client is using a different protocol or connection state is corrupted. "
                        f"Message type: 0x{msg_type:02x}. "
                        f"Closing connection and requesting client to reconnect."
                    )
                    # Close connection to force client to reconnect with clean state
                    try:
                        writer.close()
                        await writer.wait_closed()
                    except:
                        pass
                    raise ValueError(
                        f"Protocol mismatch: expected {expected_len} bytes (suspicious round number), "
                        f"got {partial_len} bytes. Connection closed. Please reconnect."
                    )
                elif expected_len > 100 * 1024 * 1024:  # > 100MB is suspicious
                    logger.error(
                        f"⚠️  Suspicious large expected size ({expected_len} bytes = {expected_len / (1024*1024):.1f} MB) "
                        f"suggests protocol mismatch. "
                        f"Got {partial_len} bytes. "
                        f"Check if binary protocol detection is working correctly or if there's leftover data."
                    )
                else:
                    logger.error(
                        f"Incomplete read from {addr}: got {partial_len} bytes, expected {expected_len} bytes "
                        f"({expected_len / (1024*1024):.1f} MB). "
                        f"Possible causes: protocol mismatch, connection closed, or leftover data."
                    )
                raise RuntimeError(
                    f"Incomplete read: connection closed during transfer "
                    f"(got {partial_len} bytes, expected {expected_len} bytes)"
                )
            
            logger.info(f"✅ Received message type 0x{msg_type:02x} from {addr}, length={length} bytes")
            
            # Import protocol constants
            from .transport.protocol import MessageType
            
            # Log finalization messages specifically
            if msg_type == MessageType.REGISTER_MODEL_FINALIZE:
                logger.info(f"🔍 FINALIZE message detected at TCP level: type=0x{msg_type:02x}, length={length} bytes from {addr}")
            
            # Extract request ID for correlation (if present)
            request_id = request.get('_request_id', 'unknown')
            if request_id != 'unknown':
                logger.debug(f"Request ID: {request_id}")
            
            # ✅ FIX: Initialize flow control variables
            # Check flow control AFTER deserialization so we can detect binary protocol
            client_id = f"{addr[0]}:{addr[1]}"
            credits_needed = max(1, length // (1024 * 1024))  # 1 credit per MB, min 1
            credits_acquired = False
            
            # ✅ Bypass flow control for chunk transfers and binary protocol (they're already rate-limited by client)
            # Chunk transfers are expected to be concurrent, and the client's semaphore already
            # limits concurrency. Server-side flow control would block legitimate parallel transfers.
            # Binary protocol (single large message) should also bypass flow control to avoid blocking.
            request_type = request.get('type', '')
            is_chunk_transfer = request_type in ['REGISTER_MODEL_CHUNK', 'REGISTER_MODEL_CHUNKED']
            is_binary_protocol = request.get('_binary_protocol', False) or 'weights_binary' in request
            
            if not is_chunk_transfer and not is_binary_protocol:
                # ✅ Acquire flow control credits before processing (skip for chunk transfers)
                try:
                    credits_acquired = await self.flow_controller.acquire(
                        size=credits_needed,
                        client_id=client_id,
                        timeout=5.0
                    )
                    
                    if not credits_acquired:
                        # Server overloaded - send error response
                        logger.warning(f"Server overloaded, rejecting request from {client_id}")
                        error_response = {
                            'status': 'error',
                            'message': 'Server overloaded, please retry',
                            '_request_id': request.get('_request_id')
                        }
                        response_bytes = SecureSerializer.serialize_response(error_response)
                        response_len = len(response_bytes)
                        writer.write(bytes([MessageType.ERROR]))
                        writer.write(response_len.to_bytes(8, 'big'))
                        writer.write(response_bytes)
                        await writer.drain()
                        return
                except Exception as flow_error:
                    logger.error(f"Flow control error: {flow_error}")
                    # Continue without flow control (graceful degradation)
                    credits_acquired = False
            else:
                # Chunk transfers bypass flow control
                logger.debug(f"Bypassing flow control for chunk transfer from {client_id}")
                credits_acquired = True  # Mark as acquired to skip release later
            
            # Handle based on message type
            logger.info(f"Processing message type 0x{msg_type:02x}...")
            try:
                if msg_type == MessageType.REGISTER_MODEL:
                    # ✅ OPTIMIZATION: Check if this is direct binary protocol
                    if 'weights_binary' in request:
                        response = await self._handle_register_model_binary(request)
                    else:
                        response = await self._handle_register_model(request)
                elif msg_type == MessageType.EXECUTE_MODEL:
                    response = await self._handle_execute_model(request)
                elif msg_type == MessageType.INIT_MODEL:
                    response = await self._handle_init_model(request)
                elif msg_type == MessageType.WARMUP_GPU:
                    response = await self._handle_warmup_gpu(request)
                elif msg_type == MessageType.CACHE_QUERY:
                    response = await self._handle_cache_query(request)
                elif msg_type == MessageType.QUERY_RESULT:
                    response = await self._handle_query_result(request)
                elif msg_type == MessageType.EXECUTE_BATCH:
                    response = await self._handle_execute_batch(request)
                elif msg_type == MessageType.REGISTER_MODEL_CHUNKED:
                    logger.info(f"Handling REGISTER_MODEL_CHUNKED message...")
                    response = await self._handle_register_model_chunked(request, reader, writer, addr)
                elif msg_type == MessageType.REGISTER_MODEL_CHUNK:
                    logger.info(f"📥 Handling REGISTER_MODEL_CHUNK message from {addr}...")
                    # ✅ OPTIMIZATION: Process chunk asynchronously (fire-and-forget)
                    # Don't wait for processing, send response immediately
                    # This allows client to send next chunk without waiting
                    response_task = asyncio.create_task(self._handle_register_model_chunk(request))
                    # Send immediate success response (chunk received, processing async)
                    response = {
                        'status': 'success',
                        'message': 'Chunk received (processing async)'
                    }
                    logger.info(f"✅ REGISTER_MODEL_CHUNK received (processing async)")
                elif msg_type == MessageType.REGISTER_MODEL_FINALIZE:
                    logger.info(f"Handling REGISTER_MODEL_FINALIZE message...")
                    response = await self._handle_register_model_finalize(request)
                else:
                    logger.warning(f"Unknown message type: 0x{msg_type:02x}")
                    response = {
                        'status': 'error',
                        'message': f'Unknown message type: {msg_type:02x}'
                    }
            finally:
                # ✅ Always release flow control credits
                if credits_acquired:
                    try:
                        await self.flow_controller.release(credits_needed, client_id=client_id)
                    except Exception as release_error:
                        logger.error(f"Error releasing flow control credits: {release_error}")
            
            # ✅ OPTIMIZATION: Skip response for fire-and-forget requests
            if request.get('_fire_and_forget', False):
                logger.debug(f"Skipping response for fire-and-forget request (chunk_id={request.get('chunk_id', 'unknown')})")
                return  # Client closed connection, no response needed
            
            logger.info(f"Sending response (status={response.get('status', 'unknown')})...")
            # Add request ID to response for correlation
            if '_request_id' in request:
                response['_request_id'] = request['_request_id']
            
            # Send response (message type + length + serialized data)
            try:
                # Use pickle (FastSerializer needs more work)
                import pickle
                response_bytes = pickle.dumps(response)
                response_len = len(response_bytes)
                
                logger.info(f"📤 Writing response: type=0x{msg_type:02x}, length={response_len} bytes ({response_len / (1024*1024):.2f} MB)")
                
                # ✅ DIAGNOSTIC: Log response size breakdown
                if msg_type == MessageType.EXECUTE_MODEL and 'result' in response:
                    result = response.get('result', {})
                    if isinstance(result, dict) and 'data' in result:
                        result_size = len(result.get('data', b''))
                        logger.info(f"🔍 [DIAGNOSTIC] Result tensor size: {result_size} bytes ({result_size / (1024*1024):.2f} MB)")
                
                # ✅ PHASE 3: Optimize write calls - combine writes for large responses
                # For large responses (> 1MB), use single write() to reduce syscalls
                if response_len > 1024 * 1024:  # > 1MB: combine writes
                    logger.debug(f"Writing large response ({response_len / (1024*1024):.1f} MB) - combining writes...")
                    combined = bytearray()
                    combined.extend(bytes([msg_type]))
                    combined.extend(response_len.to_bytes(8, 'big'))
                    combined.extend(response_bytes)
                    writer.write(bytes(combined))
                else:
                    # Small response: separate writes are fine
                    writer.write(bytes([msg_type]))  # Echo message type
                    writer.write(response_len.to_bytes(8, 'big'))
                    writer.write(response_bytes)
                logger.debug("Flushing response...")
                await writer.drain()
                logger.info(f"✅ Response flushed successfully ({response_len} bytes)")
                
                # For finalize, give extra time for client to read
                if msg_type == MessageType.REGISTER_MODEL_FINALIZE:
                    await asyncio.sleep(0.1)  # Small delay to ensure response is fully sent
                logger.info(f"✅ Response sent successfully")
            except (ConnectionResetError, BrokenPipeError, OSError) as conn_error:
                # ✅ Handle connection loss gracefully (expected for fire-and-forget)
                # Only log as debug, not error, since this is expected behavior
                logger.debug(f"Connection closed by client (expected for fire-and-forget): {conn_error}")
            except Exception as send_error:
                logger.error(f"Failed to send response: {send_error}")
                # Don't re-raise - connection might be closed, but we tried
                raise
            
        except Exception as e:
            logger.error(f"Error handling message type {msg_type:02x} from {addr}: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            # Send error response
            try:
                error_response = {
                    'status': 'error',
                    'message': str(e)
                }
                try:
                    error_bytes = SecureSerializer.serialize_response(error_response)
                except Exception as serialize_error:
                    # Should never happen, but log and use minimal error response
                    logger.error(f"Failed to serialize error response: {serialize_error}")
                    # Create minimal error response
                    minimal_error = {
                        'status': 'error',
                        'message': str(e)
                    }
                    error_bytes = SecureSerializer.serialize_response(minimal_error)
                error_len = len(error_bytes)
                writer.write(bytes([MessageType.ERROR]))  # ERROR message type
                writer.write(error_len.to_bytes(8, 'big'))
                writer.write(error_bytes)
                await writer.drain()
                logger.info("✅ Error response sent")
            except (ConnectionResetError, BrokenPipeError, OSError) as conn_error:
                # ✅ Handle connection loss gracefully (expected for fire-and-forget)
                logger.debug(f"Connection closed by client (expected for fire-and-forget): {conn_error}")
            except Exception as send_error:
                logger.error(f"Failed to send error response: {send_error}")
                # Connection might be closed, but we tried
    
    async def _handle_register_model_binary(self, request: Dict) -> Dict:
        """
        Handle model registration with direct binary protocol (optimized path).
        
        This is the fast path that avoids JSON overhead and intermediate dict structures.
        """
        try:
            fingerprint = request['fingerprint']
            descriptor = request['descriptor']
            weight_ids = request['weight_ids']
            weights_binary = request['weights_binary']
            architecture_data = request.get('architecture_data')
            
            logger.info(f"✅ PHASE 2: Received direct binary protocol registration ({len(weights_binary) / (1024*1024):.1f} MB)")
            
            # Deserialize binary weights
            from djinn.core.weight_deserializer import deserialize_weights_binary
            deserialize_start = time.time()
            uncached_weights = deserialize_weights_binary(weights_binary)
            deserialize_time = (time.time() - deserialize_start) * 1000
            logger.info(f"✅ PHASE 2: Binary deserialization complete: {deserialize_time:.1f}ms for {len(uncached_weights)} weights")
            
            # Register using existing handler
            from .resilient_model_handler import ResilientModelHandler
            if self._model_handler is None:
                self._model_handler = ResilientModelHandler(gpu_id=0)
                if hasattr(self.executor, 'model_cache') and self.executor.model_cache:
                    self._model_handler.model_cache = self.executor.model_cache
            
            registration_request = {
                'fingerprint': fingerprint,
                'descriptor': descriptor,
                'weight_ids': weight_ids,
                'uncached_weights': uncached_weights,
                'architecture_data': architecture_data
            }
            
            registration_response = await self._model_handler._register_with_recovery(registration_request)
            
            if registration_response.get('status') == 'success':
                logger.info(f"✅ Model {fingerprint} registered successfully (binary protocol)")
                return registration_response
            else:
                error_msg = registration_response.get('message', 'Unknown error')
                logger.error(f"❌ Model registration failed: {error_msg}")
                return registration_response
                
        except Exception as e:
            logger.error(f"❌ Binary model registration failed: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def _handle_register_model(self, request: Dict) -> Dict:
        """Handle model registration request - v2.3 with ModelCacheV23."""
        try:
            fingerprint = request.get('fingerprint', '')
            model_id = request.get('model_id', '')
            model_class = request.get('model_class', '')
            config_dict = request.get('config')
            
            logger.info(f"📝 Registering model {fingerprint[:8]} (v2.3)...")
            logger.info(f"   Model class: {model_class}")
            logger.info(f"   Model ID: {model_id}")
            
            # Reconstruct model from config
            # For HuggingFace models, we can use from_pretrained
            if model_id and 'transformers' in model_class:
                logger.info(f"   Loading from HuggingFace: {model_id}")
                from transformers import AutoModel, AutoModelForCausalLM
                try:
                    # Try AutoModelForCausalLM first (for GPT2, etc.)
                    model = AutoModelForCausalLM.from_pretrained(model_id)
                except:
                    # Fallback to AutoModel
                    model = AutoModel.from_pretrained(model_id)
                
                model.eval()
            else:
                return {
                    'status': 'error',
                    'message': f'Unsupported model class: {model_class}. Only HuggingFace models supported currently.'
                }
            
            # Register with v2.3 model cache
            from .model_cache_v23 import get_model_cache
            model_cache = get_model_cache()
            model_cache.register_model(fingerprint, model, model_id)
            
            logger.info(f"✅ Model {fingerprint[:8]} registered in v2.3 cache")
            
            return {
                'status': 'success',
                'fingerprint': fingerprint
            }
            
        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def _handle_init_model(self, request: Dict) -> Dict:
        """Handle model initialization (warmup) request."""
        try:
            from .resilient_model_handler import ResilientModelHandler
            # Use shared handler instance to ensure model cache is shared
            if self._model_handler is None:
                self._model_handler = ResilientModelHandler(gpu_id=0)
                # Use executor's model_cache if available
                if hasattr(self.executor, 'model_cache') and self.executor.model_cache:
                    self._model_handler.model_cache = self.executor.model_cache
            return await self._model_handler._init_model(request)
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def _cleanup_task(self, task_id: str) -> None:
        """Cleanup task after timeout."""
        await asyncio.sleep(60)
        async with self._task_lock:
            self._pending_tasks.pop(task_id, None)
            self._task_results.pop(task_id, None)
            logger.debug(f"Cleaned up task {task_id}")
    
    async def _handle_query_result(self, request: Dict) -> Dict:
        """Query result of async execution task."""
        task_id = request.get('task_id', '')
        
        logger.debug(f"Querying result for task {task_id}")
        
        async with self._task_lock:
            # Check if result is ready
            if task_id in self._task_results:
                result = self._task_results[task_id]
                logger.info(f"✅ Result ready for task {task_id}")
                return result
            
            # Check if task is still pending
            if task_id in self._pending_tasks:
                task = self._pending_tasks[task_id]
                if not task.done():
                    return {
                        'status': 'pending',
                        'task_id': task_id,
                        'message': 'Still executing, try again later'
                    }
                
                # Task finished but result not stored yet
                try:
                    result = task.result()
                    self._task_results[task_id] = result
                    return result
                except Exception as e:
                    return {
                        'status': 'error',
                        'task_id': task_id,
                        'message': str(e)
                    }
            
            # Task not found
            return {
                'status': 'error',
                'task_id': task_id,
                'message': f'Task {task_id} not found or expired'
            }
    
    async def _handle_execute_batch(self, request: Dict) -> Dict:
        """Handle batch execution request - execute multiple models/inputs in one pass."""
        try:
            # Extract batch requests
            batch_requests = request.get('batch', [])
            
            if not batch_requests:
                return {'status': 'error', 'message': 'Empty batch'}
            
            logger.info(f"📦 Executing batch of {len(batch_requests)} requests")
            
            # Group by model fingerprint for efficient execution
            by_model: Dict[str, List] = {}
            for i, req in enumerate(batch_requests):
                fp = req.get('fingerprint', '')
                if fp not in by_model:
                    by_model[fp] = []
                by_model[fp].append((i, req))
            
            # Execute each model's batch
            from .hybrid_executor import get_hybrid_executor
            from .session_manager import get_session_manager
            from .model_cache_v23 import get_model_cache
            
            results = {}
            session_mgr = get_session_manager()
            model_cache = get_model_cache()
            executor = get_hybrid_executor()
            
            for fingerprint, indexed_reqs in by_model.items():
                # Get model
                model = model_cache.get_model(fingerprint)
                if model is None:
                    for idx, req in indexed_reqs:
                        results[idx] = {
                            'status': 'error',
                            'message': f'Model {fingerprint} not found'
                        }
                    continue
                
                # Stack inputs for batching
                import torch
                batch_inputs = {}
                for key in indexed_reqs[0][1].get('inputs', {}).keys():
                    # Stack all values for this key
                    values = [req[1]['inputs'][key] for _, req in indexed_reqs]
                    if isinstance(values[0], torch.Tensor):
                        batch_inputs[key] = torch.cat(values, dim=0)
                    else:
                        batch_inputs[key] = values[0]  # Use first value for non-tensors
                
                # Execute batch
                session_id = session_mgr.create_session()
                batch_output, metrics = await executor.execute_with_lazy_outputs(
                    model=model,
                    inputs=batch_inputs,
                    session_id=session_id,
                    return_lazy=False
                )
                
                # Split batch output back to individual results
                batch_size = len(indexed_reqs)
                for i, (idx, req) in enumerate(indexed_reqs):
                    # Extract item i from batch output
                    if isinstance(batch_output, dict):
                        item_output = {k: v[i] if isinstance(v, torch.Tensor) else v 
                                      for k, v in batch_output.items()}
                    elif isinstance(batch_output, torch.Tensor):
                        item_output = batch_output[i]
                    else:
                        item_output = batch_output
                    
                    results[idx] = {
                        'status': 'success',
                        'result': item_output,
                        'metrics': {
                            'duration_ms': metrics.duration_ms / batch_size,  # Amortized
                            'batch_size': batch_size
                        }
                    }
                
                logger.info(f"✅ Batch execution complete: {len(indexed_reqs)} items, {metrics.duration_ms:.2f}ms total")
            
            return {
                'status': 'success',
                'batch_results': [results.get(i, {'status': 'error', 'message': 'Missing result'}) 
                                 for i in range(len(batch_requests))]
            }
        
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def _handle_warmup_gpu(self, request: Dict) -> Dict:
        """Handle GPU warmup request (one-time, server-wide)."""
        try:
            from .server_state import ServerState
            server_state = ServerState.get_instance()
            success = server_state.warmup_gpu()
            
            if success:
                return {
                    'status': 'success',
                    'message': 'GPU warmed up successfully',
                    'already_warmed': server_state.is_gpu_warmed()
                }
            else:
                return {
                    'status': 'error',
                    'message': 'GPU warmup failed'
                }
        except Exception as e:
            logger.error(f"GPU warmup failed: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return {
                'status': 'error',
                'message': str(e)
            }

    async def _handle_cache_query_raw(self, reader, writer, addr):
        """Handle CACHE_QUERY with raw JSON protocol (not secure protocol)."""
        try:
            logger.info(f"📥 Handling CACHE_QUERY (raw JSON protocol) from {addr}")

            # Read length (8 bytes, big-endian)
            length_bytes = await asyncio.wait_for(
                reader.readexactly(8),
                timeout=10.0
            )
            length = int.from_bytes(length_bytes, 'big')

            # Validate length
            if length > 1024 * 1024:  # 1MB limit for JSON
                raise ValueError(f"Cache query too large: {length} bytes")

            # Read JSON data
            json_bytes = await asyncio.wait_for(
                reader.readexactly(length),
                timeout=10.0
            )

            # Parse JSON
            request = json.loads(json_bytes.decode('utf-8'))

            # Handle the cache query
            response = await self._handle_cache_query(request)

            # Send response as JSON
            response_json = json.dumps(response).encode('utf-8')
            response_length = len(response_json).to_bytes(8, 'big')

            writer.write((0x04).to_bytes(1, 'big'))  # CACHE_QUERY response type
            writer.write(response_length)
            writer.write(response_json)
            await writer.drain()

            logger.info(f"✅ CACHE_QUERY response sent to {addr}")

        except Exception as e:
            logger.error(f"Cache query failed: {e}")
            error_response = {
                'status': 'error',
                'message': str(e)
            }
            try:
                response_json = json.dumps(error_response).encode('utf-8')
                response_length = len(response_json).to_bytes(8, 'big')

                writer.write((0x04).to_bytes(1, 'big'))  # CACHE_QUERY response type
                writer.write(response_length)
                writer.write(response_json)
                await writer.drain()
            except Exception as send_error:
                logger.error(f"Failed to send error response: {send_error}")

    async def _handle_cache_query(self, request: Dict) -> Dict:
        """Handle cache query request - check which tensor identifiers are cached."""
        try:
            identifiers = request.get('identifiers', [])
            # For now, return empty list (no cached tensors)
            # TODO: Implement actual cache checking logic
            cached_identifiers = []
            return {
                'status': 'success',
                'cached_identifiers': cached_identifiers
            }
        except Exception as e:
            logger.error(f"Cache query failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    async def _execute_model_impl(self, request: Dict) -> Dict:
        """Internal implementation of model execution (non-blocking)."""
        try:
            fingerprint = request.get('fingerprint', '')
            inputs = request.get('inputs', {})
            profile_id = request.get('profile_id')
            
            logger.info(f"🚀 Executing model {fingerprint[:8]} via HybridExecutor (v2.3)...")
            
            # Get v2.3 components
            from .hybrid_executor import get_hybrid_executor
            from .session_manager import get_session_manager
            from .model_cache_v23 import get_model_cache
            
            # Get or create session
            session_mgr = get_session_manager()
            session_id = session_mgr.create_session()
            
            # Get model from cache
            model_cache = get_model_cache()
            model = model_cache.get_model(fingerprint)
            
            if model is None:
                return {
                    'status': 'error',
                    'message': f'Model {fingerprint} not found in cache. Register it first.'
                }
            
            # Execute via HybridExecutor
            executor = get_hybrid_executor()
            result, metrics = await executor.execute_with_lazy_outputs(
                model=model,
                inputs=inputs,
                session_id=session_id,
                return_lazy=False  # Return concrete tensors over network
            )
            
            logger.info(f"✅ Execution complete: {metrics.duration_ms:.2f}ms")
            
            # Return result
            return {
                'status': 'success',
                'result': result,
                'metrics': {
                    'duration_ms': metrics.duration_ms,
                    'memory_peak_mb': metrics.memory_peak_mb
                }
            }
            
        except Exception as e:
            logger.error(f"Model execution failed: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def _handle_execute_model(self, request: Dict) -> Dict:
        """Handle model execution request - uses HybridExecutor (v2.3) with async execution."""
        import uuid
        
        # Check if client supports async mode (opt-in via request flag)
        async_mode = request.get('_async_mode', False)
        
        if not async_mode:
            # Blocking mode for backward compatibility (default)
            return await self._execute_model_impl(request)
        
        # Async mode: queue execution and return immediately
        task_id = str(uuid.uuid4())[:8]
        
        # Create async task for execution (don't await)
        task = asyncio.create_task(self._execute_model_impl(request))
        
        # Store task
        async with self._task_lock:
            self._pending_tasks[task_id] = task
        
        # Add callback to store result when done
        def store_result(fut):
            try:
                result = fut.result()
                # Schedule storing result
                asyncio.create_task(self._store_result_async(task_id, result))
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                asyncio.create_task(self._store_result_async(
                    task_id,
                    {'status': 'error', 'message': str(e)}
                ))
        
        task.add_done_callback(store_result)
        
        # Return immediately with task_id
        logger.info(f"📋 Task {task_id} queued for async execution (non-blocking)")
        return {
            'status': 'queued',
            'task_id': task_id,
            'message': 'Execution queued, poll with QUERY_RESULT for status'
        }
    
    async def _store_result_async(self, task_id: str, result: Any) -> None:
        """Store result asynchronously and schedule cleanup."""
        async with self._task_lock:
            self._task_results[task_id] = result
            logger.info(f"✅ Result stored for task {task_id}")
            # Schedule cleanup after 60s
            asyncio.create_task(self._cleanup_task(task_id))
    
    async def _handle_register_model_chunked(
        self, 
        request: Dict, 
        reader: Optional[asyncio.StreamReader] = None,
        writer: Optional[asyncio.StreamWriter] = None,
        addr: Optional[tuple] = None
    ) -> Dict:
        """Handle chunked model registration header."""
        try:
            fingerprint = request['fingerprint']
            total_chunks = request['total_chunks']
            architecture_data = request.get('architecture_data')
            
            # Initialize chunked registration state
            if not hasattr(self, '_chunked_registrations'):
                self._chunked_registrations = {}
            
            self._chunked_registrations[fingerprint] = {
                'fingerprint': fingerprint,
                'descriptor': request['descriptor'],
                'weight_ids': request['weight_ids'],
                'architecture_data': architecture_data,  # May be None if sent separately
                'total_chunks': total_chunks,
                'deserialized_chunks': {},  # ✅ Progressive deserialization storage (binary protocol only)
                'deserialization_tasks': {},  # ✅ Background deserialization tasks
                'deserialization_lock': asyncio.Lock(),  # ✅ Thread-safe access
                'start_time': time.time()  # time is imported at top of file
            }
            
            arch_size = len(architecture_data) if architecture_data else 0
            logger.info(
                f"Chunked registration started: {fingerprint}, "
                f"{total_chunks} chunks expected, "
                f"arch_data={arch_size / (1024*1024):.1f} MB"
            )
            
            return {
                'status': 'success',
                'message': f'Chunked registration initialized, expecting {total_chunks} chunks'
            }
        except Exception as e:
            logger.error(f"Chunked registration header failed: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def _handle_register_model_chunk(self, request: Dict) -> Dict:
        """Handle a single chunk of model registration.
        
        Optimized for speed: minimal processing, immediate response.
        Chunk data is already deserialized by SecureSerializer.
        """
        try:
            fingerprint = request['fingerprint']
            chunk_id = request['chunk_id']
            total_chunks = request['total_chunks']
            
            # Check if this is architecture_data chunk (chunk_id = -1)
            if chunk_id == -1:
                architecture_data = request.get('architecture_data')
                if not hasattr(self, '_chunked_registrations'):
                    return {
                        'status': 'error',
                        'message': 'No chunked registration in progress'
                    }
                if fingerprint not in self._chunked_registrations:
                    return {
                        'status': 'error',
                        'message': f'Chunked registration not initialized for {fingerprint}'
                    }
                reg_state = self._chunked_registrations[fingerprint]
                reg_state['architecture_data'] = architecture_data
                logger.info(f"Architecture data received for {fingerprint} ({len(architecture_data) / (1024*1024):.1f} MB)")
                return {
                    'status': 'success',
                    'chunk_id': chunk_id,
                    'message': 'Architecture data received'
                }
            
            # Regular weight chunk - optimized for speed
            # ✅ Binary protocol only (legacy dict-based removed)
            if 'chunk_data_binary' not in request:
                return {
                    'status': 'error',
                    'message': 'No chunk_data_binary provided (binary protocol required)'
                }
            
            chunk_data_binary = request['chunk_data_binary']
            logger.debug(f"✅ Received binary protocol chunk {chunk_id+1}/{total_chunks}")
            
            if not hasattr(self, '_chunked_registrations'):
                return {
                    'status': 'error',
                    'message': 'No chunked registration in progress'
                }
            
            if fingerprint not in self._chunked_registrations:
                return {
                    'status': 'error',
                    'message': f'Chunked registration not initialized for {fingerprint}'
                }
            
            # ✅ OPTIMIZATION: Store chunk and deserialize progressively (non-blocking)
            reg_state = self._chunked_registrations[fingerprint]
            
            # ✅ Binary protocol only - direct deserialization
            async def deserialize_chunk_background():
                """Deserialize chunk in background for progressive processing."""
                try:
                    # Binary protocol - direct deserialization
                    from djinn.core.weight_deserializer import deserialize_weights_binary
                    weights = await asyncio.to_thread(deserialize_weights_binary, chunk_data_binary)
                    
                    # Store deserialized weights (thread-safe access)
                    async with reg_state.get('deserialization_lock', asyncio.Lock()):
                        reg_state['deserialized_chunks'][chunk_id] = weights
                    
                    logger.debug(f"✅ Chunk {chunk_id+1}/{total_chunks} deserialized (background, binary protocol)")
                except Exception as e:
                    logger.error(f"❌ Failed to deserialize chunk {chunk_id}: {e}")
                    # Store error for later handling
                    async with reg_state.get('deserialization_lock', asyncio.Lock()):
                        reg_state['deserialized_chunks'][chunk_id] = None
            
            # Start background deserialization (fire-and-forget)
            if chunk_id not in reg_state.get('deserialization_tasks', {}):
                reg_state['deserialization_tasks'][chunk_id] = asyncio.create_task(
                    deserialize_chunk_background()
                )
            
            # ✅ OPTIMIZATION: Calculate size asynchronously (don't block response)
            received_count = len(reg_state.get('deserialized_chunks', {}))
            
            # Log every chunk for debugging (can be reduced to every 10th in production)
            if chunk_id % 10 == 0 or received_count == total_chunks or chunk_id < 5:
                # Calculate size only when logging (lazy evaluation)
                chunk_size_mb = len(chunk_data_binary) / (1024 * 1024)
                logger.info(
                    f"📥 Chunk {chunk_id+1}/{total_chunks} received for {fingerprint} "
                    f"({received_count}/{total_chunks} total, ~{chunk_size_mb:.1f} MB, binary protocol)"
                )
            
            return {
                'status': 'success',
                'chunk_id': chunk_id,
                'deserialized_chunks': received_count,
                'total_chunks': total_chunks
            }
        except Exception as e:
            logger.error(f"Chunk registration failed: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def _handle_register_model_finalize(self, request: Dict) -> Dict:
        """Finalize chunked model registration by reassembling and registering."""
        try:
            fingerprint = request['fingerprint']
            logger.info(f"🔚 FINALIZE called for {fingerprint}")
            logger.info(f"📥 FINALIZE request received: {request}")
            
            if not hasattr(self, '_chunked_registrations'):
                logger.error("No chunked registrations state found")
                return {
                    'status': 'error',
                    'message': 'No chunked registration in progress'
                }
            
            if fingerprint not in self._chunked_registrations:
                logger.error(f"Chunked registration not found for {fingerprint}")
                logger.error(f"Available registrations: {list(self._chunked_registrations.keys())}")
                return {
                    'status': 'error',
                    'message': f'Chunked registration not found for {fingerprint}'
                }
            
            reg_state = self._chunked_registrations[fingerprint]
            total_chunks = reg_state['total_chunks']
            deserialized_chunks = reg_state.get('deserialized_chunks', {})
            
            logger.info(
                f"📊 FINALIZE: {len(deserialized_chunks)}/{total_chunks} chunks deserialized "
                f"for {fingerprint}"
            )
            
            # Verify all chunks deserialized
            if len(deserialized_chunks) != total_chunks:
                missing = set(range(total_chunks)) - set(deserialized_chunks.keys())
                logger.error(
                    f"Missing chunks for {fingerprint}: {missing} "
                    f"({len(deserialized_chunks)}/{total_chunks} deserialized)"
                )
                return {
                    'status': 'error',
                    'message': f'Missing chunks: {missing} ({len(deserialized_chunks)}/{total_chunks} deserialized)'
                }
            
            # ✅ OPTIMIZATION: Progressive reassembly - use pre-deserialized chunks if available
            logger.info(f"🔄 Reassembling {total_chunks} chunks for {fingerprint}...")
            reassemble_start = time.time()
            
            # Check if chunks were deserialized progressively (background tasks)
            deserialized_chunks = reg_state.get('deserialized_chunks', {})
            deserialization_tasks = reg_state.get('deserialization_tasks', {})
            
            # Wait for any remaining background deserialization tasks
            if deserialization_tasks:
                logger.info(f"⏳ Waiting for {len(deserialization_tasks)} background deserialization tasks...")
                wait_start = time.time()
                try:
                    # Wait with timeout to prevent hanging indefinitely
                    await asyncio.wait_for(
                        asyncio.gather(*deserialization_tasks.values(), return_exceptions=True),
                        timeout=300.0  # 5 minute timeout
                    )
                    wait_time = time.time() - wait_start
                    logger.info(f"✅ Background deserialization tasks completed in {wait_time:.1f}s")
                except asyncio.TimeoutError:
                    logger.warning(f"⚠️ Timeout waiting for deserialization tasks (after {time.time() - wait_start:.1f}s)")
                    # Continue anyway - some chunks may be deserialized
            
            # Use pre-deserialized chunks if available, otherwise deserialize now
            if len(deserialized_chunks) == total_chunks and all(
                chunk_id in deserialized_chunks and deserialized_chunks[chunk_id] is not None 
                for chunk_id in range(total_chunks)
            ):
                # All chunks already deserialized progressively - just reassemble
                logger.info(f"✅ Using pre-deserialized chunks (progressive processing worked!)")
                uncached_weights = {}
                for chunk_id in sorted(deserialized_chunks.keys()):
                    if deserialized_chunks[chunk_id] is not None:
                        uncached_weights.update(deserialized_chunks[chunk_id])
                deserialize_time = 0.0  # Already done
            else:
                # Some chunks not deserialized yet - wait for background tasks to complete
                logger.info(f"🔄 Waiting for background deserialization to complete...")
                deserialize_start = time.time()
                
                # Wait for all background deserialization tasks to complete
                pending_tasks = [
                    task for chunk_id, task in reg_state.get('deserialization_tasks', {}).items()
                    if chunk_id not in deserialized_chunks or deserialized_chunks[chunk_id] is None
                ]
                
                if pending_tasks:
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                
                # Check if all chunks are now deserialized
                missing_chunks = [
                    chunk_id for chunk_id in range(total_chunks)
                    if chunk_id not in deserialized_chunks or deserialized_chunks[chunk_id] is None
                ]
                
                if missing_chunks:
                    logger.error(f"❌ Failed to deserialize chunks: {missing_chunks}")
                    raise RuntimeError(f"Failed to deserialize chunks: {missing_chunks}")
                
                deserialize_time = time.time() - deserialize_start
                logger.info(f"✅ All chunks deserialized (waited {deserialize_time:.1f}s for background tasks)")
                
                # Reassemble weights from all deserialized chunks
                uncached_weights = {}
                for chunk_id in sorted(deserialized_chunks.keys()):
                    if deserialized_chunks[chunk_id] is not None:
                        uncached_weights.update(deserialized_chunks[chunk_id])
            
            reassemble_time = time.time() - reassemble_start
            logger.info(
                f"✅ Reassembled {len(uncached_weights)} weights from {total_chunks} chunks "
                f"in {reassemble_time:.1f}s (deserialize: {deserialize_time:.1f}s)"
            )
            
            # Register model with reassembled weights
            logger.info(f"Registering model with {len(uncached_weights)} weights...")
            register_start = time.time()
            
            from .resilient_model_handler import ResilientModelHandler
            if self._model_handler is None:
                self._model_handler = ResilientModelHandler(gpu_id=0)
                if hasattr(self.executor, 'model_cache') and self.executor.model_cache:
                    self._model_handler.model_cache = self.executor.model_cache
            
            # Create registration request
            registration_request = {
                'fingerprint': fingerprint,
                'descriptor': reg_state['descriptor'],
                'weight_ids': reg_state['weight_ids'],
                'uncached_weights': uncached_weights,
                'architecture_data': reg_state.get('architecture_data')
            }
            
            # Register using existing handler
            try:
                registration_response = await self._model_handler._register_with_recovery(registration_request)
                register_time = time.time() - register_start
                logger.info(f"Model registration completed in {register_time:.1f}s")
            except Exception as e:
                logger.error(f"Model registration failed during finalize: {e}")
                logger.info(f"⚠️ Registration failed, but preserving state for {fingerprint} to allow retries")
                import traceback
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                # ✅ FIX: Don't raise - return error but preserve state for retry
                # Raising would cause outer handler to catch, but we want to preserve state
                return {
                    'status': 'error',
                    'message': str(e),
                    'retry_safe': True  # Signal to client that retry is safe
                }
            
            # ✅ Cleanup only on success
            if fingerprint in self._chunked_registrations:
                del self._chunked_registrations[fingerprint]
                logger.info(f"✅ Cleaned up chunked registration state for {fingerprint}")
            
            elapsed = time.time() - reg_state['start_time']
            logger.info(
                f"Chunked registration complete: {fingerprint} "
                f"({total_chunks} chunks, {elapsed:.1f}s)"
            )
            
            return registration_response
            
        except Exception as e:
            logger.error(f"Chunked registration finalization failed: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            # ✅ FIX: Preserve state on error to allow client retries
            # Only clean up on success (line 1069), not on error
            # This allows client to retry finalization if registration fails (e.g., CUDA OOM)
            logger.info(f"⚠️ Preserving chunked registration state for {fingerprint} to allow retries")
            return {
                'status': 'error',
                'message': str(e)
            }

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to connected clients."""
        while self.is_running:
            try:
                # Send heartbeat to all connected clients
                await asyncio.sleep(30)  # Every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def _start_health_reporting(self):
        """Start health reporting to global fleet coordinator (Phase 3)."""
        try:
            # Check if global coordinator is enabled
            fleet_config = self._central_config.fleet
            if not fleet_config.enable_global_coordinator:
                logger.debug("Global fleet coordinator disabled, skipping health reporting")
                return
            
            # Get global coordinator from coordinator if available
            global_coordinator = None
            if self.coordinator and hasattr(self.coordinator, 'global_coordinator'):
                global_coordinator = self.coordinator.global_coordinator
            
            if not global_coordinator:
                logger.debug("Global coordinator not available, skipping health reporting")
                return
            
            # Get server address (host:port format)
            import socket
            hostname = socket.gethostname()
            server_address = f"{hostname}:{self.data_port}"
            
            # Create and start health reporter
            from ..fleet.server_reporter import ServerHealthReporter
            self._health_reporter = ServerHealthReporter(
                server_address=server_address,
                global_coordinator=global_coordinator,
                report_interval=30.0  # Report every 30 seconds
            )
            
            await self._health_reporter.start()
            logger.info(f"✅ Started health reporting to global coordinator")
        except Exception as e:
            logger.warning(f"⚠️  Failed to start health reporting: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")

    async def _transfer_handler_loop(self):
        """Handle incoming transfer requests and execute operations."""
        while self.is_running:
            try:
                # Check for completed transfers from coordinator
                # In a real implementation, this would integrate with control_plane
                # to get notified of completed transfers

                # For now, we'll poll the coordinator for completed transfers
                # This is a simplified implementation - in practice you'd use callbacks
                if hasattr(self.coordinator, 'active_transfers'):
                    for transfer_id, transfer in list(self.coordinator.active_transfers.items()):
                        # Check if transfer is complete (simplified check)
                        # In practice, this would be handled by proper callbacks

                        # For demo purposes, we'll execute a simple operation on received tensors
                        try:
                            # Get received tensor (this would be implemented properly in coordinator)
                            # tensor = await self.coordinator.receive_tensor(transfer_id, transfer.metadata)

                            # Get operation from metadata
                            operation = transfer.metadata.get('operation', 'aten::relu')

                            # Execute operation (simplified - would need actual tensor)
                            print(f"Executing {operation} for transfer {transfer_id}")

                            # In a real implementation, you'd:
                            # 1. Receive the actual tensor
                            # 2. Execute the operation using the executor
                            # 3. Send the result back to the client

                        except Exception as e:
                            logger.error(f"Error executing operation for {transfer_id}: {e}")

                await asyncio.sleep(0.1)  # Small delay to avoid busy waiting
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Transfer handler error: {e}")

    async def _handle_operation_request(
        self,
        transfer_id: str,
        tensor_or_tensors,  # Can be single tensor or list of tensors
        metadata: Dict
    ):
        """
        Execute operation on received tensor(s) and send result back.

        This is called when a client sends a remote operation request.

        Args:
            transfer_id: Unique transfer identifier
            tensor_or_tensors: Single tensor or list of tensors from multi-tensor protocol
            metadata: Operation metadata
        """
        import traceback
        import numpy as np

        logger.info(f"OPERATION REQUEST: transfer_id={transfer_id}")
        logger.info(f"OPERATION REQUEST: metadata={metadata}")
        logger.info(f"OPERATION REQUEST: metadata keys={list(metadata.keys())}")

        operation = metadata.get('operation')
        result_id = metadata.get('result_id')
        source_node = metadata.get('source_node', 'unknown')

        # Use client_port if provided (more reliable than source_node)
        client_port = metadata.get('client_port')
        logger.debug(f"Received metadata keys: {list(metadata.keys())}")
        logger.debug(f"Original source_node: {source_node}")
        logger.debug(f"Client port from metadata: {client_port}")

        # Always try to use client_port if available, otherwise use hardcoded for testing
        client_port = metadata.get('client_port')
        logger.info(f"OPERATION REQUEST: transfer_id={transfer_id}")
        logger.info(f"OPERATION REQUEST: metadata={metadata}")
        logger.info(f"OPERATION REQUEST: metadata keys={list(metadata.keys())}")
        logger.info(f"OPERATION REQUEST: original source_node={source_node}")
        logger.info(f"OPERATION REQUEST: client_port={client_port}")

        # Always use the correct client port for localhost connections
        if ':' in source_node:
            client_ip = source_node.split(':')[0]
        else:
            client_ip = '127.0.0.1'  # localhost

        # Use client_port from metadata if available (preferred method)
        client_port = metadata.get('client_port')
        if client_port:
            source_node = f"{client_ip}:{client_port}"
            logger.info(f"OPERATION REQUEST: Using client_port {client_port} from metadata")
        else:
            # Fallback: use the same port as the server for localhost
            source_node = f"{client_ip}:{self.data_port}"
            logger.info(f"OPERATION REQUEST: No client_port in metadata, using server data port {self.data_port}")

        if not all([operation, result_id]):
            logger.warning(f"Missing critical metadata for {transfer_id}: op={operation}, result_id={result_id}")
            return

        try:
            # Handle both single and multi-tensor inputs
            if isinstance(tensor_or_tensors, list):
                tensors = tensor_or_tensors
                logger.info(f"🔧 Executing {operation} on {len(tensors)} tensors: {[t.shape for t in tensors]}")
            else:
                tensors = [tensor_or_tensors]
                logger.info(f"🔧 Executing {operation} on {tensor_or_tensors.shape}")

            # Create subgraph request for the executor
            subgraph_request = {
                'operations': [{
                    'op_id': 0,
                    'operation': operation,
                    'inputs': list(range(len(tensors))),  # Input indices
                    'kwargs': {}
                }],
                'output_id': 0,
                'semantic_metadata': {}  # Add basic semantic metadata
            }

            # Prepare input data dictionary
            input_data = {str(i): tensor for i, tensor in enumerate(tensors)}

            # Call executor with proper signature
            result, stats = await self.executor.execute(
                subgraph_request=subgraph_request,
                input_data=input_data,
                model_id=None,  # Single operation, no model context
                timeout=30.0
            )

            logger.info(f"📤 Sending result {result.shape} to {source_node}")

            # ✅ Send success result with result_id
            result_metadata = create_result_metadata(
                result_id=result_id,
                original_transfer=transfer_id,
                result=result
            ).to_dict()

            # ✅ FIX: Use client's listening port from metadata (not source_node)
            client_port = metadata.get('client_port')
            if client_port:
                # Get client IP from source_node or use localhost
                if ':' in source_node:
                    client_ip = source_node.split(':')[0]
                else:
                    client_ip = '127.0.0.1'  # localhost
                result_target = f"{client_ip}:{client_port}"
                logger.info(f"📤 Sending result {result.shape} to {result_target}")
            else:
                logger.error(f"Missing client_port in metadata for {transfer_id}")
                result_target = source_node
                logger.info(f"📤 Sending result {result.shape} to {result_target} (fallback)")

            # Send result back to client using the result transport
            success = await self.result_transport.send(
                result,
                target=result_target,  # ✅ Use correct target with client port
                transfer_id=result_id,
                metadata=result_metadata
            )

            if success:
                logger.info(f"✅ Result sent for {transfer_id}")
            else:
                logger.error(f"Failed to send result for {transfer_id}")

        except Exception as e:
            logger.error(f"❌ Operation execution failed: {e}")
            logger.error(traceback.format_exc())

            # ✅ Send error response to client
            error_metadata = create_error_metadata(
                result_id=result_id,
                original_transfer=transfer_id,
                error=e
            ).to_dict()

            # Send empty tensor with error metadata
            error_tensor = torch.zeros(0)

            try:
                # ✅ FIX: Use same client port routing for error responses
                client_port = metadata.get('client_port')
                if client_port:
                    if ':' in source_node:
                        client_ip = source_node.split(':')[0]
                    else:
                        client_ip = '127.0.0.1'
                    error_target = f"{client_ip}:{client_port}"
                else:
                    error_target = source_node

                await self.result_transport.send(
                    error_tensor,
                    target=error_target,  # ✅ Use correct target for errors too
                    transfer_id=result_id,
                    metadata=error_metadata
                )
                logger.info(f"📤 Error response sent for {transfer_id}")
            except Exception as send_error:
                logger.error(f"Failed to send error response: {send_error}")


def main():
    """CLI entry point for Djinn server."""
    import argparse

    parser = argparse.ArgumentParser(description="Djinn disaggregated GPU server")
    parser.add_argument("--node-id", required=True, help="Unique server identifier")
    parser.add_argument("--control-port", type=int, default=5555, help="Control plane port")
    parser.add_argument("--data-port", type=int, default=5556, help="Data plane port")
    parser.add_argument("--gpus", nargs="*", type=int, help="GPU indices to use (default: all)")
    parser.add_argument("--no-dpdk", action="store_true", help="Disable DPDK (use TCP only)")

    args = parser.parse_args()

    # Create config (can be None to use centralized defaults)
    config = ServerConfig(
        node_id=args.node_id if args.node_id else None,
        control_port=args.control_port if args.control_port != 5555 else None,
        data_port=args.data_port if args.data_port != 5556 else None,
        gpu_indices=args.gpus,
        prefer_dpdk=not args.no_dpdk if args.no_dpdk else None,
        tcp_fallback=True
    )

    # Start server
    async def run_server():
        server = DjinnServer(config)
        success = await server.start()
        if not success:
            exit(1)

        try:
            # Keep server running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await server.stop()

    # Run server
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
