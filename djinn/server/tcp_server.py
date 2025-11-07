"""
TCP-based server for remote tensor execution.

Replaces FastAPI HTTP server with pure asyncio TCP server.
Uses length-prefixed message framing for binary safety.
Implements efficient subgraph execution.
"""

import asyncio
import struct
import torch
import json
import logging
import io
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
DEVICE = None
OPTIMIZATION_EXECUTOR = None  # Changed from SUBGRAPH_EXECUTOR to OptimizationExecutor
GPU_CACHE = None
GRAPH_CACHE = None

STATS = {
    'requests_total': 0,
    'requests_success': 0,
    'requests_failed': 0,
    'subgraph_requests': 0,
    'start_time': None,
    'gpu_cache_hits': 0,
    'gpu_cache_misses': 0,
    'graph_cache_hits': 0,
    'graph_cache_misses': 0,
}

# Import optimized serialization
try:
    from .serialization import serialize_tensor, deserialize_tensor
    OPTIMIZED_SERIALIZATION_AVAILABLE = True
    logger.info("✅ Optimized serialization module loaded")
except ImportError:
    logger.warning("⚠️  Optimized serialization not available")
    OPTIMIZED_SERIALIZATION_AVAILABLE = False

USE_OPTIMIZED_SERIALIZATION = os.getenv('GENIE_USE_OPTIMIZED_SERIALIZATION', 'true').lower() == 'true'


async def initialize_server():
    """Initialize server components."""
    global DEVICE, OPTIMIZATION_EXECUTOR, GPU_CACHE, GRAPH_CACHE, STATS

    # Set device
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
        logger.info(f"🚀 Server starting with GPU: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device("cpu")
        logger.warning("⚠️  No GPU available, using CPU")

    # Initialize GPU cache
    try:
        from .gpu_cache import get_global_cache
        GPU_CACHE = get_global_cache(max_models=5)
        logger.info("✅ GPU cache initialized (max_models=5)")
    except Exception as e:
        logger.warning(f"⚠️  Failed to initialize GPU cache: {e}")

    # Initialize graph cache
    try:
        from .graph_cache import get_global_graph_cache
        GRAPH_CACHE = get_global_graph_cache(max_graphs=100)
        logger.info("✅ Graph cache initialized (max_graphs=100)")
    except Exception as e:
        logger.warning(f"⚠️  Failed to initialize graph cache: {e}")

    # Initialize optimization executor (wraps SubgraphExecutor with optimizations)
    try:
        from .optimization_executor import OptimizationExecutor
        OPTIMIZATION_EXECUTOR = OptimizationExecutor(gpu_id=0)
        logger.info("✅ OptimizationExecutor initialized (Registry + Fusion enabled)")
    except Exception as e:
        logger.error(f"❌ Failed to initialize optimization executor: {e}")
        raise

    STATS['start_time'] = datetime.now()


async def handle_execute_subgraph(request_data: bytes, writer) -> None:
    """Handle subgraph execution request."""
    global STATS
    
    try:
        STATS['subgraph_requests'] += 1
        
        # Parse request
        request_str = request_data[:request_data.find(b'\0')].decode() if b'\0' in request_data else ""
        
        # Try to extract JSON from the message
        try:
            # Message format: JSON with request and tensors_size, followed by tensor data
            json_end = request_data.find(b'}') + 1
            if json_end > 0:
                metadata = json.loads(request_data[:json_end].decode())
                request_json = metadata['request']
                tensors_size = metadata['tensors_size']
                tensors_data = request_data[json_end:json_end + tensors_size]
            else:
                raise ValueError("Could not parse request metadata")
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse request: {e}")
            error_response = str(e).encode()
            await _send_message(writer, 0x05, error_response)  # ERROR
            return

        # Parse subgraph request
        subgraph_request = json.loads(request_json)
        
        # Deserialize input tensors
        try:
            input_data = deserialize_tensor(tensors_data)
        except Exception as e:
            logger.error(f"Failed to deserialize tensors: {e}")
            error_response = str(e).encode()
            await _send_message(writer, 0x05, error_response)
            return

        logger.info(f"🎯 Executing subgraph: {len(subgraph_request['operations'])} operations")

        # Execute subgraph with optimizations
        try:
            result = OPTIMIZATION_EXECUTOR.executor.execute(subgraph_request, input_data)
        except Exception as e:
            logger.error(f"❌ Subgraph execution failed: {e}")
            error_response = str(e).encode()
            await _send_message(writer, 0x05, error_response)
            return

        # Serialize result
        try:
            if USE_OPTIMIZED_SERIALIZATION and OPTIMIZED_SERIALIZATION_AVAILABLE:
                logger.info("✅ Using optimized numpy.save serialization")
                result_bytes = serialize_tensor(result, use_numpy=True)
            else:
                logger.warning("⚠️ Using fallback torch.save serialization")
                result_buffer = io.BytesIO()
                torch.save(result, result_buffer)
                result_bytes = result_buffer.getvalue()
        except Exception as e:
            logger.error(f"Failed to serialize result: {e}")
            error_response = str(e).encode()
            await _send_message(writer, 0x05, error_response)
            return

        # Send result (type=0x04: RESULT)
        await _send_message(writer, 0x04, result_bytes)

        STATS['requests_success'] += 1
        logger.info(f"✅ Subgraph execution complete: {result.shape}")

    except Exception as e:
        logger.error(f"❌ Error handling subgraph: {e}")
        STATS['requests_failed'] += 1
        error_response = str(e).encode()
        await _send_message(writer, 0x05, error_response)


async def handle_execute_operation(request_data: bytes, writer) -> None:
    """Handle single operation execution request."""
    global STATS
    
    try:
        # Parse request (find JSON part)
        json_end = request_data.find(b'\0')
        if json_end < 0:
            json_end = request_data.find(request_data[0:1])

        # Simple approach: first part is JSON
        try:
            request_json = json.loads(request_data.split(b'\x00')[0].decode())
            tensor_data = request_data[len(request_data.split(b'\x00')[0]) + 1:]
        except:
            # Fallback
            for i, b in enumerate(request_data):
                if b == 0:
                    request_json = json.loads(request_data[:i].decode())
                    tensor_data = request_data[i+1:]
                    break

        operation = request_json.get('operation')
        logger.info(f"🎯 Executing operation: {operation}")

        # Deserialize tensor
        tensor = torch.load(io.BytesIO(tensor_data))

        # Execute operation (placeholder - should use operation registry)
        # For now, just move to GPU and back
        result = tensor.to(DEVICE).cpu()

        # Serialize result
        result_buffer = io.BytesIO()
        torch.save(result, result_buffer)
        result_bytes = result_buffer.getvalue()

        # Send result
        await _send_message(writer, 0x04, result_bytes)

        STATS['requests_success'] += 1
        logger.info(f"✅ Operation execution complete")

    except Exception as e:
        logger.error(f"❌ Error handling operation: {e}")
        STATS['requests_failed'] += 1
        error_response = str(e).encode()
        await _send_message(writer, 0x05, error_response)


async def _send_message(writer, message_type: int, data: bytes) -> None:
    """Send length-prefixed message."""
    message = struct.pack('>BI', message_type, len(data)) + data
    writer.write(message)
    await writer.drain()


async def _recv_message(reader, timeout: float = 300.0) -> Tuple[int, bytes]:
    """Receive length-prefixed message."""
    try:
        header = await asyncio.wait_for(
            reader.readexactly(5),
            timeout=timeout
        )
        msg_type, length = struct.unpack('>BI', header)
        
        data = await asyncio.wait_for(
            reader.readexactly(length),
            timeout=timeout
        )
        return msg_type, data
    except asyncio.TimeoutError:
        raise RuntimeError("Message reception timeout")


async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Handle incoming TCP connection."""
    global STATS
    
    client_addr = writer.get_extra_info('peername')
    logger.info(f"✅ New connection from {client_addr}")

    try:
        while not reader.at_eof():
            # Receive message
            msg_type, data = await _recv_message(reader)

            STATS['requests_total'] += 1

            if msg_type == 0x01:  # EXECUTE_SUBGRAPH
                logger.info("📊 Handling EXECUTE_SUBGRAPH")
                await handle_execute_subgraph(data, writer)

            elif msg_type == 0x02:  # EXECUTE_OPERATION
                logger.info("📊 Handling EXECUTE_OPERATION")
                await handle_execute_operation(data, writer)

            else:
                logger.warning(f"Unknown message type: {msg_type}")
                await _send_message(writer, 0x05, b"Unknown message type")

    except asyncio.IncompleteReadError:
        logger.info(f"Client {client_addr} disconnected")
    except Exception as e:
        logger.error(f"Error handling connection: {e}")
    finally:
        writer.close()
        await writer.wait_closed()
        logger.info(f"Connection from {client_addr} closed")


async def start_server(host: str = "0.0.0.0", port: int = 5556) -> None:
    """Start TCP server."""
    logger.info(f"🚀 Starting TCP server on {host}:{port}")

    server = await asyncio.start_server(handle_connection, host, port)

    addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
    logger.info(f"✅ TCP Server listening on {addrs}")

    STATS['start_time'] = datetime.now()

    try:
        async with server:
            await server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")


async def main():
    """Initialize and start server."""
    await initialize_server()
    await start_server()


if __name__ == "__main__":
    asyncio.run(main())
