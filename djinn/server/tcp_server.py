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
import pickle
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
OPTIMIZATION_EXECUTOR = None  # Changed from SUBGRAPH_EXECUTOR to OptimizationExecutor - disabled for network testing
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
        OPTIMIZATION_EXECUTOR = OptimizationExecutor(gpu_id=0)  # Enabled for network execution
        logger.info("✅ OptimizationExecutor initialized for network execution")
    except Exception as e:
        logger.error(f"❌ Failed to initialize optimization executor: {e}")
        raise

    STATS['start_time'] = datetime.now()


async def execute_subgraph_simple(subgraph: Dict[str, Any], input_data: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Simple subgraph execution fallback when optimization executor fails."""
    logger.warning("Using simple subgraph execution (no optimizations)")

    # Handle mock operations for testing
    operations = subgraph.get('operations', [])
    if operations:
        op = operations[0]  # Get first operation
        op_name = op.get('operation', '')

        if op_name == 'aten::linear':
            # Mock linear operation - just return a tensor with the expected shape
            expected_shape = op.get('shape', (1, 10, 50257))
            return torch.randn(*expected_shape)

    # Default fallback
    if input_data:
        # Return the first input tensor as a simple fallback
        first_tensor = next(iter(input_data.values()))
        return first_tensor + 1.0  # Dummy operation
    else:
        return torch.tensor([1.0])


async def handle_coordinator_subgraph(request_data: bytes, writer) -> None:
    """Handle subgraph execution request from coordinator (pickled format)."""
    global STATS

    try:
        STATS['subgraph_requests'] += 1
        print(f"🧪 SERVER: Received subgraph request, data length: {len(request_data)}")
        logger.info(f"🧪 SERVER: Received subgraph request, data length: {len(request_data)}")

        # Parse coordinator's pickled message
        try:
            message = pickle.loads(request_data)
            logger.info(f"🧪 SERVER: Coordinator message keys: {list(message.keys())}")
            logger.info(f"🧪 SERVER: Message subgraph_data keys: {list(message.get('subgraph_data', {}).keys()) if isinstance(message.get('subgraph_data'), dict) else 'N/A'}")
        except Exception as e:
            logger.error(f"Failed to unpickle coordinator message: {e}")
            error_response = f"Failed to parse message: {e}".encode()
            await _send_message(writer, 0x05, error_response)
            return

        # Extract data from coordinator message
        subgraph_data = message.get('subgraph_data', {})
        input_data_pickled = subgraph_data.get('input_data', {})
        timeout = message.get('timeout', 30)

        # Handle differential graph protocol
        if isinstance(subgraph_data, dict) and subgraph_data.get('type') == 'delta':
            # TODO: Handle differential updates
            logger.warning("Differential updates not yet implemented")
            subgraph = subgraph_data.get('subgraph', {})
        else:
            # Full subgraph
            subgraph = subgraph_data.get('subgraph', {}) if isinstance(subgraph_data, dict) else subgraph_data

        # Deserialize input tensors
        input_data = {}
        try:
            for key, pickled_tensor in input_data_pickled.items():
                if isinstance(pickled_tensor, bytes):
                    input_data[key] = pickle.loads(pickled_tensor)
                else:
                    input_data[key] = pickled_tensor
        except Exception as e:
            logger.error(f"Failed to deserialize input data: {e}")
            error_response = f"Failed to deserialize inputs: {e}".encode()
            await _send_message(writer, 0x05, error_response)
            return

        # Execute subgraph
        try:
            print(f"🔍 SERVER: Subgraph keys: {list(subgraph.keys()) if isinstance(subgraph, dict) else type(subgraph)}")
            print(f"🔍 SERVER: Subgraph operations: {len(subgraph.get('operations', [])) if isinstance(subgraph, dict) else 'N/A'}")
            logger.info(f"🔍 Subgraph keys: {list(subgraph.keys()) if isinstance(subgraph, dict) else type(subgraph)}")
            logger.info(f"🔍 Subgraph operations: {len(subgraph.get('operations', [])) if isinstance(subgraph, dict) else 'N/A'}")

            if OPTIMIZATION_EXECUTOR:
                result, stats = await OPTIMIZATION_EXECUTOR.execute(
                    subgraph_request=subgraph,
                    input_data=input_data,
                    timeout=timeout
                )
                logger.debug(f"Optimization stats: {stats}")

                # Ensure result is materialized
                if hasattr(result, 'materialize'):
                    print(f"🔍 SERVER: Result is LazyTensor, materializing...")
                    result = result.materialize()
                    print(f"🔍 SERVER: Materialized result shape: {result.shape}")
                else:
                    print(f"🔍 SERVER: Result is already concrete: {type(result)}")

                logger.info(f"✅ Subgraph execution completed: result shape {result.shape}")
            else:
                logger.error("❌ No optimization executor available - execution failed")
                error_response = "No optimization executor available".encode()
                await _send_message(writer, 0x05, error_response)
                return

        except Exception as e:
            logger.error(f"Subgraph execution failed: {e}")
            error_response = f"Execution failed: {e}".encode()
            await _send_message(writer, 0x05, error_response)
            return

        # Serialize and send result
        try:
            if USE_OPTIMIZED_SERIALIZATION and OPTIMIZED_SERIALIZATION_AVAILABLE:
                result_bytes = serialize_tensor(result, use_numpy=True)
            else:
                result_buffer = io.BytesIO()
                torch.save(result, result_buffer)
                result_bytes = result_buffer.getvalue()

            # Send result with proper message format (RESULT = 0x04)
            await _send_message(writer, 0x04, result_bytes)

            STATS['requests_success'] += 1
            logger.info(f"✅ Coordinator subgraph execution complete: {result.shape}")

        except Exception as e:
            logger.error(f"Failed to serialize result: {e}")
            error_response = f"Failed to serialize result: {e}".encode()
            await _send_message(writer, 0x05, error_response)

    except Exception as e:
        logger.error(f"Error in coordinator subgraph handler: {e}")
        STATS['requests_failed'] += 1
        try:
            error_response = f"Internal error: {e}".encode()
            await _send_message(writer, 0x05, error_response)
        except:
            pass  # Connection might be closed


async def handle_execute_subgraph(request_data: bytes, writer) -> None:
    """Handle subgraph execution request."""
    global STATS
    
    # ✅ PROFILING: Record request handling time
    from .profiling_context import record_phase
    with record_phase('request_handling'):
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
            
            # Deserialize input tensors (profiling happens inside deserialize_tensor)
            try:
                input_data = deserialize_tensor(tensors_data)
            except Exception as e:
                logger.error(f"Failed to deserialize tensors: {e}")
                error_response = str(e).encode()
                await _send_message(writer, 0x05, error_response)
                return

            logger.info(f"🎯 Executing subgraph: {len(subgraph_request['operations'])} operations")

            # Execute subgraph with optimizations (profiling happens inside executor)
            try:
                result = OPTIMIZATION_EXECUTOR.executor.execute(subgraph_request, input_data)
            except Exception as e:
                logger.error(f"❌ Subgraph execution failed: {e}")
                error_response = str(e).encode()
                await _send_message(writer, 0x05, error_response)
                return

            # Serialize result (profiling happens inside serialize_tensor)
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
    print(f"🔗 SERVER: New connection from {client_addr}")
    logger.info(f"✅ New connection from {client_addr}")

    try:
        while not reader.at_eof():
            # Receive message
            msg_type, data = await _recv_message(reader)

            STATS['requests_total'] += 1

            if msg_type == 0x01:  # EXECUTE_SUBGRAPH (legacy)
                logger.info("📊 Handling EXECUTE_SUBGRAPH (legacy)")
                await handle_execute_subgraph(data, writer)

            elif msg_type == 0x02:  # EXECUTE_OPERATION (legacy)
                logger.info("📊 Handling EXECUTE_OPERATION (legacy)")
                await handle_execute_operation(data, writer)

            elif msg_type == 0x03:  # EXECUTE_SUBGRAPH (coordinator)
                logger.info("📊 Handling EXECUTE_SUBGRAPH (coordinator)")
                await handle_coordinator_subgraph(data, writer)

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
