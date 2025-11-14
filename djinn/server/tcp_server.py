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
import time
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


async def handle_cache_query(request_data: bytes, writer) -> None:
    """
    Handle cache query request from client.
    
    Protocol:
    1. Receive: JSON with {'identifiers': [...]}
    2. Check cache: which identifiers are cached?
    3. Respond: JSON with {'cached_identifiers': [...]}
    
    This is Phase 1, Component 3 of the enhancement plan.
    """
    try:
        # Parse JSON request
        request = json.loads(request_data.decode('utf-8'))
        identifiers = request.get('identifiers', [])
        
        logger.debug(f"Cache query: checking {len(identifiers)} identifiers")
        
        # Check which identifiers are cached
        from .gpu_cache import get_global_cache
        cache = get_global_cache()
        
        # Ensure cache is migrated to identifier-based format
        if hasattr(cache, '_migrate_to_new_format'):
            cache._migrate_to_new_format()
        
        # Lookup in identifier-based cache
        cached_identifiers = []
        if hasattr(cache, 'cache_new'):
            # New identifier-based cache
            cached_identifiers = [
                ident for ident in identifiers
                if ident in cache.cache_new
            ]
        elif hasattr(cache, 'cache'):
            # Old model_id-based cache - check if we can look up by identifier
            # For now, return empty (cache not migrated yet)
            logger.debug("Cache not yet migrated to identifier-based format")
            cached_identifiers = []
        
        logger.debug(
            f"Cache query result: {len(cached_identifiers)}/{len(identifiers)} cached "
            f"({100*len(cached_identifiers)/len(identifiers):.1f}% hit rate)" if identifiers else "0%"
        )
        
        # Send response using standard message protocol
        response = {
            'cached_identifiers': cached_identifiers
        }
        response_json = json.dumps(response).encode('utf-8')
        
        # Use _send_message for consistent protocol (8-byte length)
        await _send_message(writer, 0x04, response_json)
        
    except Exception as e:
        logger.error(f"Cache query failed: {e}")
        import traceback
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        # Send empty response on error (safe fallback)
        response_json = json.dumps({'cached_identifiers': []}).encode('utf-8')
        await _send_message(writer, 0x04, response_json)


async def handle_register_model(request_data: bytes, writer) -> None:
    """
    Handle model registration request.
    
    Protocol:
    1. Receive: Pickled dict with registration data
    2. Validate security
    3. Register model in cache
    4. Respond: Success/error
    
    This is Week 2/3 integration.
    """
    global STATS
    
    try:
        logger.info(f"📝 Received model registration request ({len(request_data)} bytes)")
        
        # Deserialize request
        try:
            logger.debug("🔧 Attempting to deserialize request...")
            request = pickle.loads(request_data)
            logger.info(f"✅ Request deserialized successfully: {list(request.keys())}")
        except Exception as e:
            logger.error(f"❌ Failed to deserialize request: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            error_response = {
                'status': 'error',
                'message': f"Deserialization failed: {str(e)}"
            }
            try:
                response_bytes = pickle.dumps(error_response)
                await _send_message(writer, 0x05, response_bytes)
            except Exception as send_error:
                logger.error(f"❌ Failed to send error response: {send_error}")
            return
        
        fingerprint = request.get('fingerprint')
        descriptor = request.get('descriptor')
        weight_ids = request.get('weight_ids', {})
        uncached_weights = request.get('uncached_weights', {})
        architecture_data = request.get('architecture_data')
        
        if not fingerprint:
            raise ValueError("Missing fingerprint in request")
        
        logger.info(f"📝 Registering model {fingerprint} ({len(uncached_weights)} weights)")
        
        # Initialize resilient handler (singleton pattern)
        # Use shared handler instance for both registration and execution
        if not hasattr(handle_register_model, '_shared_handler'):
            from .resilient_model_handler import ResilientModelHandler
            handle_register_model._shared_handler = ResilientModelHandler(gpu_id=0)
            # Also set for execute handler
            handle_execute_model._shared_handler = handle_register_model._shared_handler
        
        handler = handle_register_model._shared_handler
        
        # Deserialize uncached weights (base64-encoded pickle)
        import base64
        
        logger.debug(f"🔧 Deserializing {len(uncached_weights)} weights...")
        deserialized_weights = {}
        for i, (name, weight_data) in enumerate(uncached_weights.items()):
            try:
                if isinstance(weight_data.get('data'), str):
                    # Base64-encoded pickle format
                    tensor_bytes = base64.b64decode(weight_data['data'].encode('ascii'))
                    tensor = pickle.loads(tensor_bytes)
                else:
                    # Legacy numpy array format (for backward compatibility)
                    import numpy as np
                    tensor_data = np.array(weight_data['data'], dtype=weight_data.get('dtype', 'float32'))
                    tensor = torch.from_numpy(tensor_data)
                    if list(tensor.shape) != weight_data['shape']:
                        tensor = tensor.reshape(weight_data['shape'])
                deserialized_weights[name] = tensor
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"  Deserialized {i + 1}/{len(uncached_weights)} weights")
            except Exception as e:
                logger.error(f"❌ Failed to deserialize weight {name}: {e}")
                raise
        
        logger.info(f"✅ Deserialized {len(deserialized_weights)} weights")
        
        # Register model
        logger.debug(f"🔧 Registering model with handler...")
        try:
            response = await handler._register_with_recovery({
                'fingerprint': fingerprint,
                'descriptor': descriptor,
                'weight_ids': weight_ids,
                'uncached_weights': deserialized_weights,
                'architecture_data': architecture_data
            })
            logger.debug(f"✅ Handler returned response: {response.get('status')}")
        except Exception as e:
            logger.error(f"❌ Handler registration failed: {e}")
            import traceback
            logger.debug(f"Traceback:\n{traceback.format_exc()}")
            raise
        
        # Send response
        logger.debug(f"📤 Sending response...")
        try:
            response_bytes = pickle.dumps(response)
            await _send_message(writer, 0x05, response_bytes)
            logger.info(f"✅ Response sent successfully")
        except Exception as e:
            logger.error(f"❌ Failed to send response: {e}")
            import traceback
            logger.debug(f"Traceback:\n{traceback.format_exc()}")
            raise
        
        if response.get('status') == 'success':
            STATS['requests_success'] += 1
        else:
            STATS['requests_failed'] += 1
        
    except Exception as e:
        logger.error(f"❌ Model registration failed: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        try:
            error_response = {
                'status': 'error',
                'message': str(e)
            }
            response_bytes = pickle.dumps(error_response)
            await _send_message(writer, 0x05, response_bytes)
        except Exception as send_error:
            logger.error(f"❌ Failed to send error response: {send_error}")
            # Connection might be closed, can't send error
        STATS['requests_failed'] += 1


async def handle_execute_model(request_data: bytes, writer) -> None:
    """
    Handle model execution request.
    
    Protocol:
    1. Receive: Pickled dict with fingerprint + inputs
    2. Execute via model cache
    3. Respond: Result tensor
    
    This is Week 2/3 integration.
    """
    global STATS
    
    try:
        # Deserialize request
        request = pickle.loads(request_data)
        
        fingerprint = request['fingerprint']
        inputs = request['inputs']
        hints = request.get('hints', {})
        
        logger.info(f"Executing model {fingerprint}")
        
        # Initialize resilient handler (singleton pattern)
        # Use shared handler instance for both registration and execution
        if not hasattr(handle_execute_model, '_shared_handler'):
            # Check if register handler already created one
            if hasattr(handle_register_model, '_shared_handler'):
                handle_execute_model._shared_handler = handle_register_model._shared_handler
            else:
                from .resilient_model_handler import ResilientModelHandler
                handle_execute_model._shared_handler = ResilientModelHandler(gpu_id=0)
                handle_register_model._shared_handler = handle_execute_model._shared_handler
        
        handler = handle_execute_model._shared_handler
        
        # Deserialize inputs
        deserialized_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, dict) and 'data' in value:
                import numpy as np
                # Fix dtype conversion - handle torch dtype strings
                dtype_str = value.get('dtype', 'float32')
                if dtype_str.startswith('torch.'):
                    dtype_str = dtype_str.replace('torch.', '')
                
                tensor_data = np.array(value['data'], dtype=dtype_str)
                tensor = torch.from_numpy(tensor_data)
                if list(tensor.shape) != value['shape']:
                    tensor = tensor.reshape(value['shape'])
                deserialized_inputs[key] = tensor
            else:
                deserialized_inputs[key] = value
        
        # Execute model
        response = await handler.handle_request({
            'type': 'EXECUTE_MODEL',
            'fingerprint': fingerprint,
            'inputs': deserialized_inputs,
            'hints': hints
        })
        
        # Send response
        response_bytes = pickle.dumps(response)
        await _send_message(writer, 0x06, response_bytes)
        
        if response.get('status') == 'success':
            STATS['requests_success'] += 1
        else:
            STATS['requests_failed'] += 1
        
    except Exception as e:
        logger.error(f"Model execution failed: {e}")
        import traceback
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        
        error_response = {
            'status': 'error',
            'message': str(e)
        }
        response_bytes = pickle.dumps(error_response)
        await _send_message(writer, 0x06, response_bytes)
        STATS['requests_failed'] += 1


async def handle_coordinator_subgraph(request_data: bytes, writer) -> None:
    """Handle subgraph execution request from coordinator (V1 pickle or V2 binary format)."""
    global STATS

    try:
        STATS['subgraph_requests'] += 1
        print(f"🧪 SERVER: Received subgraph request, data length: {len(request_data)}")
        logger.info(f"🧪 SERVER: Received subgraph request, data length: {len(request_data)}")

        # ✅ PHASE 2: Detect serialization format
        # V2_BINARY format: [version (1)][metadata_size (4)][metadata_json][tensor_blob]
        # V1_PICKLE format: pickled dict
        # Try to detect V2 by checking first byte (should be 1 or 2 for SerializationVersion)
        use_v2 = False
        message = None
        subgraph_data = None
        input_data_pickled = None
        cached_identifiers = []
        cached_identifier_map = {}
        uncached_identifier_map = {}
        timeout = 30
        
        if len(request_data) >= 1:
            first_byte = request_data[0]
            # V2_BINARY uses SerializationVersion enum (1 or 2)
            if first_byte in (1, 2):  # V1_PICKLE=1, V2_BINARY=2
                try:
                    from .fast_serialization import FastSerializer, SerializationVersion
                    # Try to deserialize as V2_BINARY
                    subgraph_message, input_data = FastSerializer.deserialize_subgraph(request_data)
                    use_v2 = True
                    logger.info("✅ Using V2_BINARY fast serialization format")
                    
                    # Extract data from SubgraphMessage
                    subgraph_data = {
                        'type': 'full_subgraph',
                        'subgraph': {
                            'operations': subgraph_message.operations,
                            'output_id': subgraph_message.output_id
                        }
                    }
                    # Convert input_data dict to pickled format (for compatibility with existing code)
                    import pickle
                    input_data_pickled = {k: pickle.dumps(v) for k, v in input_data.items()}
                    cached_identifiers = subgraph_message.cached_identifiers
                    cached_identifier_map = subgraph_message.cached_identifier_map
                    uncached_identifier_map = subgraph_message.uncached_identifier_map
                except (ImportError, ValueError, Exception) as e:
                    logger.debug(f"V2 deserialization failed: {e}, trying V1 pickle")
                    use_v2 = False
        
        # Fallback to V1_PICKLE format
        if not use_v2:
            try:
                message = pickle.loads(request_data)
                logger.info("✅ Using V1_PICKLE serialization format (compatibility mode)")
                logger.info(f"🧪 SERVER: Coordinator message keys: {list(message.keys())}")
                logger.info(f"🧪 SERVER: Message subgraph_data keys: {list(message.get('subgraph_data', {}).keys()) if isinstance(message.get('subgraph_data'), dict) else 'N/A'}")
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(f"Failed to parse coordinator message (both V1 and V2 failed): {error_msg}")
                error_response = f"Failed to parse message: {error_msg}".encode('utf-8')
                await _send_message(writer, 0x05, error_response)
                return

        # Extract data from coordinator message (V1 format)
        if not use_v2:
            subgraph_data = message.get('subgraph_data', {})
            # ✅ FIX: input_data is at the top level of the message, not nested in subgraph_data
            input_data_pickled = message.get('input_data', {})
            if not input_data_pickled:
                # Fallback: check if it's nested in subgraph_data (old format)
                input_data_pickled = subgraph_data.get('input_data', {})
            
            # ✅ PHASE 1: Extract cached identifiers (NEW)
            cached_identifiers = message.get('cached_identifiers', [])
            cached_identifier_map = message.get('cached_identifier_map', {})
            uncached_identifier_map = message.get('uncached_identifier_map', {})
            timeout = message.get('timeout', 30)
        
        if cached_identifiers:
            logger.info(f"📦 Received {len(cached_identifiers)} cached identifiers, {len(input_data_pickled)} tensors to deserialize")
        
        if not use_v2 and message:
            logger.info(f"🔍 Message keys: {list(message.keys())}")
        logger.info(f"🔍 input_data_pickled keys: {list(input_data_pickled.keys()) if input_data_pickled else 'None'}")
        logger.info(f"🔍 subgraph_data keys: {list(subgraph_data.keys()) if isinstance(subgraph_data, dict) else type(subgraph_data)}")

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
        meta_tensors_filtered = 0
        try:
            for key, pickled_tensor in input_data_pickled.items():
                if isinstance(pickled_tensor, bytes):
                    tensor = pickle.loads(pickled_tensor)
                    # ✅ FIX: Filter meta tensors after deserialization (final defense)
                    if isinstance(tensor, torch.Tensor) and tensor.device.type == 'meta':
                        logger.warning(
                            f"⚠️  Deserialized meta tensor[{key}] - filtering before execution. "
                            f"Meta tensors cannot be moved to GPU. This should have been filtered on client."
                        )
                        meta_tensors_filtered += 1
                        continue
                    input_data[key] = tensor
                else:
                    # ✅ FIX: Also check non-pickled tensors
                    if isinstance(pickled_tensor, torch.Tensor) and pickled_tensor.device.type == 'meta':
                        logger.warning(
                            f"⚠️  Non-pickled meta tensor[{key}] - filtering before execution."
                        )
                        meta_tensors_filtered += 1
                        continue
                    input_data[key] = pickled_tensor
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Failed to deserialize input data: {error_msg}")
            error_response = f"Failed to deserialize inputs: {error_msg}".encode('utf-8')
            await _send_message(writer, 0x05, error_response)
            return
        
        if meta_tensors_filtered > 0:
            logger.warning(
                f"⚠️  Filtered {meta_tensors_filtered} meta tensor(s) after deserialization. "
                f"This may cause execution errors if these tensors are needed."
            )
        
        # ✅ PHASE 1: Resolve cached identifiers from GPU cache (NEW)
        # Extract cached identifier map (V1 format only - V2 already extracted above)
        if not use_v2 and message:
            cached_identifier_map = message.get('cached_identifier_map', {})
        if cached_identifiers:
            from .gpu_cache import get_global_cache
            cache = get_global_cache()
            cache._migrate_to_new_format()
            
            # Resolve cached tensors and map to original tensor_id_str
            cached_tensors = {}
            for identifier in cached_identifiers:
                if identifier in cache.cache_new:
                    # Get cached tensor (already on GPU)
                    cached_tensor = cache.cache_new[identifier]
                    
                    # Map back to original tensor_id_str using the mapping
                    tensor_id_str = cached_identifier_map.get(identifier)
                    if tensor_id_str:
                        # Add with original tensor_id_str key (operations expect this)
                        cached_tensors[tensor_id_str] = cached_tensor.cpu()  # Move to CPU for consistency
                        logger.debug(f"✓ Resolved cached tensor: {identifier} → {tensor_id_str}")
                    else:
                        logger.warning(
                            f"⚠️  No tensor_id_str mapping for cached identifier {identifier}. "
                            f"Using identifier as key."
                        )
                        cached_tensors[f"cached:{identifier}"] = cached_tensor.cpu()
                else:
                    logger.warning(
                        f"⚠️  Identifier {identifier} marked as cached but not found in cache! "
                        f"Client cache state may be stale. This could happen if: "
                        f"(1) server restarted, (2) cache was evicted, or (3) migration incomplete."
                    )
            
            # Merge cached tensors into input_data
            input_data.update(cached_tensors)
            logger.info(
                f"🔍 Tensor resolution: {len(input_data)} total tensors "
                f"({len(cached_tensors)} from cache, {len(input_data_pickled)} transferred)"
            )
        
        # ✅ PHASE 1: Cache uncached tensors for future requests (NEW)
        # Extract uncached identifier map (V1 format only - V2 already extracted above)
        if not use_v2 and message:
            uncached_identifier_map = message.get('uncached_identifier_map', {})
        if uncached_identifier_map:
            from .gpu_cache import get_global_cache
            cache = get_global_cache()
            cache._migrate_to_new_format()
            
            # Cache newly received tensors by identifier
            tensors_to_cache = {}
            for tensor_id_str, identifier in uncached_identifier_map.items():
                if tensor_id_str in input_data:
                    tensor = input_data[tensor_id_str]
                    if isinstance(tensor, torch.Tensor):
                        tensors_to_cache[identifier] = tensor
            
            if tensors_to_cache:
                # Use new identifier-based cache API
                cache.get_weights_by_identifier(tensors_to_cache)
                logger.debug(f"💾 Cached {len(tensors_to_cache)} new tensors by identifier")

        # Execute subgraph
        try:
            print(f"🔍 SERVER: Subgraph keys: {list(subgraph.keys()) if isinstance(subgraph, dict) else type(subgraph)}")
            print(f"🔍 SERVER: Subgraph operations: {len(subgraph.get('operations', [])) if isinstance(subgraph, dict) else 'N/A'}")
            logger.info(f"🔍 Subgraph keys: {list(subgraph.keys()) if isinstance(subgraph, dict) else type(subgraph)}")
            logger.info(f"🔍 Subgraph operations: {len(subgraph.get('operations', [])) if isinstance(subgraph, dict) else 'N/A'}")

            if OPTIMIZATION_EXECUTOR:
                logger.info(f"🚀 Starting subgraph execution: {len(subgraph.get('operations', []))} operations, {len(input_data)} inputs")
                execution_start = time.time()
                try:
                    # Try to infer model_id from subgraph (for weight caching)
                    # For GPT2, we can detect it from the structure
                    model_id = None
                    if len(input_data) > 500:  # Likely a large model with many weights
                        # Try to infer from operation patterns or use a default
                        model_id = "gpt2"  # Default for now
                    
                    result, stats = await OPTIMIZATION_EXECUTOR.execute(
                        subgraph_request=subgraph,
                        input_data=input_data,
                        model_id=model_id,  # Enable weight caching
                        timeout=timeout
                    )
                    execution_time = time.time() - execution_start
                    logger.info(f"✅ Subgraph execution completed in {execution_time:.2f}s")
                    logger.debug(f"Optimization stats: {stats}")
                except Exception as e:
                    execution_time = time.time() - execution_start
                    logger.error(f"❌ Subgraph execution failed after {execution_time:.2f}s: {type(e).__name__}: {str(e)}")
                    import traceback
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    raise

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
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Subgraph execution failed: {error_msg}")
            logger.debug(f"Traceback:\n{traceback.format_exc()}")
            error_response = error_msg.encode('utf-8')
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

            # ✅ FIX: Send result in format expected by coordinator client
            # Format: [type (1)] [profiling_size (4)] [profiling_json] [tensor_size (4)] [tensor_bytes]
            # Coordinator expects profiling metadata even if empty (for protocol consistency)
            profiling_data = {}  # Empty profiling data for coordinator path
            profiling_json = json.dumps(profiling_data).encode('utf-8')
            profiling_size = len(profiling_json)
            
            # Send result with profiling metadata (type=0x04: RESULT)
            # Message format: [type (1)] [profiling_size (4)] [profiling_json] [tensor_size (4)] [tensor_bytes]
            message = struct.pack('>BI', 0x04, profiling_size) + profiling_json + struct.pack('>I', len(result_bytes)) + result_bytes
            writer.write(message)
            await writer.drain()

            STATS['requests_success'] += 1
            logger.info(f"✅ Coordinator subgraph execution complete: {result.shape}")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Failed to serialize result: {error_msg}")
            error_response = f"Failed to serialize result: {error_msg}".encode('utf-8')
            await _send_message(writer, 0x05, error_response)

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Error in coordinator subgraph handler: {error_msg}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        STATS['requests_failed'] += 1
        try:
            error_response = f"Internal error: {error_msg}".encode('utf-8')
            await _send_message(writer, 0x05, error_response)
        except:
            pass  # Connection might be closed


async def handle_execute_subgraph(request_data: bytes, writer) -> None:
    """Handle subgraph execution request."""
    global STATS
    
    # ✅ PROFILING: Set up server-side profiling context
    from .profiling_context import ProfilingContext, set_profiler
    server_profiler = ProfilingContext(enabled=True)
    server_profiler.start()
    set_profiler(server_profiler)
    
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
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(f"Failed to parse request: {error_msg}")
                error_response = error_msg.encode('utf-8')
                await _send_message(writer, 0x05, error_response)  # ERROR
                return

            # Parse subgraph request
            subgraph_request = json.loads(request_json)
            
            # Deserialize input tensors (profiling happens inside deserialize_tensor)
            try:
                input_data = deserialize_tensor(tensors_data)
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(f"Failed to deserialize tensors: {error_msg}")
                error_response = error_msg.encode('utf-8')
                await _send_message(writer, 0x05, error_response)
                return

            logger.info(f"🎯 Executing subgraph: {len(subgraph_request['operations'])} operations")

            # Execute subgraph with optimizations (profiling happens inside executor)
            try:
                result = OPTIMIZATION_EXECUTOR.executor.execute(subgraph_request, input_data)
            except Exception as e:
                import traceback
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(f"❌ Subgraph execution failed: {error_msg}")
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
                error_response = error_msg.encode('utf-8')
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
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(f"Failed to serialize result: {error_msg}")
                error_response = error_msg.encode('utf-8')
                await _send_message(writer, 0x05, error_response)
                return

            # ✅ PROFILING: Collect server-side profiling data before sending response
            profiling_data = {}
            try:
                from .profiling_context import get_profiler
                current_profiler = get_profiler()
                if current_profiler:
                    profiling_data = current_profiler.get_phase_dict()
                    logger.debug(f"📊 Server-side profiling data: {profiling_data}")
            except Exception as e:
                logger.debug(f"Could not collect profiling data: {e}")
            
            # ✅ PROFILING: Include profiling data in response
            # Format: [message_type (1 byte)] [profiling_json_size (4 bytes)] [profiling_json] [tensor_size (4 bytes)] [tensor_bytes]
            # This allows client to extract profiling data before deserializing tensor
            profiling_json = json.dumps(profiling_data).encode('utf-8')
            profiling_size = len(profiling_json)
            
            # Send result with profiling metadata (type=0x04: RESULT)
            # Message format: [type (1)] [profiling_size (4)] [profiling_json] [tensor_size (4)] [tensor_bytes]
            message = struct.pack('>BI', 0x04, profiling_size) + profiling_json + struct.pack('>I', len(result_bytes)) + result_bytes
            writer.write(message)
            await writer.drain()

            STATS['requests_success'] += 1
            logger.info(f"✅ Subgraph execution complete: {result.shape}")
            
            # Clean up profiling context
            set_profiler(None)

        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"❌ Error handling subgraph: {error_msg}")
            logger.debug(f"Traceback:\n{traceback.format_exc()}")
            STATS['requests_failed'] += 1
            error_response = error_msg.encode('utf-8')
            await _send_message(writer, 0x05, error_response)
            # Clean up profiling context on error too
            set_profiler(None)


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
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"❌ Error handling operation: {error_msg}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        STATS['requests_failed'] += 1
        error_response = error_msg.encode('utf-8')
        await _send_message(writer, 0x05, error_response)


async def _send_message(writer, message_type: int, data: bytes) -> None:
    """Send length-prefixed message.
    
    Protocol: [message_type (1 byte)] [size (8 bytes)] [data]
    Matches _recv_message protocol for large messages.
    """
    message = struct.pack('>BQ', message_type, len(data)) + data  # Q = 8-byte unsigned long long
    writer.write(message)
    await writer.drain()


async def _recv_message(reader, timeout: float = 300.0) -> Tuple[int, bytes]:
    """Receive length-prefixed message.
    
    Protocol: [message_type (1 byte)] [size (8 bytes)] [data]
    Updated to support large messages > 4GB (for model weights).
    """
    try:
        logger.debug("📥 Reading message type...")
        # Read message type (1 byte)
        msg_type_bytes = await asyncio.wait_for(
            reader.readexactly(1),
            timeout=timeout
        )
        msg_type = msg_type_bytes[0]
        logger.debug(f"📥 Message type: {msg_type} (0x{msg_type:02x})")
        
        # Read size (8 bytes for large messages)
        logger.debug("📥 Reading message length...")
        size_bytes = await asyncio.wait_for(
            reader.readexactly(8),
            timeout=timeout
        )
        length = int.from_bytes(size_bytes, 'big')
        logger.debug(f"📥 Message length: {length} bytes")
        
        # Read data
        logger.debug(f"📥 Reading {length} bytes of data...")
        data = await asyncio.wait_for(
            reader.readexactly(length),
            timeout=timeout
        )
        logger.debug(f"✅ Received complete message: type={msg_type}, length={length}")
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
        logger.debug(f"🔍 Entering message loop for {client_addr}")
        while not reader.at_eof():
            logger.debug(f"🔍 Waiting for message from {client_addr}...")
            # Receive message
            try:
                msg_type, data = await _recv_message(reader)
                logger.info(f"📨 Received message type {msg_type} (0x{msg_type:02x}) from {client_addr}, {len(data)} bytes")
            except Exception as recv_error:
                logger.error(f"❌ Error receiving message from {client_addr}: {recv_error}")
                import traceback
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                raise

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

            elif msg_type == 0x04:  # CACHE_QUERY (NEW - Phase 1)
                logger.debug("📊 Handling CACHE_QUERY")
                await handle_cache_query(data, writer)

            elif msg_type == 0x05:  # REGISTER_MODEL (NEW - Week 2/3)
                logger.info(f"📊 Handling REGISTER_MODEL (received {len(data)} bytes)")
                try:
                    await handle_register_model(data, writer)
                    logger.info("✅ REGISTER_MODEL handler completed")
                except Exception as e:
                    import traceback
                    logger.error(f"❌ REGISTER_MODEL handler failed: {e}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    # Re-raise to be caught by outer exception handler
                    raise

            elif msg_type == 0x06:  # EXECUTE_MODEL (NEW - Week 2/3)
                logger.info("📊 Handling EXECUTE_MODEL")
                await handle_execute_model(data, writer)

            else:
                logger.warning(f"Unknown message type: {msg_type}")
                await _send_message(writer, 0xFF, b"Unknown message type")

    except asyncio.IncompleteReadError:
        logger.info(f"Client {client_addr} disconnected (incomplete read)")
    except Exception as e:
        import traceback
        logger.error(f"❌ Error handling connection from {client_addr}: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        # Try to send error response before closing
        try:
            error_msg = f"Server error: {str(e)}".encode('utf-8')
            await _send_message(writer, 0xFF, error_msg)
        except:
            pass  # Connection might already be closed
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except:
            pass
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
