"""
Simple HTTP server for remote tensor execution.
File: genie/runtime/simple_server.py
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import Response, JSONResponse
import torch
import io
import json
import logging
from datetime import datetime
import traceback
import os
import numpy as np
import time
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import optimized serialization
try:
    from ..core.serialization import serialize_tensor, deserialize_tensor
    OPTIMIZED_SERIALIZATION_AVAILABLE = True
    logger.info("✅ Optimized serialization module loaded")
except ImportError:
    logger.warning("⚠️  Optimized serialization not available, using torch.save/load")
    OPTIMIZED_SERIALIZATION_AVAILABLE = False

# Feature flag for optimized serialization (can be disabled via environment variable)
USE_OPTIMIZED_SERIALIZATION = os.getenv('GENIE_USE_OPTIMIZED_SERIALIZATION', 'true').lower() == 'true'
if USE_OPTIMIZED_SERIALIZATION and OPTIMIZED_SERIALIZATION_AVAILABLE:
    logger.info("🚀 Using optimized numpy serialization (44% faster)")
else:
    logger.info("📦 Using standard torch.save serialization")

# Create FastAPI app
app = FastAPI(
    title="Genie Remote Execution Server",
    description="Executes PyTorch operations on remote GPU",
    version="0.1.0"
)

# Global device (will be set on startup)
DEVICE = None

# Subgraph executor (will be initialized on startup)
SUBGRAPH_EXECUTOR = None

# GPU cache (will be initialized on startup)
GPU_CACHE = None

# Graph cache (will be initialized on startup)
GRAPH_CACHE = None

# Statistics
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


@app.on_event("startup")
async def startup_event():
    """Initialize server on startup."""
    global DEVICE, SUBGRAPH_EXECUTOR, GPU_CACHE, GRAPH_CACHE, STATS

    # Set device
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
        logger.info(f"🚀 Server starting with GPU: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device("cpu")
        logger.warning("⚠️  No GPU available, using CPU")

    # Initialize GPU cache
    try:
        from ..server.gpu_cache import get_global_cache
        GPU_CACHE = get_global_cache(max_models=5)
        logger.info("✅ GPU cache initialized (max_models=5)")
    except Exception as e:
        logger.warning(f"⚠️  Failed to initialize GPU cache: {e}")
        logger.info("   GPU caching will be disabled")

    # Initialize graph cache
    try:
        from ..server.graph_cache import get_global_graph_cache
        GRAPH_CACHE = get_global_graph_cache(max_graphs=100)
        logger.info("✅ Graph cache initialized (max_graphs=100)")
    except Exception as e:
        logger.warning(f"⚠️  Failed to initialize graph cache: {e}")
        logger.info("   Graph caching will be disabled")

    # Initialize subgraph executor
    try:
        from ..server.subgraph_executor import SubgraphExecutor
        SUBGRAPH_EXECUTOR = SubgraphExecutor(gpu_id=0)
        logger.info("✅ SubgraphExecutor initialized")
    except Exception as e:
        logger.warning(f"⚠️  Failed to initialize SubgraphExecutor: {e}")
        logger.info("   Subgraph execution will be disabled")

    STATS['start_time'] = datetime.now()
    logger.info(f"✅ Server ready on device: {DEVICE}")


@app.get("/")
async def root():
    """Root endpoint - just returns info."""
    return {
        "service": "Genie Remote Execution Server",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "execute": "/execute (POST) - single operation",
            "execute_subgraph": "/execute_subgraph (POST) - optimized subgraph"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Test with: curl http://localhost:8888/health
    """
    uptime = None
    if STATS['start_time']:
        uptime = str(datetime.now() - STATS['start_time'])

    return {
        "status": "healthy",
        "device": str(DEVICE),
        "device_type": DEVICE.type if DEVICE else "unknown",
        "cuda_available": torch.cuda.is_available(),
        "subgraph_executor": "available" if SUBGRAPH_EXECUTOR else "unavailable",
        "uptime": uptime,
        "stats": {
            "total_requests": STATS['requests_total'],
            "successful": STATS['requests_success'],
            "failed": STATS['requests_failed'],
            "subgraph_requests": STATS['subgraph_requests']
        }
    }


@app.post("/execute")
async def execute_operation(
    operation: str = Form(...),
    tensor_file: UploadFile = File(...)
):
    """
    Execute tensor operation on server GPU.

    Args:
        operation: Operation name (e.g., "relu", "sigmoid")
        tensor_file: Binary file containing torch tensor

    Returns:
        Binary tensor result

    Test with:
        curl -X POST http://localhost:8888/execute \
             -F "operation=relu" \
             -F "tensor_file=@test_tensor.pt" \
             --output result.pt
    """
    global STATS
    STATS['requests_total'] += 1

    start_time = datetime.now()
    server_total_start = time.perf_counter()

    try:
        logger.info(f"📥 Received request: operation={operation}")

        # Read uploaded tensor
        tensor_bytes = await tensor_file.read()
        logger.debug(f"   Received {len(tensor_bytes)} bytes")

        # Deserialize tensor
        tensor = torch.load(io.BytesIO(tensor_bytes))
        logger.info(f"   Loaded tensor: shape={tensor.shape}, dtype={tensor.dtype}")

        # Move to GPU (handle both torch.Tensor and LazyTensor)
        if hasattr(tensor, 'materialize'):
            # It's a LazyTensor - materialize first
            tensor = tensor.materialize()
        tensor = tensor.to(DEVICE)

        # Execute operation
        result = _execute_single_operation(operation, tensor)

        # Serialize result
        result_bytes = io.BytesIO()
        torch.save(result.cpu(), result_bytes)
        result_bytes.seek(0)

        # Statistics
        elapsed = (datetime.now() - start_time).total_seconds()
        STATS['requests_success'] += 1

        logger.info(f"✅ Success: {operation} completed in {elapsed:.3f}s")

        return Response(
            content=result_bytes.read(),
            media_type="application/octet-stream"
        )

    except Exception as e:
        STATS['requests_failed'] += 1
        logger.error(f"❌ Error executing {operation}: {e}")
        logger.error(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail=f"Execution failed: {str(e)}"
        )


@app.post("/execute_subgraph")
async def handle_subgraph_execution(request: Request):
    """
    NEW ENDPOINT: Execute subgraph on server GPU.

    This implements the core optimization from the network enhancement plan:
    execute entire computation graphs in a single network request.

    Request (multipart/form-data):
        - request: JSON subgraph specification
        - tensors: Serialized tensor data

    Response:
        - Binary tensor data (final result)

    Test with:
        curl -X POST http://localhost:8888/execute_subgraph \
             -F "request=@request.json" \
             -F "tensors=@tensors.pt" \
             --output result.pt
    """
    global STATS, SUBGRAPH_EXECUTOR
    STATS['requests_total'] += 1
    STATS['subgraph_requests'] += 1

    start_time = datetime.now()
    server_total_start = time.perf_counter()

    try:
        logger.info(f"📥 Received subgraph execution request")
        logger.info(
            "   Server CUDA status: available=%s current_device=%s target_device=%s",
            torch.cuda.is_available(),
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
            DEVICE,
        )

        # Check if subgraph executor is available
        if SUBGRAPH_EXECUTOR is None:
            raise HTTPException(
                status_code=503,
                detail="Subgraph execution not available. Server may be running without GPU support."
            )

        # Parse multipart form data
        form = await request.form()

        # Parse subgraph request (with graph caching)
        request_data = await form['request'].read()
        if GRAPH_CACHE is not None:
            subgraph_request = GRAPH_CACHE.get_graph(request_data)
            graph_stats = GRAPH_CACHE.get_stats()
            STATS['graph_cache_hits'] = graph_stats['hits']
            STATS['graph_cache_misses'] = graph_stats['misses']
        else:
            subgraph_request = torch.load(io.BytesIO(request_data))
        
        logger.info(f"   Subgraph: {len(subgraph_request['operations'])} operations, "
                   f"{len(subgraph_request['input_tensors'])} inputs")
        if GRAPH_CACHE is not None:
            graph_hash = hashlib.md5(request_data).hexdigest()[:8]
            cache_status = "HIT" if graph_hash in [h[:8] for h in GRAPH_CACHE.cache.keys()] else "MISS"
            logger.info(f"   Graph cache: {cache_status}")

        # Load input tensors with timing
        deserialize_start = time.perf_counter()
        tensors_data = await form['tensors'].read()
        input_tensors_cpu = torch.load(io.BytesIO(tensors_data))
        
        # Try to use GPU cache if available
        model_id = subgraph_request.get('model_id', 'default')
        if GPU_CACHE is not None:
            # Convert tensors to numpy for caching
            input_tensors_numpy = {
                int(k): v.numpy() if isinstance(v, torch.Tensor) else v
                for k, v in input_tensors_cpu.items()
            }
            # Get cached GPU tensors (or load if cache miss)
            input_tensors_gpu = GPU_CACHE.get_weights(model_id, input_tensors_numpy)
            # Convert back to string keys for executor
            input_tensors = {str(k): v for k, v in input_tensors_gpu.items()}
            
            # Update stats
            cache_stats = GPU_CACHE.get_stats()
            STATS['gpu_cache_hits'] = cache_stats['hits']
            STATS['gpu_cache_misses'] = cache_stats['misses']
        else:
            # No cache - use CPU tensors directly
            input_tensors = input_tensors_cpu
        
        deserialize_ms = (time.perf_counter() - deserialize_start) * 1000
        logger.info(f"   Input tensors: {len(input_tensors)} tensors (deser {deserialize_ms:.2f} ms)")
        if GPU_CACHE is not None:
            cache_status = "HIT" if model_id in GPU_CACHE.cache else "MISS"
            logger.info(f"   GPU cache: {cache_status} for model_id={model_id}")
        for tensor_key, tensor_value in list(input_tensors.items())[:5]:  # Log first 5 only
            logger.info(
                "      tensor[%s]: shape=%s dtype=%s device=%s",
                tensor_key,
                tuple(tensor_value.shape),
                tensor_value.dtype,
                tensor_value.device,
            )

        # Execute subgraph
        execute_start = time.perf_counter()
        result = SUBGRAPH_EXECUTOR.execute(subgraph_request, input_tensors)
        execution_ms = (time.perf_counter() - execute_start) * 1000

        # Serialize result (with optimized serialization if available)
        serialize_start = time.perf_counter()
        if USE_OPTIMIZED_SERIALIZATION and OPTIMIZED_SERIALIZATION_AVAILABLE:
            # Use optimized numpy serialization (44% faster!)
            serialized_result = serialize_tensor(result, use_numpy=True)
            logger.debug(f"Serialized result with numpy: {len(serialized_result)} bytes")
        else:
            # Fallback to torch.save
            result_bytes = io.BytesIO()
            torch.save(result, result_bytes)
            serialized_result = result_bytes.getvalue()
            logger.debug(f"Serialized result with torch.save: {len(serialized_result)} bytes")
        serialization_ms = (time.perf_counter() - serialize_start) * 1000

        # Statistics
        elapsed = (datetime.now() - start_time).total_seconds()
        STATS['requests_success'] += 1

        logger.info(
            "✅ Subgraph execution successful: %s in %.3fs (result device before send=%s)",
            result.shape,
            elapsed,
            result.device,
        )
        total_ms = (time.perf_counter() - server_total_start) * 1000
        server_timing = {
            "deserialize_ms": deserialize_ms,
            "execution_ms": execution_ms,
            "serialize_ms": serialization_ms,
            "total_ms": total_ms
        }

        return Response(
            content=serialized_result,
            media_type="application/octet-stream",
            headers={
                "X-Genie-Server-Timing": json.dumps(server_timing),
                "X-Genie-Result-Bytes": str(len(serialized_result))
            }
        )

    except Exception as e:
        STATS['requests_failed'] += 1
        logger.error(f"❌ Subgraph execution failed: {e}")
        logger.error(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail=f"Subgraph execution failed: {str(e)}"
        )


def _execute_single_operation(operation: str, tensor: torch.Tensor) -> torch.Tensor:
    """
    Execute a single operation on a tensor.

    Supported operations (Phase 1):
    - relu, sigmoid, tanh, abs
    - neg, exp, log, sqrt
    """
    # Define supported operations
    SUPPORTED = {
        'relu': torch.relu,
        'sigmoid': torch.sigmoid,
        'tanh': torch.tanh,
        'abs': torch.abs,
        'neg': torch.neg,
        'exp': torch.exp,
        'log': torch.log,
        'sqrt': torch.sqrt,
    }

    if operation not in SUPPORTED:
        raise ValueError(
            f"Operation '{operation}' not supported. "
            f"Supported: {list(SUPPORTED.keys())}"
        )

    # Execute
    func = SUPPORTED[operation]
    result = func(tensor)

    return result


def start_server(host: str = "0.0.0.0", port: int = 8888):
    """Start the FastAPI server."""
    import uvicorn

    logger.info("=" * 60)
    logger.info("🚀 Starting Genie Remote Execution Server")
    logger.info(f"   Host: {host}")
    logger.info(f"   Port: {port}")
    logger.info(f"   URL: http://{host}:{port}")
    logger.info("=" * 60)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Genie Remote Execution Server'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8888,
        help='Port to bind (default: 8888)'
    )

    args = parser.parse_args()
    start_server(host=args.host, port=args.port)
