"""
Simple HTTP server for remote tensor execution.
File: genie/runtime/simple_server.py
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
import torch
import io
import logging
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Genie Remote Execution Server",
    description="Executes PyTorch operations on remote GPU",
    version="0.1.0"
)

# Global device (will be set on startup)
DEVICE = None

# Statistics
STATS = {
    'requests_total': 0,
    'requests_success': 0,
    'requests_failed': 0,
    'start_time': None
}


@app.on_event("startup")
async def startup_event():
    """Initialize server on startup."""
    global DEVICE, STATS

    # Set device
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
        logger.info(f"🚀 Server starting with GPU: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device("cpu")
        logger.warning("⚠️  No GPU available, using CPU")

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
            "execute": "/execute (POST)"
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
        "uptime": uptime,
        "stats": {
            "total_requests": STATS['requests_total'],
            "successful": STATS['requests_success'],
            "failed": STATS['requests_failed']
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
