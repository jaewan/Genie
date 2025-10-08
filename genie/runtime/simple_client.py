"""
Simple HTTP client for remote execution.
File: genie/runtime/simple_client.py
"""

import requests
import torch
import io
import logging
from typing import Optional, Dict, Any
import time

logger = logging.getLogger(__name__)


class RemoteExecutionClient:
    """
    Client for executing tensors on remote server.
    Uses standard HTTP/REST.
    """

    def __init__(self, server_url: str = "http://localhost:8888"):
        """
        Initialize client.

        Args:
            server_url: Base URL of server (default: http://localhost:8888)
        """
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()  # Reuse connections

        # Statistics
        self.stats = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_failed': 0,
            'total_bytes_sent': 0,
            'total_bytes_received': 0,
            'total_time_seconds': 0.0
        }

        logger.info(f"Created RemoteExecutionClient: {self.server_url}")

    def health_check(self, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Check server health.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Health status dict

        Raises:
            requests.RequestException: If health check fails
        """
        try:
            response = self.session.get(
                f"{self.server_url}/health",
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Health check failed: {e}")
            raise

    def execute(self,
                operation: str,
                tensor: torch.Tensor,
                timeout: float = 30.0) -> torch.Tensor:
        """
        Execute operation on remote server.

        Args:
            operation: Operation name (e.g., "relu")
            tensor: Input tensor
            timeout: Request timeout in seconds

        Returns:
            Result tensor

        Raises:
            requests.RequestException: If request fails
            RuntimeError: If execution fails
        """
        start_time = time.time()
        self.stats['requests_total'] += 1

        try:
            # Log request
            logger.debug(f"Executing {operation} on tensor {tensor.shape}")

            # Serialize tensor
            tensor_bytes = io.BytesIO()

            # Move to CPU if on GPU (Phase 1 limitation)
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                logger.warning(
                    "Moving GPU tensor to CPU for transfer "
                    "(Phase 1 limitation)"
                )
                tensor_cpu = tensor.cpu()
            else:
                tensor_cpu = tensor

            torch.save(tensor_cpu, tensor_bytes)
            tensor_bytes.seek(0)

            # Track size
            tensor_size = len(tensor_bytes.getvalue())
            self.stats['total_bytes_sent'] += tensor_size

            # Send HTTP POST
            response = self.session.post(
                f"{self.server_url}/execute",
                files={
                    'tensor_file': (
                        'tensor.pt',
                        tensor_bytes,
                        'application/octet-stream'
                    )
                },
                data={'operation': operation},
                timeout=timeout
            )

            # Check for errors
            response.raise_for_status()

            # Track received size
            self.stats['total_bytes_received'] += len(response.content)

            # Deserialize result
            result = torch.load(io.BytesIO(response.content))

            # Track statistics
            elapsed = time.time() - start_time
            self.stats['requests_success'] += 1
            self.stats['total_time_seconds'] += elapsed

            logger.debug(
                f"Executed {operation}: "
                f"{tensor.shape} -> {result.shape} "
                f"in {elapsed:.3f}s"
            )

            return result

        except requests.RequestException as e:
            self.stats['requests_failed'] += 1
            logger.error(f"HTTP error during remote execution: {e}")
            raise
        except Exception as e:
            self.stats['requests_failed'] += 1
            logger.error(f"Error during remote execution: {e}", exc_info=True)
            raise RuntimeError(f"Remote execution failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        stats = self.stats.copy()

        # Calculate derived stats
        if stats['requests_total'] > 0:
            stats['success_rate'] = (
                stats['requests_success'] / stats['requests_total']
            )
            stats['avg_time_seconds'] = (
                stats['total_time_seconds'] / stats['requests_success']
                if stats['requests_success'] > 0 else 0.0
            )
        else:
            stats['success_rate'] = 0.0
            stats['avg_time_seconds'] = 0.0

        return stats

    def close(self):
        """Close the session."""
        self.session.close()
        logger.info("Closed RemoteExecutionClient")


# Global client instance (singleton)
_global_client: Optional[RemoteExecutionClient] = None


def get_client(server_url: str = "http://localhost:8888") -> RemoteExecutionClient:
    """
    Get global client instance.

    Args:
        server_url: Server URL (default: http://localhost:8888)

    Returns:
        RemoteExecutionClient instance
    """
    global _global_client

    if _global_client is None:
        _global_client = RemoteExecutionClient(server_url)

    return _global_client


def set_server_url(server_url: str):
    """
    Set server URL for global client.

    Args:
        server_url: New server URL
    """
    global _global_client
    _global_client = RemoteExecutionClient(server_url)
