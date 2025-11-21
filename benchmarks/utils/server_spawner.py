"""Remote server spawner for benchmark evaluation."""

import asyncio
import logging
import multiprocessing as mp
import socket
import time
import sys
from typing import Optional
from pathlib import Path

# Set multiprocessing start method to 'spawn' to avoid CUDA reinitialization issues
# This must be done before any processes are created
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Reduce verbosity for Djinn components during end-to-end testing
logging.getLogger('djinn').setLevel(logging.ERROR)
logging.getLogger('djinn.server').setLevel(logging.ERROR)
logging.getLogger('djinn.core').setLevel(logging.ERROR)
logging.getLogger('djinn.frontend').setLevel(logging.ERROR)
logging.getLogger('djinn.backend').setLevel(logging.ERROR)


class RemoteServerManager:
    """Manages spawning and lifecycle of remote Djinn server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5556, timeout: int = 60):
        """
        Initialize server manager.
        
        Args:
            host: Server host address
            port: Server port
            timeout: Timeout for server startup in seconds (increased to 60 for initialization)
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.process: Optional[mp.Process] = None
        self.server_ready = False

    def start(self) -> bool:
        """
        Spawn server process in background.
        
        Returns:
            True if server started successfully, False otherwise
        """
        try:
            # Don't check port availability - let the server handle port conflicts
            # The server will automatically fall back to alternative ports if needed
            logger.info(f"Starting server on port {self.port} (server will handle port conflicts)")
            
            # Spawn server in daemon process using 'spawn' context (CUDA-safe)
            ctx = mp.get_context('spawn')
            self.process = ctx.Process(
                target=self._run_server_subprocess,
                daemon=True,
                name="djinn-server"
            )
            self.process.start()
            logger.info(f"Server process started (PID: {self.process.pid})")

            # Wait for server to be ready with more aggressive checking
            if self._wait_for_server():
                self.server_ready = True
                logger.info(f"✓ Server ready on {self.host}:{self.port}")
                return True
            else:
                logger.warning(f"Server did not respond within timeout ({self.timeout}s)")
                self.stop()
                return False

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            import traceback
            traceback.print_exc()
            return False

    def stop(self):
        """Terminate server process."""
        if self.process:
            try:
                self.process.terminate()
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join()
                logger.info("✓ Server process terminated")
            except Exception as e:
                logger.error(f"Error stopping server: {e}")

    def is_ready(self) -> bool:
        """Check if server is ready."""
        return self.server_ready
    
    def _is_port_available(self, host: str, port: int) -> bool:
        """Check if a port is available."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            return result != 0
        except Exception:
            return True

    def _wait_for_server(self, check_interval: float = 0.2, max_attempts: int = None) -> bool:
        """
        Wait for server to be ready by checking TCP connection.
        
        Args:
            check_interval: Interval between checks in seconds
            max_attempts: Maximum number of checks (overrides timeout)
            
        Returns:
            True if server is ready, False if timeout
        """
        if max_attempts is None:
            max_attempts = int(self.timeout / check_interval)
        
        start_time = time.time()
        attempt = 0
        
        # Try connecting to the expected port and fallback ports
        ports_to_try = [self.port, 5557, 5558, 5559, 5560]

        while attempt < max_attempts and (time.time() - start_time < self.timeout):
            attempt += 1
            for port in ports_to_try:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex((self.host, port))
                    sock.close()
                    if result == 0:
                        # Update self.port to the actual port the server is using
                        if port != self.port:
                            logger.info(f"Server is using port {port} instead of {self.port}")
                            self.port = port
                        logger.info(f"✓ Server connection successful on attempt {attempt} (port {port})")
                        time.sleep(0.2)  # Extra delay to ensure server is fully ready
                        return True
                except Exception as e:
                    logger.debug(f"Connection attempt {attempt} failed on port {port}: {e}")
                    continue
            
            time.sleep(check_interval)
        
        elapsed = time.time() - start_time
        logger.warning(f"Server not ready after {elapsed:.1f}s ({attempt} attempts)")
        return False

    def _run_server_subprocess(self):
        """Run server in separate process."""
        try:
            # Set up environment for subprocess (important for spawn context)
            import sys
            import os

            # Ensure we're using the same Python executable and environment
            venv_python = sys.executable
            logger.info(f"Using Python executable: {venv_python}")

            # Add current directory to Python path
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
                logger.info(f"Added to Python path: {current_dir}")

            # Import Djinn server - use the main server implementation
            from djinn.server.server import DjinnServer, ServerConfig
            import asyncio

            logger.info(f"Starting Djinn server on port {self.port}...")

            # Start server (this blocks)
            try:
                async def run_server():
                    """Initialize and start server."""
                    config = ServerConfig(
                        node_id="test-server",
                        control_port=self.port - 1,  # Control port is typically data_port - 1
                        data_port=self.port,
                        tcp_fallback=True
                    )
                    server = DjinnServer(config)
                    success = await server.start()
                    if not success:
                        raise RuntimeError("Server failed to start")
                    # Keep server running
                    try:
                        while True:
                            await asyncio.sleep(1)
                    except KeyboardInterrupt:
                        await server.stop()

                asyncio.run(run_server())
            except Exception as e:
                logger.error(f"TCP server start failed: {e}")
                import traceback
                traceback.print_exc()

        except ImportError as e:
            logger.error(f"Failed to import Djinn TCP server: {e}")
            logger.error("Make sure Djinn is properly installed")
            import traceback
            traceback.print_exc()
        except Exception as e:
            logger.error(f"Server error: {e}")
            import traceback
            traceback.print_exc()
