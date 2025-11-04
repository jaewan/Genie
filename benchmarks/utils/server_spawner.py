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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RemoteServerManager:
    """Manages spawning and lifecycle of remote Genie server."""

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
            # First, check if port is available
            if not self._is_port_available(self.host, self.port):
                logger.warning(f"Port {self.port} is in use, trying next port...")
                # Try alternative ports
                for alt_port in [5557, 5558, 5559, 5560]:
                    if self._is_port_available(self.host, alt_port):
                        self.port = alt_port
                        logger.info(f"Using alternative port: {alt_port}")
                        break
                else:
                    logger.error("No available ports found")
                    return False
            
            # Spawn server in daemon process using 'spawn' context (CUDA-safe)
            ctx = mp.get_context('spawn')
            self.process = ctx.Process(
                target=self._run_server_subprocess,
                daemon=True,
                name="genie-server"
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
        
        while attempt < max_attempts and (time.time() - start_time < self.timeout):
            attempt += 1
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((self.host, self.port))
                sock.close()
                if result == 0:
                    logger.info(f"✓ Server connection successful on attempt {attempt}")
                    time.sleep(0.2)  # Extra delay to ensure server is fully ready
                    return True
            except Exception as e:
                logger.debug(f"Connection attempt {attempt} failed: {e}")
            
            time.sleep(check_interval)
        
        elapsed = time.time() - start_time
        logger.warning(f"Server not ready after {elapsed:.1f}s ({attempt} attempts)")
        return False

    def _run_server_subprocess(self):
        """Run server in separate process."""
        try:
            # Import Genie TCP server - the new async implementation
            from genie.server.tcp_server import start_server, initialize_server
            
            logger.info(f"Starting TCP server on port {self.port}...")
            
            # Start TCP server (this blocks)
            try:
                async def run_tcp_server():
                    """Initialize and start TCP server."""
                    await initialize_server()
                    await start_server(host="127.0.0.1", port=self.port)
                
                asyncio.run(run_tcp_server())
            except Exception as e:
                logger.error(f"TCP server start failed: {e}")
                import traceback
                traceback.print_exc()
            
        except ImportError as e:
            logger.error(f"Failed to import Genie TCP server: {e}")
            logger.error("Make sure Genie is properly installed")
            import traceback
            traceback.print_exc()
        except Exception as e:
            logger.error(f"Server error: {e}")
            import traceback
            traceback.print_exc()
