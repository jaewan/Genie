"""Remote server spawner for benchmark evaluation."""

import asyncio
import logging
import multiprocessing as mp
import socket
import time
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RemoteServerManager:
    """Manages spawning and lifecycle of remote Genie server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5556, timeout: int = 30):
        """
        Initialize server manager.
        
        Args:
            host: Server host address
            port: Server port
            timeout: Timeout for server startup in seconds
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
            # Spawn server in daemon process
            self.process = mp.Process(
                target=self._run_server,
                daemon=True
            )
            self.process.start()
            logger.info(f"Server process started (PID: {self.process.pid})")

            # Wait for server to be ready
            if self._wait_for_server():
                self.server_ready = True
                logger.info(f"✓ Server ready on {self.host}:{self.port}")
                return True
            else:
                logger.warning("Server did not respond within timeout")
                self.stop()
                return False

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
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

    def _wait_for_server(self, check_interval: float = 0.5) -> bool:
        """
        Wait for server to be ready by checking TCP connection.
        
        Args:
            check_interval: Interval between checks in seconds
            
        Returns:
            True if server is ready, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((self.host, self.port))
                sock.close()
                if result == 0:
                    time.sleep(0.5)  # Extra delay to ensure server is fully ready
                    return True
            except Exception:
                pass
            
            time.sleep(check_interval)
        
        return False

    def _run_server(self):
        """Run server in separate process."""
        try:
            # Import Genie server
            from genie.server.server import GenieServer, ServerConfig
            
            # Create server configuration
            config = ServerConfig(
                node_id='benchmark-server',
                data_port=self.port
            )
            
            # Start server
            server = GenieServer(config)
            asyncio.run(server.start())
            
        except ImportError as e:
            logger.error(f"Failed to import Genie server: {e}")
            logger.error("Make sure Genie is properly installed")
        except Exception as e:
            logger.error(f"Server error: {e}")
