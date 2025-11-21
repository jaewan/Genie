"""
Enhanced Model Manager: Client-side integration for model cache system.

Integrates Week 1 & Week 2 components:
- ModelFingerprint for stable identification
- ModelRegistry for tensor identity
- CacheQueryClient for efficient weight transfer
- New model cache protocol (register + execute)

This is part of the redesign plan (Week 3).
"""

import asyncio
import logging
import socket
import threading
from collections import defaultdict
from typing import Dict, Optional, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class ModelNotRegisteredError(RuntimeError):
    """Raised when executing a model that hasn't been registered."""


class _UsageTracker:
    """Simple usage tracker to back future cache policies."""

    def __init__(self):
        self._lock = threading.Lock()
        self._usage: Dict[str, int] = defaultdict(int)

    def record(self, fingerprint: str) -> None:
        with self._lock:
            self._usage[fingerprint] += 1

    def reset(self, fingerprint: str) -> None:
        with self._lock:
            self._usage[fingerprint] = 0

    def get(self, fingerprint: str) -> int:
        with self._lock:
            return self._usage.get(fingerprint, 0)


class EnhancedModelManager:
    """
    Client-side model manager for the new model cache system.
    
    Handles:
    - Model fingerprinting
    - Explicit model registration (opt-in caching)
    - Efficient execution requests (model ID + inputs only)
    
    **Design Philosophy**:
    - Explicit registration is required to use the fast path
    - No graph fallback exists in Phase 3 (fail fast with error)
    """
    
    def __init__(self, coordinator=None, server_address=None):
        """
        Initialize enhanced model manager.
        
        Args:
            coordinator: DjinnCoordinator instance (optional, will get global if None)
            server_address: Optional server address (host:port). If None, uses runtime state or default.
        """
        from .model_fingerprint import ModelFingerprint
        from .model_registry import get_model_registry
        from .cache_query import get_cache_query_client
        
        self.fingerprint = ModelFingerprint
        self.registry = get_model_registry()
        self.cache_client = get_cache_query_client()
        self._server_address = server_address  # Store explicit server address
        
        # Get coordinator
        if coordinator is None:
            from .coordinator import get_coordinator
            try:
                self.coordinator = get_coordinator()
            except RuntimeError:
                self.coordinator = None
                logger.warning("Coordinator not available, will use direct TCP")
        else:
            self.coordinator = coordinator
        
        # Track registered models
        self.registered_models: Dict[str, Dict] = {}  # fingerprint -> registration info
        
        # Connection pool for TCP requests (reuse connections)
        # Changed to support multiple concurrent connections per target
        self._connection_pool: Dict[str, List[tuple]] = {}  # target -> [(reader, writer, last_used, error_count), ...]
        self._connection_lock = asyncio.Lock()
        self._connection_timeout = 60.0  # Close idle connections after 60s
        self._max_connection_errors = 3  # Close connection after N errors
        self._max_connections_per_target = 20  # Allow up to 20 concurrent connections per target
        
        # Thread pool executor for CPU-bound serialization work
        # Use a shared executor to avoid creating too many threads
        self._serialization_executor = ThreadPoolExecutor(
            max_workers=min(20, (asyncio.get_event_loop().get_debug() and 4) or 8),
            thread_name_prefix="djinn-serialize"
        )
        
        # Usage tracking for analytics and future cache policies
        self.usage_tracker = _UsageTracker()

        # ✅ Local TTL cache for server addresses (Phase 3 optimization)
        # Cache key: (fingerprint, hints_hash) -> (server_address, expires_at)
        self._server_address_cache: Dict[tuple, tuple] = {}  # (fingerprint, hints_hash) -> (address, expires_at)
        self._cache_lock = asyncio.Lock()
    
    def _optimize_tcp_socket(self, sock) -> None:
        """
        Optimize TCP socket for high-throughput tensor transfer.
        
        Optimizations:
        1. TCP_NODELAY: Disable Nagle's algorithm (reduce latency)
        2. Large send/recv buffers: 16MB (better TCP window utilization)
        3. TCP window scaling: Enable via large buffers (better throughput)
        
        Args:
            sock: Socket object from writer.get_extra_info('socket')
        """
        import socket
        try:
            # 1. Disable Nagle's algorithm (reduce latency for large transfers)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # 2. Increase send buffer to 16MB (better TCP window utilization)
            # This allows TCP to send more data before waiting for ACK
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)
            
            # 3. Increase receive buffer to 16MB (better TCP window utilization)
            # This allows TCP to receive more data before sending ACK
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
            
            # 4. Enable TCP window scaling (Linux automatically enables if buffers are large)
            # Explicitly set window clamp for maximum throughput
            try:
                # TCP_WINDOW_CLAMP is Linux-specific, may not exist on all systems
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_WINDOW_CLAMP, 64 * 1024 * 1024)  # 64MB window
            except (AttributeError, OSError):
                # Not available on this system, skip (large buffers are enough)
                pass
            
            # 5. Get actual buffer sizes (may be adjusted by OS)
            actual_sndbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
            actual_rcvbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            
            logger.debug(
                f"✅ TCP optimized: NODELAY=1, SNDBUF={actual_sndbuf/(1024*1024):.1f}MB, "
                f"RCVBUF={actual_rcvbuf/(1024*1024):.1f}MB"
            )
        except Exception as e:
            logger.warning(f"⚠️  Failed to optimize TCP socket: {e} (continuing with defaults)")
        
        logger.info("EnhancedModelManager initialized")
    
    def __del__(self):
        """Cleanup thread pool executor on destruction."""
        if hasattr(self, '_serialization_executor'):
            try:
                self._serialization_executor.shutdown(wait=False)
            except:
                pass
    
    async def register_model(self, 
                            model: nn.Module,
                            model_id: Optional[str] = None) -> str:
        """
        Explicitly register model with server for caching (opt-in).
        
        Registered models use the fast cache path during execution.
        Unregistered models automatically use graph execution fallback.
        
        **Note**: This is an explicit opt-in operation. Models are NOT
        automatically registered during execution.
        
        Args:
            model: PyTorch model
            model_id: Optional explicit model identifier
        
        Returns:
            Model fingerprint
        
        Example:
            # Register production model (explicit opt-in)
            fingerprint = await manager.register_model(production_model)
            
            # Execute uses cache (fast)
            result = await manager.execute_model(production_model, inputs)
        """
        
        # Compute fingerprint
        fingerprint = self.fingerprint.compute(model, model_id)
        
        # Check if already registered locally
        if fingerprint in self.registered_models:
            logger.debug(f"Model {fingerprint[:8]} already registered locally")
            return fingerprint
            
        # Store model object for local execution
        # In production, server would load model by fingerprint
        self.registered_models[fingerprint] = {
            'model_id': model_id,
            'model': model,  # Store reference to model
            'registered_at': 0
        }
        
        # If we have a coordinator, register with remote server
        if self.coordinator:
            try:
                # Send model to server for caching
                # Server will load it into VMU and keep it ready
                await self.coordinator.register_remote_model(
                    fingerprint=fingerprint,
                    model=model,
                    model_id=model_id
                )
                logger.info(f"✅ Model registered on server: {fingerprint[:8]}...")
            except Exception as e:
                logger.warning(f"Server registration failed: {e}, model cached locally only")
        else:
            logger.info(f"✅ Model registered locally: {fingerprint[:8]}...")
        
        return fingerprint

    async def execute_model(self,
                           model: nn.Module,
                           inputs: Dict[str, Any],
                           model_id: Optional[str] = None,
                           profile_id: Optional[str] = None,
                           hints: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Execute model using new cache system (or fallback to graph).
        
        **Phase 3 Compromise**: Unregistered models fall back to graph execution
        with heavy warnings. Pre-trained models should be registered explicitly
        for optimal performance (9-300x faster).
        
        Args:
            model: PyTorch model
            inputs: Input tensors dict
            model_id: Optional explicit model identifier
            profile_id: Optional profile identifier (Phase 0 inert placeholder)
        
        Returns:
            Output tensor
        """
        
        fingerprint = self.fingerprint.compute(model, model_id)

        if fingerprint not in self.registered_models:
            logger.warning(
                f"⚠️  Model {fingerprint[:16]}... is not registered. "
                f"Auto-registering now (first execution will be slow)..."
            )
            # Auto-register on first use (shadow path)
            await self.register_model(model, model_id)
            logger.info(f"✅ Model {fingerprint[:8]} auto-registered")

        profile_id = None  # Phase 2 will populate this from hints/annotations
        return await self._execute_via_cache(
            fingerprint,
            inputs,
            hints=hints,
            profile_id=profile_id
        )
    
    async def _execute_via_graph(self, model, inputs):
        # Placeholder for graph execution fallback
        # In real implementation, this would use SubgraphExecutor
        logger.info("Executing via graph fallback (SLOW)...")
        # Simulate network delay
        await asyncio.sleep(0.5)
        
        # For local simulation, just run the model
        # This creates a valid tensor to return
        if isinstance(model, nn.Module):
            if 'input_ids' in inputs:
                return model(inputs['input_ids'])
            elif 'x' in inputs:
                return model(inputs['x'])
            else:
                # Try passing kwargs
                return model(**inputs)
        return torch.zeros(1)

    async def _execute_via_cache(self, fingerprint, inputs, hints=None, profile_id=None):
        # Execute model via server using model cache
        logger.info(f"Executing via cache for {fingerprint[:8]} (FAST)...")
        
        # Get model from registered models
        if fingerprint not in self.registered_models:
            raise RuntimeError(f"Model {fingerprint} not found in registered models")
        
        model_info = self.registered_models[fingerprint]
        
        # Try remote execution first if coordinator is available
        if self.coordinator:
            try:
                logger.info(f"🌐 Executing model {fingerprint[:8]} remotely...")
                result = await self.coordinator.execute_remote_model(
                    fingerprint=fingerprint,
                    inputs=inputs,
                    profile_id=profile_id
                )
                return result
            except Exception as e:
                logger.warning(f"Remote execution failed: {e}, falling back to local")
                # Fall through to local execution
        
        # No coordinator available - cannot execute remotely
        # Client should not execute locally as it doesn't have the model weights
        raise RuntimeError(
            f"Cannot execute model {fingerprint}: no coordinator available. "
            "Ensure Djinn server is running and client is connected."
        )


# Global instance
_global_manager: Optional[EnhancedModelManager] = None

def get_model_manager() -> EnhancedModelManager:
    """Get or create global model manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = EnhancedModelManager()
    return _global_manager
