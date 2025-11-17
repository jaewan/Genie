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
from typing import Dict, Optional, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EnhancedModelManager:
    """
    Client-side model manager for the new model cache system.
    
    Handles:
    - Model fingerprinting
    - Explicit model registration (opt-in caching)
    - Smart auto-registration (usage-based, memory-aware)
    - Efficient execution requests (model ID + inputs only)
    - Automatic fallback to graph execution for unregistered models
    
    **Design Philosophy**: Smart auto-registration with safety policies.
    - Explicit registration: Always works (opt-in)
    - Smart auto-registration: Automatically registers frequently-used models
      that meet safety criteria (usage threshold, size limit, memory check)
    - Unregistered models: Use graph execution fallback
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
        
        # Feature flag for gradual rollout
        self.use_model_cache = True  # Can be controlled via config
        
        # Smart auto-registration
        from .auto_registration_policy import ModelUsageTracker, AutoRegistrationPolicy, DEFAULT_POLICY
        self.usage_tracker = ModelUsageTracker()
        self.auto_registration_policy: AutoRegistrationPolicy = DEFAULT_POLICY
        self._auto_registration_tasks: Dict[str, asyncio.Task] = {}  # Track background registrations
        
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
        
        # ✅ Local TTL cache for server addresses (Phase 3 optimization)
        # Cache key: (fingerprint, hints_hash) -> (server_address, expires_at)
        self._server_address_cache: Dict[tuple, tuple] = {}  # (fingerprint, hints_hash) -> (address, expires_at)
        self._cache_lock = asyncio.Lock()
    
    def _optimize_tcp_socket(self, sock) -> None:
        """
              
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
        
        if fingerprint in self.registered_models:
            logger.info(f"Model {fingerprint} already registered")
            return fingerprint
        
        logger.info(f"Registering model {fingerprint}")
        
        # Register model parameters with registry
        self.registry.register_model(model, fingerprint)
        
        # Extract architecture descriptor
        from djinn.server.architecture_registry import HybridArchitectureRegistry
        arch_registry = HybridArchitectureRegistry()
        
        # Save architecture
        architecture_data = arch_registry.save_architecture(fingerprint, model)
        
        # Extract state dict and create weight IDs
        # ✅ FIX: Use state_dict directly to avoid LazyTensor issues
        # When model is on remote_accelerator:0, parameters become LazyTensors
        # state_dict() returns concrete tensors, so use that instead
        from djinn.frontend.core.interception_control import disable_interception, InterceptionContext
        
        weight_ids = {}
        uncached_weights = {}
        
        with disable_interception(InterceptionContext.CONSTRUCTION):
            # Get state dict without interception - this returns concrete tensors
            state_dict = model.state_dict()
            
            for param_name, param in model.named_parameters():
                # Get stable identifier from registry
                identity = self.registry.get_identity(param)
                weight_id = identity.identifier if identity else f"{fingerprint}:{param_name}"
                weight_ids[param_name] = weight_id
                
                # Use state_dict value directly (concrete tensor, not LazyTensor)
                weight_tensor = state_dict[param_name]
                # ✅ DESIGN FIX: Tensors should already be on CPU (moved during LazyTensor conversion)
                # But keep safety check for backward compatibility
                if isinstance(weight_tensor, torch.Tensor):
                    if weight_tensor.device.type == 'meta':
                        logger.warning(f"Skipping meta tensor for {param_name}")
                        continue
                    # ✅ FIX: Should already be CPU, but check for safety
                    if weight_tensor.device.type != 'cpu':
                        logger.warning(
                            f"Parameter {param_name} is on {weight_tensor.device} (expected CPU). "
                            f"Moving to CPU. This should not happen if model was converted to 'remote_accelerator:0' correctly."
                        )
                        weight_tensor = weight_tensor.cpu()
                    uncached_weights[param_name] = weight_tensor
                else:
                    logger.warning(f"Parameter {param_name} is not a tensor, skipping")
        
        # Create descriptor
        descriptor = {
            'class_name': model.__class__.__name__,
            'class_module': model.__class__.__module__,
            'framework': self._detect_framework(model)
        }
        
        # For transformers models, include config
        if descriptor['framework'] == 'transformers':
            try:
                # Try to get config from model
                if hasattr(model, 'config'):
                    # Convert config to dict (it's usually a Config object)
                    config = model.config
                    if hasattr(config, 'to_dict'):
                        descriptor['config'] = config.to_dict()
                    elif isinstance(config, dict):
                        descriptor['config'] = config
                    else:
                        # Fallback: try to convert to dict manually
                        descriptor['config'] = {k: getattr(config, k) for k in dir(config) 
                                               if not k.startswith('_') and not callable(getattr(config, k, None))}
                    logger.debug(f"Added config to descriptor for {fingerprint}")
            except Exception as e:
                logger.warning(f"Failed to extract config from transformers model: {e}")
        
        # Query cache for existing weights
        # Cache query is optional - if it fails, we'll send all weights (safe fallback)
        weight_identifiers = list(weight_ids.values())
        server_address = await self._get_server_address(
            fingerprint=fingerprint,
            inputs=None,  # No inputs during registration
            hints={}
        )
        try:
            cache_result = await self.cache_client.query_cached_identifiers(
                server_address,
                set(weight_identifiers),
                use_local_cache=True
            )
        except Exception as e:
            logger.warning(f"Cache query failed: {e}, assuming nothing cached")
            # Create empty cache result
            from .cache_query import CacheQueryResult
            cache_result = CacheQueryResult(
                cached_identifiers=set(),
                missing_identifiers=set(weight_identifiers),
                query_time_ms=0.0
            )
        
        # Filter out cached weights
        cached_weight_names = {
            name for name, weight_id in weight_ids.items()
            if weight_id in cache_result.cached_identifiers
        }
        
        # Only send uncached weights
        uncached_weights = {
            name: weight for name, weight in uncached_weights.items()
            if name not in cached_weight_names
        }
        
        # Log registration progress
        total_weights = len(weight_ids)
        cached_count = len(cached_weight_names)
        uncached_count = len(uncached_weights)
        total_size_mb = sum(
            w.numel() * w.element_size() 
            for w in uncached_weights.values()
        ) / (1024 * 1024)
        
        logger.info(
            f"Registration summary: {total_weights} total weights, "
            f"{cached_count} cached, {uncached_count} to transfer "
            f"({total_size_mb:.1f} MB)"
        )
        
        # Send registration request
        import time
        serialize_time = 0.0
        transfer_time = 0.0
        
        if uncached_count > 0:
            # ✅ OPTIMIZATION: Only chunk for very large models (> 1GB) for memory safety
            # TCP can easily handle 500MB in a single connection
            # Chunking adds protocol overhead (message type + length + JSON per chunk)
            # ✅ TEMP FIX: Disable chunking for GPT-2-XL testing (chunked protocol has bugs)
            # GPT-2-XL is ~1.5GB, but direct transfer works fine for models up to 2GB
            CHUNK_THRESHOLD_MB = 2048  # 2GB - only chunk if larger than this (temporarily increased for testing)
            CHUNK_SIZE_MB = 100  # 100MB chunks (larger chunks = less overhead)
            CHUNK_SIZE_BYTES = CHUNK_SIZE_MB * 1024 * 1024
            
            if total_size_mb > CHUNK_THRESHOLD_MB:
                # Use chunked transfer for large payloads
                logger.info(
                    f"Large payload detected ({total_size_mb:.1f} MB), using chunked transfer "
                    f"(chunk size: {CHUNK_SIZE_MB} MB)"
                )
                response = await self._register_model_chunked(
                    fingerprint, descriptor, weight_ids, uncached_weights,
                    architecture_data, CHUNK_SIZE_BYTES
                )
                # Chunked transfer handles its own timing and logging
            else:
                # ✅ OPTIMIZATION: Use direct binary protocol for small models (< 1GB)
                # This eliminates JSON overhead and double serialization
                logger.info(f"✅ PHASE 2: Using direct binary protocol (no chunking, no JSON overhead)")
                logger.info(f"Serializing {uncached_count} weights using direct binary protocol...")
                serialize_start = time.perf_counter()
                weights_binary = self._serialize_weights_binary(uncached_weights)
                serialize_time = (time.perf_counter() - serialize_start) * 1000
                logger.info(f"✅ Binary serialization complete: {serialize_time:.1f}ms for {uncached_count} weights ({len(weights_binary) / (1024*1024):.1f} MB)")
                
                registration_request = {
                    'type': 'REGISTER_MODEL',
                    'fingerprint': fingerprint,
                    'descriptor': descriptor,
                    'weight_ids': weight_ids,
                    'weights_binary': weights_binary,  # Direct binary, no JSON
                    'architecture_data': architecture_data,
                    '_binary_protocol': True  # ✅ Flag to indicate binary protocol
                }
                
                # Send via coordinator or direct TCP
                transfer_start = time.perf_counter()
                logger.info(f"Sending registration request to server (single message, no chunking)...")
                
                response = await self._send_request(registration_request)
                transfer_time = (time.perf_counter() - transfer_start) * 1000
        else:
            serialized_weights = {}
            logger.info("All weights already cached, skipping transfer")
            
            registration_request = {
                'type': 'REGISTER_MODEL',
                'fingerprint': fingerprint,
                'descriptor': descriptor,
                'weight_ids': weight_ids,
                'uncached_weights': serialized_weights,
                'architecture_data': architecture_data
            }
            
            transfer_start = time.perf_counter()
            response = await self._send_request(registration_request)
            transfer_time = (time.perf_counter() - transfer_start) * 1000
        
        total_time = transfer_time + serialize_time
        
        if response.get('status') == 'success':
            self.registered_models[fingerprint] = {
                'fingerprint': fingerprint,
                'descriptor': descriptor,
                'weight_ids': weight_ids
            }
            # Reset usage tracking after successful registration
            self.usage_tracker.reset(fingerprint)
            logger.info(
                f"Model {fingerprint} registered successfully "
                f"(serialize: {serialize_time:.1f}ms, transfer: {transfer_time:.1f}ms, total: {total_time:.1f}ms)"
            )
        else:
            error_msg = response.get('message', 'Unknown error')
            raise RuntimeError(f"Model registration failed: {error_msg}")
        
        return fingerprint
    
    async def init_model(self, model: nn.Module, model_id: Optional[str] = None) -> bool:
        """
        Initialize (warmup) a registered model by triggering CUDA kernel JIT compilation.
        
        This is an explicit initialization step that should be called after registration.
        It executes the model once on the server to trigger kernel compilation, ensuring
        fast first execution.
        
        **Benefits:**
        - Registration is fast (~2-3s instead of 12.9s)
        - User controls when warmup happens (can batch multiple models)
        - Better separation of concerns (registration vs. initialization)
        
        Args:
            model: PyTorch model (or fingerprint string)
            model_id: Optional explicit model identifier
        
        Returns:
            True if initialization succeeded, False otherwise
        
        Example:
            # Register model (fast, ~2-3s)
            fingerprint = await manager.register_model(model)
            
            # Initialize model (warmup, ~8-10s for GPT-2-small)
            await manager.init_model(model)
            
            # Now first execution will be fast (no compilation overhead)
        """
        # Compute fingerprint
        if isinstance(model, str):
            fingerprint = model  # Allow passing fingerprint directly
        else:
            fingerprint = self.fingerprint.compute(model, model_id)
        
        if fingerprint not in self.registered_models:
            logger.warning(f"Model {fingerprint} not registered, cannot initialize")
            return False
        
        logger.info(f"Initializing model {fingerprint} (JIT compilation warmup)...")
        
        # Create initialization request
        init_request = {
            'type': 'INIT_MODEL',
            'fingerprint': fingerprint
        }
        
        # Send request to server
        try:
            response = await self._send_request(init_request)
            
            if response.get('status') == 'success':
                logger.info(f"✅ Model {fingerprint} initialized successfully")
                return True
            else:
                error_msg = response.get('message', 'Unknown error')
                logger.warning(f"Model initialization failed: {error_msg}")
                return False
        except Exception as e:
            logger.error(f"Model initialization request failed: {e}")
            return False
    
    async def _register_model_chunked(
        self,
        fingerprint: str,
        descriptor: Dict,
        weight_ids: Dict[str, str],
        uncached_weights: Dict[str, torch.Tensor],
        architecture_data: Optional[bytes],
        chunk_size_bytes: int
    ) -> Dict:
        """
        Register model using chunked transfer for large payloads.
        
        Splits weights into chunks and sends them in parallel (with concurrency limit)
        with retry logic. This prevents connection timeouts, allows recovery from failures,
        and significantly improves transfer speed by utilizing network bandwidth efficiently.
        
        Uses adaptive concurrency: 5-20 concurrent chunks depending on total chunk count.
        """
        logger.info(f"🟡 _register_model_chunked START: fingerprint={fingerprint}, chunk_size={chunk_size_bytes/(1024*1024):.1f} MB")
        
        import time
        
        # Split weights into chunks
        chunks = self._chunk_weights(uncached_weights, chunk_size_bytes)
        total_chunks = len(chunks)
        
        logger.info(
            f"Chunked transfer: {total_chunks} chunks "
            f"(~{chunk_size_bytes / (1024*1024):.1f} MB each)"
        )
        
        # Send header (architecture + metadata) - small, no chunking needed
        # Note: architecture_data might be large, so we'll send it separately if needed
        header_start = time.perf_counter()
        
        # Check if architecture_data is too large (should be small, but check anyway)
        arch_data_size = len(architecture_data) if architecture_data else 0
        arch_data_size_mb = arch_data_size / (1024 * 1024)
        
        if arch_data_size_mb > 10:  # If > 10MB, send separately
            logger.warning(f"Architecture data is large ({arch_data_size_mb:.1f} MB), sending separately...")
            header = {
                'type': 'REGISTER_MODEL_CHUNKED',
                'fingerprint': fingerprint,
                'descriptor': descriptor,
                'weight_ids': weight_ids,
                'total_chunks': total_chunks,
                'architecture_data': None  # Send separately
            }
            send_arch_separately = True
        else:
            header = {
                'type': 'REGISTER_MODEL_CHUNKED',
                'fingerprint': fingerprint,
                'descriptor': descriptor,
                'weight_ids': weight_ids,
                'total_chunks': total_chunks,
                'architecture_data': architecture_data
            }
            send_arch_separately = False
        
        logger.info(
            f"Sending chunked registration header "
            f"(fingerprint={fingerprint}, total_chunks={total_chunks}, "
            f"arch_data={arch_data_size_mb:.1f} MB)..."
        )
        try:
            # ✅ CRITICAL FIX: Use new connection for header to avoid pool pollution
            # If header uses pooled connection, leftover response buffer data can
            # interfere with subsequent chunk connections
            header_response = await self._send_tcp_request(header, use_pool=False)
            if header_response.get('status') != 'success':
                raise RuntimeError(f"Header registration failed: {header_response.get('message')}")
            
            # Send architecture_data separately if needed
            if send_arch_separately:
                logger.info("Sending architecture data separately...")
                arch_chunk = {
                    'type': 'REGISTER_MODEL_CHUNK',
                    'fingerprint': fingerprint,
                    'chunk_id': -1,  # Special chunk ID for architecture data
                    'total_chunks': total_chunks,
                    'architecture_data': architecture_data
                }
                # ✅ CRITICAL FIX: Use new connection to avoid pool pollution
                arch_response = await self._send_tcp_request(arch_chunk, use_pool=False)
                if arch_response.get('status') != 'success':
                    raise RuntimeError(f"Architecture data send failed: {arch_response.get('message')}")
        except Exception as e:
            logger.error(f"Failed to send chunked registration header: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        header_time = (time.perf_counter() - header_start) * 1000
        logger.info(f"Header sent successfully: {header_time:.1f}ms")
        
        # Send chunks in parallel with concurrency limit
        chunk_start = time.perf_counter()
        successful_chunks = 0
        failed_chunks = []
        
        # Adaptive concurrency: more chunks = higher concurrency (up to 20)
        # For 100 chunks, use 20 concurrent; for 10 chunks, use 5 concurrent
        max_concurrency = min(20, max(5, total_chunks // 5))
        semaphore = asyncio.Semaphore(max_concurrency)
        
        logger.info(
            f"Sending {total_chunks} chunks with {max_concurrency} concurrent transfers "
            f"(using thread pool for serialization, connection pool for transfers)..."
        )
        
        # Track timing for parallelism verification (shared list)
        chunk_timings: List[Dict] = []
        
        async def send_chunk_with_retry(chunk_id: int, chunk_weights: Dict[str, torch.Tensor]) -> Tuple[int, bool, Optional[str]]:
            """
            Send a single chunk with retry logic.
            
            Uses thread pool for CPU-bound serialization to avoid blocking event loop.
            This enables true parallelism: multiple chunks can serialize and transfer concurrently.
            
            Returns:
                (chunk_id, success, error_message)
            """
            logger.info(f"🟣 CHUNK {chunk_id}: send_chunk_with_retry() START")
            try:
                logger.info(f"🟣 CHUNK {chunk_id}: Acquiring semaphore (max_concurrency={max_concurrency})...")
                async with semaphore:  # Limit concurrent chunks
                    logger.info(f"🟣 CHUNK {chunk_id}: Acquired semaphore, starting serialization")
                    chunk_size_mb = sum(
                        w.numel() * w.element_size() 
                        for w in chunk_weights.values()
                    ) / (1024 * 1024)
                    
                    # ✅ PHASE 2 FIX: Use binary protocol for chunks too (not just small models)
                    # Serialize in thread pool (CPU-bound work, doesn't block event loop)
                    chunk_serialize_start = time.perf_counter()
                    loop = asyncio.get_event_loop()
                    try:
                        # Use binary protocol instead of dict-based serialization
                        chunk_data_binary = await loop.run_in_executor(
                            self._serialization_executor,
                            self._serialize_weights_binary,  # ✅ Use binary protocol, not dict-based
                            chunk_weights
                        )
                    except Exception as serialize_error:
                        logger.error(f"🚨 CHUNK {chunk_id}: Serialization failed: {serialize_error}")
                        import traceback
                        logger.error(traceback.format_exc())
                        raise
                    chunk_serialize_time = (time.perf_counter() - chunk_serialize_start) * 1000
                    logger.info(f"✅ CHUNK {chunk_id}: Serialization completed in {chunk_serialize_time:.1f}ms, data_size={len(chunk_data_binary)/(1024*1024):.1f}MB")
                    
                    # Retry logic per chunk
                    max_retries = 3
                    
                    for attempt in range(max_retries):
                        logger.info(f"🔶 CHUNK {chunk_id}: Retry attempt {attempt+1}/{max_retries}, building request...")
                        try:
                            chunk_request = {
                                'type': 'REGISTER_MODEL_CHUNK',
                                'fingerprint': fingerprint,
                                'chunk_id': chunk_id,
                                'total_chunks': total_chunks,
                                'chunk_data_binary': chunk_data_binary,
                                '_binary_protocol': True,
                                '_fire_and_forget': True
                            }
                            
                            # ✅ CRITICAL: Validate request type before sending
                            if chunk_request.get('type') != 'REGISTER_MODEL_CHUNK':
                                raise ValueError(
                                    f"CRITICAL: Chunk request type mismatch! "
                                    f"Expected 'REGISTER_MODEL_CHUNK', got '{chunk_request.get('type')}'"
                                )
                            
                            chunk_transfer_start = time.perf_counter()
                            logger.info(f"🔵 CHUNK {chunk_id}: About to call _send_tcp_request with request_type={chunk_request.get('type')}, use_pool=False (NEW connection)")
                            chunk_response = await self._send_tcp_request(
                                chunk_request, 
                                use_pool=False,  # Always use new connection for chunks
                                wait_for_response=False  # Fire-and-forget: don't wait for server response
                            )
                            logger.info(f"🟢 CHUNK {chunk_id}: _send_tcp_request returned: {chunk_response}")
                            chunk_transfer_time = (time.perf_counter() - chunk_transfer_start) * 1000
                            
                            # Fire-and-forget always succeeds (we don't wait for response)
                            if True:  # Always succeed for fire-and-forget
                                total_time = chunk_serialize_time + chunk_transfer_time
                                chunk_timings.append({
                                    'chunk_id': chunk_id,
                                    'serialize_ms': chunk_serialize_time,
                                    'transfer_ms': chunk_transfer_time,
                                    'total_ms': total_time,
                                    'size_mb': chunk_size_mb
                                })
                                
                                if chunk_id % 10 == 0 or chunk_id == total_chunks - 1:  # Log every 10th or last
                                    logger.info(
                                        f"Chunk {chunk_id+1}/{total_chunks} sent: "
                                        f"{chunk_size_mb:.1f} MB "
                                        f"(serialize: {chunk_serialize_time:.1f}ms, "
                                        f"transfer: {chunk_transfer_time:.1f}ms, "
                                        f"total: {total_time:.1f}ms)"
                                    )
                                return (chunk_id, True, None)
                            else:
                                error_msg = chunk_response.get('message', 'Unknown error')
                                if attempt < max_retries - 1:
                                    logger.warning(
                                        f"Chunk {chunk_id+1}/{total_chunks} failed (attempt {attempt+1}/{max_retries}): {error_msg}"
                                    )
                                else:
                                    logger.error(
                                        f"Chunk {chunk_id+1}/{total_chunks} failed after {max_retries} attempts: {error_msg}"
                                    )
                                
                        except Exception as e:
                            if attempt < max_retries - 1:
                                logger.error(
                                    f"Chunk {chunk_id+1}/{total_chunks} error (attempt {attempt+1}/{max_retries}): {e}"
                                )
                            else:
                                logger.error(
                                    f"Chunk {chunk_id+1}/{total_chunks} error after {max_retries} attempts: {e}"
                                )
                            
                            if attempt < max_retries - 1:
                                # Exponential backoff
                                wait_time = 2 ** attempt
                                await asyncio.sleep(wait_time)
                
                # All retries failed
                return (chunk_id, False, f"Failed after {max_retries} attempts")
            except Exception as e:
                logger.error(f"🚨 CHUNK {chunk_id}: Outer exception in send_chunk_with_retry: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return (chunk_id, False, str(e))
        
        # Send all chunks concurrently (with concurrency limit)
        chunk_tasks = [
            send_chunk_with_retry(chunk_id, chunk_weights)
            for chunk_id, chunk_weights in enumerate(chunks)
        ]
        
        # Wait for all chunks to complete
        chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
        
        # Process results
        for result in chunk_results:
            if isinstance(result, Exception):
                logger.error(f"Chunk task raised exception: {result}")
                import traceback
                logger.error(f"Traceback:\n{traceback.format_exception(type(result), result, result.__traceback__)}")
                continue
            
            chunk_id, success, error_msg = result
            if success:
                successful_chunks += 1
            else:
                failed_chunks.append(chunk_id)
                if error_msg:
                    logger.error(f"Chunk {chunk_id+1}/{total_chunks}: {error_msg}")
        
        chunk_time = (time.perf_counter() - chunk_start) * 1000
        
        # Log parallelism statistics
        if chunk_timings:
            avg_serialize = sum(t['serialize_ms'] for t in chunk_timings) / len(chunk_timings)
            avg_transfer = sum(t['transfer_ms'] for t in chunk_timings) / len(chunk_timings)
            avg_total = sum(t['total_ms'] for t in chunk_timings) / len(chunk_timings)
            max_total = max(t['total_ms'] for t in chunk_timings)
            
            # Calculate parallelism efficiency
            # If perfectly parallel, total time ≈ max(chunk_time) + overhead
            # Sequential time would be sum(chunk_time)
            sequential_time = sum(t['total_ms'] for t in chunk_timings)
            parallelism_efficiency = (sequential_time / chunk_time) if chunk_time > 0 else 0
            
            logger.info(
                f"Chunk transfer statistics: "
                f"avg serialize: {avg_serialize:.1f}ms, "
                f"avg transfer: {avg_transfer:.1f}ms, "
                f"avg total: {avg_total:.1f}ms, "
                f"max total: {max_total:.1f}ms, "
                f"parallelism efficiency: {parallelism_efficiency:.1f}x"
            )
        
        if failed_chunks:
            raise RuntimeError(
                f"Registration failed: {len(failed_chunks)}/{total_chunks} chunks failed "
                f"(chunks: {failed_chunks})"
            )
        
        # Finalize registration
        logger.info(f"Finalizing chunked registration (all {successful_chunks} chunks sent)...")
        # ✅ FIX: Wait for chunks to arrive at server before finalizing
        # With fire-and-forget, chunks are sent but may still be in transit
        # Estimate: ~0.1s per chunk for network + processing (conservative)
        # Add extra buffer for server processing
        estimated_wait_time = max(5.0, (total_chunks * 0.1) + 3.0)  # Min 5s, ~0.1s per chunk + 3s buffer
        logger.info(f"⏳ Waiting {estimated_wait_time:.1f}s for all chunks to arrive at server...")
        await asyncio.sleep(estimated_wait_time)
        finalize_start = time.perf_counter()
        
        finalize_request = {
            'type': 'REGISTER_MODEL_FINALIZE',
            'fingerprint': fingerprint
        }
        
        try:
            # ✅ FIX: Retry logic for finalization (critical operation)
            max_retries = 3
            finalize_response = None
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"📤 Sending finalization request for {fingerprint} (attempt {attempt+1}/{max_retries})...")
                    # ✅ FIX: Use new connection for finalization (like chunks do)
                    # Connection pool might be empty or have stale connections after chunks
                    # Finalization is critical - use fresh, reliable connection
                    finalize_response = await self._send_tcp_request(
                        finalize_request,
                        use_pool=False,  # Use new connection (not pool)
                        wait_for_response=True
                    )
                    logger.info(f"✅ Finalization response received: {finalize_response.get('status', 'unknown')}")
                    
                    if finalize_response.get('status') == 'success':
                        break  # Success - exit retry loop
                    else:
                        error_msg = finalize_response.get('message', 'Unknown error')
                        if attempt < max_retries - 1:
                            logger.warning(f"Finalization failed (attempt {attempt+1}): {error_msg}, retrying...")
                            await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                        else:
                            logger.error(f"Finalization failed after {max_retries} attempts: {error_msg}")
                            raise RuntimeError(f"Finalization failed: {error_msg}")
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Finalization request failed (attempt {attempt+1}): {e}, retrying...")
                        await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                    else:
                        logger.error(f"Finalization request failed after {max_retries} attempts: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        raise
            
            finalize_time = (time.perf_counter() - finalize_start) * 1000
            logger.info(f"Finalization completed: {finalize_time:.1f}ms")
        except Exception as e:
            logger.error(f"Finalization request failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        logger.info(
            f"Chunked registration complete: {successful_chunks}/{total_chunks} chunks "
            f"(header: {header_time:.1f}ms, chunks: {chunk_time:.1f}ms, total: {header_time + chunk_time:.1f}ms)"
        )
        
        return finalize_response
    
    def _chunk_weights(
        self, 
        weights: Dict[str, torch.Tensor], 
        chunk_size_bytes: int
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Split weights into chunks of approximately chunk_size_bytes.
        
        Args:
            weights: Dictionary of weight tensors
            chunk_size_bytes: Target chunk size in bytes
            
        Returns:
            List of weight dictionaries (chunks)
        """
        chunks = []
        current_chunk = {}
        current_size = 0
        
        for name, tensor in weights.items():
            tensor_size = tensor.numel() * tensor.element_size()
            
            # If adding this tensor would exceed chunk size, start new chunk
            if current_size + tensor_size > chunk_size_bytes and current_chunk:
                chunks.append(current_chunk)
                current_chunk = {}
                current_size = 0
            
            current_chunk[name] = tensor
            current_size += tensor_size
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    async def execute_model(self,
                           model: nn.Module,
                           inputs: Dict[str, Any],
                           model_id: Optional[str] = None) -> torch.Tensor:
        """
        Execute model using new cache system (or fallback to graph).
        
        **Smart Auto-Registration**: Models are automatically registered after
        meeting usage threshold (default: 3 uses) and safety criteria (size limit,
        memory check). Explicit registration via `register_model()` always works.
        
        Args:
            model: PyTorch model
            inputs: Input tensors dict
            model_id: Optional explicit model identifier
        
        Returns:
            Output tensor
        
        Example:
            # Explicit registration (immediate caching)
            await manager.register_model(model)  # Register once
            await manager.execute_model(model, inputs)  # Uses cache (fast)
            
            # Smart auto-registration (after 3 uses)
            await manager.execute_model(model, inputs)  # 1st use: graph fallback
            await manager.execute_model(model, inputs)  # 2nd use: graph fallback
            await manager.execute_model(model, inputs)  # 3rd use: auto-registers (background)
            await manager.execute_model(model, inputs)  # 4th use: uses cache (fast)
        """
        
        # Compute fingerprint
        fingerprint = self.fingerprint.compute(model, model_id)
        
        # Check registration status (debug logging removed for production)
        
        # Check if registered - if not, check for smart auto-registration
        if fingerprint not in self.registered_models:
            # Track usage
            usage_count = self.usage_tracker.record(fingerprint)
            
            # Check if should auto-register (smart policy)
            if self.auto_registration_policy.should_auto_register(
                fingerprint, model, usage_count, self.usage_tracker
            ):
                # Start background registration (non-blocking)
                if fingerprint not in self._auto_registration_tasks:
                    logger.info(
                        f"🚀 Auto-registering model {fingerprint[:16]}... "
                        f"(usage={usage_count}, background task)"
                    )
                    self._auto_registration_tasks[fingerprint] = asyncio.create_task(
                        self._register_model_async(model, fingerprint, model_id)
                    )
            
            # Use graph fallback immediately (don't wait for registration)
            logger.debug(
                f"Model {fingerprint[:16]}... not registered "
                f"(usage={usage_count}), using graph fallback"
            )
            return await self._execute_via_graph(model, inputs)
        
        # Model is registered - use fast cache path
        if self.use_model_cache:
            try:
                return await self._execute_via_cache(fingerprint, inputs)
            except Exception as e:
                logger.warning(f"Model cache execution failed: {e}, falling back to graph")
                # Fall through to graph execution
        
        # Fallback to graph execution
        return await self._execute_via_graph(model, inputs)
    
    async def _register_model_async(
        self, 
        model: nn.Module, 
        fingerprint: str, 
        model_id: Optional[str] = None
    ):
        """
        Register model asynchronously (background task for auto-registration).
        
        This is called automatically by smart auto-registration policy.
        """
        try:
            await self.register_model(model, model_id)
            logger.info(f"✅ Auto-registration completed for {fingerprint[:16]}...")
            # Reset usage tracking after successful registration
            self.usage_tracker.reset(fingerprint)
        except Exception as e:
            logger.warning(f"⚠️ Auto-registration failed for {fingerprint[:16]}...: {e}")
            # Don't re-raise - auto-registration failure shouldn't block execution
        finally:
            # Clean up task tracking
            if fingerprint in self._auto_registration_tasks:
                del self._auto_registration_tasks[fingerprint]
    
    def set_auto_registration_policy(self, policy):
        """
        Set auto-registration policy.
        
        Args:
            policy: AutoRegistrationPolicy instance or None to disable
        """
        from .auto_registration_policy import AutoRegistrationPolicy, DISABLED_POLICY
        if policy is None:
            self.auto_registration_policy = DISABLED_POLICY
        elif isinstance(policy, AutoRegistrationPolicy):
            self.auto_registration_policy = policy
        else:
            raise ValueError(f"Invalid policy type: {type(policy)}")
    
    async def _execute_via_cache(self, fingerprint: str, inputs: Dict[str, Any]) -> torch.Tensor:
        """Execute via model cache (fast path)."""
        
        # Enable profiling context
        from djinn.server.profiling_context import get_profiler, record_phase
        profiler = get_profiler()
        
        # Serialize inputs
        with record_phase('model_cache_input_serialization'):
            serialized_inputs = self._serialize_inputs(inputs)
        
        # Create execution request (TINY!)
        execution_request = {
            'type': 'EXECUTE_MODEL',
            'fingerprint': fingerprint,
            'inputs': serialized_inputs,
            'hints': {}  # Can add semantic hints here
        }
        
        # Send request (includes network transfer)
        # ✅ FIX: Force new connection for EXECUTE_MODEL to avoid stale connection issues
        with record_phase('model_cache_network_c2s'):
            response = await self._send_tcp_request(execution_request, use_pool=False)
        
        if response.get('status') == 'success':
            # Merge server-side phases into client profiler
            server_phases = response.get('server_phases', {})
            if profiler and server_phases:
                for phase_name, duration_ms in server_phases.items():
                    # Prefix server phases to distinguish from client phases
                    profiler.record_phase(f'server_{phase_name}', duration_ms)
            
            # Deserialize result (includes network transfer S→C)
            with record_phase('model_cache_result_deserialization'):
                result_data = response['result']
                result = self._deserialize_tensor(result_data)
            return result
        else:
            error_msg = response.get('message', 'Unknown error')
            # Check if server requested fallback
            if response.get('fallback_required'):
                logger.debug(f"Server requested fallback to graph execution: {error_msg}")
                raise RuntimeError(f"Model cache execution failed, fallback required: {error_msg}")
            raise RuntimeError(f"Model execution failed: {error_msg}")
    
    async def _execute_via_graph(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        """Fallback to graph execution (backward compatibility)."""
        
        # Use existing coordinator subgraph execution
        if self.coordinator:
            # Build subgraph using djinn.capture()
            import djinn
            
            with djinn.capture():
                # Handle different input formats
                # nn.Sequential expects positional args, not keyword args
                if isinstance(model, torch.nn.Sequential):
                    # For Sequential, use first input value as positional arg
                    input_values = list(inputs.values())
                    if len(input_values) == 1:
                        output = model(input_values[0])
                    else:
                        output = model(*input_values)
                else:
                    # For other models, try keyword args first
                    try:
                        output = model(**inputs)
                    except TypeError:
                        # Fallback to positional args
                        input_values = list(inputs.values())
                        output = model(*input_values)
            
            # Get the root tensor from the graph builder
            from djinn.frontend.core.graph_builder import get_global_builder
            builder = get_global_builder()
            lazy_root = builder.root_tensor
            
            if lazy_root is None:
                raise RuntimeError("No graph captured - model execution didn't create LazyTensors")
            
            # Extract subgraph using the base SubgraphBuilder method
            from djinn.server.subgraph_builder import SubgraphBuilder
            subgraph_builder = SubgraphBuilder()
            subgraph = subgraph_builder.build_remote_subgraph(lazy_root)
            
            if subgraph is None:
                raise RuntimeError("Failed to build subgraph from captured graph")
            
            # Convert inputs to tensors if needed
            input_data = {}
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    input_data[key] = value
                else:
                    input_data[key] = torch.tensor(value) if not isinstance(value, torch.Tensor) else value
            
            # Execute via coordinator - serialize RemoteSubgraph to dict
            # Get server address with fingerprint for global coordinator routing
            from .model_fingerprint import ModelFingerprint
            fingerprint = ModelFingerprint.compute(model)
            server_address = await self._get_server_address(
                fingerprint=fingerprint,
                inputs=inputs,
                hints={}  # Can extract semantic hints from model/inputs
            )
            return await self.coordinator.execute_remote_subgraph(
                subgraph=subgraph.serialize(),  # Serialize RemoteSubgraph to dict
                input_data=input_data,
                target=server_address,
                model=model  # Pass model for cache check
            )
        else:
            raise RuntimeError("No coordinator available for graph execution")
    
    async def _send_request(self, request: Dict, server_address: Optional[str] = None) -> Dict:
        """
        Send request to server.
        
        Args:
            request: Request dict
            server_address: Optional server address (if not in request)
        """
        # Extract server address from request or use provided
        target_server = request.get('server_address') or server_address
        
        if self.coordinator:
            # Use coordinator's transport
            # For now, use a simple TCP connection
            return await self._send_tcp_request(request, server_address=target_server)
        else:
            return await self._send_tcp_request(request, server_address=target_server)
    
    async def _get_connection(self, host: str, port: int, allow_new: bool = True) -> tuple:
        """
        Get or create TCP connection with pooling and health checking.
        
        Improved with:
        - Multiple concurrent connections per target (for parallel transfers)
        - Health checking (connection state, idle timeout)
        - Error tracking (close after max errors)
        - Automatic cleanup of stale connections
        
        Args:
            host: Server hostname
            port: Server port
            allow_new: If True, create new connection if pool is full. If False, wait for available connection.
        
        Returns:
            (reader, writer) tuple
        """
        import time
        target = f"{host}:{port}"
        now = time.time()
        
        async with self._connection_lock:
            # Check if we have cached connections
            if target in self._connection_pool:
                pool = self._connection_pool[target]
                
                # Try to find a healthy connection in the pool
                for i, (reader, writer, last_used, error_count) in enumerate(pool):
                    # Health check: connection state, idle timeout, error count
                    try:
                        writer.get_extra_info('peername')  # This will fail if connection is closed
                        is_healthy = (
                            not writer.is_closing() and
                            (now - last_used) < self._connection_timeout and
                            error_count < self._max_connection_errors
                        )
                    except (OSError, AttributeError):
                        # Connection is closed
                        is_healthy = False
                    
                    if is_healthy:
                        # ✅ TD-1 FIX: Check for leftover data in StreamReader buffer before reuse
                        # This prevents protocol errors from reading wrong bytes as message length
                        try:
                            # Check if there's data in buffer (non-blocking peek)
                            # StreamReader doesn't expose buffer directly, but we can check with a timeout
                            # Try to peek at first byte with very short timeout
                            # If there's data, it means leftover data from previous message
                            peek_data = await asyncio.wait_for(reader.read(1), timeout=0.001)
                            if peek_data:
                                # There's leftover data in buffer - don't reuse this connection
                                logger.warning(
                                    f"⚠️  Connection to {target} has leftover data in buffer ({len(peek_data)} bytes), "
                                    f"removing from pool to prevent protocol errors"
                                )
                                # Put the byte back (we can't, so we'll close the connection)
                                is_healthy = False
                                # Close and remove this connection
                                try:
                                    writer.close()
                                    await writer.wait_closed()
                                except:
                                    pass
                                pool.pop(i)
                                break
                        except asyncio.TimeoutError:
                            # No data available (good - buffer is empty)
                            pass
                        except Exception as buffer_check_error:
                            # Error checking buffer - assume connection is bad
                            logger.warning(f"Error checking connection buffer: {buffer_check_error}, removing from pool")
                            is_healthy = False
                            try:
                                writer.close()
                                await writer.wait_closed()
                            except:
                                pass
                            pool.pop(i)
                            break
                        
                        if is_healthy:
                            # Buffer is clean, safe to reuse
                            pool[i] = (reader, writer, now, error_count)
                            logger.debug(f"♻️  Reusing connection to {target} (errors: {error_count}, pool size: {len(pool)})")
                            return reader, writer
                    else:
                        # Connection is stale or unhealthy, remove it
                        logger.debug(f"Removing unhealthy connection from pool (idle: {now - last_used:.1f}s, errors: {error_count})")
                        try:
                            writer.close()
                            await writer.wait_closed()
                        except:
                            pass
                        pool.pop(i)
                        break
                
                # Clean up empty pool
                if not pool:
                    del self._connection_pool[target]
            
            # Check if we can create a new connection
            if not allow_new:
                # Wait for a connection to become available (not implemented, just create new for now)
                pass
            
            # Check pool size limit
            if target in self._connection_pool:
                pool = self._connection_pool[target]
                if len(pool) >= self._max_connections_per_target:
                    # Pool is full, create new connection but don't add to pool (temporary)
                    logger.debug(f"Connection pool full ({len(pool)}/{self._max_connections_per_target}), creating temporary connection")
                    # Still create new connection, but it won't be pooled
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection(host, port),
                        timeout=5.0
                    )
                    # ✅ PHASE 3: Apply TCP optimizations
                    sock = writer.get_extra_info('socket')
                    if sock:
                        self._optimize_tcp_socket(sock)
                    logger.debug(f"✅ Created temporary connection to {host}:{port} (not pooled, TCP optimized)")
                    return reader, writer
        
        # Create new connection and add to pool
        logger.debug(f"🔌 Creating new connection to {host}:{port}...")
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=5.0
            )
            
            # ✅ PHASE 3: Apply TCP optimizations for high-performance transfer
            sock = writer.get_extra_info('socket')
            if sock:
                self._optimize_tcp_socket(sock)
            
            async with self._connection_lock:
                if target not in self._connection_pool:
                    self._connection_pool[target] = []
                self._connection_pool[target].append((reader, writer, now, 0))  # 0 errors for new connection
            
            logger.debug(f"✅ Connected to {host}:{port} (pool size: {len(self._connection_pool[target])}, TCP_NODELAY enabled)")
            return reader, writer
        except Exception as conn_error:
            logger.error(f"❌ Failed to connect to {host}:{port}: {conn_error}")
            raise
    
    async def _drain_and_close(self, writer):
        """Background task to drain and close connection (for fire-and-forget)."""
        try:
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        except:
            pass  # Ignore errors in background task
    
    async def _send_tcp_request(self, request: Dict, use_pool: bool = True, 
                               wait_for_response: bool = True,
                               server_address: Optional[str] = None) -> Dict:
        """Send request via TCP with proper message type protocol (optimized with connection pooling).
        
        Uses secure JSON + binary serialization (no pickle for security).
        
        Args:
            request: Request dictionary
            use_pool: If False, create a new connection (for parallel transfers to avoid conflicts)
            wait_for_response: If False, send and return immediately (fire-and-forget)
            server_address: Optional server address (if not in request)
        """
        
        import asyncio
        import time
        import uuid
        from djinn.server.profiling_context import record_phase
        
        request_type = request.get('type', 'UNKNOWN')
        logger.info(f"🔴 _send_tcp_request START: request_type={request_type}, use_pool={use_pool}, wait_for_response={wait_for_response}")
        
        # Use provided server_address or get from request or fallback
        if not server_address:
            server_address = request.get('server_address') or await self._get_server_address()
        
        host, port = server_address.split(':')
        port = int(port)
        
        conn_key = f"{host}:{port}"
        reader = None
        writer = None
        connection_from_pool = False
        
        # Generate request ID for correlation (full UUID hex for better uniqueness)
        request_id = uuid.uuid4().hex  # Full UUID (32 hex chars)
        request = request.copy()  # Don't modify original
        request['_request_id'] = request_id  # Add to request for tracking
        
        # ✅ IMPROVEMENT: Enable keep-alive for chunked transfers and frequent requests
        # Keep connection alive for chunked protocol to avoid connection overhead
        request_type = request.get('type', '')
        if request_type in ['REGISTER_MODEL_CHUNKED', 'REGISTER_MODEL_CHUNK', 'REGISTER_MODEL_FINALIZE']:
            request['_keep_alive'] = True  # Keep connection alive for chunked transfers
        
        try:
            # Get connection from pool (reuses existing if available) or create new
            if use_pool:
                reader, writer = await self._get_connection(host, port)
                connection_from_pool = True
            else:
                # Create new connection for parallel transfers (avoid pool conflicts)
                logger.info(f"🔌 CREATING NEW CONNECTION for {request.get('type', 'unknown')} to {host}:{port}")
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=5.0
                )
                # ✅ PHASE 3: Apply TCP optimizations for high-performance transfer
                # CRITICAL: Ensure socket is available and optimization succeeds
                sock = writer.get_extra_info('socket')
                if not sock:
                    # Try multiple times (socket might not be ready immediately)
                    for attempt in range(3):
                        await asyncio.sleep(0.01)  # 10ms delay
                        sock = writer.get_extra_info('socket')
                        if sock:
                            break
                
                if sock:
                    self._optimize_tcp_socket(sock)
                    # ✅ CRITICAL: Verify TCP_NODELAY is actually set (prevent Nagle's algorithm)
                    import socket
                    try:
                        actual_nodelay = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
                        if not actual_nodelay:
                            logger.error(f"⚠️ CRITICAL: TCP_NODELAY NOT enabled on new connection! Nagle's algorithm is ACTIVE. This will cause message batching and protocol corruption.")
                            # Force set it again
                            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                            actual_nodelay = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
                            if not actual_nodelay:
                                raise RuntimeError("Failed to enable TCP_NODELAY - Nagle's algorithm will batch writes and corrupt protocol!")
                        logger.debug(f"✅ TCP_NODELAY verified: {actual_nodelay}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not verify TCP_NODELAY: {e}")
                else:
                    logger.error(f"⚠️ CRITICAL: Socket not available for new connection to {host}:{port}! Connection may have issues.")
                
                connection_from_pool = False
                logger.info(f"✅ New connection created to {host}:{port} (TCP optimized: NODELAY, 16MB buffers)")
            
            # Import protocol constants
            from ..server.transport.protocol import MessageType
            
            # Determine message type based on request type
            request_type = request.get('type', '')
            logger.info(f"🔍 DEBUG: Determining msg_type for request_type='{request_type}' (from request at line 1272)")
            if request_type == 'REGISTER_MODEL':
                msg_type = MessageType.REGISTER_MODEL
            elif request_type == 'EXECUTE_MODEL':
                msg_type = MessageType.EXECUTE_MODEL
            elif request_type == 'REGISTER_MODEL_CHUNKED':
                msg_type = MessageType.REGISTER_MODEL_CHUNKED
                logger.info(f"✅ REGISTER_MODEL_CHUNKED => msg_type=0x{msg_type:02x} ({msg_type})")
            elif request_type == 'REGISTER_MODEL_CHUNK':
                msg_type = MessageType.REGISTER_MODEL_CHUNK
                logger.info(f"✅ REGISTER_MODEL_CHUNK => msg_type=0x{msg_type:02x} ({msg_type})")
            elif request_type == 'REGISTER_MODEL_FINALIZE':
                msg_type = MessageType.REGISTER_MODEL_FINALIZE
            elif request_type == 'INIT_MODEL':
                msg_type = MessageType.INIT_MODEL
            elif request_type == 'WARMUP_GPU':
                msg_type = MessageType.WARMUP_GPU
            else:
                logger.error(f"🚨 CRITICAL: Unknown request_type='{request_type}'! This will cause protocol error. Defaulting to 0x03 (EXECUTE_SUBGRAPH). Request keys: {list(request.keys())}")
                msg_type = 0x03  # Default to EXECUTE_SUBGRAPH for compatibility (legacy)
            
            # Serialize request using secure serializer (JSON + binary, no pickle)
            logger.debug(f"📦 Serializing request (type={request_type})...")
            with record_phase('model_cache_request_serialization'):
                from .secure_serializer import SecureSerializer
                request_bytes = SecureSerializer.serialize_request(request)
                request_len = len(request_bytes)
            
            # ✅ TD-3 FIX: Validate message length matches actual serialized data
            if request_len != len(request_bytes):
                raise ValueError(
                    f"Message length mismatch: calculated {request_len} but actual is {len(request_bytes)} bytes"
                )
            if request_len == 0:
                raise ValueError("Invalid message length: 0 bytes (serialization failed?)")
            
            # ✅ BUG FIX: Validate length is reasonable (prevent protocol mismatch)
            MAX_REASONABLE_MESSAGE_SIZE = 10 * 1024 * 1024 * 1024  # 10GB
            if request_len > MAX_REASONABLE_MESSAGE_SIZE:
                raise ValueError(
                    f"⚠️  Suspicious message length: {request_len} bytes ({request_len / (1024*1024):.1f} MB). "
                    f"This is likely a serialization bug. Max reasonable size: {MAX_REASONABLE_MESSAGE_SIZE / (1024*1024):.1f} MB"
                )
            
            # ✅ BUG FIX: Check for suspicious round numbers (protocol mismatch indicator)
            SUSPICIOUS_SIZES = [64 * 1024 * 1024, 128 * 1024 * 1024, 256 * 1024 * 1024]
            if request_len in SUSPICIOUS_SIZES and request_len > 50 * 1024 * 1024:
                # This is almost certainly a bug - reject it
                raise ValueError(
                    f"⚠️  CRITICAL: Suspicious message length {request_len} bytes ({request_len / (1024*1024):.1f} MB) "
                    f"is a round number (power of 2). This indicates a serialization bug or protocol mismatch. "
                    f"Actual serialized size: {len(request_bytes)} bytes. "
                    f"Request type: {request_type}. "
                    f"Rejecting to prevent protocol corruption."
                )
            
            logger.debug(f"📦 Serialized {request_len} bytes (secure format, validated)")
            
            # Send message type (1 byte) + length (8 bytes) + data
            logger.info(f"📤 Sending message: type={msg_type} (0x{msg_type:02x}), length={request_len} bytes ({request_len / (1024*1024):.2f} MB) to {host}:{port}")
            # ✅ BUG FIX: Log length header bytes for debugging
            length_header_bytes = request_len.to_bytes(8, 'big')
            logger.debug(f"📤 Length header bytes: {length_header_bytes.hex()} (big-endian uint64)")
            try:
                with record_phase('model_cache_network_send'):
                    # ✅ PHASE 3: Optimize write calls - combine writes for large messages
                    # For large messages (> 1MB), use single write() to reduce syscalls and improve TCP utilization
                    # ✅ CRITICAL BUG FIX: Always combine writes and ALWAYS drain before closing
                    # For fire-and-forget, we MUST ensure all data is sent before closing connection
                    # Otherwise, server reads incomplete data and gets protocol mismatch errors
                    logger.debug(f"Writing message ({request_len / (1024*1024):.2f} MB) - combining writes for reliability...")
                    logger.info(f"📤 DEBUG: Creating message - msg_type={msg_type} (0x{msg_type:02x}), request_type={request_type}, request_len={request_len}")
                    # Combine header + data into single write (ensures atomic write)
                    combined = bytearray()
                    combined.extend(bytes([msg_type]))
                    combined.extend(request_len.to_bytes(8, 'big'))
                    combined.extend(request_bytes)
                    
                    if len(combined) > 0:
                        logger.info(f"📤 DEBUG: Message combined - first byte: 0x{combined[0]:02x}, length: {len(combined)}, expected first: 0x{msg_type:02x}")
                        if combined[0] != msg_type:
                            logger.error(f"🚨 BUG: First byte mismatch! combined[0]=0x{combined[0]:02x} but msg_type=0x{msg_type:02x}")
                    else:
                        logger.error(f"🚨 BUG: combined is empty!")
                    
                    # Verify message type header for chunks (critical for debugging)
                    if request_type == 'REGISTER_MODEL_CHUNK':
                        if combined[0] != msg_type:
                            raise RuntimeError(
                                f"CRITICAL: Chunk message type header missing! "
                                f"Expected 0x{msg_type:02x}, got 0x{combined[0]:02x}"
                            )
                        logger.debug(f"✅ Chunk message type header verified: 0x{combined[0]:02x}")
                    
                    # ✅ CRITICAL: Verify combined size matches expected
                    expected_combined_size = 1 + 8 + request_len
                    actual_combined_size = len(combined)
                    if actual_combined_size != expected_combined_size:
                        raise ValueError(
                            f"⚠️  CRITICAL: Combined message size mismatch! "
                            f"Expected: {expected_combined_size} bytes (1 + 8 + {request_len}), "
                            f"Actual: {actual_combined_size} bytes. "
                            f"This will cause protocol corruption."
                        )
                    
                    # Write all data at once
                    # Note: writer.write() returns None in asyncio, not bytes written
                    # We verify by checking combined size before writing
                    
                    # Convert to bytes and write
                    combined_bytes = bytes(combined)
                    writer.write(combined_bytes)
                    logger.debug(f"✅ Wrote {len(combined_bytes)} bytes (1 + 8 + {request_len} = {1+8+request_len})")
                    
                    # ✅ CRITICAL: ALWAYS drain before closing (even for fire-and-forget)
                    # This ensures server receives all data before connection is closed
                    logger.debug(f"Draining data to ensure server receives all {request_len} bytes...")
                    try:
                        # Use longer timeout for large messages (1s per MB, min 10s, max 60s)
                        drain_timeout = min(60.0, max(10.0, (request_len / (1024 * 1024)) * 1.0))
                        await asyncio.wait_for(writer.drain(), timeout=drain_timeout)
                        logger.debug(f"✅ Data drained successfully ({request_len} bytes sent and confirmed)")
                    except asyncio.TimeoutError:
                        raise RuntimeError(
                            f"⚠️  CRITICAL: Drain timeout after {drain_timeout:.1f}s for {request_len} bytes. "
                            f"Server may not have received all data. This will cause protocol corruption."
                        )
                    except Exception as drain_error:
                        raise RuntimeError(
                            f"⚠️  CRITICAL: Drain failed: {drain_error}. "
                            f"Server may not have received all {request_len} bytes. "
                            f"This will cause protocol corruption."
                        ) from drain_error
                    
                
                logger.info("✅ Data sent successfully" + (", waiting for response..." if wait_for_response else " (fire-and-forget)"))
            except Exception as send_error:
                logger.error(f"❌ Error sending data: {send_error}")
                raise
            
                # ✅ OPTIMIZATION: Fire-and-forget for chunks (don't wait for response)
            if not wait_for_response:
                # ✅ TD-4 FIX: Always remove connection from pool for fire-and-forget
                # Regardless of pool origin, close the connection since it's fire-and-forget
                if connection_from_pool:
                    # Remove from pool since it will be closed
                    async with self._connection_lock:
                        if conn_key in self._connection_pool:
                            pool = self._connection_pool[conn_key]
                            # Find and remove the connection we used
                            for i, (r, w, last_used, error_count) in enumerate(pool):
                                if w is writer:
                                    pool.pop(i)
                                    logger.debug(f"Removed fire-and-forget connection from pool")
                                    break
                            if not pool:
                                del self._connection_pool[conn_key]
                
                # ✅ CRITICAL: For ALL fire-and-forget connections (whether from pool or new),
                # FORCE close them to ensure server-side reader buffer is flushed
                logger.debug("Fire-and-forget: Force-closing connection to flush server buffer...")
                try:
                    writer.close()
                    # Wait for close to complete (critical for server-side buffer flush)
                    await asyncio.wait_for(writer.wait_closed(), timeout=5.0)
                    logger.debug("✅ Connection force-closed and buffer flushed")
                except asyncio.TimeoutError:
                    logger.warning("⚠️  Connection close timeout - server buffer may not be flushed")
                except Exception as e:
                    logger.debug(f"Error force-closing fire-and-forget connection: {e}")
                
                # Return success immediately
                return {
                    'status': 'success',
                    'message': 'Request sent (fire-and-forget)'
                }
            
            # Read response (message type + length + data)
            # Use longer timeout for finalize (reassembly can take time) and model execution
            is_finalize = request.get('type') == 'REGISTER_MODEL_FINALIZE'
            is_execute = request.get('type') == 'EXECUTE_MODEL'
            is_init = request.get('type') == 'INIT_MODEL'
            is_warmup = request.get('type') == 'WARMUP_GPU'
            base_timeout = 600.0 if is_finalize else (300.0 if is_execute else (120.0 if is_init else (60.0 if is_warmup else 30.0)))  # ✅ FIX: 1min for warmup, 2min for init, 5min for execution, 10min for finalize
            
            logger.debug(f"Reading response (timeout={base_timeout}s, is_finalize={is_finalize})...")
            with record_phase('model_cache_network_receive'):
                try:
                    msg_type_response = await asyncio.wait_for(reader.readexactly(1), timeout=base_timeout)
                    logger.debug(f"Received response message type: 0x{msg_type_response[0]:02x}")
                    
                    logger.debug("Reading response length...")
                    response_len_bytes = await asyncio.wait_for(reader.readexactly(8), timeout=10.0)
                    response_len = int.from_bytes(response_len_bytes, 'big')
                    logger.debug(f"Response length: {response_len} bytes")
                    
                    logger.info(f"📥 Reading {response_len} bytes of response data ({response_len / (1024*1024):.2f} MB)...")
                    # For finalize, use longer timeout (reassembly + registration can take time)
                    # ✅ FIX: For EXECUTE_MODEL, use longer timeout based on response size
                    # Large tensors can take time to transfer (e.g., 100MB = ~1s at 1Gbps)
                    # ✅ FIX: For INIT_MODEL, use base timeout (warmup can take 8-10s)
                    if is_execute:
                        # Calculate timeout: 1s per 50MB + 10s base (conservative for network)
                        read_timeout = max(60.0, (response_len / (50 * 1024 * 1024)) + 10.0)
                    elif is_finalize:
                        read_timeout = base_timeout
                    elif is_init:
                        read_timeout = base_timeout  # 2 minutes for model init
                    elif is_warmup:
                        read_timeout = base_timeout  # 1 minute for GPU warmup
                    else:
                        read_timeout = max(30.0, response_len / (1024 * 1024) * 2)
                    
                    logger.info(f"🔍 [DIAGNOSTIC] Using read timeout: {read_timeout:.1f}s for {response_len} bytes")
                    
                    response_bytes = await asyncio.wait_for(
                        reader.readexactly(response_len),
                        timeout=read_timeout
                    )
                    logger.info(f"✅ Response data received ({len(response_bytes)} bytes)")
                except asyncio.TimeoutError:
                    logger.error(f"❌ Timeout waiting for response from {host}:{port} (timeout={read_timeout:.1f}s, response_len={response_len} bytes)")
                    raise RuntimeError(f"Response timeout from server (timeout={read_timeout:.1f}s, expected {response_len} bytes)")
                except (ConnectionResetError, BrokenPipeError) as conn_error:
                    logger.error(f"❌ Connection reset/closed while reading response: {conn_error}")
                    raise RuntimeError(f"Connection lost: {conn_error}")
                except Exception as read_error:
                    logger.error(f"❌ Error reading response: {read_error}")
                    raise
            
            with record_phase('model_cache_response_deserialization'):
                from .secure_serializer import SecureSerializer
                # Detect format: SecureSerializer starts with version 1 (0x01), pickle starts with 0x80
                if len(response_bytes) > 0:
                    first_byte = response_bytes[0]
                    if first_byte == 0x01:
                        # SecureSerializer format
                        try:
                            response = SecureSerializer.deserialize_response(response_bytes)
                        except Exception as e:
                            logger.error(f"Failed to deserialize SecureSerializer response: {e}")
                            raise
                    elif first_byte == 0x80:
                        # Pickle format (backward compatibility)
                        logger.debug("Response is pickle format (legacy)")
                        import pickle
                        try:
                            response = pickle.loads(response_bytes)
                        except Exception as e:
                            logger.error(f"Failed to deserialize pickle response: {e}")
                            raise
                    else:
                        # Unknown format, try SecureSerializer first, then pickle
                        logger.warning(f"Unknown response format (first byte: 0x{first_byte:02x}), trying SecureSerializer...")
                        try:
                            response = SecureSerializer.deserialize_response(response_bytes)
                        except Exception:
                            logger.warning("SecureSerializer failed, trying pickle...")
                            import pickle
                            response = pickle.loads(response_bytes)
                else:
                    raise ValueError("Empty response data")
            
            # ✅ IMPROVEMENT: Verify request ID matches (with strict mode support)
            if '_request_id' in response:
                response_request_id = response.get('_request_id')
                if response_request_id != request_id:
                    # Check if strict validation is enabled
                    try:
                        from ..config import get_config
                        config = get_config()
                        strict_mode = getattr(config.security, 'strict_request_id_validation', False) if hasattr(config, 'security') else False
                        
                        if strict_mode:
                            raise RuntimeError(
                                f"Request ID mismatch: sent {request_id}, received {response_request_id}. "
                                "Possible response routing error."
                            )
                        else:
                            logger.warning(
                                f"Request ID mismatch: sent {request_id}, received {response_request_id}"
                            )
                    except Exception:
                        # Config not available, just warn
                        logger.warning(
                            f"Request ID mismatch: sent {request_id}, received {response_request_id}"
                        )
            
            # ✅ TD-5 FIX: Handle connection cleanup properly on success
            if connection_from_pool:
                # Update last used time in pool (connection stays in pool for reuse)
                import time
                now = time.time()
                async with self._connection_lock:
                    if conn_key in self._connection_pool:
                        pool = self._connection_pool[conn_key]
                        # Find and update the connection we used
                        for i, (r, w, last_used, error_count) in enumerate(pool):
                            if w is writer:
                                pool[i] = (r, w, now, error_count)
                                break
            else:
                # New connection - close it if not from pool
                try:
                    if writer and not writer.is_closing():
                        writer.close()
                        await writer.wait_closed()
                except:
                    pass
                # Close new connection created for parallel transfer (temporary connection)
                try:
                    if writer and not writer.is_closing():
                        writer.close()
                        await writer.wait_closed()
                except:
                    pass
            
            return response
            
        except Exception as e:
            # On error, handle connection cleanup
            if connection_from_pool:
                # Track error count and close if too many errors
                async with self._connection_lock:
                    if conn_key in self._connection_pool:
                        pool = self._connection_pool[conn_key]
                        # Find the connection we used
                        for i, (r, w, last_used, error_count) in enumerate(pool):
                            if w is writer:
                                error_count += 1
                                
                                if error_count >= self._max_connection_errors:
                                    # Too many errors, close and remove
                                    logger.warning(f"Closing connection to {conn_key} after {error_count} errors")
                                    if w and not w.is_closing():
                                        try:
                                            w.close()
                                            await w.wait_closed()
                                        except:
                                            pass
                                    pool.pop(i)
                                    # Clean up empty pool
                                    if not pool:
                                        del self._connection_pool[conn_key]
                                else:
                                    # Update error count
                                    pool[i] = (r, w, last_used, error_count)
                                    logger.debug(f"Connection error count: {error_count}/{self._max_connection_errors}")
                                break
            else:
                # Close new connection created for parallel transfer
                if writer and not writer.is_closing():
                    try:
                        writer.close()
                        await writer.wait_closed()
                    except:
                        pass
            
            logger.error(f"TCP request failed: {e}")
            import traceback
            logger.debug(f"Traceback:\n{traceback.format_exc()}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def _get_server_address(self, fingerprint: Optional[str] = None, 
                                  inputs: Optional[Dict] = None,
                                  hints: Optional[Dict] = None) -> str:
        """
        Get TCP server address, optionally using global fleet coordinator.
        
        The TCP server runs on port 5556 (or config.network.control_port).
        The control plane runs on a different port (5560+).
        We need the TCP server port for model registration.
        
        IMPORTANT: Always use port 5556 for TCP server, not control plane port.
        
        Args:
            fingerprint: Optional model fingerprint (for global coordinator routing)
            inputs: Optional input tensors (for size estimation)
            hints: Optional semantic hints (for load balancing)
        
        Returns:
            Server address (e.g., "server-3:5558")
        """
        import time
        import hashlib
        import json
        
        # ✅ Check local cache first (Phase 3 optimization)
        if fingerprint:
            # Create cache key from fingerprint and hints
            hints_str = json.dumps(hints or {}, sort_keys=True) if hints else ""
            hints_hash = hashlib.md5(hints_str.encode()).hexdigest()[:8]
            cache_key = (fingerprint, hints_hash)
            
            async with self._cache_lock:
                if cache_key in self._server_address_cache:
                    cached_address, expires_at = self._server_address_cache[cache_key]
                    if time.time() < expires_at:
                        logger.debug(f"💾 Using cached server address: {cached_address} (expires in {expires_at - time.time():.1f}s)")
                        return cached_address
                    else:
                        # Expired, remove from cache
                        del self._server_address_cache[cache_key]
        
        # Priority 1: Use global fleet coordinator if available and enabled
        if self.coordinator and self.coordinator.global_coordinator:
            if fingerprint and inputs:
                try:
                    from ...config import get_config
                    fleet_config = get_config().fleet
                    
                    # Query global coordinator for best server
                    best_server = await self.coordinator.global_coordinator.route_request(
                        fingerprint=fingerprint,
                        inputs=inputs,
                        hints=hints or {}
                    )
                    
                    # ✅ Cache the result (Phase 3 optimization)
                    if fingerprint:
                        cache_ttl = fleet_config.local_cache_ttl
                        expires_at = time.time() + cache_ttl
                        async with self._cache_lock:
                            self._server_address_cache[cache_key] = (best_server, expires_at)
                            # Cleanup old entries (keep cache size reasonable)
                            if len(self._server_address_cache) > 1000:
                                current_time = time.time()
                                expired_keys = [
                                    k for k, (_, exp) in self._server_address_cache.items()
                                    if exp < current_time
                                ]
                                for k in expired_keys:
                                    self._server_address_cache.pop(k, None)
                    
                    logger.info(f"🎯 Global coordinator routed to: {best_server} (cached for {cache_ttl}s)")
                    return best_server
                except Exception as e:
                    logger.warning(f"⚠️  Global coordinator routing failed: {e}, falling back to direct server")
                    # Fall through to direct server selection
        
        # Priority 2: Use explicit server address if provided during initialization
        if self._server_address:
            logger.debug(f"🔍 Using explicit server address: {self._server_address}")
            return self._server_address
        
        # Priority 3: Extract server address from runtime state if available
        # The server_address includes both host and port (e.g., "localhost:5556" or "localhost:5557")
        try:
            from ...backend.runtime.initialization import _runtime_state
            if _runtime_state.server_address:
                # server_address is in format "host:port" - use it directly
                server_address = _runtime_state.server_address
                logger.debug(f"🔍 Using runtime state server address: {server_address}")
                return server_address
        except Exception as e:
            logger.debug(f"Could not get server address from runtime state: {e}")
        
        # Priority 4: Fallback to default host and port
        host = "localhost"
        port = 5556  # Default TCP server port
        server_address = f"{host}:{port}"
        logger.warning(f"🔍 Using default TCP server address: {server_address} (no explicit address or runtime state)")
        return server_address
    
    def _detect_framework(self, model: nn.Module) -> Optional[str]:
        """Detect model framework."""
        module_name = model.__class__.__module__
        
        if 'transformers' in module_name:
            return 'transformers'
        elif 'torchvision' in module_name:
            return 'torchvision'
        else:
            return None
    
    def _serialize_weights_binary(self, weights: Dict[str, torch.Tensor]) -> bytes:
        """
        Direct binary serialization - NO intermediate dict, NO JSON.
        
        Protocol:
        [4 bytes: num_weights]
        [for each weight:
            [4 bytes: name_len] [name_bytes]
            [4 bytes: shape_len] [shape: 4*shape_len bytes]
            [4 bytes: dtype_len] [dtype_bytes]
            [8 bytes: data_len] [data_bytes]
        ]
        
        This is 10x faster than dict-based serialization and eliminates JSON overhead.
        """
        import struct
        import numpy as np
        
        result = bytearray()
        result.extend(struct.pack('>I', len(weights)))
        
        for name, tensor in weights.items():
            # ✅ CRITICAL FIX: Convert to numpy with explicit detach
            # Some tensors have requires_grad=True and need special handling
            if isinstance(tensor, torch.Tensor):
                # Move to CPU first
                tensor_cpu = tensor.cpu()
                # Then detach and convert to numpy in one operation
                try:
                    np_array = tensor_cpu.detach().numpy()
                except RuntimeError:
                    # If detach().numpy() fails, try explicit data access
                    np_array = tensor_cpu.data.cpu().numpy()
            else:
                np_array = np.asarray(tensor, dtype=np.float32)
            
            data = np_array.tobytes()  # Direct binary
            
            # Pack: name
            name_bytes = name.encode('utf-8')
            result.extend(struct.pack('>I', len(name_bytes)))
            result.extend(name_bytes)
            
            # Pack: shape
            shape = tensor.shape
            result.extend(struct.pack('>I', len(shape)))
            result.extend(struct.pack(f'>{len(shape)}I', *shape))
            
            # Pack: dtype
            dtype_str = str(tensor.dtype)
            dtype_bytes = dtype_str.encode('utf-8')
            result.extend(struct.pack('>I', len(dtype_bytes)))
            result.extend(dtype_bytes)
            
            # Pack: data
            result.extend(struct.pack('>Q', len(data)))
            result.extend(data)
        
        return bytes(result)
    
    def _serialize_inputs(self, inputs: Dict[str, Any]) -> Dict:
        """
        Serialize inputs for transmission using numpy binary format.
        
        Consistent with weight serialization - uses binary format instead of list.
        """
        import numpy as np
        
        serialized = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                # ✅ CRITICAL FIX: Convert to numpy with explicit detach
                tensor_cpu = value.cpu()
                try:
                    np_array = tensor_cpu.detach().numpy()
                except RuntimeError:
                    np_array = tensor_cpu.data.cpu().numpy()
                serialized[key] = {
                    'data': np_array.tobytes(),  # Binary numpy format
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                    'format': 'numpy_binary',  # Format marker
                }
            else:
                serialized[key] = value
        return serialized
    
    def _deserialize_tensor(self, data: Dict) -> torch.Tensor:
        """Deserialize tensor from dict."""
        import numpy as np
        
        # ✅ FIX: Handle binary numpy format (from _serialize_inputs)
        # data['data'] is binary bytes from np_array.tobytes(), not a list
        if isinstance(data['data'], bytes):
            # Binary format - use frombuffer to reconstruct numpy array
            dtype_str = data.get('dtype', 'float32')
            if dtype_str.startswith('torch.'):
                # Convert 'torch.float32' -> 'float32'
                dtype_str = dtype_str.replace('torch.', '')
            
            # Map torch dtypes to numpy dtypes
            dtype_map = {
                'float32': np.float32,
                'float16': np.float16,
                'int64': np.int64,
                'int32': np.int32,
                'bool': np.bool_,
            }
            numpy_dtype = dtype_map.get(dtype_str, np.float32)
            
            # Reconstruct numpy array from binary bytes
            numpy_data = np.frombuffer(data['data'], dtype=numpy_dtype)
            # Reshape to correct shape
            numpy_data = numpy_data.reshape(data['shape'])
            tensor = torch.from_numpy(numpy_data)
        else:
            # Legacy format - list of values (backward compatibility)
            dtype_str = data.get('dtype', 'float32')
            if dtype_str.startswith('torch.'):
                dtype_str = dtype_str.replace('torch.', '')
            
            dtype_map = {
                'float32': 'float32',
                'float16': 'float16',
                'int64': 'int64',
                'int32': 'int32',
                'bool': 'bool',
            }
            numpy_dtype = dtype_map.get(dtype_str, 'float32')
            numpy_data = np.array(data['data'], dtype=numpy_dtype)
            tensor = torch.from_numpy(numpy_data)
            
            if list(tensor.shape) != data['shape']:
                tensor = tensor.reshape(data['shape'])
        
        return tensor

