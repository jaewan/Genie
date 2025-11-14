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
from typing import Dict, Optional, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EnhancedModelManager:
    """
    Client-side model manager for the new model cache system.
    
    Handles:
    - Model fingerprinting
    - Model registration (one-time)
    - Efficient execution requests (model ID + inputs only)
    - Fallback to graph execution
    """
    
    def __init__(self, coordinator=None):
        """
        Initialize enhanced model manager.
        
        Args:
            coordinator: DjinnCoordinator instance (optional, will get global if None)
        """
        from .model_fingerprint import ModelFingerprint
        from .model_registry import get_model_registry
        from .cache_query import get_cache_query_client
        
        self.fingerprint = ModelFingerprint
        self.registry = get_model_registry()
        self.cache_client = get_cache_query_client()
        
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
        
        logger.info("EnhancedModelManager initialized")
    
    async def register_model(self, 
                            model: nn.Module,
                            model_id: Optional[str] = None) -> str:
        """
        Register model with server (one-time operation).
        
        Args:
            model: PyTorch model
            model_id: Optional explicit model identifier
        
        Returns:
            Model fingerprint
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
                # Ensure it's on CPU for serialization
                if isinstance(weight_tensor, torch.Tensor):
                    if weight_tensor.device.type == 'meta':
                        logger.warning(f"Skipping meta tensor for {param_name}")
                        continue
                    uncached_weights[param_name] = weight_tensor.cpu() if weight_tensor.device.type != 'cpu' else weight_tensor
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
        server_address = self._get_server_address()
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
        
        # Send registration request
        registration_request = {
            'type': 'REGISTER_MODEL',
            'fingerprint': fingerprint,
            'descriptor': descriptor,
            'weight_ids': weight_ids,
            'uncached_weights': self._serialize_weights(uncached_weights),
            'architecture_data': architecture_data
        }
        
        # Send via coordinator or direct TCP
        response = await self._send_request(registration_request)
        
        if response.get('status') == 'success':
            self.registered_models[fingerprint] = {
                'fingerprint': fingerprint,
                'descriptor': descriptor,
                'weight_ids': weight_ids
            }
            logger.info(f"Model {fingerprint} registered successfully")
        else:
            error_msg = response.get('message', 'Unknown error')
            raise RuntimeError(f"Model registration failed: {error_msg}")
        
        return fingerprint
    
    async def execute_model(self,
                           model: nn.Module,
                           inputs: Dict[str, Any],
                           model_id: Optional[str] = None) -> torch.Tensor:
        """
        Execute model using new cache system (or fallback to graph).
        
        Args:
            model: PyTorch model
            inputs: Input tensors dict
            model_id: Optional explicit model identifier
        
        Returns:
            Output tensor
        """
        
        # Compute fingerprint
        fingerprint = self.fingerprint.compute(model, model_id)
        
        # Check if registered
        if fingerprint not in self.registered_models:
            # Auto-register if not registered
            logger.info(f"Auto-registering model {fingerprint}")
            await self.register_model(model, model_id)
        
        # Use model cache if enabled
        if self.use_model_cache:
            try:
                return await self._execute_via_cache(fingerprint, inputs)
            except Exception as e:
                logger.warning(f"Model cache execution failed: {e}, falling back to graph")
                # Fall through to graph execution
        
        # Fallback to graph execution
        return await self._execute_via_graph(model, inputs)
    
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
        with record_phase('model_cache_network_c2s'):
            response = await self._send_request(execution_request)
        
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
            return await self.coordinator.execute_remote_subgraph(
                subgraph=subgraph.serialize(),  # Serialize RemoteSubgraph to dict
                input_data=input_data,
                target=self._get_server_address(),
                model=model  # Pass model for cache check
            )
        else:
            raise RuntimeError("No coordinator available for graph execution")
    
    async def _send_request(self, request: Dict) -> Dict:
        """Send request to server."""
        
        if self.coordinator:
            # Use coordinator's transport
            # For now, use a simple TCP connection
            return await self._send_tcp_request(request)
        else:
            return await self._send_tcp_request(request)
    
    async def _send_tcp_request(self, request: Dict) -> Dict:
        """Send request via TCP with proper message type protocol."""
        
        import pickle
        import asyncio
        from djinn.server.profiling_context import record_phase
        
        server_address = self._get_server_address()
        host, port = server_address.split(':')
        port = int(port)
        
        try:
            logger.info(f"🔌 Opening connection to {host}:{port}...")
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=5.0
                )
                logger.info(f"✅ Connected to {host}:{port}")
            except Exception as conn_error:
                logger.error(f"❌ Failed to connect to {host}:{port}: {conn_error}")
                raise
            
            # Determine message type based on request type
            request_type = request.get('type', '')
            if request_type == 'REGISTER_MODEL':
                msg_type = 0x05  # REGISTER_MODEL
            elif request_type == 'EXECUTE_MODEL':
                msg_type = 0x06  # EXECUTE_MODEL
            else:
                msg_type = 0x03  # Default to EXECUTE_SUBGRAPH for compatibility
            
            # Serialize request
            logger.debug(f"📦 Serializing request (type={request_type})...")
            with record_phase('model_cache_request_serialization'):
                request_bytes = pickle.dumps(request)
                request_len = len(request_bytes)
            logger.debug(f"📦 Serialized {request_len} bytes")
            
            # Send message type (1 byte) + length (8 bytes) + data
            logger.debug(f"📤 Sending message: type={msg_type} (0x{msg_type:02x}), length={request_len}")
            with record_phase('model_cache_network_send'):
                writer.write(bytes([msg_type]))
                writer.write(request_len.to_bytes(8, 'big'))
                writer.write(request_bytes)
                logger.debug("📤 Flushing data...")
                await writer.drain()
            logger.debug("✅ Data sent successfully")
            
            # Read response (message type + length + data)
            with record_phase('model_cache_network_receive'):
                msg_type_response = await reader.readexactly(1)
                response_len_bytes = await reader.readexactly(8)
                response_len = int.from_bytes(response_len_bytes, 'big')
                response_bytes = await reader.readexactly(response_len)
            
            with record_phase('model_cache_response_deserialization'):
                response = pickle.loads(response_bytes)
            
            writer.close()
            await writer.wait_closed()
            
            return response
            
        except Exception as e:
            logger.error(f"TCP request failed: {e}")
            import traceback
            logger.debug(f"Traceback:\n{traceback.format_exc()}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _get_server_address(self) -> str:
        """Get TCP server address (not control plane).
        
        The TCP server runs on port 5556 (or config.network.control_port).
        The control plane runs on a different port (5560+).
        We need the TCP server port for model registration.
        
        IMPORTANT: Always use port 5556 for TCP server, not control plane port.
        """
        # Always use port 5556 for TCP server (where model registration happens)
        # The control plane (5560+) is for coordinator communication, not model registration
        
        # Extract host from runtime state if available
        host = "localhost"
        try:
            from ...backend.runtime.initialization import _runtime_state
            if _runtime_state.server_address:
                # Extract host from server_address (e.g., "localhost:5556")
                host = _runtime_state.server_address.split(':')[0]
                logger.debug(f"🔍 Using runtime state host: {host}")
        except Exception as e:
            logger.debug(f"Could not get host from runtime state: {e}")
        
        # Always use port 5556 for TCP server
        server_address = f"{host}:5556"
        logger.info(f"🔍 Using TCP server address: {server_address}")
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
    
    def _serialize_weights(self, weights: Dict[str, torch.Tensor]) -> Dict:
        """Serialize weights for transmission."""
        import pickle
        import base64
        
        # Use pickle + base64 for efficient binary serialization
        serialized = {}
        for name, tensor in weights.items():
            # Detach to avoid grad issues, ensure CPU, then pickle
            tensor_cpu = tensor.detach().cpu() if hasattr(tensor, 'detach') else tensor.cpu()
            tensor_bytes = pickle.dumps(tensor_cpu)
            tensor_b64 = base64.b64encode(tensor_bytes).decode('ascii')
            
            serialized[name] = {
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'data': tensor_b64,  # Base64-encoded pickle
            }
        return serialized
    
    def _serialize_inputs(self, inputs: Dict[str, Any]) -> Dict:
        """Serialize inputs for transmission."""
        serialized = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                serialized[key] = {
                    'data': value.cpu().numpy().tolist(),
                    'shape': list(value.shape),
                    'dtype': str(value.dtype)
                }
            else:
                serialized[key] = value
        return serialized
    
    def _deserialize_tensor(self, data: Dict) -> torch.Tensor:
        """Deserialize tensor from dict."""
        import numpy as np
        
        # Convert torch dtype string to numpy dtype
        dtype_str = data.get('dtype', 'float32')
        if dtype_str.startswith('torch.'):
            # Convert 'torch.float32' -> 'float32'
            dtype_str = dtype_str.replace('torch.', '')
        
        # Map torch dtypes to numpy dtypes
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

