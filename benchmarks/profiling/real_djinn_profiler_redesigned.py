"""
Redesigned Djinn Profiler - Uses Enhanced Model Manager

This profiler compares the redesigned Djinn model cache system against
vanilla PyTorch to measure performance characteristics.
"""

import torch
import torch.nn as nn
import time
import logging
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# ✅ FIX: Configure logging to reduce verbosity
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(message)s',
    force=True
)

# Set Djinn loggers to ERROR to reduce noise (only show errors, not warnings)
for logger_name in ['djinn', 'djinn.core', 'djinn.server', 'djinn.backend', 'djinn.fleet']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Keep enhanced_model_manager at ERROR level (debug logs removed)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Profiler itself uses INFO

# ✅ FIX: Suppress tensor printing by overriding __repr__ temporarily
import torch
_original_tensor_repr = torch.Tensor.__repr__
def _suppressed_tensor_repr(self):
    if hasattr(self, 'shape'):
        return f"<Tensor shape={self.shape} dtype={self.dtype} device={self.device}>"
    return _original_tensor_repr(self)
torch.Tensor.__repr__ = _suppressed_tensor_repr

# ✅ FIX: Suppress binary data output - don't redirect, just prevent tensor printing
# The issue is tensors being printed, not stderr

# Import Djinn components
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from djinn.core.enhanced_model_manager import EnhancedModelManager
from djinn.core.model_fingerprint import ModelFingerprint
from djinn.core.model_tracker import get_model_tracker


class ExecutionMode(Enum):
    """Execution mode for profiling."""
    NEW_MODEL_CACHE = "new_model_cache"  # New model cache system


@dataclass
class RedesignedProfile:
    """Profile result for redesigned system."""
    mode: str
    model_name: str
    batch_size: int
    
    # Timing (ms)
    registration_time: float = 0.0  # One-time model registration
    registration_breakdown: Dict[str, float] = None  # ✅ NEW: Registration phase breakdown
    init_time: float = 0.0  # Model initialization (warmup) time
    execution_time: float = 0.0  # Model execution (warm)
    first_execution_time: float = 0.0  # ✅ NEW: First execution (cold start)
    cold_start_time: float = 0.0  # ✅ NEW: Cold start (registration + init + first exec)
    total_time: float = 0.0
    
    # Phase-level breakdown (ms)
    phase_breakdown: Dict[str, float] = None  # Phase name -> duration
    
    # Network stats
    network_sent_mb: float = 0.0
    network_received_mb: float = 0.0
    
    # Cache stats
    cache_hit: bool = False
    model_fingerprint: Optional[str] = None
    
    # Correctness
    output_shape: Optional[tuple] = None
    matches_baseline: bool = False
    
    def __post_init__(self):
        if self.phase_breakdown is None:
            self.phase_breakdown = {}
        if self.registration_breakdown is None:
            self.registration_breakdown = {}


class RedesignedDjinnProfiler:
    """Profiler for redesigned Djinn system."""
    
    def __init__(self):
        self.profiles: List[RedesignedProfile] = []
        self.server_process = None
    
    def start_server(self):
        """Start Djinn server using RemoteServerManager."""
        # ✅ FIX: Import directly to avoid triggering models import
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        # Import server_spawner directly without going through __init__.py
        from benchmarks.utils.server_spawner import RemoteServerManager
        
        self.server_manager = RemoteServerManager(host="127.0.0.1", port=5556, timeout=60)
        if self.server_manager.start():
            # Store actual server port (might be different if 5556 was in use)
            self.server_port = self.server_manager.port
            logger.info(f"✅ Server started on port {self.server_port}")
            return True
        else:
            logger.error("❌ Server failed to start")
            return False
    
    def stop_server(self):
        """Stop Djinn server."""
        if hasattr(self, 'server_manager') and self.server_manager:
            self.server_manager.stop()
            logger.info("✅ Server stopped")
    
    async def profile_new_model_cache(self, model_func, inputs: List[torch.Tensor],
                                     model_name: str, batch_size: int) -> RedesignedProfile:
        """Profile new model cache system."""
        import djinn
        
        logger.info(f"\n{'='*80}")
        logger.info(f"NEW MODEL CACHE SYSTEM - {model_name}")
        logger.info(f"{'='*80}")
        
        # Ensure server is running
        import socket
        import asyncio
        
        server_started = False
        server_manager = None
        server_port = 5556  # Default port
        
        # Use server manager from start_server if available (from main())
        if hasattr(self, 'server_manager') and self.server_manager and hasattr(self.server_manager, 'port'):
            server_port = self.server_manager.port
            logger.info(f"✅ Using server from start_server() on port {server_port}")
        else:
            try:
                # Check if server is already running
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', 5556))
                sock.close()
                
                if result != 0:
                    # Server not running - start it
                    logger.info("🚀 Starting Djinn server...")
                    # ✅ FIX: Import directly to avoid triggering models import
                    import sys
                    from pathlib import Path
                    project_root = Path(__file__).parent.parent.parent
                    sys.path.insert(0, str(project_root))
                    from benchmarks.utils.server_spawner import RemoteServerManager
                    server_manager = RemoteServerManager(host="127.0.0.1", port=5556, timeout=60)
                    if server_manager.start():
                        server_started = True
                        server_port = server_manager.port  # Get actual port (might be 5557, 5558, etc.)
                        self.server_manager = server_manager  # Store for cleanup
                        await asyncio.sleep(2.0)  # Wait for server to be ready
                        logger.info(f"✅ Server started on port {server_port}")
                    else:
                        raise RuntimeError("Failed to start server")
                else:
                    logger.info("✅ Server already running on port 5556")
                    server_port = 5556
            except Exception as e:
                logger.error(f"❌ Server check/start failed: {e}")
                raise
        
        # Initialize Djinn client (use async init in async context)
        # Use actual server port
        server_address = f"localhost:{server_port}"
        from djinn.backend.runtime.initialization import init_async
        init_result = await init_async(server_address=server_address, auto_connect=True)
        if init_result.get('status') != 'success':
            raise RuntimeError(f"Failed to initialize Djinn: {init_result.get('error')}")
        logger.info("✅ Djinn client initialized")
        
        # Load model (from HuggingFace, always on CPU by default)
        model = model_func()
        model.eval()  # Set to eval mode
        
        # ✅ CRITICAL: Move model to remote_accelerator:0 (this is what Djinn intercepts)
        # This is the ONLY non-transparent part of Djinn - users must use 'remote_accelerator:0'
        # instead of 'cuda:0'. Everything else is transparent.
        # ✅ FIX: Do NOT call .to('cuda') first - models should stay on CPU until moved to remote_accelerator
        # HuggingFace models are loaded on CPU by default, so we move directly to remote_accelerator:0
        model = model.to('remote_accelerator:0')
        
        # Initialize enhanced model manager with correct server address
        server_address = f"localhost:{server_port}"
        manager = EnhancedModelManager(server_address=server_address)
        manager.use_model_cache = True
        
        # Register model (one-time)
        logger.info("📝 Registering model...")
        registration_start = time.perf_counter()
        
        try:
            fingerprint = await manager.register_model(model)
            registration_time = (time.perf_counter() - registration_start) * 1000
            logger.info(f"✅ Model registered: {fingerprint} ({registration_time:.2f}ms)")
            
            # ✅ NEW: Explicit initialization (warmup) - optional but recommended
            # This separates fast registration from slow warmup
            init_start = time.perf_counter()
            init_success = await manager.init_model(model)
            init_time = (time.perf_counter() - init_start) * 1000
            if init_success:
                logger.info(f"✅ Model initialized: {fingerprint} ({init_time:.2f}ms)")
            else:
                logger.warning(f"⚠️  Model initialization failed: {fingerprint}")
                init_time = 0.0  # Failed initialization
        except Exception as e:
            logger.error(f"❌ Registration failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Execute model (cached path)
        logger.info("🚀 Executing model via cache...")
        
        # Convert inputs to dict - handle GPT-2 models which expect 'input_ids'
        if isinstance(model, nn.Sequential):
            inputs_dict = {'x': inputs[0]}
        else:
            # For transformers models (GPT-2, etc.), use 'input_ids' key
            inputs_dict = {'input_ids': inputs[0]}
        
        # ✅ OSDI FIX: Measure FIRST execution (cold start) separately
        logger.info("Measuring first execution (cold start)...")
        first_execution_start = time.perf_counter()
        try:
            first_result = await asyncio.wait_for(
                manager.execute_model(model, inputs_dict),
                timeout=120.0
            )
            first_execution_time = (time.perf_counter() - first_execution_start) * 1000
            if hasattr(first_result, 'shape'):
                logger.info(f"✅ First execution (cold start): {first_execution_time:.2f}ms (shape: {first_result.shape})")
            del first_result
        except Exception as e:
            logger.error(f"❌ First execution failed: {e}")
            first_execution_time = 0.0
            raise
        
        # Warmup: Execute model 2 more times to match PyTorch baseline (3 total)
        # This ensures CUDA kernels are fully compiled
        logger.info("Warming up model (2 more runs)...")
        for i in range(2):
            logger.info(f"  Warmup run {i+2}/3...")
            try:
                result = await asyncio.wait_for(
                    manager.execute_model(model, inputs_dict),
                    timeout=120.0
                )
                if hasattr(result, 'shape'):
                    logger.debug(f"  Warmup run {i+2}/3 result shape: {result.shape}")
                del result
                logger.info(f"  ✅ Warmup run {i+2}/3 completed")
            except asyncio.TimeoutError:
                logger.error(f"  ❌ Warmup run {i+2}/3 timed out after 120s")
                raise
            except Exception as e:
                logger.error(f"  ❌ Warmup run {i+2}/3 failed: {type(e).__name__}: {str(e)[:200]}")
                raise
        
        # Enable profiling context
        from djinn.server.profiling_context import ProfilingContext, set_profiler, get_profiler
        profiler = ProfilingContext(enabled=True)
        set_profiler(profiler)
        profiler.start()
        
        execution_start = time.perf_counter()
        
        try:
            # Execute model via cache
            logger.info("Executing model (measured run)...")
            # ✅ FIX: Add timeout to prevent hanging
            result = await asyncio.wait_for(
                manager.execute_model(model, inputs_dict),
                timeout=120.0  # 2 minute timeout for measured run
            )
            execution_time = (time.perf_counter() - execution_start) * 1000
            logger.info(f"Model execution completed in {execution_time:.2f}ms")
            
            # ✅ FIX: Extract shape immediately and prevent tensor from being printed
            if hasattr(result, 'shape'):
                output_shape = tuple(result.shape)
            else:
                output_shape = None
            
            # Extract phase breakdown from profiler
            phase_breakdown = profiler.get_phase_dict() if profiler else {}
            
            # ✅ OSDI FIX: Calculate cold start time
            cold_start_time = registration_time + init_time + first_execution_time
            
            profile = RedesignedProfile(
                mode=ExecutionMode.NEW_MODEL_CACHE.value,
                model_name=model_name,
                batch_size=batch_size,
                registration_time=registration_time,
                registration_breakdown={},  # TODO: Extract from manager if available
                init_time=init_time,
                execution_time=execution_time,
                first_execution_time=first_execution_time,  # ✅ NEW: Store first execution
                cold_start_time=cold_start_time,  # ✅ NEW: Store cold start
                total_time=registration_time + init_time + execution_time,
                phase_breakdown=phase_breakdown,
                model_fingerprint=fingerprint,
                cache_hit=True,
                output_shape=output_shape
            )
            
            logger.info(f"✅ Model cache execution: {execution_time:.2f}ms (output shape: {output_shape})")
            
            # ✅ FIX: Explicitly delete result to prevent any accidental printing
            del result
            
            # Log phase breakdown
            if phase_breakdown:
                logger.info("📊 Phase Breakdown:")
                total_phases = sum(phase_breakdown.values())
                for phase_name, duration_ms in sorted(phase_breakdown.items(), key=lambda x: -x[1]):
                    percentage = (duration_ms / total_phases * 100) if total_phases > 0 else 0
                    logger.info(f"  {phase_name}: {duration_ms:.2f}ms ({percentage:.1f}%)")
            
            return profile
        except Exception as e:
            logger.error(f"❌ Execution failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # ✅ FIX: Clean up manager connections
            if manager is not None:
                try:
                    # Close all connections in the manager
                    if hasattr(manager, '_connection_pool'):
                        async with manager._connection_lock:
                            for target, connections in list(manager._connection_pool.items()):
                                for reader, writer, _, _ in connections:
                                    try:
                                        writer.close()
                                        await writer.wait_closed()
                                    except:
                                        pass
                            manager._connection_pool.clear()
                    logger.debug("Manager connections cleaned up")
                except Exception as e:
                    logger.debug(f"Error cleaning up manager: {e}")
            
            # Stop server if we started it
            if server_started and server_manager is not None:
                try:
                    server_manager.stop()
                    logger.info("✅ Server stopped")
                except Exception as e:
                    logger.warning(f"Error stopping server: {e}")
    
    def profile_pytorch_baseline(self, model_func, inputs: List[torch.Tensor],
                                 model_name: str, batch_size: int, num_runs: int = 5) -> dict:
        """Profile vanilla PyTorch on local GPU (baseline)."""
        import torch
        
        logger.info(f"\n{'='*80}")
        logger.info(f"PYTORCH BASELINE (Local GPU) - {model_name}")
        logger.info(f"{'='*80}")
        
        # ✅ FIX: For PyTorch baseline, use local GPU (not remote_accelerator)
        # This is the baseline comparison - local GPU execution
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # ✅ OSDI FIX: Measure model loading time (cold start component)
        model_load_start = time.perf_counter()
        model = model_func()
        # ✅ FIX: For PyTorch baseline, move to local GPU (this is the baseline)
        model = model.to(device)
        model.eval()
        model_load_time = (time.perf_counter() - model_load_start) * 1000
        logger.info(f"Model loading: {model_load_time:.2f}ms")
        
        # Prepare inputs - handle both Float and Long (token IDs)
        input_tensor = inputs[0].to(device)
        
        # ✅ OSDI FIX: Measure first execution (cold start)
        first_exec_start = time.perf_counter()
        with torch.no_grad():
            if isinstance(model, nn.Sequential):
                _ = model(input_tensor)
            else:
                try:
                    _ = model(input_tensor, use_cache=False)
                except TypeError:
                    _ = model(input_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        first_exec_time = (time.perf_counter() - first_exec_start) * 1000
        pytorch_cold_start = model_load_time + first_exec_time
        logger.info(f"First execution: {first_exec_time:.2f}ms")
        logger.info(f"PyTorch cold start: {pytorch_cold_start:.2f}ms")
        
        # Warmup (2 more runs)
        with torch.no_grad():
            for _ in range(2):
                if isinstance(model, nn.Sequential):
                    _ = model(input_tensor)
                else:
                    try:
                        _ = model(input_tensor, use_cache=False)
                    except TypeError:
                        _ = model(input_tensor)
        
        # Synchronize GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure warm execution
        times = []
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            with torch.no_grad():
                if isinstance(model, nn.Sequential):
                    output = model(input_tensor)
                else:
                    try:
                        output = model(input_tensor, use_cache=False)
                    except TypeError:
                        output = model(input_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            times.append(elapsed_ms)
        
        avg_time = sum(times) / len(times)
        logger.info(f"PyTorch Warm Execution: {avg_time:.2f}ms (avg of {num_runs} runs)")
        logger.info(f"  Min: {min(times):.2f}ms, Max: {max(times):.2f}ms")
        
        # Return both cold start and warm execution
        return {
            'cold_start': pytorch_cold_start,
            'warm_execution': avg_time,
            'model_load': model_load_time,
            'first_execution': first_exec_time
        }
    
    async def run_profiling(self, model_func, inputs: List[torch.Tensor],
                          model_name: str, batch_size: int):
        """Run profiling for PyTorch baseline and redesigned Djinn (model cache)."""
        
        # Profile PyTorch baseline (local GPU)
        pytorch_results = self.profile_pytorch_baseline(model_func, inputs, model_name, batch_size)
        pytorch_cold_start = pytorch_results['cold_start']
        pytorch_warm_exec = pytorch_results['warm_execution']
        
        # Profile redesigned Djinn (model cache system)
        new_profile = await self.profile_new_model_cache(
            model_func, inputs, model_name, batch_size
        )
        
        # ✅ OSDI FIX: Compare both cold start and warm execution
        cold_start_overhead = ((new_profile.cold_start_time - pytorch_cold_start) / pytorch_cold_start * 100) if pytorch_cold_start > 0 else 0
        warm_exec_overhead = ((new_profile.execution_time - pytorch_warm_exec) / pytorch_warm_exec * 100) if pytorch_warm_exec > 0 else 0
        warm_speedup = pytorch_warm_exec / new_profile.execution_time if new_profile.execution_time > 0 else 0
        
        logger.info(f"\n{'='*80}")
        logger.info(f"PERFORMANCE COMPARISON - {model_name}")
        logger.info(f"{'='*80}")
        
        # ✅ OSDI FIX: Show cold start comparison (most important!)
        logger.info(f"COLD START (First Execution):")
        logger.info(f"  PyTorch: {pytorch_cold_start:.2f}ms (load: {pytorch_results['model_load']:.2f}ms + first: {pytorch_results['first_execution']:.2f}ms)")
        logger.info(f"  Djinn:   {new_profile.cold_start_time:.2f}ms (reg: {new_profile.registration_time:.2f}ms + init: {new_profile.init_time:.2f}ms + first: {new_profile.first_execution_time:.2f}ms)")
        logger.info(f"  Overhead: {cold_start_overhead:+.1f}% ({new_profile.cold_start_time/pytorch_cold_start:.1f}x slower)")
        logger.info(f"")
        
        # ✅ OSDI FIX: Show warm execution comparison
        logger.info(f"WARM EXECUTION (Subsequent Executions):")
        logger.info(f"  PyTorch: {pytorch_warm_exec:.2f}ms")
        logger.info(f"  Djinn:   {new_profile.execution_time:.2f}ms")
        logger.info(f"  Overhead: {warm_exec_overhead:+.1f}% ({warm_speedup:.2f}x {'faster' if warm_speedup > 1 else 'slower'})")
        logger.info(f"")
        
        # Show breakdown
        logger.info(f"BREAKDOWN:")
        logger.info(f"  Registration (1x):     {new_profile.registration_time:.2f}ms")
        logger.info(f"  Init/Warmup (1x):      {new_profile.init_time:.2f}ms")
        logger.info(f"  First execution:       {new_profile.first_execution_time:.2f}ms")
        logger.info(f"  Warm execution:       {new_profile.execution_time:.2f}ms")
        
        # Show amortized costs (assuming 1000 requests)
        amortized_registration = new_profile.registration_time / 1000
        amortized_init = new_profile.init_time / 1000
        total_per_request = new_profile.execution_time + amortized_registration + amortized_init
        logger.info(f"")
        logger.info(f"AMORTIZED COST (over 1000 requests):")
        logger.info(f"  Per request: {total_per_request:.3f}ms")
        logger.info(f"    (registration: {amortized_registration:.3f}ms, init: {amortized_init:.3f}ms, exec: {new_profile.execution_time:.2f}ms)")
        
        # Show phase breakdown if available
        if new_profile.phase_breakdown:
            logger.info(f"\n📊 Execution Phase Breakdown:")
            # Extract GPU execution time separately
            gpu_time = new_profile.phase_breakdown.get('server_model_cache_gpu_execution', 0)
            overhead = new_profile.execution_time - gpu_time
            
            logger.info(f"  Pure GPU execution:     {gpu_time:8.2f}ms (vs PyTorch warm: {pytorch_warm_exec:.2f}ms)")
            logger.info(f"  Djinn overhead:        {overhead:8.2f}ms ({overhead/new_profile.execution_time*100:.1f}% of total)")
            
            # Show network components
            network_c2s = new_profile.phase_breakdown.get('model_cache_network_c2s', 0)
            network_s2c = new_profile.phase_breakdown.get('model_cache_network_receive', 0)
            if network_c2s > 0 or network_s2c > 0:
                logger.info(f"    Network: {network_c2s + network_s2c:.2f}ms (C→S: {network_c2s:.2f}ms, S→C: {network_s2c:.2f}ms)")
            
            # Show other phases
            other_phases = {k: v for k, v in new_profile.phase_breakdown.items() 
                          if k not in ['server_model_cache_gpu_execution', 'model_cache_network_c2s', 'model_cache_network_receive']}
            if other_phases:
                logger.info(f"    Other phases:")
                for phase_name, duration_ms in sorted(other_phases.items(), key=lambda x: -x[1]):
                    logger.info(f"      {phase_name:38s}: {duration_ms:6.2f}ms")
        
        # Log optimization statistics
        logger.info(f"\n🔍 Overhead Analysis:")
        gpu_time = new_profile.phase_breakdown.get('server_model_cache_gpu_execution', new_profile.execution_time) if new_profile.phase_breakdown else new_profile.execution_time
        overhead = new_profile.execution_time - gpu_time
        logger.info(f"  Pure GPU: {gpu_time:.2f}ms (vs PyTorch warm: {pytorch_warm_exec:.2f}ms)")
        logger.info(f"  Overhead: {overhead:.2f}ms ({overhead/new_profile.execution_time*100:.1f}% of total)")
        logger.info(f"  Total: {new_profile.execution_time:.2f}ms")
        
        self.profiles.append(new_profile)
        
        return pytorch_results, new_profile
    
    def save_results(self, output_file: str):
        """Save profiling results to JSON."""
        results = {
            'profiles': [asdict(p) for p in self.profiles],
            'summary': {
                'total_profiles': len(self.profiles),
                'new_cache_profiles': len([p for p in self.profiles if p.mode == ExecutionMode.NEW_MODEL_CACHE.value])
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Results saved to {output_file}")


def create_test_model():
    """Create GPT-2-XL model for profiling (tests Phase 1 & 2 optimizations)."""
    # ✅ FIX: Always try to load real GPT-2-XL from HuggingFace first
    # This is the correct model for profiling - no synthetic fallback unless transformers is unavailable
    # Models are loaded on CPU by default (correct for GPU disaggregation)
    try:
        from transformers import GPT2LMHeadModel
        logger.info("Loading GPT-2-XL from HuggingFace for profiling...")
        # ✅ FIX: Load on CPU (default) - do NOT move to GPU here
        # Models should stay on CPU until moved to remote_accelerator:0
        model = GPT2LMHeadModel.from_pretrained('gpt2-xl', trust_remote_code=True).eval()
        logger.info("✅ GPT-2-XL loaded successfully (on CPU, ready for remote_accelerator:0)")
        return model
    except Exception as e:
        logger.warning(f"⚠️  Failed to load GPT-2-XL ({str(e)[:80]}), trying GPT-2-large...")
        try:
            from transformers import GPT2LMHeadModel
            logger.info("Loading GPT-2-large as fallback...")
            model = GPT2LMHeadModel.from_pretrained('gpt2-large', trust_remote_code=True).eval()
            logger.info("✅ GPT-2-large loaded successfully")
            return model
        except Exception as e2:
            logger.warning(f"⚠️  Failed to load GPT-2-large ({str(e2)[:80]}), trying GPT-2-medium...")
            try:
                from transformers import GPT2LMHeadModel
                logger.info("Loading GPT-2-medium as fallback...")
                model = GPT2LMHeadModel.from_pretrained('gpt2-medium', trust_remote_code=True).eval()
                logger.info("✅ GPT-2-medium loaded successfully")
                return model
            except Exception as e3:
                logger.warning(f"⚠️  Failed to load GPT-2-medium ({str(e3)[:80]}), using GPT-2-small...")
                try:
                    from transformers import GPT2LMHeadModel
                    logger.info("Loading GPT-2-small as fallback...")
                    model = GPT2LMHeadModel.from_pretrained('gpt2', trust_remote_code=True).eval()
                    logger.info("✅ GPT-2-small loaded successfully")
                    return model
                except Exception as e4:
                    logger.warning(f"⚠️  All GPT-2 models failed, using synthetic model")
                    # Fallback to synthetic model similar to GPT-2 small
                    # Include embedding layer to accept token IDs (Long dtype)
                    return nn.Sequential(
                        nn.Embedding(50257, 768),  # Token embedding (accepts Long token IDs)
                        nn.Linear(768, 3072),  # Attention projection
                        nn.ReLU(),
                        nn.Linear(3072, 768),  # Output projection
                        nn.LayerNorm(768),
                        nn.Linear(768, 50257)  # Language modeling head
                    )


async def main():
    """Main profiling function."""
    import asyncio
    import os
    
    # ✅ FIX: Set logging level early to reduce noise
    os.environ['GENIE_LOG_LEVEL'] = 'warning'  # Reduce Djinn logging verbosity
    
    # Set timeout before any Djinn initialization
    os.environ['GENIE_OPERATION_TIMEOUT'] = '1800'  # 30 minutes for GPT-2-XL
    
    profiler = RedesignedDjinnProfiler()
    
    try:
        # Start server
        logger.info("🚀 Starting Djinn server for profiling...")
        if not profiler.start_server():
            logger.error("❌ Failed to start server, aborting profiling")
            return
        
        # Wait a bit for server to fully initialize
        await asyncio.sleep(1.0)
        
        # Determine model type and create appropriate inputs
        test_model = create_test_model()
        # Detect model size from model name or parameters
        if "GPT2" in str(type(test_model)):
            # Try to detect model size from config or parameters
            if hasattr(test_model, 'config'):
                config = test_model.config
                if hasattr(config, 'n_embd'):
                    if config.n_embd >= 1600:  # GPT-2-XL has 1600 embedding dim
                        model_name = "GPT-2-XL"
                    elif config.n_embd >= 1280:  # GPT-2-large has 1280
                        model_name = "GPT-2-large"
                    elif config.n_embd >= 1024:  # GPT-2-medium has 1024
                        model_name = "GPT-2-medium"
                    else:
                        model_name = "GPT-2-small"
                else:
                    # Estimate from parameter count
                    total_params = sum(p.numel() for p in test_model.parameters())
                    if total_params > 1_500_000_000:  # > 1.5B params = XL
                        model_name = "GPT-2-XL"
                    elif total_params > 750_000_000:  # > 750M params = large
                        model_name = "GPT-2-large"
                    elif total_params > 350_000_000:  # > 350M params = medium
                        model_name = "GPT-2-medium"
                    else:
                        model_name = "GPT-2-small"
            else:
                model_name = "GPT-2"
        else:
            model_name = "SyntheticGPT2"
        
        # Create inputs appropriate for GPT-2 (token IDs)
        batch_size = 1
        seq_length = 5  # Small sequence for faster testing
        vocab_size = 50257  # GPT-2 vocabulary size
        inputs = [torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long)]
        
        # Create model function
        def model_func():
            return create_test_model()
        
        # Run profiling
        logger.info("📊 Starting profiling...")
        await profiler.run_profiling(
            model_func,
            inputs,
            model_name,
            batch_size=batch_size
        )
        
        # Save results
        logger.info("💾 Saving results...")
        profiler.save_results("redesigned_profiling_results.json")
        logger.info("✅ Profiling completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Profiling interrupted by user")
        raise
    except Exception as e:
        logger.error(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # ✅ FIX: Ensure server is stopped
        logger.info("🛑 Stopping server...")
        profiler.stop_server()
        
        # ✅ FIX: Give a moment for cleanup
        await asyncio.sleep(0.5)
        
        # ✅ FIX: Force cleanup of any remaining tasks
        try:
            # Cancel any remaining tasks
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if tasks:
                logger.debug(f"Cancelling {len(tasks)} remaining tasks...")
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.debug(f"Error during cleanup: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

