"""
Redesigned Djinn Profiler - Uses Enhanced Model Manager

This profiler uses the new model cache system to measure the actual
performance improvement from caching models server-side.
"""

import torch
import torch.nn as nn
import time
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

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
    OLD_GRAPH = "old_graph"  # Old graph execution (baseline)
    NEW_MODEL_CACHE = "new_model_cache"  # New model cache system


@dataclass
class RedesignedProfile:
    """Profile result for redesigned system."""
    mode: str
    model_name: str
    batch_size: int
    
    # Timing (ms)
    registration_time: float = 0.0  # One-time model registration
    execution_time: float = 0.0  # Model execution
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


class RedesignedDjinnProfiler:
    """Profiler for redesigned Djinn system."""
    
    def __init__(self):
        self.profiles: List[RedesignedProfile] = []
        self.server_process = None
    
    def start_server(self):
        """Start Djinn server."""
        import subprocess
        import os
        import socket
        
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root)
        
        self.server_process = subprocess.Popen(
            [sys.executable, '-m', 'djinn.server.tcp_server'],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(project_root)
        )
        
        # Wait for server to be ready
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', 5556))
                sock.close()
                if result == 0:
                    logger.info("✅ Server started")
                    return True
            except Exception:
                pass
            time.sleep(0.1)
        
        logger.error("❌ Server failed to start")
        return False
    
    def stop_server(self):
        """Stop Djinn server."""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            logger.info("✅ Server stopped")
    
    async def profile_old_graph_execution(self, model_func, inputs: List[torch.Tensor], 
                                          model_name: str, batch_size: int) -> RedesignedProfile:
        """Profile old graph execution (baseline)."""
        import djinn
        
        logger.info(f"\n{'='*80}")
        logger.info(f"OLD GRAPH EXECUTION (Baseline) - {model_name}")
        logger.info(f"{'='*80}")
        
        # Initialize Djinn (use async init in async context)
        from djinn.backend.runtime.initialization import init_async
        await init_async(server_address="localhost:5556", auto_connect=True)
        
        # Load model
        model = model_func()
        model = model.to('remote_accelerator:0')
        
        # Convert inputs
        from djinn.frontend.core.lazy_tensor import LazyTensor
        input_tensor = LazyTensor.tensor(
            data=inputs[0],
            device='remote_accelerator:0',
            dtype=inputs[0].dtype
        )
        
        # Execute via old path
        start_time = time.perf_counter()
        with djinn.capture():
            if isinstance(model, nn.Sequential):
                output = model(input_tensor)
            else:
                output = model(input_tensor, use_cache=False)
        
        # Materialize
        if hasattr(output, 'logits'):
            logits = output.logits
        else:
            logits = output
        
        result = logits.cpu()  # Triggers materialization
        execution_time = (time.perf_counter() - start_time) * 1000
        
        profile = RedesignedProfile(
            mode=ExecutionMode.OLD_GRAPH.value,
            model_name=model_name,
            batch_size=batch_size,
            execution_time=execution_time,
            total_time=execution_time,
            output_shape=tuple(result.shape)
        )
        
        logger.info(f"✅ Old graph execution: {execution_time:.2f}ms")
        return profile
    
    async def profile_new_model_cache(self, model_func, inputs: List[torch.Tensor],
                                     model_name: str, batch_size: int) -> RedesignedProfile:
        """Profile new model cache system."""
        import djinn
        
        logger.info(f"\n{'='*80}")
        logger.info(f"NEW MODEL CACHE SYSTEM - {model_name}")
        logger.info(f"{'='*80}")
        
        # Server should be started by main() or start_server() method
        # Just verify it's running
        from benchmarks.utils.server_spawner import RemoteServerManager
        server_manager = RemoteServerManager(host="127.0.0.1", port=5556, timeout=60)
        server_started = False
        try:
            # Try to connect to verify server is running
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', 5556))
            sock.close()
            if result != 0:
                logger.info("🚀 Starting Djinn server...")
                if server_manager.start():
                    server_started = True
                    await asyncio.sleep(1.0)  # Wait for server to be ready
                    logger.info("✅ Server started")
                else:
                    logger.warning("⚠️  Server already running or failed to start")
            else:
                logger.info("✅ Server already running")
        except Exception as e:
            logger.warning(f"⚠️  Server check failed: {e}, assuming server is running")
        
        # Initialize Djinn (use async init in async context)
        from djinn.backend.runtime.initialization import init_async
        await init_async(server_address="localhost:5556", auto_connect=True)
        
        # Load model
        model = model_func()
        model = model.to('remote_accelerator:0')
        
        # Initialize enhanced model manager
        manager = EnhancedModelManager()
        manager.use_model_cache = True
        
        # Register model (one-time)
        logger.info("📝 Registering model...")
        registration_start = time.perf_counter()
        
        try:
            fingerprint = await manager.register_model(model)
            registration_time = (time.perf_counter() - registration_start) * 1000
            logger.info(f"✅ Model registered: {fingerprint} ({registration_time:.2f}ms)")
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
        
        # Warmup: Execute model 3 times to match PyTorch baseline
        # This ensures CUDA kernels are compiled and model is warmed up
        logger.debug("Warming up model (3 runs)...")
        for _ in range(3):
            await manager.execute_model(model, inputs_dict)
            # Don't measure these runs
        
        # Enable profiling context
        from djinn.server.profiling_context import ProfilingContext, set_profiler, get_profiler
        profiler = ProfilingContext(enabled=True)
        set_profiler(profiler)
        profiler.start()
        
        execution_start = time.perf_counter()
        
        try:
            result = await manager.execute_model(model, inputs_dict)
            execution_time = (time.perf_counter() - execution_start) * 1000
            
            # Extract phase breakdown
            phase_breakdown = profiler.get_phase_dict()
            
            profile = RedesignedProfile(
                mode=ExecutionMode.NEW_MODEL_CACHE.value,
                model_name=model_name,
                batch_size=batch_size,
                registration_time=registration_time,
                execution_time=execution_time,
                total_time=registration_time + execution_time,
                phase_breakdown=phase_breakdown,
                model_fingerprint=fingerprint,
                cache_hit=True,
                output_shape=tuple(result.shape)
            )
            
            logger.info(f"✅ Model cache execution: {execution_time:.2f}ms")
            
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
            # Stop server if we started it
            if server_started:
                server_manager.stop()
                logger.info("✅ Server stopped")
    
    def profile_pytorch_baseline(self, model_func, inputs: List[torch.Tensor],
                                 model_name: str, batch_size: int, num_runs: int = 5) -> float:
        """Profile vanilla PyTorch on local GPU (baseline)."""
        import torch
        
        logger.info(f"\n{'='*80}")
        logger.info(f"PYTORCH BASELINE (Local GPU) - {model_name}")
        logger.info(f"{'='*80}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load model
        model = model_func()
        model = model.to(device)
        model.eval()
        
        # Prepare inputs - handle both Float and Long (token IDs)
        input_tensor = inputs[0].to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                if isinstance(model, nn.Sequential):
                    _ = model(input_tensor)
                else:
                    # Try HuggingFace model signature first, fallback to positional
                    try:
                        _ = model(input_tensor, use_cache=False)
                    except TypeError:
                        _ = model(input_tensor)
        
        # Synchronize GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure
        times = []
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            with torch.no_grad():
                if isinstance(model, nn.Sequential):
                    output = model(input_tensor)
                else:
                    # Try HuggingFace model signature first, fallback to positional
                    try:
                        output = model(input_tensor, use_cache=False)
                    except TypeError:
                        output = model(input_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            times.append(elapsed_ms)
        
        avg_time = sum(times) / len(times)
        logger.info(f"PyTorch Baseline: {avg_time:.2f}ms (avg of {num_runs} runs)")
        logger.info(f"  Min: {min(times):.2f}ms, Max: {max(times):.2f}ms")
        
        return avg_time
    
    async def run_profiling(self, model_func, inputs: List[torch.Tensor],
                          model_name: str, batch_size: int):
        """Run profiling for PyTorch baseline, old system, and new system."""
        
        # Profile PyTorch baseline (local GPU)
        pytorch_time = self.profile_pytorch_baseline(model_func, inputs, model_name, batch_size)
        
        # Profile old system
        old_profile = await self.profile_old_graph_execution(
            model_func, inputs, model_name, batch_size
        )
        
        # Profile new system
        new_profile = await self.profile_new_model_cache(
            model_func, inputs, model_name, batch_size
        )
        
        # Compare
        speedup_vs_old = old_profile.execution_time / new_profile.execution_time if new_profile.execution_time > 0 else 0
        speedup_vs_pytorch = pytorch_time / new_profile.execution_time if new_profile.execution_time > 0 else 0
        overhead_vs_pytorch = ((new_profile.execution_time - pytorch_time) / pytorch_time * 100) if pytorch_time > 0 else 0
        network_reduction = ((old_profile.execution_time - new_profile.execution_time) / old_profile.execution_time * 100) if old_profile.execution_time > 0 else 0
        
        logger.info(f"\n{'='*80}")
        logger.info(f"PERFORMANCE COMPARISON - {model_name}")
        logger.info(f"{'='*80}")
        logger.info(f"PyTorch (Local GPU):      {pytorch_time:.2f}ms")
        logger.info(f"Old Graph Execution:      {old_profile.execution_time:.2f}ms ({old_profile.execution_time/pytorch_time:.1f}x slower)")
        logger.info(f"New Model Cache:")
        logger.info(f"  Registration (1x):     {new_profile.registration_time:.2f}ms")
        logger.info(f"  Execution (cached):    {new_profile.execution_time:.2f}ms")
        logger.info(f"  vs PyTorch:            {speedup_vs_pytorch:.1f}x faster, {overhead_vs_pytorch:+.1f}% overhead")
        logger.info(f"  vs Old System:         {speedup_vs_old:.1f}x faster")
        logger.info(f"  Network reduction:     {network_reduction:.1f}%")
        
        # Show phase breakdown if available
        if new_profile.phase_breakdown:
            logger.info(f"\n📊 Execution Phase Breakdown:")
            total_phases = sum(new_profile.phase_breakdown.values())
            for phase_name, duration_ms in sorted(new_profile.phase_breakdown.items(), key=lambda x: -x[1]):
                percentage = (duration_ms / total_phases * 100) if total_phases > 0 else 0
                logger.info(f"  {phase_name:40s}: {duration_ms:8.2f}ms ({percentage:5.1f}%)")
        
        # Log optimization statistics (phase detection, OOM events)
        logger.info(f"\n🔍 Optimization Statistics:")
        logger.info(f"  GPU Execution: {new_profile.execution_time:.2f}ms")
        logger.info(f"  vs PyTorch Baseline: {pytorch_time:.2f}ms")
        logger.info(f"  Performance: {((pytorch_time - new_profile.execution_time) / pytorch_time * 100):.1f}% {'faster' if new_profile.execution_time < pytorch_time else 'slower'}")
        
        self.profiles.extend([old_profile, new_profile])
        
        return pytorch_time, old_profile, new_profile
    
    def save_results(self, output_file: str):
        """Save profiling results to JSON."""
        results = {
            'profiles': [asdict(p) for p in self.profiles],
            'summary': {
                'total_profiles': len(self.profiles),
                'old_graph_profiles': len([p for p in self.profiles if p.mode == ExecutionMode.OLD_GRAPH.value]),
                'new_cache_profiles': len([p for p in self.profiles if p.mode == ExecutionMode.NEW_MODEL_CACHE.value])
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Results saved to {output_file}")


def create_test_model():
    """Create GPT-2-XL model for realistic profiling (with fallbacks)."""
    # Try to load GPT-2-XL from HuggingFace
    try:
        from transformers import GPT2LMHeadModel
        logger.info("Loading GPT-2-XL for realistic profiling...")
        model = GPT2LMHeadModel.from_pretrained('gpt2-xl', trust_remote_code=True).eval()
        logger.info("✅ GPT-2-XL loaded successfully")
        return model
    except Exception as e:
        logger.warning(f"⚠️  Failed to load GPT-2-XL ({str(e)[:80]}), falling back to GPT-2-medium")
        try:
            from transformers import GPT2LMHeadModel
            model = GPT2LMHeadModel.from_pretrained('gpt2-medium', trust_remote_code=True).eval()
            logger.info("✅ GPT-2-medium loaded as fallback")
            return model
        except Exception as e2:
            logger.warning(f"⚠️  Failed to load any GPT-2 model ({str(e2)[:80]}), using synthetic model")
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
    
    profiler = RedesignedDjinnProfiler()
    
    try:
        # Start server
        if not profiler.start_server():
            return
        
        # Determine model type and create appropriate inputs
        test_model = create_test_model()
        model_name = "GPT-2-XL" if "GPT2" in str(type(test_model)) else "SyntheticGPT2"
        
        # Create inputs appropriate for GPT-2 (token IDs)
        batch_size = 1
        seq_length = 5  # Small sequence for faster testing
        vocab_size = 50257  # GPT-2 vocabulary size
        inputs = [torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long)]
        
        # Create model function
        def model_func():
            return create_test_model()
        
        # Run profiling
        await profiler.run_profiling(
            model_func,
            inputs,
            model_name,
            batch_size=batch_size
        )
        
        # Save results
        profiler.save_results("redesigned_profiling_results.json")
        
    finally:
        profiler.stop_server()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

