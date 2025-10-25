"""
Diagnostic Profiler: Identifies where the 140ms overhead is spent.

This tool profiles each component of the graph capture process to identify
bottlenecks. Run this to get a detailed breakdown of where time is spent.

Usage:
    python benchmarks/diagnostic_profiler.py
    
Output:
    Shows timing breakdown for:
    - Graph construction
    - Metadata annotation
    - Pattern matching
    - Serialization
    - Remote execution
    - Deserialization
"""

import torch
import time
import sys
from pathlib import Path

# Add genie to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genie.profiling import get_detailed_profiler
from benchmarks.workloads_detailed import (
    LLMDecodeWorkload,
    LLMPrefillWorkload,
    VisionCNNWorkload,
)
from benchmarks.baselines import (
    LocalPyTorchBaseline,
    GenieCaptureOnlyBaseline,
)


class DiagnosticProfiler:
    """Profile Genie components to identify bottlenecks."""
    
    def __init__(self):
        self.profiler = get_detailed_profiler()
        self.results = {}
    
    def profile_baseline(self, baseline_name, baseline, workload_name, workload):
        """Profile a single baseline × workload combination."""
        
        print(f"\n{'='*80}")
        print(f"Profiling: {baseline_name} × {workload_name}")
        print(f"{'='*80}")
        
        # Load model
        if hasattr(workload, 'load_model'):
            print("  Loading model...")
            workload.load_model()
        
        # Warm up
        print("  Warming up (3 runs)...")
        for _ in range(3):
            try:
                _ = baseline.run(workload)
            except:
                pass
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Run measurements
        print("  Running measurements (5 runs)...")
        latencies = []
        
        self.profiler.clear()
        
        for run in range(5):
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                result = baseline.run(workload)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                elapsed = (time.perf_counter() - start) * 1000  # ms
                latencies.append(elapsed)
                
                print(f"    Run {run+1}: {elapsed:.1f}ms")
            
            except Exception as e:
                print(f"    Run {run+1}: FAILED ({e})")
        
        # Print results
        import numpy as np
        if latencies:
            print(f"\n  📊 Statistics:")
            print(f"    Mean:       {np.mean(latencies):.1f}ms")
            print(f"    Std Dev:    {np.std(latencies):.1f}ms")
            print(f"    Min:        {np.min(latencies):.1f}ms")
            print(f"    Max:        {np.max(latencies):.1f}ms")
            print(f"    Coefficient of Variation: {(np.std(latencies)/np.mean(latencies))*100:.1f}%")
        
        # Print component breakdown
        print(f"\n  🔍 Component Breakdown:")
        self.profiler.print_summary()
        
        return {
            'baseline': baseline_name,
            'workload': workload_name,
            'latencies': latencies,
        }
    
    def run_full_diagnostic(self):
        """Run diagnostics on all key baselines and workloads."""
        
        print("\n" + "="*80)
        print("🔬 GENIE DIAGNOSTIC PROFILER")
        print("Identifying the 140ms overhead bottleneck")
        print("="*80)
        
        # Test configurations
        configurations = [
            ('local_pytorch', LocalPyTorchBaseline(), 'llm_decode', LLMDecodeWorkload()),
            ('genie_capture', GenieCaptureOnlyBaseline(), 'llm_decode', LLMDecodeWorkload()),
            ('local_pytorch', LocalPyTorchBaseline(), 'llm_prefill', LLMPrefillWorkload()),
            ('genie_capture', GenieCaptureOnlyBaseline(), 'llm_prefill', LLMPrefillWorkload()),
            ('local_pytorch', LocalPyTorchBaseline(), 'vision_cnn', VisionCNNWorkload()),
            ('genie_capture', GenieCaptureOnlyBaseline(), 'vision_cnn', VisionCNNWorkload()),
        ]
        
        for baseline_name, baseline, workload_name, workload in configurations:
            try:
                self.profile_baseline(baseline_name, baseline, workload_name, workload)
            except Exception as e:
                print(f"❌ FAILED: {baseline_name} × {workload_name}: {e}")
        
        # Print summary
        print("\n" + "="*80)
        print("✅ DIAGNOSTIC COMPLETE")
        print("="*80)
        print("\nRecommendations based on profiling:")
        print("1. If lazy_tensor_capture > 50ms: Optimize shape inference (implement caching)")
        print("2. If pattern_matching > 20ms: Implement lazy pattern matching")
        print("3. If metadata_annotation > 10ms: Parallelize metadata annotation")
        print("4. If variance > 20%: Improve warm-up protocol (add GPU pinning)")
        print("\nFor details, see docs/optimization_guide.md")


if __name__ == '__main__':
    diagnostic = DiagnosticProfiler()
    diagnostic.run_full_diagnostic()
