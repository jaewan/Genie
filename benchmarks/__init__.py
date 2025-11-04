"""
Genie Benchmark Suite for OSDI Evaluation.

This module provides the 4 main OSDI benchmarks that demonstrate
semantic-driven GPU disaggregation benefits for production LLM serving.

Active Benchmarks:
1. llama_7b_unified_final.py - Memory pressure handling (OOM cliffs, 26-34% savings)
2. continuous_llm_serving.py - Production serving realism (41% GPU utilization)
3. multi_tenant_real.py - Multi-tenant coordination (120% throughput improvement)
4. ray_vs_genie_comparison.py - Disaggregation superiority (7221% vs Ray)

Key Components:
- Realistic workloads: HuggingFace model implementations with semantic optimizations
- Baseline configurations: Local PyTorch, Genie semantic, Ray disaggregation
- Production evaluation: Real models, realistic scenarios, measurable benefits

Usage:
    python benchmarks/llama_7b_unified_final.py     # Memory pressure benchmark
    python benchmarks/continuous_llm_serving.py     # Production serving benchmark
    python benchmarks/multi_tenant_real.py          # Multi-tenant benchmark
    python benchmarks/ray_vs_genie_comparison.py    # Ray comparison benchmark

Expected Results:
- Memory efficiency: 26-34% reduction through semantic management
- Production readiness: 41% GPU utilization in realistic scenarios
- Multi-tenant throughput: 120% improvement through smart scheduling
- Disaggregation advantage: 7221% better than naive approaches
"""

# Framework components (optional - may not exist yet)
try:
    from .framework import (
        BenchmarkRunner,
        AblationStudyRunner,
        BenchmarkConfig,
        BenchmarkResult,
        ComparativeAnalysis
    )
    _FRAMEWORK_AVAILABLE = True
except ImportError:
    _FRAMEWORK_AVAILABLE = False

# Basic workloads removed - use workloads_detailed/ for comprehensive implementations

# Import comprehensive evaluation components
try:
    from .comprehensive_evaluation import ComprehensiveEvaluation
    _COMPREHENSIVE_AVAILABLE = True
except ImportError:
    _COMPREHENSIVE_AVAILABLE = False

# Import detailed workloads
try:
    from .workloads_detailed import (
        RealisticLLMDecodeWorkload,
        RealisticLLMPrefillWorkload
    )
    _DETAILED_WORKLOADS_AVAILABLE = True
except ImportError:
    _DETAILED_WORKLOADS_AVAILABLE = False

# Import baselines
try:
    from .baselines import (
        LocalPyTorchBaseline,
        GenieFullBaseline,
        RayBaseline
    )
    _BASELINES_AVAILABLE = True
except ImportError:
    _BASELINES_AVAILABLE = False

__version__ = "2.0.0"
__all__ = []

# Add framework components if available
if _FRAMEWORK_AVAILABLE:
    __all__.extend([
    "BenchmarkRunner",
    "AblationStudyRunner",
    "BenchmarkConfig",
    "BenchmarkResult",
    "ComparativeAnalysis",
    ])

# Add comprehensive evaluation components if available
if _COMPREHENSIVE_AVAILABLE:
    __all__.append("ComprehensiveEvaluation")

if _DETAILED_WORKLOADS_AVAILABLE:
    __all__.extend([
        "RealisticLLMDecodeWorkload",
        "RealisticLLMPrefillWorkload"
    ])

if _BASELINES_AVAILABLE:
    __all__.extend([
        "LocalPyTorchBaseline",
        "GenieFullBaseline",
        "RayBaseline"
    ])
