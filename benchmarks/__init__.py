"""
Genie Benchmark Suite for OSDI Evaluation.

This module provides a comprehensive benchmarking framework that proves
semantic awareness in accelerator disaggregation provides performance benefits.

Key Components:
- BenchmarkRunner: Unified experiment runner with statistical analysis
- AblationStudyRunner: Feature ablation studies to isolate impact
- Detailed workloads: HuggingFace model implementations with semantic optimizations
- Baseline configurations: 7 different baselines from best to worst performance
- Comprehensive evaluation: 35 configurations (7 baselines × 5 workloads)
- Result generation: Paper-ready tables and figures

Usage:
    python3 test_comprehensive.py                # Run test suite
    python -m benchmarks.comprehensive_evaluation  # Full OSDI evaluation

Expected Results:
- LLM co-location: 3-5x speedup, 90%+ network reduction
- Multimodal parallel: 50%+ speedup from parallelization
- Cost model validation: >0.7 correlation with reality
- Ablation studies: Quantify impact of each semantic feature
"""

from .framework import (
    BenchmarkRunner,
    AblationStudyRunner,
    BenchmarkConfig,
    BenchmarkResult,
    ComparativeAnalysis
)

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
        LLMDecodeWorkload,
        LLMPrefillWorkload,
        VisionCNNWorkload,
        MultimodalVQAWorkload,
        MicrobenchmarkWorkload
    )
    _DETAILED_WORKLOADS_AVAILABLE = True
except ImportError:
    _DETAILED_WORKLOADS_AVAILABLE = False

# Import baselines
try:
    from .baselines import (
        LocalPyTorchBaseline,
        GenieCaptureOnlyBaseline,
        GenieLocalRemoteBaseline,
        GenieNoSemanticsBaseline,
        GenieFullBaseline,
        PyTorchRPCBaseline,
        RayBaseline
    )
    _BASELINES_AVAILABLE = True
except ImportError:
    _BASELINES_AVAILABLE = False

__version__ = "2.0.0"
__all__ = [
    # Framework
    "BenchmarkRunner",
    "AblationStudyRunner",
    "BenchmarkConfig",
    "BenchmarkResult",
    "ComparativeAnalysis",
]

# Add comprehensive evaluation components if available
if _COMPREHENSIVE_AVAILABLE:
    __all__.append("ComprehensiveEvaluation")

if _DETAILED_WORKLOADS_AVAILABLE:
    __all__.extend([
        "LLMDecodeWorkload",
        "LLMPrefillWorkload",
        "VisionCNNWorkload",
        "MultimodalVQAWorkload",
        "MicrobenchmarkWorkload"
    ])

if _BASELINES_AVAILABLE:
    __all__.extend([
        "LocalPyTorchBaseline",
        "GenieCaptureOnlyBaseline",
        "GenieLocalRemoteBaseline",
        "GenieNoSemanticsBaseline",
        "GenieFullBaseline",
        "PyTorchRPCBaseline",
        "RayBaseline"
    ])
