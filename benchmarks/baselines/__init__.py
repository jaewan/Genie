"""
Baseline configurations for OSDI evaluation.

This module implements the active baseline configurations used in the
4 main OSDI benchmarks.
"""

from .local_pytorch import LocalPyTorchBaseline
from .genie_full_semantics import GenieFullBaseline
from .ray_baseline import RayBaseline

__all__ = [
    'LocalPyTorchBaseline',
    'GenieFullBaseline',
    'RayBaseline',
]
