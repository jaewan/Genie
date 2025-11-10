"""
Baseline configurations for OSDI evaluation.

This module implements the active baseline configurations used in the
4 main OSDI benchmarks.
"""

from .local_pytorch import LocalPyTorchBaseline
from .djinn_full_semantics import DjinnFullBaseline
from .ray_baseline import RayBaseline

__all__ = [
    'LocalPyTorchBaseline',
    'DjinnFullBaseline',
    'RayBaseline',
]
