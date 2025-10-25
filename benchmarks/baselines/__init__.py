"""
Baseline configurations for comprehensive OSDI evaluation.

This module implements the 7 baseline configurations from best to worst
expected performance to prove semantic awareness matters.
"""

from .local_pytorch import LocalPyTorchBaseline
from .genie_capture_only import GenieCaptureOnlyBaseline
from .genie_local_remote import GenieLocalRemoteBaseline
from .genie_no_semantics import GenieNoSemanticsBaseline
from .genie_full_semantics import GenieFullBaseline
from .pytorch_rpc import PyTorchRPCBaseline
from .ray_baseline import RayBaseline

__all__ = [
    'LocalPyTorchBaseline',
    'GenieCaptureOnlyBaseline',
    'GenieLocalRemoteBaseline',
    'GenieNoSemanticsBaseline',
    'GenieFullBaseline',
    'PyTorchRPCBaseline',
    'RayBaseline'
]
