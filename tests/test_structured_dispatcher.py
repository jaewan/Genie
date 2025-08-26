"""
Tests for structured dispatcher have been deprecated in favor of the enhanced
unified dispatcher fallback. Keeping a minimal smoke test to ensure interfaces
still import and basic capture works.
"""
import pytest
import torch

from genie.core.enhanced_dispatcher import enhanced_dispatcher, get_enhanced_stats
from genie.core.lazy_tensor import LazyTensor


def test_basic_capture_via_enhanced_dispatcher():
    # Ensure we are in lazy mode and wrappers exist
    enhanced_dispatcher.set_lazy_mode(True)
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)
    # Even if impls are disabled, wrapper should return LazyTensor
    lt = enhanced_dispatcher._create_lazy_tensor("aten::add", (x, y), {"alpha": 1})
    assert isinstance(lt, LazyTensor)
    assert lt.operation == "aten::add"


def test_enhanced_stats_available():
    stats = get_enhanced_stats()
    assert "registered_ops" in stats
    assert "fallback_ops" in stats
