"""Tests for Semantic Enricher (Refactoring #2)."""

import pytest
import torch
from genie.semantic.enricher import SemanticEnricher, get_semantic_enricher
from genie.semantic.metadata_registry import SemanticMetadata


def test_enricher_singleton():
    """Test enricher is reusable."""
    e1 = get_semantic_enricher()
    e2 = get_semantic_enricher()
    assert e1 is e2


def test_enrich_matmul():
    """Test enriching matmul operation."""
    enricher = get_semantic_enricher()
    
    result = enricher.enrich(
        operation="aten::matmul",
        inputs=[torch.randn(10, 10), torch.randn(10, 10)],
        kwargs={},
        shape=torch.Size([10, 10]),
        dtype=torch.float32
    )
    
    assert result.is_ok
    metadata = result.unwrap()
    
    assert metadata.operation_type == "aten::matmul"
    assert metadata.tensor_shape == (10, 10)
    assert metadata.compute_intensity == 10.0  # High for matmul
    assert metadata.can_parallelize is True


def test_enrich_attention_pattern():
    """Test detecting attention patterns."""
    enricher = get_semantic_enricher()
    
    # 3D tensors suggest attention
    result = enricher.enrich(
        operation="aten::matmul",
        inputs=[torch.randn(1, 8, 64), torch.randn(1, 64, 8)],  # Batch, heads, dim
        kwargs={},
        shape=torch.Size([1, 8, 8]),
        dtype=torch.float32
    )
    
    assert result.is_ok
    metadata = result.unwrap()
    assert metadata.semantic_role == "attention_score_computation"


def test_enrich_activation():
    """Test enriching activation functions."""
    enricher = get_semantic_enricher()
    
    result = enricher.enrich(
        operation="aten::relu",
        inputs=[torch.randn(10, 10)],
        kwargs={},
        shape=torch.Size([10, 10]),
        dtype=torch.float32
    )
    
    assert result.is_ok
    metadata = result.unwrap()
    assert metadata.semantic_role == "relu_activation"
    assert metadata.compute_intensity == 1.0  # Low for activations
    assert metadata.is_activation is True


def test_enrich_conv2d():
    """Test enriching convolution."""
    enricher = get_semantic_enricher()
    
    result = enricher.enrich(
        operation="aten::conv2d",
        inputs=[torch.randn(1, 3, 224, 224), torch.randn(64, 3, 7, 7)],
        kwargs={},
        shape=torch.Size([1, 64, 224, 224]),
        dtype=torch.float32
    )
    
    assert result.is_ok
    metadata = result.unwrap()
    assert metadata.operation_type == "aten::conv2d"
    assert metadata.compute_intensity == 8.0
    assert metadata.memory_pattern == "streaming"


def test_enrich_kv_cache():
    """Test detecting KV cache operations."""
    enricher = get_semantic_enricher()
    
    result = enricher.enrich(
        operation="aten::cat",
        inputs=[],
        kwargs={'cache': True, 'past_key_value': (None, None)},
        shape=torch.Size([1, 12, 2048, 64]),
        dtype=torch.float32
    )
    
    assert result.is_ok
    metadata = result.unwrap()
    assert metadata.kv_cache_related is True
    assert metadata.memory_pattern == "persistent"
    assert metadata.colocation_group == "kv_cache"


def test_enrich_normalization():
    """Test enriching normalization layers."""
    enricher = get_semantic_enricher()
    
    result = enricher.enrich(
        operation="aten::layer_norm",
        inputs=[torch.randn(10, 768)],
        kwargs={},
        shape=torch.Size([10, 768]),
        dtype=torch.float32
    )
    
    assert result.is_ok
    metadata = result.unwrap()
    assert metadata.semantic_role == "normalization"


def test_enrich_with_lazy_tensor_inputs():
    """Test enriching with LazyTensor inputs that have metadata."""
    enricher = get_semantic_enricher()
    
    # Create a mock LazyTensor-like object
    class MockLazyTensor:
        def __init__(self):
            self.id = "lt_mock"
            self.shape = torch.Size([10, 10])
    
    mock_input = MockLazyTensor()
    
    result = enricher.enrich(
        operation="aten::add",
        inputs=[mock_input, torch.randn(10, 10)],
        kwargs={},
        shape=torch.Size([10, 10]),
        dtype=torch.float32
    )
    
    assert result.is_ok
    metadata = result.unwrap()
    assert metadata.operation_type == "aten::add"
    
    # Check lineage tracking
    if metadata.data_lineage:
        assert "lt_mock" in metadata.data_lineage['source_tensors']


def test_enrich_priority_calculation():
    """Test priority calculation for different operations."""
    enricher = get_semantic_enricher()
    
    # High priority operation
    result_high = enricher.enrich(
        operation="aten::matmul",
        inputs=[torch.randn(10, 10), torch.randn(10, 10)],
        kwargs={},
        shape=torch.Size([10, 10]),
        dtype=torch.float32
    )
    
    # Low priority operation
    result_low = enricher.enrich(
        operation="aten::reshape",
        inputs=[torch.randn(10, 10)],
        kwargs={},
        shape=torch.Size([100]),
        dtype=torch.float32
    )
    
    assert result_high.is_ok
    assert result_low.is_ok
    
    metadata_high = result_high.unwrap()
    metadata_low = result_low.unwrap()
    
    assert metadata_high.priority > metadata_low.priority


def test_enrich_parallel_capability():
    """Test parallel capability detection."""
    enricher = get_semantic_enricher()
    
    # Parallelizable operation
    result_parallel = enricher.enrich(
        operation="aten::matmul",
        inputs=[torch.randn(10, 10), torch.randn(10, 10)],
        kwargs={},
        shape=torch.Size([10, 10]),
        dtype=torch.float32
    )
    
    # Non-parallelizable operation
    result_sequential = enricher.enrich(
        operation="aten::lstm",
        inputs=[torch.randn(10, 10)],
        kwargs={},
        shape=torch.Size([10, 10]),
        dtype=torch.float32
    )
    
    assert result_parallel.is_ok
    assert result_sequential.is_ok
    
    assert result_parallel.unwrap().can_parallelize is True
    assert result_sequential.unwrap().can_parallelize is False


def test_enrich_error_handling():
    """Test error handling in enrichment."""
    enricher = get_semantic_enricher()
    
    # Even with invalid inputs, should return Result with error, not crash
    result = enricher.enrich(
        operation="aten::unknown_op",
        inputs=[],
        kwargs={},
        shape=None,
        dtype=None
    )
    
    # Should still succeed (enricher is best-effort)
    assert result.is_ok
    metadata = result.unwrap()
    assert metadata.operation_type == "aten::unknown_op"

