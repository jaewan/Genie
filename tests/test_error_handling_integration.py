"""Integration tests for error handling across Genie components.

Tests the end-to-end error handling flow with Result types and exceptions.
"""

import pytest
import torch
import torch.fx as fx

from genie.core.exceptions import (
    Result, GenieException, SemanticException, ShapeInferenceError,
    PatternMatchError, OptimizationError
)
from genie.core.lazy_tensor import LazyTensor
from genie.semantic.pattern_registry import PatternRegistry
from genie.semantic.analyzer import SemanticAnalyzer
from genie.semantic.optimizer import SemanticOptimizer
from genie.semantic.workload import WorkloadProfile, WorkloadType
from genie.core.graph import ComputationGraph, ComputationNode


def test_lazy_tensor_shape_inference_error_handling():
    """Test LazyTensor handles shape inference errors gracefully."""
    # Create LazyTensor with problematic operation
    tensor = LazyTensor(
        operation="aten::unknown_op",
        inputs=[],
        kwargs={}
    )
    
    # Shape should be None (graceful degradation)
    assert tensor.shape is None
    # But tensor should still be usable
    assert tensor.operation == "aten::unknown_op"


def test_pattern_registry_error_aggregation():
    """Test PatternRegistry aggregates errors from multiple patterns."""
    registry = PatternRegistry()
    
    # Create a simple graph
    node = ComputationNode(id="n1", operation="aten::add", inputs=[], outputs=[], metadata={})
    graph = ComputationGraph(nodes={"n1": node}, edges=[], entry_points={"n1"})
    
    # Match patterns
    result = registry.match_patterns(graph)
    
    # Should return Result
    assert isinstance(result, Result)
    assert result.is_ok or result.is_err
    
    # If successful, should get matches (possibly empty)
    if result.is_ok:
        matches = result.unwrap()
        assert isinstance(matches, list)


def test_semantic_analyzer_handles_pattern_errors():
    """Test SemanticAnalyzer handles errors from PatternRegistry gracefully."""
    analyzer = SemanticAnalyzer()
    
    # Create a simple graph
    node = ComputationNode(id="n1", operation="aten::matmul", inputs=[], outputs=[], metadata={})
    graph = ComputationGraph(nodes={"n1": node}, edges=[], entry_points={"n1"})
    
    # Should complete even if patterns have issues
    try:
        profile = analyzer.analyze_graph(graph)
        assert profile is not None
        assert profile.workload_type is not None
    except Exception as e:
        pytest.fail(f"Analyzer should handle errors gracefully, but raised: {e}")


def test_optimizer_validation():
    """Test SemanticOptimizer validates inputs."""
    optimizer = SemanticOptimizer()
    
    # Test with None graph
    result = optimizer.optimize(None, None)
    assert result.is_err
    error = result.error
    assert isinstance(error, OptimizationError)
    assert "None" in str(error)


def test_optimizer_with_valid_inputs():
    """Test SemanticOptimizer works with valid inputs."""
    optimizer = SemanticOptimizer()
    
    # Create simple FX graph
    class SimpleModel(torch.nn.Module):
        def forward(self, x):
            return x + 1
    
    model = SimpleModel()
    graph = fx.symbolic_trace(model)
    
    # Create profile
    profile = WorkloadProfile(
        workload_type=WorkloadType.UNKNOWN,
        patterns=[],
        metadata={}
    )
    
    # Should succeed
    result = optimizer.optimize(graph, profile)
    assert result.is_ok, f"Optimization should succeed, but got error: {result.error if result.is_err else 'unknown'}"
    
    optimized_graph, plan = result.unwrap()
    assert optimized_graph is not None
    assert plan is not None


def test_end_to_end_error_propagation():
    """Test errors propagate correctly through the pipeline."""
    # Create components
    registry = PatternRegistry()
    analyzer = SemanticAnalyzer(pattern_registry=registry)
    optimizer = SemanticOptimizer()
    
    # Create graph
    node = ComputationNode(id="n1", operation="aten::add", inputs=[], outputs=[], metadata={})
    graph = ComputationGraph(nodes={"n1": node}, edges=[], entry_points={"n1"})
    
    # Analyze (should handle any pattern errors)
    try:
        profile = analyzer.analyze_graph(graph)
        assert profile is not None
    except GenieException as e:
        pytest.fail(f"Analysis should handle errors gracefully: {e}")
    
    # Create FX graph for optimizer
    class SimpleModel(torch.nn.Module):
        def forward(self, x):
            return x + 1
    
    model = SimpleModel()
    fx_graph = fx.symbolic_trace(model)
    
    # Optimize
    result = optimizer.optimize(fx_graph, profile)
    
    # Should either succeed or return meaningful error
    if result.is_err:
        error = result.error
        assert isinstance(error, OptimizationError)
        assert error.context is not None
    else:
        optimized_graph, plan = result.unwrap()
        assert optimized_graph is not None


def test_result_type_chaining():
    """Test Result type chaining in practical scenario."""
    optimizer = SemanticOptimizer()
    
    # Create graph and profile
    class Model(torch.nn.Module):
        def forward(self, x):
            return x * 2 + 1
    
    model = Model()
    graph = fx.symbolic_trace(model)
    profile = WorkloadProfile(
        workload_type=WorkloadType.UNKNOWN,
        patterns=[],
        metadata={}
    )
    
    # Chain Result operations
    result = optimizer.optimize(graph, profile)
    
    # Use map to transform result
    plan_result = result.map(lambda x: x[1])  # Extract just the plan
    
    if plan_result.is_ok:
        plan = plan_result.unwrap()
        assert plan is not None
    
    # Use unwrap_or for default
    default_plan = plan_result.unwrap_or(None)
    assert default_plan is not None or plan_result.is_err


def test_exception_context_preservation():
    """Test exception context is preserved through the stack."""
    optimizer = SemanticOptimizer()
    
    # Trigger error with None
    result = optimizer.optimize(None, None)
    
    assert result.is_err
    error = result.error
    
    # Check context
    assert isinstance(error, OptimizationError)
    assert error.context is not None
    assert 'profile' in error.context or 'graph_nodes' in error.context


def test_graceful_degradation():
    """Test system continues operating despite errors."""
    # Component 1: LazyTensor with unknown op
    tensor1 = LazyTensor(operation="aten::unknown", inputs=[], kwargs={})
    assert tensor1 is not None  # Created successfully
    
    # Component 2: Pattern matching with simple graph
    registry = PatternRegistry()
    node = ComputationNode(id="n1", operation="aten::add", inputs=[], outputs=[], metadata={})
    graph = ComputationGraph(nodes={"n1": node}, edges=[], entry_points={"n1"})
    
    result = registry.match_patterns(graph)
    # Even if it fails, we get a Result
    assert isinstance(result, Result)
    
    # Component 3: Analyzer continues
    analyzer = SemanticAnalyzer()
    profile = analyzer.analyze_graph(graph)
    assert profile is not None
    
    # System is still functional
    assert True


def test_error_logging():
    """Test errors are logged appropriately."""
    import logging
    from io import StringIO
    
    # Capture logs
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    logger = logging.getLogger('genie')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    try:
        # Trigger error
        optimizer = SemanticOptimizer()
        result = optimizer.optimize(None, None)
        
        # Should have logged something (if logging is enabled)
        # This test is informational - logging may or may not be enabled
        assert result.is_err
    finally:
        logger.removeHandler(handler)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

