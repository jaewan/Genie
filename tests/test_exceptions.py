"""
Tests for Genie exception hierarchy and Result type.

Tests the error handling consolidation from Refactoring #1.
"""
import pytest
from genie.core.exceptions import (
    GenieException,
    SemanticException,
    PatternMatchError,
    GraphBuildError,
    ShapeInferenceError,
    TransportException,
    TransferTimeoutError,
    NetworkError,
    ExecutionException,
    MaterializationError,
    DeviceError,
    ConfigurationError,
    Result,
    try_result,
)


# ============================================================================
# Exception Hierarchy Tests
# ============================================================================

def test_exception_hierarchy():
    """Test exception inheritance."""
    assert issubclass(SemanticException, GenieException)
    assert issubclass(PatternMatchError, SemanticException)
    assert issubclass(PatternMatchError, GenieException)
    assert issubclass(TransportException, GenieException)
    assert issubclass(ExecutionException, GenieException)


def test_exception_context():
    """Test exception context tracking."""
    error = PatternMatchError(
        "Failed to match pattern",
        context={'pattern_name': 'llm', 'graph_nodes': 150}
    )
    
    assert error.message == "Failed to match pattern"
    assert error.context['pattern_name'] == 'llm'
    assert error.context['graph_nodes'] == 150
    assert 'pattern_name=llm' in str(error)
    assert 'graph_nodes=150' in str(error)


def test_exception_without_context():
    """Test exception without context."""
    error = ShapeInferenceError("Could not infer shape")
    
    assert error.message == "Could not infer shape"
    assert error.context == {}
    assert str(error) == "ShapeInferenceError: Could not infer shape"


def test_catch_all_genie_exceptions():
    """Test catching all Genie exceptions with base class."""
    try:
        raise PatternMatchError("test")
    except GenieException as e:
        assert isinstance(e, GenieException)
        assert isinstance(e, SemanticException)
        assert isinstance(e, PatternMatchError)


def test_catch_specific_exception():
    """Test catching specific exception types."""
    with pytest.raises(TransferTimeoutError):
        raise TransferTimeoutError("Transfer timed out")
    
    with pytest.raises(MaterializationError):
        raise MaterializationError("Materialization failed")


def test_exception_types_separate():
    """Test different exception types are separate hierarchies."""
    semantic_error = PatternMatchError("test")
    transport_error = NetworkError("test")
    execution_error = DeviceError("test")
    
    assert isinstance(semantic_error, SemanticException)
    assert not isinstance(semantic_error, TransportException)
    assert not isinstance(semantic_error, ExecutionException)
    
    assert isinstance(transport_error, TransportException)
    assert not isinstance(transport_error, SemanticException)
    
    assert isinstance(execution_error, ExecutionException)
    assert not isinstance(execution_error, SemanticException)


# ============================================================================
# Result Type Tests
# ============================================================================

def test_result_ok():
    """Test successful result."""
    result = Result.ok(42)
    
    assert result.is_ok
    assert not result.is_err
    assert result.unwrap() == 42
    assert result.error is None


def test_result_err():
    """Test error result."""
    error = ValueError("test error")
    result = Result.err(error)
    
    assert result.is_err
    assert not result.is_ok
    assert result.error == error


def test_result_unwrap_raises():
    """Test unwrap raises on error."""
    result = Result.err(ValueError("test"))
    
    with pytest.raises(ValueError, match="test"):
        result.unwrap()


def test_result_unwrap_or():
    """Test unwrap_or returns default on error."""
    result = Result.err(ValueError("test"))
    assert result.unwrap_or(99) == 99
    
    # Ok case returns value, not default
    result_ok = Result.ok(42)
    assert result_ok.unwrap_or(99) == 42


def test_result_unwrap_or_else():
    """Test unwrap_or_else computes from error."""
    result = Result.err(ValueError("test"))
    
    def handle_error(e):
        return f"Error: {e}"
    
    value = result.unwrap_or_else(handle_error)
    assert value == "Error: test"


def test_result_map():
    """Test mapping over result."""
    result = Result.ok(5)
    mapped = result.map(lambda x: x * 2)
    
    assert mapped.is_ok
    assert mapped.unwrap() == 10


def test_result_map_error():
    """Test map on error result."""
    result = Result.err(ValueError("test"))
    mapped = result.map(lambda x: x * 2)
    
    assert mapped.is_err
    assert isinstance(mapped.error, ValueError)


def test_result_map_raises():
    """Test map that raises becomes error."""
    result = Result.ok(5)
    
    def divide_by_zero(x):
        return x / 0
    
    mapped = result.map(divide_by_zero)
    
    assert mapped.is_err
    assert isinstance(mapped.error, ZeroDivisionError)


def test_result_and_then():
    """Test chaining Result-returning operations."""
    def safe_divide(x: int, y: int) -> Result[float]:
        if y == 0:
            return Result.err(ValueError("Division by zero"))
        return Result.ok(x / y)
    
    # Success case
    result = Result.ok(10)
    chained = result.and_then(lambda x: safe_divide(x, 2))
    
    assert chained.is_ok
    assert chained.unwrap() == 5.0
    
    # Error in chain
    result = Result.ok(10)
    chained = result.and_then(lambda x: safe_divide(x, 0))
    
    assert chained.is_err
    assert isinstance(chained.error, ValueError)
    
    # Error propagates
    result = Result.err(TypeError("test"))
    chained = result.and_then(lambda x: safe_divide(x, 2))
    
    assert chained.is_err
    assert isinstance(chained.error, TypeError)


def test_try_result_decorator():
    """Test try_result decorator."""
    @try_result
    def divide(a: int, b: int) -> float:
        return a / b
    
    # Success case
    result = divide(10, 2)
    assert result.is_ok
    assert result.unwrap() == 5.0
    
    # Error case
    result = divide(10, 0)
    assert result.is_err
    assert isinstance(result.error, ZeroDivisionError)


def test_try_result_with_custom_exception():
    """Test try_result with custom exceptions."""
    @try_result
    def validate_positive(x: int) -> int:
        if x < 0:
            raise ValueError("Must be positive")
        return x * 2
    
    # Valid input
    result = validate_positive(5)
    assert result.is_ok
    assert result.unwrap() == 10
    
    # Invalid input
    result = validate_positive(-1)
    assert result.is_err
    assert isinstance(result.error, ValueError)
    assert "Must be positive" in str(result.error)


def test_result_chaining_complex():
    """Test complex Result chaining scenario."""
    @try_result
    def parse_int(s: str) -> int:
        return int(s)
    
    @try_result
    def double(x: int) -> int:
        return x * 2
    
    @try_result
    def validate_range(x: int) -> int:
        if x > 100:
            raise ValueError("Too large")
        return x
    
    # Success path
    result = (
        parse_int("5")
        .and_then(lambda x: double(x))
        .and_then(lambda x: validate_range(x))
    )
    
    assert result.is_ok
    assert result.unwrap() == 10
    
    # Fail at parsing
    result = (
        parse_int("not_a_number")
        .and_then(lambda x: double(x))
        .and_then(lambda x: validate_range(x))
    )
    
    assert result.is_err
    assert isinstance(result.error, ValueError)
    
    # Fail at validation
    result = (
        parse_int("100")
        .and_then(lambda x: double(x))
        .and_then(lambda x: validate_range(x))
    )
    
    assert result.is_err
    assert "Too large" in str(result.error)


def test_result_with_genie_exceptions():
    """Test Result type works with Genie exceptions."""
    def risky_semantic_operation() -> Result[str]:
        try:
            # Simulate some operation
            raise PatternMatchError(
                "Pattern not found",
                context={'pattern': 'llm', 'confidence': 0.3}
            )
        except GenieException as e:
            return Result.err(e)
    
    result = risky_semantic_operation()
    
    assert result.is_err
    assert isinstance(result.error, PatternMatchError)
    assert isinstance(result.error, SemanticException)
    assert isinstance(result.error, GenieException)
    assert result.error.context['pattern'] == 'llm'


def test_result_practical_usage():
    """Test Result in practical scenario."""
    def safe_operation(value: int) -> Result[str]:
        """Simulates a complex operation that may fail."""
        if value < 0:
            return Result.err(ValueError("Negative values not allowed"))
        
        if value == 0:
            return Result.err(ZeroDivisionError("Cannot divide by zero"))
        
        result_value = f"Success: {100 / value}"
        return Result.ok(result_value)
    
    # Test all paths
    result = safe_operation(10)
    assert result.is_ok
    assert "Success: 10.0" in result.unwrap()
    
    result = safe_operation(-5)
    assert result.is_err
    assert isinstance(result.error, ValueError)
    
    result = safe_operation(0)
    assert result.is_err
    assert isinstance(result.error, ZeroDivisionError)
    
    # Use unwrap_or for default
    result = safe_operation(-5)
    value = result.unwrap_or("Default value")
    assert value == "Default value"


# ============================================================================
# Integration Tests
# ============================================================================

def test_exception_in_try_catch():
    """Test exceptions work with standard try/catch."""
    try:
        raise ShapeInferenceError(
            "Shape mismatch",
            context={'expected': (10, 10), 'got': (5, 5)}
        )
    except SemanticException as e:
        assert isinstance(e, ShapeInferenceError)
        assert 'expected' in e.context
        assert 'got' in e.context
    except Exception:
        pytest.fail("Should have caught as SemanticException")


def test_result_error_propagation():
    """Test error propagation through Result chain."""
    def step1() -> Result[int]:
        return Result.ok(10)
    
    def step2(x: int) -> Result[int]:
        if x < 20:
            return Result.err(ValueError("Too small"))
        return Result.ok(x * 2)
    
    def step3(x: int) -> Result[str]:
        return Result.ok(f"Final: {x}")
    
    # Chain that fails at step2
    result = (
        step1()
        .and_then(step2)
        .and_then(step3)
    )
    
    assert result.is_err
    assert isinstance(result.error, ValueError)
    assert "Too small" in str(result.error)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

