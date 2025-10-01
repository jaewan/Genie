"""
Simple test runner for exception hierarchy without pytest dependency.

Tests the error handling consolidation from Refactoring #1.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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


def test_exception_hierarchy():
    """Test exception inheritance."""
    assert issubclass(SemanticException, GenieException)
    assert issubclass(PatternMatchError, SemanticException)
    assert issubclass(PatternMatchError, GenieException)
    assert issubclass(TransportException, GenieException)
    assert issubclass(ExecutionException, GenieException)
    print("✓ Exception hierarchy test passed")


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
    print("✓ Exception context test passed")


def test_exception_without_context():
    """Test exception without context."""
    error = ShapeInferenceError("Could not infer shape")
    
    assert error.message == "Could not infer shape"
    assert error.context == {}
    assert str(error) == "ShapeInferenceError: Could not infer shape"
    print("✓ Exception without context test passed")


def test_catch_all_genie_exceptions():
    """Test catching all Genie exceptions with base class."""
    try:
        raise PatternMatchError("test")
    except GenieException as e:
        assert isinstance(e, GenieException)
        assert isinstance(e, SemanticException)
        assert isinstance(e, PatternMatchError)
    print("✓ Catch all Genie exceptions test passed")


def test_result_ok():
    """Test successful result."""
    result = Result.ok(42)
    
    assert result.is_ok
    assert not result.is_err
    assert result.unwrap() == 42
    assert result.error is None
    print("✓ Result.ok test passed")


def test_result_err():
    """Test error result."""
    error = ValueError("test error")
    result = Result.err(error)
    
    assert result.is_err
    assert not result.is_ok
    assert result.error == error
    print("✓ Result.err test passed")


def test_result_unwrap_raises():
    """Test unwrap raises on error."""
    result = Result.err(ValueError("test"))
    
    try:
        result.unwrap()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "test"
    print("✓ Result unwrap raises test passed")


def test_result_unwrap_or():
    """Test unwrap_or returns default on error."""
    result = Result.err(ValueError("test"))
    assert result.unwrap_or(99) == 99
    
    # Ok case returns value, not default
    result_ok = Result.ok(42)
    assert result_ok.unwrap_or(99) == 42
    print("✓ Result unwrap_or test passed")


def test_result_map():
    """Test mapping over result."""
    result = Result.ok(5)
    mapped = result.map(lambda x: x * 2)
    
    assert mapped.is_ok
    assert mapped.unwrap() == 10
    print("✓ Result map test passed")


def test_result_map_error():
    """Test map on error result."""
    result = Result.err(ValueError("test"))
    mapped = result.map(lambda x: x * 2)
    
    assert mapped.is_err
    assert isinstance(mapped.error, ValueError)
    print("✓ Result map error test passed")


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
    print("✓ try_result decorator test passed")


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
    print("✓ Result and_then test passed")


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
    print("✓ Result with Genie exceptions test passed")


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
    print("✓ Exception types separate test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running Exception Hierarchy Tests")
    print("="*60 + "\n")
    
    tests = [
        test_exception_hierarchy,
        test_exception_context,
        test_exception_without_context,
        test_catch_all_genie_exceptions,
        test_exception_types_separate,
        test_result_ok,
        test_result_err,
        test_result_unwrap_raises,
        test_result_unwrap_or,
        test_result_map,
        test_result_map_error,
        test_try_result_decorator,
        test_result_and_then,
        test_result_with_genie_exceptions,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} raised unexpected exception: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
