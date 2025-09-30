"""
Genie exception hierarchy.

All Genie exceptions inherit from GenieException for easy catching.
"""
from typing import TypeVar, Generic, Optional, Callable
from dataclasses import dataclass


class GenieException(Exception):
    """Base exception for all Genie errors."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self):
        base = f"{self.__class__.__name__}: {self.message}"
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{base} (context: {ctx_str})"
        return base


# Semantic layer exceptions
class SemanticException(GenieException):
    """Base for semantic analysis errors."""
    pass


class PatternMatchError(SemanticException):
    """Pattern matching failed."""
    pass


class GraphBuildError(SemanticException):
    """Graph construction failed."""
    pass


class ShapeInferenceError(SemanticException):
    """Shape inference failed."""
    pass


class OptimizationError(SemanticException):
    """Graph optimization failed."""
    pass


class SchedulingError(SemanticException):
    """Scheduling failed."""
    pass


class PlacementError(SemanticException):
    """Device placement failed."""
    pass


# Transport layer exceptions
class TransportException(GenieException):
    """Base for transport errors."""
    pass


class TransferTimeoutError(TransportException):
    """Transfer timed out."""
    pass


class NetworkError(TransportException):
    """Network communication failed."""
    pass


# Execution exceptions
class ExecutionException(GenieException):
    """Base for execution errors."""
    pass


class MaterializationError(ExecutionException):
    """LazyTensor materialization failed."""
    pass


class DeviceError(ExecutionException):
    """Device operation failed."""
    pass


# Configuration exceptions
class ConfigurationError(GenieException):
    """Invalid configuration."""
    pass


# ============================================================================
# Result Type for Operations That May Fail
# ============================================================================

T = TypeVar('T')
U = TypeVar('U')


@dataclass
class Result(Generic[T]):
    """
    Result type for operations that may fail.
    
    Inspired by Rust's Result<T, E> pattern. Forces explicit error handling.
    
    Usage:
        result = risky_operation()
        if result.is_ok:
            value = result.unwrap()
        else:
            handle_error(result.error)
    """
    
    _value: Optional[T] = None
    _error: Optional[Exception] = None
    
    @staticmethod
    def ok(value: T) -> 'Result[T]':
        """Create successful result."""
        return Result(_value=value, _error=None)
    
    @staticmethod
    def err(error: Exception) -> 'Result[T]':
        """Create error result."""
        return Result(_value=None, _error=error)
    
    @property
    def is_ok(self) -> bool:
        """Check if result is successful."""
        return self._error is None
    
    @property
    def is_err(self) -> bool:
        """Check if result is error."""
        return self._error is not None
    
    def unwrap(self) -> T:
        """
        Get value, raising exception if error.
        
        Use when you're certain the result is Ok.
        """
        if self.is_err:
            raise self._error
        return self._value
    
    def unwrap_or(self, default: T) -> T:
        """Get value or return default if error."""
        if self.is_err:
            return default
        return self._value
    
    def unwrap_or_else(self, f: Callable[[Exception], T]) -> T:
        """Get value or compute from error."""
        if self.is_err:
            return f(self._error)
        return self._value
    
    @property
    def error(self) -> Optional[Exception]:
        """Get error if present."""
        return self._error
    
    def map(self, f: Callable[[T], U]) -> 'Result[U]':
        """Transform value if Ok."""
        if self.is_ok:
            try:
                return Result.ok(f(self._value))
            except Exception as e:
                return Result.err(e)
        return Result.err(self._error)
    
    def and_then(self, f: Callable[[T], 'Result[U]']) -> 'Result[U]':
        """Chain Result-returning operations."""
        if self.is_ok:
            try:
                return f(self._value)
            except Exception as e:
                return Result.err(e)
        return Result.err(self._error)


# Convenience function
def try_result(f: Callable[..., T]) -> Callable[..., Result[T]]:
    """
    Decorator to wrap function in Result.
    
    Usage:
        @try_result
        def risky_operation(x: int) -> str:
            if x < 0:
                raise ValueError("Negative")
            return str(x)
        
        result = risky_operation(5)  # Returns Result[str]
    """
    def wrapper(*args, **kwargs) -> Result[T]:
        try:
            value = f(*args, **kwargs)
            return Result.ok(value)
        except Exception as e:
            return Result.err(e)
    return wrapper

