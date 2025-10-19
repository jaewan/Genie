class LazyTensorError(Exception):
	"""Base error for LazyTensor-related failures."""
	pass


class MaterializationError(LazyTensorError):
	"""Raised when materialization fails."""
	pass


class UnsupportedOperationError(LazyTensorError):
	"""Raised when an operation is not supported by the executor."""
	pass
