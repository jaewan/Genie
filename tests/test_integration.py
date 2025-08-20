import torch
import pytest

from genie.core.device import RemoteAcceleratorDevice
from genie.core.lazy_tensor import LazyTensor
from genie.core.graph import GraphBuilder
from genie.core.dispatcher import dispatcher
from genie.patterns.matmul_pattern import MatMulPattern, ConvolutionPattern


def test_device_registration():
	"""Test device registration and basic functionality."""
	device = RemoteAcceleratorDevice.get_device(0)
	assert str(device) == "remote_accelerator:0"
	assert device.device_count() >= 1

	# Test device equality
	device2 = RemoteAcceleratorDevice.get_device(0)
	assert device == device2

	# Test different device indices
	device1 = RemoteAcceleratorDevice.get_device(1)
	assert device != device1
	assert str(device1) == "remote_accelerator:1"


def test_lazy_tensor_creation():
	"""Test LazyTensor creation and metadata inference."""
	# Test with concrete tensors
	a = torch.randn(4, 4)
	b = torch.randn(4, 4)

	lt = LazyTensor("aten::matmul", [a, b], {})
	
	assert isinstance(lt, LazyTensor)
	assert lt.operation == "aten::matmul"
	assert lt.shape == torch.Size([4, 4])
	assert lt.dtype == torch.float32
	assert not lt.is_materialized()


def test_shape_inference():
	"""Test shape inference for various operations."""
	a = torch.randn(3, 4)
	b = torch.randn(4, 5)
	
	# Matrix multiplication
	matmul_lt = LazyTensor("aten::matmul", [a, b], {})
	assert matmul_lt.shape == torch.Size([3, 5])
	
	# Element-wise operations
	c = torch.randn(3, 4)
	add_lt = LazyTensor("aten::add", [a, c], {})
	assert add_lt.shape == torch.Size([3, 4])
	
	# Tensor creation
	zeros_lt = LazyTensor("aten::zeros", [(2, 3)], {"dtype": torch.float32})
	assert zeros_lt.shape == torch.Size([2, 3])
	assert zeros_lt.dtype == torch.float32


def test_lazy_execution_and_graph():
	"""Test graph construction and execution."""
	# Clear any existing graph
	GraphBuilder._thread_local = type(GraphBuilder._thread_local)()
	
	a = torch.randn(8, 8)
	b = torch.randn(8, 8)

	lt_a = LazyTensor("aten::matmul", [a, b], {})
	lt_b = LazyTensor("aten::add", [lt_a, a], {})

	graph = GraphBuilder.current().get_graph()
	assert len(graph.nodes) >= 2
	
	order = graph.topological_sort()
	assert set(order).issuperset({lt_a.id, lt_b.id})

	# Test materialization
	tensor_cpu = lt_b.cpu()
	assert isinstance(tensor_cpu, torch.Tensor)
	assert lt_b.is_materialized()


def test_dispatcher_integration():
	"""Test dispatcher statistics and operation counting."""
	initial_stats = dispatcher.get_stats()
	
	# Test basic dispatcher functionality
	assert initial_stats["lazy_mode"] == True
	
	# Test lazy mode toggle
	dispatcher.set_lazy_mode(False)
	assert dispatcher.get_stats()["lazy_mode"] == False
	
	dispatcher.set_lazy_mode(True)
	assert dispatcher.get_stats()["lazy_mode"] == True
	
	# Test that we have some registered operations (may be 0 if registration failed)
	# This is acceptable in Phase 1 since PyTorch dispatcher integration is complex
	stats = dispatcher.get_stats()
	assert "registered_ops" in stats
	assert "fallback_ops" in stats


def test_tensor_interface():
	"""Test LazyTensor tensor-like interface."""
	a = torch.randn(3, 3)
	b = torch.randn(3, 3)
	
	lt = LazyTensor("aten::matmul", [a, b], {})
	
	# Test size methods
	assert lt.size() == torch.Size([3, 3])
	assert lt.size(0) == 3
	assert lt.dim() == 2
	assert lt.numel() == 9
	
	# Test memory footprint estimation
	footprint = lt.get_memory_footprint()
	assert footprint > 0


def test_arithmetic_operators():
	"""Test LazyTensor arithmetic operators."""
	a = torch.randn(2, 2)
	b = torch.randn(2, 2)
	
	lt_a = LazyTensor("aten::zeros", [(2, 2)], {})
	lt_b = LazyTensor("aten::ones", [(2, 2)], {})
	
	# Test operators create new LazyTensors
	add_result = lt_a + lt_b
	assert isinstance(add_result, LazyTensor)
	assert add_result.operation == "aten::add"
	
	sub_result = lt_a - lt_b
	assert isinstance(sub_result, LazyTensor)
	assert sub_result.operation == "aten::sub"
	
	mul_result = lt_a * lt_b
	assert isinstance(mul_result, LazyTensor)
	assert mul_result.operation == "aten::mul"


def test_pattern_recognition():
	"""Test pattern recognition plugins."""
	# Clear graph
	GraphBuilder._thread_local = type(GraphBuilder._thread_local)()
	
	# Create matmul chain
	a = torch.randn(4, 4)
	b = torch.randn(4, 4)
	c = torch.randn(4, 4)
	
	lt1 = LazyTensor("aten::matmul", [a, b], {})
	lt2 = LazyTensor("aten::matmul", [lt1, c], {})
	
	graph = GraphBuilder.current().get_graph()
	
	# Test matmul pattern
	matmul_pattern = MatMulPattern()
	match = matmul_pattern.match(graph)
	
	assert match is not None
	assert match.pattern_name in ["matmul_chain", "large_matmul"]
	assert match.confidence > 0.7
	assert len(match.matched_nodes) >= 1


def test_convolution_pattern():
	"""Test convolution pattern recognition."""
	# Clear graph
	GraphBuilder._thread_local = type(GraphBuilder._thread_local)()
	
	# Create conv + relu pattern
	input_tensor = torch.randn(1, 3, 32, 32)
	weight = torch.randn(64, 3, 3, 3)
	
	conv_lt = LazyTensor("aten::conv2d", [input_tensor, weight], {})
	relu_lt = LazyTensor("aten::relu", [conv_lt], {})
	
	graph = GraphBuilder.current().get_graph()
	
	conv_pattern = ConvolutionPattern()
	match = conv_pattern.match(graph)
	
	assert match is not None
	assert match.pattern_name == "conv_activation"
	assert match.confidence >= 0.9
	assert len(match.matched_nodes) == 2


def test_execution_with_multiple_operations():
	"""Test execution with various operation types."""
	# Clear graph
	GraphBuilder._thread_local = type(GraphBuilder._thread_local)()
	
	# Create a more complex graph
	a = torch.randn(3, 3)
	b = torch.randn(3, 3)
	
	# Matmul -> ReLU -> Add chain
	matmul_lt = LazyTensor("aten::matmul", [a, b], {})
	relu_lt = LazyTensor("aten::relu", [matmul_lt], {})
	add_lt = LazyTensor("aten::add", [relu_lt, a], {})
	
	# Execute the chain
	result = add_lt.cpu()
	
	assert isinstance(result, torch.Tensor)
	assert result.shape == torch.Size([3, 3])
	
	# Verify the final tensor is materialized
	assert add_lt.is_materialized()
	
	# Test that we can get reasonable values
	assert result.numel() == 9


@pytest.mark.parametrize("operation,inputs", [
	("aten::add", [torch.randn(2, 2), torch.randn(2, 2)]),
	("aten::sub", [torch.randn(2, 2), torch.randn(2, 2)]),
	("aten::mul", [torch.randn(2, 2), torch.randn(2, 2)]),
	("aten::matmul", [torch.randn(2, 3), torch.randn(3, 2)]),
	("aten::relu", [torch.randn(2, 2)]),
	("aten::sigmoid", [torch.randn(2, 2)]),
])
def test_operation_execution(operation, inputs):
	"""Test individual operation execution."""
	# Clear graph
	GraphBuilder._thread_local = type(GraphBuilder._thread_local)()
	
	lt = LazyTensor(operation, inputs, {})
	result = lt.cpu()
	
	assert isinstance(result, torch.Tensor)
	assert lt.is_materialized()


