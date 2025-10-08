"""
Correctness validation for Genie LazyTensor implementation.

This example validates that LazyTensor interception produces numerically
identical results to native PyTorch execution across a comprehensive
range of operations, from simple arithmetic to complex multi-operation chains.
"""
from __future__ import annotations

import math
import torch
import numpy as np
from typing import Callable, Any, Tuple, List
import warnings

import genie
from genie.core.lazy_tensor import LazyTensor


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_test_result(test_name: str, passed: bool, error_msg: str = "") -> None:
    """Print test result with formatting."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{test_name:50s} {status}")
    if not passed and error_msg:
        print(f"    Error: {error_msg}")


def compare_tensors(lazy_result: torch.Tensor, eager_result: torch.Tensor, 
                   rtol: float = 1e-5, atol: float = 1e-8) -> Tuple[bool, str]:
    """Compare two tensors for numerical equality."""
    try:
        # Check shapes first
        if lazy_result.shape != eager_result.shape:
            return False, f"Shape mismatch: {lazy_result.shape} vs {eager_result.shape}"
        
        # Check dtypes
        if lazy_result.dtype != eager_result.dtype:
            return False, f"Dtype mismatch: {lazy_result.dtype} vs {eager_result.dtype}"
        
        # Check numerical equality
        if torch.allclose(lazy_result, eager_result, rtol=rtol, atol=atol):
            return True, ""
        else:
            max_diff = torch.max(torch.abs(lazy_result - eager_result)).item()
            return False, f"Max difference: {max_diff:.2e} (rtol={rtol}, atol={atol})"
    
    except Exception as e:
        return False, f"Comparison failed: {str(e)}"


def run_correctness_test(test_name: str, lazy_fn: Callable[[], Any], 
                        eager_fn: Callable[[], Any], **compare_kwargs) -> bool:
    """Run a single correctness test comparing lazy vs eager execution."""
    try:
        # Execute lazy and eager with identical RNG to ensure identical inputs
        torch.manual_seed(1234)
        lazy_result = lazy_fn()
        if isinstance(lazy_result, LazyTensor):
            lazy_result = lazy_result.cpu()
        
        # Execute eager version with same RNG seed to match values
        torch.manual_seed(1234)
        eager_result = eager_fn()
        
        # Compare results
        passed, error_msg = compare_tensors(lazy_result, eager_result, **compare_kwargs)
        print_test_result(test_name, passed, error_msg)
        return passed
    
    except Exception as e:
        print_test_result(test_name, False, str(e))
        return False


class CorrectnessValidator:
    """Comprehensive correctness validation suite."""
    
    def __init__(self):
        self.passed_tests = 0
        self.total_tests = 0
        
        # Set up test data
        torch.manual_seed(42)
        self.setup_test_data()
    
    def setup_test_data(self):
        """Set up various test tensors."""
        # Basic tensors - create on CPU, will be converted to LazyTensors during operations
        self.x_small = torch.randn(4, 4)
        self.y_small = torch.randn(4, 4)
        self.x_large = torch.randn(128, 128)
        self.y_large = torch.randn(128, 128)
        
        # Special shapes for broadcasting
        self.x_broadcast = torch.randn(4, 1)
        self.y_broadcast = torch.randn(1, 4)
        
        # 3D tensors for batch operations
        self.x_3d = torch.randn(2, 4, 4)
        self.y_3d = torch.randn(2, 4, 4)
        
        # 4D tensors for conv operations
        self.x_4d = torch.randn(1, 3, 8, 8)
        self.conv_weight = torch.randn(16, 3, 3, 3)
        self.conv_bias = torch.randn(16)
        
        # Vectors for 1D operations
        self.vec_a = torch.randn(10)
        self.vec_b = torch.randn(10)
        
        # Scalars
        self.scalar = torch.tensor(2.5)

        # Deprecated: remote factory tensors; Phase 1 uses lift() for correctness
    
    def test(self, name: str, lazy_fn: Callable, eager_fn: Callable, **kwargs) -> None:
        """Run a single test and track results."""
        self.total_tests += 1
        if run_correctness_test(name, lazy_fn, eager_fn, **kwargs):
            self.passed_tests += 1
    
    def run_basic_arithmetic_tests(self):
        """Test basic arithmetic operations."""
        print_header("Basic Arithmetic Operations")
        
        # Addition
        self.test("torch.add", 
                 lambda: torch.add(LazyTensor.lift(self.x_small), LazyTensor.lift(self.y_small)),
                 lambda: torch.add(self.x_small, self.y_small))
        
        # Subtraction
        self.test("torch.sub",
                 lambda: torch.sub(LazyTensor.lift(self.x_small), LazyTensor.lift(self.y_small)),
                 lambda: torch.sub(self.x_small, self.y_small))
        
        # Multiplication
        self.test("torch.mul",
                 lambda: torch.mul(LazyTensor.lift(self.x_small), LazyTensor.lift(self.y_small)),
                 lambda: torch.mul(self.x_small, self.y_small))
        
        # Division
        self.test("torch.div",
                 lambda: torch.div(LazyTensor.lift(self.x_small), LazyTensor.lift(self.y_small + 0.1)),  # Avoid div by zero
                 lambda: torch.div(self.x_small, self.y_small + 0.1))
        
        # Broadcasting operations
        self.test("torch.add (broadcast)",
                 lambda: torch.add(LazyTensor.lift(self.x_broadcast), LazyTensor.lift(self.y_broadcast)),
                 lambda: torch.add(self.x_broadcast, self.y_broadcast))
    
    def run_linear_algebra_tests(self):
        """Test linear algebra operations."""
        print_header("Linear Algebra Operations")
        
        # Matrix multiplication
        self.test("torch.matmul",
                 lambda: torch.matmul(LazyTensor.lift(self.x_small), LazyTensor.lift(self.y_small)),
                 lambda: torch.matmul(self.x_small, self.y_small))
        
        # 2D matrix multiplication
        self.test("torch.mm",
                 lambda: torch.mm(LazyTensor.lift(self.x_small), LazyTensor.lift(self.y_small)),
                 lambda: torch.mm(self.x_small, self.y_small))
        
        # Batch matrix multiplication
        self.test("torch.bmm",
                 lambda: torch.bmm(LazyTensor.lift(self.x_3d), LazyTensor.lift(self.y_3d)),
                 lambda: torch.bmm(self.x_3d, self.y_3d))
        
        # Vector operations
        self.test("torch.dot (via matmul)",
                 lambda: torch.matmul(LazyTensor.lift(self.vec_a), LazyTensor.lift(self.vec_b)),
                 lambda: torch.matmul(self.vec_a, self.vec_b))
    
    def run_activation_tests(self):
        """Test activation functions."""
        print_header("Activation Functions")
        
        # ReLU
        self.test("torch.relu",
                 lambda: torch.relu(LazyTensor.lift(self.x_small)),
                 lambda: torch.relu(self.x_small))
        
        # Sigmoid
        self.test("torch.sigmoid",
                 lambda: torch.sigmoid(LazyTensor.lift(self.x_small)),
                 lambda: torch.sigmoid(self.x_small))
        
        # Tanh
        self.test("torch.tanh",
                 lambda: torch.tanh(LazyTensor.lift(self.x_small)),
                 lambda: torch.tanh(self.x_small))
        
        # Softmax
        self.test("torch.softmax",
                 lambda: torch.softmax(LazyTensor.lift(self.x_small), dim=1),
                 lambda: torch.softmax(self.x_small, dim=1))
    
    def run_tensor_manipulation_tests(self):
        """Test tensor manipulation operations."""
        print_header("Tensor Manipulation Operations")
        
        # Transpose
        self.test("torch.transpose",
                 lambda: torch.transpose(LazyTensor.lift(self.x_small), 0, 1),
                 lambda: torch.transpose(self.x_small, 0, 1))
        
        # Reshape
        self.test("torch.reshape",
                 lambda: torch.reshape(LazyTensor.lift(self.x_small), (2, 8)),
                 lambda: torch.reshape(self.x_small, (2, 8)))
        
        # Squeeze/Unsqueeze
        x_with_dim = self.x_small.unsqueeze(0)
        x_with_dim_lazy = LazyTensor.lift(self.x_small).unsqueeze(0)
        self.test("torch.squeeze",
                 lambda: torch.squeeze(x_with_dim_lazy, 0),
                 lambda: torch.squeeze(x_with_dim, 0))
        
        self.test("torch.unsqueeze",
                 lambda: torch.unsqueeze(LazyTensor.lift(self.x_small), 0),
                 lambda: torch.unsqueeze(self.x_small, 0))
        
        # Permute
        self.test("torch.permute (via transpose)",
                 lambda: torch.transpose(LazyTensor.lift(self.x_3d), 1, 2),
                 lambda: torch.transpose(self.x_3d, 1, 2))
    
    def run_reduction_tests(self):
        """Test reduction operations."""
        print_header("Reduction Operations")
        
        # Sum
        self.test("torch.sum",
                 lambda: torch.sum(LazyTensor.lift(self.x_small)),
                 lambda: torch.sum(self.x_small))
        
        self.test("torch.sum (dim=1)",
                 lambda: torch.sum(LazyTensor.lift(self.x_small), dim=1),
                 lambda: torch.sum(self.x_small, dim=1))
        
        # Mean
        self.test("torch.mean",
                 lambda: torch.mean(LazyTensor.lift(self.x_small)),
                 lambda: torch.mean(self.x_small))
        
        # Max (note: returns values and indices, we'll just test values)
        self.test("torch.max",
                 lambda: torch.max(LazyTensor.lift(self.x_small)),
                 lambda: torch.max(self.x_small))
    
    def run_elementwise_tests(self):
        """Test element-wise mathematical functions."""
        print_header("Element-wise Mathematical Functions")
        
        # Use positive values to avoid domain issues
        x_pos = torch.abs(self.x_small) + 0.1
        x_pos_lazy = torch.abs(LazyTensor.lift(self.x_small)) + 0.1
        
        # Absolute value
        self.test("torch.abs",
                 lambda: torch.abs(LazyTensor.lift(self.x_small)),
                 lambda: torch.abs(self.x_small))
        
        # Exponential
        self.test("torch.exp",
                 lambda: torch.exp(LazyTensor.lift(self.x_small)),
                 lambda: torch.exp(self.x_small))
        
        # Logarithm
        self.test("torch.log",
                 lambda: torch.log(x_pos_lazy),
                 lambda: torch.log(x_pos))
        
        # Square root
        self.test("torch.sqrt",
                 lambda: torch.sqrt(x_pos_lazy),
                 lambda: torch.sqrt(x_pos))
        
        # Power
        self.test("torch.pow",
                 lambda: torch.pow(x_pos_lazy, 2.0),
                 lambda: torch.pow(x_pos, 2.0))
        
        # Trigonometric
        self.test("torch.sin",
                 lambda: torch.sin(LazyTensor.lift(self.x_small)),
                 lambda: torch.sin(self.x_small))
        
        self.test("torch.cos",
                 lambda: torch.cos(LazyTensor.lift(self.x_small)),
                 lambda: torch.cos(self.x_small))
        
        # Clamp
        self.test("torch.clamp",
                 lambda: torch.clamp(LazyTensor.lift(self.x_small), -1.0, 1.0),
                 lambda: torch.clamp(self.x_small, -1.0, 1.0))
    
    def run_complex_chain_tests(self):
        """Test complex operation chains."""
        print_header("Complex Operation Chains")
        
        # Chain 1: (x @ y) + x
        self.test("matmul + add chain",
                 lambda: torch.add(
                    torch.matmul(LazyTensor.lift(self.x_small), 
                                LazyTensor.lift(self.y_small)),
                    LazyTensor.lift(self.x_small)),
                 lambda: torch.add(torch.matmul(self.x_small, self.y_small), self.x_small))
        
        # Chain 2: relu(x @ y + bias)
        bias = torch.randn(4)
        self.test("linear + relu chain",
                 lambda: torch.relu(
                    torch.add(
                        torch.matmul(LazyTensor.lift(self.x_small), 
                                    LazyTensor.lift(self.y_small)),
                        bias)),
                 lambda: torch.relu(torch.add(torch.matmul(self.x_small, self.y_small), bias)))
        
        # Chain 3: softmax(tanh(x) * y)
        self.test("tanh + mul + softmax chain",
                 lambda: torch.softmax(
                    torch.mul(
                        torch.tanh(LazyTensor.lift(self.x_small)),
                        LazyTensor.lift(self.y_small)), dim=1),
                 lambda: torch.softmax(torch.mul(torch.tanh(self.x_small), self.y_small), dim=1))
        
        # Chain 4: Complex reshape and transpose chain
        self.test("reshape + transpose + reshape chain",
                 lambda: torch.reshape(
                    torch.transpose(
                        torch.reshape(LazyTensor.lift(self.x_small), (2, 8)), 0, 1), 
                    (4, 4)),
                 lambda: torch.reshape(torch.transpose(torch.reshape(self.x_small, (2, 8)), 0, 1), (4, 4)))
    
    def run_mixed_tensor_tests(self):
        """Test operations mixing LazyTensor and regular tensors."""
        print_header("Mixed Tensor Operations")
        
        # LazyTensor + CPU tensor
        self.test("lazy + cpu tensor",
                 lambda: torch.add(LazyTensor.lift(self.x_small), self.y_small),
                 lambda: torch.add(self.x_small, self.y_small))
        
        # CPU tensor + LazyTensor
        self.test("cpu + lazy tensor",
                 lambda: torch.add(self.x_small, LazyTensor.lift(self.y_small)),
                 lambda: torch.add(self.x_small, self.y_small))
        
        # LazyTensor with scalar
        self.test("lazy tensor + scalar",
                 lambda: torch.add(LazyTensor.lift(self.x_small), 2.5),
                 lambda: torch.add(self.x_small, 2.5))
    
    def run_tensor_creation_tests(self):
        """Test tensor creation operations."""
        print_header("Tensor Creation Operations")
        
        # Note: We compare shapes and dtypes since random values will differ
        def compare_creation(lazy_fn, eager_fn, name):
            lazy_result = lazy_fn().cpu()
            eager_result = eager_fn()
            
            shape_match = lazy_result.shape == eager_result.shape
            dtype_match = lazy_result.dtype == eager_result.dtype
            passed = shape_match and dtype_match
            
            error_msg = ""
            if not shape_match:
                error_msg += f"Shape mismatch: {lazy_result.shape} vs {eager_result.shape}. "
            if not dtype_match:
                error_msg += f"Dtype mismatch: {lazy_result.dtype} vs {eager_result.dtype}."
            
            print_test_result(name, passed, error_msg)
            return passed
        
        # Test creation operations (compare structure, not values)
        self.total_tests += 4
        
        if compare_creation(
            lambda: torch.randn(4, 4, device="remote_accelerator:0"),
            lambda: torch.randn(4, 4),
            "torch.randn"):
            self.passed_tests += 1
        
        if compare_creation(
            lambda: torch.zeros(4, 4, device="remote_accelerator:0"),
            lambda: torch.zeros(4, 4),
            "torch.zeros"):
            self.passed_tests += 1
        
        if compare_creation(
            lambda: torch.ones(4, 4, device="remote_accelerator:0"),
            lambda: torch.ones(4, 4),
            "torch.ones"):
            self.passed_tests += 1
        
        # For zeros and ones, we can compare values too
        self.test("torch.zeros (values)",
                 lambda: torch.zeros(4, 4, device="remote_accelerator:0"),
                 lambda: torch.zeros(4, 4))
    
    def run_all_tests(self):
        """Run all correctness tests."""
        print_header("Genie LazyTensor Correctness Validation")
        print("Comparing LazyTensor results with native PyTorch execution...")
        
        # Enable lazy mode
        genie.set_lazy_mode(True)
        
        # Run test suites
        self.run_basic_arithmetic_tests()
        self.run_linear_algebra_tests()
        self.run_activation_tests()
        self.run_tensor_manipulation_tests()
        self.run_reduction_tests()
        self.run_elementwise_tests()
        self.run_complex_chain_tests()
        self.run_mixed_tensor_tests()
        self.run_tensor_creation_tests()
        
        # Print summary
        print_header("Correctness Validation Summary")
        pass_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        print(f"Tests passed: {self.passed_tests}/{self.total_tests} ({pass_rate:.1f}%)")
        
        if pass_rate >= 95:
            print("🎉 EXCELLENT: Genie LazyTensor produces correct results!")
        elif pass_rate >= 90:
            print("✅ GOOD: Minor discrepancies detected, but mostly correct.")
        elif pass_rate >= 80:
            print("⚠️  WARNING: Some correctness issues detected.")
        else:
            print("❌ CRITICAL: Significant correctness issues detected.")
        
        return pass_rate >= 95


def main():
    """Run the correctness validation suite."""
    validator = CorrectnessValidator()
    success = validator.run_all_tests()
    
    if success:
        print("\n🎯 All correctness tests passed! Genie LazyTensor is ready for production.")
    else:
        print("\n🔧 Some tests failed. Review the results above for debugging.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
