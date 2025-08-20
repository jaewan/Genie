from __future__ import annotations

import statistics
import time
from typing import Callable, Tuple

import torch

from genie.core.lazy_tensor import LazyTensor


def benchmark(fn: Callable[[], None], iters: int = 1000) -> Tuple[float, float, float]:
	"""Return mean, p50, p95 in microseconds."""
	# Warmup
	for _ in range(50):
		fn()
	# Measure
	meas = []
	for _ in range(iters):
		t0 = time.perf_counter()
		fn()
		meas.append((time.perf_counter() - t0) * 1e6)
	meas.sort()
	return (statistics.mean(meas), meas[int(0.5 * iters)], meas[int(0.95 * iters)])


def print_row(name: str, stats: Tuple[float, float, float]) -> None:
	mean, p50, p95 = stats
	print(f"{name:28s}  mean={mean:8.2f}µs  p50={p50:8.2f}µs  p95={p95:8.2f}µs")


def bench_add(size: int = 256) -> None:
	print("\nAdd (size size):", size)
	a = torch.randn(size, size)
	b = torch.randn(size, size)

	# Overhead: constructing LazyTensor for add (no materialization)
	def create_lazy_add():
		_ = LazyTensor("aten::add", [a, b], {})

	# Eager add compute
	def eager_add():
		_ = torch.add(a, b)

	# Lazy materialization after creation
	def lazy_materialize_once():
		lt = LazyTensor("aten::add", [a, b], {})
		_ = lt.cpu()

	print_row("lazy add creation", benchmark(create_lazy_add))
	print_row("eager add compute", benchmark(eager_add))
	print_row("lazy add materialize", benchmark(lazy_materialize_once, iters=100))


def bench_matmul(m: int = 256, k: int = 256, n: int = 256) -> None:
	print("\nMatmul (m,k,n):", (m, k, n))
	x = torch.randn(m, k)
	y = torch.randn(k, n)

	def create_lazy_mm():
		_ = LazyTensor("aten::matmul", [x, y], {})

	def eager_mm():
		_ = torch.matmul(x, y)

	def lazy_mm_materialize_once():
		lt = LazyTensor("aten::matmul", [x, y], {})
		_ = lt.cpu()

	print_row("lazy mm creation", benchmark(create_lazy_mm))
	print_row("eager mm compute", benchmark(eager_mm))
	print_row("lazy mm materialize", benchmark(lazy_mm_materialize_once, iters=50))


def main() -> None:
	print("Genie Phase 1 Performance Comparison (CPU)")
	bench_add(256)
	bench_matmul(128, 128, 128)


if __name__ == "__main__":
	main()


