from __future__ import annotations

import time
from typing import Dict

import torch

from .lazy_tensor import LazyTensor
from .graph import GraphBuilder


def _timeit(fn, iters: int = 1000) -> float:
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    end = time.perf_counter()
    return (end - start) / iters


def bench_creation(size=(64, 64), iters: int = 200) -> float:
    def _op():
        GraphBuilder.reset_current()
        _ = torch.randn(*size, device="remote_accelerator:0")
    return _timeit(_op, iters)


def bench_intercept_add(size=(64, 64), iters: int = 200) -> float:
    x = torch.randn(*size)
    y = torch.randn(*size)

    def _op():
        GraphBuilder.reset_current()
        lx = LazyTensor.lift(x)
        ly = LazyTensor.lift(y)
        _ = lx + ly
    return _timeit(_op, iters)


def bench_materialize_matmul(m=128, k=128, n=128, iters: int = 30) -> float:
    x = torch.randn(m, k)
    y = torch.randn(k, n)

    def _op():
        GraphBuilder.reset_current()
        lx = LazyTensor.lift(x)
        ly = LazyTensor.lift(y)
        _ = (lx @ ly).materialize()
    return _timeit(_op, iters)


def run_all() -> Dict[str, float]:
    return {
        "creation_randn_us": bench_creation() * 1e6,
        "intercept_add_us": bench_intercept_add() * 1e6,
        "materialize_matmul_us": bench_materialize_matmul() * 1e6,
    }


