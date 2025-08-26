import sys
sys.path.insert(0, "/home/jaewan/Genie")

import pytest

import genie
from genie.core import benchmarks


def test_microbenchmarks_run():
    stats = benchmarks.run_all()
    assert set(stats.keys()) == {
        "creation_randn_us",
        "intercept_add_us",
        "materialize_matmul_us",
    }
    for k, v in stats.items():
        assert isinstance(v, float)
        assert v > 0.0


