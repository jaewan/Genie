import os
import pytest
import torch


def _ensure_import_path():
    import sys
    root = "/home/jaewan/Genie"
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_import_path()
import genie  # noqa: E402
from genie.core.lazy_tensor import LazyTensor  # noqa: E402


@pytest.fixture(autouse=True)
def enable_lazy_and_logging(monkeypatch):
    monkeypatch.setenv("GENIE_LOG_INTERCEPTS", "0")
    monkeypatch.setenv("GENIE_ENABLE_ATEN_IMPL", "0")
    yield


def test_reductions_sum_mean_var_std():
    x = torch.randn(4, 5)
    lx = LazyTensor.lift(x)
    s = lx.sum(dim=1)
    m = lx.mean(dim=1)
    v = lx.var(dim=1)
    sd = lx.std(dim=1)

    torch.testing.assert_close(s.materialize(), x.sum(dim=1))
    torch.testing.assert_close(m.materialize(), x.mean(dim=1))
    torch.testing.assert_close(v.materialize(), x.var(dim=1))
    torch.testing.assert_close(sd.materialize(), x.std(dim=1))


def test_broadcasting_add_mm():
    a = torch.randn(3, 1)
    b = torch.randn(1, 4)
    la = LazyTensor.lift(a)
    lb = LazyTensor.lift(b)
    c = (la + lb).materialize()
    torch.testing.assert_close(c, a + b)

    x = torch.randn(2, 3)
    y = torch.randn(3, 4)
    lx = LazyTensor.lift(x)
    ly = LazyTensor.lift(y)
    z = (lx @ ly).materialize()
    torch.testing.assert_close(z, x @ y)


def test_mixed_dtypes_add():
    a = torch.randn(2, 2, dtype=torch.float32)
    b = torch.randn(2, 2, dtype=torch.float16).to(torch.float32)
    la = LazyTensor.lift(a)
    lb = LazyTensor.lift(b)
    out = (la + lb).materialize()
    torch.testing.assert_close(out, a + b)


def test_argmax_argmin_softmax():
    x = torch.randn(5, 7)
    lx = LazyTensor.lift(x)
    amx = lx.argmax(dim=1).materialize()
    amn = lx.argmin(dim=1).materialize()
    smx = lx.softmax(dim=1).materialize()
    torch.testing.assert_close(amx, x.argmax(dim=1))
    torch.testing.assert_close(amn, x.argmin(dim=1))
    torch.testing.assert_close(smx, torch.softmax(x, dim=1))


