import sys
from pathlib import Path
import pytest
import os


# Ensure repository root is on sys.path so `import genie` works without editable install
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


@pytest.fixture(autouse=True)
def reset_graph_state():
    """Reset thread-local FX builder and LazyTensor IDs before each test.
    Prevents 'Cannot add operations to finalized graph' across tests.
    """
    try:
        from genie.core.fx_graph_builder import FXGraphBuilder
        FXGraphBuilder.reset()
    except Exception:
        pass


@pytest.fixture(autouse=True, scope="session")
def default_disable_cpp_dataplane():
    """Disable C++ data plane by default for test stability.
    Can be overridden per-test by setting GENIE_DISABLE_CPP_DATAPLANE explicitly.
    """
    if os.environ.get("GENIE_DISABLE_CPP_DATAPLANE") is None:
        os.environ["GENIE_DISABLE_CPP_DATAPLANE"] = "1"

    try:
        from genie.core.lazy_tensor import LazyTensor
        if hasattr(LazyTensor, "reset_id_counter"):
            LazyTensor.reset_id_counter()
    except Exception:
        pass


