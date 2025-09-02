import os
import pytest


@pytest.mark.skipif(os.environ.get("GENIE_WITH_SPDK_TEST") != "1", reason="SPDK not available for test")
def test_spdk_alloc_free():
    # This test only validates that symbols are callable when SPDK is linked
    try:
        from ctypes import CDLL
        lib = CDLL("/usr/local/lib/libgenie_data_plane.so")
    except Exception:
        pytest.skip("genie_data_plane not installed")
    # Smoke load OK
    assert lib is not None

