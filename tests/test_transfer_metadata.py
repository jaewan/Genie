from genie.runtime.transport_coordinator import DataPlaneBindings


def test_set_transfer_metadata(monkeypatch):
    b = DataPlaneBindings()

    class FakeLib:
        def __init__(self):
            self.calls = []

        def genie_data_plane_create(self, cfg):
            return 1

        def genie_set_transfer_metadata(self, dp, transfer_id, dtype_code, phase, shape_rank, shape_dims, dims_len):
            # Capture values for assertion
            try:
                n = int(getattr(dims_len, 'value', dims_len))
            except Exception:
                n = 0
            dims = [shape_dims[i] for i in range(n)]
            to_int = lambda x: int(getattr(x, 'value', x))
            self.calls.append((transfer_id.decode(), to_int(dtype_code), to_int(phase), to_int(shape_rank), dims))
            return 0

    b.lib = FakeLib()
    b.data_plane_ptr = None
    # No need to call create; we're testing signature pass-through via FakeLib

    # Invoke the C API through ctypes directly to validate signature
    import ctypes
    tid = b"tx123"
    dims = (ctypes.c_uint16 * 3)(1, 224, 224)
    rc = b.lib.genie_set_transfer_metadata(1, tid, ctypes.c_uint8(1), ctypes.c_uint8(0), ctypes.c_uint8(3), dims, ctypes.c_size_t(3))
    assert rc == 0
    assert b.lib.calls and b.lib.calls[-1] == ("tx123", 1, 0, 3, [1, 224, 224])

