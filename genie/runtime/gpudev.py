from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Optional


def _load(names: list[str]) -> Optional[ctypes.CDLL]:
    for n in names:
        try:
            return ctypes.CDLL(n)
        except OSError:
            continue
    return None


@dataclass
class GpuDevLibs:
    gpudev: Optional[ctypes.CDLL]

    @property
    def ok(self) -> bool:
        return self.gpudev is not None


def load_gpudev() -> GpuDevLibs:
    lib = _load(["librte_gpudev.so", "librte_gpudev.so.23", "librte_gpudev.so.23.0"])
    return GpuDevLibs(gpudev=lib)


class GPUDevInterface:
    def __init__(self) -> None:
        self.libs = load_gpudev()
        self.gpudev_available = self.libs.ok

        if not self.libs.ok:
            return
        # Stubs; wiring would require proper headers and CTypes structures
        # int rte_gpu_count(void)
        try:
            self.libs.gpudev.rte_gpu_count.argtypes = []
            self.libs.gpudev.rte_gpu_count.restype = ctypes.c_int
        except Exception:
            pass

    def is_available(self) -> bool:
        return self.gpudev_available


