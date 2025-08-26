from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Optional


def _load_first(names: list[str]) -> Optional[ctypes.CDLL]:
    for n in names:
        try:
            return ctypes.CDLL(n)
        except OSError:
            continue
    return None


@dataclass
class DpdkLibraries:
    eal: Optional[ctypes.CDLL]
    mempool: Optional[ctypes.CDLL]
    mbuf: Optional[ctypes.CDLL]

    @property
    def ok(self) -> bool:
        return self.eal is not None and self.mempool is not None and self.mbuf is not None


def load_dpdk() -> DpdkLibraries:
    # Try our installed DPDK first, then common system locations
    dpdk_lib_path = "/opt/dpdk/dpdk-23.11/install/lib/x86_64-linux-gnu"
    
    eal = _load_first([
        f"{dpdk_lib_path}/librte_eal.so.24.0",
        f"{dpdk_lib_path}/librte_eal.so.24",
        f"{dpdk_lib_path}/librte_eal.so",
        "librte_eal.so", "librte_eal.so.20", "librte_eal.so.23", "librte_eal.so.24",
        "libdpdk.so", "libdpdk.so.20", "libdpdk.so.23", "libdpdk.so.24",
        "/usr/lib/x86_64-linux-gnu/librte_eal.so",
    ])
    mempool = _load_first([
        f"{dpdk_lib_path}/librte_mempool.so.24.0",
        f"{dpdk_lib_path}/librte_mempool.so.24",
        f"{dpdk_lib_path}/librte_mempool.so",
        "librte_mempool.so", "librte_mempool.so.20", "librte_mempool.so.23", "librte_mempool.so.24",
        "libdpdk.so", "libdpdk.so.20", "libdpdk.so.23", "libdpdk.so.24",
        "/usr/lib/x86_64-linux-gnu/librte_mempool.so",
    ])
    mbuf = _load_first([
        f"{dpdk_lib_path}/librte_mbuf.so.24.0",
        f"{dpdk_lib_path}/librte_mbuf.so.24",
        f"{dpdk_lib_path}/librte_mbuf.so",
        "librte_mbuf.so", "librte_mbuf.so.20", "librte_mbuf.so.23", "librte_mbuf.so.24",
        "libdpdk.so", "libdpdk.so.20", "libdpdk.so.23", "libdpdk.so.24",
        "/usr/lib/x86_64-linux-gnu/librte_mbuf.so",
    ])
    return DpdkLibraries(eal=eal, mempool=mempool, mbuf=mbuf)


class DpdkRuntime:
    def __init__(self) -> None:
        self.libs = load_dpdk()
        self.eal_initialized = False
        self._pool = None

        if not self.libs.ok:
            return

        # Resolve and validate required symbols; any failure disables availability
        try:
            # int rte_eal_init(int argc, char **argv)
            getattr(self.libs.eal, "rte_eal_init")
            self.libs.eal.rte_eal_init.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
            self.libs.eal.rte_eal_init.restype = ctypes.c_int

            # struct rte_mempool* rte_pktmbuf_pool_create(...)
            getattr(self.libs.mbuf, "rte_pktmbuf_pool_create")
            self.libs.mbuf.rte_pktmbuf_pool_create.argtypes = [
                ctypes.c_char_p,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint16,
                ctypes.c_uint16,
                ctypes.c_int,
            ]
            self.libs.mbuf.rte_pktmbuf_pool_create.restype = ctypes.c_void_p

            # struct rte_mbuf* rte_pktmbuf_alloc(struct rte_mempool *mp)
            getattr(self.libs.mbuf, "rte_pktmbuf_alloc")
            self.libs.mbuf.rte_pktmbuf_alloc.argtypes = [ctypes.c_void_p]
            self.libs.mbuf.rte_pktmbuf_alloc.restype = ctypes.c_void_p
        except AttributeError:
            # Mark as unavailable if any symbol is missing
            self.libs = DpdkLibraries(eal=None, mempool=None, mbuf=None)

    def is_available(self) -> bool:
        return self.libs.ok

    def init_eal(self, args: Optional[list[str]] = None) -> bool:
        if not self.libs.ok:
            return False
        if self.eal_initialized:
            return True
        # Minimal EAL init; users should set hugepages separately
        argv = [b"genie-dpdk"]
        if args:
            argv.extend([a.encode("utf-8") for a in args])
        argc = len(argv)
        c_argv = (ctypes.c_char_p * argc)(*argv)
        ret = self.libs.eal.rte_eal_init(ctypes.c_int(argc), c_argv)
        if ret < 0:
            return False
        self.eal_initialized = True
        return True

    def create_default_pool(self, name: str = "genie_pool", n: int = 8192, data_room: int = 4096) -> Optional[ctypes.c_void_p]:
        if not self.eal_initialized:
            return None
        pool = self.libs.mbuf.rte_pktmbuf_pool_create(
            name.encode("utf-8"),
            ctypes.c_uint(n),
            ctypes.c_uint(256),
            ctypes.c_uint16(0),
            ctypes.c_uint16(data_room),
            ctypes.c_int(-1),
        )
        self._pool = pool
        return pool

    def alloc_mbuf(self, pool: Optional[ctypes.c_void_p] = None) -> Optional[ctypes.c_void_p]:
        if not self.eal_initialized:
            return None
        mp = pool or self._pool
        if not mp:
            return None
        mbuf = self.libs.mbuf.rte_pktmbuf_alloc(mp)
        return mbuf


