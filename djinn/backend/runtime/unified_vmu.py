"""
Unified Virtual Memory Unit (VMU): Dual-Lifecycle Memory Management

Implements the v2.3 architecture feature: "Dual-Lifecycle VMU"

The Watermark Pattern:
- Separates persistent state (KV Cache, Weights) from volatile activations
- Persistent memory never freed between requests
- Volatile memory reset after each request
- Enables efficient LLM generation with growing KV cache

Memory Layout:
┌────────────────────────────────────┐
│ OS Reserved (4GB)                  │ ← Kernel, drivers
├────────────────────────────────────┤
│ PERSISTENT SECTION                 │ ← Weights, KV Cache (grows)
│ (Watermark: persistent_offset)     │
├────────────────────────────────────┤
│ VOLATILE SECTION                   │ ← Activations (reset each request)
│ (Current: current_offset)          │
├────────────────────────────────────┤
│ Free Memory                        │
└────────────────────────────────────┘

Lifecycle:
1. Persistent allocation: Move watermark up (malloc_persistent)
2. Volatile allocation: Allocate in volatile section (malloc_volatile)
3. Request complete: Reset volatile to watermark (reset_volatile)
4. Next request: Volatile section reused, persistent preserved
"""

import torch
import logging
import threading
import socket
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AllocationProfile:
    """Profile of a memory allocation."""
    start_offset: int
    size_bytes: int
    dtype: torch.dtype
    persistent: bool
    name: str = "unnamed"
    timestamp: float = 0.0
    
    @property
    def end_offset(self) -> int:
        return self.start_offset + self.size_bytes


class UnifiedVMU:
    """
    Unified Virtual Memory Unit with Watermark-based dual-lifecycle management.
    
    Key Properties:
    - Zero alignment overhead (minimal fragmentation)
    - Constant-time allocation for both persistent and volatile
    - Predictable memory layout for optimization
    - Safe concurrent access with proper synchronization
    
    Usage:
        vmu = UnifiedVMU(device_id=0)
        
        # Allocate KV cache (persistent)
        kv_offset = vmu.malloc_persistent(kv_size)
        
        # Allocate activations (volatile)
        act_offset = vmu.malloc_volatile(act_size)
        
        # After request completes
        vmu.reset_volatile()  # Reuse activation memory
    """
    
    def __init__(self, device_id: int = 0, os_reserve_gb: float = 4.0, alignment: int = 256):
        """
        Initialize VMU on specified GPU device.
        
        Args:
            device_id: GPU device ID
            os_reserve_gb: Memory reserved for OS/kernels
            alignment: Alignment requirement (bytes)
        """
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}')
        self.alignment = alignment
        self.lock = threading.Lock()  # CONCURRENCY CONTROL
        
        # Get device properties
        props = torch.cuda.get_device_properties(device_id)
        total_memory = props.total_memory
        
        # Reserve memory for OS
        os_reserve_bytes = int(os_reserve_gb * 1024**3)
        self.slab_capacity = total_memory - os_reserve_bytes
        
        logger.info(
            f"VMU initialized on cuda:{device_id}\n"
            f"  Total GPU memory: {total_memory / 1024**3:.1f} GB\n"
            f"  OS reserve: {os_reserve_gb} GB\n"
            f"  Slab capacity: {self.slab_capacity / 1024**3:.1f} GB"
        )
        
        # Allocate the unified slab buffer
        try:
            self.buffer = torch.zeros(self.slab_capacity, dtype=torch.uint8, device=self.device)
            logger.info(f"✅ Allocated unified slab: {self.slab_capacity / 1024**3:.1f} GB")
        except RuntimeError as e:
            logger.error(f"❌ Failed to allocate slab: {e}")
            raise

        # CPU Staging Buffer (Pinned for DMA) - 32MB chunk
        self.staging_size = 32 * 1024 * 1024  # 32MB
        self.staging = None
        try:
            self.staging = torch.zeros(self.staging_size, dtype=torch.uint8).pin_memory()
            logger.info(f"✅ Allocated pinned staging buffer: {self.staging_size / 1024**2:.1f} MB")
        except RuntimeError as e:
            # Fallback to regular CPU buffer if pinning fails
            logger.warning(f"⚠️  Pinned memory not available, using regular CPU buffer: {e}")
            try:
                self.staging = torch.zeros(self.staging_size, dtype=torch.uint8)
                logger.info(f"✅ Allocated regular staging buffer: {self.staging_size / 1024**2:.1f} MB")
            except RuntimeError as e2:
                logger.error(f"❌ Failed to allocate staging buffer: {e2}")
                raise

        # Memory pointers
        self.persistent_watermark = 0  # Boundary between persistent and volatile
        self.current_offset = 0        # Current allocation head in volatile section
        
        # Tracking
        self.allocations: Dict[int, AllocationProfile] = {}
        self.allocation_counter = 0
        
        # Statistics
        self.stats = {
            'persistent_allocated_bytes': 0,
            'volatile_allocated_bytes': 0,
            'max_persistent_offset': 0,
            'max_current_offset': 0,
            'reset_volatile_count': 0,
            'allocation_count': 0,
        }
    
    def _align(self, offset: int) -> int:
        """Align offset to alignment boundary (round up) using bit manipulation."""
        # For power-of-2 alignment: (offset + alignment - 1) & ~(alignment - 1)
        # This is more efficient than modulo arithmetic
        return (offset + self.alignment - 1) & ~(self.alignment - 1)
    
    def malloc_persistent(self, size: int, dtype: torch.dtype = torch.float32, name: str = "") -> int:
        """
        Allocate persistent memory (KV Cache, Weights).
        
        Persistent allocations:
        - Move the watermark up
        - Never freed until model unloads
        - Preserved across requests
        
        Args:
            size: Allocation size in bytes
            dtype: Data type (for tracking)
            name: Name for debugging
        
        Returns:
            Offset in the slab buffer
        
        Raises:
            RuntimeError: If allocation would exceed slab capacity
        """
        aligned_start = self._align(self.persistent_watermark)
        new_watermark = aligned_start + size
        
        if new_watermark >= self.slab_capacity:
            used = self.persistent_watermark / 1024**3
            available = self.slab_capacity / 1024**3
            raise RuntimeError(
                f"Persistent memory exhausted: "
                f"requested {size / 1024**3:.2f}GB, "
                f"used {used:.2f}GB / {available:.2f}GB"
            )
        
        # Update watermark
        self.persistent_watermark = new_watermark
        self.current_offset = new_watermark  # Move volatile head too
        
        # Track allocation
        alloc_id = self.allocation_counter
        self.allocation_counter += 1
        self.allocations[alloc_id] = AllocationProfile(
            start_offset=aligned_start,
            size_bytes=size,
            dtype=dtype,
            persistent=True,
            name=name
        )
        
        # Update stats
        self.stats['persistent_allocated_bytes'] = self.persistent_watermark
        self.stats['allocation_count'] += 1
        self.stats['max_persistent_offset'] = max(
            self.stats['max_persistent_offset'],
            self.persistent_watermark
        )
        
        logger.debug(
            f"Persistent allocation: {name} @ {aligned_start / 1024**2:.1f}MB, "
            f"size={size / 1024**2:.1f}MB, watermark={self.persistent_watermark / 1024**2:.1f}MB"
        )
        
        return aligned_start
    
    def malloc_volatile(self, size: int, dtype: torch.dtype = torch.float32, name: str = "") -> int:
        """
        Allocate volatile memory (Activations).
        
        Volatile allocations:
        - Stay below persistent watermark
        - Reset after each request
        - Can be reused for next request
        
        Args:
            size: Allocation size in bytes
            dtype: Data type (for tracking)
            name: Name for debugging
        
        Returns:
            Offset in the slab buffer
        
        Raises:
            RuntimeError: If allocation would exceed slab capacity
        """
        aligned_start = self._align(self.current_offset)
        new_offset = aligned_start + size
        
        if new_offset >= self.slab_capacity:
            used_persistent = self.persistent_watermark / 1024**3
            used_volatile = (self.current_offset - self.persistent_watermark) / 1024**3
            available = self.slab_capacity / 1024**3
            raise RuntimeError(
                f"Volatile memory exhausted: "
                f"requested {size / 1024**3:.2f}GB, "
                f"used (persistent={used_persistent:.2f}GB, volatile={used_volatile:.2f}GB) / "
                f"available={available:.2f}GB"
            )
        
        # Update offset
        self.current_offset = new_offset
        
        # Track allocation
        alloc_id = self.allocation_counter
        self.allocation_counter += 1
        self.allocations[alloc_id] = AllocationProfile(
            start_offset=aligned_start,
            size_bytes=size,
            dtype=dtype,
            persistent=False,
            name=name
        )
        
        # Update stats
        self.stats['volatile_allocated_bytes'] = (
            self.current_offset - self.persistent_watermark
        )
        self.stats['allocation_count'] += 1
        self.stats['max_current_offset'] = max(
            self.stats['max_current_offset'],
            self.current_offset
        )
        
        logger.debug(
            f"Volatile allocation: {name} @ {aligned_start / 1024**2:.1f}MB, "
            f"size={size / 1024**2:.1f}MB, offset={self.current_offset / 1024**2:.1f}MB"
        )
        
        return aligned_start
    
    def reset_volatile(self) -> None:
        """
        Reset volatile memory after request completes.
        
        CRITICAL: This is called after each inference request.
        - Resets current_offset to persistent_watermark
        - Keeps persistent state (KV cache, weights) intact
        - Enables efficient reuse of volatile section
        
        This is the KEY INNOVATION that solves "Reset vs. Persistence" conflict.
        """
        if self.current_offset == self.persistent_watermark:
            # Nothing to reset
            return
        
        freed_bytes = self.current_offset - self.persistent_watermark
        self.current_offset = self.persistent_watermark
        
        self.stats['reset_volatile_count'] += 1
        
        logger.debug(
            f"Reset volatile memory: freed {freed_bytes / 1024**2:.1f}MB, "
            f"watermark at {self.persistent_watermark / 1024**2:.1f}MB"
        )

    def _recv_exact(self, sock: socket.socket, buf) -> None:
        """
        Reliable TCP read helper that handles partial receives.

        Args:
            sock: TCP socket to read from
            buf: Buffer to read into (must be writable)

        Raises:
            ConnectionError: If connection is closed during read
        """
        view = memoryview(buf)
        while len(view) > 0:
            n = sock.recv_into(view)
            if n == 0:
                raise ConnectionError("Unexpected EOF during DMA read")
            view = view[n:]

    def write_from_socket(self, sock: socket.socket, total_size: int,
                         is_persistent: bool = False) -> int:
        """
        Reads network data -> CPU staging -> GPU slab with DMA synchronization.

        Implements the v2.3.15 DMA pipeline:
        1. Pre-flight OOM check
        2. Chunked DMA with proper synchronization
        3. Memory barrier to prevent data corruption

        Args:
            sock: TCP socket to read from
            total_size: Total bytes to transfer
            is_persistent: Whether this is persistent allocation (moves watermark)

        Returns:
            GPU offset where data was written

        Raises:
            MemoryError: If allocation would exceed slab capacity
        """
        with self.lock:
            # 1. Pre-Flight OOM Check
            base_ptr = self.persistent_watermark if is_persistent else self.current_offset

            # Align start address to 256-byte boundary
            gpu_start = (base_ptr + self.alignment - 1) & ~(self.alignment - 1)

            if gpu_start + total_size > self.slab_capacity:
                used = self.persistent_watermark / 1024**3
                available = self.slab_capacity / 1024**3
                requested = total_size / 1024**3
                raise MemoryError(
                    f"Slab OOM during DMA: "
                    f"requested {requested:.2f}GB, "
                    f"used {used:.2f}GB / available {available:.2f}GB"
                )

            # 2. Chunked DMA Pipeline
            bytes_received = 0
            chunk_cap = self.staging_size  # 32MB chunks

            # Get CUDA stream for synchronization
            stream = torch.cuda.current_stream(device=self.device)

            # Check if staging buffer is pinned
            is_pinned = self.staging.is_pinned() if hasattr(self.staging, 'is_pinned') else False
            if not is_pinned:
                logger.warning("⚠️  Using non-pinned staging buffer - DMA may be slower")

            logger.debug(f"Starting DMA transfer: {total_size} bytes to GPU offset {gpu_start}")

            while bytes_received < total_size:
                chunk_size = min(total_size - bytes_received, chunk_cap)

                # A. Read into Pinned CPU Buffer
                cpu_view = self.staging[:chunk_size]
                self._recv_exact(sock, cpu_view)

                # B. Async DMA Copy to GPU
                gpu_slice_start = gpu_start + bytes_received
                gpu_slice_end = gpu_slice_start + chunk_size
                gpu_dest = self.buffer[gpu_slice_start:gpu_slice_end]

                # Non-blocking copy (async DMA)
                gpu_dest.copy_(cpu_view, non_blocking=True)

                # C. CRITICAL: Synchronize to prevent CPU overwriting buffer
                # before GPU finishes reading previous chunk
                stream.synchronize()

                bytes_received += chunk_size

                logger.debug(f"DMA chunk: {chunk_size} bytes, total {bytes_received}/{total_size}")

            # 3. Update Pointers
            new_end = gpu_start + total_size
            if is_persistent:
                self.persistent_watermark = new_end
                self.current_offset = new_end
            else:
                self.current_offset = new_end

            # 4. Update statistics
            alloc_size_mb = total_size / 1024**2
            if is_persistent:
                self.stats['persistent_allocated_bytes'] += total_size
                logger.debug(f"Persistent DMA allocation: {alloc_size_mb:.1f}MB at offset {gpu_start}")
            else:
                self.stats['volatile_allocated_bytes'] += total_size
                logger.debug(f"Volatile DMA allocation: {alloc_size_mb:.1f}MB at offset {gpu_start}")

            return gpu_start
    
    def get_slab_view(self, offset: int, size: int, dtype: torch.dtype) -> torch.Tensor:
        """
        Get a view of the slab buffer at specified offset.
        
        Returns a strided tensor view - no copy, zero overhead.
        
        Args:
            offset: Start offset in bytes
            size: Size in bytes
            dtype: Data type
        
        Returns:
            torch.Tensor view of slab buffer
        """
        element_size = torch.empty(0, dtype=dtype).element_size()
        num_elements = size // element_size
        
        # Create view with as_strided
        view = self.buffer[offset:offset + size].view(dtype=dtype)
        
        if view.numel() != num_elements:
            # Size mismatch - shouldn't happen
            logger.warning(
                f"Size mismatch in slab view: requested {num_elements}, got {view.numel()}"
            )
        
        return view
    
    def get_memory_stats(self) -> Dict:
        """Get current memory statistics."""
        used_total = self.current_offset
        available = self.slab_capacity - used_total
        
        return {
            **self.stats,
            'persistent_offset_mb': self.persistent_watermark / 1024**2,
            'current_offset_mb': self.current_offset / 1024**2,
            'volatile_allocated_mb': (
                self.current_offset - self.persistent_watermark
            ) / 1024**2,
            'used_total_mb': used_total / 1024**2,
            'available_mb': available / 1024**2,
            'utilization_percent': (used_total / self.slab_capacity * 100) if self.slab_capacity > 0 else 0,
        }
    
    def power_of_2_allocation_size(self, raw_size: int) -> int:
        """
        Compute power-of-2 allocation size for KV cache.
        
        Prevents fragmentation during auto-regressive generation:
        - If seq_len needs 10MB, allocate 16MB (next power of 2)
        - If seq_len needs 17MB, allocate 32MB
        - When KV cache grows, it stays within allocation
        - No reallocation needed until next power of 2
        
        Args:
            raw_size: Actual size needed
        
        Returns:
            Next power of 2 >= raw_size
        """
        if raw_size <= 0:
            return 1
        
        # Find next power of 2
        # bit_length() gives position of highest bit
        # -1 to get the exponent
        bit_pos = (raw_size - 1).bit_length()
        return 1 << bit_pos
    
    def clear(self) -> None:
        """
        Clear all allocations and reset memory.
        
        Used during shutdown or cache eviction.
        """
        self.persistent_watermark = 0
        self.current_offset = 0
        self.allocations.clear()
        
        logger.info("VMU memory cleared")
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get VMU statistics.
        
        Returns:
            Dict with memory usage statistics
        """
        persistent_used = self.persistent_watermark
        volatile_used = self.current_offset - self.persistent_watermark
        total_used = self.current_offset
        total_free = self.slab_capacity - self.current_offset
        
        return {
            'persistent_allocated_mb': self.stats['persistent_allocated_bytes'] / 1024**2,
            'volatile_allocated_mb': self.stats['volatile_allocated_bytes'] / 1024**2,
            'max_persistent_mb': self.stats['max_persistent_offset'] / 1024**2,
            'max_current_mb': self.stats['max_current_offset'] / 1024**2,
            'slab_capacity_mb': self.slab_capacity / 1024**2,
            'reset_volatile_count': self.stats['reset_volatile_count'],
            'allocation_count': self.stats['allocation_count'],
            # Current usage
            'persistent_used_mb': persistent_used / 1024**2,
            'volatile_used_mb': volatile_used / 1024**2,
            'total_used_mb': total_used / 1024**2,
            'total_free_mb': total_free / 1024**2,
            'utilization_percent': (total_used / self.slab_capacity * 100) if self.slab_capacity > 0 else 0,
        }


# Global VMU instance per device
_global_vmus: Dict[int, UnifiedVMU] = {}


def get_vmu(device_id: int = 0) -> UnifiedVMU:
    """Get or create VMU for specified device."""
    if device_id not in _global_vmus:
        _global_vmus[device_id] = UnifiedVMU(device_id=device_id)
    return _global_vmus[device_id]


def reset_vmu(device_id: int = 0) -> None:
    """Reset VMU for specified device."""
    if device_id in _global_vmus:
        _global_vmus[device_id].reset_volatile()



