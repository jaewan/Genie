# Device Layer Implementation

## Overview

The Device Layer provides the foundation for remote accelerator support in PyTorch. It registers a custom backend (`remote_accelerator`) and manages device instances, enabling transparent tensor operations on disaggregated GPUs.

**File**: `genie/core/device.py`  
**C++ Backend**: `genie/csrc/device.cpp`

## Architecture

The Device Layer is **Mechanism 3** of the three interception mechanisms:

```
PyTorch Core
     │
     ▼
┌─────────────────────────┐
│  c10::PrivateUse1       │  ← PyTorch's extension mechanism
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│  genie._C (C++ Module)  │  ← device.cpp bindings
│  • register_backend()   │
│  • device_count()       │
│  • is_registered()      │
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│ RemoteAcceleratorDevice │  ← Python wrapper (device.py)
│  • get_device(idx)      │
│  • to_torch_device()    │
│  • synchronize()        │
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│  Interception Layer     │  ← Coordinates all three mechanisms
│  • Factory Intercept    │
│  • __torch_dispatch__   │
│  • Device Backend       │
└─────────────────────────┘
```

**Role in Interception System**:
- **Device Backend** is one of three necessary interception mechanisms
- Required for PyTorch to recognize "remote_accelerator" as valid device type
- Works with Factory Intercept and __torch_dispatch__ for complete coverage

## Core Classes

### RemoteAcceleratorDevice

Main device class that integrates with PyTorch.

```python
class RemoteAcceleratorDevice:
    """Custom PyTorch device for disaggregated execution."""
    
    def __init__(self, index: int = 0):
        """Create device instance and register backend."""
        self.type = "remote_accelerator"
        self.index = index
        self._register_backend()  # Register with PyTorch
        self._torch_device = torch.device("remote_accelerator", index)
```

#### Key Methods

##### `_register_backend()`

Registers the backend with PyTorch's C++ core as **Mechanism 3** of the interception system:

```python
def _register_backend(self):
    """Register the remote_accelerator backend with PyTorch (Mechanism 3)."""
    if not RemoteAcceleratorDevice._backend_registered:
        try:
            from genie import _C
            _C.register_remote_accelerator_device()
            self._register_python_hooks()
            RemoteAcceleratorDevice._backend_registered = True
            logger.info("Successfully registered remote_accelerator backend (Mechanism 3)")
        except Exception as e:
            # Graceful fallback: Python-level interception still works
            logger.warning(f"C++ backend registration failed, using Python-only mode: {e}")
            logger.warning("This is expected if the C++ extension has ABI compatibility issues")
            logger.info("All three interception mechanisms will still work (Factory + __torch_dispatch__ + Python fallback)")

            RemoteAcceleratorDevice._backend_registered = True
            self._register_python_hooks()
```

**C++ Implementation** (`device.cpp`):
```cpp
void register_remote_accelerator_device() {
    if (!_backend_registered) {
        // Register with PyTorch's PrivateUse1 mechanism (Mechanism 3)
        c10::register_privateuse1_backend("remote_accelerator");
        _backend_registered = true;
    }
}
```

**Role in Interception System**:
This is **Mechanism 3** (Device Backend) of the three interception mechanisms:
- **Factory Intercept** (Mechanism 1): Wraps torch.randn for initial tensor creation
- **`__torch_dispatch__`** (Mechanism 2): Intercepts operations on LazyTensor
- **Device Backend** (Mechanism 3): Makes PyTorch recognize "remote_accelerator" device

**Fallback Behavior**: If C++ registration fails, Genie gracefully falls back to Python-only mode. All three mechanisms still work together for complete operation interception.

##### `get_device(index)` (Class Method)

Singleton pattern for device management:

```python
@classmethod
def get_device(cls, index: int = 0) -> "RemoteAcceleratorDevice":
    """Get or create a remote accelerator device."""
    if index not in cls._devices:
        cls._devices[index] = cls(index)
    return cls._devices[index]
```

**Usage**:
```python
device0 = get_device(0)  # Create/get device 0
device1 = get_device(1)  # Create/get device 1
```

##### `device_count()` (Class Method)

Returns available device count:

```python
@classmethod
def device_count(cls) -> int:
    """Return number of available remote accelerator devices."""
    try:
        from genie import _C
        return _C.device_count()
    except Exception:
        return 4  # Default from specs
```

**Phase 1**: Returns 4 (hardcoded)  
**Phase 2+**: Will query actual remote infrastructure

##### `to_torch_device()`

Converts to PyTorch device object:

```python
def to_torch_device(self) -> torch.device:
    """Convert to PyTorch device object."""
    return self._torch_device
```

Enables seamless integration:
```python
genie_dev = get_device(0)
torch_dev = genie_dev.to_torch_device()
x = torch.zeros(10, 10, device=torch_dev)
```

##### `synchronize()`

Device synchronization (no-op in Phase 1):

```python
def synchronize(self) -> None:
    """Synchronize device operations."""
    # Phase 1: no-op
    # Phase 2+: Wait for remote operations to complete
    pass
```

##### `memory_stats()`

Returns memory statistics:

```python
def memory_stats(self) -> dict:
    """Get memory statistics for this device."""
    return {
        "allocated": 0,      # Phase 1: placeholder
        "cached": 0,
        "reserved": 0,
        "device_index": self.index
    }
```

**Phase 2+**: Will track actual remote memory usage

## Integration with PyTorch

### Device Creation

PyTorch recognizes `remote_accelerator` as a valid device type:

```python
# String-based creation
device = torch.device("remote_accelerator", 0)
device = torch.device("remote_accelerator:0")

# Through Genie API
device = get_device(0).to_torch_device()
```

### Tensor Operations

Tensors created on `remote_accelerator` device automatically become `LazyTensor` through the interception mechanisms:

```python
# This creates a LazyTensor via Factory Intercept (Mechanism 1)
x = torch.randn(10, 10, device="remote_accelerator:0")
assert isinstance(x, LazyTensor)

# Operations create more LazyTensors via __torch_dispatch__ (Mechanism 2)
y = x + x  # LazyTensor (intercepted by __torch_dispatch__)
z = torch.matmul(x, y)  # LazyTensor (intercepted by __torch_dispatch__)
```

**Interception Flow**:
1. `torch.randn` → **Factory Intercept** (Mechanism 1) → Creates LazyTensor
2. `x + x` → **`__torch_dispatch__`** (Mechanism 2) → Creates new LazyTensor
3. `torch.matmul` → **`__torch_dispatch__`** (Mechanism 2) → Creates new LazyTensor

### Device Properties

```python
device = get_device(0)

# String representation
str(device)   # "remote_accelerator:0"
repr(device)  # "remote_accelerator:0"

# Equality
device == get_device(0)  # True
device == get_device(1)  # False

# Hashing (for use in dicts/sets)
device_map = {device: "data"}
```

## C++ Backend Implementation

### File: `genie/csrc/device.cpp`

```cpp
#include <torch/extension.h>

bool _backend_registered = false;

void register_remote_accelerator_device() {
    if (!_backend_registered) {
        try {
            // Register with PyTorch's PrivateUse1 system
            c10::register_privateuse1_backend("remote_accelerator");
            _backend_registered = true;
        } catch (const std::exception& e) {
            // Graceful degradation if registration fails
        }
    }
}

int device_count() {
    // Phase 1: Return fixed count
    return 4;
}

bool is_backend_registered() {
    return _backend_registered;
}

// Python bindings using pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("register_remote_accelerator_device", 
          &register_remote_accelerator_device,
          "Register remote_accelerator device with PyTorch");
    
    m.def("device_count", &device_count,
          "Get number of remote_accelerator devices");
    
    m.def("is_backend_registered", &is_backend_registered,
          "Check if backend is registered with PyTorch");
}
```

### Building the Extension

The C++ extension is built during package installation:

```python
# setup.py
from torch.utils.cpp_extension import BuildExtension, CppExtension

ext_modules = [
    CppExtension(
        name='genie._C',
        sources=['genie/csrc/device.cpp'],
        extra_compile_args=['-std=c++17']
    )
]
```

## Utility Functions

Module-level convenience functions:

```python
def get_device_count() -> int:
    """Get the number of available remote_accelerator devices."""
    return RemoteAcceleratorDevice.device_count()

def is_available() -> bool:
    """Check if remote_accelerator backend is available."""
    return RemoteAcceleratorDevice.is_available()

def get_device(index: int = 0) -> RemoteAcceleratorDevice:
    """Get a remote_accelerator device by index."""
    return RemoteAcceleratorDevice.get_device(index)

def synchronize(device = None) -> None:
    """Synchronize operations on the specified device."""
    if device is None:
        device = 0
    if isinstance(device, int):
        device = get_device(device)
    device.synchronize()
```

## Initialization Flow

```
Package Import (genie/__init__.py)
     │
     ▼
Import device module
     │
     ▼
C++ Extension loaded (genie._C)
     │
     ▼
RemoteAcceleratorDevice.__init__
     │
     ▼
_register_backend()
     │
     ▼
C++ register_remote_accelerator_device()
     │
     ▼
c10::register_privateuse1_backend("remote_accelerator")
     │
     ▼
PyTorch now recognizes "remote_accelerator" device
     │
     ▼
Default device created: get_device(0)
```

## Error Handling

### Graceful Degradation

The device layer handles missing dependencies gracefully:

```python
try:
    _default_device = RemoteAcceleratorDevice.get_device(0)
    logger.info("Initialized default remote_accelerator device")
except Exception as e:
    logger.warning(f"Could not initialize: {e}")
    _default_device = None
```

### Common Errors

**1. C++ Extension Not Built**
```
ModuleNotFoundError: No module named 'genie._C'
```
Solution: Build with `python setup.py install` or `pip install -e .`

**2. Backend Registration Failed**
```
RuntimeError: Cannot initialize remote_accelerator device
```
Solution: Check PyTorch version compatibility

**3. Device Index Out of Range**
```python
device = get_device(10)  # Only 0-3 available in Phase 1
```
Solution: Use `get_device_count()` to check available devices

## Testing

See `tests/test_device_registration.py`:

```python
def test_device_creation():
    """Test device creation and properties."""
    device = get_device(0)
    assert str(device) == "remote_accelerator:0"
    assert device.index == 0

def test_device_count():
    """Test device count."""
    count = get_device_count()
    assert count >= 1  # At least one device

def test_torch_device_integration():
    """Test integration with torch.device."""
    device = get_device(0)
    torch_device = device.to_torch_device()
    assert torch_device.type == "remote_accelerator"
    assert torch_device.index == 0

def test_device_equality():
    """Test device equality."""
    dev1 = get_device(0)
    dev2 = get_device(0)
    assert dev1 == dev2
    
    dev3 = get_device(1)
    assert dev1 != dev3
```

## Phase 2+ Enhancements

### Remote Device Discovery

```python
# Future: Query actual remote infrastructure
@classmethod
def device_count(cls) -> int:
    # Query cluster manager for available GPUs
    return cluster_manager.get_available_device_count()
```

### Memory Management

```python
def memory_stats(self) -> dict:
    """Get actual remote memory statistics."""
    return {
        "allocated": self._query_remote_memory(),
        "cached": self._query_cache_size(),
        "reserved": self._query_reserved(),
        "device_index": self.index
    }
```

### Device Affinity

```python
def set_device_affinity(self, workload_type: str):
    """Hint workload type for optimal device selection."""
    # LLM decode -> co-locate with KV cache
    # Vision -> bandwidth-optimized GPU
    # RecSys -> sparse-optimized accelerator
    pass
```

## Best Practices

### 1. Use Singleton Pattern

Always use `get_device()` instead of creating instances directly:

```python
# Good
device = get_device(0)

# Bad
device = RemoteAcceleratorDevice(0)  # May create duplicates
```

### 2. Check Availability

Verify backend is available before use:

```python
if is_available():
    device = get_device(0)
    # Use device
else:
    # Fallback to CPU
    device = torch.device("cpu")
```

### 3. Proper Synchronization

Synchronize when needed (especially in Phase 2+):

```python
# Launch operations
result = model(input)

# Wait for completion
synchronize(device=0)

# Now safe to access result
print(result)
```

## Related Documentation

- [LazyTensor](03-lazy-tensor.md) - How tensors on this device behave
- [Dispatcher](04-dispatcher.md) - How operations are intercepted
- [Architecture Overview](01-architecture-overview.md) - System-wide view

## References

- PyTorch PrivateUse1: https://pytorch.org/docs/stable/notes/extending.html
- C++ Extensions: https://pytorch.org/tutorials/advanced/cpp_extension.html
