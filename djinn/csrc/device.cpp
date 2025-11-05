#include <torch/extension.h>

// Simple device registration for Phase 1
// This is a minimal implementation that focuses on getting the backend name registered

bool _backend_registered = false;

void register_remote_accelerator_device() {
  if (!_backend_registered) {
    try {
      // Register the backend name with PyTorch
      // This uses PyTorch's PrivateUse1 device type
      c10::register_privateuse1_backend("remote_accelerator");
      _backend_registered = true;
    } catch (const std::exception& e) {
      // If registration fails, we'll handle it gracefully in Python
      // This allows the system to work even without full C++ integration
    }
  }
}

int device_count() {
  // Phase 1: report 4 devices as per specs
  return 4;
}

bool is_backend_registered() {
  return _backend_registered;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("register_remote_accelerator_device", &register_remote_accelerator_device,
        "Register remote_accelerator device with PyTorch");
  
  m.def("device_count", &device_count,
        "Get number of remote_accelerator devices");
        
  m.def("is_backend_registered", &is_backend_registered,
        "Check if backend is registered with PyTorch");
}


