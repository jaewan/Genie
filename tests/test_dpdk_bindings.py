#!/usr/bin/env python3
"""
Test script for DPDK Python bindings.

This script tests the basic functionality of our DPDK bindings:
- Library loading
- EAL initialization
- Mempool creation
- Ethernet device discovery
- GPU device discovery (if available)

Run with: python tests/test_dpdk_bindings.py
"""

import sys
import os
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from genie.runtime.dpdk_bindings import get_dpdk, eal_init, is_dpdk_available

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_library_loading():
    """Test DPDK library loading."""
    print("=" * 60)
    print("Testing DPDK Library Loading")
    print("=" * 60)
    
    dpdk = get_dpdk()
    
    print(f"Core libraries available: {dpdk.libs.core_ok}")
    print(f"EAL library: {'✓' if dpdk.libs.eal else '✗'}")
    print(f"Mempool library: {'✓' if dpdk.libs.mempool else '✗'}")
    print(f"Mbuf library: {'✓' if dpdk.libs.mbuf else '✗'}")
    print(f"Ethdev library: {'✓' if dpdk.libs.ethdev else '✗'}")
    print(f"GPUDev library: {'✓' if dpdk.libs.gpudev else '✗'}")
    
    if not dpdk.is_available():
        print("❌ DPDK libraries not available - check installation")
        return False
    
    print("✅ DPDK libraries loaded successfully")
    return True

def test_eal_initialization():
    """Test EAL initialization."""
    print("\n" + "=" * 60)
    print("Testing EAL Initialization")
    print("=" * 60)
    
    # Test with default arguments
    success = eal_init()
    
    if not success:
        print("❌ EAL initialization failed")
        print("Common issues:")
        print("  - Run as root or with CAP_NET_ADMIN capability")
        print("  - Check hugepages are allocated: cat /proc/meminfo | grep Huge")
        print("  - Check hugepages are mounted: mount | grep huge")
        return False
    
    print("✅ EAL initialized successfully")
    
    dpdk = get_dpdk()
    print(f"EAL initialized: {dpdk.eal_initialized}")
    
    return True

def test_mempool_creation():
    """Test mempool creation."""
    print("\n" + "=" * 60)
    print("Testing Mempool Creation")
    print("=" * 60)
    
    dpdk = get_dpdk()
    
    if not dpdk.eal_initialized:
        print("❌ EAL not initialized - skipping mempool test")
        return False
    
    # Create a test mempool
    pool = dpdk.create_mempool("test_pool", n_mbufs=1024, data_room_size=2048)
    
    if not pool:
        print("❌ Failed to create mempool")
        return False
    
    print("✅ Mempool created successfully")
    print(f"Pool pointer: {hex(pool.value) if pool else 'None'}")
    
    # Test mbuf allocation
    mbuf = dpdk.alloc_mbuf(pool)
    if mbuf:
        print("✅ Mbuf allocation successful")
        print(f"Mbuf pointer: {hex(mbuf.value)}")
        dpdk.free_mbuf(mbuf)
        print("✅ Mbuf freed successfully")
    else:
        print("❌ Mbuf allocation failed")
        return False
    
    return True

def test_ethernet_devices():
    """Test ethernet device discovery."""
    print("\n" + "=" * 60)
    print("Testing Ethernet Device Discovery")
    print("=" * 60)
    
    dpdk = get_dpdk()
    
    if not dpdk.libs.ethdev_ok:
        print("⚠️  Ethdev library not available - skipping ethernet tests")
        return True
    
    eth_count = dpdk.get_eth_dev_count()
    print(f"Available ethernet devices: {eth_count}")
    
    if eth_count == 0:
        print("⚠️  No ethernet devices found")
        print("This is expected if NICs are not bound to DPDK")
        return True
    
    print("✅ Ethernet devices detected")
    
    # Try to configure the first device (if available)
    if eth_count > 0:
        print(f"Attempting to configure port 0...")
        success = dpdk.configure_eth_dev(0, nb_rx_queues=1, nb_tx_queues=1)
        if success:
            print("✅ Port 0 configured successfully")
        else:
            print("❌ Port 0 configuration failed")
    
    return True

def test_gpu_devices():
    """Test GPU device discovery."""
    print("\n" + "=" * 60)
    print("Testing GPU Device Discovery")
    print("=" * 60)
    
    dpdk = get_dpdk()
    
    if not dpdk.libs.gpudev_ok:
        print("⚠️  GPUDev library not available - skipping GPU tests")
        return True
    
    try:
        gpu_count = dpdk.get_gpu_count()
        print(f"Available GPU devices: {gpu_count}")
        
        if gpu_count == 0:
            print("⚠️  No GPU devices found")
            print("This is expected if CUDA toolkit is not installed")
        else:
            print("✅ GPU devices detected")
            
    except Exception as e:
        print(f"❌ GPU device discovery failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("DPDK Python Bindings Test Suite")
    print("=" * 60)
    
    # Check if running as root
    if os.geteuid() != 0:
        print("⚠️  WARNING: Not running as root")
        print("   Some tests may fail due to insufficient permissions")
        print("   Consider running with: sudo python tests/test_dpdk_bindings.py")
        print()
    
    tests = [
        ("Library Loading", test_library_loading),
        ("EAL Initialization", test_eal_initialization),
        ("Mempool Creation", test_mempool_creation),
        ("Ethernet Devices", test_ethernet_devices),
        ("GPU Devices", test_gpu_devices),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! DPDK bindings are working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
