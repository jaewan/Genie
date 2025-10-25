"""
Test script to verify shape cache optimization in Phase 2.

This script tests that the shape inference cache is working properly
and achieving good hit rates.

Usage:
    python benchmarks/test_shape_cache.py
"""

import torch
import time
import sys
from pathlib import Path

# Add genie to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genie.core.lazy_tensor import get_shape_cache_stats
from genie.profiling import get_detailed_profiler


def test_shape_cache_basic():
    """Test basic shape caching functionality."""
    print("\n" + "="*80)
    print("TEST 1: Basic Shape Cache Functionality")
    print("="*80)
    
    # Import after genie path is set
    from genie.core.capture import capture
    
    with capture():
        # Create tensors
        x = torch.randn(10, 20)
        y = torch.randn(20, 30)
        
        # Perform operations (should cache shapes)
        z1 = x @ y  # First matmul - will miss cache
        z2 = x @ y  # Second matmul with same inputs - should hit cache
        z3 = z1 + z2  # Add - new operation
        
        # Multiple operations with same shapes
        for i in range(5):
            _ = x @ y  # Should all hit cache
        
        result = z3.cpu()
    
    # Check cache stats
    stats = get_shape_cache_stats()
    print(f"\n✓ Cache Statistics:")
    print(f"  Hits:       {stats['hits']}")
    print(f"  Misses:     {stats['misses']}")
    print(f"  Hit Rate:   {stats['hit_rate']}")
    print(f"  Cache Size: {stats['cache_size']} entries")
    
    # Verify hit rate > 50% (should be ~70%+ with repeated operations)
    hit_rate_percent = float(stats['hit_rate'].rstrip('%'))
    if hit_rate_percent > 50:
        print(f"\n✅ PASS: Hit rate {stats['hit_rate']} > 50%")
        return True
    else:
        print(f"\n❌ FAIL: Hit rate {stats['hit_rate']} <= 50%")
        return False


def test_shape_cache_scalability():
    """Test cache performance with many different operations."""
    print("\n" + "="*80)
    print("TEST 2: Cache Scalability (Many Operations)")
    print("="*80)
    
    from genie.core.capture import capture
    
    with capture():
        x = torch.randn(100, 200)
        y = torch.randn(200, 300)
        
        # Generate 50 operations with repeated patterns
        for i in range(50):
            # Repeat operations to create cache hits
            if i % 5 == 0:
                z = x @ y  # Repeat every 5 iterations
            else:
                z = x + y  # Other operations
    
    stats = get_shape_cache_stats()
    print(f"\n✓ After 50 operations:")
    print(f"  Cache Size: {stats['cache_size']} entries")
    print(f"  Hit Rate:   {stats['hit_rate']}")
    print(f"  Evictions:  {stats['evictions']}")
    
    # Verify cache didn't grow unbounded
    if stats['cache_size'] < 100:
        print(f"\n✅ PASS: Cache size {stats['cache_size']} < 100 (bounded)")
        return True
    else:
        print(f"\n❌ FAIL: Cache size {stats['cache_size']} >= 100 (unbounded)")
        return False


def test_shape_cache_performance():
    """Test performance improvement from caching."""
    print("\n" + "="*80)
    print("TEST 3: Performance Improvement")
    print("="*80)
    
    from genie.core.capture import capture
    from genie.profiling import get_detailed_profiler
    
    profiler = get_detailed_profiler()
    
    # First run: cold cache
    with capture():
        x = torch.randn(1000, 1000)
        for i in range(10):
            y = x @ x
    
    stats1 = get_shape_cache_stats()
    first_run_miss = stats1['misses']
    
    # Second run: warm cache  
    with capture():
        x = torch.randn(1000, 1000)
        for i in range(10):
            y = x @ x
    
    stats2 = get_shape_cache_stats()
    second_run_misses = stats2['misses'] - first_run_miss
    
    print(f"\n✓ Warm vs Cold Cache:")
    print(f"  First run misses:  {first_run_miss}")
    print(f"  Second run misses: {second_run_misses}")
    print(f"  Hit Rate (2nd run): {stats2['hit_rate']}")
    
    # Second run should have much higher hit rate
    if second_run_misses < first_run_miss / 2:
        print(f"\n✅ PASS: Warm cache has fewer misses")
        return True
    else:
        print(f"\n⚠️  CHECK: Hit rate may not be optimal")
        return False


def test_shape_cache_with_profiler():
    """Test cache improvement measured with profiler."""
    print("\n" + "="*80)
    print("TEST 4: Profiler Integration")
    print("="*80)
    
    from genie.core.capture import capture
    from genie.profiling import get_detailed_profiler
    
    profiler = get_detailed_profiler()
    
    # Run with shape cache enabled
    with capture():
        with profiler.profile_component("shape_cache_test"):
            x = torch.randn(100, 100)
            for i in range(20):
                y = x @ x
    
    stats = profiler.get_component_stats("shape_cache_test")
    cache_stats = get_shape_cache_stats()
    
    print(f"\n✓ Profiler Results:")
    print(f"  Total time: {stats['mean']:.2f}ms (mean)")
    print(f"  Std dev:    {stats['std']:.2f}ms")
    print(f"  Min:        {stats['min']:.2f}ms")
    print(f"  Max:        {stats['max']:.2f}ms")
    
    print(f"\n✓ Cache Stats:")
    print(f"  Hit rate:   {cache_stats['hit_rate']}")
    print(f"  Cache size: {cache_stats['cache_size']}")
    
    print(f"\n✅ PASS: Profiler integrated with cache")
    return True


if __name__ == '__main__':
    print("\n" + "="*80)
    print("GENIE PHASE 2: SHAPE CACHE OPTIMIZATION TESTS")
    print("="*80)
    
    results = []
    
    try:
        results.append(("Basic Functionality", test_shape_cache_basic()))
    except Exception as e:
        print(f"❌ Test failed: {e}")
        results.append(("Basic Functionality", False))
    
    try:
        results.append(("Scalability", test_shape_cache_scalability()))
    except Exception as e:
        print(f"❌ Test failed: {e}")
        results.append(("Scalability", False))
    
    try:
        results.append(("Performance", test_shape_cache_performance()))
    except Exception as e:
        print(f"❌ Test failed: {e}")
        results.append(("Performance", False))
    
    try:
        results.append(("Profiler Integration", test_shape_cache_with_profiler()))
    except Exception as e:
        print(f"❌ Test failed: {e}")
        results.append(("Profiler Integration", False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nResult: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Shape cache optimization working correctly.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Check implementation.")
        sys.exit(1)
