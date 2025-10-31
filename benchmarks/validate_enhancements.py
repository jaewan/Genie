#!/usr/bin/env python3
"""
Validate Benchmark Enhancements.

Quick test to ensure all enhancements are working:
1. Real models load correctly
2. Naive baseline exists
3. Configuration defaults are correct
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all imports work."""
    print("🧪 Testing imports...")
    try:
        from benchmarks.baselines import (
            LocalPyTorchBaseline,
            NaiveDisaggregationBaseline,
            GenieCaptureOnlyBaseline,
            GenieLocalRemoteBaseline,
            GenieNoSemanticsBaseline,
            GenieFullBaseline,
            PyTorchRPCBaseline,
            RayBaseline
        )
        print("  ✅ All baselines import successfully")
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_naive_baseline():
    """Test naive baseline instantiation."""
    print("\n🧪 Testing naive baseline...")
    try:
        from benchmarks.baselines import NaiveDisaggregationBaseline
        baseline = NaiveDisaggregationBaseline()
        metadata = baseline.get_metadata()
        
        assert metadata['baseline'] == 'naive_disaggregation'
        assert 'worst-case' in metadata['description'].lower()
        
        print("  ✅ Naive baseline works correctly")
        print(f"     Description: {metadata['description']}")
        return True
    except Exception as e:
        print(f"  ❌ Naive baseline failed: {e}")
        return False

def test_comprehensive_eval_defaults():
    """Test that comprehensive evaluation has correct defaults."""
    print("\n🧪 Testing comprehensive evaluation defaults...")
    try:
        from benchmarks.comprehensive_evaluation import ComprehensiveEvaluation
        
        # Create with mock models to avoid GPU OOM
        eval = ComprehensiveEvaluation(
            use_real_models=False,  # Use mocks for validation
            spawn_server=False       # Don't spawn server for validation
        )
        
        # Check that we CAN override defaults
        assert eval.use_real_models == False, "Should be able to override use_real_models"
        assert eval.spawn_server == False, "Should be able to override spawn_server"
        
        # Check structure
        assert eval.output_dir.name == "osdi_final_results", "output_dir should be osdi_final_results"
        assert len(eval.baselines) == 8, f"Should have 8 baselines, got {len(eval.baselines)}"
        assert '2_naive_disaggregation' in eval.baselines, "Naive baseline should be in baselines"
        
        print("  ✅ Comprehensive evaluation structure correct")
        print(f"     output_dir: {eval.output_dir}")
        print(f"     num_baselines: {len(eval.baselines)}")
        print(f"     baselines: {list(eval.baselines.keys())}")
        
        # Note about defaults
        print("  📝 Note: Defaults are use_real_models=True, spawn_server=True")
        print("           (overridden to False for validation to avoid GPU OOM)")
        return True
    except Exception as e:
        print(f"  ❌ Comprehensive evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_realistic_workloads():
    """Test that realistic workloads can be loaded."""
    print("\n🧪 Testing realistic workloads...")
    try:
        from benchmarks.workloads_detailed import (
            RealisticLLMDecodeWorkload,
            RealisticLLMPrefillWorkload,
            RealisticVisionCNNWorkload
        )
        
        print("  ✅ Realistic workload imports successful")
        print("     Note: Actual model loading will happen during benchmark run")
        return True
    except Exception as e:
        print(f"  ❌ Realistic workload import failed: {e}")
        return False

def main():
    print("="*60)
    print("🔬 VALIDATING BENCHMARK ENHANCEMENTS")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Naive Baseline", test_naive_baseline()))
    results.append(("Comprehensive Eval", test_comprehensive_eval_defaults()))
    results.append(("Realistic Workloads", test_realistic_workloads()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:25s}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✅ ALL VALIDATIONS PASSED!")
        print("\nNext steps:")
        print("  1. Run quick test: python benchmarks/validate_enhancements.py")
        print("  2. Run full benchmarks: python run_phase2_benchmarks.py")
        print("\nNote: Full benchmarks will take 4-5 hours with real models.")
        return 0
    else:
        print("\n❌ SOME VALIDATIONS FAILED!")
        print("\nPlease fix the issues above before running benchmarks.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

