# Experiment 2.1: LLM Decode with KV Cache - Results Summary

**Model:** GPT2-XL  
**Prompt Length:** 72 tokens  
**New Tokens:** 50 tokens  
**Runs per Baseline:** 30  
**Date:** 2025-11-25

## Results

| Baseline | Mean Latency (ms) | P95 Latency (ms) | Mean Per-Token (ms) | Data Transfer (MB) |
|----------|-------------------|------------------|---------------------|-------------------|
| **Native PyTorch** | 979.87 | 983.91 | 19.60 | 0.00 |
| **Semantic-Blind** | 6054.17 | 6096.33 | 121.08 | 925.10 |
| **Partially-Aware** | 6084.93 | 6194.48 | 121.70 | 925.10 |
| **Full Djinn** | 6072.84 | 6126.18 | 121.46 | 925.10 |

## Key Observations

1. **Native PyTorch Performance:** 979ms for 50 tokens (~19.6ms/token) - this is the upper bound for local execution.

2. **Remote Baselines:** All three remote baselines show similar performance (~6 seconds), which is unexpected. This suggests:
   - Semantic optimizations may not be fully enabled/working
   - KV cache persistence might not be reducing data transfer
   - Model caching is working (all baselines use cached model)

3. **Data Transfer:** All remote baselines transfer ~925MB per run, indicating:
   - Semantic hints are not reducing data transfer for decode phase
   - KV cache handles are not being used (full cache is being sent each time)
   - Semantic-blind baseline is correctly not using hints, but model caching still applies

4. **Performance Gap:** Remote execution is ~6× slower than native, primarily due to:
   - Network overhead
   - Serialization/deserialization overhead
   - Lack of semantic optimizations (KV cache persistence)

## Comparison Metrics

- **Full Djinn vs Native PyTorch:** 0.16× speedup (6.2× slower)
- **Full Djinn vs Semantic-Blind:** 1.00× speedup (no improvement)
- **Data Savings:** 0% (semantic hints not reducing transfer)

## Next Steps

1. **Investigate why semantic optimizations aren't working:**
   - Check if KV cache persistence is enabled on server
   - Verify semantic hints are being processed correctly
   - Ensure decode phase detection is working

2. **For LLaMA-7B evaluation:**
   - Need to resolve GPU memory issues
   - LLaMA-7B should show more dramatic differences between baselines
   - Expected: 50× speedup, 99.8% data savings (per EvaluationPlan.md)

3. **Improve semantic-blind baseline:**
   - Ensure it truly disables all semantic features
   - Should show worse performance than partially-aware/full-djinn

## Files Generated

- `native_pytorch/llm_decode_native_pytorch_20251125T002654Z.json`
- `semantic_blind/llm_decode_semantic_blind_20251125T003056Z.json`
- `partially_aware/llm_decode_partially_aware_20251125T003454Z.json`
- `full_djinn/llm_decode_full_djinn_20251125T003851Z.json`
- `comparison_report_gpt2xl_full_20251125T003852Z.json`
