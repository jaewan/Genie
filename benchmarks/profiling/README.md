# Profiling Tools

This directory contains essential profiling tools for network performance analysis.

## Available Tools

### 1. Network Breakdown Profiler
**File**: `network_breakdown_profiler.py`

Detailed breakdown of network communication overhead:
- Serialization time
- Network transfer time
- Deserialization time
- End-to-end latency

**Usage**:
```bash
python3 -m benchmarks.profiling.network_breakdown_profiler
```

### 2. Network Measurements Reconciliation
**File**: `reconcile_network_measurements.py`

Validates and reconciles network measurements across different baselines:
- Compares real network vs simulated
- Validates consistency
- Identifies anomalies

**Usage**:
```bash
python3 -m benchmarks.profiling.reconcile_network_measurements
```

## Purpose

These tools help:
1. **Debug network issues**: Identify bottlenecks in network communication
2. **Validate measurements**: Ensure network measurements are accurate
3. **Optimize performance**: Guide optimization efforts based on profiling data

## Integration with Main Benchmarks

The profiling tools complement the main benchmark suite (`comprehensive_evaluation.py`) by providing:
- Detailed breakdown of where time is spent
- Validation that measurements are correct
- Insights for optimization

## Output

Profiling results are typically saved as:
- JSON files (raw data)
- Console output (summary statistics)
- Can be integrated into benchmark reports

---

**Note**: All old profiling scripts have been removed. Only essential tools for OSDI evaluation remain.

