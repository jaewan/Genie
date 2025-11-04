# Benchmark Scripts Guide

This directory contains **optional tmux helpers** for running benchmarks as detachable background jobs.

## ⚠️ Important: Use the Python CLI by Default

For most users and use cases, **use the Python CLI** instead:

```bash
# This is the recommended way
python3 run_benchmarks.py --baselines genie_full local_pytorch ray
python3 run_benchmarks.py --all-baselines
python3 run_profiling.py --workload gpt2
```

## 📝 When to Use These Scripts

Use the tmux scripts **only if** you need:
- Long-running benchmarks (3+ hours)
- The ability to detach and reattach (work on other things)
- Session persistence across terminal closes
- Real-time monitoring in a separate tmux window

## 🚀 Available Scripts

### 1. `run_full_tmux.sh` - Full Benchmark Suite

Runs all 8 baselines across all 5 workloads in a single tmux session.

```bash
bash benchmarks/scripts/run_full_tmux.sh
```

**Use when**: You want complete OSDI evaluation as a detachable job

**Configuration**:
- Baselines: 8 (all)
- Workloads: 5
- Runs: 20 measurement + 3 warmup per combination
- Total: ~800 experiments
- Duration: ~4-5 hours
- Output: `osdi_final_results/`

**Monitoring**:
```bash
tmux attach -t genie_benchmarks
Ctrl+B, then D to detach
```

---

### 2. `run_sequential_tmux.sh` - Sequential with GPU Cleanup

Runs baselines sequentially, cleaning GPU memory between each.

```bash
bash benchmarks/scripts/run_sequential_tmux.sh
```

**Use when**: You want clean GPU state between runs (prevents memory accumulation)

**Configuration**:
- Mode: Sequential (one baseline at a time)
- GPU cleanup: Yes (between each)
- Baselines: 8 (all)
- Duration: ~4-5 hours
- Output: `benchmark_results_<TIMESTAMP>/`

**Monitoring**:
```bash
tmux attach -t genie_benchmarks_seq
```

---

### 3. `run_sustained_tmux.sh` - Sustained Execution Benchmarks

Runs benchmarks under sustained load with cache statistics.

```bash
# Default: 1000 requests
bash benchmarks/scripts/run_sustained_tmux.sh

# Custom number of requests
bash benchmarks/scripts/run_sustained_tmux.sh 500
```

**Use when**: You want to measure cache hit rates and performance under repeated workloads

**Configuration**:
- Requests: 1000 (customizable)
- Measurement: Every 10 requests
- Models: GPT-2-XL (1.5B), ResNet-50 (25M)
- Baselines: 3 (Local, Genie Full, PyTorch RPC)
- Duration: ~2-3 hours
- Output: `sustained_execution_real_results/`

**Monitoring**:
```bash
tmux attach -t genie_sustained
```

---

## 🔄 Comparison: CLI vs Tmux Scripts

| Feature | Python CLI | Tmux Scripts |
|---------|-----------|--------------|
| **Ease of use** | ✅ Simpler | ⚠️ More setup |
| **Flexibility** | ✅ Many flags | ❌ Fixed behavior |
| **Detachable** | ❌ Blocks terminal | ✅ Runs in background |
| **Long jobs** | ❌ Can't close laptop | ✅ Stays running |
| **Logging** | ✅ Built-in | ✅ Auto-logged |
| **Error handling** | ✅ Better | ⚠️ Less obvious |

---

## 💡 Recommended Usage Patterns

### Quick Validation
```bash
python3 run_benchmarks.py --only-genie-full
```

### Full OSDI Evaluation (Interactive)
```bash
python3 run_benchmarks.py --all-baselines
```

### Full Evaluation (Detachable)
```bash
bash benchmarks/scripts/run_full_tmux.sh
tmux attach -t genie_benchmarks
# ... monitor progress ...
# Ctrl+B, D to detach
```

### Specific Baselines
```bash
python3 run_benchmarks.py --baselines genie_full local_pytorch ray
```

### Sustained Execution
```bash
bash benchmarks/scripts/run_sustained_tmux.sh 500
tmux attach -t genie_sustained
```

### Profiling
```bash
python3 run_profiling.py --workload gpt2
```

---

## 📊 Output Locations

| Script | Output Directory | Files |
|--------|------------------|-------|
| `run_full_tmux.sh` | `osdi_final_results/` | `results.json`, LaTeX tables, PDFs |
| `run_sequential_tmux.sh` | `benchmark_results_<TIMESTAMP>/` | Per-baseline directories with results |
| `run_sustained_tmux.sh` | `sustained_execution_real_results/` | Cache statistics, latency data |

---

## 🔍 Troubleshooting

### Tmux Session Already Exists
```bash
# List sessions
tmux ls

# Kill old session
tmux kill-session -t genie_benchmarks

# Then run script again
bash benchmarks/scripts/run_full_tmux.sh
```

### Check Benchmark Progress
```bash
# Attach and view output
tmux attach -t genie_benchmarks

# Or tail the log file
tail -f benchmark_logs_*/full_suite_*.log
```

### GPU Out of Memory
The sequential runner cleans GPU memory between runs. Use:
```bash
bash benchmarks/scripts/run_sequential_tmux.sh
```

### Kill Running Benchmarks
```bash
# Inside tmux session
Ctrl+C

# Or from outside
tmux send-keys -t genie_benchmarks C-c
```

---

## 📚 Integration with Main Benchmarks

These scripts are **wrappers** around:
- `run_benchmarks.py` - Main Python entry point
- `benchmarks/comprehensive_evaluation.py` - Core implementation
- `benchmarks/sustained_execution_real.py` - Sustained execution

For advanced users who want custom orchestration, edit the scripts or use the Python API directly:

```python
from benchmarks.comprehensive_evaluation import ComprehensiveEvaluation

eval = ComprehensiveEvaluation(use_real_models=True, spawn_server=True)
await eval.run_all(num_runs=20, num_warmup=3)
```

---

## ✅ Quick Reference

| Goal | Command |
|------|---------|
| Quick test | `python3 run_benchmarks.py --only-genie-full` |
| Full suite (interactive) | `python3 run_benchmarks.py --all-baselines` |
| Full suite (tmux) | `bash benchmarks/scripts/run_full_tmux.sh` |
| Sequential (clean GPU) | `bash benchmarks/scripts/run_sequential_tmux.sh` |
| Sustained execution | `bash benchmarks/scripts/run_sustained_tmux.sh` |
| Profiling | `python3 run_profiling.py --workload gpt2` |
| Specific baselines | `python3 run_benchmarks.py --baselines genie_full pytorch_rpc` |

---

**Last Updated**: November 3, 2025  
**Status**: Tmux scripts are optional helpers - use Python CLI for primary usage
