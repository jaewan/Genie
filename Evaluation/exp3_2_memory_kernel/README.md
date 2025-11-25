# Experiment 3.2 – Memory Kernel Deep Dive

Demonstrates VMU’s OS-level guarantees with Djinn-specific workloads (concurrent
stateful sessions). Mirrors §6.3 in `docs/EvaluationPlan.md`.

## Goals
1. Show zero external fragmentation under allocate/use/free churn.
2. Prove sub-20 µs allocation latency vs. cudaMalloc / PyTorch caching allocator.
3. Verify session isolation: each conversation’s KV cache is fenced.

## Structure
- `configs/session_stress.yaml` – parameters for session workloads and allocator
  backends.
- `scripts/run_memory_stress.py` – orchestrates synthetic sessions, collects
  fragmentation/latency stats.
- `results/` – raw metrics (JSON/CSV), one per allocator/run.

## Usage
```bash
source .venv/bin/activate
python Evaluation/exp3_2_memory_kernel/scripts/run_memory_stress.py \
  --config Evaluation/exp3_2_memory_kernel/configs/session_stress.yaml \
  --allocator vmu --duration 30 --output Evaluation/exp3_2_memory_kernel/results/vmu_run1.json
```

You can swap `--allocator` between `vmu`, `cudacache`, and `torch_cache`.

## Next steps
- Integrate with actual VMU API (current script uses simulated allocators to
  confirm harness behavior).
- Plug in device telemetry to capture real fragmentation data on the GPU.


