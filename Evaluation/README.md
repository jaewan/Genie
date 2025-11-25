# Djinn Evaluation Workspace

This directory hosts runnable artifacts for the OSDI evaluation plan described in
`docs/EvaluationPlan.md`.  Each experiment lives in its own subdirectory with
configs, scripts, and raw result storage so we can iterate without touching the
core Djinn tree.

## Structure
- `exp2_1_llm_decode/` – Experiment 2.1 (hero evaluation). Future experiments
  will follow the same layout (`configs/`, `scripts/`, `results/`, `notes/`).

## Workflow
1. **Plan** – translate the requirements from `docs/EvaluationPlan.md` into a
   per-experiment README that specifies workload, metrics, and pass criteria.
2. **Implement** – add scripts in `scripts/` plus any helper configs/templates.
3. **Run** – capture ≥30 runs per configuration, write JSON/CSV into
   `results/<baseline>/<timestamp>.json`.
4. **Analyze** – notebooks or scripts emit the figures referenced in the plan.

## Environment
- Python ≥3.10
- PyTorch + CUDA toolchain that matches Djinn runtime
- Hugging Face `transformers` for baseline workloads
- Optional: `pynvml` for GPU utilization sampling (see experiment README)

Each script declares its own requirements; prefer `pip install -r
requirements-dev.txt` at the repo root before running evaluations.

## Contributing
Keep experiment-specific assets confined to their directory, add short docstrings
to each script, and store raw outputs (not checked-in) under `results/`.  Derived
tables/figures should be checked into `notes/` or `figures/` once finalized to
keep provenance clear.

