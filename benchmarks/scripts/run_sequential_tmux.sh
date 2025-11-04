#!/bin/bash
#
# Run Sequential Benchmarks in Tmux (With GPU Cleanup)
# ====================================================
#
# Runs baselines sequentially, cleaning GPU memory between each.
# Ensures clean state and prevents memory accumulation.
#
# For simpler CLI usage without tmux, use instead:
#   python3 run_benchmarks.py --baselines all
#
# Usage:
#   bash benchmarks/scripts/run_sequential_tmux.sh
#
# To monitor:
#   tmux attach -t genie_benchmarks_seq
#
# To detach (leave running):
#   Press Ctrl+B, then D
#

set -e

PROJECT_DIR="/home/jae/Genie"
SESSION_NAME="genie_benchmarks_seq"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="benchmark_results_${TIMESTAMP}"
LOG_DIR="$RESULTS_DIR/logs"

mkdir -p "$LOG_DIR"

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
sleep 1

# Create new session
tmux new-session -d -s "$SESSION_NAME" -x 200 -y 50

# Set up environment
tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_DIR && source .venv/bin/activate" Enter
sleep 1

echo "========================================"
echo "🚀 SEQUENTIAL BENCHMARKS - TMUX"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Mode: Sequential (one baseline at a time)"
echo "  GPU Cleanup: Yes (between each baseline)"
echo "  Baselines: 8 (all)"
echo "  Workloads: 5 (LLM Decode, Prefill, Vision, Multimodal, Micro)"
echo "  Runs: 20 measurement + 3 warmup per combination"
echo "  Total: ~800 experiments"
echo "  Duration: ~4-5 hours"
echo ""
echo "Output: $RESULTS_DIR/"
echo "Logs: $LOG_DIR/"
echo ""

# Define baselines
baselines=("local_pytorch" "naive_disaggregation" "genie_capture" "genie_local_remote" "genie_no_semantics" "genie_full" "pytorch_rpc" "ray")

# Run each baseline sequentially
for baseline in "${baselines[@]}"; do
    log_file="$LOG_DIR/${baseline}.log"
    
    tmux send-keys -t "$SESSION_NAME" "python3 run_benchmarks.py --baselines $baseline --output-dir '$RESULTS_DIR/${baseline}' 2>&1 | tee '$log_file'" Enter
    
    # Wait for completion and clean GPU
    tmux send-keys -t "$SESSION_NAME" "python3 -c \"import torch; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()\" 2>/dev/null || true; sleep 2" Enter
done

tmux send-keys -t "$SESSION_NAME" "echo ''; echo '✅ Sequential benchmarks complete!'; echo 'Results: $RESULTS_DIR/'" Enter

echo "✅ Sequential benchmarks started in tmux session: $SESSION_NAME"
echo ""
echo "Commands:"
echo "  Attach:    tmux attach -t $SESSION_NAME"
echo "  Detach:    Ctrl+B, then D"
echo ""
echo "Results: $RESULTS_DIR/"
echo "Logs: $LOG_DIR/"
echo ""
