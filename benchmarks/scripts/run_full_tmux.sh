#!/bin/bash
#
# Run Full Benchmark Suite in Tmux (All 8 Baselines)
# ====================================================
#
# This script runs all 8 baselines across all 5 workloads in a tmux session.
# Use this for complete OSDI evaluation when you want a detachable long-running job.
#
# For simpler CLI usage without tmux, use instead:
#   python3 run_benchmarks.py --baselines all
#
# Usage:
#   bash benchmarks/scripts/run_full_tmux.sh
#
# To monitor:
#   tmux attach -t genie_benchmarks
#
# To detach (leave running):
#   Press Ctrl+B, then D
#

set -e

PROJECT_DIR="/home/jae/Genie"
SESSION_NAME="genie_benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_DIR}/benchmark_logs_${TIMESTAMP}"

mkdir -p "$LOG_DIR"

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
sleep 1

# Create new session
tmux new-session -d -s "$SESSION_NAME" -x 200 -y 50

# Set up environment
tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_DIR && source .venv/bin/activate" Enter
sleep 1

echo "=================================="
echo "🚀 FULL BENCHMARK SUITE - TMUX"
echo "=================================="
echo ""
echo "Configuration:"
echo "  Baselines: 8 (all)"
echo "  Workloads: 5 (LLM Decode, Prefill, Vision, Multimodal, Micro)"
echo "  Runs: 20 measurement + 3 warmup per combination"
echo "  Total: ~800 experiments"
echo "  Duration: ~4-5 hours"
echo ""
echo "Output: osdi_final_results/"
echo "Logs: $LOG_DIR/"
echo ""

# Run the benchmarks
tmux send-keys -t "$SESSION_NAME" "python3 run_benchmarks.py --output-dir osdi_final_results 2>&1 | tee '$LOG_DIR/full_suite_$TIMESTAMP.log'" Enter

echo "✅ Benchmark started in tmux session: $SESSION_NAME"
echo ""
echo "Commands:"
echo "  Attach:    tmux attach -t $SESSION_NAME"
echo "  Detach:    Ctrl+B, then D"
echo "  List:      tmux list-windows -t $SESSION_NAME"
echo ""
echo "Logs: $LOG_DIR/"
echo ""
