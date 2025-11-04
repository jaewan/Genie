#!/bin/bash
#
# Run Sustained Execution Benchmarks in Tmux
# ==========================================
#
# Runs sustained execution benchmarks (many requests, cache statistics).
# Useful for measuring performance under repeated workloads.
#
# For CLI usage without tmux, use instead:
#   python3 -m benchmarks.sustained_execution --num-requests 1000
#
# Usage:
#   bash benchmarks/scripts/run_sustained_tmux.sh [num_requests]
#
# Examples:
#   bash benchmarks/scripts/run_sustained_tmux.sh        (default: 1000 requests)
#   bash benchmarks/scripts/run_sustained_tmux.sh 500    (500 requests)
#
# To monitor:
#   tmux attach -t genie_sustained
#
# To detach (leave running):
#   Press Ctrl+B, then D
#

set -e

PROJECT_DIR="/home/jae/Genie"
SESSION_NAME="genie_sustained"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NUM_REQUESTS="${1:-1000}"  # Default: 1000 requests
LOG_DIR="sustained_execution_logs_${TIMESTAMP}"

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
echo "🚀 SUSTAINED EXECUTION - TMUX"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Mode: Sustained execution"
echo "  Requests: $NUM_REQUESTS"
echo "  Measurement: Every 10 requests"
echo "  Models: GPT-2-XL (1.5B), ResNet-50 (25M)"
echo "  Baselines: 3 (Local PyTorch, Genie Full, PyTorch RPC)"
echo "  Duration: ~2-3 hours"
echo ""
echo "Output: sustained_execution_real_results/"
echo "Logs: $LOG_DIR/"
echo ""

# Run sustained execution benchmarks
tmux send-keys -t "$SESSION_NAME" "python3 -m benchmarks.sustained_execution_real --num-requests $NUM_REQUESTS --measure-every 10 --output-dir sustained_execution_real_results 2>&1 | tee '$LOG_DIR/sustained_$TIMESTAMP.log'" Enter

echo "✅ Sustained execution started in tmux session: $SESSION_NAME"
echo ""
echo "Commands:"
echo "  Attach:    tmux attach -t $SESSION_NAME"
echo "  Detach:    Ctrl+B, then D"
echo ""
echo "Logs: $LOG_DIR/"
echo ""
