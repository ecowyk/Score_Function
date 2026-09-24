#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?Usage: bash scripts/train_tmux.sh ROOT [GPU_LIST] [CONFIG] [fresh|resume]}"
GPUS="${2:-0,1,2,3,4,5,6,7}"
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${3:-$CODE/configs/score_function.json}"
MODE="${4:-fresh}"
[[ "$MODE" == fresh || "$MODE" == resume ]] || { echo 'Use fresh or resume'; exit 1; }
SESSION="score_function_train_wyk"
tmux has-session -t "$SESSION" 2>/dev/null && { echo "Session exists: tmux attach -t $SESSION"; exit 1; }
PYTHON="$(command -v python)"
printf -v CMD 'cd %q && export PYTHONPATH=%q CUDA_VISIBLE_DEVICES=%q; bash scripts/run_company.sh %q %q %q %q; result=$?; echo "Pipeline exit code: $result"; exec bash' "$CODE" "$CODE${PYTHONPATH:+:$PYTHONPATH}" "$GPUS" "$ROOT" "$CONFIG" "$MODE" "$PYTHON"
tmux new-session -d -s "$SESSION" -c "$CODE" "$CMD"
echo "Started: tmux attach -t $SESSION"
