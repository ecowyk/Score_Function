#!/usr/bin/env bash
# One resumable pipeline. No offline or nuPlan evaluation is launched.
set -euo pipefail
ROOT="${1:?Workspace root required}"
CONFIG="${2:?Configuration required}"
MODE="${3:-fresh}"
PYTHON="${4:-python}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NPROC="${#DEVICES[@]}"
OUT="$("$PYTHON" -c 'import json,sys,pathlib; c=json.load(open(sys.argv[1])); p=pathlib.Path(c["paths"]["run_dir"]); print(p if p.is_absolute() else pathlib.Path(sys.argv[2])/p)' "$CONFIG" "$ROOT")"
mkdir -p "$OUT"
# flock prevents duplicate runs even from different shells/tmux sessions.
exec 9>"$OUT/pipeline.lock"
flock -n 9 || { echo "A pipeline already owns $OUT"; exit 1; }
exec > >(tee -a "$OUT/pipeline.log") 2>&1
TRAIN_EXTRA=()
if [[ "$MODE" == resume ]]; then
  [[ -f "$OUT/score/last.pt" ]] || { echo 'No last.pt to resume'; exit 1; }
  TRAIN_EXTRA=(--resume)
else
  [[ ! -d "$OUT/score" ]] || { echo 'Training output exists; use resume or a new run_dir'; exit 1; }
fi
"$PYTHON" -m score_function preflight --config "$CONFIG" --root "$ROOT" --gpus "$NPROC"
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  -m score_function check-ddp --config "$CONFIG" --root "$ROOT"
if [[ "$MODE" != resume ]]; then
  "$PYTHON" -m score_function prepare --config "$CONFIG" --root "$ROOT"
  "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    -m score_function cache --config "$CONFIG" --root "$ROOT"
  "$PYTHON" -m score_function smoke --config "$CONFIG" --root "$ROOT"
fi
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  -m score_function train --config "$CONFIG" --root "$ROOT" "${TRAIN_EXTRA[@]}"
