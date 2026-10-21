#!/usr/bin/env bash
# Paired original baseline -> S10-G K20, eight GPUs with one scenario per GPU.
set -euo pipefail
CODE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export UD_ROOT=${UD_ROOT:-/home/data/wyk}
export NUPLAN_DEVKIT_ROOT="$UD_ROOT/nuplan-devkit"
export NUPLAN_DATA_ROOT="$UD_ROOT/dataset/nuplan"
export NUPLAN_MAPS_ROOT="$NUPLAN_DATA_ROOT/maps"
export NUPLAN_EXP_ROOT="$NUPLAN_DATA_ROOT/exp"
export FROZEN_SCORE_CODE=${FROZEN_SCORE_CODE:-$NUPLAN_EXP_ROOT/score_G_transfer_1014_v1/code/lg}
export SCORE_CONFIG=${SCORE_CONFIG:-$NUPLAN_EXP_ROOT/score_G_transfer_1014_v1/models/S10-G/config.json}
export SCORE_CHECKPOINT=${SCORE_CHECKPOINT:-$NUPLAN_EXP_ROOT/score_G_transfer_1014_v1/models/S10-G/final_ema.pt}
PYTHON=${PYTHON:-$UD_ROOT/envs/urbandriver/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HYDRA_FULL_ERROR=1 NODE_RANK=0 NUM_NODES=1
export PYTHONPATH="$FROZEN_SCORE_CODE:$CODE_ROOT:$UD_ROOT/Diffusion-Planner:$NUPLAN_DEVKIT_ROOT:$UD_ROOT/tuplan_garage"
unset NUPLAN_HYDRA_CONFIG_PATH RAY_ADDRESS
cd "$UD_ROOT/tuplan_garage"
test -s "$SCORE_CHECKPOINT"
test -s "$FROZEN_SCORE_CODE/score_function/model/refinement.py"
CUDA_VISIBLE_DEVICES="" "$PYTHON" "$CODE_ROOT/tests/test_native_planner_score.py"
QUEUE_ROOT=${QUEUE_ROOT:-$NUPLAN_EXP_ROOT/native_score_queue_$(date +%Y%m%d_%H%M%S)}
args=(--output "$QUEUE_ROOT" --workers 8)
if [[ -n "${BASELINE_FULL_ROOT:-}" ]]; then
  args+=(--raw-suite "$BASELINE_FULL_ROOT")
fi
exec "$PYTHON" -u "$CODE_ROOT/scripts/evaluate_native_score_queue.py" "${args[@]}"
