#!/usr/bin/env bash
# Published Flow Planner, then frozen S10-G K20; full runs use eight GPUs.
set -euo pipefail
CODE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export UD_ROOT=${UD_ROOT:-/home/data/wyk}
export FLOW_CODE=${FLOW_CODE:-$UD_ROOT/Flow-Planner}
export FLOW_CHECKPOINT=${FLOW_CHECKPOINT:-$FLOW_CODE/checkpoints/model.pth}
export FLOW_CONFIG=${FLOW_CONFIG:-$FLOW_CODE/checkpoints/model_config.yaml}
export NUPLAN_DEVKIT_ROOT=${NUPLAN_DEVKIT_ROOT:-$UD_ROOT/nuplan-devkit}
export NUPLAN_DATA_ROOT=${NUPLAN_DATA_ROOT:-$UD_ROOT/dataset/nuplan}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-$NUPLAN_DATA_ROOT/maps}
export NUPLAN_EXP_ROOT=${NUPLAN_EXP_ROOT:-$NUPLAN_DATA_ROOT/exp}
export FROZEN_SCORE_CODE=${FROZEN_SCORE_CODE:-$NUPLAN_EXP_ROOT/score_G_transfer_1014_v1/code/lg}
export SCORE_CONFIG=${SCORE_CONFIG:-$NUPLAN_EXP_ROOT/score_G_transfer_1014_v1/models/S10-G/config.json}
export SCORE_CHECKPOINT=${SCORE_CHECKPOINT:-$NUPLAN_EXP_ROOT/score_G_transfer_1014_v1/models/S10-G/final_ema.pt}
PYTHON=${PYTHON:-$UD_ROOT/envs/flow_planner/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HYDRA_FULL_ERROR=1 NODE_RANK=0 NUM_NODES=1
export PYTHONPATH="$FROZEN_SCORE_CODE:$CODE_ROOT:$FLOW_CODE:$UD_ROOT/Diffusion-Planner:$NUPLAN_DEVKIT_ROOT:$UD_ROOT/tuplan_garage"
unset NUPLAN_HYDRA_CONFIG_PATH RAY_ADDRESS
cd "$FLOW_CODE"
test -s "$FLOW_CHECKPOINT"
test -s "$FLOW_CONFIG"
test -s "$SCORE_CHECKPOINT"
test -s "$FROZEN_SCORE_CODE/score_function/model/refinement.py"
CUDA_VISIBLE_DEVICES="" "$PYTHON" "$CODE_ROOT/tests/test_flow_score.py"
CUDA_VISIBLE_DEVICES="" "$PYTHON" "$CODE_ROOT/tests/test_native_score_metrics.py"
QUEUE_ROOT=${QUEUE_ROOT:-$NUPLAN_EXP_ROOT/flow_score_queue_$(date +%Y%m%d_%H%M%S)}
exec "$PYTHON" -u "$CODE_ROOT/scripts/evaluate_flow_score_queue.py" --output "$QUEUE_ROOT"
