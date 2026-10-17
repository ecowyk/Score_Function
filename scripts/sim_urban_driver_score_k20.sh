#!/usr/bin/env bash
# Official nuPlan run_simulation.py entry; native UrbanDriver's saved config.
# One one-scenario K20 smoke check, followed by one full K20 evaluation.
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
export HYDRA_FULL_ERROR=1 NODE_RANK=0
unset NUPLAN_HYDRA_CONFIG_PATH RAY_ADDRESS
# Use exactly the score implementation already frozen for the DP experiments.
export PYTHONPATH="$FROZEN_SCORE_CODE:$CODE_ROOT:$UD_ROOT/Diffusion-Planner:$NUPLAN_DEVKIT_ROOT:$UD_ROOT/tuplan_garage"
cd "$UD_ROOT/tuplan_garage"
test -s "$SCORE_CHECKPOINT"
test -s "$FROZEN_SCORE_CODE/score_function/model/refinement.py"
"$PYTHON" "$CODE_ROOT/tests/test_urban_driver_score.py"
RESULT_DIR="$NUPLAN_EXP_ROOT/exp/simulation/closed_loop_nonreactive_agents/urbandriver_score/test14-hard/S10G_K20_$(date +%Y%m%d_%H%M%S)"
"$PYTHON" "$CODE_ROOT/scripts/prepare_urban_driver_score.py" prepare "$RESULT_DIR"
printf '结果目录：%s\n' "$RESULT_DIR"
# First run uses GPU 0 only and one scenario. The full run reserves one GPU per
# Ray task, with the same workers/resources as the completed native 8-GPU run.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} "$PYTHON" -u \
  "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
  --config-path "$RESULT_DIR/launch/smoke" --config-name config \
  2>&1 | tee "$RESULT_DIR/smoke.console.log"
"$PYTHON" "$CODE_ROOT/scripts/prepare_urban_driver_score.py" verify-smoke "$RESULT_DIR"
"$PYTHON" -u "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
  --config-path "$RESULT_DIR/launch/full" --config-name config \
  2>&1 | tee "$RESULT_DIR/console.log"
"$PYTHON" "$CODE_ROOT/scripts/prepare_urban_driver_score.py" verify "$RESULT_DIR"
