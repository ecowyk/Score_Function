#!/usr/bin/env bash
# Run in the original baseline node; the other node is reserved for Energy.
set -euo pipefail
CODE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT=${UD_ROOT:-/home/data/wyk}
QUEUE_ROOT=${QUEUE_ROOT:-$RUN_ROOT/dataset/nuplan/exp/random_baselines_1025_v1}
PYTHON=${PYTHON:-$RUN_ROOT/envs/urbandriver/bin/python}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
exec "$PYTHON" -u "$CODE_ROOT/scripts/evaluate_random_baselines_1025.py" run \
  --output "$QUEUE_ROOT" --root "$RUN_ROOT" "$@"
