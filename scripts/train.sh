#!/usr/bin/env bash
# Train from an existing cache. Data preparation and simulation are separate stages.
set -euo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="${2:?--python requires an interpreter}"; shift 2 ;;
    --gpus) GPUS="${2:?--gpus requires a comma-separated GPU list}"; shift 2 ;;
    -h|--help)
      cat <<'USAGE'
Usage: bash scripts/train.sh --config CONFIG [--root ROOT] [options]
  --python PATH       Python interpreter (default: $PYTHON or active python)
  --gpus 0[,1,...]     Visible GPUs; one process per GPU (default: existing mask or 0)
  --resume            Restore this run's score/last.pt with the same code/config
  --progress MODE     auto (terminal bars), on (force bars), off (plain logs)
  --set KEY=JSON      Configuration override, repeatable

All other options are forwarded to train_score_branch.py. This command only
trains; prepare the feature cache first. Relative config paths use your current
directory. Plain logs: paths.run_dir/score/console.log. Use a new run_dir for a
fresh experiment; do not update the checkout of an active or resumable run.
USAGE
      exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[[ ${#ARGS[@]} -gt 0 ]] || { echo 'Missing --config; use --help.' >&2; exit 2; }
[[ "$GPUS" != ,* && "$GPUS" != *, && "$GPUS" != *,,* && "$GPUS" != *[[:space:]]* ]] \
  || { echo 'Invalid --gpus list.' >&2; exit 2; }
PYTHON="$(command -v -- "$PYTHON")" || { echo 'Python interpreter not found.' >&2; exit 1; }
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPUS" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
IFS=',' read -ra DEVICES <<< "$GPUS"
printf 'Training only | Python: %s | GPUs: %s | code: %s\n' "$PYTHON" "$GPUS" "$CODE"
if [[ ${#DEVICES[@]} -eq 1 ]]; then
  exec "$PYTHON" -u "$CODE/train_score_branch.py" "${ARGS[@]}"
else
  exec "$PYTHON" -u -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node="${#DEVICES[@]}" "$CODE/train_score_branch.py" "${ARGS[@]}"
fi
