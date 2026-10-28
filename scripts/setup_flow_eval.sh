#!/usr/bin/env bash
# Isolated official Flow Planner inference environment and checkpoint checks.
set -euo pipefail

CODE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
UD_ROOT=${UD_ROOT:-/home/data/wyk}
FLOW_CODE=${FLOW_CODE:-$UD_ROOT/Flow-Planner}
FLOW_ENV=${FLOW_ENV:-$UD_ROOT/envs/flow_planner}
FLOW_BASE_ENV=${FLOW_BASE_ENV:-$UD_ROOT/envs/pluto}
FLOW_ASSET_DIR=${FLOW_ASSET_DIR:-$FLOW_CODE/checkpoints}
FLOW_CHECKPOINT=$FLOW_ASSET_DIR/model.pth
FLOW_MODEL_CONFIG=$FLOW_ASSET_DIR/model_config.yaml
FLOW_COMMIT=366df7654d7b7fa77f9d2577279da63127c6d2f8
NUPLAN_DEVKIT_ROOT=${NUPLAN_DEVKIT_ROOT:-$UD_ROOT/nuplan-devkit}

command -v conda >/dev/null
test -x "$FLOW_BASE_ENV/bin/python"
test -f "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py"
for protected_env in "$FLOW_BASE_ENV" "$UD_ROOT/envs/pluto" \
                     "$UD_ROOT/envs/urbandriver" "$UD_ROOT/envs/plantf"; do
  if [[ "$(realpath -m "$FLOW_ENV")" == "$(realpath -m "$protected_env")" ]]; then
    echo 'FLOW_ENV must be separate from the existing baseline environments.' >&2
    exit 1
  fi
done
"$FLOW_BASE_ENV/bin/python" - <<'PY'
import sys
if sys.version_info[:2] != (3, 9):
    raise SystemExit("FLOW_BASE_ENV must use Python 3.9")
PY

echo '[1/5] Verify pinned official Flow Planner source and published weights.'
if [[ ! -e "$FLOW_CODE" ]]; then
  git clone https://github.com/DiffusionAD/Flow-Planner.git "$FLOW_CODE"
  git -C "$FLOW_CODE" checkout --detach "$FLOW_COMMIT"
fi
test "$(git -C "$FLOW_CODE" rev-parse HEAD)" = "$FLOW_COMMIT"
git -C "$FLOW_CODE" diff --quiet HEAD --
download_args=(--output-dir "$FLOW_ASSET_DIR")
# Use the same supplied company CA for the downloader and pip in this process;
# no trusted-host exceptions or global certificate settings are added.
flow_ca_bundle=${FLOW_CA_BUNDLE:-${PIP_CERT:-}}
if [[ -n "$flow_ca_bundle" ]]; then
  download_args+=(--ca-bundle "$flow_ca_bundle")
  export PIP_CERT="$flow_ca_bundle"
fi
"$FLOW_BASE_ENV/bin/python" "$CODE_ROOT/scripts/download_flow_checkpoint.py" "${download_args[@]}"

echo '[2/5] Prepare isolated Flow Planner environment.'
if [[ ! -e "$FLOW_ENV" ]]; then
  echo "Cloning $FLOW_BASE_ENV into $FLOW_ENV."
  echo 'Conda may copy many thousands of files without progress output on shared storage.'
  conda create --yes --prefix "$FLOW_ENV" --clone "$FLOW_BASE_ENV"
  printf '%s\n' "$FLOW_BASE_ENV" > "$FLOW_ENV/.flow_score_setup"
elif [[ ! -f "$FLOW_ENV/.flow_score_setup" ]]; then
  echo "Existing environment has no completed Flow setup clone marker: $FLOW_ENV" >&2
  echo 'Preserve it and choose a new FLOW_ENV; an interrupted clone cannot be resumed here.' >&2
  exit 1
else
  echo 'Completed environment clone found; resuming dependency installation.'
fi
PYTHON="$FLOW_ENV/bin/python"

echo '[3/5] Install official Torch 2.3.0 / CUDA 11.8 versions.'
"$PYTHON" -m pip install 'pip==24.0' 'setuptools==69.5.1' 'wheel==0.43.0'
# These extensions belong to the cloned older Torch installation. Flow does not
# use either one, and keeping their incompatible binaries would be misleading.
"$PYTHON" -m pip uninstall --yes natten torchaudio
"$PYTHON" -m pip install --upgrade 'torch==2.3.0' 'torchvision==0.18.0' \
  --index-url https://download.pytorch.org/whl/cu118

echo '[4/5] Install pinned inference and frozen-score dependencies.'
CONSTRAINTS=$(mktemp)
trap 'rm -f "$CONSTRAINTS"' EXIT
cat > "$CONSTRAINTS" <<'EOF'
torch==2.3.0
torchvision==0.18.0
numpy==1.23.5
shapely==2.0.6
scipy==1.13.1
pandas==1.5.3
pyarrow==15.0.2
numba==0.58.1
timm==1.0.10
pytorch-lightning==2.0.1
torchmetrics==0.10.2
opencv-python==4.8.1.78
opencv-python-headless==4.8.1.78
hydra-core==1.3.2
omegaconf==2.3.0
einops==0.8.0
flow-matching==1.0.10
torchdiffeq==0.2.5
casadi==3.6.7
EOF
"$PYTHON" -m pip install -c "$CONSTRAINTS" numpy shapely scipy pandas pyarrow numba \
  timm pytorch-lightning torchmetrics opencv-python opencv-python-headless \
  hydra-core omegaconf einops flow-matching torchdiffeq casadi 'rich==13.3.4' \
  -r "$CODE_ROOT/requirements_urban_driver_score.txt"

echo '[5/5] Strictly load published weights and run a synthetic CUDA inference.'
export PYTHONPATH="$FLOW_CODE:$NUPLAN_DEVKIT_ROOT:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export FLOW_CODE FLOW_ENV FLOW_MODEL_CONFIG FLOW_CHECKPOINT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$FLOW_CODE"
"$PYTHON" - <<'PY'
from importlib.metadata import version
from pathlib import Path
import os
import torch
from flow_planner.data.dataset.nuplan import NuPlanDataSample
from flow_score.planner import load_flow_model

assert torch.__version__.startswith("2.3.0"), torch.__version__
assert torch.version.cuda == "11.8", torch.version.cuda
assert torch.cuda.is_available(), "CUDA is unavailable in the Flow Planner environment"
assert version("flow-matching") == "1.0.10", version("flow-matching")
assert version("hydra-core") == "1.3.2", version("hydra-core")
assert version("omegaconf") == "2.3.0", version("omegaconf")
model = load_flow_model(os.environ["FLOW_MODEL_CONFIG"], os.environ["FLOW_CHECKPOINT"], device="cuda")

# Exercise the actual normalization, encoder, decoder and four-step midpoint
# sampler on a small synthetic scene. This is an environment check, not a
# benchmark result; the queue separately checks actual nuPlan scenes.
def zeros(*shape, dtype=torch.float32):
    return torch.zeros(shape, dtype=dtype, device="cuda")

ego_current = zeros(1, 16)
ego_current[..., 2] = 1.0
ego_current[..., 4] = 5.0
ego_past = zeros(1, 21, 14)
ego_past[..., 2] = 1.0
lanes = zeros(1, 70, 20, 12)
lanes[0, 0, :, 0] = torch.arange(20, device="cuda") * 2.0
lanes[0, 0, :, 2] = 1.0
routes = zeros(1, 25, 20, 12)
routes[:, 0] = lanes[:, 0]
sample = NuPlanDataSample(
    batched=True, ego_past=ego_past, ego_current=ego_current,
    neighbor_past=zeros(1, 32, 21, 11),
    lanes=lanes, lanes_speedlimit=zeros(1, 70, 1),
    lanes_has_speedlimit=zeros(1, 70, 1, dtype=torch.bool),
    routes=routes, routes_speedlimit=zeros(1, 25, 1),
    routes_has_speedlimit=zeros(1, 25, 1, dtype=torch.bool),
    map_objects=zeros(1, 5, 10),
)
torch.manual_seed(3402)
with torch.no_grad():
    output = model(sample, mode="inference", use_cfg=True, cfg_weight=1.0)
torch.cuda.synchronize()
assert tuple(output.shape) == (1, 1, 80, 4), output.shape
assert torch.isfinite(output).all(), "Flow Planner returned nonfinite synthetic predictions"
print("Flow Planner environment ready:", torch.__version__, "flow-matching", version("flow-matching"))
print("Visible GPUs:", torch.cuda.device_count())
print("Published weights loaded strictly; synthetic CUDA output:", tuple(output.shape))
print("Native model source:", Path(__import__(type(model).__module__, fromlist=["x"]).__file__).resolve())
print("Python:", Path(os.environ["FLOW_ENV"]) / "bin/python")
PY
