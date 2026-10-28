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
export FLOW_TORCH_CHANNEL=${FLOW_TORCH_CHANNEL:-cu118}
case "$FLOW_TORCH_CHANNEL" in
  cu118) export FLOW_EXPECTED_CUDA=11.8 ;;
  cu121-pypi) export FLOW_EXPECTED_CUDA=12.1 ;;
  *) echo 'FLOW_TORCH_CHANNEL must be cu118 or cu121-pypi.' >&2; exit 1 ;;
esac

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

echo "[3/5] Install official Torch 2.3.0 / CUDA $FLOW_EXPECTED_CUDA ($FLOW_TORCH_CHANNEL)."
if [[ "$FLOW_TORCH_CHANNEL" == cu121-pypi ]]; then
  # PyPI's Linux wheels use CUDA 12.1. Check the node before replacing packages;
  # no system CUDA toolkit or driver is installed/modified by this script.
  "$PYTHON" - <<'PY'
import platform
import re
import subprocess
import sys

if sys.version_info[:2] != (3, 9) or platform.system() != "Linux" or platform.machine() != "x86_64":
    raise SystemExit("Pinned PyPI wheels require Python 3.9 / Linux x86_64")
try:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True, timeout=15)
except (OSError, subprocess.SubprocessError) as error:
    raise SystemExit(f"Cannot verify NVIDIA driver for CUDA 12.1: {error}")
drivers = sorted(set(output.split()))
if not drivers or any(not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", v) for v in drivers):
    raise SystemExit(f"Unrecognized NVIDIA driver output: {output!r}")
for driver in drivers:
    parts = tuple(map(int, driver.split(".")))
    if (parts + (0,) * (3 - len(parts))) < (525, 60, 13):
        raise SystemExit(f"Driver {driver} is too old for this CUDA 12.1 fallback; "
                         "need >=525.60.13. Environment packages have not been changed. "
                         "Keep CUDA 11.8 and obtain its wheels offline instead.")
print("CUDA 12.1 minimum driver check passed:", ", ".join(drivers))
PY
fi
FLOW_SETUP_TMP=$(mktemp -d)
trap 'rm -rf -- "$FLOW_SETUP_TMP"' EXIT
"$PYTHON" -m pip install 'pip==24.0' 'setuptools==69.5.1' 'wheel==0.43.0'
# These extensions belong to the cloned older Torch installation. Flow does not
# use either one, and keeping their incompatible binaries would be misleading.
"$PYTHON" -m pip uninstall --yes natten torchaudio
# Reuse only a pair that actually imports with the selected CUDA runtime.
if "$PYTHON" - <<'PY'
import os
try:
    import torch
    import torchvision
    suffix = "+cu" + os.environ["FLOW_EXPECTED_CUDA"].replace(".", "")
    ready = (torch.__version__ == "2.3.0" + suffix
             and torchvision.__version__ == "0.18.0" + suffix
             and torch.version.cuda == os.environ["FLOW_EXPECTED_CUDA"])
except Exception:
    ready = False
raise SystemExit(0 if ready else 1)
PY
then
  echo "Matching Torch/torchvision CUDA $FLOW_EXPECTED_CUDA packages found; reusing them."
elif [[ "$FLOW_TORCH_CHANNEL" == cu118 ]]; then
  # Exact official R2 release wheels; dependencies use the configured pip index.
  "$PYTHON" -m pip install --upgrade --timeout 60 --retries 2 \
    'https://download-r2.pytorch.org/whl/cu118/torch-2.3.0%2Bcu118-cp39-cp39-linux_x86_64.whl#sha256=2c74bcc48e187153e56c9e3cb87c02831ee767630fd0a6c3d699063f12a2f471' \
    'https://download-r2.pytorch.org/whl/cu118/torchvision-0.18.0%2Bcu118-cp39-cp39-linux_x86_64.whl#sha256=dcc015a913a025d7b1ff43d444d2cba016e1c3a31f580fdeca20791025040fc6'
else
  # Use the configured PyPI mirror only. Exact public versions exclude +cu118
  # builds; validate against upstream PyPI hashes before installing either wheel.
  "$PYTHON" -m pip download --dest "$FLOW_SETUP_TMP" --no-deps --only-binary=:all: \
    --timeout 60 --retries 2 'torch===2.3.0' 'torchvision===0.18.0'
  "$PYTHON" - "$FLOW_SETUP_TMP" <<'PY'
import hashlib
from pathlib import Path
import sys

expected = {
    "torch-2.3.0-cp39-cp39-manylinux1_x86_64.whl":
        "cd0dc498b961ab19cb3f8dbf0c6c50e244f2f37dbfa05754ab44ea057c944ef9",
    "torchvision-0.18.0-cp39-cp39-manylinux1_x86_64.whl":
        "4c334b3e719ba0a9ba6e15d4aff1178f5e6d029174f346163fed525f0ccfffd3",
}
for name, checksum in expected.items():
    path = Path(sys.argv[1]) / name
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != checksum:
        raise SystemExit(f"Official PyPI SHA256 mismatch: {name}")
    print("Official PyPI wheel verified:", name)
PY
  # CUDA 11/12 pip packages can share paths (notably libcudnn.so.8). Remove
  # overlapping old packages before installing the new dependencies, never after.
  mapfile -t old_cuda_packages < <("$PYTHON" - <<'PY'
from importlib.metadata import distributions
names = sorted({d.metadata["Name"] for d in distributions()
                if d.metadata["Name"] and d.metadata["Name"].startswith("nvidia-")})
if any(name.endswith("-cu11") for name in names):
    print("\n".join(name for name in names if name.endswith(("-cu11", "-cu12"))))
PY
  )
  if ((${#old_cuda_packages[@]})); then
    "$PYTHON" -m pip uninstall --yes "${old_cuda_packages[@]}"
  fi
  "$PYTHON" -m pip install --upgrade --timeout 60 --retries 2 \
    "$FLOW_SETUP_TMP/torch-2.3.0-cp39-cp39-manylinux1_x86_64.whl" \
    "$FLOW_SETUP_TMP/torchvision-0.18.0-cp39-cp39-manylinux1_x86_64.whl"
fi

echo '[4/5] Install pinned inference and frozen-score dependencies.'
CONSTRAINTS="$FLOW_SETUP_TMP/constraints.txt"
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

assert torch.__version__.split("+")[0] == "2.3.0", torch.__version__
assert version("torchvision").split("+")[0] == "0.18.0", version("torchvision")
assert torch.version.cuda == os.environ["FLOW_EXPECTED_CUDA"], torch.version.cuda
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
print("Actual CUDA runtime:", torch.version.cuda, "cuDNN:", torch.backends.cudnn.version())
print("Visible GPUs:", torch.cuda.device_count())
print("Published weights loaded strictly; synthetic CUDA output:", tuple(output.shape))
print("Native model source:", Path(__import__(type(model).__module__, fromlist=["x"]).__file__).resolve())
print("Python:", Path(os.environ["FLOW_ENV"]) / "bin/python")
PY
