#!/usr/bin/env bash
# Isolated PLUTO evaluation environment, preserving the working baseline envs.
set -euo pipefail

CODE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
UD_ROOT=${UD_ROOT:-/home/data/wyk}
PLUTO_CODE=${PLUTO_CODE:-$UD_ROOT/pluto}
PLUTO_ENV=${PLUTO_ENV:-$UD_ROOT/envs/pluto}
PLUTO_BASE_ENV=${PLUTO_BASE_ENV:-$UD_ROOT/envs/urbandriver}
PLUTO_CHECKPOINT=${PLUTO_CHECKPOINT:-$PLUTO_CODE/checkpoints/pluto_1M_aux_cil.ckpt}
PLUTO_COMMIT=b9964b649c660f1f4a971d614c66f5992e24c18a
NUPLAN_DEVKIT_ROOT=${NUPLAN_DEVKIT_ROOT:-$UD_ROOT/nuplan-devkit}

command -v conda >/dev/null
test -x "$PLUTO_BASE_ENV/bin/python"
test -f "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py"
if [[ "$(realpath -m "$PLUTO_ENV")" == "$(realpath -m "$PLUTO_BASE_ENV")" ||
      "$(realpath -m "$PLUTO_ENV")" == "$(realpath -m "$UD_ROOT/envs/plantf")" ]]; then
  echo 'PLUTO_ENV must be separate from the existing UrbanDriver and PlanTF environments.' >&2
  exit 1
fi
"$PLUTO_BASE_ENV/bin/python" - <<'PY'
import sys
if sys.version_info[:2] != (3, 9):
    raise SystemExit("PLUTO_BASE_ENV must use Python 3.9 for the pinned NATTEN wheel")
PY

if [[ ! -e "$PLUTO_CODE" ]]; then
  git clone https://github.com/jchengai/pluto.git "$PLUTO_CODE"
  git -C "$PLUTO_CODE" checkout --detach "$PLUTO_COMMIT"
fi
test "$(git -C "$PLUTO_CODE" rev-parse HEAD)" = "$PLUTO_COMMIT"
git -C "$PLUTO_CODE" diff --quiet HEAD --
mkdir -p "$(dirname -- "$PLUTO_CHECKPOINT")"
"$PLUTO_BASE_ENV/bin/python" "$CODE_ROOT/scripts/download_pluto_checkpoint.py" --output "$PLUTO_CHECKPOINT"

# Reuse the already working nuPlan/Ray/geospatial stack, but clone it before
# installing the official PLUTO Torch/NATTEN versions. No source env is edited.
if [[ ! -e "$PLUTO_ENV" ]]; then
  conda create --yes --prefix "$PLUTO_ENV" --clone "$PLUTO_BASE_ENV"
  printf '%s\n' "$PLUTO_BASE_ENV" > "$PLUTO_ENV/.pluto_score_setup"
elif [[ ! -f "$PLUTO_ENV/.pluto_score_setup" ]]; then
  echo "Existing environment was not created by this setup: $PLUTO_ENV" >&2
  echo 'Set PLUTO_ENV to a new directory to preserve that environment.' >&2
  exit 1
fi
PYTHON="$PLUTO_ENV/bin/python"
"$PYTHON" -m pip install 'pip==24.0' 'setuptools==69.5.1' 'wheel==0.43.0'
# torchaudio is not used here; a cloned Torch-1.12 build would conflict with 2.0.
"$PYTHON" -m pip uninstall --yes torchaudio
"$PYTHON" -m pip install --upgrade 'torch==2.0.1' 'torchvision==0.15.2' \
  --index-url https://download.pytorch.org/whl/cu118
# Use the official cp39 wheel directly. Some proxies return the archive index
# without a usable Content-Type, causing pip to silently skip all its wheels.
# Avoid downloading the 74 MB wheel again when resuming later setup stages.
if "$PYTHON" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
try:
    installed = version("natten")
except PackageNotFoundError:
    installed = None
raise SystemExit(0 if installed == "0.14.6+torch200cu118" else 1)
PY
then
  echo 'NATTEN 0.14.6+torch200cu118 already installed; CUDA validation follows.'
else
  # SHA256 is published by GitHub for official release asset 326070037.
  "$PYTHON" -m pip install --no-index --no-deps --only-binary=:all: \
    'https://github.com/SHI-Labs/NATTEN/releases/download/v0.14.6/natten-0.14.6%2Btorch200cu118-cp39-cp39-linux_x86_64.whl#sha256=f12b90c03d9c8635b0491922de3b6452e4e33cc0cb041521f3ea04a0acca0ac4'
fi

# Explicit compatibility pins for Python 3.9 / nuPlan 1.2.2 and the frozen DP
# encoder. In particular avoid NumPy 2 and recent pyarrow's NumPy-2 typing API.
CONSTRAINTS=$(mktemp)
trap 'rm -f "$CONSTRAINTS"' EXIT
cat > "$CONSTRAINTS" <<'EOF'
torch==2.0.1
torchvision==0.15.2
numpy==1.23.5
shapely==2.0.6
scipy==1.10.1
pandas==1.5.3
pyarrow==15.0.2
numba==0.58.1
timm==1.0.10
pytorch-lightning==2.0.1
torchmetrics==0.10.2
opencv-python==4.8.1.78
opencv-python-headless==4.8.1.78
EOF
"$PYTHON" -m pip install -c "$CONSTRAINTS" numpy shapely scipy pandas pyarrow numba \
  timm pytorch-lightning torchmetrics opencv-python opencv-python-headless 'rich==13.3.4' \
  -r "$CODE_ROOT/requirements_urban_driver_score.txt"

export PYTHONPATH="$PLUTO_CODE:$NUPLAN_DEVKIT_ROOT:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PLUTO_CODE PLUTO_ENV
cd "$PLUTO_CODE"
"$PYTHON" - <<'PY'
from importlib.metadata import version
from pathlib import Path
import os
import torch
import natten
from natten import NeighborhoodAttention1D
from src.models.pluto.pluto_model import PlanningModel
from src.feature_builders.pluto_feature_builder import PlutoFeatureBuilder
import nuplan
assert torch.__version__.startswith("2.0.1"), torch.__version__
assert torch.version.cuda == "11.8", torch.version.cuda
assert version("natten").startswith("0.14.6+torch200cu118"), version("natten")
assert torch.cuda.is_available(), "CUDA is unavailable in the PLUTO environment"
layer = NeighborhoodAttention1D(dim=32, kernel_size=3, num_heads=4).cuda().eval()
with torch.no_grad():
    output = layer(torch.zeros(1, 21, 32, device="cuda"))
assert torch.isfinite(output).all(), "NATTEN CUDA kernel returned nonfinite values"
print("PLUTO environment ready:", torch.__version__, "natten", version("natten"))
print("Visible GPUs:", torch.cuda.device_count())
print("Native model source:", Path(__import__(PlanningModel.__module__, fromlist=["x"]).__file__).resolve())
print("Python:", Path(os.environ["PLUTO_ENV"]) / "bin/python")
PY
