#!/usr/bin/env bash
# Prepare an independent cached-VLM ChainFlow runtime; never modify the DD env.
set -euo pipefail

REPRO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CF_ROOT=/home/data/wyk
CF_INSECURE=${CF_INSECURE:-0}
CF_SKIP_ENV=0
CF_SKIP_ASSETS=0
CF_ONLY=all
CF_SCOPE=native

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_navsim_chainflow.sh [options]
  --root PATH       Existing project root (default /home/data/wyk)
  --insecure        Skip TLS verification only for these setup subprocesses
  --skip-env        Reuse an already prepared ROOT/envs/chainflow environment
  --skip-assets     Prepare code/environment only; no checkpoint or VLM downloads
  --only KIND       Asset selection: all, weights, navtest, trainval (default all)
  --scope KIND      native (default): weights + navtest VLM (~109 GB)
                    all: also trainval VLM (~1.035 TB cache in total)

The official full VLM cache is 1,035,243,638,403 bytes compressed (~1.035 TB).
This script reuses ROOT/dataset/navsim and ROOT/nuplan-devkit. It does not download
NAVSIM sensor data, does not merge val into train, and does not replace existing
environments or source checkouts. Inspect the asset plan before large downloads:
  python3 score_function/navsim_chainflow/assets.py plan

Optional environment variables:
  CF_CONDA_CHANNEL   Conda Python channel (default https://repo.anaconda.com/pkgs/main)
  CF_TORCH_INDEX    Torch wheel index (default https://download.pytorch.org/whl/cu121)
  HF_ENDPOINT       Hugging Face origin or compatible mirror
Proxy variables are inherited; proxy passwords are never printed or written.
EOF
}

while (($#)); do
  case "$1" in
    --root) CF_ROOT=$2; shift 2 ;;
    --insecure) CF_INSECURE=1; shift ;;
    --skip-env) CF_SKIP_ENV=1; shift ;;
    --skip-assets) CF_SKIP_ASSETS=1; shift ;;
    --only) CF_ONLY=$2; shift 2 ;;
    --scope) CF_SCOPE=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$CF_ONLY" in all|weights|navtest|trainval) ;; *) echo 'Invalid --only value' >&2; exit 2 ;; esac
case "$CF_SCOPE" in native|all) ;; *) echo 'Invalid --scope value' >&2; exit 2 ;; esac
if [[ "$CF_ONLY" == trainval && "$CF_SCOPE" != all ]]; then
  echo '--only trainval requires --scope all' >&2; exit 2
fi
case "$CF_INSECURE" in 0|1) ;; *) echo 'CF_INSECURE must be 0 or 1' >&2; exit 2 ;; esac

CF_ROOT=$(realpath -- "$CF_ROOT")
CF_CODE="$CF_ROOT/ChainFlow-VLA"
CF_ENV="$CF_ROOT/envs/chainflow"
CF_PYTHON="$CF_ENV/bin/python"
CF_DEVKIT="$CF_ROOT/nuplan-devkit"
CF_COMMIT=0f3565f66546139a0183bc5807d5f7234ae2d4b0
CF_GIT=(git)
CF_PIP_TLS=()
if [[ "$CF_INSECURE" == 1 ]]; then
  CF_GIT+=( -c http.sslVerify=false )
  export CONDA_SSL_VERIFY=false
  # Process-scoped only; no .condarc, pip.conf or git config is changed.
  CF_PIP_TLS+=( --trusted-host pypi.org --trusted-host files.pythonhosted.org
                --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org )
  for CF_INDEX in "${PIP_INDEX_URL:-}" "${PIP_EXTRA_INDEX_URL:-}" "${CF_TORCH_INDEX:-}"; do
    if [[ -n "$CF_INDEX" ]]; then
      CF_HOST=$(python3 -c 'import sys,urllib.parse; print(urllib.parse.urlparse(sys.argv[1]).hostname or "")' "$CF_INDEX")
      [[ -z "$CF_HOST" ]] || CF_PIP_TLS+=( --trusted-host "$CF_HOST" )
    fi
  done
fi

[[ -d "$CF_DEVKIT/nuplan" ]] || { echo "Missing existing nuplan-devkit: $CF_DEVKIT" >&2; exit 1; }
[[ -d "$CF_ROOT/dataset/navsim" ]] || { echo 'Existing NAVSIM data root is missing' >&2; exit 1; }
[[ -d "$CF_ROOT/dataset/nuplan/maps" ]] || { echo 'Existing shared nuPlan maps are missing' >&2; exit 1; }

if [[ -e "$CF_CODE" ]]; then
  [[ -d "$CF_CODE/.git" || -f "$CF_CODE/.git" ]] || { echo "Refusing non-git existing directory: $CF_CODE" >&2; exit 1; }
  [[ $(git -C "$CF_CODE" rev-parse HEAD) == "$CF_COMMIT" ]] || {
    echo "Existing ChainFlow checkout is not the pinned commit; left untouched: $CF_CODE" >&2; exit 1;
  }
  git -C "$CF_CODE" diff --quiet
  git -C "$CF_CODE" diff --cached --quiet
else
  # Sparse checkout excludes the repository's vendored duplicate nuplan-devkit.
  "${CF_GIT[@]}" clone --filter=blob:none --no-checkout https://github.com/AFARI-Research/ChainFlow-VLA.git "$CF_CODE"
  "${CF_GIT[@]}" -C "$CF_CODE" sparse-checkout set --no-cone '/*' '!/nuplan-devkit/'
  "${CF_GIT[@]}" -C "$CF_CODE" checkout --detach "$CF_COMMIT"
fi

if [[ "$CF_SKIP_ENV" == 0 ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo 'conda is unavailable in this shell; initialize your existing Conda installation first.' >&2
    exit 1
  fi
  if [[ ! -x "$CF_PYTHON" ]]; then
    [[ ! -e "$CF_ENV" ]] || { echo "Refusing incomplete existing environment: $CF_ENV" >&2; exit 1; }
    mkdir -p "$CF_ROOT/envs"
    conda create -y --prefix "$CF_ENV" --override-channels \
      -c "${CF_CONDA_CHANNEL:-https://repo.anaconda.com/pkgs/main}" python=3.10 pip
  fi
  "$CF_PYTHON" -m pip install "${CF_PIP_TLS[@]}" --upgrade 'pip==25.3' 'setuptools==75.8.2' 'wheel==0.45.1'
  "$CF_PYTHON" -m pip install "${CF_PIP_TLS[@]}" \
    --index-url "${CF_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}" \
    'torch==2.5.1' 'torchvision==0.20.1'
  "$CF_PYTHON" -m pip install "${CF_PIP_TLS[@]}" -r "$REPRO_ROOT/configs/navsim_chainflow_requirements.txt"
  # Upstream's declared requirements also include optional VLM finetuning stacks.
  # Install source without those extras; use the audited cached-feature requirements above.
  "$CF_PYTHON" -m pip install --no-deps --no-build-isolation -e "$CF_DEVKIT" -e "$CF_CODE"
fi
[[ -x "$CF_PYTHON" ]] || { echo "Missing Python: $CF_PYTHON" >&2; exit 1; }

PYTHONNOUSERSITE=1 PYTHONPATH="$REPRO_ROOT:$CF_CODE:$CF_DEVKIT" "$CF_PYTHON" - <<'PY'
import importlib.metadata
from pathlib import Path
import torch
import torchvision
import timm
import navsim
import nuplan
import zstandard
from navsim.agents.chainflow_vla.chainflow_vla_agent import ChainFlowVLAAgent
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
assert torch.__version__.split('+')[0] == '2.5.1', torch.__version__
assert torchvision.__version__.split('+')[0] == '0.20.1', torchvision.__version__
assert timm.__version__ == '1.0.15', timm.__version__
print('Native ChainFlow and PDM imports passed')
print('navsim:', navsim.__file__)
print('nuplan:', nuplan.__file__)
print('Torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('Visible GPUs:', torch.cuda.device_count())
PY

CF_SETUP_DIR="$CF_ROOT/dataset/navsim/exp/chainflow_setup"
mkdir -p "$CF_SETUP_DIR"
"$CF_PYTHON" -m pip freeze > "$CF_SETUP_DIR/pip-freeze.txt"
if [[ "$CF_SKIP_ASSETS" == 0 ]]; then
  CF_ASSET_ARGS=(fetch --root "$CF_ROOT" --only "$CF_ONLY" --scope "$CF_SCOPE")
  [[ "$CF_INSECURE" != 1 ]] || CF_ASSET_ARGS+=(--insecure)
  "$CF_PYTHON" -u "$REPRO_ROOT/score_function/navsim_chainflow/assets.py" "${CF_ASSET_ARGS[@]}"
else
  echo 'Code/environment ready. Asset downloads were explicitly skipped.'
fi
echo "ChainFlow Python: $CF_PYTHON"
echo "ChainFlow source: $CF_CODE ($CF_COMMIT)"
