#!/usr/bin/env bash
# Bootstrap and launch eight independent single-GPU experiments inside tmux.
set -euo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ORIGINAL_ARGS=("$@")
ROOT="" RUN_NAME=score_matrix_1m GPUS=0,1,2,3,4,5,6,7
PYTHON="" ENV_NAME=score_function_wyk FOREGROUND=0 DRY_RUN=0
while (( $# )); do
  case "$1" in
    --root) ROOT="${2:?Missing root}"; shift 2 ;;
    --run-name) RUN_NAME="${2:?Missing run name}"; shift 2 ;;
    --gpus) GPUS="${2:?Missing GPU list}"; shift 2 ;;
    --python) PYTHON="${2:?Missing Python executable}"; shift 2 ;;
    --env-name) ENV_NAME="${2:?Missing environment name}"; shift 2 ;;
    --foreground) FOREGROUND=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --resume) shift ;;
    --database-dir|--maps-dir|--planner-dir|--devkit-dir|--checkpoint|--planner-args|--data-output|--base-config|--suite-config|--reuse-features-from|--total-scenarios)
      [[ $# -ge 2 ]] || { echo "Missing value for $1"; exit 2; }; shift 2 ;;
    -h|--help)
      echo 'Usage: bash scripts/launch_experiments.sh --root /workspace [options]'
      echo '  --database-dir PATH --maps-dir PATH --data-output PATH'
      echo '  --planner-dir PATH --devkit-dir PATH --checkpoint FILE --planner-args FILE'
      echo '  --python /existing/environment/bin/python (reuse a compatible environment)'
      echo '  --env-name score_function_wyk (otherwise create this conda environment)'
      echo '  --gpus 0,1,2,3,4,5,6,7 --run-name score_matrix_1m'
      echo '  --total-scenarios 1000000 --reuse-features-from /old/data_output'
      echo '  --resume | --dry-run | --foreground'
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done
[[ -n "$ROOT" ]] || { echo '--root is required'; exit 2; }
[[ "$RUN_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid run-name'; exit 2; }
[[ "$ENV_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid env-name'; exit 2; }
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
if (( DRY_RUN )); then
  "${PYTHON:-python3}" -m score_function.tools.experiment_suite "${ORIGINAL_ARGS[@]}"
  exit
fi
[[ "$(uname -s)" == Linux ]] || { echo 'Production launcher requires Linux'; exit 2; }
[[ -z "$PYTHON" || -x "$PYTHON" ]] || {
  PYTHON="$(command -v "$PYTHON")" || { echo 'Python executable not found'; exit 2; }
}
CONDA=""
if command -v conda >/dev/null 2>&1; then
  CONDA="$(conda info --base)/bin/conda"
else
  for candidate in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" /opt/conda/bin/conda; do
    if [[ -x "$candidate" ]]; then CONDA="$candidate"; break; fi
  done
fi
if ! command -v tmux >/dev/null 2>&1; then
  [[ -n "$CONDA" ]] || { echo 'Install tmux or make conda available first'; exit 2; }
  "$CONDA" install -y -n base -c conda-forge tmux
  export PATH="$(dirname "$CONDA"):$PATH"
fi
if (( ! FOREGROUND )); then
  SESSION="${RUN_NAME}_wyk"
  if tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "tmux session exists: $SESSION. Inspect it before starting another run."
    exit 1
  fi
  printf -v RUN 'exec bash %q ' "$CODE/scripts/launch_experiments.sh"
  printf -v QUOTED '%q ' "${ORIGINAL_ARGS[@]}"
  RUN+="$QUOTED --foreground"
  tmux new-session -d -s "$SESSION" -n pipeline -c "$PWD"
  tmux set-option -t "$SESSION" remain-on-exit on
  tmux send-keys -t "$SESSION:0.0" -l "$RUN"
  tmux send-keys -t "$SESSION:0.0" Enter
  echo "Started: tmux attach -t $SESSION"
  echo "Bootstrap/status: $ROOT/outputs/$RUN_NAME/"
  exit
fi
mkdir -p "$ROOT/outputs/$RUN_NAME"
ROOT="$(cd "$ROOT" && pwd)"
exec 9>"$ROOT/outputs/$RUN_NAME/bootstrap.lock"
flock -n 9 || { echo 'This suite is already bootstrapping/running'; exit 1; }
exec > >(tee -a "$ROOT/outputs/$RUN_NAME/bootstrap.log") 2>&1
trap 'echo "Launcher failed at line $LINENO. See bootstrap.log and per-stage logs."' ERR
if [[ -z "$PYTHON" ]]; then
  [[ -n "$CONDA" ]] || { echo 'Provide --python or install Miniconda first'; exit 2; }
  PREFIX="$("$CONDA" info --base)/envs/$ENV_NAME"
  if [[ ! -x "$PREFIX/bin/python" ]]; then
    "$CONDA" create -y -p "$PREFIX" -c conda-forge python=3.9 pip
  fi
  PYTHON="$PREFIX/bin/python"
  # This named environment belongs to Score Function; existing environments are
  # reused without dependency replacement only when explicitly passed via --python.
  REQUIREMENTS_HASH="$(sha256sum "$CODE/requirements_company.txt" | cut -d ' ' -f1)"
  if [[ ! -f "$PREFIX/.score_function_requirements" ]] || \
     [[ "$(cat "$PREFIX/.score_function_requirements")" != "$REQUIREMENTS_HASH" ]]; then
    "$PYTHON" -m pip install 'pip==23.3.2' 'setuptools==59.5.0' wheel
    "$PYTHON" -m pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 \
      --extra-index-url https://download.pytorch.org/whl/cu118
    "$PYTHON" -m pip install -r "$CODE/requirements_company.txt"
    printf '%s\n' "$REQUIREMENTS_HASH" > "$PREFIX/.score_function_requirements"
  fi
fi
export PYTHONUNBUFFERED=1
"$PYTHON" -m score_function.tools.bootstrap_company "${ORIGINAL_ARGS[@]}"
"$PYTHON" -m pip install --no-deps --no-build-isolation -e "$CODE"
"$PYTHON" -m pip freeze > "$ROOT/outputs/$RUN_NAME/environment.txt"
"$PYTHON" -m score_function.tools.experiment_suite "${ORIGINAL_ARGS[@]}"
