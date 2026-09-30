#!/usr/bin/env bash
# Place this file at the Score_Function repository root, next to train_score_branch.py.
# Based on Diffusion-Planner/sim_diffusion_planner_runner.sh. All simulation,
# metric computation, aggregation and NuBoard serialization use nuPlan itself.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash sim_score_planner.sh --config CONFIG --checkpoint BEST_PT [options]

Required (or export SCORE_CONFIG and SCORE_CHECKPOINT):
  --config PATH       The configuration used for this trained score/energy model
  --checkpoint PATH   Selected score/best.pt, not the resumable score/last.pt

Options (corresponding environment variable in parentheses):
  --name NAME         Experiment label, e.g. E05-L or S05-L (RUN_NAME; score_function)
  --steps N           Score refinement iterations, not diffusion steps (STEPS; 5)
  --gamma NUMBER      Refinement step factor; 0 disables refinement (GAMMA; 0.1)
  --gpus LIST         Comma-separated available GPU IDs (EVAL_GPUS; existing mask or 0)
  --python PATH       Python in the prepared diffusion_planner environment (PY; python)
  --root PATH         Workspace containing data and Diffusion-Planner (ROOT; repo parent)
  --split NAME        test14-hard, test14-random or val14 (SPLIT; test14-hard)
  --challenge NAME    closed_loop_nonreactive_agents or closed_loop_reactive_agents
                      (CHALLENGE; closed_loop_nonreactive_agents)
  --gpu-fraction N    Ray GPU allocation per scenario (SIM_GPU_FRACTION; 0.5)
  --threads N         Ray worker threads (SIM_THREADS; 128)
  --dry-run           Print the command and paths without creating or running anything
  -h, --help          Show help

Paths can also be set with DP_DIR and NUPLAN_{DEVKIT,DATA,MAPS,EXP}_ROOT.
Defaults under ROOT: Diffusion-Planner, nuplan-devkit, dataset/nuplan,
dataset/nuplan/maps, and dataset/nuplan/exp respectively.

Results use nuPlan's original output layout:
  $NUPLAN_EXP_ROOT/exp/simulation/$CHALLENGE/score_function/$SPLIT/
    ${RUN_NAME}_g${GAMMA_WITHOUT_DOT}_k${STEPS}_YYYY-MM-DD-HH-MM-SS/
Each directory contains console.log, code/hydra/config.yaml, metrics/,
aggregator_metric/, simulation/ and nuboard_*.nuboard (as produced by nuPlan).

This starts one evaluation. It reuses existing weights; it never trains,
rebuilds caches, replaces experiment outputs, or modifies the model code.
USAGE
}

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(dirname -- "$CODE")}"
PY="${PY:-python}"
SCORE_CONFIG="${SCORE_CONFIG:-}"
SCORE_CHECKPOINT="${SCORE_CHECKPOINT:-}"
RUN_NAME="${RUN_NAME:-score_function}"
STEPS="${STEPS:-5}"
GAMMA="${GAMMA:-0.1}"
EVAL_GPUS="${EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
SPLIT="${SPLIT:-test14-hard}"
CHALLENGE="${CHALLENGE:-closed_loop_nonreactive_agents}"
SIM_GPU_FRACTION="${SIM_GPU_FRACTION:-0.5}"
SIM_THREADS="${SIM_THREADS:-128}"
DRY_RUN=0

while (( $# )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift; continue ;;
    --config|--checkpoint|--name|--steps|--gamma|--gpus|--python|--root|--split|--challenge|--gpu-fraction|--threads)
      (( $# >= 2 )) && [[ -n "$2" && "$2" != --* ]] || die "$1 requires a value." ;;
    *) die "Unknown option: $1 (use --help)." ;;
  esac
  case "$1" in
    --config) SCORE_CONFIG="$2" ;;
    --checkpoint) SCORE_CHECKPOINT="$2" ;;
    --name) RUN_NAME="$2" ;;
    --steps) STEPS="$2" ;;
    --gamma) GAMMA="$2" ;;
    --gpus) EVAL_GPUS="$2" ;;
    --python) PY="$2" ;;
    --root) ROOT="$2" ;;
    --split) SPLIT="$2" ;;
    --challenge) CHALLENGE="$2" ;;
    --gpu-fraction) SIM_GPU_FRACTION="$2" ;;
    --threads) SIM_THREADS="$2" ;;
  esac
  shift 2
done

[[ -f "$CODE/score_function/planner/planner.py" && -f "$CODE/configs/nuplan_planner.yaml" ]] \
  || die 'Place sim_score_planner.sh in the Score_Function repository root.'
[[ -n "$SCORE_CONFIG" && -n "$SCORE_CHECKPOINT" ]] \
  || die 'Supply --config and --checkpoint (or SCORE_CONFIG and SCORE_CHECKPOINT).'
[[ "$RUN_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || die '--name must contain only letters, digits, underscores or hyphens.'
[[ "$STEPS" =~ ^[0-9]+$ ]] || die '--steps must be a nonnegative integer.'
[[ "$GAMMA" =~ ^[0-9]+([.][0-9]+)?$ ]] || die '--gamma must be a nonnegative number, e.g. 0.1.'
[[ "$SIM_GPU_FRACTION" =~ ^[0-9]+([.][0-9]+)?$ ]] || die '--gpu-fraction must be a number, e.g. 0.5.'
[[ "$SIM_THREADS" =~ ^[1-9][0-9]*$ ]] || die '--threads must be a positive integer.'
[[ -n "$EVAL_GPUS" && "$EVAL_GPUS" != ,* && "$EVAL_GPUS" != *, && "$EVAL_GPUS" != *,,* && "$EVAL_GPUS" != *[[:space:]]* ]] \
  || die '--gpus must be a nonempty comma-separated GPU list.'
case "$SPLIT" in
  val14) SCENARIO_BUILDER=nuplan ;;
  test14-hard|test14-random) SCENARIO_BUILDER=nuplan_challenge ;;
  *) die 'Supported splits: test14-hard, test14-random, val14.' ;;
esac
case "$CHALLENGE" in
  closed_loop_nonreactive_agents|closed_loop_reactive_agents) ;;
  *) die 'Unsupported --challenge.' ;;
esac
PY="$(command -v -- "$PY")" || die 'Python interpreter not found.'
ROOT="$(realpath -e -- "$ROOT")"
SCORE_CONFIG="$(realpath -e -- "$SCORE_CONFIG")"
SCORE_CHECKPOINT="$(realpath -e -- "$SCORE_CHECKPOINT")"
[[ -f "$SCORE_CONFIG" && -f "$SCORE_CHECKPOINT" ]] || die 'Config/checkpoint must be files.'

DP_DIR="${DP_DIR:-$ROOT/Diffusion-Planner}"
export NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-$ROOT/nuplan-devkit}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-$ROOT/dataset/nuplan}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$ROOT/dataset/nuplan/maps}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-$ROOT/dataset/nuplan/exp}"
[[ -d "$DP_DIR/diffusion_planner" ]] || die "Diffusion-Planner not found: $DP_DIR"
[[ -f "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" ]] \
  || die "nuPlan run_simulation.py not found under $NUPLAN_DEVKIT_ROOT"
export CUDA_VISIBLE_DEVICES="$EVAL_GPUS" HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 NODE_RANK=0
export PYTHONPATH="$CODE:$DP_DIR:$NUPLAN_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

STAMP="$(date +%Y-%m-%d-%H-%M-%S)"
RUN_LABEL="${RUN_NAME}_g${GAMMA//./}_k${STEPS}_${STAMP}"
EXPERIMENT_UID="score_function/$SPLIT/$RUN_LABEL"
RESULT_DIR="$NUPLAN_EXP_ROOT/exp/simulation/$CHALLENGE/$EXPERIMENT_UID"
HYDRA_CONFIG_DIR="$RESULT_DIR/code/score_function_hydra"

# Like the official DP launcher, set experiment_uid and let nuPlan construct
# output_dir. The challenge remains in the path, which its aggregator requires.
CMD=(
  "$PY" -u "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py"
  "+simulation=$CHALLENGE"
  planner=score_function_planner
  "planner.score_function_planner.score_config=$SCORE_CONFIG"
  "planner.score_function_planner.score_checkpoint=$SCORE_CHECKPOINT"
  "planner.score_function_planner.root=$ROOT"
  planner.score_function_planner.device=cuda
  "planner.score_function_planner.gamma=$GAMMA"
  "planner.score_function_planner.steps=$STEPS"
  planner.score_function_planner.heading_projection=null
  planner.score_function_planner.trace_dir=null
  "scenario_builder=$SCENARIO_BUILDER" "scenario_filter=$SPLIT"
  "group=$NUPLAN_EXP_ROOT/exp" "experiment_uid=$EXPERIMENT_UID"
  verbose=true worker=ray_distributed "worker.threads_per_node=$SIM_THREADS"
  distributed_mode=SINGLE_NODE
  "number_of_gpus_allocated_per_simulation=$SIM_GPU_FRACTION"
  enable_simulation_progress_bar=true exit_on_failure=true run_metric=true
  "hydra.searchpath=[file://$HYDRA_CONFIG_DIR,pkg://diffusion_planner.config.scenario_filter,pkg://diffusion_planner.config,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]"
)

printf 'Code: %s\nConfig: %s\nCheckpoint: %s\nGPUs: %s\nResults: %s\n' \
  "$CODE" "$SCORE_CONFIG" "$SCORE_CHECKPOINT" "$EVAL_GPUS" "$RESULT_DIR"
printf 'Command: '; printf '%q ' "${CMD[@]}"; printf '\n'
(( DRY_RUN )) && exit 0
[[ ! -e "$RESULT_DIR" ]] || die "Output already exists; use another --name: $RESULT_DIR"
mkdir -p "$HYDRA_CONFIG_DIR/planner"
cp -- "$CODE/configs/nuplan_planner.yaml" "$HYDRA_CONFIG_DIR/planner/score_function_planner.yaml"
cd -- "$CODE"
"${CMD[@]}" 2>&1 | tee "$RESULT_DIR/console.log"

# nuPlan can exit successfully even if aggregation finds no matching files.
# Catch that specific failure here; do not silently start the next experiment.
shopt -s nullglob
AGGREGATED_FILES=("$RESULT_DIR/aggregator_metric/"*.parquet)
(( ${#AGGREGATED_FILES[@]} )) \
  || die "Simulation exited without an aggregated result. See $RESULT_DIR/console.log"
printf '\nEvaluation finished. Official aggregated results:\n'
printf '  %s\n' "${AGGREGATED_FILES[@]}"
NUBOARD_FILES=("$RESULT_DIR/"*.nuboard)
if (( ${#NUBOARD_FILES[@]} )); then
  printf 'NuBoard files:\n'; printf '  %s\n' "${NUBOARD_FILES[@]}"
fi
