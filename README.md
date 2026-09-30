# Score Function

Time-independent conditional score learning for ego-trajectory refinement with [Diffusion Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner).

The model learns a fixed-noise-scale score from expert trajectories, conditioned on frozen scene and route features. At inference, it refines the ego future predicted by Diffusion Planner while keeping the current state and neighboring-agent predictions fixed.

## Method

The score network combines temporal residual convolutions, scene cross-attention, and route conditioning. Its normalized trajectory input and score output both have shape `[B, 80, 4]`, with trajectory coordinates `(x, y, cos(theta), sin(theta))`. Scores are derivatives with respect to these normalized coordinates.

Training uses fixed-scale denoising score matching:

$$
y = x + \sigma\epsilon, \qquad \epsilon \sim \mathcal{N}(0,I),
$$

$$
\mathcal{L}_{\mathrm{DSM}} =
\mathbb{E}\left[\left\|\sigma s_\theta(y,C,R)+\epsilon\right\|^2\right].
$$

Refinement starts from the planner prediction and applies:

$$
x_{k+1} = x_k + \gamma\sigma^2 s_\theta(x_k,C,R).
$$

The score network receives no diffusion timestep. Scene and route features remain fixed during refinement. Optional heading projection normalizes the heading vectors after each update.

Two parameterizations share this same fixed-sigma DSM objective and fixed-step refinement:

| `model.parameterization` | Score computation |
|---|---|
| `"score"` (default) | Directly predicts the `[B, 80, 4]` score. |
| `"energy"` | Predicts one scalar energy per trajectory and returns $s_\theta=-\nabla_x E_\theta$, also `[B, 80, 4]`. |

The energy is the sum of learned per-token scalar contributions. It adds an input-gradient computation at inference and double backward during DSM training; its attention uses explicit math operations to support those derivatives. Expect higher runtime and memory cost. This is an engineering option, with no evidence yet of better trajectory quality or closed-loop performance. See the [parameterization guide](docs/parameterization.md) for the API, checks, and checkpoint rules.

| Setting | Default |
|---|---:|
| Hidden dimension / attention heads | 192 / 6 |
| Noise scale $\sigma$ | 0.05 |
| Refinement step factor $\gamma$ | 0.1 |
| Refinement steps | 5 |
| Global batch size | 2048 |
| Optimizer | AdamW |
| Peak learning rate / warmup | 5e-4 / 5 epochs from 5e-5 |
| AdamW weight decay | 0.01 |
| Training budget | At least 500 epochs and 244,000 optimizer updates |
| Learning-rate decay / early stopping | Disabled in the aligned defaults |
| Paired state augmentation | Official DP `StatePerturbation`, probability 0.5 |
| EMA decay | 0.999 |

The current defaults align training budget and state augmentation with the official
DP recipe. Actual epochs increase when the internal validation split leaves fewer
training frames, so both budget floors are met. Architecture and the fixed-sigma
ego DSM objective remain unchanged. See [alignment and long-refinement diagnostics](docs/alignment.md)
for migration from existing runs and the exact scope of this alignment.

## Installation

Use a Linux environment with Python 3.9+, PyTorch 2.x, NumPy < 2, and the dependencies required by Diffusion Planner and the [nuPlan devkit](https://github.com/motional/nuplan-devkit).

```bash
git clone https://github.com/ecowyk/Score_Function.git
cd Score_Function
python -m pip install --no-deps --no-build-isolation -e .
```

Download the official Diffusion Planner checkpoint and prepare the nuPlan training databases and maps separately.

## Configuration

Edit [configs/score_function.json](configs/score_function.json) to configure paths, data processing, training, and refinement. Relative paths are resolved against `--root`.

Expected workspace layout with the default paths:

```text
workspace/
├── Score_Function/
├── Diffusion-Planner/
│   └── checkpoints/
│       ├── args.json
│       └── model.pth
├── nuplan-devkit/
├── dataset/nuplan/
│   ├── nuplan-v1.1/trainval/
│   └── maps/
├── score_data/score_function/
└── outputs/score_function_aligned/
```

```bash
ROOT=/path/to/workspace

python -m score_function check-config \
  --config configs/score_function.json --root "$ROOT"
```

For energy training, use [configs/score_function_energy.json](configs/score_function_energy.json). It changes only `model.parameterization` to `"energy"` and `paths.run_dir` to `outputs/score_function_energy_aligned`; it reuses the same feature cache. The equivalent CLI override is:

```bash
python -m score_function check-config \
  --config configs/score_function.json --root "$ROOT" \
  --set 'model.parameterization="energy"' \
  --set 'paths.run_dir="outputs/score_function_energy_aligned"'
```

The single quotes preserve the JSON double quotes required by `--set`. Keep a separate run directory for each parameterization.

## Training

### Train from an existing cache

The dedicated entry point is `train_score_branch.py`. For an already prepared
experiment, start only training with the active Python environment:

```bash
bash scripts/train.sh --gpus 0 \
  --config /path/to/E05-L.json --root /path/to/workspace
```

Use `--python /path/to/env/bin/python` to select an environment explicitly.
The equivalent single-GPU Python command is:

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_score_branch.py \
  --config /path/to/E05-L.json --root /path/to/workspace
```

Both score and energy configurations use this same entry. `scripts/train.sh`
only starts training: it does not install dependencies, prepare data, rebuild
caches, or run simulation. Pass `--gpus 0,1` for two processes; global batch must
be divisible by `GPU count * microbatch_size`. A comma-separated mask alone does
not distribute a plain Python invocation.

Terminal/tmux runs display progress bars for cache-index checks, initial and
epoch validation, and each training epoch. Training shows completed updates,
elapsed time, current-epoch ETA, updates/second, rank-zero DSM loss, learning
rate, global step and peak allocated GiB. ETA is for the current phase, not the
whole run including future validation or early stopping. In DDP only rank zero
displays a bar; periodic DSM log records still reduce the loss across ranks.

When output is redirected/piped, progress defaults to plain lines every 30
seconds plus phase boundaries; training metrics also follow `log_every_updates`.
Choose `--progress on` to force terminal bars, or `--progress off` for plain logs.
Plain output and training tracebacks are also saved automatically in
`paths.run_dir/score/console.log`; `status.json` identifies the current phase.

To keep training attached to a durable terminal, first open a tmux shell, then
run the command above inside it:

```bash
tmux new-session -s e05_l_train
# Activate your Python environment here, or pass --python explicitly.
# Run scripts/train.sh with explicit --config, --root and --gpus arguments.
```

Detach with `Ctrl-b d`; return with `tmux attach -t e05_l_train`. Opening the
shell first keeps errors visible even if training exits. Use `--resume` for an
interrupted run with the same checkout, configuration, data and GPU count.
**Keep existing runs on their original checkout.** This update changes
source hashes, so it cannot resume a pre-update `last.pt`; old selected `best.pt`
files remain usable for evaluation. Use a separate checkout/run directory for
new experiments. Existing generated configurations retain their original budget.
Use `scripts/prepare_aligned_config.py` to create a new aligned configuration while
preserving an existing model variant, sigma and data paths; see the alignment guide.

### Prepare data and run a full pipeline

For the eight-model sigma/temporal/neighbor ablation on eight A100 GPUs:

```bash
bash scripts/launch_experiments.sh \
  --root /path/to/workspace \
  --database-dir /path/to/nuplan/trainval \
  --maps-dir /path/to/nuplan/maps
tmux attach -t score_matrix_1m_wyk
```

This bootstraps the `score_function_wyk` environment and official dependencies,
prepares shared caches, and runs one experiment per GPU in tmux. The official
training logs supply candidates for a global maximum of 1,000,000 scenarios,
without timestamp thinning. See the
[experiment suite guide](docs/experiment_suite.md) for the matrix, existing-environment
option, paths, logs, and resume commands.

The training pipeline prepares nuPlan features and a clean cache for split identity
and validation. With online augmentation enabled, training reads the original NPZs,
jointly perturbs the scene and target using official DP code, then recomputes frozen
scene/route features. Neighbor-conditioned variants also predict fresh neighbors
from that same augmented observation. Validation remains clean and deterministic.
Checkpoint selection uses validation DSM with EMA. The aligned defaults complete
the full budget without plateau LR reduction or early stopping; older configs
without the new options retain their original behavior.

Run on eight GPUs in tmux:

```bash
bash scripts/train_tmux.sh "$ROOT" 0,1,2,3,4,5,6,7
tmux attach -t score_function_train_wyk
```

Change the GPU list to match the available devices. The global batch size must be divisible by the number of GPUs times the microbatch size.

To use the energy configuration with this launcher:

```bash
bash scripts/train_tmux.sh "$ROOT" 0,1,2,3,4,5,6,7 \
  "$PWD/configs/score_function_energy.json" fresh
```

The launcher uses the same tmux session name for both configurations; finish or close an existing launcher session before starting another. Run the guide's [GPU and cache smoke checks](docs/parameterization.md#server-checks-and-training) before a full energy run.

Resume an interrupted run:

```bash
bash scripts/train_tmux.sh "$ROOT" 0,1,2,3,4,5,6,7 \
  "$PWD/configs/score_function.json" resume
```

Checkpoints are saved under `outputs/score_function_aligned/score/`:

- `best.pt`: selected score model for evaluation.
- `last.pt`: full training state for resuming.
- `history.json`: training and validation metrics.

The energy configuration writes these files under `outputs/score_function_energy_aligned/score/`. Older selected direct-score checkpoints without a `parameterization` field load as `"score"`; score/energy mismatches are rejected. Strict resume still requires unchanged source hashes, data, world size, and training configuration, so selected-weight compatibility does not imply that an older-version training run can resume after this code change.

## Evaluation

Inspect long refinement on existing score or energy weights before further
closed-loop evaluations:

```bash
python -m score_function visualize-refinement \
  --config /path/to/the/checkpoints/original-config.json --root "$ROOT" \
  --checkpoint /path/to/score/best.pt \
  --split val --max-samples 8 --steps 5000 \
  --output "$ROOT/outputs/refinement_5000_val"
```

This fixed-scene diagnostic uses both DP outputs and expert-plus-fixed-noise
initializations. It writes sparse trajectory snapshots, physical motion plots,
per-update diagnostics, energy curves for energy models, and an HTML index. It
does not advance the simulator or retrain weights. Use a new output directory
for each call. Details and train/validation commands are in the alignment guide.

Evaluate fixed-noise expert-trajectory diagnostics:

```bash
python eval_score_branch.py \
  --config configs/score_function.json --root "$ROOT" \
  --checkpoint "$ROOT/outputs/score_function_aligned/score/best.pt" \
  --split val
```

Compare planner trajectories before and after refinement:

```bash
python -m score_function evaluate-planner \
  --config configs/score_function.json --root "$ROOT" \
  --checkpoint "$ROOT/outputs/score_function_aligned/score/best.pt" \
  --split val --max-samples 100
```

Offline evaluation saves metrics, per-sample results, refinement traces, and trajectory plots. Use `--set refinement.steps=3` or other existing configuration keys to override evaluation settings.

For an energy checkpoint, use `--config configs/score_function_energy.json` and `--checkpoint "$ROOT/outputs/score_function_energy_aligned/score/best.pt"` in both commands. Evaluation must match the checkpoint's parameterization, architecture, and training sigma.

For official nuPlan closed-loop evaluation, configure [configs/nuplan_planner.yaml](configs/nuplan_planner.yaml) and use `score_function.planner.planner.ScoreFunctionPlanner` in the nuPlan simulation runner. Setting `gamma=0` disables refinement for a paired baseline.

## Project Structure

```text
score_function/
├── model/          # Score network, neural modules, and refinement
├── loss.py         # Fixed-scale DSM objective
├── train_epoch.py  # Learning and validation loops
├── train.py        # Training orchestration and checkpoint selection
├── data_process/   # nuPlan preprocessing and feature caching
├── planner/        # Official nuPlan planner integration
├── evaluation/     # Metrics, visualization, and offline evaluation
├── utils/          # Configuration, datasets, DDP, and checkpoint utilities
└── tools/          # Environment and small-batch checks
```

Training and evaluation entry points are in the repository root. Launch scripts are in `scripts/`, configurations in `configs/`, and tests in `tests/`.
