# Score Function

Time-independent conditional score learning for ego-trajectory refinement with [Diffusion Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner).

The model learns a fixed-noise-scale score from expert trajectories, conditioned on frozen scene and route features. At inference, it refines the ego future predicted by Diffusion Planner while keeping the current state and neighboring-agent predictions fixed.

## Method

The score network combines temporal residual convolutions, scene cross-attention, and route conditioning. Its input and output are normalized ego trajectories with shape `[B, 80, 4]`, where each point contains `(x, y, cos(theta), sin(theta))`.

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

| Setting | Default |
|---|---:|
| Hidden dimension / attention heads | 192 / 6 |
| Noise scale $\sigma$ | 0.05 |
| Refinement step factor $\gamma$ | 0.1 |
| Refinement steps | 5 |
| Global batch size | 2048 |
| Optimizer | AdamW |
| Initial learning rate | 1e-4 |
| EMA decay | 0.999 |

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
└── outputs/score_function/
```

```bash
ROOT=/path/to/workspace

python -m score_function check-config \
  --config configs/score_function.json --root "$ROOT"
```

## Training

For the eight-model sigma/temporal/neighbor ablation on eight A100 GPUs:

```bash
bash scripts/launch_experiments.sh \
  --root /path/to/workspace \
  --database-dir /path/to/nuplan/trainval \
  --maps-dir /path/to/nuplan/maps
tmux attach -t score_matrix_v1_wyk
```

This bootstraps the `score_function_wyk` environment and official dependencies,
prepares shared caches, and runs one experiment per GPU in tmux. Full eligible
training data is used without timestamp thinning. See the
[experiment suite guide](docs/experiment_suite.md) for the matrix, existing-environment
option, paths, logs, and resume commands.

The training pipeline prepares nuPlan features, caches the frozen scene and route encodings, and trains the score branch. Checkpoint selection uses validation DSM with EMA and plateau-based learning-rate reduction and early stopping.

Run on eight GPUs in tmux:

```bash
bash scripts/train_tmux.sh "$ROOT" 0,1,2,3,4,5,6,7
tmux attach -t score_function_train_wyk
```

Change the GPU list to match the available devices. The global batch size must be divisible by the number of GPUs times the microbatch size.

Resume an interrupted run:

```bash
bash scripts/train_tmux.sh "$ROOT" 0,1,2,3,4,5,6,7 \
  "$PWD/configs/score_function.json" resume
```

Checkpoints are saved under `outputs/score_function/score/`:

- `best.pt`: selected score model for evaluation.
- `last.pt`: full training state for resuming.
- `history.json`: training and validation metrics.

## Evaluation

Evaluate fixed-noise expert-trajectory diagnostics:

```bash
python eval_score_branch.py \
  --config configs/score_function.json --root "$ROOT" \
  --checkpoint "$ROOT/outputs/score_function/score/best.pt" \
  --split val
```

Compare planner trajectories before and after refinement:

```bash
python -m score_function evaluate-planner \
  --config configs/score_function.json --root "$ROOT" \
  --checkpoint "$ROOT/outputs/score_function/score/best.pt" \
  --split val --max-samples 100
```

Offline evaluation saves metrics, per-sample results, refinement traces, and trajectory plots. Use `--set refinement.steps=3` or other existing configuration keys to override evaluation settings.

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
