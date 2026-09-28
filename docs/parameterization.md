# Score / energy parameterization

`model.parameterization` selects how the branch represents the score. The default is `"score"`; `"energy"` learns a scalar potential and differentiates it with respect to the normalized ego future. Both use the existing temporal and conditional backbone, fixed-sigma DSM loss, AdamW training, EMA selection, and fixed-step refinement. This change adds no line search, distillation, or new optimizer.

## Method and API

For normalized ego trajectories `x` of shape `[B, 80, 4]` and fixed conditioning `C, R`:

| Mode | Learned output head | Public score |
|---|---|---|
| `score` | Four values per trajectory token | Direct output `s_theta(x, C, R)` |
| `energy` | One scalar contribution per trajectory token | `-grad_x E_theta(x, C, R)` |

The energy uses a **sum**, not a mean, across the 80 token contributions:

$$
E_\theta(x,C,R)=\sum_{t=1}^{80} e_{\theta,t}(x,C,R),
\qquad s_\theta(x,C,R)=-\nabla_x E_\theta(x,C,R).
$$

Each contribution can depend on other tokens through the configured backbone. The gradient is taken in normalized ego coordinates. Scene, route, and optional predicted-neighbor conditioning stay fixed during refinement; current ego state and neighbor predictions are not optimization variables.

For a local backbone with receptive-field upper bound `R`, differentiating the sum of local energies can give the score a receptive-field upper bound of `min(T, 2*R-1)`. The model records the backbone bound as `feature_receptive_field` and the score bound as `temporal_receptive_field`. For the default 80-point backbone these are 37 and 73 in energy mode; direct-score mode retains its 37-point bound. Global temporal attention gives a full-horizon bound in either mode.

```python
score = branch(ego_traj_norm, scene_context, route_embedding)
score = branch.predict_score(ego_traj_norm, scene_context, route_embedding)
# Both return [B, 80, 4] in either parameterization.

energy = branch.energy_value(ego_traj_norm, scene_context, route_embedding)
# Returns [B]; available only when parameterization == "energy".
```

The optional `neighbor_future` and `neighbor_valid` arguments are supported by all three methods when neighbor conditioning is enabled. Calling `energy_value` on a direct-score branch raises an error.

Energy `forward` / `predict_score` work inside `torch.no_grad()` and `torch.inference_mode()`, including when all model parameters are frozen. They locally enable the input differentiation needed to compute the score and return detached scores for these inference callers. In normal gradient-enabled training, they preserve the derivative graph so DSM backward can train the energy parameters. `energy_value` itself follows the caller's ordinary gradient mode. Set `branch.eval()` for deterministic inference and refinement.

The energy's absolute additive offset is unidentifiable from DSM. The energy head omits its final scalar bias because its input gradient would always be zero. Raw energy values are not calibrated likelihoods or comparable quality scores across independently trained models.

## Unchanged training and refinement

Both parameterizations train with one fixed noise scale per run:

$$
y=x+\sigma\epsilon,\quad \epsilon\sim\mathcal{N}(0,I),\qquad
\mathcal{L}=\operatorname{mean}\big((\sigma s_\theta(y,C,R)+\epsilon)^2\big).
$$

Both use exactly the configured number of refinement steps:

$$
x_{k+1}=x_k+\gamma\sigma^2 s_\theta(x_k,C,R).
$$

For energy mode, this is a fixed-step negative-energy-gradient update. Optional heading projection remains the same. A fixed step size, especially with projection, does not guarantee decreasing energy at every step, convergence, or improved planner quality. `gamma=0` or `steps=0` keeps the existing exact disabled-baseline path.

The energy branch uses explicit math attention for scene attention and, when enabled, temporal and neighbor attention so that the derivative can be differentiated again during DSM training. Training requires double backward; each inference score also requires an input-gradient computation. This increases computation and memory use relative to direct prediction. Measure the actual GPU footprint and latency before choosing a production batch size. No trajectory-quality or closed-loop improvement has been established by this implementation alone.

## Configuration and cache reuse

The supplied configurations differ in exactly two fields:

| Field | `configs/score_function.json` | `configs/score_function_energy.json` |
|---|---|---|
| `model.parameterization` | `"score"` | `"energy"` |
| `paths.run_dir` | `outputs/score_function` | `outputs/score_function_energy` |

All other settings, including sigma, optimizer, data paths, cache, and refinement, are identical. Frozen scene/route features and targets do not depend on this parameterization, so the energy run can reuse a valid existing cache at the configured path. Cache provenance checks still apply; switching parameterization is not a reason to bypass them or rebuild a valid cache.

Run commands from the repository root in the prepared Python environment:

```bash
ROOT=/path/to/workspace

python -m score_function check-config \
  --config configs/score_function_energy.json --root "$ROOT"
```

Alternatively, override the default config with JSON string values. The outer single quotes below must preserve the inner double quotes:

```bash
python -m score_function check-config \
  --config configs/score_function.json --root "$ROOT" \
  --set 'model.parameterization="energy"' \
  --set 'paths.run_dir="outputs/score_function_energy"'
```

`check-config` only validates and prints the resolved configuration. If using overrides for training, pass the same overrides to every subsequent command. Keep each trained parameterization in its own output directory to preserve its checkpoints and logs. The same separation is needed for repeated independent runs.

## Server checks and training

These are commands to run on a configured CUDA server, not a report that they have already passed there. The distributed example uses two visible GPUs; adjust the list and `--nproc_per_node` together. The global batch size must be divisible by `world_size * microbatch_size`.

```bash
# Synthetic two-update forward/backward and distributed-reduction check.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m score_function check-ddp \
  --config configs/score_function_energy.json --root "$ROOT"

# Small fixed-corruption fit using the existing training cache.
CUDA_VISIBLE_DEVICES=0 python -m score_function smoke \
  --config configs/score_function_energy.json --root "$ROOT"
```

`check-ddp` reports the parameterization, number of optimizer updates, finite-gradient checks, and peak allocated GPU memory. It does not need the real feature cache. `smoke` needs a valid cache, uses the configured four frames and 200 updates, writes `smoke.json` into the run directory, and discards the fitted weights. Passing it only checks finite optimization and a decrease in the fixed-corruption loss; it is not a generalization result.

If the configured microbatch is too large for energy training, repeat `check-ddp` with a smaller divisor such as `--set training.microbatch_size=8`, and use the same override for training. Keeping `training.batch_size` unchanged preserves the effective global batch through accumulation. For the tmux launcher, make that setting in a separate JSON configuration because the launcher accepts a config path rather than arbitrary CLI overrides.

If the shared cache has not yet been prepared, run the existing `prepare` and `cache` stages using the configured data and official planner paths before `smoke`. A full training launch after successful checks can use:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m score_function train \
  --config configs/score_function_energy.json --root "$ROOT"
```

For the complete tmux pipeline and eight GPUs:

```bash
bash scripts/train_tmux.sh "$ROOT" 0,1,2,3,4,5,6,7 \
  "$PWD/configs/score_function_energy.json" fresh
tmux attach -t score_function_train_wyk
```

The launcher session name is shared with the direct-score launcher; an existing session blocks a second launch. The energy run writes selected weights to `outputs/score_function_energy/score/best.pt` and resumable state to `outputs/score_function_energy/score/last.pt`, relative to `$ROOT`.

## Evaluation and checkpoints

Use the matching energy configuration for both evaluation paths:

```bash
python eval_score_branch.py \
  --config configs/score_function_energy.json --root "$ROOT" \
  --checkpoint "$ROOT/outputs/score_function_energy/score/best.pt" \
  --split val

python -m score_function evaluate-planner \
  --config configs/score_function_energy.json --root "$ROOT" \
  --checkpoint "$ROOT/outputs/score_function_energy/score/best.pt" \
  --split val --max-samples 100
```

Selected-checkpoint loading checks parameterization, architecture, and training sigma. Loading direct-score weights as an energy branch, or energy weights as a direct-score branch, is rejected. The heads have different output shapes; this option is not a conversion of already trained direct-score weights into energy weights.

Older direct-score configurations and selected checkpoints that omit `model.parameterization` are interpreted as `"score"`. This compatibility concerns selected-weight loading for evaluation and inference. Training resume continues to enforce its strict source-hash, configuration, data, and world-size guard. An older-version training run is not promised to resume after this source change, and changing parameterization requires a new run with matching weights.

For a paired comparison, keep the same data, frozen planner, sigma, refinement settings, and evaluation samples. Report DSM, trajectory diagnostics, synchronized refinement latency, and actual GPU memory separately. Offline diagnostics and implementation checks do not establish a nuPlan closed-loop gain; see [validation records](validation.md) for the scope of completed checks.
