# UrbanDriver + S10-G: one K=20 hard14 NR run

This experiment takes the completed native UrbanDriver evaluation (approximately
43/100) as its paired baseline. It performs one full K=20 evaluation, not a
seed-by-K matrix. There is no training and no DP trajectory sampling.

Fixed settings: existing S10-G final EMA, epoch 529, sigma 0.1, gamma 0.1,
20 updates, heading projection enabled. The existing frozen score source and
weights under `score_G_transfer_1014_v1` are reused without modification.

## Server commands

Use a separate checkout/worktree of this branch under
`/home/data/wyk/Score_Function_UrbanDriver_1017`. Do not switch the checkout used
by a running DP matrix job.

```bash
conda activate /home/data/wyk/envs/urbandriver
cd /home/data/wyk/Score_Function_UrbanDriver_1017
python -m pip install --no-deps -r requirements_urban_driver_score.txt
bash scripts/sim_urban_driver_score_k20.sh
```

The launcher selects the latest `raw_8gpu_*` result under
`/home/data/wyk/dataset/nuplan/exp/exp/simulation/closed_loop_nonreactive_agents/urbandriver/test14-hard/`.
It prints that exact baseline path and score before starting. If the native
result lives elsewhere, specify it explicitly:

```bash
BASELINE_RESULT=/absolute/path/to/completed/raw_result \
  bash scripts/sim_urban_driver_score_k20.sh
```

It requires all 272 native scenarios to have succeeded and compares their
`(log_name, token)` identities with the existing DP hard14 matrix plan when that
plan exists. It uses the native run's saved, resolved Hydra configuration,
including its scenario filter/builder, model configuration, controller, metrics,
seed and Ray resource settings. Result paths are relocated into a fresh directory.
The official nuPlan `run_simulation.py` remains the simulation entry point.

The script runs a dataset-free interface check, a one-scenario **K20** smoke
simulation on the first visible GPU, then one complete K20 simulation with
eight visible GPUs and one GPU reserved per scenario. A failed check stops the
script before the complete run. There is no complete K0 evaluation.

## Adapter contract

1. Load native UrbanDriver through nuPlan's original `build_planners` / MLPlanner
   checkpoint path. Model loading occurs in the assigned worker's `initialize`.
2. Generate the unmodified native 16-pose / 8-second trajectory. Sample nuPlan's
   own `InterpolatedTrajectory` at t=0.1,...,8.0 seconds, including interpolation
   between the current ego anchor and the first future point. Angular wrap is
   handled by the native interpolation. Express rear-axle poses relative to the
   current rear axle as x,y,cos(theta),sin(theta).
3. Normalize with the frozen DP ego statistics used to train S10-G. Extract
   current conditions with DP's official observation adapter and frozen scene
   and route encoders. Only those encoder weights are loaded; the DP trajectory
   generator is not constructed or run. No future ground truth or invented
   neighbor predictions are used.
4. Call the existing `refine_ego` unchanged: x <- x + gamma * sigma^2 * score,
   followed by its existing physical heading projection, exactly 20 times.
5. Convert the refined 80 poses through nuPlan's original
   `transform_predictions_to_states` and `InterpolatedTrajectory`.

K=0 is supported only for an interface check: it returns the exact native
trajectory object before sampling, normalization or condition encoding. It does
not change baseline sampling or require score checkpoint loading. The K20 output
uses the score model's 0.1-second grid; the native baseline remains on its original
0.5-second grid.

`runtime.py` only adapts checkpoint deserialization for torch 1.12 and extracts
the original encoder weights. It checks architecture, sigma, EMA selection,
normalizer and frozen DP provenance. No global monkeypatching is performed.

## Results

The launcher prints the result directory under
`.../closed_loop_nonreactive_agents/urbandriver_score/test14-hard/S10G_K20_<timestamp>`.
Its `smoke/` directory is separate from the full run. Important files:

- `console.log`: full evaluation log.
- `protocol.json`: native reference, settings, checkpoint hashes and scenario IDs.
- `code/hydra/config.yaml`: actual full evaluation configuration.
- `refinement_audit/*.json`: one first-frame K20 execution audit per scenario.
- `comparison.json`: native score, refined score, difference, better/worse/tied counts.
- `paired_scores.csv`: all 272 paired scenario scores.

The original metric aggregator determines the reported final score. Scores in
`comparison.json` use 0..100; per-scenario score columns in the CSV remain 0..1
and `delta_100` uses points on 0..100. Failures or missing scenarios are not
silently dropped.

Local tests exercise official nuPlan trajectory conversion with synthetic
inputs, K0 object identity, sampling/frame/heading correctness, an exact known
20-step update, and old-torch loading arguments. The user's actual checkpoints,
CUDA runtime, maps and databases are validated by the server-side preflight and
smoke simulation; a local CPU test is not a completed closed-loop evaluation.
