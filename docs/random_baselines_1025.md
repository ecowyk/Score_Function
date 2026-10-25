# Test14-Random NR: six native / S10-G K20 pairs on eight GPUs

The queue runs UrbanDriver, PlanTF epoch17, PlanTF epoch24 (`last.ckpt`),
PDM-Open, GC-PGP and PLUTO w/o rule refinement. Each runs K0 and K20: 12 full
evaluations. Diffusion Planner is never scheduled. Its already completed
`random_NR_seed20260920_K0` result and matrix plan supply the evaluation settings
and exact `(log_name, scenario_token)` identities. The number of scenarios comes
from that actual cohort; it is not assumed to be 272, and this frozen subset is
not mislabeled as all scenarios in the Test14 databases.

All full runs expose GPU 0–7, use Ray, and reserve one GPU per simulation with
eight concurrent simulations. Existing `urbandriver`, `plantf` and `pluto`
environments are selected automatically in separate processes. Their packages
are not upgraded or reinstalled. PDM-Open retains its official CPU native model;
its score/conditioning run on GPU, as in the completed hard_NR experiment.

Each pair uses the same native checkpoint and fixed gamma=0.1, sigma=0.1,
K=20, heading projection, and the existing S10-G **final EMA at epoch 529**.
PLUTO retains reference lines and learned trajectory selection, without its rule
postprocessing. K0 wrappers return the exact native trajectory without score
loading, normalization or resampling. PlanTF's two checkpoint variants remain
separate rows.

## Run on the original baseline node

```bash
cd /home/data/wyk/Score_Function_TwoNode_1025
export QUEUE_ROOT=/home/data/wyk/dataset/nuplan/exp/random_baselines_1025_v1
bash scripts/sim_random_baselines_1025.sh
```

This is a foreground command; use the provided top-level detached launcher or
an existing tmux session for unattended execution. The output root is fixed so
the same command resumes validated completed stages instead of creating an
accidental duplicate experiment. Do not run it on both nodes.

The script first checks the completed DP reference and all six native hard_NR
sources, hashes the checkpoints/configurations/source, then checks every model
environment before launching the first full evaluation. On occupied GPUs it
waits without terminating other tasks. Each pair performs K0/K20 one-scene
integration checks before its full evaluations. Every full result must contain
all expected successful runner reports and official aggregate scores. K20 must
also contain one valid execution audit per scene. Missing/failed scenes stop
the queue and are never discarded from the comparison.

Prior native result directories are discovered from the completed experiment
layout: UrbanDriver's `raw_8gpu_*`, PlanTF's `plantf_native_hard_NR_*` and
`plantf_last_hard_NR_*`, native-score queue `raw/` or its summary's `raw_dir`, and
PLUTO queue's `pluto_wo_refine/raw`. A discovered failed result is rejected;
there is no silent fallback to older results. If a source lives elsewhere,
pass `--sources /absolute/source_map.json`, with only the needed overrides:

```json
{"plantf_epoch17": "/absolute/completed/epoch17/full",
 "plantf_epoch24": "/absolute/completed/last/full"}
```

Inspect without launching simulations:

```bash
/home/data/wyk/envs/urbandriver/bin/python scripts/evaluate_random_baselines_1025.py prepare \
  --output "$QUEUE_ROOT"
/home/data/wyk/envs/urbandriver/bin/python scripts/evaluate_random_baselines_1025.py status \
  --output "$QUEUE_ROOT"
```

If an interrupted simulation left an incomplete output, inspect its log, then
rerun the same shell command with `--retry-failed`. That preserves each failed
directory under a `.failed.<timestamp>` sibling and retries it. Completed,
validated stages are reused. The original hard_NR and DP outputs are read-only.

## Outputs

- `plan.json`: exact random cohort, native source results, settings, inputs and hashes.
- `status.json`, `FAILED.json` or `COMPLETE.json`: explicit queue state.
- `<variant>.preflight.log`, `<variant>.worker.log`: environment and stage logs.
- `<variant>/native`, `<variant>/S10G_K20`: simulation, metric and NuBoard outputs.
- `<variant>/paired_scores.csv`, `comparison.json`: paired individual scores.
- `summary.csv`, `summary.json`: native / K20 / delta / improved / worsened / tied.

Local tests cover cohort identity, count independence, native selection, separate
PlanTF/PLUTO imports, GPU reservation, all 12 configurations, paired result
validation, execution audits and preservation when retrying. Actual CUDA, models,
maps and databases are checked on the server; these tests do not claim a completed
nuPlan evaluation.
