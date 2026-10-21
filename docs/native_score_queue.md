# PDM-Open / GC-PGP: native baseline followed by S10-G K20

This queue uses the completed UrbanDriver hard14 NR run as the authoritative
272-scenario cohort and evaluation configuration. It takes each native planner
configuration from its successful two-scenario smoke result. It executes:

1. PDM-Open native (or reuses its completed, matching result).
2. PDM-Open K20 one-scenario integration check, then all 272 K20 scenarios.
3. GC-PGP native (or reuses its completed, matching result).
4. GC-PGP K20 one-scenario integration check, then all 272 K20 scenarios.

It uses the existing S10-G final EMA at epoch 529, sigma=0.1, gamma=0.1,
20 updates and heading projection. No training or DP trajectory sampling occurs.
The queue uses eight visible GPUs, with one GPU reserved per simulation and
eight scenarios running concurrently. GC-PGP native inference, score inference
and frozen DP conditioning run on GPU. PDM-Open retains the official CPU native
implementation, while its score and conditioning run on GPU. This queue is
intended for the user's separate eight-GPU evaluation node.

## Server launch

Use a separate checkout of `codex/pdm-gcpgp-score-k20-1021` at
`/home/data/wyk/Score_Function_NativeScore_1021`.

```bash
conda activate /home/data/wyk/envs/urbandriver
cd /home/data/wyk/Score_Function_NativeScore_1021
python -m pip install --no-deps -r requirements_urban_driver_score.txt
export QUEUE_ROOT=/home/data/wyk/dataset/nuplan/exp/native_score_queue_$(date +%Y%m%d_%H%M%S)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p "$QUEUE_ROOT"
nohup bash scripts/sim_native_score_queue.sh > "$QUEUE_ROOT/queue.log" 2>&1 &
echo "$!" > "$QUEUE_ROOT/queue.pid"
tail -f "$QUEUE_ROOT/queue.log"
```

By default this starts the four stages on the evaluation node. To explicitly
reuse an earlier native-only suite, set `BASELINE_FULL_ROOT` to its directory.
The launcher checks the old PID's inherited suite path before waiting for a
queue on the same host to finish. Completed PDM-Open results may be reused if all
272 scenario IDs, model and evaluation settings agree. A completed CPU GC-PGP
result is preserved, but a fresh GPU-native baseline is run to match GPU K20.
Failed or incomplete results are not accepted as completed and are not
overwritten. There is no automatic fallback to an older successful result when
the selected one failed. When using a different host with shared storage, only
explicitly select an old suite that has finished on its host.

The default UrbanDriver reference is the latest `raw_8gpu_*` in
`.../exp/simulation/closed_loop_nonreactive_agents/urbandriver/test14-hard/`.
Override it with `BASELINE_RESULT=/absolute/path/to/result` when needed.
`FROZEN_SCORE_CODE`, `SCORE_CONFIG` and `SCORE_CHECKPOINT` retain the defaults from
the existing UrbanDriver launcher under `score_G_transfer_1014_v1`.

## Adapter and checks

The native planner is built through nuPlan's official `build_planners`, in the
simulation worker after Ray assigns its GPU. The native baseline uses a lazy K0
wrapper which returns the identical native trajectory without score loading or
resampling. This avoids constructing 272 GPU models in the driver. PDM-Open uses
its original CPU implementation; GC-PGP uses the original CUDA MLPlanner. Its
native sampling and candidate selection are unchanged. Finished
16-pose/8-second trajectories use nuPlan interpolation on the same 80-point grid
and rear-axle frame as the UrbanDriver score integration. The existing frozen
score/refinement implementation and DP normalizers are reused.

The score work preserves Python, NumPy and Torch RNG state. K0 returns the exact
native trajectory object without resampling or running the score. K20 audits
record epoch, sigma, completed steps and finite output for each scenario.
Checkpoint/config hashes are checked before and after evaluation.

Each new run relocates all nested metric and callback output paths. Old outputs
remain untouched. Failed scenarios, missing metrics, mismatched cohorts and
invalid K20 audits stop the queue. Completed stages can be resumed with the same
`QUEUE_ROOT`; incomplete output directories require inspection and a fresh queue
directory rather than being silently overwritten.

## Outputs

- `queue.log`: progress and errors from the whole queue.
- `protocol.json`: reference cohort, seed, inference devices and checkpoint hashes.
- `pdm_open/comparison.json`, `gc_pgp/comparison.json`: native score, K20 score,
  delta and per-scenario improvement counts on the 0..100 scale.
- Each model's `paired_scores.csv`: one-to-one scenario comparisons.
- `summary.json`: completed model comparisons, updated after each pair.
- Each new simulation's `code/hydra/config.yaml`, `console.log` and metric files.

Dataset-free checks exercise real nuPlan trajectory conversion, the exact K20
update, K0 identity, device selection and RNG isolation. Synthetic result tests
exercise queue scheduling, reuse and output integrity. These are not a completed
nuPlan benchmark: actual checkpoints, maps and databases are checked by the
server preflight and automatic K20 integration run.
