# Official Flow Planner followed by frozen S10-G K20

Run the released Flow Planner checkpoint without retraining, then apply the
existing score refiner to its completed trajectory. This is a learned
flow-matching baseline. Its native flow integration remains **four midpoint
steps**; **K20 means twenty subsequent S10-G updates**, not twenty flow steps.

## Native model and published assets

Official source: https://github.com/DiffusionAD/Flow-Planner

Pinned commit: `366df7654d7b7fa77f9d2577279da63127c6d2f8`.

The official README links https://huggingface.co/ttwhy/flow-planner. Both files
are downloaded from immutable revision
`b627934a8db3021892afd13f5278ba0149532efc`:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `model.pth` | 57244066 | `9231b4c02968c9e42dd2881ad3e7dd4bd362796f7b2c0158e0458c4ab3bde88e` |
| `model_config.yaml` | 7036 | `2914c4f174e7fba8baf9ee7d42300c640a285f9d6fa13d36b78f0a3cbb9a03fe` |

The downloader uses HTTPS directly, verifies length and SHA256, rejects HTML,
and reuses an existing matching file. It preserves a mismatched existing file
and reports an error. Each asset gets a `.source.json` provenance record.

The released weights contain 338 `module.*` tensors in a bare state dictionary.
The official planner therefore receives `enable_ema=False` to read this format;
the file does not establish whether the authors exported EMA or raw training
weights. Every model parameter is loaded strictly. The adapter does not modify
the official source or the downloaded config.

The published YAML refers to two missing training dataset fields. A temporary
inference config fills them using the pinned official
`flow_planner/script/data/dataset/nuplan_data.yaml`: downsampling `uniform` and
predicted neighbor count 0 (the model's `neighbor_pred_num`). Only model/core
subtrees are resolved; training recorder and environment references are unused.
The protocol records these defaults and a digest of the effective config.

Native behavior follows the released YAML: `x_start`, four midpoint steps,
temperature 1, CFG weight 1, average assembly, 80 rear-axle-relative poses over
8 seconds. Keep the released CFG value 1 rather than the repository training
default 1.8. Official preprocessing, normalization, model, inference core and
nuPlan conversion are retained, with no extra rule postprocessor.

## Refinement and comparison

Models are constructed inside simulation workers after GPU assignment. K0
returns the same native trajectory object. K20 refines the completed physical
trajectory with frozen S10-G EMA epoch 529, sigma 0.1, gamma 0.1 and heading
projection. Frozen DP conditioning encoders and normalizers are reused; no DP
trajectory is sampled. Each scenario initializes Python, NumPy and Torch RNGs
once from the inherited reference evaluation seed, before constructing the
native model. This makes raw/K20 sampling independent of Ray worker scheduling;
the seed is not reset per frame. Additional score loading and refinement
preserve RNG state, so they do not advance Flow's native sampling stream.

Both paths use the same nuPlan conversion, including the observed ego state at
t=0 and 80 future states at t=0.1..8 seconds. The completed UrbanDriver result
supplies the exact 272 hard14 nonreactive scenario identities, seed, controller
and official metric settings. Full evaluations run eight workers with one GPU
and two CPUs per simulation. Flow and the score both use CUDA.

## Setup on the evaluation node

This assumes the existing dataset, nuPlan devkit, frozen S10-G assets and working
Python 3.9 PLUTO environment under `/home/data/wyk`. Setup clones PLUTO into
`envs/flow_planner`, then installs official Torch 2.3.0/cu118, Hydra 1.3.2 and
pinned inference dependencies. The completed clone is reused on retry. Conda
may spend several minutes copying files without progress output.

Torch and torchvision use their exact official `download-r2.pytorch.org`
Python 3.9/Linux x86-64 cu118 wheels with upstream SHA256 checks. This skips
the `download.pytorch.org` index hostname that timed out on the evaluation
node; transitive dependencies use the configured pip index. Installed matching
cu118 versions are reused on resume. The final CUDA inference check still runs.
This changes the download route, not the Torch, torchvision or CUDA versions.

```bash
cd /home/data/wyk
git clone --single-branch --branch codex/flow-score-k20-1028 \
  https://github.com/ecowyk/Score_Function.git Score_Function_FLOW_1028
cd /home/data/wyk/Score_Function_FLOW_1028

PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:+$PIP_TRUSTED_HOST }mirrors.tools.huawei.com download.pytorch.org download-r2.pytorch.org" \
PIP_DEFAULT_TIMEOUT=120 \
bash scripts/setup_flow_eval.sh
```

The scoped pip host exceptions above match the already observed company proxy
issue. The Hugging Face downloader validates TLS separately. If that endpoint
requires the company's CA, pass its PEM path as `FLOW_CA_BUNDLE`; `PIP_CERT` is
also accepted. No global certificate settings are changed. Setup finishes only
after strict checkpoint loading and a full synthetic CUDA inference succeed.

Defaults can be overridden using `FLOW_BASE_ENV`, `FLOW_ENV`, `FLOW_CODE` and
`FLOW_ASSET_DIR`. For launch, use `PYTHON` to select a custom environment and
`FLOW_CHECKPOINT`/`FLOW_CONFIG` to select assets in a custom directory.

If the node cannot connect to either Hugging Face or HF-Mirror, the dated
`scripts/download_flow_cdn_1028.py` recovery helper downloads the identical model
directly from its official CDN and reconstructs the byte-identical configuration
from an embedded copy. It verifies the same sizes and SHA256 values before
publishing either file. This emergency URL expires at 2026-10-28 09:01:35 UTC;
it is not a permanent mirror and is not used by normal setup. Existing correct
assets remain reusable after expiry. The helper checks TLS normally and retries
with certificate verification disabled only after curl reports certificate
error 60; pinned file hashes remain mandatory.

Alternatively, copy the two unchanged files from a connected machine into
`/home/data/wyk/Flow-Planner/checkpoints`, then rerun setup. When both files match
the pinned hashes, setup skips the download entirely. Do not keep retrying
certificate options for a TCP connection timeout.

## Start native then K20 automatically

```bash
cd /home/data/wyk/Score_Function_FLOW_1028
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHON=/home/data/wyk/envs/flow_planner/bin/python
export BASELINE_RESULT=/home/data/wyk/dataset/nuplan/exp/exp/simulation/closed_loop_nonreactive_agents/urbandriver/test14-hard/raw_8gpu_20261017_130942
export QUEUE_ROOT="/home/data/wyk/dataset/nuplan/exp/flow_score_queue_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$QUEUE_ROOT"
nohup bash scripts/sim_flow_score_queue.sh \
  > "$QUEUE_ROOT/queue.log" 2>&1 < /dev/null &
echo "$!" > "$QUEUE_ROOT/queue.pid"
tail -f "$QUEUE_ROOT/queue.log"
```

The sequence is K0 single-scenario check, K20 single-scenario check, full native
272-scenario evaluation, and full K20 evaluation. The short checks use the first
GPU; the full evaluations use all eight. Official aggregation, exact successful
scenario identities, finite score updates and native80-to-score80 audits must
pass before a stage is accepted. The queue records source/config/weight hashes
and rejects changes between stages.

`ALL COMPLETE` marks successful completion. Outputs:

- `QUEUE_ROOT/flow_planner/raw/` and `S10G_K20/`: full simulations.
- `QUEUE_ROOT/flow_planner/K0_smoke/` and `K20_smoke/`: short checks.
- `QUEUE_ROOT/flow_planner/paired_scores.csv` and `comparison.json`: paired results.
- `QUEUE_ROOT/summary.json`: native and K20 scores on a 0..100 scale and delta.
- `QUEUE_ROOT/protocol.json`: cohort, seed, effective config and provenance.

From a new terminal:

```bash
RUN=$(cat /home/data/wyk/dataset/nuplan/exp/flow_score_queue_latest.txt)
cat "$RUN/summary.json"
tail -n 40 "$RUN/queue.log"
```

Reusing the same `QUEUE_ROOT` validates and reuses completed stages. Incomplete
stage directories are preserved and rejected; inspect their logs before
starting a fresh queue. A separate lock prevents overlapping Flow queues.

## Verification limits

Local checks use actual published weights and full synthetic CPU inference,
plus real nuPlan/Flow trajectory conversion, K0 identity, K20 count, audits and
RNG preservation. Queue tests validate stage order, eight-GPU configuration,
exact cohort, official aggregation paths and resume rejection. These are not
benchmark results. Setup checks CUDA on the server; the two single-scenario
runs check real map/features and frozen-score integration before full runs.
The paper's reported results are reference values, not guaranteed scores for
this cohort or environment.
