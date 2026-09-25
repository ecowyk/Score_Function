# Eight-GPU Training Suite

Run eight independent experiments on eight GPUs. The launcher prepares one
shared dataset and frozen scene/route cache, then assigns one GPU to each model.
It does not launch offline sweeps or nuPlan simulations.

## Start

Requirements: Linux, eight CUDA GPUs with a compatible NVIDIA driver, Git,
Miniconda (or an existing Diffusion Planner Python environment), and extracted
nuPlan training databases and maps. Downloads require access to GitHub, PyPI,
the PyTorch wheel index and Hugging Face. No sudo is used.

```bash
git clone https://github.com/ecowyk/Score_Function.git
cd Score_Function
bash scripts/launch_experiments.sh \
  --root /path/to/workspace \
  --database-dir /path/to/nuplan/trainval \
  --maps-dir /path/to/nuplan/maps
```

Default tmux session: `score_matrix_1m_wyk`.
The launcher creates a Python 3.9 environment named `score_function_wyk`,
installs CUDA 11.8 PyTorch 2.0 and the pinned production dependencies, and
downloads missing official repositories and released checkpoint files.
Existing official repositories and checkpoint files are reused without replacement.
The system NVIDIA driver and nuPlan dataset are not installed or downloaded.
If tmux is missing, the launcher installs it into the base conda environment.

To reuse an already configured Diffusion Planner environment:

```bash
bash scripts/launch_experiments.sh \
  --root /path/to/workspace \
  --database-dir /path/to/nuplan/trainval \
  --maps-dir /path/to/nuplan/maps \
  --python /path/to/conda/envs/diffusion_planner/bin/python
```

This installs the project and official source packages in editable mode, without
replacing that environment's third-party dependencies. Preflight validates imports,
checkpoint loading, maps, training DB selection, and the configured model backward.

Useful path options:

- `--planner-dir`, `--devkit-dir`: existing official source directories.
- `--checkpoint`, `--planner-args`: existing released planner files.
- `--data-output`: shared NPZ/tensor cache location; use local SSD/NVMe if available.
- `--gpus 0,1,2,3,4,5,6,7`: exactly eight distinct physical GPU indices.
- `--run-name score_matrix_v2`: separate output and default data/cache directories.
- `--foreground`: run in the current terminal instead of creating tmux.
- `--dry-run`: print all resolved configurations without installing, writing or training.

Relative source/data paths in the base configuration resolve against `--root`.
Explicit CLI paths resolve against the current directory. The default repository
locations are `ROOT/Diffusion-Planner` and `ROOT/nuplan-devkit`.

## Protocol

| ID | Sigma | Ego temporal interaction | Predicted neighbors |
|---|---:|---|---|
| S02-L | 0.02 | RF37 TCN | No |
| S05-L | 0.05 | RF37 TCN | No |
| S10-L | 0.10 | RF37 TCN | No |
| S20-L | 0.20 | RF37 TCN | No |
| S05-W | 0.05 | RF77 TCN | No |
| S10-W | 0.10 | RF77 TCN | No |
| S05-G | 0.05 | RF37 TCN + temporal self-attention | No |
| S05-N | 0.05 | RF77 TCN | Yes |

L uses dilations `[1,2] -> [2,4]`; W/N use `[1,2] -> [4,12]`.
Each of the four residual blocks contains two kernel-3 convolutions.
L and W have identical parameter counts and depth. RF77 is a larger local window;
only G gives every ego point direct access to the whole ego horizon.
G uses one pre-norm temporal self-attention/FFN block after scene attention.
N encodes each complete predicted neighbor future into one token and cross-attends
to the ten neighbor tokens plus an always-valid null token.

All groups use width 192, six heads, dropout 0.1, FP32, effective batch 2048,
microbatch 128 and accumulation 16 on a single GPU. The default CUDA allocation
limit is 50% of each GPU (40 GiB on an 80 GiB A100).
This is an allocator limit, not a reservation or a guarantee about total GPU memory.
Preprocessing uses 16 CPU workers; each trainer uses two data-loader workers.

Training is fixed-sigma DSM on normalized expert ego futures only.
The frozen planner supplies scene/route features. No timestep enters the score
network; neither current ego pose nor neighbor futures are optimization targets.
AdamW uses LR 1e-4, weight decay 1e-4, gradient clipping 5, EMA 0.999,
and one epoch of linear warmup from 1e-5.
Every epoch evaluates eight fixed validation corruptions.
Three checks without a 0.5% significant EMA DSM improvement halve LR; eight checks
trigger early stopping after at least five epochs. Thirty epochs is the ceiling.
The selected checkpoint is the minimum validation DSM among the initial branch
and subsequent EMA branches. An initial selection remains explicitly labeled.

Data selection follows the official Diffusion Planner preprocessing script:
all locally available official training logs are candidate sources, then one
global nuPlan builder call selects at most **1,000,000 scenarios** with
`shuffle=true`, `expand_scenarios=true`, `remove_invalid_goals=false`,
`timestamp_spacing_s=null` and no per-DB limit.
The cap includes both the training and internal validation partitions.
The selection seed and per-log token lists are frozen before feature extraction.
The final number of NPZs can be smaller if selected targets are incomplete/nonfinite;
rejected samples are not replaced with extra draws.
Override the total explicitly with `--total-scenarios N`.
Validation holds out 5% of recording groups; official validation/test logs do not enter training.
Incomplete final global batches are dropped as in the existing trainer.
The selected DB list and recording split are frozen at preprocessing start:
finish extraction first; adding DBs later requires a new data/output directory.

## Execution and artifacts

1. Bootstrap dependencies, sources and released planner checkpoint.
2. Validate all eight GPU/model forward/backward configurations.
3. Select and freeze the global scenario set, then extract only those NPZ features
   in parallel. Selection still scans candidate DB metadata; the cap saves the
   expensive feature extraction, storage and subsequent encoding for unselected frames.
4. Encode shared scene/route features with eight-GPU distributed caching.
5. Start seven trainers immediately. On GPU 7, prepare predicted-neighbor cache,
   then start S05-N. Thus N starts later; it does not delay the other seven.
6. Write a suite summary after all eight jobs exit.

Neighbor caching calls the official frozen decoder, reusing cached scene features.
Each fixed batch has a reproducible seed derived from its frame tokens. Batch size
16, sample membership, seed, source hashes and shared-cache identity are recorded.
No expert neighbor future is read. Neighbor padding is masked; absent-agent outputs
cannot affect the model. Normalization uses the official neighbor statistics.
The extra cache stores only neighbor futures/masks and reuses the original
scene/route/target arrays. It supports resuming completed shards.
At inference the N branch conditions on neighbors from the same joint planner
sample as the ego candidate; it does not generate another sample.

Planner-predicted neighbors paired with expert ego are an auxiliary prediction
feature, not guaranteed behaviorally matched joint demonstrations.

```text
ROOT/outputs/score_matrix_1m/
├── bootstrap.log
├── environment.txt
├── suite.json                 # frozen eight configurations and source hashes
├── status.json                # suite and per-experiment stages/failures
├── summary.json               # published once all workers finish
├── configs/S05-L.json         # standalone configs for later evaluation
└── S05-L/
    ├── train.log
    └── score/
        ├── best.pt
        ├── last.pt
        ├── history.json
        ├── selection.json
        ├── model_info.json
        └── status.json
```

Other experiment directories have the same layout. N also has
`S05-N/cache-neighbors.log`; shared preparation logs are under S02-L.
Do not rank different sigma models by raw DSM alone.

## Monitor and resume

### Migrate from the previous unlimited preprocessing run

Stop the old pipeline **before** updating its code. For the original default session:

```bash
tmux send-keys -t score_matrix_v1_wyk:0.0 C-c
```

Wait for its workers to exit (GPU/CPU jobs from that pipeline should stop), then
update the repository and start the new default `score_matrix_1m` run:

```bash
git pull --ff-only
bash scripts/launch_experiments.sh \
  --root "$ROOT" --database-dir "$DB" --maps-dir "$MAPS" \
  --reuse-features-from "$ROOT/score_data/score_matrix_v1" \
  --python /path/to/existing/environment/bin/python
```

If the old run used a custom `--data-output`, pass that directory to
`--reuse-features-from` instead. The new selection is made independently of
which samples finished earlier. Only selected tokens with matching DB/frame metadata
and intact NPZ checksums are reused; missing/corrupt samples are recomputed in the
new directory. Old validation labels are replaced with the current recording split.
Reused NPZs are referenced in place: retain the old features directory.
Old tensor caches and model checkpoints are not reused across this data-protocol change.
Do not pass `--resume` when switching from the old unlimited run to this new run.

### Continue an interrupted run under the new protocol

```bash
tmux attach -t score_matrix_1m_wyk
tail -f /path/to/workspace/outputs/score_matrix_1m/S05-L/train.log
cat /path/to/workspace/outputs/score_matrix_1m/status.json
```

Detach with Ctrl-b then d; closing the local computer does not terminate tmux.
After a failed/completed tmux pane, remove that inactive session before resuming.
Do not kill an active session merely to start a duplicate launcher.

Resume using exactly the original options plus `--resume`:

```bash
bash scripts/launch_experiments.sh \
  --root /path/to/workspace \
  --database-dir /path/to/nuplan/trainval \
  --maps-dir /path/to/nuplan/maps \
  --resume
```

Completed groups are retained; interrupted groups restore model, optimizer,
EMA, RNG and data cursor from `last.pt`. Raw preprocessing and caches resume
completed work. Source/configuration/data changes are rejected instead of silently
changing an experiment. A job that failed before its first checkpoint needs a
new run directory. Failures are logged and produce a nonzero suite exit status;
other independent trainers can finish.

The launcher's locks prevent duplicate suite runs. Use exclusive GPU allocations
from your cluster scheduler; the script does not evict other users' processes.
No offline or closed-loop evaluation is launched by this script.
