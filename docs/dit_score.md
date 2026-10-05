# DP-aligned DiT score branch

`model.architecture="dit_score_function"` adds an ego-only, direct-score DiT to
the existing training and refinement pipeline. The default is 80 temporal tokens,
width 192, six heads, three blocks, MLP ratio 4, attention dropout 0.1, and a
512-wide input projection MLP. Inputs and scores both have shape `[B,80,4]`.

## Alignment with Diffusion Planner

The reference is official DP commit
[`a3a621f`](https://github.com/ZhengYinan-AIR/Diffusion-Planner/tree/a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df),
specifically `model/module/dit.py`, `model/module/decoder.py`, and the decoder
initializer in `model/diffusion_planner.py` under `diffusion_planner/`.

| Component | Score DiT implementation |
|---|---|
| Input projection | Same two-linear MLP and GELU, `4 → 512 → 192` per time point |
| DiT blocks | Same self-attention → MLP1 → scene cross-attention → MLP2 order |
| Conditioning | Same route-conditioned adaLN shift, scale and residual gates |
| Attention and norms | Same batch-first PyTorch MHA, affine LayerNorm and dropout |
| Block MLPs | Same 4× expansion and approximate-tanh GELU, MLP dropout 0 |
| Cross-attention / MLP2 | Same **non-residual** assignments as official DP |
| Output head | Same adaptive normalization and two-linear projection structure |
| Initialization | Xavier linear weights, zero linear biases, unit LayerNorm weights; zero block/final modulation and zero final output linear |
| Frozen features | Existing DP scene encoder and route encoder, same cache/online-augmentation path |

`score_function/model/module/dit.py` is a small PyTorch adaptation of those
shared components, retaining their parameter names. Its MLP implements only the
dense, zero-dropout timm path used by DP. This keeps cached-feature training
independent of planner/timm/torchvision imports and avoids silently depending on
a server's modified DP checkout. Component parity tests strictly transfer state
dictionaries to the actual official module with real `timm==1.0.10`, then compare
forward outputs, input gradients and parameter gradients, including training
dropout and attention masks. CI pins the reference commit above.

The task-specific differences are explicit:

- DP packs each agent's current state and future into one token. Score DiT uses
  one token per ego future point, enabling attention across all 80 future times.
  A learned temporal embedding replaces the agent-type embedding; it uses the
  same `Normal(0,0.02)` embedding initialization. There is no current-state token
  or neighbor-future input; the existing frozen context supplies scene history.
- The adaLN condition is the frozen route embedding, without a diffusion-time
  embedding. Sigma is fixed for a run and bound to the checkpoint.
- The final head emits four score coordinates per token. It does **not** apply
  DP's diffusion-SDE standard-deviation division, predict a clean trajectory,
  predict noise, or invoke a diffusion sampler. There is no pretrained decoder
  weight import; this score branch is initialized and trained from scratch.
- A context projection is an identity at the default width 192; a linear
  projection allows narrower test/custom models.

Training still minimizes `mean((sigma * score + epsilon)**2)` for
`noisy = clean + sigma * epsilon`. Refinement still applies
`x += gamma * sigma**2 * score`, with the existing optional heading projection.
The initial score is exactly zero, matching DP's zero output initialization.
Initially, upstream gradients are zero; they become active as the output head
and gates learn. This is expected and covered by backward and DDP checks.

The temporal-convolution score and energy models remain available. This first
DiT experiment supports only `parameterization="score"` and no neighbor-future
conditioning. Old model settings and checkpoints are not converted into DiT
weights. The different tokenization also means a full DP decoder checkpoint is
not a score-model checkpoint, even though shared block tensors are compatible.

## Start from the current experiment configuration

Keep any running baseline on its original checkout. Create a separate worktree
for the new code; production resume checks bind checkpoints to source hashes.
For a local clone with this remote branch:

```bash
git fetch origin codex/dit-score-branch
git worktree add ../Score_Function-DiT -b experiment/dit-score origin/codex/dit-score-branch
cd ../Score_Function-DiT
```

Activate the existing training environment, and replace the paths below with
the workspace root and the JSON used by the current direct-score experiment:

```bash
ROOT=/path/to/workspace
CURRENT_CONFIG=/path/to/current-aligned-score.json
DIT_CONFIG="$ROOT/configs/score_dit.json"

python scripts/prepare_dit_config.py \
  --source-config "$CURRENT_CONFIG" \
  --output "$DIT_CONFIG" \
  --root "$ROOT" \
  --run-dir outputs/score_function_dit_aligned

python -m score_function check-config --config "$DIT_CONFIG" --root "$ROOT"

bash scripts/train.sh --gpus 0,1,2,3,4,5,6,7 \
  --config "$DIT_CONFIG" --root "$ROOT"
```

Choose the GPUs available for this run. Global batch must remain divisible by
`GPU count * microbatch_size`. The generator preserves **all** training, data,
sigma, seed, augmentation, refinement and evaluation settings from the source.
It also retains shared model dimensions and dropout, installs the DiT-specific
settings, and assigns a fresh run directory. It refuses to overwrite a config
or reuse an existing run directory. Use the same `--root` as the source run so
relative dataset/cache/planner paths still resolve to the same artifacts.

The first run starts without `--resume`; subsequent resumes require the same
DiT configuration, source checkout, data and GPU count. Existing scene/route
caches can be reused. No data preparation or simulation is launched by the
config generator or training wrapper. Online augmentation continues to use the
existing raw-feature and frozen-planner dependencies when enabled.

`configs/score_function_dit.json` is also supplied as a standalone template based
on the repository's aligned defaults (`sigma=0.05`). Deriving from the actual
server config is preferable when paths or training settings have been tuned.
If DiT needs a smaller microbatch, adjust it explicitly while preserving the
global batch and its divisibility constraint; no GPU memory fit is assumed.

## Checkpoints and evaluation

Use this run's `score/best.pt` and its matching DiT JSON with the existing
`evaluate`, `evaluate-planner`, `visualize-refinement`, and nuPlan integration.
EMA selection, deterministic validation noise, checkpoint provenance, score
coordinates and refinement gamma/steps are unchanged. Loading a convolutional
checkpoint as DiT, an energy checkpoint as score, or a mismatched sigma is rejected.

For example, the existing long-refinement diagnostic is:

```bash
python -m score_function visualize-refinement \
  --config "$DIT_CONFIG" --root "$ROOT" \
  --checkpoint "$ROOT/outputs/score_function_dit_aligned/score/best.pt" \
  --steps 5000 --max-samples 8
```

## Engineering verification

```bash
python -m unittest discover -s tests -v
SCORE_FUNCTION_OFFICIAL_ROOT=/path/to/pinned/Diffusion-Planner \
  python -m unittest discover -s tests -p test_dit_score.py -v
```

The first command exercises shape/conditioning contracts, finite gradients,
refinement arithmetic, config migration, training/validation/EMA selection,
checkpoint guards, and exact single-process and two-rank Gloo resume. The second
adds comparisons with the actual official source and needs timm/torchvision in
addition to the CPU test dependencies. These synthetic engineering checks make
no claim about nuPlan quality, GPU throughput, or convergence of a trained score.
The scientific comparison uses the existing real-data experiment and evaluation
protocol; a separate toy study is not required by this implementation.
