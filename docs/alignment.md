# DP training alignment and long-refinement diagnostics

This update addresses training budget, paired data augmentation, and fixed-scene
trajectory inspection. It retains the current temporal score/energy architecture,
fixed sigma, frozen official encoders, and ego-only DSM objective. It does not
establish that these models have converged or improve nuPlan closed-loop scores.

## Training budget

The reference is the official DP paper Appendix C.1 and repository at
`a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df`: one million training scenarios,
global batch 2048, 500 epochs, AdamW LR 5e-4, warmup five epochs, EMA 0.999.
The released training code uses AdamW's default weight decay 0.01 and clips
gradient norm at 5. Its scheduler is epoch-wise linear warmup followed by a
constant LR, despite its cosine-related function name.

New score and energy defaults use these optimizer settings. The budget is:

```text
updates_per_epoch = floor(actual_training_frames / global_batch)
planned_epochs = max(500, ceil(244000 / updates_per_epoch))
planned_updates = planned_epochs * updates_per_epoch
sample_presentations = completed_updates * global_batch
```

Both epoch and update floors must be satisfied. The 244,000 reference updates
are `500 * floor(1000000 / 2048)`. Because our one-million selection includes
an internal recording-level validation split, 500 epochs alone usually supplies
fewer updates. The trainer reports the actual counts and any epoch extension.
The default has no early stop or validation-driven LR reduction. Validation
and best-checkpoint selection continue throughout; the best checkpoint may
come from an earlier epoch, and its selection metadata must be reported
separately from the completed training budget.

Global batch is `GPU_count * microbatch_per_rank * accumulation`. With eight
GPUs, microbatch 256 gives accumulation 1; microbatch 128 gives accumulation 2.
On one GPU, microbatch 1024 gives accumulation 2. Increasing microbatch alone
does not increase effective training exposure. Online encoder work is separately
chunked by `training.data_augmentation.encoding_batch_size` (default 64).

Older saved configs without the new schedule, budget and augmentation fields
keep the old behavior. Changing the repository default does not rewrite files
under an existing experiment's output directory.

## Reuse existing data and create a fresh training configuration

The raw NPZs and clean cache can be reused when their saved planner/normalizer
identity matches the installed official planner. Raw files must still exist at
their recorded paths. Every NPZ used by online training is checked against its
published SHA-256 while reading the same bytes; there is no separate full-data
startup scan. Missing feature paths can be resolved from the matching manifest.

Create a new configuration from an existing S05-L, E05-L or S05-LN config:

```bash
python scripts/prepare_aligned_config.py \
  --source-config /home/data/wyk/outputs/score_matrix_1m/configs/S05-L.json \
  --output /home/data/wyk/outputs/score_aligned/configs/S05-L.json \
  --run-dir /home/data/wyk/outputs/score_aligned/S05-L \
  --root /home/data/wyk --gpus 8 --microbatch 256

python -m score_function check-config \
  --config /home/data/wyk/outputs/score_aligned/configs/S05-L.json \
  --root /home/data/wyk
```

The source config must be the one belonging to the intended variant. The helper
preserves model architecture, score/energy parameterization, sigma, seeds, data
paths and data splits; it copies the new training protocol from the checked-out
default template. It refuses to overwrite the source config, an existing output
config, or an existing run directory. It does not load weights or start a job.

When server access is restored, explicitly start a fresh run with:

```bash
bash scripts/train.sh --gpus 0,1,2,3,4,5,6,7 \
  --config /home/data/wyk/outputs/score_aligned/configs/S05-L.json \
  --root /home/data/wyk
```

This starts from random score/energy weights. `--resume` is only for an interrupted
run of this same code, config, data and GPU count. Old `best.pt` files remain usable
for offline diagnosis, but old `last.pt` files cannot be resumed after changing
the code or training protocol. Keep an old checkout if that old run must resume.

## Paired data augmentation

The new training path executes the actual official `StatePerturbation` implementation:

1. Read unnormalized scene inputs, ego future and neighbor future from the NPZ.
2. Perturb the ego state with probability 0.5 where the official speed gate permits it.
   Reconstruct the initial expert future and transform scene/trajectories together.
3. Pack target heading as cosine/sine; apply official observation and state normalizers.
4. Recompute scene and route features using the frozen official modules in eval mode.
5. For neighbor-conditioned models, predict neighbor futures from this same augmented
   observation. A clean cached neighbor prediction is never paired with an augmented target.
6. Add fresh fixed-sigma DSM noise and update only the score/energy network.

`probability` denotes actual perturbation probability. The adapter accounts for
the upstream implementation's `rand >= augment_prob` comparison. At the official
value 0.5 these definitions coincide. A setting of zero keeps the online path
but does not select perturbations; setting `enabled=false` selects the legacy
cached-feature training path.

Validation remains on clean cached observations with reproducible corruption
noise. There is no validation augmentation or planner-gradient training.
Neighbor-conditioned validation still needs its matching clean neighbor cache.
Online encoding and especially neighbor sampling add real computation; previous
training wall times do not predict the runtime of this protocol.

This aligns preprocessing conventions and augmentation mechanics. It does not
make our selected train/validation token lists identical to the paper's training
set, nor change ego-only DSM into DP's joint ego-and-neighbor diffusion objective.

## Inspect existing checkpoints over 5,000 refinement steps

Use the checkpoint's original saved configuration, not a newly changed model
configuration. This command performs offline inspection only:

```bash
python -m score_function visualize-refinement \
  --config /path/to/original-config.json --root /home/data/wyk \
  --checkpoint /path/to/score/best.pt \
  --split val --max-samples 8 --steps 5000 \
  --snapshot-steps 0 5 20 100 500 1000 2000 5000 \
  --output /home/data/wyk/outputs/refinement_5000_val
```

Use `--split train` and a different output directory to inspect training frames.
The command uses a deterministic subset, never chooses cases by their result,
and by default inspects both `planner` and `expert_noise` initializations.
`--initializations expert_noise` permits inspecting the cached condition without
generating a DP candidate. Planner initialization needs the official checkpoint,
normalizers and raw NPZs. `--device cpu` is supported for diagnosis when those
dependencies are available, although thousands of energy evaluations can be slow.

Each trajectory is refined with fixed scene, route and neighbor conditions.
The update is the existing `gamma * sigma**2 * score`, followed by the configured
heading projection. `--gamma` overrides only the diagnostic step size. The
simulation defaults remain at five refinement steps.

Outputs include an HTML index, sparse normalized/physical trajectory snapshots,
map/trajectory overlays, physical-time motion plots, per-update CSV diagnostics,
and summary/provenance JSON. Full trajectories are retained only at requested
snapshot steps and the last finite step, rather than serializing every score
vector for all 5,000 iterations. Scalar traces include update and cumulative
displacement in meters, score norm, expert ADE/FDE, and heading projection effects.
Energy models additionally record actual energy values. Numerical divergence is
reported with the failed iteration and last finite trajectory; it is not silently
treated as convergence. Completing the iteration budget is not a convergence proof,
and a small update or decreasing energy is not a closed-loop quality guarantee.

## Verification scope

CPU tests cover budget extension, resume behavior, paired transformation order,
frozen planner gradients, neighbor-condition consistency, long-horizon analytic
score/energy behavior and bounded snapshot storage. Optional official-code tests
compare against upstream transformation/encoder implementations. These are code
checks, not training or nuPlan driving experiments. Real-data throughput, numerical
trajectory results and completed training budgets require the server and assets.
