# PLUTO w/o refine.* followed by S10-G K20

This experiment implements the no-rule-postprocessing PLUTO baseline used in
Diffusion Planner's comparison. The asterisk means that pre-searched reference
lines remain model inputs. Use the pretrained model; no retraining is needed.
`w/o refine.` refers to PLUTO's original rule postprocessor, not to our S10-G
refinement. The two compared runs are:

- `PLUTO w/o refine.*`: official learned trajectory selection, K=0.
- `PLUTO w/o refine.* + S10-G`: the same native path followed by K=20.

## Implementation

Official PLUTO source is pinned to
`b9964b649c660f1f4a971d614c66f5992e24c18a`. The model and feature configuration
come from its `config/planner/pluto_planner.yaml`. Its eval-mode
`output_trajectory` selects the highest learned probability over valid reference
lines and longitudinal modes; when no reference line exists, the official model
uses its learned reference-free output. The adapter calls this path directly.
It does not invoke the rule evaluator, candidate simulation, emergency brake,
or the hybrid planner's hand-weighted extra proposal.

The original feature builder, ScenarioManager with a 60m radius, route updates,
and reference lines are retained. The model is constructed and strictly loaded
inside the simulation worker after Ray assigns its GPU. K0 returns the exact
native trajectory object without running the score.

PLUTO predicts 80 rear-axle-relative poses at 0.1s intervals over 8s. Both K0 and
K20 use PLUTO's original `global_trajectory_to_states(include_ego_state=False)`:
future states occupy t=0.1..8s, with the first predicted state copied to t=0.
Preserving that wrapper on both sides avoids confounding the score update with
a change to the controller's initial reference. The shared UrbanDriver adapter
retains its previous conversion and 16-pose audit default.

The existing frozen S10-G implementation, DP conditioning encoders and
normalizers are reused: EMA epoch 529, sigma=0.1, gamma=0.1, K=20, heading
projection. Additional score work preserves Python, NumPy and Torch RNG state.
There is no DP trajectory generation.

## Checkpoint provenance

On 2026-10-25 the original PLUTO README OneDrive checkpoint link returned 404.
The ForSim authors publicly link a PLUTO pretrained checkpoint folder in their
README. We downloaded `pluto_1M_aux_cil.ckpt` from that folder:

- Source: https://github.com/CurryChen77/ForSim#data-and-ckpt
- Folder: https://drive.google.com/drive/folders/18bXqs8nT1gr3-xnTUlxGN2jxnQV5BqFQ
- Direct endpoint: https://drive.usercontent.google.com/download?id=1rI_we4zkk3Jk6wr7IzF-nMBnRMJsuOGT&export=download&confirm=t
- Size: 51,431,136 bytes.
- SHA256: `ce60d7c854d10b310c4e1705099d799a6ea008baa4288e668626f4a2f22943e1`.

The inspected Lightning file contains 438 state-dict entries, epoch 24,
global_step 179224, a 25-epoch training configuration and contrastive loss.
This is a public secondary copy of the pretrained checkpoint, not a ForSim
training experiment being run here. The unavailable original OneDrive file
could not be compared byte-for-byte. The downloader pins the inspected bytes,
rejects HTML and incorrect hashes, and preserves any different existing file.

## Setup and launch on the evaluation node

This uses the existing dataset, maps, frozen score assets and working
`urbandriver` environment under `/home/data/wyk`. Setup clones that environment
into a separate `envs/pluto` before installing PLUTO's Torch 2.0.1/cu118 and
NATTEN 0.14.6 dependencies. It checks a real NATTEN CUDA operation. Existing
baseline environments are not edited.

```bash
cd /home/data/wyk
git clone --single-branch --branch codex/pluto-score-k20-1025 \
  https://github.com/ecowyk/Score_Function.git Score_Function_PLUTO_1025
cd /home/data/wyk/Score_Function_PLUTO_1025
bash scripts/setup_pluto_eval.sh
```

The setup downloads and verifies the checkpoint automatically. If the company
network blocks the direct endpoint, run the downloader on a connected machine
and copy the resulting file to
`/home/data/wyk/pluto/checkpoints/pluto_1M_aux_cil.ckpt`, then rerun setup. It will
verify an existing correct file without downloading it again.

```bash
cd /home/data/wyk/Score_Function_PLUTO_1025
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export QUEUE_ROOT="/home/data/wyk/dataset/nuplan/exp/pluto_score_queue_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$QUEUE_ROOT"
nohup bash scripts/sim_pluto_score_queue.sh \
  > "$QUEUE_ROOT/queue.log" 2>&1 < /dev/null &
echo "$!" > "$QUEUE_ROOT/queue.pid"
tail -f "$QUEUE_ROOT/queue.log"
```

The launcher uses `envs/pluto/bin/python` explicitly, so activating that
environment in the terminal is unnecessary. `PLUTO_CODE`, `PLUTO_CHECKPOINT`
and `PYTHON` can override its defaults. If setup used a custom `PLUTO_ENV`, set
`PYTHON` to that environment's `bin/python` for launch.

## Evaluation and outputs

The completed UrbanDriver hard14 NR result supplies the exact 272 scenarios,
seed, controller, observation and metric settings. `BASELINE_RESULT` can select
an explicit reference; otherwise the latest `raw_8gpu_*` reference is used on
the first run and retained in `protocol.json` for resume.

The queue first checks one fixed cohort scenario at K0 and K20, including
official aggregation and the K20 audit. Then it runs all 272 native scenarios,
followed automatically by all 272 K20 scenarios. Full evaluations reserve one
GPU and two CPUs per scenario across eight GPUs. Both the PLUTO model and the
score use CUDA. It preserves the official nonreactive challenge in metric file
paths so nuPlan's official aggregator finds the metrics.

Results are under `QUEUE_ROOT/pluto_wo_refine/`:

- `raw/`, `S10G_K20/`: full official simulation outputs.
- `K0_smoke/`, `K20_smoke/`: single-scenario integration outputs.
- `paired_scores.csv`, `comparison.json`: paired scores and improvement counts.
- `QUEUE_ROOT/summary.json`: final 0..100 scores and delta.
- `QUEUE_ROOT/protocol.json`: source, model, cohort and checkpoint provenance.

`ALL COMPLETE` marks completion. Resume uses the same `QUEUE_ROOT` and validates
completed stages; incomplete directories are preserved and rejected. To inspect
the latest queue from a new terminal:

```bash
RUN=$(cat /home/data/wyk/dataset/nuplan/exp/pluto_score_queue_latest.txt)
cat "$RUN/summary.json"
tail -n 40 "$RUN/queue.log"
```

## Verification limits

Local regression checks exercise real nuPlan and official PLUTO trajectory
conversion, native K0 identity, K20 updates/audits and RNG preservation. Queue
checks use synthetic completed results to validate order, GPU settings, cohort
checks, resume and rejection of incomplete outputs. These are not full nuPlan
benchmark runs. Real checkpoint strict loading, NATTEN execution and scene
feature inference are checked on the server before the full evaluations.
The paper's reported 70.03 is a comparison value, not an asserted reproduction.

Protocol sources:

- https://github.com/jchengai/pluto
- https://github.com/ZhengYinan-AIR/Diffusion-Planner
- DP Appendix C.4 and Table 1: https://proceedings.iclr.cc/paper_files/paper/2025/file/5c0cd43063f087eb32e572d040a19cd2-Paper-Conference.pdf
