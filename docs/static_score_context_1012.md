# Final S05-G static trajectory inspection

The two new scripts read the frozen `score_full_k_1009_v1` experiment and selected
saved DP_K0 simulation logs. They do not launch a new nuPlan simulation or alter
the active evaluation queue. Run with the existing diffusion_planner Python.

```bash
cd /home/data/wyk/Score_Function_aligned_v1
SCORE_PY=/home/data/wyk/envs/diffusion_planner/bin/python

# First check one frame and inspect its two-page PDF.
"$SCORE_PY" -u scripts/static_score_context.py run --cohort I01 --frames 0

# Then six priority scenarios, three fixed temporal probes per scenario.
# The completed frame above is reused.
"$SCORE_PY" -u scripts/static_score_context.py run --cohort priority
```

Output root:
`/home/data/wyk/dataset/nuplan/exp/score_G_static_context_1012_v1`

- `static_trajectories.pdf`: combined report for the requested cases.
- `figures/<case>/frame_<index>/`: two PNGs and a two-page vector PDF.
- `captures/<case>/frame_<index>/payload.json`: physical trajectories, environment,
  provenance, replay error and exact frame identity.
- `inputs.npz`: normalized planner inputs, DP start and fixed scene/route features.

Page one uses six separate panels for K=5,20,50,100,200,500. Each compares a thin
dashed DP trajectory with one refined trajectory. Every future sample has a
small point. Axes share the same scale, in metres. Page two shows a common local
zoom for K20/50/200 and signed displacement along/across the DP path. The zoom is
chosen near the largest displacement among these three K values; it is not a
detected collision or metric failure. Displacement arrows are not magnified.

Road/lane polygons, lane centerlines, route, crosswalks, present vehicles and
pedestrians, current ego footprint and traffic-light connector colors are drawn
in the same rear-axle coordinate frame. Actors are at the capture time, not at
each future trajectory time: overlap with a present actor box alone does not
establish a future collision.

## Comparison protocol

All K values at a given frame share the same recorded DP_K0 closed-loop history,
normalized inputs, scene/route features and one seeded DP prediction. Only final
G EMA weights are used. These are **counterfactual refinement endpoints at a
common input**, not the actual later trajectories from separate K closed-loop
runs. The original timestamp-based seed rule is used unchanged.

The default probe indices are start, one-third and two-thirds of saved baseline
history. These are temporal probes, not detected event frames. To inspect a
specific history index, use `--frames 40 50 60` with one case.

The script first verifies frozen input hashes, then compares reconstructed DP
K0 against the saved baseline trajectory. Maximum XY error above 0.02 m stops
capture; it is not silently accepted. CPU and original CUDA inference may differ.
If that check fails, inspect the printed error; an explicit single idle GPU can
be used with a separate output directory:

```bash
# Replace N with an idle physical GPU index.
"$SCORE_PY" -u scripts/static_score_context.py run --cohort I01 --frames 0 \
  --device cuda --gpu N \
  --output /home/data/wyk/dataset/nuplan/exp/score_G_static_context_1012_cuda_v1
```

CPU is the default, with two threads and no visible GPUs. The script sets DP,
nuPlan devkit, data/maps/exp roots, frozen-code import precedence and the Agg
plot backend internally. It reads the user's trusted pickle-based simulation
logs with Torch storage restored onto CPU.

## Selected cases

Priority six:

| ID | Token | Reason to inspect from aggregate metrics |
|---|---|---|
| I01 | cf87d58064425ee7 | G improves collision/TTC metrics |
| W01 | 5d392fa38ff65c3f | Drivable metric fails at G20/50, recovers at G200 |
| K01 | 6d0875dbcec45b3a | Drivable metric fails only at G50 |
| K02 | c2c649fcd5325b3d | Collision metric fails only at G50 |
| K04 | c6277e74a4f054a7 | Comfort metric fails only at G50 |
| W04 | 54da00bc66c7575d | TTC degrades near pedestrian crossing |

`--cohort all` includes the full selected 16 cases, including four controls.
These cases are selected for diagnosis, not unbiased aggregate performance
estimation. The score/metric changes motivate inspection; they do not prove
the causal mechanism visible in a particular frame.

To regenerate figures from captures, without any planner inference:

```bash
"$SCORE_PY" -u scripts/static_score_context.py render --cohort priority
```

Local checks cover coordinate transforms, recorded-history reconstruction,
trusted GPU-tagged pickle restoration to CPU, replay timestamp truncation,
refinement chunking and PDF layout using synthetic geometry. Actual nuPlan
capture requires the server's data, frozen code and model weights.
