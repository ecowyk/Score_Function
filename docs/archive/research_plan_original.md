# Diffusion Planner Clean-Space Score Refinement
## Research & Implementation Specification for Codex

**Goal:** extend the official Diffusion Planner with a lightweight, time-independent, scene-conditioned score branch that learns a local density gradient in clean ego-trajectory space and uses it to refine the ego trajectory produced by the original planner.

**Repository baseline:** official `ZhengYinan-AIR/Diffusion-Planner` codebase, current `main` structure as of 2026-09-24.

---

## 0. Executive Summary

We do **not** want to reuse Diffusion Planner's diffusion-time-dependent score, nor convert its default `x_start` prediction into a score. Previous experiments indicate that near the clean endpoint this conversion becomes numerically/relatively unstable and does not reliably provide a useful optimization direction.

The new target is instead a **clean-space conditional trajectory score**:

$$
s_\phi(\tau, C, R) \approx \nabla_\tau \log p_\sigma(\tau \mid C,R),
$$

where:

- $\tau$ is the **ego future trajectory** only;
- $C$ is the frozen scene representation from the original Diffusion Planner encoder;
- $R$ is the route condition from the original route encoder;
- $p_\sigma$ is a slightly Gaussian-smoothed clean trajectory distribution with a **fixed** $\sigma$;
- the score branch has **no diffusion timestep input**;
- $\sigma$ is a training hyperparameter, not an input to the network.

The original planner remains responsible for global trajectory generation:

$$
\epsilon \xrightarrow{\text{Diffusion Planner}} \tau_{\text{DP}}.
$$

The new branch performs lightweight local refinement:

$$
\tau^{(k+1)} = \tau^{(k)} + \gamma \sigma^2 s_\phi(\tau^{(k)}, C,R),
$$

starting from $\tau^{(0)}=\tau_{\text{DP}}$.

The primary architecture is:

```text
Frozen Diffusion Planner Scene Encoder ──────────────┐
                                                      │ scene context C
Frozen Diffusion Planner Route Encoder ───────┐       │
                                               │ R     │
Candidate ego trajectory τ                     │       │
[B, T, 4]                                       │       │
       │                                        │       │
       ▼                                        │       │
Point embedding + temporal position             │       │
       │                                        │       │
       ▼                                        │       │
Temporal residual blocks                        │       │
       │                                        │       │
       ▼                                        ▼       ▼
Trajectory tokens ───────────────→ Scene Cross-Attention
                                      │
                                      ▼
                             Temporal residual blocks
                                      │
                                      ▼
                               Point-wise score head
                                      │
                                      ▼
                              ego score [B,T,4]
```

The first implementation should **freeze the entire pretrained Diffusion Planner** and train only the new score branch. This isolates the research question and avoids changing the original planner distribution during score learning.

---

# 1. Research Question

The project should answer the following main question:

> Can a lightweight, time-independent conditional score model learn a useful local density field around clean driving trajectories and improve a pretrained generative planner through learned trajectory-space refinement?

Subquestions:

1. Can a fixed-$\sigma$ conditional DSM objective learn a stable score field for structured 80-step trajectories?
2. Does explicit temporal modeling improve score quality compared with a plain MLP/flattened representation?
3. Does conditioning on the original Diffusion Planner scene representation improve score quality over trajectory-only score learning?
4. Does score ascent improve the generated ego trajectory in offline trajectory metrics?
5. Most importantly, does it improve **nuPlan closed-loop planning score**?
6. Is local trajectory likelihood/density improvement aligned with closed-loop planning utility, or are the two objectives partly misaligned?

Do not assume the answer to Question 5 or 6 in advance. The experiment should be designed so that either outcome is scientifically interpretable.

---

# 2. Important Constraints

## 2.1 Do not change the original planner in Phase 1

Load the released/pretrained Diffusion Planner checkpoint and freeze:

- scene encoder;
- diffusion decoder / DiT;
- route encoder;
- all normalization parameters.

Only the new score branch is trainable.

This is crucial because the first experiment should answer whether an **additional learned local density field** can refine an already-trained planner.

## 2.2 No diffusion timestep in the score branch

The score branch must **not** take diffusion time $t$.

Do not implement:

```python
score = score_branch(traj, t, scene)
```

Implement:

```python
score = score_branch(traj, scene_context, route_embedding)
```

The branch estimates one fixed-smoothed-distribution score field.

## 2.3 Optimize ego only

The nuPlan planner ultimately executes/evaluates the ego trajectory. Therefore the score branch should output:

```text
[B, T, 4]
```

for the ego future only.

Neighbors should be used as **conditioning information through the shared scene encoder**, not as optimization variables in the first version.

Do **not** jointly update predicted neighbor trajectories in Phase 1.

## 2.4 Keep the method learning-based

Do not add rule-based costs such as:

- lane-center cost;
- hand-coded collision cost;
- TTC gradient;
- comfort penalty;
- manually constructed drivable-area gradient.

The only allowed post-update projection in the primary experiment is representation validity handling for heading `(cos, sin)`; this is not a planning rule.

---

# 3. Verified Baseline Code Structure

The current official repository uses the following relevant files.

## 3.1 Original planner wrapper

`diffusion_planner/model/diffusion_planner.py`

Current top-level flow:

```python
encoder_outputs = self.encoder(inputs)
decoder_outputs = self.decoder(encoder_outputs, inputs)
return encoder_outputs, decoder_outputs
```

Use the existing encoder rather than reimplementing scene processing.

## 3.2 Scene encoder

`diffusion_planner/model/module/encoder.py`

The encoder creates fused context tokens from agents, static objects and lanes and returns:

```python
encoder_outputs['encoding']
```

Use this tensor as the scene context $C$.

Do not redesign the scene encoder in the first experiment.

## 3.3 Original decoder

`diffusion_planner/model/module/decoder.py`

The original decoder:

- constructs current ego + predicted neighbor states;
- trains on joint ego/neighbor trajectories;
- flattens each agent's whole `[current + future]` trajectory into one token;
- uses DiT;
- uses `RouteEncoder`;
- during inference starts from Gaussian noise and calls DPM-Solver++;
- inverse-normalizes the generated states and returns `prediction`.

Default output representation per future point is:

```text
(x, y, cos(theta), sin(theta))
```

The original `RouteEncoder` is already part of `decoder.dit` and should be reused/frozen for this project.

## 3.4 Original diffusion training loss

`diffusion_planner/loss.py`

The current code samples random diffusion time `t`, perturbs the future, and trains either score or `x_start`. The default repository configuration is `x_start`.

This new score training should **not** reuse this variable-$t$ objective. Implement a separate fixed-$\sigma$ score loss.

## 3.5 Original training data preparation

`diffusion_planner/train_epoch.py`

Reuse the same preprocessing conventions:

- ego future from `batch[1]`;
- neighbor history / map / route / static object inputs;
- convert heading to `(cos, sin)`;
- apply the existing observation normalization before the scene encoder.

For the first score-training experiment, turn off the repository's additional `StatePerturbation` augmentation unless explicitly enabled by a later ablation. Fixed-$\sigma$ DSM already supplies the trajectory perturbation needed by the score objective.

## 3.6 nuPlan output

`diffusion_planner/planner/planner.py`

The planner consumes:

```python
outputs['prediction'][0, 0]
```

as the ego future, converts `(cos, sin)` back to heading with `atan2`, and transforms it to nuPlan states.

Therefore the first score-refinement implementation only needs to replace/refine:

```python
outputs['prediction'][:, 0]
```

while leaving neighbor predictions unchanged.

---

# 4. Score Definition

## 4.1 Desired object

We want a time-independent score field:

$$
s_\phi(\tau,C,R) \approx \nabla_\tau \log p_\sigma(\tau\mid C,R).
$$

Do not describe this mathematically as the exact unsmoothed empirical-data score $\nabla\log p_{data}$, because clean trajectory data lie near a thin structured manifold. We intentionally learn the score of a small Gaussian-smoothed distribution $p_\sigma$.

## 4.2 Fixed-noise DSM

Given a normalized clean ego future trajectory:

$$
\tau_0 \sim p_{data}(\tau\mid C,R),
$$

sample:

$$
\epsilon \sim \mathcal N(0,I),
$$

and construct:

$$
\tilde\tau = \tau_0 + \sigma\epsilon.
$$

The conditional denoising-score target is:

$$
s^*(\tilde\tau\mid\tau_0) = -\frac{\epsilon}{\sigma}.
$$

Train the branch using the numerically stable equivalent objective:

$$
\mathcal L_{score}
=
\mathbb E\left[\left\|\sigma s_\phi(\tilde\tau,C,R)+\epsilon\right\|_2^2\right].
$$

Since $\sigma$ is fixed, this has the same optimum as direct regression to $-\epsilon/\sigma$ but avoids unnecessarily large target magnitudes when $\sigma$ is small.

## 4.3 Coordinate system

Perform score learning in the **Diffusion Planner normalized ego trajectory space**.

Reason:

- x/y and heading components otherwise have different scales;
- isotropic fixed Gaussian noise is much better defined after normalization;
- optimization step sizes become easier to compare.

Implement explicit ego-only helpers, for example:

```python
def normalize_ego_future(traj, state_normalizer):
    # traj: [B,T,4]
    ego_mean = state_normalizer.mean[0].to(traj.device)
    ego_std = state_normalizer.std[0].to(traj.device)
    return (traj - ego_mean) / ego_std


def denormalize_ego_future(traj_norm, state_normalizer):
    ego_mean = state_normalizer.mean[0].to(traj_norm.device)
    ego_std = state_normalizer.std[0].to(traj_norm.device)
    return traj_norm * ego_std + ego_mean
```

Do not call the multi-agent normalizer on a `[B,1,T,4]` tensor and accidentally broadcast it to all configured agents.

---

# 5. Proposed Score Branch Architecture

Create a new file:

```text
diffusion_planner/model/module/score_branch.py
```

Primary class:

```python
class ScoreFunctionBranch(nn.Module):
    def __init__(
        self,
        future_len=80,
        input_dim=4,
        hidden_dim=192,
        num_heads=6,
        dropout=0.1,
        pre_dilations=(1, 2),
        post_dilations=(2, 4),
    ):
        ...

    def forward(
        self,
        ego_traj_norm,      # [B,T,4]
        scene_context,      # [B,N,H]
        route_embedding,    # [B,H]
    ):
        # return [B,T,4]
        ...
```

No diffusion timestep argument.

## 5.1 Point embedding

Input:

```text
[B, T, 4]
```

Apply:

```python
nn.Linear(4, hidden_dim)
```

Result:

```text
[B, T, H]
```

Use `H=192` by default so it matches the frozen Diffusion Planner scene and route representations.

## 5.2 Temporal positional embedding

Add a learnable future-time positional embedding:

```python
self.temporal_pos = nn.Parameter(torch.zeros(1, future_len, hidden_dim))
```

This indicates whether a token is future step 1, 2, ..., 80.

This is **trajectory-time position**, not diffusion time.

## 5.3 Route conditioning

Reuse the frozen original route encoder:

```python
route_embedding = base_model.decoder.decoder.dit.route_encoder(inputs['route_lanes'])
```

The exact wrapper nesting may differ depending on the local branch, so Codex should inspect the instantiated model and keep the implementation consistent with the repository.

Broadcast the route embedding to all trajectory tokens:

```python
h = h + route_embedding[:, None, :]
```

Do not add a diffusion timestep embedding.

## 5.4 Temporal residual blocks

Temporal modeling means the model explicitly couples neighboring future trajectory points instead of treating all `T*4` coordinates independently.

Implement a reusable residual block operating along the trajectory-time axis.

Recommended first version:

```python
class TemporalResidualBlock(nn.Module):
    def __init__(self, dim, dilation=1, dropout=0.1):
        ...
```

Conceptual block:

```text
input [B,T,H]
   │
LayerNorm
   │
transpose -> [B,H,T]
   │
Conv1d(H,H,kernel=3,dilation=d,padding=d)
   │
GELU
   │
Conv1d(H,H,kernel=3,dilation=d,padding=d)
   │
transpose -> [B,T,H]
   │
Dropout
   │
+ residual
```

Use two blocks before scene fusion, with default dilations:

```text
1, 2
```

and two after scene fusion:

```text
2, 4
```

This gives local smoothness bias plus a growing receptive field without a large extra transformer.

Do not use causal convolution: the whole candidate future trajectory is available simultaneously, so the score at time $i$ may use future points both before and after $i$ in trajectory order.

## 5.5 Scene cross-attention

Use trajectory tokens as queries and the frozen scene tokens as keys/values:

$$
Q=H_\tau,\qquad K=V=C.
$$

Implement one residual cross-attention block:

```text
h
 │
LayerNorm
 │
MultiheadAttention(Q=h, K=C, V=C)
 │
+ residual
 │
LayerNorm
 │
MLP(H -> 4H -> H)
 │
+ residual
```

Use:

```python
nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
```

Default `num_heads=6`, matching the original model.

The original Diffusion Planner DiT cross-attention does not expose/use a scene key-padding mask. For the first implementation, stay consistent with the baseline and use the fused scene context directly. Do not modify the encoder just to expose a mask in Phase 1. Mask-aware attention can be a later ablation if necessary.

## 5.6 Point-wise score head

After the second temporal stage:

```python
LayerNorm(H)
Linear(H, H)
GELU
Linear(H, 4)
```

Output:

```text
[B,T,4]
```

This is the gradient in normalized trajectory coordinates.

The final output should **not** be flattened to `[B, T*4]` inside the public interface.

## 5.7 Why this architecture

The architecture deliberately separates three roles:

1. **Temporal blocks:** model trajectory structure and coordinated changes across neighboring time points.
2. **Scene cross-attention:** make the score conditional on lanes, current/history agents and static objects already encoded by Diffusion Planner.
3. **Score head:** convert the learned token representation into a per-coordinate gradient.

This avoids training a second full planner while still giving the score branch enough capacity to learn a structured conditional vector field.

---

# 6. Optional Neighbor-Future Conditioning — Not Phase 1

The original Diffusion Planner jointly predicts ego + neighbor futures. However, nuPlan executes/scores the ego planner output, so the score branch should optimize ego only.

In Phase 1, condition on neighbors through the **scene encoder**, which already contains agent history/current state.

Only after the basic branch is working, add an optional experiment that also conditions on the original Diffusion Planner's predicted neighbor futures.

Possible design:

```text
predicted neighbor futures
[B,Nn,T,4]
       │
small trajectory encoder
       │
neighbor-future tokens
       │
additional cross-attention into ego trajectory tokens
```

Do not update those neighbor futures during score ascent.

Reason to postpone this:

- in reactive nuPlan simulation, true future neighbor behavior can change when ego behavior changes;
- treating initial predicted neighbor futures as fixed ground truth may create a mismatch after a large ego update;
- local score refinement should first be tested with the simpler and cleaner condition set.

---

# 7. Training Pipeline

Create a separate training entry point rather than modifying `train_predictor.py` heavily:

```text
train_score_branch.py
```

Also create:

```text
diffusion_planner/score_loss.py
```

if useful for separation.

## 7.1 Load pretrained Diffusion Planner

Training sequence:

1. Build the original Diffusion Planner using its existing config.
2. Load the pretrained checkpoint, preferably the EMA state used by official inference.
3. Set the base model to `.eval()`.
4. Set every base parameter `requires_grad=False`.
5. Instantiate `ScoreFunctionBranch`.
6. Only pass score-branch parameters to the optimizer.

The frozen encoder should run under `torch.no_grad()`.

## 7.2 Data preparation

Reuse the original training dataset and batch conventions.

For each batch:

1. Construct `inputs` exactly as in `train_epoch.py`.
2. Convert ego future heading to `(cos, sin)` exactly as the original code does.
3. Normalize observation inputs with the existing observation normalizer.
4. For Phase 1 set repository `StatePerturbation` augmentation off.
5. Extract only ego future GT:

```text
[B,T,4]
```

6. Normalize it with ego state-normalizer statistics.

## 7.3 Frozen conditioning computation

Under `torch.no_grad()`:

```python
encoder_outputs = base_model.encoder(inputs)
scene_context = encoder_outputs['encoding']
route_embedding = base_model.decoder.decoder.dit.route_encoder(inputs['route_lanes'])
```

Verify the exact module path in the local code before hardcoding it.

Expected shapes:

```text
scene_context   [B,N,192]
route_embedding [B,192]
```

## 7.4 Fixed-noise DSM sample

For normalized ego future `tau0`:

```python
eps = torch.randn_like(tau0)
tau_noisy = tau0 + sigma_score * eps
score_pred = score_branch(tau_noisy, scene_context, route_embedding)
loss = ((sigma_score * score_pred + eps) ** 2).mean()
```

No diffusion `t` is sampled.

No SDE schedule is used.

## 7.5 Initial hyperparameters

Make all values CLI-configurable. Suggested starting point:

```text
hidden_dim       = 192
num_heads        = 6
dropout          = 0.1
sigma_score      = 0.05   # normalized space, initial pilot only
learning_rate    = 1e-4
weight_decay     = 1e-4
grad_clip        = 5.0
optimizer        = AdamW
```

Do not treat `sigma_score=0.05` as a final value. It must be swept after the pipeline is verified.

Recommended sigma sweep in normalized space:

```text
0.01, 0.03, 0.05, 0.10
```

For every sigma, log the equivalent **denormalized** perturbation magnitude for x/y and heading representation so the scale is physically interpretable.

## 7.6 Training schedule

Use two stages:

### Smoke stage

- very small subset;
- 1–3 epochs or a fixed small number of iterations;
- verify loss decreases;
- verify no NaNs;
- verify score branch alone receives gradients;
- verify frozen Diffusion Planner parameters do not change.

### Full stage

After smoke tests pass, train on the normal training split.

Do not commit to a large number of epochs until validation curves are observed. Add early checkpointing and save best validation loss.

## 7.7 Checkpoint format

Save score-specific metadata:

```python
{
    'score_branch': score_branch.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'sigma_score': sigma_score,
    'score_config': {...},
    'base_planner_checkpoint': str(...),
}
```

The score checkpoint should not duplicate the full frozen base model unless needed for convenience.

---

# 8. Offline Validation Before nuPlan Simulation

Do not immediately launch 10-hour closed-loop experiments. The branch must pass offline diagnostics first.

Create:

```text
eval_score_branch.py
```

and save per-sample metrics to CSV/JSON.

## 8.1 DSM validation loss

Report:

$$
\|\sigma s_\phi(\tilde\tau)+\epsilon\|^2.
$$

This is necessary but not sufficient.

## 8.2 Score-direction cosine similarity

Because the sampled conditional target is known during synthetic corruption, compute:

$$
\cos\left(s_\phi(\tilde\tau),-\epsilon/\sigma\right).
$$

Report:

- mean cosine similarity;
- median;
- fraction `cos > 0`;
- distribution/histogram.

This directly tests whether the learned score points broadly in the correct denoising direction.

## 8.3 One-step corrupted-GT recovery

Starting from:

$$
\tilde\tau=\tau_0+\sigma\epsilon,
$$

apply:

$$
\tau' = \tilde\tau + \gamma\sigma^2s_\phi(\tilde\tau,C,R).
$$

Compare error to clean GT before and after.

Metrics:

- normalized trajectory MSE;
- x/y ADE;
- x/y FDE;
- heading angular error;
- percentage of samples whose error improves.

Test several `gamma` values:

```text
0.1, 0.25, 0.5, 1.0
```

This is a critical gate. If the branch cannot reliably move synthetically perturbed clean trajectories back toward GT, do not proceed directly to closed-loop refinement.

## 8.4 Tweedie-style denoised estimate

For fixed additive Gaussian corruption, evaluate:

$$
\hat\tau_0 = \tilde\tau + \sigma^2s_\phi(\tilde\tau,C,R).
$$

This gives a useful interpretation of the learned score and should be logged separately from iterative ascent.

## 8.5 Temporal smoothness diagnostics

Log the temporal variation of the predicted score:

$$
D_1 = \frac{1}{T-1}\sum_i \|s_{i+1}-s_i\|,
$$

and optionally second difference:

$$
D_2 = \frac{1}{T-2}\sum_i \|s_{i+2}-2s_{i+1}+s_i\|.
$$

These are diagnostic metrics only. Do **not** add them as hand-crafted training losses in the first experiment.

Compare them to a simple MLP baseline to quantify whether temporal modeling actually reduces zig-zag score fields.

---

# 9. Inference-Time Refinement

Create a separate wrapper rather than rewriting DPM-Solver:

```text
diffusion_planner/model/score_refined_planner.py
```

Conceptual behavior:

```python
encoder_outputs, decoder_outputs = base_model(inputs)

prediction = decoder_outputs['prediction']
ego = prediction[:, 0]                  # [B,T,4], physical/original scale

ego_norm = normalize_ego_future(ego)
route_embedding = frozen_route_encoder(inputs['route_lanes'])
scene_context = encoder_outputs['encoding']

for _ in range(K):
    score = score_branch(ego_norm, scene_context, route_embedding)
    ego_norm = ego_norm + gamma * sigma_score**2 * score
    ego_norm = project_heading_representation_if_enabled(ego_norm)

ego_refined = denormalize_ego_future(ego_norm)
prediction[:, 0] = ego_refined

decoder_outputs['prediction'] = prediction
return encoder_outputs, decoder_outputs
```

## 9.1 Important: no inference noise

At refinement time, feed the actual clean candidate trajectory produced by Diffusion Planner directly to the fixed-$\sigma$ score model.

Do **not** artificially add Gaussian noise before every refinement step in the primary experiment.

The network is being evaluated as the score of the smoothed density at the candidate location.

## 9.2 Refinement update

Use:

$$
\boxed{
\tau_{k+1} = \tau_k + \gamma\sigma^2 s_\phi(\tau_k,C,R)
}
$$

rather than an arbitrary raw `lr * score` as the primary parameterization.

Reason:

- the score magnitude depends on the smoothing scale;
- $\sigma^2s$ has the natural scale of a denoising/mean-shift displacement;
- `gamma` becomes an interpretable dimensionless refinement strength.

Still expose `gamma` and `K` as command-line/config parameters.

Suggested initial grid after offline validation:

```text
K     = {1, 3, 5, 10}
gamma = {0.05, 0.1, 0.25, 0.5, 1.0}
```

Do not run the whole grid in expensive closed-loop simulation. Use offline diagnostics and a small nuPlan pilot to reduce it first.

## 9.3 Heading representation handling

The trajectory representation uses `(cos(theta), sin(theta))`. Score ascent can move these two values off the unit circle.

Implement an optional representation projection after each update:

1. denormalize the updated trajectory;
2. compute

$$
r=\sqrt{c^2+s^2+\epsilon};
$$

3. replace `(c,s)` by `(c/r,s/r)`;
4. re-normalize for the next score iteration.

This is a representation-validity projection, not a planning heuristic.

Log experiments with it enabled and, if useful, one ablation without it.

## 9.4 Do not modify neighbors

Keep:

```python
prediction[:, 1:]
```

unchanged in Phase 1.

Only:

```python
prediction[:, 0]
```

is refined.

---

# 10. Evaluation Strategy

The final success criterion is **nuPlan closed-loop planning score**, not score-matching loss alone.

However, use a staged evaluation so failures are diagnosable.

## 10.1 Level A — unit and architecture tests

Write tests for:

- input/output shapes;
- score branch has no timestep input;
- temporal blocks preserve length;
- cross-attention shape compatibility;
- base model parameters stay frozen;
- score branch receives nonzero gradients;
- checkpoint save/load;
- heading projection does not produce NaN;
- refinement with `gamma=0` exactly reproduces baseline output.

## 10.2 Level B — synthetic corruption validation

Use the diagnostics from Section 8.

The branch should demonstrate a clearly positive local correction ability before expensive simulation.

## 10.3 Level C — offline Diffusion Planner output refinement

Run the frozen baseline planner on validation samples and record its generated ego trajectory.

Refine it with the score branch and compare against logged expert future using:

- ADE;
- FDE;
- heading error;
- displacement introduced by refinement;
- score norm;
- temporal smoothness;
- before/after visualization.

Important: improvement in expert imitation metrics is **not** equivalent to improved closed-loop score. Treat this only as a diagnostic.

## 10.4 Level D — nuPlan smoke closed-loop

Before a full benchmark:

- run 20–50 scenarios;
- confirm simulation does not crash;
- confirm refined trajectories are finite;
- inspect NuBoard before/after;
- record per-planning-step inference overhead.

## 10.5 Level E — nuPlan validation

Hyperparameter selection should be done on validation, preferably Val14 first.

Evaluate baseline and refinement under identical:

- model checkpoint;
- scenario set;
- simulation configuration;
- random seed policy.

Start with non-reactive agents to isolate ego-trajectory effects, then evaluate reactive agents.

Report at least:

- final aggregate score;
- drivable area compliance;
- driving direction compliance;
- ego comfort;
- ego progress / progress along expert route if available;
- no ego at-fault collision;
- speed-limit compliance;
- time-to-collision metric;
- scenario count;
- score-branch latency per planning cycle.

Use paired per-scenario comparison:

$$
\Delta_i = score_i^{refined} - score_i^{baseline}.
$$

Report:

- mean delta;
- median delta;
- win/tie/loss count;
- score distribution;
- metric-level deltas.

## 10.6 Level F — held-out hard benchmark

Only after choosing `sigma`, `gamma`, `K` and architecture on validation, run the selected configuration on Test14-hard (NR/R as appropriate).

Do not tune all hyperparameters directly on the held-out hard set.

---

# 11. Required Ablations

Do not implement all ablations before the main pipeline works. Add them after the primary branch passes offline tests.

## 11.1 Architecture ablation

Compare:

### A. MLP trajectory score baseline

Flatten trajectory or use a simple per-trajectory MLP with the same scene condition.

Purpose: test whether explicit temporal structure is actually needed.

### B. Temporal-only

```text
trajectory -> TCN -> score
```

No scene context.

Purpose: test whether trajectory structure alone explains the gain.

### C. Temporal + Scene

```text
trajectory -> TCN -> cross-attention(scene) -> TCN -> score
```

This is the main model.

### D. Temporal + Scene + Route

Add the frozen route embedding. This should be the full primary model if it performs best.

## 11.2 Sigma ablation

At least:

```text
0.01 / 0.03 / 0.05 / 0.10
```

Evaluate both score quality and refinement quality.

Expected tradeoff to test:

- too small sigma: target field may become difficult/high variance and very local;
- too large sigma: score becomes easier/smoother but may represent an over-smoothed behavior distribution.

Do not assume the optimum.

## 11.3 Refinement strength and iterations

Evaluate `gamma` and `K` separately.

Track not only benchmark score but trajectory displacement from the baseline:

$$
\|\tau_{refined}-\tau_{DP}\|.
$$

This determines whether improvements come from genuinely local correction or large behavior changes.

## 11.4 Scene conditioning ablation

Compare:

```text
s(τ)
```

vs

```text
s(τ | scene)
```

This is important because a trajectory can be globally plausible but wrong for the current road/traffic scene.

## 11.5 Optional predicted-neighbor-future conditioning

Only after the main model works:

```text
s(τ_ego | scene, route, predicted neighbor futures)
```

Do not optimize neighbors themselves.

---

# 12. Instrumentation and Logging

Every experiment should save enough information to diagnose why score ascent helps or fails.

For each refined trajectory log:

```text
scenario id / token
baseline ego trajectory
refined ego trajectory
score at every refinement iteration
score norm per iteration
trajectory displacement per iteration
heading-unit-norm deviation
K
gamma
sigma
baseline nuPlan score
refined nuPlan score
metric breakdown
```

For a small visualization subset also plot:

- map / route;
- baseline trajectory;
- refined trajectory;
- expert trajectory when available;
- score vectors at selected future points;
- refinement trajectory across iterations.

Do not rely on aggregate final score alone.

---

# 13. Recommended File-Level Implementation Plan

Keep changes modular and avoid breaking the original baseline.

## New files

```text
diffusion_planner/model/module/score_branch.py
    TemporalResidualBlock
    SceneCrossAttentionBlock
    ScoreFunctionBranch

diffusion_planner/model/score_refined_planner.py
    wrapper for frozen DP + score branch refinement

diffusion_planner/score_loss.py
    fixed_sigma_dsm_loss
    ego normalize/denormalize helpers if not placed in utils

train_score_branch.py
    dedicated frozen-backbone score training

eval_score_branch.py
    offline DSM / recovery diagnostics

scripts or shell files as needed for score training/evaluation
```

## Minimal existing-file modifications

Prefer configuration hooks over invasive edits.

Potential modifications:

```text
diffusion_planner/planner/planner.py
```

Only if necessary to instantiate the score-refined wrapper/checkpoint. Prefer a separate planner class/file if that keeps the baseline untouched.

Add CLI/config values for:

```text
score_checkpoint
score_sigma
score_gamma
score_steps
score_hidden_dim
score_num_heads
score_heading_projection
```

Keep original baseline runnable with score refinement disabled.

---

# 14. Codex Execution Order

Codex should implement in this order and stop/report at each gate if something is inconsistent.

## Phase 1 — Repository inspection and shape verification

Before coding architecture assumptions, inspect the local repository and print/verify:

```text
encoder_outputs['encoding'].shape
route_encoder(route_lanes).shape
ego future shape after heading conversion
base inference prediction shape
state normalizer ego mean/std shapes
```

Expected conceptual shapes:

```text
scene_context   [B,N,192]
route_embedding [B,192]
ego future       [B,80,4]
prediction       [B,1+neighbors,80,4]
```

If local code differs, adapt implementation while preserving the research design.

## Phase 2 — Implement score branch + unit tests

Implement:

- point projection;
- temporal positional embedding;
- temporal residual blocks;
- scene cross-attention;
- route addition;
- point-wise score head.

Run synthetic shape tests before integrating data.

## Phase 3 — Implement frozen-backbone DSM training

Implement `train_score_branch.py`.

Verify:

- only score branch parameters change;
- base planner is eval/frozen;
- loss decreases on a tiny overfit subset;
- no diffusion time appears in score-branch inputs.

First overfit a very small subset. If the branch cannot overfit synthetic DSM targets, debug architecture/training before full training.

## Phase 4 — Implement offline evaluation

Implement all Section 8 metrics.

Produce a compact report file with:

- validation DSM loss;
- target/predicted cosine similarity;
- corrupted-GT before/after ADE/FDE;
- score temporal variation;
- qualitative trajectory plots.

## Phase 5 — Implement DP-output refinement

Create the wrapper that:

1. calls the untouched original planner;
2. selects ego prediction;
3. normalizes it;
4. runs K score-ascent iterations;
5. handles heading representation;
6. writes refined ego back into the output tensor.

Verify `gamma=0` produces bitwise/numerically identical planner output to baseline.

## Phase 6 — Small nuPlan smoke test

Run a very small scenario subset.

Save before/after trajectory and timing logs.

Manually inspect NuBoard.

## Phase 7 — Validation hyperparameter study

Choose a small set of promising:

```text
sigma_score
gamma
K
```

based on offline validation, then run Val14.

Do not launch the full combinatorial grid in closed-loop simulation.

## Phase 8 — Main benchmark

Freeze all choices and run the selected configuration on the target hard benchmark.

---

# 15. Failure Modes to Watch Carefully

## 15.1 DSM loss decreases but refinement direction is bad

Possible causes:

- score target is too high variance because sigma is too small;
- branch memorizes local corruption without learning useful structured direction;
- temporal architecture insufficient;
- mismatch between Gaussian-corrupted GT training points and Diffusion Planner candidate errors.

Use one-step corrupted-GT recovery and DP-output offline evaluation to distinguish these.

## 15.2 Score works on corrupted GT but damages DP trajectories

This is an important possible distribution mismatch:

```text
training input = GT + Gaussian noise
inference input = planner candidate with structured/model bias
```

Do not immediately add rule-based losses.

First quantify how far DP trajectories lie from the fixed-sigma training neighborhood.

Possible later research directions include training on more realistic learned/model-induced perturbations, but do not mix them into the first clean experiment.

## 15.3 Trajectory becomes zig-zag

Compare against the temporal-MLP ablation and inspect:

- per-step score vectors;
- first/second temporal score differences;
- whether TCN receptive field is sufficient.

Do not immediately add an explicit smoothness loss unless architecture alone is shown insufficient.

## 15.4 Score norm explodes

Check:

- normalization;
- sigma value;
- stable loss form `||sigma * s + eps||^2`;
- gradient clipping;
- refinement gamma;
- repeated iteration count.

## 15.5 Heading `(cos,sin)` drifts off unit circle

Use the representation projection described above and log the drift before projection.

## 15.6 Offline imitation improves but nuPlan score does not

Do not treat this automatically as implementation failure.

It may indicate that:

$$
\text{higher behavior density / closer expert imitation}
\not\Rightarrow
\text{higher closed-loop utility}
$$

But only make this scientific claim after verifying the score model is genuinely learned and the refinement is behaving as intended.

## 15.7 Large refinement changes interaction mode

The current method is intended as **local refinement**.

If optimized trajectories move far from the original DP plan, neighbor/context assumptions become less reliable, especially under reactive simulation.

Always report trajectory displacement vs baseline and prefer configurations whose gains occur with modest local corrections.

---

# 16. Primary Experiment Table

Codex should ultimately produce a table with at least the following variants:

| ID | Score architecture | Scene | Route | Optimized variable | sigma | K | gamma | Purpose |
|---|---|---|---|---|---:|---:|---:|---|
| B0 | none | — | — | none | — | 0 | 0 | original Diffusion Planner baseline |
| S1 | MLP baseline | yes | yes | ego | fixed | chosen | chosen | test whether simple network is enough |
| S2 | TCN | no | no/optional | ego | fixed | chosen | chosen | temporal modeling only |
| S3 | TCN + cross-attn | yes | no | ego | fixed | chosen | chosen | scene conditioning |
| S4 | TCN + cross-attn | yes | yes | ego | fixed | chosen | chosen | primary method |
| S5 | S4 + predicted neighbor future | yes | yes | ego only | fixed | chosen | chosen | optional interaction ablation |

Do not run S5 until S4 is stable.

---

# 17. What Counts as Success

The project should not define success only as “training loss went down.”

A convincing result requires a chain of evidence:

1. **Score learning works:** DSM validation and direction metrics are meaningful.
2. **Local correction works:** corrupted clean trajectories move toward the clean target.
3. **Planner candidate refinement is controlled:** updates are smooth/local rather than catastrophic.
4. **Closed-loop effect is measured:** nuPlan paired scenario metrics show whether the method helps.
5. **Ablations explain why:** temporal modeling and scene conditioning are tested rather than assumed.

The strongest positive outcome is:

```text
original DP generation
        +
lightweight learned clean-space score refinement
        -> measurable closed-loop improvement
```

A scientifically useful negative outcome is also possible if the score is demonstrably learned but density ascent systematically fails to improve closed-loop utility. In that case, the conclusion should focus on the mismatch between behavior-density refinement and planning objective, not merely “score network failed.”

---

# 18. Non-Goals for the First Version

Do not do the following until the primary pipeline is complete:

- jointly retrain the original Diffusion Planner and score branch;
- replace DPM-Solver;
- modify the original diffusion sampling process;
- learn variable-$t$ diffusion score again;
- optimize neighbor trajectories;
- add handcrafted nuPlan metric gradients;
- add RL/reward fine-tuning;
- add multiple score noise levels with sigma conditioning;
- add a second large transformer backbone;
- add physical/kinematic projection beyond basic heading representation validity.

These may be future extensions, but including them now would make the experiment difficult to interpret.

---

# 19. Short Instruction Block for Codex

Use this as the implementation priority summary:

> Implement a new time-independent ego trajectory score branch on top of the frozen official Diffusion Planner. Reuse the original scene encoder and route encoder. The branch takes a normalized `[B,80,4]` ego candidate trajectory, frozen scene tokens, and frozen route embedding; it contains point embedding, temporal residual Conv1D blocks, scene cross-attention, more temporal blocks, and a point-wise `[B,80,4]` score head. Train only this branch using fixed-sigma Gaussian DSM with loss `mean((sigma * score_pred + eps)**2)`. No diffusion timestep is an input. At inference, run the original Diffusion Planner unchanged, select the ego prediction, normalize it, and apply `K` local refinement steps `tau += gamma * sigma**2 * score(tau, scene, route)`. Update ego only. Keep neighbor predictions unchanged. Add offline score-direction and corrupted-GT recovery diagnostics before running nuPlan closed-loop evaluation. Keep the baseline runnable and avoid invasive modifications to original files.

