# Full DD scene-reader Score experiment

This is the complete **C** model: fixed DD scene processing plus its pretrained trajectory-dependent reader, followed by a newly trained conditional Score branch. It does not run the smaller B ablation. The conditional DSM objective is unchanged. The initial trajectory can still come from any NAVSIM planner.

## Model and data

The frozen path reconstructs the original DD BEV tokens, ego/agent queries and spatial BEV. It then applies the original whole-trajectory position encoder and the first decoder layer's BEV/agent/ego attention and FFN, ending before timestep modulation. Every training example reads at its **noisy** trajectory; every refinement step reads at the **current** trajectory. No clean-target-based local features are supplied to the network.

The Score receives 96 tokens: 64 BEV, 1 ego, 30 agents and 1 trajectory-dependent reader token. Its temporal processing and four-channel direct-score output are retained. Frozen DD weights stay in eval mode and FP32; the trainable Score uses FP16 mixed precision. All required frozen reader weights are included in exported checkpoints.

The new cache stores `[512,8,8]` FP16 backbone latents and 256-dimensional status features. It is approximately 6.4 GiB of tensors for the complete 103,288 training/validation scenes, avoiding a much larger dense-BEV cache. It reuses the old cache's exact scene selection, target bytes and train-only normalizer, while re-encoding the original camera/LiDAR observations. It never tries to recover full backbone features from the old compressed tokens. Existing caches and outputs remain intact.

## Full training and automatic evaluation

| Setting | Value |
| --- | --- |
| Training | From scratch, 500 epochs |
| GPUs / global batch | 8 / 512 (64 per GPU) |
| Optimizer | AdamW, peak LR 6e-4, weight decay 1e-4 |
| LR schedule | 3-epoch warmup, cosine over the full 500-epoch horizon, minimum 1e-6 |
| Noise / refinement | sigma 0.1, gamma 0.1 |
| Seed | 20260922 |
| Weights | RAW; EMA disabled |
| Snapshots | Epochs 50, 100, ..., 500 |
| Automatic evaluation | Final RAW, K=5,10,15,20,30; exact saved K0 reused |

The queue performs cache preparation, a real-data native-reader consistency check during extraction, a real-batch GPU DSM optimizer/validation check, full training, then full 12,146-scene paired evaluation. These checks validate implementation and readiness; they are not small scientific ablation runs. A failed prerequisite stops dependent stages and records the failing log.

All paths follow the existing server layout under `/home/data/wyk`:

- Source cache: `dataset/navsim/exp/dd_score_cache_v1`
- New cache: `dataset/navsim/exp/dd_score_reader_cache_v1`
- Training: `dataset/navsim/exp/dd_score_reader_v1`
- Evaluation: `dataset/navsim/exp/dd_score_reader_navtest_v1`
- Queue state and stage logs: `dataset/navsim/exp/dd_score_reader_pipeline/`
- Existing K0 pointer: `dataset/navsim/exp/dd_score_e200_eval_latest.txt`

Run from the checkout containing this version:

```bash
cd /home/data/wyk/Score_Function_NAVSIM_DDAligned
PY=/home/data/wyk/envs/diffusion_drive/bin/python
JOBS=/home/data/wyk/dataset/navsim/exp/dd_score_reader_pipeline
mkdir -p "$JOBS"
nohup "$PY" -u scripts/run_navsim_dd_reader.py run \
  > "$JOBS/queue.log" 2>&1 < /dev/null &
echo "DD reader pipeline PID=$!"
```

Inspect status and the active stage's log:

```bash
/home/data/wyk/envs/diffusion_drive/bin/python \
  /home/data/wyk/Score_Function_NAVSIM_DDAligned/scripts/run_navsim_dd_reader.py status
tail -n 30 /home/data/wyk/dataset/navsim/exp/dd_score_reader_pipeline/queue.log
```

The four stage logs are `cache.log`, `preflight.log`, `train.log`, and `evaluate.log`; cache GPU workers also log to `dd_score_reader_cache_v1/worker*.log`. The queue prints a heartbeat every 30 seconds. After evaluation, `status` prints all five PDMS scores, deltas against K0 and improved/worsened/tied counts.

For recovery, execute the **same** `run` command with `>>` instead of `>` to preserve the queue log. Completed cache shards are validated and reused. Training resumes from the last durable epoch with optimizer/scaler/RNG state. Evaluation resumes compatible saved output. Locks prevent duplicate writers, and changed model/config/provenance is rejected. No global proxy or certificate configuration is changed by these scripts.

The local verification environment does not have the user's sensor data, official checkpoint or A100 GPUs. CPU tests cover native-module math, cache integrity and exact membership, actual DSM parameter updates with frozen DD weights, checkpoint/resume, inference and queue behavior. Real-weight/data/GPU checks run automatically on the server before full training.
