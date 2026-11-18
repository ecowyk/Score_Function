# ChainFlow Score · B64 / 40 轮实验交接

在已经完成 ChainFlow 原生复现、缓存和第一版 Score 训练的节点上做一次独立训练。主结果预先指定为第 40 轮 EMA 在全量 navtest 的 K=20 配对 PDMS 差值。沿用每个 token 当时保存的原生 top-1 轨迹和指标，原生基准约为 93.8194。

## 实验配置

| 参数 | 本次设置 |
|---|---|
| 训练对象 | 同一条件 Score 网络，固定噪声 DSM，复用冻结的 ChainFlow 场景与 VLM 条件 |
| 数据 | train 85,109，val 18,179，navtest 12,146 |
| 批量 | 8 卡 × 每卡 8，梯度累积 1，全局 64 |
| 更新预算 | 每轮 1,329 次，共 40 轮、53,160 次更新。每轮打乱后舍去尾部 53 个样本 |
| 学习率 | AdamW，前 5,316 步线性从 2e-10 升至 2e-4，随后按更新步余弦降至 0 |
| 其余训练参数 | FP32，weight decay 0.01，梯度裁剪 5，EMA 0.999，Score seed 20260922，σ=0.1 |
| 保存与评测 | 每 5 轮保存 RAW 和 EMA。第 40 轮 final EMA 自动评测 K=5、10、15、20、30，γ=0.1 |

这个对照检验整组优化设置对 Score 的影响。全局 batch 从 512 缩到 64，一轮内的更新从 166 增至 1,329。作为参照，旧实验第 50 和 200 轮分别更新了 8,300 和 33,200 次。本次沿用项目原有的 DSM、EMA 和逐步学习率实现。40 轮、参考 batch 和学习率曲线参照 ChainFlow Stage 2 的[官方配置](https://github.com/AFARI-Research/ChainFlow-VLA/blob/0f3565f66546139a0183bc5807d5f7234ae2d4b0/navsim/planning/script/config/common/agent/chainflow_vla_stage2.yaml)及[官方调度实现](https://github.com/AFARI-Research/ChainFlow-VLA/blob/0f3565f66546139a0183bc5807d5f7234ae2d4b0/navsim/agents/chainflow_vla/chainflow_vla_agent.py)。

## 在公司节点启动或恢复

在独立 worktree 中运行以下命令。首次准备 worktree 时，在原 Score 仓库 fetch 分支 `codex/navsim-chainflow-score-b64-e40`，将交接消息中指定的 commit 以 detached worktree 放在 `/home/data/wyk/Score_Function_NAVSIM_ChainFlow_B64_E40`。

```bash
(
  set -euo pipefail
  cd /home/data/wyk/Score_Function_NAVSIM_ChainFlow_B64_E40
  exp=/home/data/wyk/dataset/navsim/exp
  py=/home/data/wyk/envs/chainflow/bin/python
  plan="$exp/chainflow_score_pipeline/plan.json"
  q="$exp/chainflow_score_pipeline_b64_e40"
  test -x "$py"
  test -r "$plan"
  mkdir -p "$q"
  if ! flock -n "$q/pipeline.lock" true; then
    echo "新队列已经运行，使用 status 查看进度"
    exit 0
  fi
  nohup nice -n 19 "$py" -u scripts/run_navsim_chainflow.py run \
    --recipe b64-e40 --assets-from-plan "$plan" \
    --workers 8 --microbatch-size 8 --gpu-policy wait-idle \
    >> "$q/queue.log" 2>&1 < /dev/null &
  echo "PID=$!  日志=$q/queue.log"
)
```

从旧 `plan.json` 读取实际的 cache、权重、两套 VLM 路径、原生交接及 GPU 租约目录。新实验要求已发布的 `TRAINING_READY.json`。依次复核原生轨迹与缓存分片校验和、单卡真实样本预检、8 卡正式训练、配对评测与结果校验。GPU 已有任务时排队。大型 NFS 缓存逐分片校验的进度写在 `verify_cache.log`。使用旧实验产物时保留其工作树、checkpoint、日志与进程。

## 监控

```bash
/home/data/wyk/envs/chainflow/bin/python \
  /home/data/wyk/Score_Function_NAVSIM_ChainFlow_B64_E40/scripts/run_navsim_chainflow.py \
  status --recipe b64-e40 --watch
```

Ctrl+C 退出监控。查看详细输出用 `tail -F /home/data/wyk/dataset/navsim/exp/chainflow_score_pipeline_b64_e40/queue.log`。状态里的 `log` 指向当前阶段日志。中断后回到同一 commit，使用上面的原启动命令；训练从本次实验最近一个完整 epoch 的 `last.pt` 恢复。遇到数据或哈希错误时保留现场，用阶段日志定位具体文件。

## 验收和汇报

训练开始后核对 `chainflow_score_b64_e40/run.json` 中的 world size、batch、累积和更新预算。队列最终显示 `status=complete` 后运行下列命令。表格统一使用 0–100 分，差值来自相同 token 的 K 与 K0。

```bash
/home/data/wyk/envs/chainflow/bin/python - <<'PY'
import json
from pathlib import Path

exp = Path("/home/data/wyk/dataset/navsim/exp")
train = exp / "chainflow_score_b64_e40"
queue = exp / "chainflow_score_pipeline_b64_e40"
evaluation = exp / "chainflow_score_navtest_b64_e40"

def read(path):
    return json.loads(path.read_text())

assert read(queue / "state.json")["status"] == "complete"
run = read(train / "run.json")
assert (run["protocol"]["world_size"], run["microbatch_per_rank"],
        run["gradient_accumulation"], run["global_batch"]) == (8, 8, 1, 64)
assert (run["updates_per_epoch"], run["planned_optimizer_updates"]) == (1329, 53160)
history = read(train / "history.json")
assert history[-1]["epoch"] == 40 and history[-1]["step"] == 53160
for epoch in range(5, 41, 5):
    for kind in ("raw", "ema"):
        assert (train / f"epoch_{epoch:04d}_{kind}.pt").is_file()
summary = read(evaluation / "summary.json")
assert summary["count"] == summary["valid_count"] == 12146
protocol = summary["protocol"]
assert (protocol["score_epoch"], protocol["score_weight_kind"], protocol["gamma"]) == (40, "ema", 0.1)
assert protocol["score_checkpoint_sha256"] == read(train / "TRAINING_COMPLETE.json")["final_ema_sha256"]
ks = (0, 5, 10, 15, 20, 30)
assert set(summary["by_k"]) == set(map(str, ks))
assert abs(100 * summary["by_k"]["0"]["mean"]["score"] - 93.8194331) < 0.0001
print(" K     PDMS      ΔK0    improved / worsened / tied")
for k in ks:
    row = summary["by_k"][str(k)]
    print(f"{k:2}  {100*row['mean']['score']:7.4f}  {100*row['mean_delta']['score']:+8.4f}"
          f"    {row['improved']} / {row['worsened']} / {row['tied']}")
print("EMA SHA-256", protocol["score_checkpoint_sha256"])
print("结果文件", evaluation / "summary.json")
PY
```

交付 Git commit、最终 checkpoint 路径和 SHA-256、训练样本数与更新数、六行结果、总结文件绝对路径。用 K20 对比旧实验第 50 和 200 轮在同一批场景上的结果，注明旧权重类型。如果旧汇总文件不可用，保留本次表格并明确缺少哪一个对照。
