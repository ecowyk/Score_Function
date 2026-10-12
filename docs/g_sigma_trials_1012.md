# S05-G 续训与固定 sigma 对照

只运行三组实验，不包含多 sigma。保留已完成的原 S05-G 结果，所有新输出写入独立目录。

| 顺序 | 任务 | 初始化 | sigma | 目标 epoch |
|---|---|---|---|---|
| 1 | S05-G_plus100 | 原 S05-G 最终完整断点 | 0.05 | 529 → 629 |
| 2 | S10-G | 从头训练，与原配置相同 seed | 0.10 | 529 |
| 3 | S025-G | 从头训练，与原配置相同 seed | 0.025 | 529 |

sigma 沿用原配置的归一化坐标定义。三个任务均使用原训练代码的精确源码快照；网络、数据、增强、学习率、batch、验证方式和其他参数保持原配置。续训保留 raw/EMA 权重、优化器、全部 rank 的随机数状态、步数和历史，不重新 warmup。

原配置 `max_epochs=500` 同时要求至少 244000 次更新，因此完整预算为 529 epoch、244398 updates。新训练仍保留这个预算规则。续训目标为 629 epoch、290598 updates。

默认在 GPU 0、1、2、3 上按顺序运行，保持原 S05-G 的四卡设置：global batch 2048、每卡 microbatch 256、梯度累积 2。GPU 被其他任务占用时等待，不终止其他进程。

## 准备与启动

从 GitHub 获取 `codex/g-sigma-trials-1012` 分支，在独立 worktree 运行：

```bash
(
set -euo pipefail
cd /home/data/wyk/Score_Function_aligned_v1
git -c http.sslCAInfo=/etc/ssl/certs/ca-certificates.crt fetch origin codex/g-sigma-trials-1012
git worktree add --detach /home/data/wyk/Score_Function_G_sigma_1012 FETCH_HEAD
/home/data/wyk/envs/diffusion_planner/bin/python -u /home/data/wyk/Score_Function_G_sigma_1012/scripts/prepare_g_sigma_trials.py
tmux new-session -d -s score_G_sigma_1012 'exec /home/data/wyk/envs/diffusion_planner/bin/python -u /home/data/wyk/Score_Function_G_sigma_1012/scripts/queue_g_sigma_trials.py run >> /home/data/wyk/outputs/score_aligned_1004/G_sigma_1012_v1/queue.log 2>&1'
)
```

准备脚本读取原 G 运行的 `score/config.json`，生成三份完整配置和原始结果副本，核对训练源码、数据身份、四卡完整断点、预算，并做 CPU 模型前反向检查。它不会启动 GPU 训练。完成后才启动 tmux 队列。

如果希望使用 GPU 4、5、6、7，在首次准备命令末尾添加 `--gpus 4,5,6,7`。不要在队列启动后修改冻结的配置或代码。

## 查看进度

```bash
/home/data/wyk/envs/diffusion_planner/bin/python /home/data/wyk/Score_Function_G_sigma_1012/scripts/queue_g_sigma_trials.py status --watch --poll-seconds 5
```

```bash
tail -n 50 /home/data/wyk/outputs/score_aligned_1004/G_sigma_1012_v1/queue.log
tail -n 30 /home/data/wyk/outputs/score_aligned_1004/G_sigma_1012_v1/logs/S05-G_plus100.launcher.log
```

离开前确认第一组显示 running，且 step 从 244398 向上增长。`waiting_gpu` 表示等待所选四卡释放。训练失败会停止后续队列，并保留已有结果；修正失败原因后重新运行同一个 `run` 命令，会跳过已完成任务并从完整断点续训。不要再次执行 prepare，也不要启动重复 tmux 会话。

## 输出

队列目录：`/home/data/wyk/outputs/score_aligned_1004/G_sigma_1012_v1`。

- `reference/score/`：原 S05-G 529 epoch 结果副本。
- `configs/`：三组完整配置。
- `code/fixed/`：与原训练 provenance 完全一致的源码。
- `runs/<任务名>/score/last.pt`：该组完整训练状态。
- `runs/<任务名>/score/final_ema.pt`：该组最终 epoch 的 EMA 权重，供后续统一评测。
- `runs/<任务名>/score/best.pt`：按验证 DSM 选出的权重。续训组可能保留先前 epoch 的 best；需要最终结果时使用 `final_ema.pt`。

本队列不自动运行 nuPlan 闭环评测。
