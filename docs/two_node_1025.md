# 2026-10-25 两个八卡节点实验

节点 A 运行 **Test14-Random / NR** 的 12 组配对评测：UrbanDriver、PlanTF epoch17、PlanTF epoch24（last）、PDM-Open、GC-PGP、PLUTO w/o refine，各自 Vanilla / S10-G K20。DP 已有结果不重新运行。沿用已完成 hard_NR 的同一模型权重和模型推理，使用 DP 矩阵中已固定的 Random 场景清单，γ=0.1、σ=0.1。两个 PlanTF checkpoint 都保留。

节点 B 从实际完成的 S10-G 配置和原始源码快照创建 **E10-G**，仅改变 score/energy 参数化及输出路径。保持数据、增强、归一化、训练 seed、学习率、EMA、global batch2048、每卡256、八卡累积1、529 epochs / 244398 updates。完成训练后，从最后一轮 `last.pt` 中导出 EMA `final_ema.pt`，自动运行 **Test14-hard / NR、seed20260922、K=5/10/20/50/100**，复用该 seed 已完成的 DP K0，并列出原 direct-score 同 seed 同 K 结果。

`last.pt` 是 raw+EMA+优化器+采样/RNG 的完整恢复状态。此前 direct-score 评测用的是最后一轮 EMA `final_ema.pt`，不是验证集选出的 `best.pt`。Energy 沿用此规则。20260922 只改变评测随机种子，不改变训练 seed。

## 获取代码

两个节点均执行以下代码。若 `/home/data/wyk` 为共享目录，第二次会检查并复用同一 worktree，不重复添加。

```bash
(
set -euo pipefail
SOURCE=/home/data/wyk/Score_Function_G_sigma_8gpu_1012
TARGET=/home/data/wyk/Score_Function_TwoNode_1025
git -C "$SOURCE" fetch origin codex/random-baselines-energy-1025
EXPECTED=$(git -C "$SOURCE" rev-parse FETCH_HEAD)
if [ ! -e "$TARGET" ]; then
  git -C "$SOURCE" worktree add --detach "$TARGET" "$EXPECTED"
else
  test "$(git -C "$TARGET" rev-parse HEAD)" = "$EXPECTED"
  test -z "$(git -C "$TARGET" status --porcelain --untracked-files=no)"
fi
)
```

## 节点 A 启动

在已完成 baseline hard_NR 的节点执行：

```bash
/home/data/wyk/envs/diffusion_planner/bin/python /home/data/wyk/Score_Function_TwoNode_1025/scripts/launch_two_node_1025.py baselines
```

脚本在前台准备和核对输入，然后自动转入后台，不需要手动运行 tmux/nohup。每组正式评测使用八张 GPU（0–7），Ray 每个场景保留一张卡。PDM-Open 的原生推理保留其官方 CPU 实现；K20 score 在 GPU 上执行，与此前 hard_NR 协议一致。队列逐组运行，不同时启动 12 个八卡进程。

```bash
tail -f /home/data/wyk/dataset/nuplan/exp/random_baselines_1025_v1.queue.log
/home/data/wyk/envs/urbandriver/bin/python /home/data/wyk/Score_Function_TwoNode_1025/scripts/evaluate_random_baselines_1025.py status --output /home/data/wyk/dataset/nuplan/exp/random_baselines_1025_v1
```

结果目录为 `/home/data/wyk/dataset/nuplan/exp/random_baselines_1025_v1`。已完成且通过场景/配置/权重校验的任务会跳过。失败任务默认停止并保留现场；修正原因后，使用同一启动命令追加 `--retry-failed`，会保留失败尝试再重跑对应阶段。

## 节点 B 启动

节点 B 必须能访问原 S10-G 的配置、checkpoint、源码快照、原训练缓存、DP 模型、nuPlan 数据和矩阵结果，使用相同绝对路径。准备阶段会逐项检查；缺文件时停止并明确报告，不偷偷重建缓存或更换数据。无需节点 A 的 PlanTF/PLUTO 环境。

```bash
/home/data/wyk/envs/diffusion_planner/bin/python /home/data/wyk/Score_Function_TwoNode_1025/scripts/launch_two_node_1025.py energy
```

```bash
tail -f /home/data/wyk/outputs/score_aligned_1004/E10_G_aligned_1025_v1.queue.log
/home/data/wyk/envs/diffusion_planner/bin/python /home/data/wyk/Score_Function_TwoNode_1025/scripts/run_energy_aligned_1025.py status --queue-dir /home/data/wyk/outputs/score_aligned_1004/E10_G_aligned_1025_v1
```

正式训练前先用相同模型和每卡 256 的 microbatch 做八卡 NCCL、两次 Energy DSM 反向更新检查；该检查使用独立模型，不计入正式训练预算。训练源码保留原始快照；评测源码保留旧矩阵的优化版本，并核对两者的 Energy 输出与更新结果一致。

训练结果与评测队列均放在 `/home/data/wyk/outputs/score_aligned_1004/E10_G_aligned_1025_v1` 下。重新运行同一启动命令会验证并恢复同一个实验，保留训练进度和已完成的评测。不要换输出目录来“续训”。训练未完整结束或最终 EMA 校验失败时，不会开始评测。

Energy 默认评测 hard_NR，以接续已完成的 seed20260922 矩阵。若要单独在 Random 上运行同样五个 K，可使用底层准备脚本的 `--setting random_NR` 和独立输出目录；同一个已准备队列不允许中途改场景设置。

两套队列均等待当前节点 GPU 释放，不终止其他进程。后台启动成功不等于训练/仿真已经完成；查看日志与完成标记确认状态。本地测试只覆盖控制逻辑、配置、checkpoint 和 CPU 数值；真实八卡训练及 nuPlan 仿真由服务器预检/小规模试运行验证。
