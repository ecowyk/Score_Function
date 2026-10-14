# 暂停 S025-G，评测已完成的两组 G

使用 `scripts/evaluate_g_sigma_1014.py`，复用已有 `full_test14hard_suite.py`、原 sim 脚本与冻结的 `code/lg`。不重启或修改训练配置，不重建数据缓存。

## 本轮内容

| 模型 | 权重 | K | 是否新跑 |
|---|---|---|---|
| DP | 原基线 | 0 | 复用原结果 |
| 原 S05-G | 529 epoch 最终 EMA，sigma=0.05 | 20 | 复用原结果 |
| S05-G_plus100 | 629 epoch 最终 EMA，sigma=0.05 | 20 | 新评测 |
| S10-G | 529 epoch 最终 EMA，sigma=0.10 | 20 | 新评测 |

采用完整 test14-hard 272 场景、closed_loop_nonreactive_agents、八卡 Ray，gamma=0.1。seed 和 DP 噪声生成规则保持原套件设置。闭环状态分叉后，DP 的场景条件和轨迹也会变化，固定 seed 不意味着所有模型整段闭环的 DP 轨迹完全相同。

各模型推理时使用自己的训练 sigma。更新公式为 `x += gamma * sigma**2 * score`，因此固定 K/gamma 的结果反映整套模型与更新规则的效果；不能仅凭两组分数把变化全部归因于模型拟合质量。

## 暂停与启动

先在八卡实验 worktree 中获取新脚本，再执行：

```bash
(
set -euo pipefail
cd /home/data/wyk/Score_Function_G_sigma_8gpu_1012
/home/data/wyk/envs/diffusion_planner/bin/python -u scripts/evaluate_g_sigma_1014.py pause-training
/home/data/wyk/envs/diffusion_planner/bin/python -u scripts/evaluate_g_sigma_1014.py prepare
tmux new-session -d -s score_G_eval_1014 'exec /home/data/wyk/envs/diffusion_planner/bin/python -u /home/data/wyk/Score_Function_G_sigma_8gpu_1012/scripts/evaluate_g_sigma_1014.py run >> /home/data/wyk/dataset/nuplan/exp/score_G_sigma_closed_loop_1014_v1/queue.log 2>&1'
)
```

暂停命令先检查可恢复断点，停止该训练队列的协调器，再等下一次原子写入的 `last.pt`（最多 180 秒），之后仅终止该训练任务的进程。若等待超时，仍使用最近已经保存的断点。打印 `PAUSED` 后会给出真实保存的 epoch/step，记录在训练队列的 `pause_record.json`。

该操作不等于立即保存当前内存中的每一步；保存之后、进程退出之前的少量未保存更新，恢复时会重新执行。权重、优化器、EMA、随机状态和采样游标均由已有完整断点恢复。

准备阶段验证两个模型的最终 EMA、训练完成标志、DP 身份、模型实现、normalizer 和 seed。原 DP/G20 结果会重新校验 272 场景完整性后复用。新评测期间持有训练队列锁，阻止恢复训练争抢 GPU。

## 查看进度与结果

```bash
watch -n 2 /home/data/wyk/envs/diffusion_planner/bin/python /home/data/wyk/Score_Function_G_sigma_8gpu_1012/scripts/evaluate_g_sigma_1014.py status
```

状态显示与之前完整闭环套件一致，包含当前任务、日志位置、运行进度和完成得分。退出 watch 不会终止评测。

队列与汇总：`/home/data/wyk/dataset/nuplan/exp/score_G_sigma_closed_loop_1014_v1/`，其中 `summary.csv` 保存四组结果。实际仿真、metrics 和 NuBoard 文件继续使用原 nuPlan 输出布局：`dataset/nuplan/exp/exp/simulation/closed_loop_nonreactive_agents/score_function/test14-hard/`。每组的精确结果目录见 `jobs/<任务名>.json` 的 `result_dir`。

## 评测后恢复 S025-G

确认评测结束后运行原队列，不要再次 prepare 或 fork：

```bash
tmux new-session -d -s score_G_sigma_8gpu_1012 'exec /home/data/wyk/envs/diffusion_planner/bin/python -u /home/data/wyk/Score_Function_G_sigma_8gpu_1012/scripts/queue_g_sigma_trials.py run >> /home/data/wyk/outputs/score_aligned_1004/G_sigma_1012_8gpu_v1/queue.log 2>&1'
```

队列会验证并跳过已完成的前两组，然后从 S025-G 最近保存的 `last.pt` 续训，不会从第 0 个 epoch 开始。本轮不自动恢复训练，方便先查看闭环结果。
