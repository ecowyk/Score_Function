# E10-G：对齐 S10-G 的八卡 Energy 训练与自动闭环评测

本队列从已经完成的实际 S10-G 运行读取配置和源码，不采用旧 `score_function_energy.json` 模板。

- 来源：`/home/data/wyk/outputs/score_aligned_1004/G_sigma_1012_8gpu_v1/runs/S10-G/score/`。
- 训练只改 `model.parameterization=energy` 和新输出目录。保留原来的 G 结构、冻结 DP 条件、归一化、数据划分、增强、训练 seed、优化器、LR、EMA、验证与训练预算。
- 八张 GPU，global batch 2048，每卡 microbatch 256，累积 1；947,912 个训练样本；529 epoch、244,398 updates；float32；σ=0.1。
- Energy 输出标量，score 为 `-∇E`；仍用同一 DSM 损失训练。不重训 DP，不重建缓存。
- 完成后使用最后一个 epoch 的 **EMA**，与之前 S10-G 的 `final_ema.pt` 保持一致。`last.pt` 是完整恢复断点，包含 raw 和 EMA；不会将 raw 权重或验证集选出的 `best.pt` 当成最终评测权重。
- 自动运行 **Test14-hard / NR**、推理 seed **20260922**、γ=0.1、K=5/10/20/50/100。训练 seed 不改成 20260922。
- 每次评测使用八卡 Ray，一张卡一个场景。K0 复用已有同 seed 矩阵结果，不重跑 DP。
- 训练使用原 S10-G 的精确源码；评测另行复制旧矩阵的 `no_trace` 推理源码，保持原有推理效率。准备时核验其原始 SHA，并比较两套实现的 Energy 预测和 K=0/5/20 轨迹更新，防止更换参数化时悄悄改变推理规则。

## 在第二个八卡节点启动

需要原 S10-G 完整训练目录、原 seed/K 矩阵目录、数据缓存与数据集仍以原绝对路径可访问；不允许通过静默替换路径或重新采样来“对齐”。先执行准备检查，再启动训练和评测串联队列：

```bash
cd /home/data/wyk/Score_Function_TwoNodes_1025
/home/data/wyk/envs/diffusion_planner/bin/python -u scripts/prepare_energy_aligned_1025.py
/home/data/wyk/envs/diffusion_planner/bin/python -u scripts/run_energy_aligned_1025.py run
```

将最后一条放在 tmux 中长期运行。准备阶段只做 CPU 检查，核验原始最终 EMA、训练缓存、原 DP/normalizer、训练预算及 seed20260922 已完成的 K0/5/10/20/50/100 结果；用原始训练源码做两步 Energy 二阶反向检查，以及 `inference_mode` 推理检查。缺少依赖或原始文件会明确报错，不会悄悄换成另一个实验。

正式训练前，队列还会用八卡 NCCL 对实际 G-energy 模型做两步合成数据更新，每卡 microbatch256，检查所有参数梯度有限、参数实际更新且各 rank 同步。这个检查不读写正式训练 checkpoint，不占正式预算；通过记录与配置/源码/plan 的 SHA 绑定，恢复时跳过已经通过的检查。

目录默认为：

```text
/home/data/wyk/outputs/score_aligned_1004/E10_G_aligned_1025_v1
```

`--root`、`--queue-dir`、`--source-queue`、`--matrix` 可显式指定。默认评测 hard_NR；只有明确需要时才在准备时用 `--setting random_NR`，它也要求相应旧矩阵对照已经完成。

## 状态和恢复

```bash
/home/data/wyk/envs/diffusion_planner/bin/python -u scripts/run_energy_aligned_1025.py status
```

出错修复后，运行同一条 `run` 命令。从已有完整 `last.pt` 恢复训练；训练已完成就核验并跳过；已完成的评测逐项验证后跳过。不要重新 prepare 或换目录重训，也不需要恢复此前暂停的 S025-G。该队列不停止其他任务，八卡有占用时等待。

输出：

- `plan.json`、`preflight.json`：完整配置、源码 SHA、训练预算与对照记录。
- `runs/E10-G/score/last.pt`：完整恢复断点。
- `runs/E10-G/score/final_ema.pt`：最后 epoch 的 EMA 权重。
- `logs/E10-G.launcher.log`：训练日志。
- `logs/energy_ddp_smoke.log`、`energy_ddp_smoke.json`：八卡二阶反向检查日志与通过记录。
- `evaluation/jobs/`、`evaluation/logs/`：逐项闭环状态与日志。
- `evaluation/energy_vs_direct.csv`：同 seed、同场景的 DP K0 / S10-G / E10-G 得分和差值。
- 原生 nuPlan 目录仍保存 metrics、simulation 与 NuBoard 文件。

本地测试不代表服务器上已完成训练或闭环。真正完成必须通过训练预算、checkpoint 和完整场景结果检查；程序只在全部五项评测通过后将 pipeline 标记为 complete。
