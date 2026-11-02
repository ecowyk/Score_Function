# Score Function 科研项目交接入口

**整理日期：2026-11-02（Asia/Shanghai）。** 目标读者是第一次接手项目、看不到原始聊天记录的研究或工程 agent。

本次核对了 GitHub 分支、核心源码、训练/评测文档、2026-10-28 v4 队列包和 2026-11-02 两个脚本。实验分数主要来自用户此前报告；本次没有连接实验服务器、读取完整运行结果或重新执行训练/闭环仿真。文中每类证据的边界见 [SOURCES.md](SOURCES.md)。

## 先理解这件事

项目研究：**学习专家轨迹的条件局部密度方向，从现有 planner 输出的轨迹出发进行确定性 score refinement，检验这个数据驱动先验能否改善闭环规划。**

起点可以来自 DP、UrbanDriver、PlanTF、PDM-Open、GC-PGP、PLUTO 或 Flow。目标是当前模式内可达的局部高密度区域；不寻找全局 MAP，不主动跨模式切换。固定 K 的实现没有到达密度峰或不跨模式的数学保证。最终以 nuPlan 闭环指标为准；NAVSIM 是下一阶段迁移方向。

当前主协议是 **S10-G 最终 EMA，sigma=0.1、gamma=0.1、K=20**。后续评测批次统一 seed=20260922，Flow CFG=1.8；DP 多种子矩阵另有 20260920–20260924。不能把这些后续设置补填到缺少配置证据的历史结果中。

## 阅读顺序

| 次序 | 文档 | 应掌握的内容 |
| --- | --- | --- |
| 1 | [RESEARCH.md](RESEARCH.md) | 为什么做、数学定义、研究演变、结论边界 |
| 2 | [IMPLEMENTATION.md](IMPLEMENTATION.md) | 输入输出、网络、DSM、归一化、训练及权重选择 |
| 3 | [EXPERIMENTS.md](EXPERIMENTS.md) | 已报告结果、负结果、公平比较规则 |
| 4 | [STATUS.md](STATUS.md) | 截至整理日的状态、缺口、优先级 |
| 5 | [OPERATIONS.md](OPERATIONS.md) | 工作目录、队列范围、只读核查与恢复注意事项 |
| 6 | [DECISIONS.md](DECISIONS.md) | 已作出的决策及原因 |
| 7 | [SOURCES.md](SOURCES.md) | 完整 commit、文件来源、证据等级、尚未取回的材料 |
| 8 | [COLLABORATION.md](COLLABORATION.md) | 多 agent 分工、更新方式和接手任务模板 |

机器可读记录：[历史结果](EXPERIMENTS.csv)、[Random-NR 多种子快照](DP_RANDOM_NR_20261014.csv)、[分支快照](branch_snapshot.json)、[脚本快照及校验值](snapshots/README.md)。

## 不要带着这些误解开始工作

1. **少步坍塌不是 DP 上已证实的核心问题。** Toy 中观察到的现象不能直接迁移成 DP 的实验结论。
2. **Score 不是闭环 reward 的梯度。** DSM 改善、ADE 改善、密度上升、闭环得分改善是不同命题。
3. **目前有正结果也有负结果。** NR 上多项改善，但 GC-PGP 的已报告结果下降，DP hard-R 也没有改善。
4. **S10-G 不是仓库默认 JSON。** 默认模板的 local temporal / sigma=0.05 / K=5 不等于当前主实验配置。
5. **DP 条件分支仍有依赖。** 迁移到其他 planner 时复用冻结 DP 的场景/路线编码器与归一化，不代表已经移除了 DP 条件依赖。
6. **72.8647 是旧 Flow CFG=1.0 结果。** 后续 CFG=1.8、单场景首帧 parity、noise-aligned 变体须分别记录。
7. **代码存在不等于实验完成。** 68 项队列和 128 并发配置已能从脚本确认，实时完成数需要服务器回执。
8. **K20 是经验主设置。** 它不是所有场景、所有 baseline、所有 seed 都逐项最优的定理。

## 仓库与运行环境的关系

`main@65524db113a4684e8dc376c841be4c8640156a0a` 是本次文档变更的基底，是 10 月 19 日整理后的核心接入版本。后续 baseline 适配和实验调度分布在其他分支及服务器工作目录。核对到的 main 与 Flow 分支 `score_branch.py` blob 一致，但不能把这一点推广成整个目录或所有运行快照完全相同。

代码、权重、配置和结果必须作为一组定位。GitHub 不含 nuPlan 数据集、训练 checkpoint、完整队列 plan、完整闭环日志和 parquet；仅克隆此仓库无法恢复服务器实验状态。

## 新 agent 的第一项任务

先用自己的话复述：研究问题、当前方法、支持/反对的证据、尚未解决的问题、拟接手任务。随后按任务核对相应源码与实验记录。

若负责补实验，先读取现有 plan、status、job receipts 和 summary，确认任务所有者及当前状态，再决定是否恢复。若负责方法分析，先阅读已完成的正负结果及限制，不把早期计划中的待办默认视为现阶段必须执行的任务。
