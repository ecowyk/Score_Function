# 进度快照与下一步

**快照日期：2026-11-02（Asia/Shanghai）。这是交接记录，不是实时运行监控。** 本次没有连接实验服务器。

## 状态表

| 工作 | 已知状态 | 尚需核实的内容 |
| --- | --- | --- |
| S10-G direct-score | 已报告529 epoch最终EMA训练完成；作为主 refiner 使用 | 取回当前使用的配置、checkpoint SHA、provenance |
| DP 多 seed × K | 有 hard-NR、random-NR 历史矩阵；K20 为主设置 | 完整 hard 矩阵、random K100 最终状态，不能沿用旧 pending 当实时状态 |
| 多 baseline hard-NR | 有 DP / UrbanDriver / PlanTF17/24 / PDM-Open / GC-PGP / PLUTO 的历史结果 | 每组原始结果目录、seed、checkpoint、完整配对表 |
| E10-G 对齐训练 | 10月28日历史报告：529 epochs、244398 steps、56.05h 完成 | 最终训练回执和权重；K5/10/20 的数值 |
| E10-G 评测 | 后续讨论记录在K20后停止、保留K5/10/20；K50/100未完成 | 以 `evaluation/jobs/` 和 `energy_vs_direct.csv` 核验，不能重启整轮训练 |
| Flow 旧 Native | 72.8647/100、272/272；仓库文档标记CFG1.0 | 后续CFG1.8完整NR/R分数尚未在此交接包补齐 |
| Flow native/wrapper parity | 用户贴出单真实场景首帧通过记录 | 不覆盖整段闭环、全部场景或官方论文协议 |
| Flow noise-aligned | 11月2日有隔离实验脚本，修改噪声chunk构造 | 没有可核验的最终aligned分数；修改尚未得到作者确认 |
| baselines8 v4 | 源码确认68项；复用原准备计划 | 服务器当前完成数、失败项、配对状态 |
| parallel128 | 脚本确认4×2GPU×16场景/GPU；用户接受该配置 | 是否已成功接管、实际吞吐/显存/完成状态 |
| NAVSIM | 已讨论迁移和embedding归因 | 尚无已验证的本项目训练或评测结果 |

## 最新已知讨论与后续安排

11月2日用户决定先补 nuPlan 实验，暂不继续深挖 Flow 与官方数值的差异。同日曾有“17/68完成，`val_NR_gc_pgp_K20`到908/1118”的历史队列描述；之后生成了 parallel128 调度脚本。**这个进度数字已可能过时，不应显示成当前运行状态。**

当前可以从 v4 脚本确认的68项顺序：Flow hard-NR的K0/K20、Flow random-NR的K0/K20，然后八个配置分别在val-NR、hard-R、random-R、val-R上跑K0/K20。较早讨论中的66项不含后来补入的Flow hard-NR两项，不能继续用66作为v4总数。

并行调度器按这个列表取待完成任务，但多个任务可以同时执行，因此启动/完成的时间顺序不再等同于串行序号。

## 当前优先级

1. **读现有状态，继续已授权的 nuPlan 补测。** 从同一输出目录恢复；先确认是否已有调度器，不重复启动。
2. **汇总完整结果。** 同步 `summary.csv`、失败清单、逐场景配对结果，补上本交接包缺失的seed、权重、路径和完成数。
3. **补录 Energy 对照。** 获取已完成K的结果再判断是否需要后续实验；不要把已完成训练当作待从头执行。
4. **分析收益边界。** 重点包括GC-PGP下降、hard-R未改善，以及不同场景的K敏感性；没有明确证据前不预设原因。
5. **推进 NAVSIM 设计。** 固定版本、baseline、条件表示和公平对照；重新训练。若用户改变优先级，记录原因和日期。

以上是已知目标的延续，不要求自动启动所有待办。新agent应明确认领其中一项；其它任务可以继续由另一个agent负责。

## Flow parity 的已报告范围

用户贴出的JSON：`trajectories_equal=true`，`sampled_states=81`，timestamps一致，xy最大绝对误差0米，heading最大误差0弧度，容差1e-6，initial/final RNG一致，native/wrapper CFG均为1.8，seed20260922，timestamp_us=1621960740300132，`scope=one_real_scene_first_frame_only`，`state=complete`。

这里的81包括当前状态，不改变score只优化80个未来点的接口。该记录支持该场景首帧等价；不能证明全部分数差异仅来自seed。用户后来描述val较接近、hard-NR偏低、test-NR偏高，但未在可用记录中给出可精确入表的全部数值。

## 下一次更新应取得的材料

- baselines8 v4 的 `plan.json`、`status.json`、`summary.csv`、`failed_jobs.csv`、`jobs/*.json`、`*/*/pairing.json`。
- parallel128 的 `manifest.json`、`status.json`、`queue.log`，以及任务对应 `resources` 记录。
- E10-G 的 `evaluation/energy_vs_direct.csv`、训练完成回执及评测job记录。
- Flow 的实际 `parity.json`、`aligned.complete.json`（若存在）及相应配置。
- S10-G/E10-G 的原始 `config.json`、`model_info.json`、`provenance.json`、checkpoint哈希；不必把大权重提交到Git。
- 各活跃worktree的 `git rev-parse HEAD` 与工作区修改状态，建立服务器目录到commit的真实映射。
