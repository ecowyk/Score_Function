# 来源、版本与证据边界

核对日期：2026-11-02，时区Asia/Shanghai。仅包含本研究相关信息。

## 证据等级

| 等级 | 含义 | 本次能支持的结论 |
| --- | --- | --- |
| source_verified | 本次读取具体commit的代码/文档或完整脚本 | 代码定义、配置约束、接口、调度范围；不证明服务器执行成功 |
| user_reported | 用户此前贴出的结果或项目讨论中的确认记录 | 历史报告的数值/意图；没有本次重新核算原始parquet |
| historical_status | 某个时间点的运行进度/完成描述 | 仅该时间点状态；不自动延续到现在 |
| hypothesis | 待检验解释或建议 | 不能当作观测事实 |
| unknown | 尚未取得或存在冲突 | 保留空值和补证任务 |

来源优先级应按问题选择：用户最新明确决定用于研究目标/优先级；某次run的冻结plan、代码、配置、checkpoint用于“实际做了什么”；完整结果和回执用于“实际得到什么”。旧聊天、旧文档和当前main的默认值都不能覆盖某次run的真实证据。

## 已核对的来源

| ID | 来源 | 核对范围 |
| --- | --- | --- |
| S01 | [main@65524db…](https://github.com/ecowyk/Score_Function/tree/65524db113a4684e8dc376c841be4c8640156a0a) | 根README、repository_map、experiments、score_branch、refinement、loss及文件树 |
| S02 | [score_branch.py](https://github.com/ecowyk/Score_Function/blob/65524db113a4684e8dc376c841be4c8640156a0a/score_function/model/score_branch.py) | direct/energy接口、G attention、heading之外的网络定义 |
| S03 | [refinement.py](https://github.com/ecowyk/Score_Function/blob/65524db113a4684e8dc376c841be4c8640156a0a/score_function/model/refinement.py)、[loss.py](https://github.com/ecowyk/Score_Function/blob/65524db113a4684e8dc376c841be4c8640156a0a/score_function/loss.py) | 固定K更新、投影、DSM |
| S04 | [G sigma训练说明](https://github.com/ecowyk/Score_Function/blob/4685b485eb003c8e3db7e5c3ba2a7595b172bd11/docs/g_sigma_trials_1012.md)、[准备脚本](https://github.com/ecowyk/Score_Function/blob/4685b485eb003c8e3db7e5c3ba2a7595b172bd11/scripts/prepare_g_sigma_trials.py)、[闭环说明](https://github.com/ecowyk/Score_Function/blob/4685b485eb003c8e3db7e5c3ba2a7595b172bd11/docs/g_sigma_closed_loop_1014.md) | S05/S10/S025来源、G结构、训练预算、final EMA、暂停/恢复协议 |
| S05 | [Energy对齐](https://github.com/ecowyk/Score_Function/blob/7f3df70601c8a6b1f4c352b50a70e70687fde8dd/docs/energy_aligned_1025.md)、[两节点安排](https://github.com/ecowyk/Score_Function/blob/7f3df70601c8a6b1f4c352b50a70e70687fde8dd/docs/two_node_1025.md) | E10-G公平对照与输出路径；不证明所有评测已完成 |
| S06 | [Flow协议](https://github.com/ecowyk/Score_Function/blob/73f47018a7fca2b265b4b74f5a722d989b42cd97/docs/flow_score_queue.md) | 官方代码/资产pin、CFG1.0与1.8、旧72.8647、RNG策略、4步midpoint |
| S07 | [v4脚本快照](snapshots/baselines8_v4/scripts/evaluate_baselines_val_reactive_1028.py)及同目录依赖 | 读取原 `baselines8_flowcfg18_seed20260922_1028_v4.tar.gz` 后逐文件保存；确认68项 |
| S08 | [parallel128脚本](snapshots/2026-11-02/run_baselines_parallel_1102.py) | 读取11月2日创建的完整版本；确认profile、接管、恢复和状态行为 |
| S09 | [Flow诊断脚本](snapshots/2026-11-02/run_flow_reproduction_1102.py) | 读取11月2日修改后的完整版本；确认parity与noise-aligned隔离设计 |
| S10 | [random-baselines前置脚本](https://github.com/ecowyk/Score_Function/blob/7f3df70601c8a6b1f4c352b50a70e70687fde8dd/scripts/evaluate_random_baselines_1025.py) | 核对VARIANTS及K0/K20前置队列定义 |
| U01 | 用户已确认研究定位，尤其2026-10-13/14相关讨论 | 局部模式内目标、非RL、以闭环为准；并非全局MAP |
| U02 | 用户固定sigma比较输出（10月训练/评测讨论） | DP75.5216、S05-G76.9929、续训76.7071、S10-G77.1190；seed原始配置本次未取回 |
| U03 | 用户2026-10-14 random-NR矩阵 | 5个seed×6个K的原始显示值；K100为当时失败/待完成 |
| U04 | 多baseline及R集项目进度记录，至2026-10-25前后 | UrbanDriver、PlanTF17/24、PDM、GC-PGP、PLUTO及DP-R分数；回执/元数据待补 |
| U05 | 用户贴出的Flow首帧parity JSON及复现讨论 | scope明确为单真实场景首帧；本包记录原数值，不扩大解释 |
| U06 | 2026-10-28 Energy训练完成及停止评测的讨论记录 | 529epoch/244398step/56.05h、K20后停止；最终CSV未取得 |
| U07 | 2026-11-02补测与资源安排讨论 | 暂缓深挖Flow差异、旧队列17/68历史描述、接受128并发；不证明新调度器已运行成功 |
| U08 | 2026-10-28 NAVSIM讨论 | 重新训练、encoder/embedding与多帧的归因关注，未作最终baseline选择 |

U类是本交接从已有项目上下文整理的证据索引，并非伪造的公开论文引用；没有可公开定位的聊天永久链接。后续用原始结果文件补证时，应新增来源并提升对应记录的证据等级，不删除这层来源历史。

## 分支快照

完整17个分支的名字和SHA见 [branch_snapshot.json](branch_snapshot.json)。它们是2026-11-02读取的远端HEAD，未来可能前进；引用方法时使用完整SHA。

| 用途 | 分支 | 核对的HEAD |
| --- | --- | --- |
| 核心接入 | main | `65524db113a4684e8dc376c841be4c8640156a0a` |
| 早期mentor交接 | codex/mentor-score-handoff | `d62cebaf40af164ee156f0eadcabadcf18f71767` |
| G固定sigma | codex/g-sigma-trials-1012 | `4685b485eb003c8e3db7e5c3ba2a7595b172bd11` |
| seed/K矩阵 | codex/g-seed-k-1015 | `fffeb75636b3b92f4409cbedc2a475890df3182a` |
| 场景GIF | codex/score-context-gif-1014 | `ef29f0c8457b0f81cf1e284be0c073b12d4e6958` |
| UrbanDriver | codex/urbandriver-score-k20-1017 | `83f39cdb27b4abc5ce95ff449a46759474bd09dc` |
| PDM / GC-PGP | codex/pdm-gcpgp-score-k20-1021 | `561ff56edf614933efa290cb5ee1cbd6fb6ce328` |
| PLUTO | codex/pluto-score-k20-1025 | `f8a159a1babda502e12e34b162d904cdf7f1063f` |
| Random / Energy | codex/random-baselines-energy-1025 | `7f3df70601c8a6b1f4c352b50a70e70687fde8dd` |
| Flow初始集成 | codex/flow-score-k20-1028 | `a4b6fa31e0954c9eb93a286ab88090967245a354` |
| Flow CFG1.8 | codex/flow-cfg18-repro-1028 | `73f47018a7fca2b265b4b74f5a722d989b42cd97` |
| DiT候选 | codex/dit-score-branch | `d480c9d3ea3ad0e4011a1785262b2998d258a1a0` |

仅核对“分支存在/HEAD”不表示逐行审查其全部内容。main与Flow分支的 `score_branch.py` blob相同；未据此宣称其他文件也一致。服务器worktree可能有cherry-pick或未提交补丁，必须独立核对。

## 官方依赖的已记录版本

S06中记录Flow官方源码 `366df7654d7b7fa77f9d2577279da63127c6d2f8`，HF资产revision `b627934a8db3021892afd13f5278ba0149532efc`。这些是现有协议文档的pin，本次未重新下载官方权重。

Flow官方来源：[DiffusionAD/Flow-Planner](https://github.com/DiffusionAD/Flow-Planner)。DP已有文档的官方引用：[ZhengYinan-AIR/Diffusion-Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner/tree/a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df)。运行时绑定的DP源码/权重可能需从provenance进一步确认；不要把文档引用自动当作所有实验实际HEAD。

## 本包明确没有覆盖

完整服务器实时状态、大权重与原始数据、所有历史聊天逐字稿、完整文献综述、每轮训练的全量配置和结果parquet。现有材料不足以补全的内容已集中在[STATUS.md](STATUS.md)。此包的用途是让agent理解现有工作并知道如何补证，不是假定已拥有整个服务器的镜像。
