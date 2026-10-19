# 仓库主线与历史分支

核对日期：2026-10-19。此文记录整理时的远端快照；后续分支可能继续前进，复核时以表中的完整提交 SHA 为准。

## 给 mentor 的入口

本次交付以 `main` 的 `b37ddf186aacfbfc1035d9ac497d17f9ea8c94f7` 为基础。
截至该快照，最新的 `codex/urbandriver-score-k20-1017` 虽然多出 65 个提交，但其
`score_function/` 和 `configs/` 与 `main` 逐文件完全相同。额外改动是 50 个研究脚本、
文档、测试和 UrbanDriver 适配文件，不是更新后的核心 Score 模型。

因此，mentor 接入 Score 模块时，从整理后的 `codex/mentor-score-handoff` 交接分支阅读接口和使用示例；需要复现某个历史实验时，
再查对应研究分支。此结论只说明代码关系，不代表已验证任意业务模型的接入效果或规划性能。

## 文件去向

| 路径 | 用途及整理决定 |
| --- | --- |
| `score_function/model/`、`score_function/loss.py` | Score/energy 网络、精炼更新和 DSM 目标，保留核心实现。 |
| `score_function/utils/` | 配置、归一化、权重、训练工具，保留与现有权重的约定。 |
| `score_function/planner/` | 现有 nuPlan / Diffusion Planner 接入，保留为参考。 |
| `score_function/train*.py`、`score_function/data_process/`、`score_function/evaluation/` | 训练、数据处理和评测流水线，保留复现能力。 |
| `configs/`、根目录训练/预处理/评测入口 | 有效配置和现有命令入口，保留。 |
| `tests/`、`.github/workflows/tests.yml` | 核心回归测试和自动检查，不能因使用小规模输入而当成临时实验删除。 |
| `scripts/` | 保留环境、训练、实验调度和诊断工具。 |
| `scripts/profile_training.py` | 独立真实数据吞吐诊断，虽不属于 mentor 首次接入流程，仍有明确用途，保留。 |
| `sim_score_planner.sh` | 参数化的官方 nuPlan 评测启动器，保留；不是一次性结果读取脚本。 |
| `nuplan_score.py` | 删除当前工作树中的一次性脚本：硬编码个人服务器路径，只读取 `S05-LN_*` 最新结果，无其他代码依赖。历史提交仍可查看。 |
| `docs/reproduction.md` | 原 README 的环境、训练、评测说明，供需要完整复现时阅读。 |
| `docs/archive/research_plan_original.md` | 原始研究计划，归档供追溯。 |
| `docs/archive/previous_projects_hashes.json` | 旧项目文件哈希，归档供追溯；不是当前交付版本的校验清单。 |
| 其他 `docs/` 文档 | 对齐协议、参数化、实验及历史验收记录；记录中的旧路径、日期和验收范围需按原文理解。 |

研究分支上的日期化评测、排队、绘图和 UrbanDriver 适配脚本继续留在原分支，避免为模块接入引入
整套历史实验目录。本次没有删除任何远端分支，也没有修改其提交。

## 分支快照

表中的名称均位于 `origin/`。`origin/HEAD` 只是指向 `origin/main`，不另算分支。

| 分支 | 完整 tip SHA | 与主线的关系及用途 |
| --- | --- | --- |
| `main` | `b37ddf186aacfbfc1035d9ac497d17f9ea8c94f7` | 本次交付基础，包含已有 score/energy 和训练对齐实现。 |
| `codex/score-energy-parameterization` | `3e6ea79f00ac0c2d7cd2a0d6750e3b0a7143fadf` | 已经 squash 合入；整个文件树与主线中的 `d165a7b` 相同。保留历史。 |
| `codex/training-progress` | `9cbb670bf7a6f51e0a9affd668f605b54c26c138` | 已经 squash 合入；整个文件树与主线中的 `a53d93f` 相同。保留历史。 |
| `codex/dit-score-branch` | `d480c9d3ea3ad0e4011a1785262b2998d258a1a0` | 主线外的独立 DiT 架构试验，有核心网络/config 改动；保留，不混入本次接口交付。 |
| `codex/test14hard-full-k-suite` | `011f70c570ad8213333d957ed380c93102765f1d` | Full K 评测套件；提交全部包含在后续研究链中。 |
| `codex/g-local-k-1012` | `b3da79e927c633dbcea93b9a78a1c5bc6a6762a1` | G/local K 后续实验；提交全部包含在后续研究链中。 |
| `codex/static-score-context-1012` | `8aacef24435e368a9c41266ce067f841ba531abe` | 静态场景与闭环轨迹诊断；提交包含在后续 sigma 和 GIF 分支中。 |
| `codex/score-context-gif-1014` | `ef29f0c8457b0f81cf1e284be0c073b12d4e6958` | 独立 GIF 可视化增量，未包含于 UrbanDriver 分支；保留。 |
| `codex/g-sigma-trials-1012` | `4685b485eb003c8e3db7e5c3ba2a7595b172bd11` | 固定 sigma 续训及迁移评测调度；提交包含在 seed/K 和 UrbanDriver 分支中。 |
| `codex/g-seed-k-1015` | `fffeb75636b3b92f4409cbedc2a475890df3182a` | 固定 S10-G 的 seed × K 评测；提交包含在 UrbanDriver 分支中。 |
| `codex/urbandriver-score-k20-1017` | `83f39cdb27b4abc5ce95ff449a46759474bd09dc` | 最新研究链，新增 UrbanDriver + S10-G K20 适配和启动器；保留作跨模型接入参考。 |

研究链的祖先关系如下；箭头只表示提交包含关系，不表示各实验已完成或结果更优：

```text
main → test14hard-full-k-suite → g-local-k-1012 → static-score-context-1012
                                                 ├→ score-context-gif-1014
                                                 └→ g-sigma-trials-1012 → g-seed-k-1015
                                                                            └→ urbandriver-score-k20-1017
main → dit-score-branch
```

`score-energy-parameterization` 和 `training-progress` 通过 squash 进入主线，所以不能仅用
“是否为 main 的祖先”判断它们是否已合入；上表使用了文件树相等的证据。

## 只读追溯示例

以下命令只展示历史内容或比较文件，不切换当前工作目录，也不修改远端。须在具有对应历史对象的
仓库副本中运行；完整 clone 会保留这些对象，只有浅克隆时可能缺失。

```bash
# 查看整理前的一次性结果读取脚本。
git show b37ddf186aacfbfc1035d9ac497d17f9ea8c94f7:nuplan_score.py

# 阅读固定版本的跨模型适配说明和实现。
git show 83f39cdb27b4abc5ce95ff449a46759474bd09dc:docs/urban_driver_score_k20.md
git show 83f39cdb27b4abc5ce95ff449a46759474bd09dc:urban_driver_score/planner.py

# 此比较在整理时为空：最新研究链没有改变这两个核心目录。
git diff b37ddf186aacfbfc1035d9ac497d17f9ea8c94f7 \
  83f39cdb27b4abc5ce95ff449a46759474bd09dc -- score_function/ configs/
```

UrbanDriver 适配可展示如何把另一个 planner 的候选轨迹转换为 Score 输入，但它仍复用训练时的
冻结 DP scene/route 编码器和 ego 归一化，且固定到特定 S10-G 权重及实验协议。
它不是任意业务模型输入均可直接运行的通用接口，也不是业务接入完成的证明。
