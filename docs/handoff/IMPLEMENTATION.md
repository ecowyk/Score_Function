# 实现、训练与运行版本

核对日期：2026-11-02。核心代码依据本次变更前的 `main@65524db113a4684e8dc376c841be4c8640156a0a`；具体训练 run 仍以其冻结源码及配置为准。

## 数据流与张量

nuPlan 提供日志数据库、地图和场景抽取接口。DP 的输入是从观测整理的 ego/邻车状态与历史、静态物体、车道/地图、路线等结构化张量，不是直接输入图像。具体字段、padding、时序长度必须核对相应 DP 特征处理代码，不从张量形状猜语义。

| 对象 | 当前接口约定 |
| --- | --- |
| 候选 ego 轨迹 | `[B,80,4]`，未来 0.1–8.0 s，步长 0.1 s |
| 每点通道 | ego 局部坐标中的 `(x,y,cos(theta),sin(theta))`，再按 DP ego normalizer 归一化 |
| 当前时刻 | 不包含在被优化的 80 个点中；nuPlan 输出可在前面加当前状态，得到 81 个 sampled states |
| 场景条件 C | 现有 DP 条件常见为 `[B,107,192]`；代码接口允许 N 个 scene token |
| 路线条件 R | `[B,192]` |
| 输出 | 同形状 `[B,80,4]`，归一化坐标下的 score |
| 可选邻车未来条件 | 类接口支持；S10-G 明确关闭 `neighbor_future` |

“不使用 predicted-neighbor future”不等于“不使用邻车信息”：邻车观测/历史仍可进入 scene encoder。只有 ego future 被更新；当前 ego state 和其他车辆的预测输出不属于该优化变量。

其他 baseline 迁移流程：原生 planner 输出 → 正确的时间采样/坐标/heading 转换 → DP ego 归一化 → 冻结 S10-G → 反归一化 → 原生 nuPlan trajectory 转换。条件继续由训练时匹配的 DP encoder 提供，不需要为这些 baseline 再采样一条 DP 轨迹。

## 主网络

`ScoreFunctionBranch` 是可训练的独立分支，冻结 DP 场景和路线编码器。默认隐藏维 192、6 heads、dropout=0.1；实际值必须以 run 的 `score/config.json` 核查。

1. 每个轨迹点由 Linear(4,192) 编码，加可学习 temporal position 和投影后的 route embedding。
2. 两个 TemporalResidualBlock，默认 dilation 为 1、2。
3. 对冻结 scene token 做 cross-attention。
4. 可选 predicted-neighbor attention；S10-G 关闭。
5. **G 结构启用轨迹时间维 self-attention**，使各未来轨迹点相互交换信息。
6. 两个 TemporalResidualBlock，默认 dilation 为 2、4。
7. LayerNorm → Linear → GELU → direct-score head，输出 4 维；Energy head 输出逐点标量并求和。

Temporal position 指轨迹中的未来采样位置，不是扩散时间 t。该分支不接收 diffusion timestep，也不直接复用带 t 的 DP decoder latent。`codex/dit-score-branch` 存在，说明做过候选实现；不代表当前 S10-G 使用 DiT 或该候选已得到最新闭环验证。

local 基型的 backbone 感受野为 37 点；启用 G attention 后可覆盖 80 点。历史文档中的 **1,387,204 参数属于 local 基型**，不能直接当作 S10-G 参数量；准确数量读取该 run 的 `model_info.json` 或按配置统计。

## DSM 与归一化

训练在完整归一化 ego future 上加固定尺度高斯噪声，`fixed_sigma_dsm_loss` 为 `(sigma * score + epsilon).square().mean()`。不以 projection 改写训练噪声。sigma 是该 run 的训练定义，不能换一个推理 sigma 就声称得到另一噪声尺度的已训练 score。

`refine_ego` 每步重算 score，做 `gamma * sigma**2` 缩放，并可在物理 heading 表示中投影。K0/gamma0 的上层 adapter 应直接返回原始物理轨迹，避免归一化往返造成基线变化。`record_trace` 和性能优化副本可能不同，实际 batch 使用的 `no_trace` 快照要由 plan 定位。

归一化坐标的 score 不能不经变换直接当作以米或弧度为单位的物理梯度。cos/sin 通道的噪声标准差也不能直接称为角度标准差。

## S10-G 训练协议

以下字段在固定 sigma 实验准备脚本、Energy 对齐文档中有依据（S04/S05），是已知运行协议；不是当前模板文件的所有默认值。

| 字段 | 已知协议 |
| --- | --- |
| 参数化 | direct score |
| 条件与结构 | 冻结 DP encoder，G temporal attention，关闭 predicted-neighbor future |
| 训练样本 | 947,912；准确缓存/manifest 身份由 run 提供 |
| 有效 global batch | 2048 |
| 每卡 microbatch | 256 |
| 八卡梯度累积 | 1；原 S05-G 四卡阶段为 2 |
| 每 epoch updates | 462 |
| 完整预算 | 529 epochs / 244,398 updates，满足至少 500 epochs 且至少 244,000 updates |
| S10-G sigma | 0.1 |
| 优化器协议 | AdamW，lr=5e-4，weight_decay=0.01，grad_clip=5；5 epoch warmup 后恒定 LR |
| EMA | 0.999；具体实现与恢复状态以源码/配置为准 |
| 数据增强 | 对齐后启用原 DP StatePerturbation；同步处理观测与目标，再用冻结 encoder 编码 |
| 正式矩阵权重 | 最终 epoch EMA `final_ema.pt` |
| 训练 seed | 从原 run 配置继承，不能用评测 seed=20260922 替换 |

早期文档既包含“关闭增强”的初稿，也包含后来的对齐修订；当前 G 准备脚本明确要求 `data_augmentation.enabled=true`。不能仅阅读历史文档靠前的一段来断定实际协议。

训练内部 validation 是训练数据的记录级划分，不等于官方 Val14 闭环集。数据文件数量或“百万帧”描述不证明与 DP 完全相同；应继续核对 allowlist、manifest、预处理与增强。

## checkpoint 的三个角色

| 文件 | 用途 |
| --- | --- |
| `last.pt` | 完整恢复状态：raw/EMA、优化器、RNG、训练游标等 |
| `best.pt` | 按内部验证 DSM 选择的权重；可用于某些部署场景，但不自动等于主矩阵权重 |
| `final_ema.pt` | 最后一个完整训练 epoch 的 EMA；S10-G / E10-G 正式对照使用它 |

S05-G sigma=0.05，529 epochs；S05-G_plus100 延长到629 epochs；S10-G sigma=0.1，从头训练到529 epochs；S025-G sigma=0.025 的历史暂停位置是 epoch71/step33000。不能在恢复时新建同名实验代替续训。

E10-G 应从实际 S10-G 配置和源码快照仅切换 energy 参数化及输出位置。它保留 DSM、数据、训练 seed、预算、EMA和更新规则。使用 `-grad(E)` 时推理也需要内部启用 autograd；训练则需要二阶反向路径，不能简单套用 direct-score 的无梯度实现。

## 代码入口

| 入口 | 作用 |
| --- | --- |
| [score_branch.py](../../score_function/model/score_branch.py) | direct score / energy 网络 |
| [refinement.py](../../score_function/model/refinement.py) | 固定 K 更新 |
| [loss.py](../../score_function/loss.py) | 固定 sigma DSM |
| [normalizer.py](../../score_function/utils/normalizer.py) | 归一化、heading 投影 |
| [checkpoint.py](../../score_function/utils/checkpoint.py) | 权重与协议核验 |
| [train.py](../../score_function/train.py)、[train_epoch.py](../../score_function/train_epoch.py) | 训练组织与单 epoch |
| [online_training.py](../../score_function/data_process/online_training.py) | 在线增强/条件处理 |
| [planner.py](../../score_function/planner/planner.py) | nuPlan / DP planner 接入 |
| [业务接入指南](../integration.md) | 接口复用与 DP 条件依赖 |
| [训练复现](../reproduction.md)、[对齐协议](../alignment.md) | 完整运行入口及历史修订 |

评测 adapter 和队列并不全部存在于 main。使用 [SOURCES.md](SOURCES.md) 的 commit 链接，以及 `plan.json` 内的 `frozen_score_code`、`sources`、`inputs` 找到真正参与运行的代码。
