# 新方案的方法审阅与复现协议

日期：2026-09-24。方法 `score_function_v1`，包 `score_function`，版本0.1.0。

当前新增的八组训练入口是 `scripts/launch_experiments.sh`，完整协议、GPU 分配、
环境安装和恢复方法见 [Eight-GPU Training Suite](experiment_suite.md)。
该入口实现四个 σ、同参数量 RF77、全局时间 attention 和预测邻车条件的对照；
以下内容保留为基础方法的设计说明。执行新入口会启动八组训练，不自动启动评测。

## 审阅结论

用户给出的 Phase1 方案没有阻碍实现的根本矛盾，可以实施。该方案与此前冻结 DiT 后克隆输出 head 有实质区别：新分支的整个轨迹表示可以学习，并且不读取带 diffusion time 的 DiT latent。只共享不接收 diffusion time 的 scene encoder 和 route encoder。

保留原文作为规格快照，而不是把文档中的后续实验清单视为本次自动执行命令。本次实现主分支、训练、离线诊断、refinement 和可选 nuPlan 接口；不启动公司服务器任务，也不自动进行消融、参数扫描或正式 benchmark。

需修正原文中可能引起误解的三种说法：

1. `-epsilon/sigma` 是单次 corruption 的条件监督标签；真实边缘 score 是其在给定 noisy trajectory / C / R 下的条件期望。cosine 和恢复率作为去噪诊断，不能命名为真 score 方向准确率、真实密度增量或峰值命中率。实际数据 DSM 存在标签条件方差，不以 loss必须趋零/cosine必须趋一为收敛标准。
2. 四个两层kernel3卷积块的dilation为1、2、2、4，候选轨迹感受野为 `1+4*(1+2+2+4)=37` 点。场景 cross-attention 按各 query 读取固定场景，不额外交换轨迹 query 信息。保留指定主模型，并暴露 dilation 配置；不宣称每个输出坐标依赖完整80点。
3. 开启 heading 重归一化后是带表示投影的 refinement，不再是纯四维欧氏空间的无约束 score ascent。不能据此保证每步增加 p_sigma。训练噪声、DSM、Tweedie一步恢复均不投影，推理投影是显式可关闭选项。

## 固定实现

截至本次核验，官方 `main` 解析为 `a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df`。读取官方 encoder、decoder、normalizer、train_epoch 和 planner 接口，不修改这些文件。运行时绑定实际 checkpoint、args 和官方 Python 源码的SHA256，因此本地修改后的不同base不能静默共用缓存和权重。

数据从官方NPZ输入字段构造观测，按官方ObservationNormalizer归一化；专家未来heading转cos/sin。关闭StatePerturbation，不生成邻车未来用于训练，也不增加自定义安全/舒适度loss。

score变量只含ego future80点，每点4维，不含当前点。ego normalizer显式取 `[agent=0, future-broadcast=0, :]`，避免误广播到11agents。C为 `[B,107,192]`、R为`[B,192]`。缓存只运行原encoder和route encoder，不经过DPM采样；无需为训练生成planner起点。

模型主分支为：

```text
Linear(4,192) + learned zero-initialized temporal position + route
→ TemporalResidualBlock(dilation1)
→ TemporalResidualBlock(dilation2)
→ residual scene cross-attention (6 heads) + residual FFN(192→768→192)
→ TemporalResidualBlock(dilation2)
→ TemporalResidualBlock(dilation4)
→ LayerNorm / Linear192→192 / GELU / Linear192→4
```

每个temporal块是LayerNorm、两个非因果kernel3卷积、GELU、dropout及residual。dropout=.1。总参数1,387,204；所有新分支参数训练，原planner所有参数冻结。未来位置embedding指轨迹采样位置，不是diffusion time。无sigma/t/SDE输入，无原x_start到score转换，无DiT权重克隆。

公开接口：

```python
score = branch(ego_traj_norm, scene_context, route_embedding)
```

输出为归一化坐标下的直接score。

## 训练目标与选择

训练采样 `epsilon~N(0,I)`，`y=X+sigma*epsilon`：

```text
L = mean((sigma * branch(y,C,R) + epsilon)^2)
```

sigma固定为该run的超参数，默认pilot=.05；没有t采样、VP换算或多尺度混合。部署checkpoint绑定sigma，改变它必须新训练。日志记录 `sigma * ego_std` 的x/y/cos/sin各通道扰动标准差；cos/sin标准差不能称为角度标准差。

使用官方 train allowlist 中的 DB 作为候选来源，调用 nuPlan 官方 builder/filter 全局随机选择最多 1,000,000 个场景（包含内部训练/验证），展开场景、不做时间抽稀、不设每 DB 上限，remove_invalid_goals=false。按 recording 分约95%训练/5%验证；测试集不参与。先冻结选中 token 名单再并行提取特征，完整且有限的未来轨迹检查仍保留。预处理开始时冻结已解压 DB 列表，之后新增 DB 不加入已有 run。旧版无限量处理迁移到新目录，可用 --reuse-features-from 复用选中且校验通过的旧 NPZ；不沿用旧缓存或 checkpoint。

这里的val是score训练的内部验证划分，不是官方Val14。冻结的原planner可能在预训练时见过这些train allowlist中的记录；正式闭环仍需要单独的官方验证/测试协议。

8卡默认有效batch2048=8×64×4累积。每epoch对全部训练帧全局shuffle，不重复取样，丢弃不足一个global batch的尾部；下epoch尾部随shuffle改变。固定8份验证噪声按frame token确定，所有rank无补齐重复，逐元素求和后聚合。

新文档没有规定精确调度，本实现明确采用：AdamW lr1e-4、weight_decay1e-4、grad_clip5；1epoch从1e-5 warmup到1e-4；EMA目标.999带初期warmup。每epoch验证，改善阈值.5%，3次无显著改善学习率减半，最低1e-6；8次无显著改善且至少5epochs时早停。30epochs为上限。全部可配置。

checkpoint选择只比较初始分支和各次验证的EMA分支，按验证DSM实际最小值；raw验证用于记录，不根据test挑选权重。若选择initial，需如实报告没有改善验证目标。早停说明所设验证目标进入plateau，不等于真实score误差已知，更不等于ascent到达峰值。

`score/last.pt`包含分支、EMA、optimizer、每rank RNG、epoch内游标、lr控制器及历史，用于严格续训；`score/best.pt`只保存选中分支与协议/normalizer/base身份。均不重复保存冻结planner。保持同一world size、数据、代码、训练协议可恢复，变更则拒绝继续。

训练前smoke用4帧和固定噪声做200步短拟合，检查finite和loss下降。该检查仅说明实现可优化，权重被丢弃，不声称验证泛化。正式训练使用每次独立噪声。

## Refinement与评估解释

起点为原planner生成的ego未来，推理不加噪声：

```text
x[k+1] = x[k] + gamma * sigma^2 * s(x[k],C,R)
```

默认gamma=.1、K=5；严格执行给定K，没有一万步/两万步搜索，也不根据预算退出宣布收敛。每次重新计算候选score，C/R固定。默认heading投影开启，在物理cos/sin坐标归一化，再映射回normalized；零向量先回退到上一步有效朝向，否则使用(1,0)。gamma=0或K=0时在物理预测层提前返回，连归一化往返和投影都不执行，保证baseline一致性。

仅替换prediction[:,0]，prediction[:,1:]逐项不变。最终继续使用官方atan2、坐标转换和nuPlan trajectory构造。当前ego state不在优化张量中。

`evaluate`计算Section8：DSM、随机条件标签cosine分布、无投影一步恢复(gamma=.1,.25,.5,1)、Tweedie(gamma1)、normalized MSE、物理ADE/FDE、heading angular MAE、改善比例、score norm及D1/D2。零范数cosine单独计数，不伪造方向；输出逐样本CSV/JSONL和固定子集图。

对于同一批noise，gamma=1的normalized MSE恰好等于sigma²乘DSM；它是同一目标的去噪解释，不是一项独立的真实score验证。

`evaluate-planner`运行原始预测并对同一起点做离线refinement，记录每步score、轨迹、位移、heading偏差和时间。baseline/refined使用同一原始输出，避免采样差异。生成按timestamp/reference_seed固定；score计算不使用此随机种子，也不产生随机噪声。

这些指标是诊断，不是nuPlan成绩。图中的专家恢复也不能用于计算未知真实密度增量。官方闭环通过独立adapter接入既有仿真配置；使用同场景、同seed、同base及同模拟器设置，按场景配对比较。Val14调参完成后再固定设置运行hard benchmark，S1/S2/S3/S5消融留在后续，不在此次实现中自动扩展。

## 缓存、目录与运行

所有命令见README。新缓存只有target/context/route，分别为80×4、107×192、192；按512帧分shard，mmap按需读取。每百万帧float32张量约84.2GB（十进制，另加NPZ和metadata）。缓存与sigma无关，可跨sigma只读复用，但与base/归一化/manifest绑定。旧两个项目的缓存schema不兼容，会被拒绝。

测试/调试代码位于tests；主代码位于src/score_function；原方案文档保留在docs；不从旧两个Python包导入。`previous_projects_hashes.json`用于确认旧目录未修改。真实运行产生的日志、checkpoint、图均写入配置的独立output，不混入源代码目录。

参考：[官方源码](https://github.com/ZhengYinan-AIR/Diffusion-Planner/tree/a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df)、[Vincent固定高斯DSM理论](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)。
