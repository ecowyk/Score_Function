# 研究定位、方法与演变

状态日期：2026-11-02。项目意图来自用户已确认的讨论；数学实现由仓库源码核对。来源编号见 [SOURCES.md](SOURCES.md)。

## 研究问题

生成或回归式 planner 已能给出轨迹，但一次输出不保证落在条件行为分布的局部密度峰附近。项目尝试利用独立学习的轨迹先验，对当前输出做局部调整，检验它是否带来可重复的闭环收益。

用户明确的目标是：**从当前轨迹出发，沿条件 score 到达当前模式内可达的局部密度极大点**。实际算法使用有限步更新，所以“到达极大点”和“保持模式”是研究意图，不是已经验证的性质。项目范围是数据驱动的 ego trajectory planning / refinement，不是 action-conditioned world model，也不使用 RL。

DP 是最初的轨迹来源和目前条件编码器的来源，研究范围后来扩展到多个 planner。不同 baseline 上的改善或退化共同决定方法的适用性。

## 数学对象与训练

令 $x\in\mathbb R^{80\times4}$ 为归一化后的 ego 专家未来轨迹，$C$ 为场景 token，$R$ 为路线 embedding。每个训练 run 固定一个噪声尺度 $\sigma$：

$$
\epsilon\sim\mathcal N(0,I),\qquad y=x+\sigma\epsilon.
$$

网络采用 denoising score matching：

$$
\mathcal L(\theta)=\mathbb E\left[\operatorname{mean}\left(\sigma s_\theta(y,C,R)+\epsilon\right)^2\right].
$$

在理想的容量/优化条件下，最优网络对应高斯平滑条件分布的 score：

$$
p_\sigma(y\mid C,R)=\int p_{\rm data}(x\mid C,R)\mathcal N(y;x,\sigma^2I)\,dx,
\qquad s^*(y,C,R)=\nabla_y\log p_\sigma(y\mid C,R).
$$

单次 corruption 的标签 $-\epsilon/\sigma$ 不是某个真实场景的可直接观测“真 score”；最优回归是给定 $y,C,R$ 的条件期望。有限数据、条件编码的信息损失及模型拟合误差都会影响结果。真实 nuPlan 场景通常只有一条记录的专家未来，不能因此声称知道真实条件分布或密度增量。

## 推理与目标边界

对 planner 输出 $x_0$，固定同一规划时刻的 $C,R$，不再注入噪声：

$$
x_{k+1}=\Pi\left(x_k+\gamma\sigma^2s_\theta(x_k,C,R)\right),\qquad k=0,\ldots,K-1.
$$

$\Pi$ 表示可选的 heading 表示投影：反归一化后把 $(\cos\theta,\sin\theta)$ 投到单位圆，再归一化。它不是碰撞避免或动力学可行性求解。关闭该选项时 $\Pi$ 为恒等映射。

S10-G 主配置下 $\gamma\sigma^2=0.001$。K 是 score 更新次数，与 DP 的去噪步数、Flow 的 ODE 积分步数不同。每个闭环规划时刻会重新取得条件；固定条件仅指一次 refinement 内。

当前直接输出的向量场不强制保守性，有限步数也不检验 Hessian 或峰值；加上 heading 投影、离散步长和估计误差，不能保证真实密度逐步上升，也不能严格保证不跨模式。实验表述应是“有限预算局部 refinement”，把峰值可达性留作需要证据的研究问题。

## Direct score 与 Energy

- **Direct score / S10-G**：直接输出与轨迹同形状的向量场。
- **Energy / E10-G**：输出标量 $E_\theta(x,C,R)$，通过 $s_\theta=-\nabla_xE_\theta$ 得到 score，仍采用相同 DSM 和推理更新规则。

Energy 的训练需要对输入梯度继续反向传播，计算代价更高。标量能量的绝对加性常数不可辨识。早期未充分对齐的 Energy 实验不足以否定这一路线；10 月 25 日后已经安排严格对齐 S10-G 的 E10-G，并有训练完成的历史报告。最终闭环对照分数尚待从服务器补齐。

## 项目演变

| 阶段 | 当时的问题或行动 | 现在应如何理解 |
| --- | --- | --- |
| 初期 toy FM / diffusion | 少步采样分布坍塌，零噪声初始化集中到中心附近 | 属于特定 toy 观察；“零噪声总是均值/中位数”不是一般定理 |
| 迁移 DP | 比较少步/多步采样和闭环效果 | 没有观察到明显的少步闭环退化；不据此断言生成分布完全不变 |
| 早期 proxy | 尝试从 DP 去噪输出或扩散中间量导出更新 | 不是目前独立、无 diffusion time 的 score 分支 |
| 独立 DSM 分支 | 固定 sigma，训练 ego 轨迹 score，条件来自冻结 encoder | 当前方法基础；主分支没有 diffusion timestep 输入 |
| 对齐训练与 G 结构 | 训练预算/增强等对齐 DP，增加全局时间 attention | 需要区分结构、训练预算与推理步长等共同变化 |
| S10-G 与 K 矩阵 | 多 seed、多个 K，检查闭环收益和可视化 | K20 成为后续主设置，不宣布所有单项最优 |
| 多 baseline 迁移 | 包括确定性/生成式方法和原生 wrapper 检查 | 多项 NR 改善，也有 GC-PGP 退化；研究泛化边界 |
| Energy 对齐补测 | E10-G 重新训练并安排 K 对照 | 训练已报告完成；对照结果需要补录 |
| NAVSIM 讨论 | 转向传感器输入下的端到端 planner | 需重新训练并控制条件编码来源；尚无本项目结果 |

## 当前证据能说明什么

已报告的闭环结果支持：这一类局部先验在部分 NR baseline 上有实际收益，且收益并不限于 DP。它尚不能证明收益来自严格的密度最大化、所有 baseline 均受益、跨 benchmark 普适，或 R 集退化有唯一确定原因。

几类诊断要分开：DSM/随机标签 cosine 检查训练行为；离线 ADE/FDE 检查相对单条 GT 的变化；轨迹可视化检查位移、形状与模式；闭环得分评价实际行为。ADE 随 K 可以下降、先降后升或持续上升，局部密度方向不必朝向该条 GT，闭环也存在状态反馈。

## NAVSIM 后续迁移

用户关注的是：如果 score 分支使用新的、更强的视觉 encoder，最终提升可能同时来自条件表示变化。需要区分 baseline 原有表示、score 读取的表示、是否共享/冻结、训练数据、帧数和 encoder 规模。

优先考虑在同一 baseline 内固定条件来源，再比较 K0/K20；若更换表示，增加能够分离 encoder 收益的对照。不同 baseline 可分别适配其表示，不能仅凭 embedding 维度一致就复用同一权重。ViT-Large 和多帧目前是讨论项，不是已经选定的结构。NAVSIM 的具体版本、指标及 baseline 选择需要开始任务时读取官方实现，不能把 nuPlan NR/R 数值直接横向比较。
