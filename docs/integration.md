# 把 Score 接到业务规划模型后面

交接对象是 `ScoreFunctionBranch` + `refine_ego`，以及匹配的条件编码与归一化。先运行 CPU 示例，再替换真实权重与真实场景。

## 1. 条件来源决定如何复用权重

| 方案 | 业务模型做什么 | Score 条件来自哪里 | 已有 S10-G 权重 |
|---|---|---|---|
| 复用现有 Score | 产生候选 ego 轨迹 | 训练时同一 DP 场景／路线编码器 | 可加载，业务效果需配对验证 |
| 使用业务条件特征 | 产生轨迹和自有场景／路线特征 | 业务编码器 | 需要训练匹配的 Score |

推荐先用第一种方案验证接入。DP 编码器读取场景、地图、路线等观测，**不需要 DP 生成候选轨迹**。相同的特征维度不代表相同语义，不能直接把业务特征送入已有 Score 权重。

`ScoreRefinedPlanner` 是 DP 专用包装器，读取 DP 内部结构；不要把任意业务模型直接塞进该类。业务端使用 [examples/business_integration.py](../examples/business_integration.py) 中的 `refine_physical_ego`，自行适配业务模型的输入输出。

## 2. 输入输出契约

| 项目 | 现有 DP / S10-G 契约 |
|---|---|
| ego 轨迹 | float32 `[B,80,4]`，列为 `x,y,cos(theta),sin(theta)` |
| 时间 | 未来 `0.1,0.2,…,8.0 s`；不含当前点，不优化历史 |
| 空间 | 当前 ego 后轴为原点、当前朝向为坐标轴的局部坐标，XY 单位米 |
| scene | 官方观测归一化后，经原冻结场景编码器得到 `[B,107,192]` |
| route | 同一观测的路线，经原冻结 route encoder 得到 `[B,192]` |
| ego normalizer | 训练所用 agent 0 的 mean/std，不是业务模型自己的归一化 |
| 输出 | 同一时间网格与局部坐标下 `[B,80,4]` 物理轨迹 |
| dtype / device | 轨迹、条件和 Score 权重一致，首次接入建议 float32 |

业务轨迹若是 `(x,y,theta)`，先转为 cos/sin；世界坐标或不同参考点需先转换。时间网格不同时先插值，处理角度 ±π 跨越，并包含当前状态到第一未来点的区间；不要按数组索引截断或重复点。候选轨迹不足 8 秒时，不能假定其余时刻有效。

每帧重新提取真实条件，同一帧 K 次更新固定 C/R。S10-G 不使用 predicted-neighbor 条件，也不需要未来真值。启用 `neighbor_future` 的其他配置须提供训练约定的邻车预测及有效 mask，不能用虚构零预测代替。

## 3. 加载配套权重

使用权重附带的**原始配置**，只迁移部署路径。仓库默认 local / σ=0.05 配置不能替代 S10-G 配置。先安装配套 DP 及依赖，并确保导入源码与配置一致：

```python
import inspect
from pathlib import Path

from diffusion_planner.utils.config import Config
from score_function.utils.checkpoint import load_selected
from score_function.utils.config import load_config
from score_function.utils.planner_utils import (
    ego_normalizer_metadata, load_planner_config, planner_identity,
)
from score_function.utils.train_utils import resolve_path

cfg = load_config("/path/to/original-score-config.json", root="/path/to/workspace")
device = "cuda:0"
branch, saved = load_selected(cfg, "/path/to/selected-score.pt", device)
if Path(inspect.getfile(Config)).resolve().parents[2] != resolve_path(cfg, "planner_dir").resolve():
    raise ValueError("Imported DP source differs from configured DP source")
if saved["planner"] != planner_identity(cfg):
    raise ValueError("DP source / args / checkpoint differ from score training")
args = load_planner_config(cfg, device)
normalizer = args.state_normalizer
if saved["ego_normalizer"] != ego_normalizer_metadata(args):
    raise ValueError("Ego normalization differs from score training")
sigma = saved["sigma_score"]
```

`load_selected` 返回 eval 模式的冻结分支，检查架构、parameterization、训练 σ 和权重结构；条件来源由上面的后续检查核验。只加载可信来源的 checkpoint。

`best.pt` 已含选中的分支权重，不要再次做 EMA。交付 S10-G 时核验 `weight_kind=ema`、global temporal attention 开启、neighbor_future 关闭、`sigma_score=0.1` 及对应 epoch；文件名本身不能证明身份。`last.pt` 用于恢复训练，不是部署文件。

## 4. 编码与调用

首次接入可加载冻结 DP，再只调用编码器：

```python
from score_function.utils.planner_utils import encode_conditions, load_frozen_planner
from examples.business_integration import refine_physical_ego

# 一次初始化：加载完整 DP 权重，后续不调用轨迹采样器。
frozen_dp, _ = load_frozen_planner(cfg, device)

# 业务侧提供 candidate 和同一帧的 dp_inputs。
# dp_inputs 已经过官方观测适配器和 ObservationNormalizer。
scene, route = encode_conditions(frozen_dp, dp_inputs)
refined, trace = refine_physical_ego(
    candidate, branch, scene, route, normalizer,
    sigma=sigma, gamma=0.1, steps=20, heading_projection=True,
)
```

`candidate`、`dp_inputs` 由业务适配器提供，上面不是独立可执行脚本。二者时间戳、ego 锚点、路线与场景必须一致。`examples` 是仓库内的参考代码，不作为 `score-function` 安装包发布；从仓库根目录运行，或把该函数接入业务适配层。

上述方案加载完整 DP，会占用相应内存；部署时可以仅加载原 scene/route encoder 权重。参考固定版本的 [UrbanDriver runtime](https://github.com/ecowyk/Score_Function/blob/83f39cd/urban_driver_score/runtime.py) 和 [轨迹适配器](https://github.com/ecowyk/Score_Function/blob/83f39cd/urban_driver_score/planner.py)。它们是特定实验／环境的参考实现。

K=0 / gamma=0 对照应在**业务调用层的插值和条件编码之前**返回原生轨迹，保留原来的时间网格与轨迹构造；示例函数也提供精确物理轨迹旁路。默认关闭完整轨迹 trace，但核心仍计算每步诊断、CPU 拷贝和 CUDA 同步；应测量含条件编码与转换的完整耗时。

## 5. 交付清单

发一个固定提交的代码链接，并通过允许的共享渠道提供以下材料（文件名是约定示例）：

```text
score-handoff/
├── score-selected.pt       # 实际选定的部署权重
├── score-config.json       # 权重对应的原始完整配置
├── dp-args.json            # 原 DP 参数与归一化
├── dp-model.pth            # 原冻结编码器权重所在 checkpoint
├── environment.txt        # 工作环境版本
└── handoff.json            # 版本、hash、设置、验收记录
```

这些二进制文件**不在 GitHub 仓库中**。从实际运行目录提取并记录：

- Score 代码 commit、DP 源码 commit；DP 有本地修改时保存源码快照及 checkpoint 内的源码哈希。
- Score checkpoint、DP checkpoint、DP args 的 SHA256；原始配置、epoch、选权方式及 σ。
- gamma、K、heading_projection，以及业务轨迹的坐标和时间网格。
- Python / PyTorch / CUDA / nuPlan / DP 版本、数据与地图要求。
- 一条允许共享的真实输入及 baseline/refined 输出，包含时间戳与坐标说明。
- 接口、真实样例、完整延迟、配对业务指标的验收状态。

不要把目录名或截图当作权重身份；交付前读取实际元数据并核对哈希。这次仓库整理没有读取服务器权重或运行真实业务评测。

## 6. 接入验收

1. `python -m examples.business_integration --smoke`：确认安装与接口，随机权重不证明效果。
2. 真实配套权重与条件跑一帧：检查 finite、shape、坐标、heading 和输入未被修改。
3. K=0 保持原业务输出精确一致；从同一候选轨迹运行 K=20，记录精炼结果及完整延迟。
4. 固定数据和随机种子，配对比较原模型／原模型+Score，再决定业务侧 K 与 gamma。

σ 必须与训练一致；K 和 gamma 可用于部署评估，但需验证。Score 更新与 heading 单位圆投影不保证碰撞、动力学约束或业务收益，继续使用业务系统的验收标准。
