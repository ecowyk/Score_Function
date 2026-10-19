# Score Function

独立的条件 Score 模块，用于细化规划模型输出的 ego 未来轨迹。业务接入从 [接入指南](docs/integration.md) 和 [最小示例](examples/business_integration.py) 开始。

```text
业务规划模型 → ego 候选轨迹 → 归一化 → Score 迭代 → 反归一化 → 业务轨迹输出
                                        ↑
                            固定的场景条件 C / 路线条件 R
```

Score 核心不调用轨迹生成器。**已有 DP 条件权重仍依赖训练时的 DP 场景／路线编码器及归一化参数**；业务模型负责提供候选轨迹。若改用业务模型自己的条件特征，需要匹配数据重新训练，不能只对齐维度就复用已有权重。

## 先跑通接口

在准备好的 Python 3.9+ / PyTorch 2.x 环境中：

```bash
git clone --branch codex/mentor-score-handoff https://github.com/ecowyk/Score_Function.git
cd Score_Function
python -m pip install --no-deps --no-build-isolation -e .
python -m examples.business_integration --smoke
```

全新环境先安装 PyTorch，以及 `numpy<2`、`matplotlib>=3.5,<4`、`tqdm>=4.64,<5`、`setuptools`、`wheel`。GPU 环境沿用已有匹配 CUDA 的 PyTorch。

这个 CPU 示例使用随机初始化模型与合成条件，只验证接线、形状和更新过程；不需要 nuPlan 数据或 DP 权重，也不代表真实规划效果。实际接入必须换成匹配的已训练权重和真实条件。

## 接入只需先看这些文件

| 入口 | 用途 |
|---|---|
| [业务接入指南](docs/integration.md) | 输入输出、权重配套、DP 依赖和交付清单 |
| [物理轨迹接入示例](examples/business_integration.py) | 归一化 → 调用原 refinement → 反归一化 |
| [score_branch.py](score_function/model/score_branch.py) | `ScoreFunctionBranch`，预测归一化坐标下的 score |
| [refinement.py](score_function/model/refinement.py) | `refine_ego`，固定次数轨迹更新 |
| [normalizer.py](score_function/utils/normalizer.py) | ego 归一化与物理 heading 投影 |
| [checkpoint.py](score_function/utils/checkpoint.py) | 加载选中的权重，核验架构及训练 σ |

输入轨迹为 `[B,80,4]`，对应未来 `0.1…8.0 s` 的 `(x,y,cosθ,sinθ)`，不含当前点。现有 DP 条件为 scene `[B,107,192]`、route `[B,192]`。每次调用固定条件，执行：

```text
x[k+1] = x[k] + gamma * sigma² * score(x[k], C, R)
```

不在推理中加噪声。默认可对物理 `(cosθ,sinθ)` 做单位圆投影；这不是动力学约束或碰撞检查。`K=0` / `gamma=0` 的业务示例会直接返回原始轨迹对象。

## 配置与权重不要混用

- `configs/score_function.json` 是基础训练配置：local temporal、σ=0.05、K=5。
- 当前 S10-G 研究权重需要它自己的原始配置：direct score、global temporal attention、无 predicted-neighbor 条件、σ=0.1。已有 K=20 设置属于该实验协议，不是所有业务场景的默认最优值。
- 提供选中的部署权重（通常为 `score/best.pt`）及配套配置；`score/last.pt` 是训练恢复文件。GitHub 不包含训练权重、数据和服务器结果。
- `load_selected` 核验架构与 σ；条件编码器和归一化来源还需按 [接入指南](docs/integration.md) 单独核验。

## 需要训练或复现时

[DP 数据、训练与评测完整流程](docs/reproduction.md) 保留了原 README 的运行命令。训练入口是 `train_score_branch.py` / `scripts/train.sh`，nuPlan 评测入口是 `sim_score_planner.sh`。八卡实验矩阵、energy 对照和吞吐诊断按需使用，不是业务接入的前置步骤。

[仓库与分支说明](docs/repository_map.md) 记录了清理内容和实验分支用途。交接版以 `main` 为基础；截至整理时，最新 UrbanDriver 实验分支的 `score_function/`、`configs/` 与 `main` 一致。实验分支保留以复现历史运行。

运行回归检查：

```bash
python -m unittest discover -s tests -v
```

核心单元测试是接口保障，应当保留；依赖官方源码、真实数据的检查会在缺失条件时明确跳过。历史实验验证记录见 [docs/validation.md](docs/validation.md)。
