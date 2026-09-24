## 2026-09-24 服务端目录重构验证

本节对应 4093 上的分层目录版本；本地文档和压缩包尚未同步。

- 在官方 diffusion_planner_wyk 环境中，完整 28 项测试通过。测试从项目目录外通过已安装包执行，包含双进程 Gloo 精确恢复、固定噪声选权重、官方模型接口和 nuPlan 父类调用。
- Ruff 代码与格式检查通过；三个根入口及 python -m score_function 配置入口通过。
- Hydra planner 目标已更新并可导入；三个 Bash 脚本通过语法检查，tmux 使用新的包路径。
- 使用兼容服务器 setuptools 59.5.0 的 setup.py；发行包识别为 score-function 0.1.0，未升级环境依赖。
- 官方 Python 源码、训练配置和网络类定义保持不变。source_hashes 覆盖整个 score_function 包，共 40 个 Python 文件。
- 本次未启动真实数据训练或 nuPlan 闭环，也未重建已有科研结果。

# 实现验收

日期：2026-09-24。以下均为本地工程测试，使用合成输入、合成标签或临时随机权重。不是本方案在真实nuPlan数据上的实验结果。

## 已通过

- 27项单元/集成测试通过：26项完整测试运行，加1项后补的生产流水线端到端测试。
- 新模型接口只有候选轨迹、scene、route；没有t或sigma输入。输出80×4，非因果卷积保持长度，默认候选感受野37点。
- 直接score的固定sigma DSM、ego-only归一化及heading投影/退化回退测试通过。
- gamma=0或K=0保持输入精确一致；实际官方DPM采样输出与关闭refinement的wrapper在恢复同一RNG后逐项一致，包括原始heading向量非单位长度时。
- 开启refinement只改变ego future，邻车输出及原始prediction张量保持不变。
- 冻结模块保持eval/无梯度；score分支有非零有限梯度。官方模块临时合成checkpoint加载、107×192 scene、192 route、80×4 ego标签及11×80×4联合预测接口通过。
- 数据分片、损坏检查、缓存方法隔离、official train allowlist、spawn DataLoader、全局无重复shuffle通过。
- 单进程中途和epoch最后update后恢复，及2-rank CPU/Gloo中途恢复，raw/EMA参数、AdamW、调度状态、验证历史与连续训练逐项相同。改变sigma、模型协议会被拒绝。
- 初始/EMA选权重及选中checkpoint验证loss复算通过。
- 离线DSM/cosine/物理ADE/FDE/heading计算、零范数处理、批次无关固定噪声、流式报告、同起点planner对照均通过；评估模块自身单测使用可解析oracle或替身，仅验证指标和IO，不替代真实planner边界测试。
- 小型固定corruption优化检查：width24测试分支100步，DSM从0.936453降至0.115639，说明反向和更新链路能拟合固定标签；不能据此宣称真实数据泛化。
- 默认width192模型CUDA microbatch64前向/反向通过，1,387,204参数，所有梯度有限。本机RTX3060 Laptop、PyTorch2.0.1+cu118、NumPy1.26.4。分支峰值allocated约227MiB；此值不含冻结encoder缓存构建及完整optimizer，不是公司机器实测。
- Ruff lint/format、三个Bash脚本语法及CLI JSON参数覆盖检查通过。
- 原两个项目53+33个文件SHA256保持不变。
- 端到端测试使用实际官方模型类、临时合成EMA权重及6帧合成NPZ，完整执行：缓存构建→再次复用（index/shard哈希及mtime不变）→smoke→1epoch训练→corruption离线诊断→1帧真实DPM输出refinement诊断。状态、有限指标、CSV/报告/图像输出均通过；全部临时产物清理，未使用真实数据。

## 尚未执行

公司的8卡A100/NCCL；实际预训练发布权重及真实DB/map的全量缓存/训练；nuPlan仿真、NuBoard和官方分数对照；架构/σ/K/gamma消融。提供启动入口不等于已经跑过这些阶段。

## 复现

```bash
cd /mnt/pai-hdd/wangyikai/Score_Function
export PYTHONPATH="$PWD/src:$PWD/tests:/mnt/pai-hdd/wangyikai/Diffusion-Planner:${PYTHONPATH:-}"
export SCORE_FUNCTION_OFFICIAL_ROOT=/mnt/pai-hdd/wangyikai/Diffusion-Planner
python -m unittest discover -s tests -v
```

不设置 `SCORE_FUNCTION_OFFICIAL_ROOT` 时，官方模块相关测试明确跳过，其他测试不需要官方/nuPlan依赖。测试不会下载任何文件。官方源码核验提交为 `a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df`。
