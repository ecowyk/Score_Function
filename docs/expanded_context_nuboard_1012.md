# 扩展场景、完整闭环轨迹与 nuBoard 对照

## 三种视图分别回答什么

1. 全程静态图：DP/G20/G50/G200 在整个场景中实际走了哪条路径。
2. 单次规划图：在同一个 DP 历史输入下，完整 8 秒预测轨迹如何被 score 推动。
3. nuBoard：动态查看实际闭环中的自车、周车、行人、预测轨迹和指标。

一张单次规划图已经包含未来 80 个点，并非只画某个轨迹点。之前只在场景
开头、三分之一、三分之二选择了三次规划，没有展示每次重新规划。
全程静态图则读取历史中的每个实际自车状态，各运行保留自己的采样时刻和点数。
不把 G20 第 i 帧与 DP 第 i 帧强行当作相同状态，也不把预测轨迹当实际行驶路径。

## 场景扩展到 22 个

保留原 16 个场景，增加以下六个，总计改善候选 6、退化候选 6、步数敏感候选 6、对照 4。
这些是诊断样本，不用于重新估计整体测试分数。

| ID | token | 类型 | 观察重点 |
| --- | --- | --- | --- |
| I05 | a2fc81fb19985d60 | high_lateral_acceleration | 三个 K 的碰撞/TTC 指标均改善 |
| I06 | 0d3490827eb45df7 | following_lane_with_lead | 跟车场景的碰撞/TTC 改善 |
| W05 | 62db9dc16ffd5135 | traversing_pickup_dropoff | 三个 K 均出现 TTC 退化 |
| W06 | 6db7f9f43c655149 | behind_long_vehicle | 长车后方的 TTC 退化 |
| K05 | 217131c79adf588d | changing_lane | G20 提高，但 G50/G200 未保持提升 |
| K06 | 0cd6cff135c25fa9 | starting_left_turn | 小幅变化对照：进度随 K 反复 |

上述观察重点来自先前逐场景汇总，不等于已经确认了具体事件原因。
实际图中得分始终从官方 parquet 读取。

```bash
cd /home/data/wyk/Score_Function_aligned_v1
SCORE_PY=/home/data/wyk/envs/diffusion_planner/bin/python
"$SCORE_PY" scripts/expanded_score_context.py list
```

完整 token、recording、类型和四组分数写入：
`/home/data/wyk/dataset/nuplan/exp/score_G_static_context_1012_v1/selection_expanded_v1/cohort.csv`

## nuBoard：读取已有四组完整评测

```bash
"$SCORE_PY" scripts/nuboard_score_compare.py --list
"$SCORE_PY" -u scripts/nuboard_score_compare.py --port 6599
```

可在独立 tmux 会话中运行第二条。脚本自动使用冻结套件 `jobs/*.json` 指向的
DP_K0、G20、G50、G200 原始结果，包括 G20/G50 复用的评测目录；不重新仿真。
脚本设置 DP、nuPlan devkit、data/maps/exp 路径、
`NUPLAN_SIMULATION_ALLOW_ANY_BUILDER=1`，并清除分布式训练环境变量。
Viewer 隐藏 GPU，在进程内把自己的可信仿真日志加载到 CPU；不改原始文件。

在 VS Code 的“端口 / Ports”面板转发服务器 **6599** 到本机 **6599**，
打开 `http://localhost:6599/`。若本机使用其他端口或代理域名，启动时追加
`--allow-websocket-origin 实际浏览器主机:端口`。

进入 Scenarios 页面，在 **Scenario token** 中输入一个 token，选中候选后点击
**Query Scenario**。例如输入 `5d392fa38ff65c3f` 查看 W01。
四组会分别显示，标题中的运行目录区分 DP/G20/G50/G200。
同一 planner 名称属正常，实验目录不同；在左侧实验目录复选框切换各组。
每组都有时间滑条和播放按钮，不能假定四组播放自动同步。
比较时看同一个场景、对应时刻，同时关注实际车身位置和显示的未来预测轨迹。

## 完整实际行驶轨迹图

```bash
"$SCORE_PY" -u scripts/closed_loop_context_overview.py --cohort all
```

输出目录：`score_G_static_context_1012_v1/closed_loop_overview_v1/`，其中
`I_全程实际行驶.pdf`、`W_全程实际行驶.pdf`、`K_全程实际行驶.pdf`、`C_全程实际行驶.pdf`
为分组报告，`cases/<ID>/` 内是单场景 PDF 和 PNG。

完整路径图由已保存的仿真历史生成，不运行推理。每页比较 DP 和一个 K，保留
全部实际位置点、道路环境、场景闭环得分。周车/行人随时间变化，用 nuBoard 看；
不会在全程路径图上摆一组固定的周车框来暗示全程碰撞关系。

## 补齐单次规划图

```bash
"$SCORE_PY" -u scripts/expanded_score_context.py run
```

这一步会为 22 个场景补采三次规划的预测与推动结果，再生成按 I/W/K/C 分开的 PDF。
兼容的旧采集帧会复用。它会运行离线规划和 score 推动，但不重跑闭环场景。
默认 CPU 和两个线程，保留相同初始噪声、模型、gamma、输入与重放检查。

输出：`score_G_static_context_1012_v1/report_cn_v4_expanded/{I,W,K,C}/trajectory_report_cn.pdf`

如果在 nuBoard 找到某个值得看的时刻，可对一个场景补采 DP 历史中的指定索引：

```bash
"$SCORE_PY" -u scripts/expanded_score_context.py run --cohort W01 --frames 40 50 60
```

这里的 40/50/60 只是索引示例，不代表已经定位到事件。应按实际日志时间戳对应，
不能把界面秒数直接当作帧号。即使选择了相同时刻，DP 与 G 的闭环状态也可能已经
不同，因此固定 DP 输入的推动图用于分析作用方向，不能单独证明某次事故的原因。
