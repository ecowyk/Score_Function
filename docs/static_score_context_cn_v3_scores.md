# 中文静态轨迹报告 v3：逐场景闭环得分

本版读取已采集的轨迹和已完成的官方 nuPlan 评测结果，只重新绘图。
不运行模型推理或仿真，不需要 GPU。使用服务器原 diffusion_planner 环境即可。

## 场景编号

| 前缀 | 含义 |
| --- | --- |
| I | 改善候选（Improve） |
| W | 退化候选（Worsen） |
| K | 步数敏感候选：不同推动次数下得分出现反复 |
| C | 对照（Control） |

这些是人工选样标签，不是 nuPlan 官方场景类型。后面的数字是组内编号，不是排名。
`K01` 是场景编号，`K=20` 则表示推动 20 次。每个 K 的实际改善、退步或持平，
由该次评测得分与 DP 得分比较决定，不由 I/W/K/C 标签决定。

## 图页新增内容

每页上方显示同一场景的 `DP 原始得分 → G 当前 K 得分`、分差及中文判断。
分数统一为 0–100 分；分差等于 G 减 DP，正值改善、负值退步。
按未四舍五入的数值比较，绝对分差不超过 0.000001 分视为数值持平。
微小但非零的变化显示“小于 0.01 分”，避免显示误导性的正负 0.00。

这些是**整段场景的闭环得分**，不是当前帧或当前画出的单条轨迹得分。
图仍使用相同 DP 闭环历史和输入，对比固定输入下不同 K 的轨迹推动；
它不表示各个 G 闭环运行分叉后实际执行的轨迹。得分持平也不表示轨迹没有移动。

分数从冻结套件 `jobs/*.json` 指向的 `aggregator_metric/*.parquet` 读取，
精确匹配 token、场景类型及 recording。不会使用全测试集平均分、类型汇总分，
也不会把缺失结果填成 0。保存的 K、gamma、配置、权重和采集来源须一致。
已复用的 G20/G50 结果沿用各自原评测目录。

## 重新生成

```bash
cd /home/data/wyk/Score_Function_aligned_v1
/home/data/wyk/envs/diffusion_planner/bin/python -u \
  scripts/render_static_context_cn.py --cases all
```

`--cases all` 处理全部已采集场景；不带此参数时处理 I01/W01/K01/K02/K04/W04。
默认每帧展示 K20、K50、K200，各占一页。可用 `--frames 0` 只看第一帧。
若采集目录不同，传 `--source /path/to/capture_output`。
评测套件默认取采集元数据的 suite 路径，也可显式传 `--suite`（须与采集来源一致）。

默认输出目录：
`/home/data/wyk/dataset/nuplan/exp/score_G_static_context_1012_v1/report_cn_v3_scores/`

- `trajectory_report_cn.pdf`：合并 PDF。
- `figures/<case>/frame_<index>/`：单页 PNG 和分帧 PDF。
- `scores_used.json`：实际使用的逐场景得分、分项和来源路径。
- `index.json`：图件和原始采集数据的对应关系。

本版保留原图页的道路环境、细轨迹线和每帧小点；随脚本附带离线中文字体。
原始 captures 和旧版报告不修改。缺失或不一致的评测结果会在绘图前报错。
