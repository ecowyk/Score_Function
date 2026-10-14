# 带场景 agent 的 GIF

入口：`scripts/score_context_gif.py`。在原有静态方案旁增加 GIF，不改动模型、评测或已有 capture 文件。默认读取现有 S05-G 冻结实验及其 22 个诊断场景。

## 两种动画

| 输出 | 动画轴 | 场景中的其他 agent | 用途 |
| --- | --- | --- | --- |
| `refinement/<case>/frame_<index>.gif` | 实际保存的 K 快照 | 保持该次规划输入时刻的观测位置；显示车辆、行人、自行车及静态对象的真实外框 | 看同一输入下轨迹如何被推、推了多少 |
| `closed_loop/<case>/DP_vs_G_K<steps>.gif` | 闭环时间 | 按各次运行自身的观测时间戳移动、出现或消失；信号灯同步更新 | 看实际驾驶过程及整场景闭环得分 |

推动 GIF 左图保留完整轨迹和道路上下文，右图放大位移最大的局部。灰虚线是原始 DP，蓝线是当前 K；箭头表示**前一个展示的 K 到当前 K 的实际坐标位移**，按米绘制，未夸大箭头长度。标题同时显示相对 DP 和相对上一快照的最大位移，轨迹标出未来 1/2/4/6/8 秒位置。

原有采集通常只保存 K=0/5/20/50/100/200/500，因此动画按这些真实快照切换，不插值制造逐步更新。K=5 到 K=20 的箭头是 15 次更新后的净位移，不是某一步的原始 score 向量；最大位移也不是累计移动距离或概率提升量。要分析每一步，需额外采集逐步轨迹或 score trace，已有稀疏快照无法恢复。

闭环 GIF 左边 DP，右边指定 K 的 S05-G；两边同一坐标系、同一时间轴、同一视窗，各自显示实际行驶轨迹及自身观测的 agent。按时间戳取不晚于回放时刻的最近一条记录，不按数组下标配对，也不外推结束后的车辆位置。结束的一侧只留下实际轨迹并标记“已结束”。上方分数来自官方整场景聚合结果，不是每一帧的即时分数。

推动 GIF 中，其他 agent 属于规划起点时刻的观测；轨迹上的未来点不能直接与这些静止外框做未来碰撞判断。闭环 GIF 显示各时刻的实际观测，可用于观察当时的交互。两类动画均不以 ADE 或 GT 接近程度作为评价依据。

## 使用

在已有 nuPlan 实验环境、仓库根目录运行。使用现有 Matplotlib、NumPy、Pillow 和仓库自带中文字体；无需 ImageMagick、ffmpeg 或浏览器。

先生成一个场景的推动动画，复用已有 `payload.json`，不运行推理：

```bash
python scripts/score_context_gif.py refinement --cohort I01
```

生成同一场景的 DP / G K20 闭环对照动画，从现有 simulation log 提取每一帧的自车和 agent，不重跑闭环：

```bash
python scripts/score_context_gif.py closed-loop --cohort I01 --ks 20
```

生成全部已采集场景的推动 GIF，以及 K20/50/200 的闭环 GIF：

```bash
python scripts/score_context_gif.py all --cohort all
```

默认路径：

- `--source /home/data/wyk/dataset/nuplan/exp/score_G_static_context_1012_v1`
- `--suite /home/data/wyk/dataset/nuplan/exp/score_full_k_1009_v1`
- 输出为 `SOURCE/gif_context_v1`，可用 `--output` 改目录。
- 单次规划源文件位于 `SOURCE/captures/<case>/frame_<index>/payload.json`。

常用参数：

```bash
# 指定已有规划帧、展示哪些已保存的 K，每张快照停留 1 秒。
python scripts/score_context_gif.py refinement --cohort I01,W01 \
  --frames 0 --refine-ks 5 20 50 200 --hold 1 --agent-ids

# 闭环 10 fps，1 倍速，自车周围留 35 米视窗边距。
python scripts/score_context_gif.py closed-loop --cohort I01,W01 \
  --ks 20 50 --fps 10 --speed 1 --radius 35

# 已提取过的同一组 K，可复用观测 JSON 调整速度、字号分辨率等。
python scripts/score_context_gif.py closed-loop --cohort I01,W01 \
  --ks 20 50 --render-only --speed 2 --dpi 120
```

`--agent-ids` 显示 track token 最后 6 位。所有观测对象都会进入绘图，视窗外对象自然裁剪；没有只保留前 N 个 agent。GIF 默认无限循环，分辨率由 `--dpi` 控制，默认 100。逐帧写文件，不将整段 RGB 图像保留在内存中。长闭环或全部场景生成可能较慢，建议先看 I01/W01。

`--render-only` 要求原 suite、源日志和完全相同的 K 集仍可访问，以校验来源。若扩大 `--radius`，需去掉 `--render-only` 重新查询更大范围的地图。闭环 JSON 保存在 `closed_loop_data/<case>.json`；每个 GIF 旁保存 `.json`，记录输入文件哈希、K 或播放参数和时序规则。

若提示缺少 capture，可以先按原流程采集选定场景，再运行 GIF 命令：

```bash
python scripts/expanded_score_context.py capture --cohort I01 --frames 0 --device cpu
python scripts/score_context_gif.py refinement --cohort I01 --frames 0
```

只有这一步补采集会运行 DP 和 refine；已有 capture 转 GIF、已有日志转闭环 GIF 均不需要 GPU。新入口复用原来的 `ScenarioScores`，因此目前对应 S05-G；不能将 S10-G 的结果目录直接替换进去并沿用标签。

## 验证

```bash
python -m unittest discover -s tests -p 'test_score_context_gif.py' -v
```

测试使用明确标注的合成场景，覆盖不同起止时间和采样频率、agent 消失、结束后不保留动态对象、统一坐标变换、实际 K 更新箭头、GIF 帧数/时长/可解码性和写入失败保护。合成测试输出不是 nuPlan 实验结果；真实 GIF 必须在有上述 capture / simulation log 的环境中生成。
