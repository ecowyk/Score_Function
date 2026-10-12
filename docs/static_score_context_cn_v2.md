# 中文静态轨迹报告 v2

只读取已有 `captures/*/frame_*/payload.json`，无需 nuPlan、模型权重或 GPU，
不重新采集数据，不改动正在运行的训练和评测。

```bash
cd /home/data/wyk/Score_Function_aligned_v1
/home/data/wyk/envs/diffusion_planner/bin/python -u \
  scripts/render_static_context_cn.py
```

默认读取：
`/home/data/wyk/dataset/nuplan/exp/score_G_static_context_1012_v1/captures`

默认输出：
`/home/data/wyk/dataset/nuplan/exp/score_G_static_context_1012_v1/report_cn_v2/`

- `trajectory_report_cn.pdf`：合并报告；原报告保持不变。
- `figures/<场景>/frame_<帧>/`：每个 K 的独立 PNG、每帧的 PDF。
- `index.json`：图件与原始采集数据的对应关系。

默认生成六个优先场景的全部已采集帧，每帧三页，对应 K20、K50、K200。
每页仅显示灰色原始 DP 和蓝色推动后轨迹，采用全景加局部放大。
长直路用上下布局，转弯和静止场景用左右布局。坐标、说明、图例均为中文。
全景与局部图各自保持等比例物理坐标，同一帧的各 K 页面使用相同图界。
细线、小点保留，每点仍对应原始采样时刻，没有放大位移或修改轨迹数据。

只看一帧：

```bash
/home/data/wyk/envs/diffusion_planner/bin/python -u \
  scripts/render_static_context_cn.py --cases I01 --frames 0 \
  --output /home/data/wyk/dataset/nuplan/exp/score_G_static_context_1012_v1/preview_cn_v2
```

包含全部已采集的 K 并加位移曲线附页：

```bash
/home/data/wyk/envs/diffusion_planner/bin/python -u \
  scripts/render_static_context_cn.py --ks 5 20 50 100 200 500 --include-curves \
  --output /home/data/wyk/dataset/nuplan/exp/score_G_static_context_1012_v1/report_cn_full_v2
```

脚本附带专供本报告使用的中文字体子集，离线即可使用，无需 apt、sudo 或下载字体。
字体来源和 SIL Open Font License 见 `scripts/assets/score_plot_cjk.*`。
也可使用 `--font /path/to/NotoSansCJK-Regular.ttc` 或环境变量 `SCORE_PLOT_FONT`。
字体必须覆盖报告字符；缺字会报错，避免输出方框。

图中周车和行人是当前帧位置。小箭头表示同一未来时刻的 DP 点到推动后点的净位移。
局部区域按位置变化选择，不是事故定位。各 K 固定使用同一 DP 闭环输入，
不代表不同 K 的闭环评测后来实际行驶的轨迹。
