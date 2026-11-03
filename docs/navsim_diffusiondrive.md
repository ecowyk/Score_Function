# DiffusionDrive + Score Function（NAVSIM v1）

第一阶段以 DiffusionDrive 作者发布的模型生成初始自车轨迹。一个独立的冻结 DD conditioner 为 score 分支提供场景条件。Score 仅在专家轨迹上训练一次。后续换初始规划器时，可复用相同的 conditioner、score checkpoint 和物理轨迹接口。所有命令在 Linux Bash 下运行。

## 准备环境与权重

使用原版 [hustvl/DiffusionDrive](https://github.com/hustvl/DiffusionDrive) 自带的 NAVSIM v1 代码。固定提交为 `9b52ed0ec06b073d82d6f392ab084c7b301c8681`。`navsim` 应从该 checkout 导入。安装依赖时可以参考[作者说明](https://github.com/hustvl/DiffusionDrive/blob/main/docs/train_eval.md)。已核对的组合包括 Python 3.9、Torch 2.0.1、torchvision 0.15.2、timm 0.9.16、diffusers 0.29.2、huggingface-hub 0.25.2 和 nuPlan devkit v1.2。

以下路径按服务器情况设置。`SF_ROOT` 指向当前 Score_Function 分支，`NAVSIM_DEVKIT_ROOT` 指向固定版本的 DiffusionDrive。

```bash
export SF_ROOT=/path/to/Score_Function
export NAVSIM_DEVKIT_ROOT=/path/to/DiffusionDrive
export OPENSCENE_DATA_ROOT=/path/to/openscene
export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps
export NAVSIM_EXP_ROOT=/path/to/navsim/exp
export DD_CKPT=/path/to/diffusiondrive_navsim_88p1_PDMS
export DD_ANCHOR=/path/to/kmeans_navsim_traj_20.npy
export DD_BACKBONE=/path/to/resnet34.a1_in1k/pytorch_model.bin
cd "$SF_ROOT"
git -C "$NAVSIM_DEVKIT_ROOT" rev-parse HEAD
python -m pip install --no-deps -e "$NAVSIM_DEVKIT_ROOT"
python -m pip install --no-deps -e "$SF_ROOT"
DD_ARGS=(--checkpoint "$DD_CKPT" --anchor "$DD_ANCHOR" --backbone "$DD_BACKBONE")
```

权重来自 [DiffusionDrive NAVSIM checkpoint](https://huggingface.co/hustvl/DiffusionDrive/tree/main)、[20 个聚类 anchors](https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_88p1_PDMS_Eval_file/kmeans_navsim_traj_20.npy) 和 [ResNet-34](https://huggingface.co/timm/resnet34.a1_in1k/tree/main)。DD checkpoint 的发布文件名没有扩展名。在内网使用预下载的文件；timm 优先尝试其预训练模型缓存，可设置 `HF_HUB_OFFLINE=1` 使用本地资产。OpenScene 根目录包含 `navsim_logs/{trainval,test}`、`sensor_blobs/{trainval,test}`；只跑 mini 单场景时使用对应 `mini` 子目录。

## 第一步：核对一个真实 scene

```bash
python -m score_function.navsim_dd.check_scene "${DD_ARGS[@]}" --device cuda:0
```

只需 mini 数据时追加 `--split mini`。已有特定样本时可提供 `--log LOG --token TOKEN`。此命令不需要 score 权重或 metric cache。它用原生输入构造器处理相机、LiDAR 和状态，分别比较独立 conditioner 与原 DD 内部的 `keyval[:, :-1]` 和 `status_encoding`，并检查在相同采样噪声下读取的 top-1 与直接调用原 agent 逐元素一致。期望 `scene_shape=[1,64,256]`、`status_shape=[1,256]`、`raw_shape=[1,8,3]` 及 `native_poses_match=true`。失败时命令以非零状态退出。

在作者发布的 DD checkpoint 上验证过 mini 日志 `2021.06.28.16.29.11_veh-38_01415_01821` 的 token `1906410e9fea5257`。CPU 下 C/g 最大绝对误差均为 0，top-1 index 为 8，原始轨迹逐元素一致。这是输入与模型接口的验收。

## 训练缓存与训练

缓存使用官方 `navtrain` filter 与 DiffusionDrive 的 train/val 日志列表。条件仅来自原生输入，监督目标是 8 个未来后轴位姿（0.5–4.0 s）。缓存把专家航向变为 cos/sin，xy 统计量只由训练日志计算，角度通道保留单位尺度。每条 C/g 分别以 FP16 存储，训练读取 FP32；部署时使用同样的条件精度。cache 清单记录日志划分、输入资产摘要和数据来源。中断后用原命令重跑，复用完整且校验通过的分片。只用小缓存做接口测试时，两个 split 都需要有数据。

```bash
python -m score_function.navsim_dd.cache "${DD_ARGS[@]}" --device cuda:0 \
  --cache-dir "$NAVSIM_EXP_ROOT/dd_score_cache_smoke" --max-per-split 8
python -m score_function.navsim_dd.train --cache-dir "$NAVSIM_EXP_ROOT/dd_score_cache_smoke" \
  --output "$NAVSIM_EXP_ROOT/dd_score_train_smoke" --epochs 1 --batch-size 2
```

全量训练用独立目录，八卡示例：

```bash
export DD_CACHE="$NAVSIM_EXP_ROOT/dd_score_cache_v1"
export DD_RUN="$NAVSIM_EXP_ROOT/dd_score_s10g_v1"
python -m score_function.navsim_dd.cache "${DD_ARGS[@]}" --cache-dir "$DD_CACHE" --device cuda:0
torchrun --standalone --nproc-per-node=8 -m score_function.navsim_dd.train \
  --cache-dir "$DD_CACHE" --output "$DD_RUN" --config "$SF_ROOT/configs/navsim_dd_s10_g.json"
```

默认每卡 batch 64，八卡全局 batch 512，100 个 epoch，固定 σ=0.1；这是第一版预算。训练复用仓库中的直接 score、DSM、全局时间注意力、EMA 和验证流程。在同一输出目录带 `--resume` 可从最近的完整 epoch 恢复。`history.json` 记录训练与验证 DSM，`run.json` 记录每个坐标上噪声的物理尺度。`final_ema.pt` 是最终 EMA，`best_ema.pt` 按验证 DSM 选择，`last.pt` 用于恢复训练。建议先在独立验证日志上确定实验配置，再报告 navtest。

## Paired PDMS：同一条原生 top-1 的 K=0 与 K=20

先用作者代码生成 NAVSIM v1 的 metric cache。和 score 训练特征缓存分别放在不同目录；数据目录选用绝对路径。

```bash
export METRIC_VAL="$NAVSIM_EXP_ROOT/metric_cache_dd_val"
export METRIC_TEST="$NAVSIM_EXP_ROOT/metric_cache_dd_test"
VAL_LOGS=$(python -c 'import json; from score_function.navsim_dd.cache import build_navtrain_filter; print(json.dumps(build_navtrain_filter("val").log_names, separators=(",", ":")))')
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
  train_test_split=navtrain "train_test_split.scene_filter.log_names=$VAL_LOGS" "cache.cache_path=$METRIC_VAL"
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
  train_test_split=navtest "cache.cache_path=$METRIC_TEST"
```

特征缓存与 metric cache 的路径分开。配对评测读取各节点生成的 metadata CSV，检查 token 覆盖和路径冲突。上游原生 evaluator 从一份 metadata CSV 读取绝对路径，使用该入口时需确保单份清单完整。

先验证一小批，再做正式配对。两次实验使用不同输出目录。

```bash
python -m score_function.navsim_dd.evaluate "${DD_ARGS[@]}" --native-only --verify-native \
  --split val --metric-cache "$METRIC_VAL" --max-scenes 3 --output "$NAVSIM_EXP_ROOT/dd_native_smoke"

python -m score_function.navsim_dd.evaluate "${DD_ARGS[@]}" --score-checkpoint "$DD_RUN/final_ema.pt" \
  --split val --metric-cache "$METRIC_VAL" --ks 0 1 5 10 20 --verify-native \
  --output "$NAVSIM_EXP_ROOT/dd_score_val_v1"

python -m score_function.navsim_dd.evaluate "${DD_ARGS[@]}" --score-checkpoint "$DD_RUN/final_ema.pt" \
  --split navtest --metric-cache "$METRIC_TEST" --ks 0 20 \
  --seed 20260922 --output "$NAVSIM_EXP_ROOT/dd_score_navtest_v1"
```

每个 token 的随机流由 seed 和 token 决定。原 DD 只生成一份 top-1；所有 K 从这一份 float32 轨迹出发。K0 直接返回原始位姿。K>0 的每一步沿用 `u += gamma * sigma**2 * score(u,C,g)` 并对物理航向投影；g 是自车状态与导航指令的编码。每个样本输出原生 mode index、原始及各 K 的位姿、PDM 子项和逐项差值。最终 `summary.json` 的分数在 0–1 尺度，论文对照需乘 100。作者的 88.1 为公开参照，项目的增益以同批运行的 K0 为基准。评测失败或缺失样本会报出错误，只有完整分片产生 `complete.json`。

多 GPU 可以用同一参数分别启动 `--num-shards 8 --shard-index 0…7`，每个进程指定自己的 `--device cuda:N`。待全部成功后合并：

```bash
python -m score_function.navsim_dd.evaluate --merge --output "$NAVSIM_EXP_ROOT/dd_score_navtest_v1"
```

## 代码入口

| 文件 | 职责 |
|---|---|
| `score_function/navsim_dd/conditioner.py` | 读取官方 DD 前端、独立提取 C/g、和原生中间值比较 |
| `score_function/navsim_dd/cache.py` | 官方日志划分、分片缓存、训练集归一化 |
| `score_function/navsim_dd/train.py` | PyTorch / torchrun 训练，EMA 导出和断点恢复 |
| `score_function/navsim_dd/runtime.py` | NAVSIM 位姿适配、checkpoint 检查、多 K 共用一条更新路径 |
| `score_function/navsim_dd/evaluate.py` | 原版 top-1 和官方 PDMS 的逐 token 配对评测 |
| `score_function/navsim_dd/agent.py` | 可由原版 NAVSIM Hydra evaluator 实例化的 AbstractAgent |

使用官方单配置 evaluator 时，Hydra 需同时看到 DD 原配置与此仓库的 agent 配置：

```bash
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
  "hydra.searchpath=[pkg://navsim.planning.script.config.common,file://$SF_ROOT/configs/navsim_dd_hydra]" \
  train_test_split=navtest agent=diffusiondrive_score_agent agent.steps=20 \
  "agent.score_checkpoint=$DD_RUN/final_ema.pt" "metric_cache_path=$METRIC_TEST" \
  worker=ray_distributed agent.device=cuda experiment_name=dd_score_v1
```

该接口供官方 agent/提交工具调用。上面的 paired evaluator 额外保证每个 K 使用同一次原生采样，因此用于正式的 K0 对 K20 比较。
