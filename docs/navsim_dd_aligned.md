**NAVSIM Score 的 DD 对齐训练**

这次实验在独立工作树 `/home/data/wyk/Score_Function_NAVSIM_DDAligned` 中从头训练 100 epoch。配置为 `configs/navsim_dd_s10_g_ddaligned.json`，继续使用已验收的 `/home/data/wyk/dataset/navsim/exp/dd_score_cache_v1` 和 Python 环境 `/home/data/wyk/envs/diffusion_drive`。

训练优化设置按 DiffusionDrive 提交 `9b52ed0ec06b073d82d6f392ab084c7b301c8681` 核对。预训练 DD 前端保持冻结，独立 Score 网络、固定 σ 的 DSM 目标、轨迹归一化、条件缓存与验证流程沿用当前 Score 实现。项目 seed 使用 `20260922`，官方 DD 训练默认 seed 为 `0`。这些差异需要在实验记录中保留。[官方训练说明](https://github.com/hustvl/DiffusionDrive/blob/9b52ed0ec06b073d82d6f392ab084c7b301c8681/docs/train_eval.md)

| 设置 | 此次配置 |
|---|---|
| GPU / batch | 8 张可见 GPU，每卡 64，全局完整 batch 512 |
| Epoch | 100 |
| 优化器 | AdamW，峰值学习率 6e-4，weight decay 1e-4 |
| 学习率计划 | 前 3 epoch 线性 warmup，随后按 epoch 更新 cosine，min_lr 1e-6 |
| Warmup 学习率 | 第 1–3 epoch 分别为 2e-4、4e-4、6e-4 |
| 精度 / 梯度 | FP16 mixed precision + GradScaler，gradient_clip=0 |
| 权重选择 | EMA decay 为 0，验证与最终导出使用 raw weights |
| 尾批次 | drop_last=False，保留尾批次，分布式采样按各 rank 等长要求补齐 |
| Score 参数 | σ=0.1，γ=0.1，固定直接 score 架构 |
| 随机种子 | 训练与验证均为 20260922 |

Cosine 的最小学习率在第 100 epoch 完成后的调度边界达到。最后一个训练 epoch 使用对应 epoch 起点的学习率。γ 是后续轨迹 refine 的步长系数，在训练配置和导出中记录。

先在目标节点执行预检。脚本读取 `TRAINING_READY.json`，核对 manifest 摘要、缓存身份和 103,288 个训练/验证样本总数，检查八张可见 CUDA 设备，并用真实训练缓存中的 64 个样本在可见 GPU 0 上执行生产代码的 FP16 forward、DSM backward 和 GradScaler optimizer step。首次 overflow 会对同一批次降低 scale 重试，八次仍无法更新则报错。

```bash
cd /home/data/wyk/Score_Function_NAVSIM_DDAligned
bash scripts/train_navsim_dd_aligned.sh --check
```

看到 `CHECK OK` 后启动。直接运行训练命令时也会先做同一预检。日志放在训练输出目录的旁边。

```bash
cd /home/data/wyk/Score_Function_NAVSIM_DDAligned
nohup bash scripts/train_navsim_dd_aligned.sh \
  >> /home/data/wyk/dataset/navsim/exp/dd_score_ddaligned_v1.log 2>&1 < /dev/null &
tail -f /home/data/wyk/dataset/navsim/exp/dd_score_ddaligned_v1.log
```

输出为 `/home/data/wyk/dataset/navsim/exp/dd_score_ddaligned_v1`，相邻的 `.launcher.lock` 防止重复启动。训练目录由训练器创建。初次启动要求输出目录为空，已有该实验的中断运行使用相同入口追加 `--resume`，从最近完整 epoch 的 `last.pt` 恢复 optimizer、GradScaler 和各 rank 的随机状态。

```bash
cd /home/data/wyk/Score_Function_NAVSIM_DDAligned
bash scripts/train_navsim_dd_aligned.sh --check --resume
nohup bash scripts/train_navsim_dd_aligned.sh --resume \
  >> /home/data/wyk/dataset/navsim/exp/dd_score_ddaligned_v1.log 2>&1 < /dev/null &
```

恢复时必须匹配本次配置、缓存与八卡协议。旧 200 epoch 实验的 checkpoint 和输出目录不能用于这个入口。若本机同用户还运行着输出到 `dd_score_s10g_v1_e200` 的确切训练进程，脚本会显示相关 PID 并退出，需要先处理该运行。脚本使用调用终端当前的 `CUDA_VISIBLE_DEVICES`，如需指定物理卡，在启动前自行设置。

主结果取 `final_raw.pt`，`best_raw.pt` 按验证 DSM 选择，`last.pt` 仅供恢复。兼容接口保留的 `final_ema.pt`、`best_ema.pt` 在这组 EMA decay 为 0 的配置下同样存放 raw weights，导出字段 `weight_kind=raw`。评测时优先使用显式的 `final_raw.pt`。每个 epoch 的学习率、训练/验证 DSM 和 AMP 跳过更新数记录在 `history.json`。

若共享 NFS 的缓存读取有延迟，可以将已验收的完整缓存复制到本节点的 SSD 或 `/dev/shm`，再通过绝对路径 `SCORE_CACHE_DIR` 指定副本。原有 manifest、分片和验收记录必须原样保留，复制不会重新提取特征。训练器的恢复签名校验缓存内容身份，不绑定缓存所在路径。

```bash
mkdir -p /dev/shm/wyk_dd_score_cache_v1
rsync -a --checksum --info=progress2 \
  /home/data/wyk/dataset/navsim/exp/dd_score_cache_v1/ \
  /dev/shm/wyk_dd_score_cache_v1/

cd /home/data/wyk/Score_Function_NAVSIM_DDAligned
nohup env SCORE_CACHE_DIR=/dev/shm/wyk_dd_score_cache_v1 \
  bash scripts/train_navsim_dd_aligned.sh \
  >> /home/data/wyk/dataset/navsim/exp/dd_score_ddaligned_v1.log 2>&1 < /dev/null &
```

只复制 Score 条件缓存，主要张量约 3.2 GiB；无需复制完整 NAVSIM 传感器数据。`/dev/shm` 属于本节点/容器的临时内存存储，重启后若副本丢失，可重新复制后以 `--resume` 继续。checkpoint 和日志仍写入 `/home/data/wyk/dataset/navsim/exp`。

每轮日志额外报告 rank 0 的 `elapsed`、`train`、`validation` 秒数；总耗时包含保存 checkpoint 与同步，训练阶段包含加载与通信。用多轮耗时比较 NFS 与本地缓存效果，不能仅凭 GPU 活跃率或一次验证变慢认定瓶颈。

训练后的配对评测使用 `scripts/eval_navsim_dd_aligned.py`。先查看训练和评测状态：

```bash
python3 scripts/eval_navsim_dd_aligned.py status
```

完成 100 epoch 后，用原 Conda 环境启动八卡 navtest 评测：

```bash
nohup /home/data/wyk/envs/diffusion_drive/bin/python -u \
  scripts/eval_navsim_dd_aligned.py run \
  >> /home/data/wyk/dataset/navsim/exp/dd_score_ddaligned_eval.log \
  2>&1 < /dev/null &
```

入口检查 `dd_score_ddaligned_v1/final_raw.pt` 的 epoch、raw 权重标记和噪声尺度。从 `dd_score_e200_eval_latest.txt` 指向的已完成评测中，按 token 复用 K0 的原始轨迹和各项分数，只重新计算新 Score 的 K20；参数固定为 seed=20260922、gamma=0.1。复用前核对 DD 资产身份、种子、数据及 metric cache 路径、全部 token、分片完整性和旧汇总分数，并将来源文件摘要记录到新结果。每个分片首场景保留一次原生前向一致性检查，不重新计算 K0 的 PDM 分数。

八个分片全部成功后才合并，并要求覆盖 12,146 个场景。评测会创建独立输出目录，启动信息和进度持续写入上述日志。再次执行 `status` 可查看进度或完整评分，并在存在历史结果时显示与旧配置 100/200 轮评测的汇总对照。结果仍包含标明复用的 K0，用于逐场景计算 K20−K0、改善/退化/持平数量。

检查结果时确认场景覆盖完整，核对 K0 基线，再看 K20−K0 是否为正。历史汇总 K0 相等只能作为快速核对，严格的逐场景一致性仍需比较保存的 token 和原始轨迹。仅验证 DSM 降低不能替代 PDMS 配对评测。

100 轮对齐版的用户回传结果为 12,146 场景，复用 K0=88.0337，K20=87.9575，
差值 −0.0762 分；改善/退化/持平为 4066/3793/4287。它仍未超过固定 K0，
因此后续训练的效果要以新 checkpoint 的 K20 评测判断。

追加 150 轮的入口为 `scripts/continue_navsim_dd_aligned.py`。它从已完成的
`dd_score_ddaligned_v1/last.pt` 完整恢复，训练到总计 250 轮，输出为
`dd_score_ddaligned_v1_e250`。原 100 轮目录保留。恢复内容包含 raw 参数、AdamW
状态、GradScaler、各 rank RNG、历史与 best 权重；不能用 `final_raw.pt` 代替完整断点。

原 100 轮 cosine 的最后一轮 LR 约为 `1.157067e-6`，下一轮自然到达下限
`1e-6`。续训配置以 `lr_schedule_epochs=100` 保留该曲线，101–250 轮固定
`1e-6`，不重新 warmup。它是低学习率续训，不等同于从头训练 250 轮的 cosine
实验。直接把原 cosine 的 horizon 改成 250 会使第 101 轮 LR 回升到约
`3.99547e-4`，因此普通 `--resume --epochs 250` 仍会被配置签名检查拒绝。

```bash
/home/data/wyk/envs/diffusion_drive/bin/python scripts/continue_navsim_dd_aligned.py check
nohup /home/data/wyk/envs/diffusion_drive/bin/python -u \
  scripts/continue_navsim_dd_aligned.py run \
  >> /home/data/wyk/dataset/navsim/exp/dd_score_ddaligned_v1_e250.log \
  2>&1 < /dev/null &
python3 scripts/continue_navsim_dd_aligned.py status
```

首次运行原子创建独立续训目录；重复执行相同 `run` 从该目录最近完整 epoch
恢复，并用锁阻止重复启动。存在 `/dev/shm/wyk_dd_score_cache_v1/TRAINING_READY.json`
时优先使用该缓存，否则使用原 NFS 缓存；`SCORE_CACHE_DIR` 可显式指定。
两种位置都必须与原训练缓存身份一致。完成后使用新目录的 `final_raw.pt`
做 K20 评测，仍可复用先前的 K0。
