# ChainFlow-VLA 原生复现：独立交接任务

## 任务边界

本任务只负责 ChainFlow-VLA 原生基线的准备、复现和可复用结果交付。它不依赖 Score 分支实现，也不训练 Score、不改 DSM、不接 refine。另一个 agent 正在 `codex/navsim-chainflow-score-1110` 分支实现 Score 条件缓存、完整训练和配对评测。

请直接完成已授权的环境、资产准备与原生评测。先检查已有文件与进程，复用正确产物；不要重复下载或覆盖已有实验。没有服务器执行权限时，交付经过核对的脚本与命令，明确区分本地检查和服务器实测。

## 已确定的实验协议

- 官方仓库：https://github.com/AFARI-Research/ChainFlow-VLA
- 固定提交：`0f3565f66546139a0183bc5807d5f7234ae2d4b0`。
- 采用 **navtrain 85k** Stage 2 权重；不要换成 trainval 103k 权重。
- 官方发布权重：`weights/stage_2/stage2_navtrain_9382.ckpt`。
- 官方 HF 模型仓库：`AFARI-Research/ChainFlow-VLA`，已核对 revision `d00244bd0e2b12575251f4f47370034831b0d6c1`。
- 此 checkpoint 大小 569,057,402 bytes。下载后必须核对对应固定 revision 的 LFS SHA-256，不能以“文件存在”判成功。
- README 对该发布权重标注 PDMS 约 93.8；论文 navtrain 表格写 93.6。94.8/94.85 对应 trainval 训练的版本，不能作为本任务验收目标。真实得分以固定权重和配置的完整 navtest 实测为准。
- 固定配置：`navsim/planning/script/config/common/agent/chainflow_vla_stage2.yaml`。当前发布配置是 DDIM、12 步、64 候选、原生 PDM scorer 选 top-1。保留官方 `diffusion_eval_noise_seed=27` 和 per-token 噪声实现，不要为“统一项目 seed”改变原生采样。
- 不改变候选生成、残差扩散、候选评分或最终选择规则；不加入我们的 Score/refine。
- 评测完整 navtest：12,146 个唯一 token。每个场景都需成功；失败不能静默排除后计算均值。

## 用户服务器目录（强约束）

| 内容 | 路径 |
|---|---|
| 根目录 | /home/data/wyk |
| ChainFlow 代码 | /home/data/wyk/ChainFlow-VLA |
| 独立 Conda 环境 | /home/data/wyk/envs/chainflow |
| 现有 DD 代码（保持不变） | /home/data/wyk/DiffusionDrive |
| 现有 DD 环境（保持不变） | /home/data/wyk/envs/diffusion_drive |
| 现有 nuplan-devkit | /home/data/wyk/nuplan-devkit |
| NAVSIM 原始数据 | /home/data/wyk/dataset/navsim |
| 共享地图 | /home/data/wyk/dataset/nuplan/maps |
| 现有 v1 metric cache | /home/data/wyk/dataset/navsim/exp/metric_cache |
| 原生基线结果 | /home/data/wyk/dataset/navsim/exp/chainflow_native_navtest_v1 |
| VLM 特征缓存 | /home/data/wyk/dataset/navsim/chainflow_vlm_cache |
| 下载日志与记录 | /home/data/wyk/dataset/navsim/exp/chainflow_prepare |
| 交接记录 | /home/data/wyk/dataset/navsim/exp/chainflow_native_navtest_v1/reproduction_ready.json |

不要把数据集放进代码目录或 src/，不要重新下载 nuplan-devkit，不要改现有 DD/nuPlan 环境。先检查官方内置 devkit 与现有 devkit 的兼容性；如需少量适配，明确记录，不能偷偷切换评分协议。

## 数据实际上已经有多少

现有 DD Reader 缓存已完成：
- train：85,109
- val：18,179
- 总计：103,288
- navtest：12,146，已完成 DD 全量评测。

现有 `navtrain_current_*.tgz` 下载覆盖 navtrain 所需当前帧子集，metadata 是 trainval；这不等于缺少 18k 验证集。ChainFlow 的 navtrainval 配置只多 `include_val_logs_in_train: true`，不需要另下一套 103k 数据。

不要直接运行官方 `download_trainval.sh`：它下载的是完整 OpenScene 原始数据，不是“补验证集”，且当前脚本 camera 循环有从 123 开始的残留。

必须另查 **传感器覆盖**：
- 主视觉分支需要当前帧 `cam_f0, cam_l0, cam_r0, cam_b0`，默认不使用 LiDAR。
- 现有 DD 成功不证明后视相机 `cam_b0` 的所有文件齐全。按真实 scene/token 检查缺项后再定向补充。
- VLM 缓存使用前视图与 ego 历史/导航等输入。不要把 metadata 中的历史 ego 状态误当作必须下载所有历史图像。
- 后续 Score 使用已有 train/val 记录和物理 GT，因此下载和缓存必须覆盖两部分；navtest 只用于最终评测。
- 现有 metric cache 要核查与该固定代码的 v1 scorer、采样时域和对象 schema 兼容性。无法证明兼容时在新目录重建，不能覆盖旧 cache。

## 资源限制：不能干扰已有任务

用户 2026-11-10 提供的状态：
- GPU 0–3：约 73 GiB 级占用，剩余约 8.4 GiB。
- GPU 4–7：约 54 GiB 级占用，剩余约 26.7 GiB。
- 全部 GPU utilization：99–100%。

这些值只是当时快照，启动前必须重新检查。显存有余量不等于有空闲算力，不能承诺忙卡并行完全无影响。

默认执行方式：
1. 下载、环境安装、CPU 数据覆盖检查先以少线程/低优先级并行。
2. GPU 阶段等待空闲卡；按可用卡数量分片评测，不强求 8 卡。
3. 如用户明确选择共享模式，只使用指定 GPU 池并设置本任务的 PyTorch 分配器显存预算；预算不是全进程显存硬隔离，也不是算力保证。
4. 不 kill、暂停、迁移其他任务，不修改 MPS、MIG、功耗、时钟或其他进程。
5. 避免 Ray 批量抢占 CPU/GPU；官方 eval 可设 `enable_ray=false`，每个进程单卡、单场景推理。
6. 监控命令要支持持续查看；后台脚本即时 flush 日志并提供 PID、阶段与完成场景数。

## 官方模型与 VLM 资产

- DINO 权重：`timm/vit_small_patch14_reg4_dinov2.lvd142m`，本地文件放 `ChainFlow-VLA/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors`。
- 原生模型含已训练 LoRA、learned scene tokens、neck，必须一起加载。不能用通用 DINO 特征替代它们。
- VLM feature dataset：https://huggingface.co/datasets/AFARI-Research/ChainFlow-VLA-VLM-Feature-Cache
- 国内官方备份：https://modelscope.cn/datasets/AFARI/ChainFlow-VLA-VLM-Feature-Cache
- 缓存字段 `last_hidden_state`，隐藏维 1536，保留完整 token 序列及有效长度，不改成均值池化。
- 官方 docs/train_eval.md 支持直接下载已发布缓存，或从 ReCogDrive-VLM-2B 构建。

**实际成本**：已核对官方发布资产列表：24 个 trainval 压缩分片 + 1 个 navtest 压缩包，合计 **1,035,243,638,403 bytes（约 1.035 TB 压缩量）**；navtest 单包约 109 GB。解压还需要额外空间，后续可训练 mmap 缓存也需空间。不能声称下载很小或很快。

先核对磁盘、现有缓存和网络速度，再选择发布缓存或本地构建路线。若改为本地构建，要核对权重、prompt、图像预处理、历史状态和缓存格式与发布流程一致，并记录差异；不能把不同特征当成同一实验。两种路线均不需要重新微调 VLM。记录 VLM 权重的训练数据来源；不能未经核实声称其预训练完全没见过 val。

下载要求：
- 继承用户已配置的代理，不打印代理账号密码。
- 公司 CA 可能失败；可提供显式单次 `--insecure` 选项，不全局关闭 Git/Conda 校验、不写入凭据。
- 断点续传核对 HTTP 状态与 Content-Range；检查最终大小和 SHA-256。
- 防止把 HTML/代理错误页当作数据，校验失败只隔离坏文件。
- tar/zstd 解压防路径逃逸；任务锁、原子完成标记、可恢复。
- 不向用户保证“一定不会报错”，提供失败位置与恢复命令。

## 复现验收与交付

先做少量真实场景的加载与原生推理检查，再完整评测；这只是管线验收，不是方法消融实验。

必须交付：
1. 固定代码 commit、原生配置解析后的内容与哈希、权重来源/revision/大小/SHA-256、实际依赖版本、设备与采样设置。
2. 四路相机、地图、VLM 缓存和 metric cache 覆盖报告。
3. 每个 navtest token 的原生 **top-1 原始轨迹**，保持原生浮点结果（建议逐场景 npy 或 npz，形状 [8,3]，列 x/y/heading，ego 局部坐标，0.5s 间隔、4s 时域）。从实际评分的同一次 forward 捕获，不另采样一遍生成“对应轨迹”。
4. 同一 token 的逐场景 PDMS 分项与 valid 状态，另有汇总 CSV/JSON。数值保存 0–1，展示乘100；去掉 average 汇总行再计数。
5. 12,146 个唯一 token 全部有效的证据，mean PDMS 与各分项，失败数量必须为0。
6. 标明与发布约93.8的差距。若偏差明显，保留失败/偏差状态并定位原因，不用放宽门限或丢场景宣布复现成功。
7. 逐场景轨迹、metrics、token清单与元数据的内容哈希，供后续 Score 原样复用 K0；绝不能复用 DD 的 88.0337 K0。
8. 持续进度查看命令、恢复命令、全部输出绝对路径。

推荐交接 JSON（schema是双方约定，不能填未验证的值）：

```json
{
  "schema": "chainflow-native-reproduction-v1",
  "state": "complete",
  "upstream_commit": "0f3565f66546139a0183bc5807d5f7234ae2d4b0",
  "training_split": "navtrain",
  "evaluation_split": "navtest",
  "count": 12146,
  "valid_count": 12146,
  "score_scale": "0..1",
  "mean_pdms": "<actual numeric result>",
  "checkpoint": "<absolute path>",
  "checkpoint_sha256": "<actual digest>",
  "backbone": "<absolute path>",
  "backbone_sha256": "<actual digest>",
  "vlm_cache": "<absolute path>",
  "vlm_provenance": "<absolute path>",
  "resolved_config": "<absolute path>",
  "resolved_config_sha256": "<actual digest>",
  "native_noise_seed": 27,
  "metric_cache": "<absolute path>",
  "metrics": "<absolute per-token metrics file>",
  "trajectories": "<absolute per-token raw trajectory directory/index>",
  "token_index": "<absolute path with tokens/log names/file hashes>"
}
```

上述 angle-bracket 字符串只是示例占位符；真实交接文件应使用实际值，不能作为 ready 标记提交。

## 与 Score agent 的接口

- 原生复现 agent：负责原生环境、资产、传感器检查、K0 与 provenance。
- Score agent：负责条件适配与缓存、DSM500轮训练、每50轮ckpt、最终EMA的K=5/10/15/20/30配对评测。
- 条件接口为视觉 scene [64,256]、ego [256]、完整 VLM [L,1536]；这些条件不包含未来真值或评测分数。
- 共享实验的原始数据、地图和权重，代码提交独立分支。不要同时修改对方新建文件。
- 已发布复现脚本若不能直接满足轨迹交付，可在独立脚本中加只读 forward hook；不得改变 forward 结果。后续统一逐场景文件格式可以通过无损导入完成。
- 完成时将 reproduction_ready.json、结果摘要和精确 GitHub commit 交给 Score agent。不要仅回复“93.8 已复现”而不给文件。

## 快速给另一个 agent 的指令

> 请按本任务书独立复现 ChainFlow-VLA 的 navtrain 85k 官方模型，完成环境和资产准备、数据覆盖检查、12,146场景原生K0评测及逐场景轨迹/指标交接。当前GPU已有任务，遵守资源约束，不操作其他进程。Score模块由另一个agent负责，不需要你训练或修改。先检查已有内容与资产下载成本，再推进可执行步骤；最终提交脚本与provenance，明确哪些已经服务器实测。
