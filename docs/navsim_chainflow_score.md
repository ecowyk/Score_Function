# ChainFlow 条件下的 Score 训练与配对评测

原生复现已拆为独立任务，见 [交接任务书](chainflow_native_reproduction_handoff.md)。这里负责冻结条件缓存、Score 训练和复用原生 K0 的评测。代码检查通过不代表已经在用户服务器完成复现、缓存或训练。

## 实验目标与边界

Score 仍然学习加噪轨迹的条件密度梯度。主视觉条件来自官方 85k Stage 2 checkpoint 中已训练的 DINO、LoRA、scene tokens、neck 与 ego 编码；语义条件使用官方完整 VLM hidden states。原生规划器的候选生成、残差扩散和 PDM 候选评分保持不变。Score 仅在原生最终选中的轨迹上运行。

固定官方提交 `0f3565f66546139a0183bc5807d5f7234ae2d4b0`，权重 `stage2_navtrain_9382.ckpt`，原生 token-dependent seed=27。项目实验 seed=20260922 不覆盖原生采样种子。85k 权重的发布参考约93.8，不能混用103k权重的94.8参考。

原生分支默认四路当前帧相机、无 LiDAR。训练时读取冻结特征缓存，因此显存用量不等于端到端图像主干训练的用量。我们不训练 VLM、DINO 或原生规划器。

## 数据和目录

| 内容 | 默认路径 |
|---|---|
| Score 工作树 | `/home/data/wyk/Score_Function_NAVSIM_ChainFlow` |
| ChainFlow 原生代码 | `/home/data/wyk/ChainFlow-VLA` |
| Conda 环境 | `/home/data/wyk/envs/chainflow` |
| 原始数据 | `/home/data/wyk/dataset/navsim` |
| 地图 | `/home/data/wyk/dataset/nuplan/maps` |
| 已有 GT 与划分缓存 | `/home/data/wyk/dataset/navsim/exp/dd_score_cache_v1` |
| 官方 VLM 缓存 | `/home/data/wyk/dataset/navsim/chainflow_vlm_cache` |
| 已完成原生 K0 | `/home/data/wyk/dataset/navsim/exp/chainflow_native_navtest_v1/run_native_stage2_navtrain` |
| 新条件缓存 | `/home/data/wyk/dataset/navsim/exp/chainflow_score_cache_v1` |
| 训练输出 | `/home/data/wyk/dataset/navsim/exp/chainflow_score_v1` |
| 配对评测 | `/home/data/wyk/dataset/navsim/exp/chainflow_score_navtest_v1` |
| 队列记录 | `/home/data/wyk/dataset/navsim/exp/chainflow_score_pipeline` |

train=85,109，val=18,179；继承已有缓存的精确 token、物理 GT 和仅由 train 得到的 normalizer。navtest 的12,146个场景不用于训练。CPU 覆盖检查会检查四路相机和 VLM 文件；张量形状、有限值与内容哈希在生成缓存时进一步检查。

不重新下载 nuplan-devkit，不另套一个全量 OpenScene trainval 数据集。代码、环境和数据保持上面的分离方式。

## 接入已经完成的原生复现

本次外部交接格式是 `chainflow-native-reproduction-v1`：`reproduction_ready.json` 指向逐场景 CSV，以及 `trajectories/token_index.json` 中记录的 NPZ 文件。NPZ 的 `trajectory` 是 float64 `[8,3]`。读取路径一律来自交接文件，保留轨迹精度，不转换并覆盖已有文件。基线为 12,146 场景、PDMS 93.8194331；不是 DD 的88.0337。

接入时验证官方提交、checkpoint/DINO SHA、配置/指标/索引/逐轨迹 SHA、完整 token 集、valid、指标范围和均值。当前评分器先在8个有不同分数的已有 K0 上做 CPU 兼容预检；完整 K 评测时仍逐 token 重评分已有 K0，分项一致才放行。这里重评分的是已经保存的轨迹，不调用原生规划器再次采样。

外部交接没有记录历史逐场景条件哈希、metric-cache 哈希和推理耗时；代码明确保留这些缺项，不能把现在计算的哈希冒充原生评测时的记录。生成条件和配对评测会记录当前内容哈希，并对每个场景执行 K0 指标重放检查。预检通过不代表未来所有文件都不会变化。

继续使用已有 `/home/data/wyk/envs/chainflow` 环境（原生评测使用 PyTorch 2.5.1+cu124）。这个续接流程不运行 `setup_navsim_chainflow.sh`，不重建环境、重新下载权重或 navtest。

训练所需的 train/val VLM 如果已有完整103,288文件则直接复用；缺少时 `--prepare-assets` 只准备官方24个 trainval 压缩分片，从空目录开始共 **926,302,926,122 bytes（约926.3 GB）**，另需解压及冻结条件缓存空间。已经由本下载器校验并解压完成的分片自动跳过。此处补的是VLM hidden states，不是全量OpenScene原始传感器数据。原有 navtest wrapper 路径直接使用，无需搬动或复制。

下载只在本进程使用当前终端代理。默认验证证书；如果公司代理仍要求临时跳过验证，可在下面命令加 `--insecure-downloads`，仅影响训练 VLM 下载。已完成的部分下载遇到错误的 HTTP200 Range响应会保留，不截断重下；最终大小与SHA通过后才解压。需要 `zstandard` 包，缺少时会在开始大文件下载前报错。

## 网络与训练配置

视觉 scene 条件 `[64,256]`、ego `[256]` 与完整变长 VLM `[L,1536]` 保存到 FP16 缓存。VLM 以 ragged offsets 存储，不截断、不预先均值池化。训练时由可学习投影和16个查询 token 读取全部有效 VLM tokens，再与视觉及 ego 条件一起输入 Score 分支。

可训练参数2,845,444；包括条件适配器与 Score 网络。训练张量和参数用 FP32，缓存读取后转 FP32。有效位置 mask 防止 padding 改变条件；推理的条件先按同样 FP16 精度量化再转 FP32。

| 配置 | 数值 |
|---|---|
| 目标 | 直接 score，DSM |
| 轨迹 | 8步 × `(x,y,cosθ,sinθ)` |
| sigma / gamma | 0.1 / 0.1 |
| epoch / 周期 checkpoint | 500 / 每50轮 RAW+EMA |
| 全局 batch | 512 |
| microbatch | 配置文件每卡64；资源队列默认每卡16，通过累积保持全局512 |
| GPU 数 | 1、2、4、8，梯度累积保持全局512 |
| LR | 2e-5起，5轮warmup至2e-4，随后cosine至1e-6 |
| weight decay / clip | 0.01 / 5.0 |
| EMA | 0.999 |
| 验证 | 固定token噪声，每场景4次，逐轮RAW与EMA |
| 增广 | 无静态特征上的伪几何增广 |

每轮166次优化更新，共83,000次；117个尾部样本在每轮随机重排后舍弃。不能把这个预算声称与之前nuPlan的244,398更新相同。物理空间的噪声尺度由已有train normalizer决定，记录在 `run.json`；不悄悄改成nuPlan的2米尺度。

除训练模式的loss之外，记录同一评估模式下固定train子集与val的RAW/EMA DSM、xy和heading分项，方便区分泛化问题与验证协议差异。早期best、每50轮和final均保留；主流程按用户约定评测final EMA，不以val最小的早期checkpoint替代。

## 运行与资源

运行入口为 `scripts/run_navsim_chainflow.py`，以 `--help` 为参数准。完整队列依次验证原生交接、检查数据、生成条件缓存、做真实小批训练/验证预检、训练500轮、自动配对评测K=5/10/15/20/30。预检只检查代码和数据通路，临时模型不进入正式训练输出。

原生交接已经完整时，使用以下命令把数据准备与训练接成一个队列。先把 Score 工作树更新到 `codex/navsim-chainflow-score-1110` 的本次交接适配版本；不要把这些脚本复制进原生 ChainFlow 仓库。

```bash
cd /home/data/wyk/Score_Function_NAVSIM_ChainFlow
mkdir -p /home/data/wyk/dataset/navsim/exp/chainflow_score_pipeline
nohup nice -n 19 /home/data/wyk/envs/chainflow/bin/python -u scripts/run_navsim_chainflow.py run \
  --baseline /home/data/wyk/dataset/navsim/exp/chainflow_native_navtest_v1/run_native_stage2_navtrain \
  --navtest-vlm-cache /home/data/wyk/dataset/navsim/chainflow_vlm_cache/navtest_recogdrive_vlm2B_traj_cache_first_forward \
  --prepare-assets --workers 4 --microbatch-size 16 \
  >> /home/data/wyk/dataset/navsim/exp/chainflow_score_pipeline/queue.log 2>&1 < /dev/null &
echo "Score 队列 PID=$!"
```

顺序是：验证原生交接与评分兼容性 → 补齐训练VLM → 检查相机/VLM完整性 → 冻结条件缓存 → 真实训练预检 → 500轮训练 → final EMA 配对评测。原生代码必须仍是官方提交且 `navsim/` 下没有已跟踪的修改；若复现 agent 曾加过导出补丁，应另建官方提交的干净 git worktree，并通过 `--devkit-root` 指向它，权重路径仍复用已有文件。不自动清理或覆盖复现 agent 的源码。

持续查看：

```bash
python3 /home/data/wyk/Score_Function_NAVSIM_ChainFlow/scripts/run_navsim_chainflow.py status --watch
```

默认 GPU 策略为等待空闲。显存剩余但利用率99–100%时，不承诺并行完全无影响。显式共享策略才使用繁忙设备；显存限制只约束本任务的PyTorch allocator，不是整张卡或整进程的隔离。所有阶段只管理自己启动的子进程，不操作现有任务或GPU全局设置。

原生复现与Score训练使用共享GPU租约目录防止本项目的队列重复选卡。训练续跑保留原有world size、microbatch和累积配置；这些设置变化时拒绝复用原输出。相同命令恢复已完成且校验通过的阶段。

监控使用 `status --watch`；Ctrl+C只退出监控。失败时首先看状态中的阶段日志，修复原因后用原命令恢复。不要删除已完成的原生轨迹、缓存或checkpoint来强制重跑。

## 原生结果交接与配对约束

同时支持本仓库的原生分片JSONL交接和本次外部CSV+NPZ交接。前者保留其严格的历史条件/metric cache哈希校验；后者使用明确的外部来源类型，保留缺失的历史字段，并额外要求当前评分器对已保存 K0 的逐指标重放一致。两者都要求12,146个token的完整集合一致。

所有K从同一次原生 forward 留下的 top-1 轨迹出发，K0 轨迹和原始 CSV 分项精确复用，不重新采样。外部NPZ的float64 K0保持不变；refine内部按训练时FP32计算，仅新增 refine 后结果。`summary.json` 的分数是0–1，展示时乘100。任何场景失败使阶段失败，不能排除失败场景后宣布完成。

## 本地验证范围

已提供CPU测试覆盖：官方forward选择不变、严格权重加载契约、变长VLM/padding、缓存GT与划分保持、损坏/中断恢复、DSM学习、RAW/EMA续训一致性、K0精确复用、全量token与指标校验、资源等待及下载校验。多卡Gloo测试在支持本地通信的主机可运行。

本开发环境没有公司服务器的GPU、相机、地图或正式checkpoint。真实加载、CUDA峰值显存、原生PDMS与完整训练耗时必须在用户节点实测；任务书和脚本不能替代这项验收。
