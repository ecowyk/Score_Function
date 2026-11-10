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
| 原生 K0 | `/home/data/wyk/dataset/navsim/exp/chainflow_native_navtest_v1` |
| 新条件缓存 | `/home/data/wyk/dataset/navsim/exp/chainflow_score_cache_v1` |
| 训练输出 | `/home/data/wyk/dataset/navsim/exp/chainflow_score_v1` |
| 配对评测 | `/home/data/wyk/dataset/navsim/exp/chainflow_score_navtest_v1` |
| 队列记录 | `/home/data/wyk/dataset/navsim/exp/chainflow_score_pipeline` |

train=85,109，val=18,179；继承已有缓存的精确 token、物理 GT 和仅由 train 得到的 normalizer。navtest 的12,146个场景不用于训练。CPU 覆盖检查会检查四路相机和 VLM 文件；张量形状、有限值与内容哈希在生成缓存时进一步检查。

不重新下载 nuplan-devkit，不另套一个全量 OpenScene trainval 数据集。代码、环境和数据保持上面的分离方式。

## 资产准备

原生复现 agent 可先运行 `scripts/setup_navsim_chainflow.sh --scope native`，只准备模型和 navtest VLM，约109.6 GB压缩资产。Score 训练需要 `--scope all`，全部发布资产合计1,035,243,638,403 bytes，另需解压和 mmap 缓存空间。下载器按已完成校验跳过资产；稳定的 `release_manifest.json` 不随下载进度改变，`assets.json` 记录完成情况。

```bash
cd /home/data/wyk/Score_Function_NAVSIM_ChainFlow
bash scripts/setup_navsim_chainflow.sh --scope all
```

如果公司代理确实需要临时绕过证书校验，增加显式 `--insecure`。继承终端代理，不输出或写入代理密码，不修改全局 Git/Conda 配置。已存在的原生代码提交或环境不符合预期时脚本停止并说明原因，不覆盖它们。

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

在 `--scope all` 资产准备完成后启动；原生K0可以由另一位agent继续跑，Score队列会等待完整交接：

```bash
cd /home/data/wyk/Score_Function_NAVSIM_ChainFlow
mkdir -p /home/data/wyk/dataset/navsim/exp/chainflow_score_pipeline
nohup /home/data/wyk/envs/chainflow/bin/python -u scripts/run_navsim_chainflow.py run \
  --workers 4 --microbatch-size 16 \
  >> /home/data/wyk/dataset/navsim/exp/chainflow_score_pipeline/queue.log 2>&1 < /dev/null &
echo "Score 队列 PID=$!"
```

持续查看：

```bash
python3 /home/data/wyk/Score_Function_NAVSIM_ChainFlow/scripts/run_navsim_chainflow.py status --watch
```

默认 GPU 策略为等待空闲。显存剩余但利用率99–100%时，不承诺并行完全无影响。显式共享策略才使用繁忙设备；显存限制只约束本任务的PyTorch allocator，不是整张卡或整进程的隔离。所有阶段只管理自己启动的子进程，不操作现有任务或GPU全局设置。

原生复现与Score训练使用共享GPU租约目录防止本项目的队列重复选卡。训练续跑保留原有world size、microbatch和累积配置；这些设置变化时拒绝复用原输出。相同命令恢复已完成且校验通过的阶段。

监控使用 `status --watch`；Ctrl+C只退出监控。失败时首先看状态中的阶段日志，修复原因后用原命令恢复。不要删除已完成的原生轨迹、缓存或checkpoint来强制重跑。

## 原生结果交接与配对约束

独立 agent 应优先用 `scripts/eval_navsim_chainflow.py run --mode baseline` 生成标准交接。原生结果包括分片JSONL、完整summary和 `reproduction_ready.json`。后续队列校验真实文件和哈希，不仅检查这个标记是否存在。

原生与refine需要相同代码、官方checkpoint、DINO、VLM release、每场景条件内容、metric cache内容与评分配置。12,146个token的完整成员集合也固定。refine读取同一次原生forward留下的top-1轨迹，K0不重新采样、不做坐标来回转换，也不复用DD的88.0337基线。

`summary.json` 中PDMS分项为0–1，展示乘100。K0来自新ChainFlow原生评测，所有K在同一批场景配对。任何场景失败使该阶段失败，不能排除失败场景后宣布完成。

若另一位agent独立实现了不同交付格式，先做无损导入和协议核对再接本队列，不能只放一个JSON标记绕过验证。

## 本地验证范围

已提供CPU测试覆盖：官方forward选择不变、严格权重加载契约、变长VLM/padding、缓存GT与划分保持、损坏/中断恢复、DSM学习、RAW/EMA续训一致性、K0精确复用、全量token与指标校验、资源等待及下载校验。多卡Gloo测试在支持本地通信的主机可运行。

本开发环境没有公司服务器的GPU、相机、地图或正式checkpoint。真实加载、CUDA峰值显存、原生PDMS与完整训练耗时必须在用户节点实测；任务书和脚本不能替代这项验收。
