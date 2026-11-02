# 服务器、队列与恢复入口

日期：2026-11-02。以下路径是已知服务器布局，不表示本次环境可以直接访问，也不表示目录仍处于旧记录中的运行状态。

## 目录与环境

| 路径/标识 | 用途 |
| --- | --- |
| `/home/data/wyk/Score_Function_aligned_v1` | 对齐训练相关工作目录 |
| `/home/data/wyk/Score_Function_G_sigma_8gpu_1012` | G固定sigma训练、闭环与矩阵脚本 |
| `/home/data/wyk/Score_Function_TwoNode_1025` | 多baseline、Energy及后续扩展主工作目录 |
| `/home/data/wyk/Score_Function_PLUTO_1025` | PLUTO适配工作目录（历史记录） |
| `/home/data/wyk/Score_Function_FLOW_1028` | Flow适配工作目录（历史记录） |
| `/home/data/wyk/Diffusion-Planner` | DP官方源码依赖 |
| `/home/data/wyk/Flow-Planner` | Flow源码/资产依赖 |
| `/home/data/wyk/nuplan-devkit` | nuPlan仿真依赖 |
| `/home/data/wyk/dataset/nuplan` | nuPlan数据根目录 |
| `/home/data/wyk/dataset/nuplan/maps` | 地图 |
| `/home/data/wyk/dataset/nuplan/exp` | 仿真输出根目录 |
| `/home/data/wyk/envs/diffusion_planner/bin/python` | DP与score相关环境 |
| `/home/data/wyk/envs/urbandriver/bin/python` | 多baseline调度器常用环境 |
| `/home/data/wyk/envs/flow_planner/bin/python` | Flow环境 |

部分历史文档写过 `Score_Function_TwoNodes_1025`（多一个s）；最新脚本使用 `Score_Function_TwoNode_1025`。以服务器实际目录及plan为准，不静默改写正在运行的路径。目录名不是Git分支名，不能据此推断HEAD。

运行脚本通常设置 `NUPLAN_DATA_ROOT`、`NUPLAN_MAPS_ROOT`、`NUPLAN_EXP_ROOT`、`NUPLAN_DEVKIT_ROOT`、`DP_DIR`。不同baseline在各自环境/进程内运行，尤其PlanTF与PLUTO同名`src`包不可混用PYTHONPATH。

## 主要输出

| 根目录（相对于 `/home/data/wyk`） | 内容 |
| --- | --- |
| `outputs/score_aligned_1004/G_sigma_1012_8gpu_v1` | S05-G续训、S10-G、S025-G训练队列 |
| `outputs/score_aligned_1004/G_sigma_1012_8gpu_v1/runs/S10-G/score` | S10-G配置、模型信息、最终EMA、恢复断点 |
| `dataset/nuplan/exp/score_G_sigma_closed_loop_1014_v1` | 固定sigma闭环比较的summary/jobs |
| `outputs/score_aligned_1004/E10_G_aligned_1025_v1` | 对齐Energy训练与评测 |
| `dataset/nuplan/exp/random_baselines_1025_v1` | 6个非DP配置×K0/K20的random-NR队列 |
| `dataset/nuplan/exp/baselines8_flowcfg18_seed20260922_1028_v4` | 当前扩展68项队列 |
| 上一目录下的 `parallel128_1102` | 新并行调度manifest/status/log及冻结runner |
| `dataset/nuplan/exp/flow_repro_noisealigned_seed20260922_1102_v1` | Flow隔离parity/noise-aligned诊断 |

G的评测权重路径为训练run内的 `score/final_ema.pt`；旧原始S05-G reference位于队列的 `reference/score/`。精确路径、hash与模型配置仍读取plan，不能只按文件名选择最新修改时间的checkpoint。

## baselines8 v4 的68项

八个配置为DP、Flow、UrbanDriver、PlanTF epoch17、PlanTF epoch24、PDM-Open、GC-PGP、PLUTO w/o refine。

| setting | split / observation | 配置数 | K | 任务数 |
| --- | --- | ---: | --- | ---: |
| hard_NR | test14-hard / TracksObservation | Flow 1个 | 0,20 | 2 |
| random_NR | test14-random / TracksObservation | Flow 1个 | 0,20 | 2 |
| val_NR | val14 / TracksObservation | 8 | 0,20 | 16 |
| hard_R | test14-hard / IDMAgents | 8 | 0,20 | 16 |
| random_R | test14-random / IDMAgents | 8 | 0,20 | 16 |
| val_R | val14 / IDMAgents | 8 | 0,20 | 16 |

总计68。Val14由trainval builder抽取一次并供NR/R共享；test split从原矩阵继承实际场景keys。不能只对比场景数量。hard14的272与历史Val14的1118属于对应记录，其他数据版本不应硬编码照搬。

原队列的主要产物为 `plan.json`、`status.json`、`jobs/<job_id>.json`、`summary.csv`、`failed_jobs.csv`、各(setting,variant)下的 `pairing.json` / `paired_scores.csv`。正常退出和 `complete_with_errors` 含义不同；配对失败也不能算作完整成功。

## parallel128 配置

11月2日用户报告CPU cgroup quota=128核、内存上限1TiB、`/dev/shm`约756GiB，随后接受以下配置。它是单节点配置，不表示跨两个节点共同调度。

| 项目 | 设置 |
| --- | --- |
| 并行评测任务 | 4 |
| GPU组 | 0,1 / 2,3 / 4,5 / 6,7 |
| 每GPU并发场景 | 16；Ray每场景GPU份额0.0625 |
| 每任务场景并发 | 32；合计最多128 |
| 每任务CPU / 每场景CPU | 32 / 1 |
| 每任务object store | 8GiB |
| 每任务Ray logical memory | 128GiB；调度配置，不是实际RSS保证 |
| 数值线程 | OMP/MKL/OpenBLAS等设为1 |

脚本使用原始plan及模型、seed、cohort，创建独立Ray运行资源。接管时暂停旧控制器继续分发，让当前worker完成后再启动新任务；保留完整结果，对中断项保留旧目录后重跑，失败记录后继续。它会核验原始输入与资源条件，但本次没有验证服务器真实吞吐或数值结果是否因并行设置完全一致。

不要因看见有GPU空闲就启动第二个队列；先确认锁、进程身份、hostname及任务归属。暂停命令的意图是当前任务完成后停止分发，不能与立即杀死所有仿真进程混为一谈。

## 接手时先做只读核查

以下命令只读实际工作区/队列。若某个文件不存在，将其记为未找到，不创建空文件来代替。

```bash
git -C /home/data/wyk/Score_Function_TwoNode_1025 rev-parse HEAD
git -C /home/data/wyk/Score_Function_TwoNode_1025 status --short
cat /home/data/wyk/dataset/nuplan/exp/baselines8_flowcfg18_seed20260922_1028_v4/status.json
cat /home/data/wyk/dataset/nuplan/exp/baselines8_flowcfg18_seed20260922_1028_v4/parallel128_1102/status.json
cat /home/data/wyk/dataset/nuplan/exp/baselines8_flowcfg18_seed20260922_1028_v4/summary.csv
cat /home/data/wyk/dataset/nuplan/exp/baselines8_flowcfg18_seed20260922_1028_v4/failed_jobs.csv
tail -n 60 /home/data/wyk/dataset/nuplan/exp/baselines8_flowcfg18_seed20260922_1028_v4/parallel128_1102/queue.log
```

Energy状态入口（需要已有对应工作区脚本）：

```bash
/home/data/wyk/envs/diffusion_planner/bin/python \
  /home/data/wyk/Score_Function_TwoNode_1025/scripts/run_energy_aligned_1025.py \
  status --queue-dir /home/data/wyk/outputs/score_aligned_1004/E10_G_aligned_1025_v1
```

后续恢复命令应匹配当前控制器：原串行队列支持 `launch --retry-failed`；parallel脚本支持 `launch/status/pause/resume/restore-old`。不要把不同脚本的选项混用。本交接不会启动或停止任何服务器任务，也不提供不经状态检查就运行的统一启动命令。

## Flow诊断的两条路径

`run_flow_reproduction_1102.py` 在隔离目录准备source/weights后依次运行：

- `parity`：原始噪声代码，单真实场景首帧native与wrapper比较。
- `aligned`：先生成完整80点噪声，再按重叠chunk切分，跑272个hard-NR场景。

后者改变了原生推理噪声的联合结构，脚本manifest明确写着尚未作者确认。它是诊断变体，不能不加标签地替换原始Flow Vanilla。若已有aligned完成回执，应单独记录配置、分数和源文件hash。

## 已遇到的工程问题

历史包括环境变量未设置、Hydra runtime输出目录/to_container解析、指标回调顺序、同名模块导入、natten/Torch/CUDA匹配以及HF访问超时。相应修复存在于具体分支或v4快照，先读实际错误和源码再处理，避免为了“统一环境”重装全部依赖。

训练从共享存储迁到本地XFS后曾明显加速；epoch耗时、显存、DPM采样耗时都是特定机器/版本的观测，不作为新环境保证。实验数据/权重放服务器，Git提交代码、配置身份与结果索引即可。
