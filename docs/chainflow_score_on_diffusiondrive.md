# 将已训练的 ChainFlow Score 用于 DiffusionDrive 轨迹

在同一批 12,146 个 navtest 场景中，把 Score 的起点换成 DiffusionDrive（DD）之前选出的 top-1 轨迹。Score 继续使用 B64 训练第 40 轮的 `final_ema.pt`，每个 token 使用原 ChainFlow 视觉、ego 和 VLM 条件，σ=0.1，γ=0.1。固定评测 K=0、5、10、15、20、30，预先指定 K20。实验只需运行推理和评分。

| 比较 | 起始轨迹 | 同一场景的 Score 条件 | 原生 K0 | K20 对 K0 的变化 |
|---|---|---|---:|---:|
| 已完成的对照 | ChainFlow top-1 | ChainFlow | 93.8194 | −1.7323 |
| 本次实验 | 保存的 DD top-1 | ChainFlow | 88.0337 | 由本次评测计算 |

主问题是新一行自己的配对差值是否大于零。两行配对差值之差衡量起始规划器改变后，Score 的收益如何变化。例如 DD 下降 0.5 分时，其下降幅度小于 ChainFlow，本次 DD 在 K20 仍为负收益。此前独立 DD 原生评测的 88.0094 来自另一份输出。

## 资源与运行

复用 `/home/data/wyk/envs/chainflow`、已验证权重、navtest 数据、地图和 metric cache。`chainflow_score_pipeline_b64_e40/plan.json` 提供训练和原 ChainFlow 评测使用的路径。`dd_score_e200_eval_latest.txt` 指向含原始 DD K0 的旧配对评测。此处读取的是每个场景对应的 K0 轨迹和七项指标。参数名中的 e200 表示这份历史评测所在的运行；本次使用的 Score 权重来自 ChainFlow B64 epoch 40。

在公司节点，从交接消息指定的提交建立独立工作树后执行：

```bash
(
  set -euo pipefail
  cd /home/data/wyk/Score_Function_NAVSIM_ChainFlow_DDTransfer
  out=/home/data/wyk/dataset/navsim/exp/chainflow_score_b64_e40_on_dd_navtest
  py=/home/data/wyk/envs/chainflow/bin/python
  mkdir -p "$out"
  nohup nice -n 19 "$py" -u scripts/eval_chainflow_score_on_dd.py run \
    --workers 8 --gpu-policy wait-idle >> "$out/queue.log" 2>&1 < /dev/null &
  echo "PID=$! 日志=$out/queue.log"
)
```

已有其他 GPU 作业时，队列默认等待空闲设备。若只分配部分卡，可在首次启动时把 `--workers` 设为 1、2 或 4。中断恢复时使用原来的 worker 数、源码和数据路径。若 DD 指针已迁移，用 `--dd-source /绝对路径/已完成的DD配对评测目录` 指向包含 `summary.json` 和 `part-*/scenes.jsonl` 的目录。

```bash
cd /home/data/wyk/Score_Function_NAVSIM_ChainFlow_DDTransfer
/home/data/wyk/envs/chainflow/bin/python scripts/eval_chainflow_score_on_dd.py status --watch
```

Ctrl+C 可以退出状态查看。完成后输出汇总：

```bash
cd /home/data/wyk/Score_Function_NAVSIM_ChainFlow_DDTransfer
/home/data/wyk/envs/chainflow/bin/python scripts/eval_chainflow_score_on_dd.py report
```

## 交付给实验负责人

1. 核对全部 12,146 个 token。每个场景的 DD K0 复用历史 float32 轨迹，并在本次评分器上重放七项指标。用保存的 ChainFlow 条件与 metric cache 哈希确认该 token 的输入与第 40 轮对照一致。
2. 汇报每个 K 的 DD PDMS、相对 DD K0 的配对变化和改善／退化／持平数。主结论使用 K20。另报与同一 checkpoint 在 ChainFlow 起点上的差值之差。
3. 保留本次的 `summary.json`、逐场景结果和运行记录。发生 K0 重放或条件校验异常时，记录第一个 token、两端的值和阶段日志，再定位数据、环境或路径差异。
