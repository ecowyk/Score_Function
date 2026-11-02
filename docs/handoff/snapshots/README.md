# 原始脚本快照

这些文件是为科研交接保存的源文件，便于新agent读取当时实际生成的调度/诊断逻辑。它们不接入main的运行入口，也不构成独立可运行的发布包。不要在读文档时直接launch。

| 目录 | 来源与用途 |
| --- | --- |
| [baselines8_v4/scripts](baselines8_v4/scripts) | 2026-10-28 `baselines8_flowcfg18_seed20260922_1028_v4.tar.gz` 的5个脚本 |
| [baselines8_v4/flow_score_cfg18](baselines8_v4/flow_score_cfg18) | 同包的CFG1.8 Flow adapter，2个文件 |
| [2026-11-02/run_baselines_parallel_1102.py](2026-11-02/run_baselines_parallel_1102.py) | 128场景并发调度器 |
| [2026-11-02/run_flow_reproduction_1102.py](2026-11-02/run_flow_reproduction_1102.py) | 单帧parity与noise-aligned隔离诊断 |

逐文件大小、SHA256和来源编号见 [manifest.json](manifest.json)。文件按读取到的内容保存，没有为适应本目录修改import、相对路径或服务器默认路径。

依赖包括对应研究分支中的 `evaluate_random_baselines_1025.py`、`evaluate_native_score_queue.py`、各planner adapter、官方源码、Python环境、权重、缓存与已准备的服务器plan。仅复制这些快照到任意机器不会自动满足依赖。`stop_energy_evaluation_1028.py` 会改变进程状态，作为历史来源保存，并非交接过程要求执行。

未来修订使用新的有日期目录并记录新hash；不要静默覆盖此快照。读取源代码能够证明行为定义，不能证明服务器已经成功运行该脚本。
