from pathlib import Path
import pandas as pd

root = Path(
    "/home/data/wyk/dataset/nuplan/exp/exp/simulation/"
    "closed_loop_nonreactive_agents/score_function/test14-hard"
)
files = list(root.glob("S05-LN_*/aggregator_metric/*.parquet"))
if not files:
    raise SystemExit("未找到 S05-LN 结果")

path = max(files, key=lambda p: p.stat().st_mtime)
df = pd.read_parquet(path)

# 使用 nuPlan 已计算好的汇总行。
summary = df.loc[
    df["num_scenarios"].notna(),
    ["scenario_type", "num_scenarios", "score"],
].copy()
summary["num_scenarios"] = summary["num_scenarios"].astype(int)

print("RUN:", path.parent.parent.name)
print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
