"""Print up to 24 diverse visualization candidates from existing paired results.

Read-only: no model inference, no changes to training/evaluation files. Scores
are in points; changed submetrics are raw 0..1, ordered DP/G20/G50/G200.
These are diagnostic candidates, not a representative evaluation subset.
"""

from pathlib import Path
from collections import Counter
import sys
import pandas as pd

path = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(
        "/home/data/wyk/dataset/nuplan/exp/score_G_local_k_1012_v1/analysis/paired_scenes.csv"
    )
)
df = pd.read_csv(path, dtype={"scenario": str, "log_name": str})
parts = {}
for k in (20, 50, 200):
    d = df[df.comparison.eq(f"DP_K0 -> S05-G_K{k}")].copy()
    if d.empty or d.duplicated(["log_name", "scenario"]).any():
        raise ValueError(f"K={k}: missing or duplicate rows; run analyze first")
    parts[k] = d.set_index(["log_name", "scenario"]).sort_index()
a = parts[20]
if any(not x.index.equals(a.index) for x in parts.values()):
    raise ValueError("Scenario keys differ between runs")
s = pd.DataFrame(
    {"DP": a.score_before, **{f"G{k}": x.score_after for k, x in parts.items()}}
)
if s.isna().any().any():
    raise ValueError("Missing scores")


def recording(log):
    p = log.rsplit("_", 2)
    return p[0] if len(p) == 3 and p[1].isdigit() and p[2].isdigit() else log


rec = {i: recording(i[0]) for i in a.index}
types = sorted(a.scenario_type.unique())
rids = {r: j + 1 for j, r in enumerate(sorted(set(rec.values())))}
gain = s.G20 - s.DP
valley = pd.concat([s.G20 - s.G50, s.G200 - s.G50], axis=1).min(axis=1)
spread = s.max(axis=1) - s.min(axis=1)
pools = [
    ("rebound", valley[valley > 0.005].sort_values(ascending=False)),
    ("improve", gain[gain > 0.005].sort_values(ascending=False)),
    ("worsen", gain[gain < -0.005].sort_values()),
    ("control", spread[(spread <= 0.005) & (s.min(axis=1) >= 0.5)].sort_values()),
]
selected, used, recordings, counts, parked = [], set(), Counter(), Counter(), 0
for group, ranking in pools:
    per_type, n = Counter(), 0
    for key in ranking.index:
        typ = a.at[key, "scenario_type"]
        stationary = "stationary" in typ.lower()
        if (
            key in used
            or recordings[rec[key]] >= 2
            or per_type[typ] >= 2
            or (stationary and parked >= 2)
        ):
            continue
        selected.append((group, key))
        used.add(key)
        recordings[rec[key]] += 1
        per_type[typ] += 1
        parked += int(stationary)
        counts[group] += 1
        n += 1
        if n == 6:
            break
print("CANDIDATES:", {g: counts[g] for g, _ in pools})
print("Types:", " | ".join(f"T{i + 1}={t}" for i, t in enumerate(types)))
print("R=same recording ID; scores=DP/G20/G50/G200 (0-100)")
print(
    "Changed metrics use DP/G20/G50/G200 (0-1); C=collision,T=TTC,H=comfort,P=progress,D=drivable,W=direction,V=speed"
)
metrics = {
    "C": "no_ego_at_fault_collisions",
    "T": "time_to_collision_within_bound",
    "H": "ego_is_comfortable",
    "P": "ego_progress_along_expert_route",
    "D": "drivable_area_compliance",
    "W": "driving_direction_compliance",
    "V": "speed_limit_compliance",
}
for group, key in selected:
    changes = []
    for short, m in metrics.items():
        v = [a.at[key, m + "_before"]] + [
            parts[k].at[key, m + "_after"] for k in (20, 50, 200)
        ]
        if max(v) - min(v) > 1e-6:
            changes.append(short + "=" + "/".join(f"{x:.3f}" for x in v))
    scores = "/".join(f"{100 * x:.2f}" for x in s.loc[key])
    print(
        f"{group}|{key[1]}|T{types.index(a.at[key, 'scenario_type']) + 1}|R{rids[rec[key]]}|{scores}|"
        + (" ".join(changes) or "metrics_unchanged")
    )
