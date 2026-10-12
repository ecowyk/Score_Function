#!/usr/bin/env python3
"""Inspect 22 selected scenes without rerunning closed-loop evaluation.

This selection wrapper preserves the existing capture implementation and its
replay checks, allowing compatible saved frames to be reused unchanged.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import static_score_context as capture_core

EXTRA_CASES = (
    ("I05", "a2fc81fb19985d60", "high_lateral_acceleration"),
    ("I06", "0d3490827eb45df7", "following_lane_with_lead"),
    ("W05", "62db9dc16ffd5135", "traversing_pickup_dropoff"),
    ("W06", "6db7f9f43c655149", "behind_long_vehicle"),
    ("K05", "217131c79adf588d", "changing_lane"),
    ("K06", "0cd6cff135c25fa9", "starting_left_turn"),
)
CASES = tuple(capture_core.CASES) + EXTRA_CASES
GROUPS = {"I": "改善候选", "W": "退化候选", "K": "步数敏感候选", "C": "对照"}
REPORT_NAME = "report_cn_v4_expanded"


def selected_cases(cohort="all"):
    if cohort == "all":
        return sorted(CASES, key=lambda case: (list(GROUPS).index(case[0][0]), case[0]))
    by_name = {case[0]: case for case in CASES}
    requested = list(dict.fromkeys(cohort.split(",")))
    if not requested or any(name not in by_name for name in requested):
        raise ValueError("--cohort must be all or comma-separated IDs such as I01,W01,K05")
    return [by_name[name] for name in requested]


def prepare_index(args):
    from static_context_scores import ScenarioScores

    scores = ScenarioScores(args.suite)
    rows = []
    details = {}
    for case_id, token, scenario_type in selected_cases(args.cohort):
        bundle = scores.for_payload({"token": token, "scenario_type": scenario_type}, (20, 50, 200))
        baseline = bundle["baseline"]
        row = dict(
            case_id=case_id,
            group=GROUPS[case_id[0]],
            token=token,
            scenario_type=scenario_type,
            log_name=baseline["log_name"],
            DP_K0=baseline["score"],
            **{f"G_K{k}": bundle["by_k"][str(k)]["score"] for k in (20, 50, 200)},
        )
        rows.append(row)
        details[case_id] = bundle
    output = args.output.resolve() / "selection_expanded_v1"
    output.mkdir(parents=True, exist_ok=True)
    capture_core.write(
        output / "cohort.json",
        dict(
            selection_script_sha256=capture_core.digest(__file__),
            capture_script_sha256=capture_core.digest(capture_core.__file__),
            cases=rows,
            score_sources=details,
            note="Diagnostic selection, not an unbiased benchmark sample. Scores cover full scenarios.",
        ),
    )
    with (output / "cohort.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("case  token                 DP     G20     G50    G200   scenario_type", flush=True)
    for row in rows:
        print(
            f"{row['case_id']:5} {row['token']}  {row['DP_K0']:6.2f}  "
            f"{row['G_K20']:6.2f}  {row['G_K50']:6.2f}  {row['G_K200']:6.2f}   "
            f"{row['scenario_type']}",
            flush=True,
        )
    print(f"场景索引（含 nuBoard 查询 token）：{output / 'cohort.csv'}", flush=True)
    return rows


def capture(args):
    prepare_index(args)
    # Only the selected list changes. The original source bytes, capture
    # protocol, fixed input, weights and existing provenance remain unchanged.
    original = capture_core.CASES
    try:
        capture_core.CASES = list(CASES)
        capture_core.capture(args)
    finally:
        capture_core.CASES = original


def render(args):
    prepare_index(args)
    selected = selected_cases(args.cohort)
    report = args.output.resolve() / REPORT_NAME
    for group, label in GROUPS.items():
        names = [case[0] for case in selected if case[0].startswith(group)]
        if not names:
            continue
        command = [
            sys.executable,
            str(Path(__file__).with_name("render_static_context_cn.py")),
            "--source",
            str(args.output.resolve()),
            "--suite",
            str(args.suite.resolve()),
            "--cases",
            *names,
            "--ks",
            "20",
            "50",
            "200",
            "--output",
            str(report / group),
        ]
        if args.frames is not None:
            command.extend(["--frames", *map(str, args.frames)])
        print(f"生成 {group}：{label}", flush=True)
        subprocess.run(command, check=True)
    print(f"分组 PDF：{report}/I、W、K、C/trajectory_report_cn.pdf", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "capture", "render", "run"))
    parser.add_argument("--suite", type=Path, default=capture_core.DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=capture_core.DEFAULT_OUTPUT)
    parser.add_argument("--cohort", default="all", help="all (22), or comma-separated case IDs")
    parser.add_argument(
        "--frames", nargs="+", type=int, help="DP history indices; default 0,N/3,2N/3"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu", type=int, help="Single idle physical GPU, only for CUDA capture")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--replay-tolerance-m", type=float, default=0.02)
    args = parser.parse_args()
    selected_cases(args.cohort)
    if args.threads < 1 or args.replay_tolerance_m <= 0:
        parser.error("threads and replay tolerance must be positive")
    if args.device == "cuda" and args.gpu is None:
        parser.error("--device cuda requires --gpu N")
    if args.command == "list":
        prepare_index(args)
    if args.command in ("capture", "run"):
        capture(args)
    if args.command in ("render", "run"):
        render(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
