#!/usr/bin/env python3
"""Read a completed Flow native run and report protocol differences; no simulation."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pandas as pd

from evaluate_native_score_queue import file_hash, read_result


def yaml_data(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


def git_info(path):
    try:
        return {
            "commit": subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip(),
            "tracked_changes": subprocess.check_output(
                ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True).splitlines(),
        }
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": str(error)}


def audit(queue_root, flow_code=None, nuplan_root=None):
    protocol = json.loads((queue_root / "protocol.json").read_text())
    flow_code = Path(flow_code or protocol["flow_source"])
    raw = queue_root / "flow_planner/raw"
    cfg, _, keys, score = read_result(raw, 272)
    filter_path = flow_code / "flow_planner/nuplan_simulation/scenario_filter/test14-hard.yaml"
    official = yaml_data(filter_path)
    expected, actual = set(map(str, official["scenario_tokens"])), {token for _, token in keys}
    flags = []
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra or len(expected) != 272:
        flags.append("Scenario tokens do not match the official 272-token test14-hard filter.")
    filter_differences = {}
    for field, value in official.items():
        if field in ("_target_", "_convert_", "scenario_tokens", "log_names"):
            continue
        saved = cfg["scenario_filter"].get(field)
        same = sorted(saved or []) == sorted(value or []) if field == "scenario_types" else saved == value
        if not same:
            filter_differences[field] = {"saved": saved, "official": value}
    if filter_differences:
        flags.append("Scenario filter semantics differ; inspect scenario_filter_differences.")
    source = git_info(flow_code)
    if source.get("commit") != protocol.get("flow_commit") or source.get("tracked_changes"):
        flags.append("Current Flow source differs from the source recorded for this run.")
    planner = cfg["planner"]["flow_score"]
    assets = {}
    for name, path in (("checkpoint", planner["checkpoint_path"]), ("config", planner["config_path"])):
        current = file_hash(path)
        saved = protocol.get("hashes", {}).get(str(Path(path).resolve()))
        assets[name] = {"path": path, "current_sha256": current, "recorded_sha256": saved,
                        "matches_recorded": current == saved}
        if current != saved:
            flags.append(f"{name} hash no longer matches this run's recorded asset.")
    model_cfg = OmegaConf.load(planner["config_path"])
    repo_model_path = flow_code / "flow_planner/script/model/flow_planner.yaml"
    repo_cfg = yaml_data(repo_model_path)["cfg_weight"]
    effective_cfg = protocol.get("native_metadata", {}).get("cfg_weight", protocol["sampling"].get("cfg_weight"))
    if effective_cfg != repo_cfg:
        flags.append(f"Saved effective CFG={effective_cfg}; official repository inference default is {repo_cfg}.")
    if planner.get("steps") != 0:
        flags.append("The selected raw directory is not configured with K=0.")
    observation = cfg.get("observation", {})
    controller = cfg.get("ego_controller", {})
    if not str(observation.get("_target_", "")).endswith(".TracksObservation"):
        flags.append("Observation is not the official nonreactive TracksObservation.")
    if not str(controller.get("tracker", {}).get("_target_", "")).endswith(".LQRTracker"):
        flags.append("The saved controller does not expose an LQRTracker; inspect ego_controller.")
    aggregate_dir = Path(cfg.get("aggregator_save_path", raw / "aggregator_metric"))
    aggregate = next(aggregate_dir.glob("*.parquet"))
    frame = pd.read_parquet(aggregate)
    final = frame.loc[frame.scenario.eq("final_score")]
    final_row = json.loads(final.to_json(orient="records"))[0]
    package_versions = {}
    for name in ("numpy", "scipy", "pandas", "nuplan-devkit", "torch", "torchvision", "flow-matching"):
        try:
            package_versions[name] = version(name)
        except PackageNotFoundError:
            package_versions[name] = None
    nuplan_root = Path(nuplan_root or os.environ.get("NUPLAN_DEVKIT_ROOT", "/home/data/wyk/nuplan-devkit"))
    builder = dict(cfg.get("scenario_builder", {}))
    if isinstance(builder.get("db_files"), list):
        builder["db_files"] = {"count": len(builder["db_files"]), "first_three": builder["db_files"][:3]}
    return {
        "status": "audit_only_not_a_reproduction_pass", "queue_root": str(queue_root),
        "native_score_100": score, "paper_score_100": 76.47, "difference_from_paper": score - 76.47,
        "flags": flags,
        "cohort": {"successful_unique_scenarios": len(keys), "official_unique_tokens": len(expected),
                   "exact_token_match": not missing and not extra, "missing_tokens": missing, "extra_tokens": extra,
                   "official_filter": str(filter_path), "official_filter_sha256": file_hash(filter_path)},
        "scenario_filter_differences": filter_differences,
        "log_filter_note": "Queue log_names narrows the reference cohort; actual token identities are checked above.",
        "flow_source": {"path": str(flow_code), "recorded_commit": protocol.get("flow_commit"), **source},
        "assets": assets,
        "guidance": {"effective_cfg_weight_saved": effective_cfg, "repository_default": repo_cfg,
                     "published_config_cfg_weight_now": OmegaConf.select(model_cfg, "model.flow_ode.cfg_weight"),
                     "repository_config": str(repo_model_path),
                     "note": "FlowODE uses its instantiated cfg_weight; planner keyword alone does not override it."},
        "saved_sampling": protocol.get("sampling"), "saved_native_metadata": protocol.get("native_metadata"),
        "seed": {"saved_simulation": cfg.get("seed"), "planner": planner.get("evaluation_seed"),
                 "rng_policy": protocol.get("native_rng_policy"), "paper_evaluation_seed": "not disclosed"},
        "evaluation": {name: cfg.get(name) for name in (
            "ego_controller", "observation", "simulation_time_controller", "simulation_history_buffer_duration",
            "simulation_metric", "metric_aggregator")},
        "scenario_builder": builder, "aggregate_path": str(aggregate), "aggregate_final_row": final_row,
        "runtime_saved": protocol.get("runtime"), "audit_environment_packages_now": package_versions,
        "nuplan_source_now": {"path": str(nuplan_root), **git_info(nuplan_root)},
        "limitations": ["No models or datasets loaded; no files changed and no simulation launched.",
                        "Evaluation configuration is reported, not fully recomposed from an unknown paper devkit version.",
                        "The paper's exact evaluation seed, repetition policy and released-weight provenance remain unverified.",
                        "A CFG mismatch does not establish how much of the score gap it explains."],
        "paper": "https://arxiv.org/html/2510.11083v1",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path)
    parser.add_argument("--flow-code", type=Path)
    parser.add_argument("--nuplan-root", type=Path)
    args = parser.parse_args()
    try:
        root = args.queue_root
        if root is None:
            exp = Path(os.environ.get("NUPLAN_EXP_ROOT", "/home/data/wyk/dataset/nuplan/exp"))
            root = Path((exp / "flow_score_queue_latest.txt").read_text().strip())
        result = audit(root.resolve(), args.flow_code, args.nuplan_root)
    except Exception as error:
        print(json.dumps({"status": "audit_incomplete", "error": f"{type(error).__name__}: {error}"}, indent=2))
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
