#!/usr/bin/env python3
"""Prepare one K20 run from the completed native UrbanDriver Hydra configuration."""

import argparse
import copy
import json
import os
from pathlib import Path

from hydra.core.utils import setup_globals
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from score_function.utils.train_utils import file_hash

ROOT = Path(os.environ.get("UD_ROOT", "/home/data/wyk"))
EXP = ROOT / "dataset/nuplan/exp/exp/simulation/closed_loop_nonreactive_agents"


def read_result(directory, expected_count):
    directory = Path(directory)
    # Importing hydra alone does not register ${now:...}. This function is also
    # called from a plain Python process, outside @hydra.main/compose.
    setup_globals()
    cfg = OmegaConf.to_container(OmegaConf.load(directory / "code/hydra/config.yaml"), resolve=True)
    report = pd.read_parquet(directory / cfg.get("runner_report_file", "runner_report.parquet"))
    if len(report) != expected_count or not report.succeeded.eq(True).all():
        raise ValueError(f"{directory}: expected {expected_count} successful scenarios")
    files = list((directory / "aggregator_metric").glob("*.parquet"))
    if len(files) != 1:
        raise ValueError(f"{directory}: expected exactly one official aggregate")
    frame = pd.read_parquet(files[0])
    detail = frame.loc[frame.log_name.notna()].copy()
    final = frame.loc[frame.scenario.eq("final_score")]
    keys = sorted(zip(detail.log_name.astype(str), detail.scenario.astype(str)))
    reports = sorted(zip(report.log_name.astype(str), report.scenario_name.astype(str)))
    if len(keys) != expected_count or len(set(keys)) != expected_count or keys != reports:
        raise ValueError("Missing, duplicate, or mismatched scenario identities")
    if len(final) != 1 or int(final.iloc[0].num_scenarios) != expected_count:
        raise ValueError("Incorrect final-score scenario count")
    if not np.isfinite(detail.score.astype(float)).all() or not np.isfinite(float(final.iloc[0].score)):
        raise ValueError("Nonfinite evaluation scores")
    return cfg, detail, keys, 100 * float(final.iloc[0].score)


def relocate(value, old, new):
    if isinstance(value, dict):
        return {k: relocate(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v, old, new) for v in value]
    if isinstance(value, str) and (value == old or value.startswith(old + "/")):
        return new + value[len(old):]
    return value


def save_config(cfg, directory):
    directory.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(cfg)
    cfg["hydra"] = {"run": {"dir": "${output_dir}"},
                    "output_subdir": "${output_dir}/code/hydra"}
    OmegaConf.save(OmegaConf.create(cfg), directory / "config.yaml")


def prepare(out):
    from urban_driver_score.runtime import load_score
    import score_function

    explicit = os.environ.get("BASELINE_RESULT")
    if explicit:
        baseline = Path(explicit).resolve()
    else:
        candidates = list((EXP / "urbandriver/test14-hard").glob("raw_8gpu_*"))
        if not candidates:
            raise FileNotFoundError("Set BASELINE_RESULT to the completed ~43-point UrbanDriver result directory")
        # Never silently fall back to an older successful run if the latest failed.
        baseline = max(candidates, key=lambda p: p.stat().st_mtime).resolve()
    cfg, _, keys, score = read_result(baseline, 272)
    if not str(cfg["observation"]["_target_"]).endswith(".TracksObservation"):
        raise ValueError("Require the completed closed-loop nonreactive UrbanDriver baseline")
    if list(cfg["planner"]) != ["ml_planner"]:
        raise ValueError("Baseline must contain the native ml_planner only")
    native = cfg["planner"]["ml_planner"]
    if not str(native["_target_"]).endswith(".MLPlanner"):
        raise ValueError("Baseline is not the native MLPlanner")
    if "urban_driver" not in str(native["model_config"]).lower():
        raise ValueError("Baseline model is not UrbanDriver")
    expected_data = ROOT / "dataset/nuplan/nuplan-v1.1/test"
    if Path(cfg["scenario_builder"]["data_root"]).resolve() != expected_data.resolve():
        raise ValueError("Baseline does not use the formal nuPlan test DB directory")
    matrix_path = ROOT / "dataset/nuplan/exp/score_G_seed_k_1015_v1/plan.json"
    if matrix_path.is_file():
        matrix = json.loads(matrix_path.read_text())
        if [list(k) for k in keys] != matrix["scenario_keys"]["test14-hard"]:
            raise ValueError("Native UrbanDriver and the completed DP hard14 matrix used different scenarios")
    frozen = Path(os.environ["FROZEN_SCORE_CODE"]).resolve()
    if Path(score_function.__file__).resolve().parent.parent != frozen:
        raise ValueError("PYTHONPATH must load the previously frozen score_function code first")
    score_config = Path(os.environ["SCORE_CONFIG"]).resolve()
    score_checkpoint = Path(os.environ["SCORE_CHECKPOINT"]).resolve()
    settings, state, branch, _ = load_score(score_config, score_checkpoint, "cpu")
    if state["epoch"] != 529:
        raise ValueError("Expected the existing S10-G final EMA at epoch 529")
    with torch.no_grad():
        pred = branch(torch.zeros(1, 80, 4), torch.zeros(1, 107, 192), torch.zeros(1, 192))
    if pred.shape != (1, 80, 4) or not torch.isfinite(pred).all():
        raise ValueError("Score forward failed in the UrbanDriver environment")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("This launcher requires eight visible GPUs")
    if cfg["number_of_gpus_allocated_per_simulation"] != 1.0:
        raise ValueError("Expected the native eight-GPU run with one GPU per simulation")
    if not str(cfg["worker"]["_target_"]).endswith(".RayDistributed"):
        raise ValueError("Native run did not use the expected Ray worker")
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"New run directory already exists: {out}")
    out.mkdir(parents=True)
    old_output = str(cfg["output_dir"])
    refined = relocate(cfg, old_output, str(out))
    refined.update(output_dir=str(out), job_name="urbandriver_score_K20",
                   experiment_uid=str(out.relative_to(EXP)), exit_on_failure=True, run_metric=True)
    refined["planner"] = {"urban_driver_score": {
        "_target_": "urban_driver_score.planner.UrbanDriverScorePlanner",
        "_recursive_": False, "_convert_": "all",
        "model_config": native["model_config"], "checkpoint_path": native["checkpoint_path"],
        "score_config": str(score_config), "score_checkpoint": str(score_checkpoint),
        "steps": 20, "gamma": .1, "sigma": .1, "heading_projection": True,
        "device": "cuda", "audit_dir": str(out / "refinement_audit"),
    }}
    save_config(refined, out / "launch/full")
    smoke_dir = out / "smoke"
    smoke = relocate(refined, str(out), str(smoke_dir))
    smoke["scenario_filter"]["limit_total_scenarios"] = 1
    smoke["worker"] = {"_target_": "nuplan.planning.utils.multithreading.worker_sequential.Sequential", "_convert_": "all"}
    smoke["disable_callback_parallelization"] = True
    save_config(smoke, out / "launch/smoke")
    record = dict(baseline=str(baseline), baseline_score_100=score,
                  baseline_config_sha256=file_hash(baseline / "code/hydra/config.yaml"),
                  scenario_keys=keys, K=20, gamma=.1, sigma=.1,
                  heading_projection=True, torch_version=torch.__version__,
                  score_code=str(frozen), score_config=str(score_config),
                  score_checkpoint=str(score_checkpoint),
                  score_checkpoint_sha256=file_hash(score_checkpoint),
                  urban_checkpoint=native["checkpoint_path"],
                  urban_checkpoint_sha256=file_hash(native["checkpoint_path"]),
                  score_epoch=state["epoch"], seed=cfg["seed"])
    (out / "protocol.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Baseline: {baseline}\nScore: {score:.4f}/100; 272/272 successful")
    print(f"READY: K=20 gamma=0.1 sigma=0.1, same native evaluation config; output={out}")


def verify(out, smoke):
    record = json.loads((out / "protocol.json").read_text())
    directory = out / "smoke" if smoke else out
    cfg, detail, keys, score = read_result(directory, 1 if smoke else 272)
    expected_keys = {tuple(k) for k in record["scenario_keys"]}
    if (smoke and not set(keys) <= expected_keys) or (not smoke and set(keys) != expected_keys):
        raise ValueError("Refined scenario membership differs from native baseline")
    planner = cfg["planner"]["urban_driver_score"]
    if (planner["steps"], planner["gamma"], planner["sigma"]) != (20, .1, .1):
        raise ValueError("Refinement parameters changed")
    for path, key in ((record["score_checkpoint"], "score_checkpoint_sha256"),
                      (record["urban_checkpoint"], "urban_checkpoint_sha256")):
        if file_hash(path) != record[key]:
            raise ValueError(f"Checkpoint changed during evaluation: {path}")
    audits = [json.loads(p.read_text()) for p in (directory / "refinement_audit").glob("*.json")]
    if len(audits) != len(keys) or any(a["completed_steps"] != 20 or not a["finite"] for a in audits):
        raise ValueError("Missing or invalid per-scenario K20 refinement audits")
    if smoke:
        print(f"K20 SMOKE PASSED: 1/1 scenario, 20 score steps, finite output; score={score:.4f}")
        return
    _, baseline, _, old_score = read_result(record["baseline"], 272)
    paired = baseline[["log_name", "scenario", "score"]].merge(
        detail[["log_name", "scenario", "score"]], on=["log_name", "scenario"],
        suffixes=("_raw", "_K20"), validate="one_to_one")
    paired["delta_100"] = 100 * (paired.score_K20.astype(float) - paired.score_raw.astype(float))
    paired.to_csv(out / "paired_scores.csv", index=False)
    summary = dict(baseline_100=old_score, refined_K20_100=score, delta_100=score-old_score,
                   scenarios=272, improved=int((paired.delta_100 > 1e-8).sum()),
                   worsened=int((paired.delta_100 < -1e-8).sum()),
                   tied=int((paired.delta_100.abs() <= 1e-8).sum()))
    (out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "verify-smoke", "verify"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.action == "prepare":
        prepare(args.output)
    else:
        verify(args.output, smoke=args.action == "verify-smoke")
