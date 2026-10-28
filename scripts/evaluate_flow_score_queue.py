#!/usr/bin/env python3
"""Published Flow Planner -> frozen S10-G K20 on the exact hard14 NR cohort."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf

# This file is both an executable and imported by the dataset-free queue tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_native_score_queue as shared


FLOW_COMMIT = "366df7654d7b7fa77f9d2577279da63127c6d2f8"
MODEL_TARGET = "flow_planner.model.flow_planner_model.flow_planner.FlowPlanner"
PLANNER_TARGET = "flow_score.planner.FlowScorePlanner"
VARIANT = "flow_planner"


def verify_flow_source(directory):
    """Require the inspected official source; do not silently use a local fork."""
    directory = Path(directory).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    if commit != FLOW_COMMIT:
        raise ValueError(f"Flow Planner source must be pinned to {FLOW_COMMIT}; found {commit}")
    dirty = subprocess.check_output(
        ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"],
        text=True).strip()
    if dirty:
        raise ValueError("Flow Planner tracked source has local changes; use the clean pinned checkout")
    return commit


def load_flow_sampling(config_path):
    """Read inference fields without resolving unrelated training/env references."""
    config = OmegaConf.load(config_path)
    if OmegaConf.select(config, "model._target_") != MODEL_TARGET:
        raise ValueError("Require the official Flow Planner model configuration")
    paths = {
        "future_len": "model.future_len", "state_dim": "model.state_dim",
        "model_type": "model.model_type", "sample_steps": "model.flow_ode.sample_steps",
        "sample_method": "model.flow_ode.sample_method",
        "sample_temperature": "model.flow_ode.sample_temperature",
        "cfg_weight": "model.flow_ode.cfg_weight",
    }
    sampling = {name: OmegaConf.select(config, path) for name, path in paths.items()}
    expected = {
        "future_len": 80, "state_dim": 4, "model_type": "x_start",
        "sample_steps": 4, "sample_method": "midpoint", "sample_temperature": 1.0,
        "cfg_weight": 1.0,
    }
    if sampling != expected:
        raise ValueError(f"Require the published Flow sampling configuration {expected}; found {sampling}")
    return sampling


def make_config(base, config_path, checkpoint, out, keys, steps, score_config, score_checkpoint):
    if steps not in (0, 20):
        raise ValueError("This queue fixes K=0 and K=20")
    cfg = shared.execution_config(base, out, 8)
    # The Flow model is loaded lazily from its own published configuration, not
    # from the UrbanDriver reference's top-level model.
    cfg.pop("model", None)
    cfg["planner"] = {"flow_score": {
        "_target_": PLANNER_TARGET, "_recursive_": False, "_convert_": "all",
        "config_path": str(config_path), "checkpoint_path": str(checkpoint),
        "score_config": str(score_config) if steps else "unused",
        "score_checkpoint": str(score_checkpoint) if steps else "unused",
        "steps": steps, "gamma": .1, "sigma": .1, "heading_projection": True,
        "evaluation_seed": base["seed"],
        "device": "cuda", "native_device": "cuda",
        "audit_dir": str(out / "refinement_audit") if steps else None,
    }}
    cfg["scenario_filter"].update(
        scenario_tokens=sorted({token for _, token in keys}),
        log_names=sorted({log for log, _ in keys}),
        num_scenarios_per_type=None, limit_total_scenarios=None, shuffle=False)
    return cfg


def make_smoke_config(cfg, out, keys):
    cfg = shared.execution_config(cfg, out, 1)
    cfg["scenario_filter"].update(scenario_tokens=[keys[0][1]], log_names=[keys[0][0]])
    cfg["worker"] = {
        "_target_": "nuplan.planning.utils.multithreading.worker_sequential.Sequential",
        "_convert_": "all"}
    cfg["number_of_cpus_allocated_per_simulation"] = 1
    return cfg


def native_preflight(config_path, checkpoint):
    from flow_score.planner import preflight_model
    return preflight_model(config_path, checkpoint)


def verify_inputs(hashes, flow_code):
    verify_flow_source(flow_code)
    for path, expected in hashes.items():
        if shared.file_hash(path) != expected:
            raise ValueError(f"Checkpoint/config changed: {path}")


def validate_stage(directory, cfg, keys):
    result = shared.validate_reuse(directory, cfg, keys)
    if cfg["planner"]["flow_score"]["steps"] == 20:
        shared.check_audits(directory, len(keys))
        for path in (directory / "refinement_audit").glob("*.json"):
            audit = json.loads(path.read_text())
            if audit.get("native_pose_count") != 80 or audit.get("score_pose_count") != 80:
                raise ValueError(f"{path}: require native80 -> score80 Flow Planner refinement")
            if (audit.get("evaluation_seed") != cfg["planner"]["flow_score"]["evaluation_seed"]
                    or audit.get("seed_policy") != "once_per_scenario_before_native_initialization"):
                raise ValueError(f"{path}: require the inherited evaluation seed once per scenario")
    return result


def run_stage(cfg, directory, keys, entry, environment, hashes, flow_code):
    verify_inputs(hashes, flow_code)
    if not directory.exists():
        shared.run_simulation(cfg, directory, entry, environment)
    else:
        print(f"CHECK existing stage {directory}", flush=True)
    # Existing incomplete directories are preserved and rejected here. They are
    # never overwritten or treated as successful simply because they exist.
    result = validate_stage(directory, cfg, keys)
    verify_inputs(hashes, flow_code)
    print(f"COMPLETE {directory.name}: {result[3]:.4f}/100; "
          f"{len(keys)}/{len(keys)} matched scenarios", flush=True)
    return result


def run_queue(args, root, exp, out):
    shared.write_json(out / "process.json", {"pid": os.getpid(), "output": str(out)})
    flow_code = Path(os.environ["FLOW_CODE"]).resolve()
    flow_checkpoint = Path(os.environ["FLOW_CHECKPOINT"]).resolve()
    config_path = Path(os.environ["FLOW_CONFIG"]).resolve()
    score_config = Path(os.environ["SCORE_CONFIG"]).resolve()
    score_checkpoint = Path(os.environ["SCORE_CHECKPOINT"]).resolve()
    frozen_source = Path(os.environ["FROZEN_SCORE_CODE"]).resolve()
    commit = verify_flow_source(flow_code)
    sampling = load_flow_sampling(config_path)
    shared.score_preflight(score_config, score_checkpoint, frozen_source)
    native_metadata = native_preflight(config_path, flow_checkpoint)
    print("Flow Planner published checkpoint loaded strictly; 4 midpoint ODE steps, "
          "temperature=1, CFG weight=1; no rule post-processing; S10-G K=20 is separate", flush=True)

    protocol_path = out / "protocol.json"
    previous = json.loads(protocol_path.read_text()) if protocol_path.exists() else None
    explicit = os.environ.get("BASELINE_RESULT")
    if explicit:
        baseline = Path(explicit).resolve()
    elif previous:
        baseline = Path(previous["baseline"])
    else:
        baseline = shared.latest(
            exp / "exp/simulation/closed_loop_nonreactive_agents/urbandriver/test14-hard",
            "raw_8gpu_*")
    base, _, keys, urban_score = shared.read_result(baseline, 272)
    if not str(base["observation"]["_target_"]).endswith(".TracksObservation"):
        raise ValueError("Reference must be the completed nonreactive UrbanDriver evaluation")
    expected_data = (root / "dataset/nuplan/nuplan-v1.1/test").resolve()
    if Path(base["scenario_builder"]["data_root"]).resolve() != expected_data:
        raise ValueError("Reference must use the formal test databases")
    matrix = exp / "score_G_seed_k_1015_v1/plan.json"
    if matrix.exists() and [list(k) for k in keys] != json.loads(
            matrix.read_text())["scenario_keys"]["test14-hard"]:
        raise ValueError("Reference scenario identities differ from the DP hard14 matrix")
    print(f"REFERENCE {baseline}; UrbanDriver={urban_score:.4f}; 272 matched scenarios", flush=True)

    hashes = {str(path): shared.file_hash(path)
              for path in (flow_checkpoint, config_path, score_config, score_checkpoint)}
    protocol = {
        "variant": VARIANT, "baseline": str(baseline), "scenario_keys": keys,
        "seed": base["seed"], "K": 20, "gamma": .1, "sigma": .1,
        "native_rng_policy": "seed_each_scenario_once_from_reference",
        "native_device": "cuda", "score_device": "cuda", "workers": 8,
        "generation": "official_flow_matching", "rule_postprocessing": False,
        "flow_source": str(flow_code), "flow_commit": commit,
        "flow_config": str(config_path), "sampling": sampling,
        "native_metadata": native_metadata,
        "frozen_score_source": str(frozen_source), "hashes": hashes,
    }
    if previous is not None and previous != json.loads(json.dumps(protocol)):
        raise ValueError("Resume protocol changed; use a fresh QUEUE_ROOT")
    shared.write_json(protocol_path, protocol)

    model_out = out / VARIANT
    raw_dir, refined_dir = model_out / "raw", model_out / "S10G_K20"
    raw_cfg = make_config(base, config_path, flow_checkpoint, raw_dir, keys, 0,
                          score_config, score_checkpoint)
    refined_cfg = make_config(base, config_path, flow_checkpoint, refined_dir, keys, 20,
                              score_config, score_checkpoint)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       NODE_RANK="0", NUM_NODES="1")
    environment.pop("NUPLAN_HYDRA_CONFIG_PATH", None)
    environment.pop("RAY_ADDRESS", None)
    entry = Path(os.environ["NUPLAN_DEVKIT_ROOT"]) / "nuplan/planning/script/run_simulation.py"

    # Verify both model and score integration on one of the exact cohort's
    # scenarios before an hours-long evaluation. Full K0 always precedes K20.
    smoke_environment = environment.copy()
    smoke_environment["CUDA_VISIBLE_DEVICES"] = environment.get(
        "CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    for name, cfg in (("K0_smoke", raw_cfg), ("K20_smoke", refined_cfg)):
        directory = model_out / name
        smoke_cfg = make_smoke_config(cfg, directory, keys)
        run_stage(smoke_cfg, directory, keys[:1], entry, smoke_environment, hashes, flow_code)
    native = run_stage(raw_cfg, raw_dir, keys, entry, environment, hashes, flow_code)
    refined = run_stage(refined_cfg, refined_dir, keys, entry, environment, hashes, flow_code)

    paired = native[1][["log_name", "scenario", "score"]].merge(
        refined[1][["log_name", "scenario", "score"]], on=["log_name", "scenario"],
        suffixes=("_raw", "_K20"), validate="one_to_one")
    paired["delta_100"] = 100 * (paired.score_K20.astype(float) - paired.score_raw.astype(float))
    paired.to_csv(model_out / "paired_scores.csv", index=False)
    comparison = {
        "variant": VARIANT, "raw_100": native[3], "K20_100": refined[3],
        "delta_100": refined[3] - native[3], "scenarios": len(keys),
        "raw_dir": str(raw_dir), "K20_dir": str(refined_dir),
        "improved": int((paired.delta_100 > 1e-8).sum()),
        "worsened": int((paired.delta_100 < -1e-8).sum()),
        "tied": int((paired.delta_100.abs() <= 1e-8).sum()),
    }
    shared.write_json(model_out / "comparison.json", comparison)
    shared.write_json(out / "summary.json", {VARIANT: comparison})
    print("PAIRED " + json.dumps(comparison), flush=True)
    print("ALL COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(os.environ.get("UD_ROOT", "/home/data/wyk")).resolve()
    exp = Path(os.environ["NUPLAN_EXP_ROOT"]).resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (exp / "flow_score_queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A Flow Planner score queue is already active; inspect its queue.log")
        marker = exp / "flow_score_queue_latest.txt"
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(str(out) + "\n")
        temporary.replace(marker)
        run_queue(args, root, exp, out)


if __name__ == "__main__":
    main()
