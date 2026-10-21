#!/usr/bin/env python3
"""Run paired PDM-Open / GC-PGP native -> S10-G K20 on the fixed hard14 cohort."""

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
from hydra.core.utils import setup_globals
from omegaconf import OmegaConf


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_saved_config(directory):
    setup_globals()
    return OmegaConf.to_container(
        OmegaConf.load(Path(directory) / "code/hydra/config.yaml"), resolve=True)


def read_result(directory, expected_count=272):
    directory = Path(directory)
    cfg = load_saved_config(directory)
    report = pd.read_parquet(directory / cfg.get("runner_report_file", "runner_report.parquet"))
    if len(report) != expected_count or not report.succeeded.eq(True).all():
        raise ValueError(f"{directory}: require {expected_count}/{expected_count} successful scenarios")
    files = list((directory / "aggregator_metric").glob("*.parquet"))
    if len(files) != 1:
        raise ValueError(f"{directory}: require exactly one official aggregate")
    frame = pd.read_parquet(files[0])
    detail = frame.loc[frame.log_name.notna()].copy()
    final = frame.loc[frame.scenario.eq("final_score")]
    keys = sorted(zip(detail.log_name.astype(str), detail.scenario.astype(str)))
    report_keys = sorted(zip(report.log_name.astype(str), report.scenario_name.astype(str)))
    if len(keys) != expected_count or len(set(keys)) != expected_count or keys != report_keys:
        raise ValueError(f"{directory}: missing, duplicate, or mismatched scenario identities")
    if len(final) != 1 or int(final.iloc[0].num_scenarios) != expected_count:
        raise ValueError(f"{directory}: incorrect final-score scenario count")
    score = 100 * float(final.iloc[0].score)
    if not math.isfinite(score) or not detail.score.map(lambda x: math.isfinite(float(x))).all():
        raise ValueError(f"{directory}: nonfinite scores")
    return cfg, detail, keys, score


def relocate(value, old, new):
    if isinstance(value, dict):
        return {key: relocate(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate(item, old, new) for item in value]
    if isinstance(value, str) and (value == old or value.startswith(old + "/")):
        return new + value[len(old):]
    return value


def execution_config(cfg, out, workers):
    cfg = relocate(copy.deepcopy(cfg), str(cfg["output_dir"]).rstrip("/"), str(out))
    cfg.update(output_dir=str(out), number_of_gpus_allocated_per_simulation=1.0,
               number_of_cpus_allocated_per_simulation=2, disable_callback_parallelization=True,
               distributed_mode="SINGLE_NODE", exit_on_failure=True, run_metric=True)
    cfg["scenario_builder"]["max_workers"] = workers
    cfg["worker"] = {"_target_": "nuplan.planning.utils.multithreading.worker_ray.RayDistributed",
                     "_convert_": "all", "master_node_ip": None, "threads_per_node": workers * 2,
                     "debug_mode": False, "log_to_driver": True, "logs_subdir": "logs",
                     "use_distributed": False}
    cfg["hydra"] = {"run": {"dir": "${output_dir}"}, "output_subdir": "${output_dir}/code/hydra"}
    return cfg


def make_native_config(base_cfg, smoke_cfg, out, keys, workers):
    cfg = execution_config(base_cfg, out, workers)
    cfg["planner"] = copy.deepcopy(smoke_cfg["planner"])
    if "model" in smoke_cfg:
        cfg["model"] = copy.deepcopy(smoke_cfg["model"])
    if "ml_planner" in cfg["planner"]:
        # Resolve the official GC-PGP model indirection before serializing again.
        cfg["planner"]["ml_planner"]["model_config"] = copy.deepcopy(smoke_cfg["model"])
    cfg["scenario_filter"].update(scenario_tokens=sorted({token for _, token in keys}),
                                  log_names=sorted({log for log, _ in keys}),
                                  num_scenarios_per_type=None, limit_total_scenarios=None,
                                  shuffle=False)
    native = cfg["planner"]
    name = "pdm_open" if "pdm_open_planner" in native else "gc_pgp"
    # Load after Ray assigns the GPU, rather than creating 272 models in the driver.
    # K0 is exact native-object passthrough, without resampling or score loading.
    cfg["planner"] = {"native_score": {
        "_target_": "native_planner_score.planner.NativePlannerScorePlanner",
        "_recursive_": False, "_convert_": "all", "native_planner_config": native,
        "planner_name": name, "score_config": "unused", "score_checkpoint": "unused",
        "steps": 0, "gamma": .1, "sigma": .1, "heading_projection": True,
        "native_device": "cpu" if name == "pdm_open" else "cuda", "device": "cuda",
    }}
    return cfg


def make_refined_config(native_cfg, out, score_config, score_checkpoint, workers,
                        device="cuda", planner_name=None):
    cfg = execution_config(native_cfg, out, workers)
    original = native_cfg["planner"]
    if "native_score" in original:
        if original["native_score"]["steps"] != 0:
            raise ValueError("K20 must wrap a native K0 planner, never another refiner")
        original = original["native_score"]["native_planner_config"]
    if planner_name is None:
        planner_name = "pdm_open" if "pdm_open_planner" in original else "gc_pgp"
    cfg["planner"] = {"native_score": {
        "_target_": "native_planner_score.planner.NativePlannerScorePlanner",
        "_recursive_": False, "_convert_": "all",
        "native_planner_config": copy.deepcopy(original),
        "planner_name": planner_name, "score_config": str(score_config),
        "score_checkpoint": str(score_checkpoint), "steps": 20, "gamma": .1, "sigma": .1,
        "heading_projection": True, "audit_dir": str(out / "refinement_audit"), "device": device,
        "native_device": "cpu" if planner_name == "pdm_open" else "cuda",
    }}
    return cfg


def evaluation_signature(cfg):
    """Compare model and evaluation semantics, excluding scheduling/output paths."""
    names = ("planner", "seed", "scenario_builder", "ego_controller", "observation",
             "simulation_time_controller", "simulation_history_buffer_duration", "simulation_metric")
    signature = {name: copy.deepcopy(cfg.get(name)) for name in names}
    planner = signature["planner"]
    if "native_score" in planner:
        wrapper = planner["native_score"]
        signature["native_device"] = wrapper["native_device"]
        if wrapper["steps"] == 0:
            signature["planner"] = copy.deepcopy(wrapper["native_planner_config"])
    elif "pdm_open_planner" in planner:
        signature["native_device"] = "cpu"  # Official implementation fixes this device.
    else:
        signature["native_device"] = "cuda" if cfg.get("number_of_gpus_allocated_per_simulation", 0) else "cpu"
    signature["scenario_builder"].pop("max_workers", None)
    signature["scenario_builder"].pop("verbose", None)
    # Cohort identities are compared independently. Preserve extraction/filter semantics.
    filters = copy.deepcopy(cfg["scenario_filter"])
    for key in ("scenario_tokens", "log_names", "num_scenarios_per_type",
                "limit_total_scenarios", "shuffle"):
        filters.pop(key, None)
    signature["scenario_filter"] = filters
    aggregators = copy.deepcopy(cfg.get("metric_aggregator", {}))
    for aggregator in aggregators.values():
        if isinstance(aggregator, dict):
            for key in ("save_path", "file_name"):
                aggregator.pop(key, None)
    signature["metric_aggregator"] = aggregators
    return signature


def validate_reuse(directory, expected_cfg, expected_keys):
    result = read_result(directory, len(expected_keys))
    if result[2] != sorted(expected_keys):
        raise ValueError(f"{directory}: cohort differs from reference")
    if evaluation_signature(result[0]) != evaluation_signature(expected_cfg):
        raise ValueError(f"{directory}: model/evaluation configuration differs; refusing reuse")
    return result


def latest(parent, pattern, required=True):
    candidates = [p for p in Path(parent).glob(pattern) if p.is_dir()]
    if not candidates:
        if required:
            raise FileNotFoundError(f"No directory matching {parent}/{pattern}")
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime).resolve()


def old_queue_active(raw_suite):
    """Identify the earlier stdin-based launcher, including its inherited suite path."""
    pid_file = Path(raw_suite) / "queue.pid"
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        process = Path("/proc") / str(pid)
        values = process.joinpath("environ").read_bytes().split(b"\0")
        env = dict(v.split(b"=", 1) for v in values if b"=" in v)
        marker = env.get(b"BASELINE_FULL_ROOT", b"").decode()
        command = process.joinpath("cmdline").read_bytes()
        return bool(marker and Path(marker).resolve() == Path(raw_suite).resolve()
                    and b"python" in command)
    except (ProcessLookupError, FileNotFoundError):
        return False
    except (PermissionError, ValueError) as error:
        raise RuntimeError(f"Cannot identify old queue safely: {pid_file}") from error


def wait_for_old_queue(raw_suite):
    while raw_suite is not None and old_queue_active(raw_suite):
        print(f"Waiting for existing native queue; it will be reused: {raw_suite}", flush=True)
        time.sleep(30)


def write_json(path, data):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_simulation(cfg, out, entry, environment):
    if out.exists():
        raise FileExistsError(f"Incomplete/unverified output already exists; preserved: {out}. Use a fresh QUEUE_ROOT.")
    launch = out / "launch"
    launch.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create(cfg), launch / "config.yaml")
    command = [sys.executable, "-u", str(entry), "--config-path", str(launch), "--config-name", "config"]
    print(f"START {out}", flush=True)
    with (out / "console.log").open("w") as log:
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f"Simulation exited {code}: {out}/console.log")


def check_audits(directory, expected):
    audits = [json.loads(p.read_text()) for p in (directory / "refinement_audit").glob("*.json")]
    if len(audits) != expected or any(
        a.get("completed_steps") != 20 or a.get("K") != 20 or a.get("gamma") != .1
        or a.get("sigma_score") != .1 or a.get("epoch") != 529
        or a.get("weight_kind") != "ema" or a.get("finite") is not True for a in audits
    ):
        raise ValueError(f"{directory}: missing or invalid K20 execution audits")


def score_preflight(score_config, score_checkpoint, frozen_source):
    import torch
    import score_function
    from urban_driver_score.runtime import load_conditions, load_score

    if Path(score_function.__file__).resolve().parent.parent != frozen_source.resolve():
        raise ValueError("PYTHONPATH did not load the existing frozen score source first")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("This queue requires eight visible GPUs (CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7)")
    cfg, state, branch, arguments = load_score(score_config, score_checkpoint, "cpu")
    if state["epoch"] != 529:
        raise ValueError("Require existing S10-G final EMA, epoch 529")
    load_conditions(cfg, arguments, "cpu")
    with torch.no_grad():
        output = branch(torch.zeros(1, 80, 4), torch.zeros(1, 107, 192), torch.zeros(1, 192))
    if tuple(output.shape) != (1, 80, 4) or not torch.isfinite(output).all():
        raise ValueError("S10-G forward preflight failed")
    print("Preflight passed: frozen S10-G EMA529, gamma=.1 sigma=.1 K=20; 8 GPUs visible", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--raw-suite", type=Path)
    args = parser.parse_args()
    if args.workers != 8:
        parser.error("This launcher fixes eight workers, one GPU per simulation")
    root = Path(os.environ.get("UD_ROOT", "/home/data/wyk"))
    exp = Path(os.environ["NUPLAN_EXP_ROOT"])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # One new queue per dataset root; never race two writers against the same raw results.
    lock = (exp / "native_score_queue.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("A native-score queue is already active; inspect its queue.log")
    try:
        run_queue(args, root, exp, out)
    finally:
        lock.close()


def run_queue(args, root, exp, out):
    write_json(out / "process.json", {"pid": os.getpid(), "output": str(out)})
    score_config = Path(os.environ["SCORE_CONFIG"]).resolve()
    score_checkpoint = Path(os.environ["SCORE_CHECKPOINT"]).resolve()
    frozen_source = Path(os.environ["FROZEN_SCORE_CODE"]).resolve()
    score_preflight(score_config, score_checkpoint, frozen_source)

    explicit = os.environ.get("BASELINE_RESULT")
    baseline = Path(explicit).resolve() if explicit else latest(
        exp / "exp/simulation/closed_loop_nonreactive_agents/urbandriver/test14-hard", "raw_8gpu_*")
    base, _, keys, urban_score = read_result(baseline)
    if not str(base["observation"]["_target_"]).endswith(".TracksObservation"):
        raise ValueError("Reference must be the completed nonreactive UrbanDriver evaluation")
    if Path(base["scenario_builder"]["data_root"]).resolve() != root / "dataset/nuplan/nuplan-v1.1/test":
        raise ValueError("Reference must use the formal test databases")
    matrix = exp / "score_G_seed_k_1015_v1/plan.json"
    if matrix.exists() and [list(k) for k in keys] != json.loads(matrix.read_text())["scenario_keys"]["test14-hard"]:
        raise ValueError("Reference scenario identities differ from the DP hard14 matrix")
    # On another node, do not guess whether a shared-filesystem CPU queue is
    # still running. Reuse that earlier suite only when explicitly supplied.
    raw_suite = args.raw_suite
    if raw_suite is not None:
        raw_suite = raw_suite.resolve()
        if not raw_suite.is_dir():
            raise FileNotFoundError(raw_suite)
    wait_for_old_queue(raw_suite)
    print(f"REFERENCE {baseline}; UrbanDriver={urban_score:.4f}; 272 matched scenarios", flush=True)

    # Prepare all model configurations before starting an hours-long simulation.
    expected = {}
    hashes = {str(score_config): file_hash(score_config), str(score_checkpoint): file_hash(score_checkpoint)}
    for name in ("pdm_open", "gc_pgp"):
        smoke = latest(exp / "baseline_smoke", name + "_*")
        smoke_cfg, _, _, _ = read_result(smoke, 2)
        expected[name] = make_native_config(base, smoke_cfg, out / name / "raw", keys, args.workers)
        planner = next(iter(expected[name]["planner"]["native_score"]["native_planner_config"].values()))
        checkpoint = Path(planner["checkpoint_path"])
        hashes[str(checkpoint)] = file_hash(checkpoint)
    protocol = {"baseline": str(baseline), "scenario_keys": keys, "seed": base["seed"],
                "K": 20, "gamma": .1, "sigma": .1,
                "native_device": {"pdm_open": "cpu", "gc_pgp": "cuda"}, "score_device": "cuda",
                "frozen_score_source": str(frozen_source), "hashes": hashes,
                "raw_suite": str(raw_suite) if raw_suite else None}
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError("Resume protocol changed; use a fresh QUEUE_ROOT")
    write_json(protocol_path, protocol)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", NODE_RANK="0", NUM_NODES="1")
    environment.pop("NUPLAN_HYDRA_CONFIG_PATH", None)
    environment.pop("RAY_ADDRESS", None)
    entry = Path(os.environ["NUPLAN_DEVKIT_ROOT"]) / "nuplan/planning/script/run_simulation.py"
    summary = {}
    for name in ("pdm_open", "gc_pgp"):
        for path, digest in hashes.items():
            if file_hash(path) != digest:
                raise ValueError(f"Checkpoint/config changed: {path}")
        cfg = expected[name]
        raw_dir = out / name / "raw"
        candidate = raw_suite / name if raw_suite else None
        if raw_dir.exists():
            native = validate_reuse(raw_dir, cfg, keys)
        elif candidate is not None and candidate.exists():
            # An incomplete result is not accepted or silently ignored.
            old = read_result(candidate, len(keys))
            old_signature, new_signature = evaluation_signature(old[0]), evaluation_signature(cfg)
            if (name == "gc_pgp" and old_signature.get("native_device") == "cpu"
                    and new_signature.get("native_device") == "cuda"):
                # CPU-native results cannot be paired with a GPU-native K20 run.
                old_signature["native_device"] = "cuda"
                if old[2] != keys or old_signature != new_signature:
                    raise ValueError("Earlier CPU GC-PGP has different model/evaluation settings")
                print(f"KEEP CPU result {candidate}; run a fresh GPU GC-PGP baseline for GPU K20", flush=True)
                run_simulation(cfg, raw_dir, entry, environment)
                native = validate_reuse(raw_dir, cfg, keys)
            else:
                native = validate_reuse(candidate, cfg, keys)
                raw_dir = candidate
                print(f"REUSE completed native {name}: {raw_dir}", flush=True)
        else:
            run_simulation(cfg, raw_dir, entry, environment)
            native = validate_reuse(raw_dir, cfg, keys)
        print(f"NATIVE {name}: {native[3]:.4f}/100, 272/272", flush=True)

        refined_dir = out / name / "S10G_K20"
        refined = make_refined_config(native[0], refined_dir, score_config, score_checkpoint,
                                      args.workers, planner_name=name)
        if not refined_dir.exists():
            smoke_dir = out / name / "K20_smoke"
            smoke_cfg = execution_config(refined, smoke_dir, 1)
            smoke_cfg["scenario_filter"].update(scenario_tokens=[keys[0][1]], log_names=[keys[0][0]])
            smoke_cfg["worker"] = {
                "_target_": "nuplan.planning.utils.multithreading.worker_sequential.Sequential",
                "_convert_": "all"}
            smoke_cfg["number_of_cpus_allocated_per_simulation"] = 1
            smoke_environment = environment.copy()
            smoke_environment["CUDA_VISIBLE_DEVICES"] = environment.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            if not smoke_dir.exists():
                run_simulation(smoke_cfg, smoke_dir, entry, smoke_environment)
            smoke_result = read_result(smoke_dir, 1)
            if smoke_result[2] != keys[:1]:
                raise ValueError("K20 smoke scenario identity mismatch")
            check_audits(smoke_dir, 1)
            run_simulation(refined, refined_dir, entry, environment)
        result = validate_reuse(refined_dir, refined, keys)
        check_audits(refined_dir, 272)
        for path, digest in hashes.items():
            if file_hash(path) != digest:
                raise ValueError(f"Checkpoint/config changed during evaluation: {path}")
        paired = native[1][["log_name", "scenario", "score"]].merge(
            result[1][["log_name", "scenario", "score"]], on=["log_name", "scenario"],
            suffixes=("_raw", "_K20"), validate="one_to_one")
        paired["delta_100"] = 100 * (paired.score_K20.astype(float) - paired.score_raw.astype(float))
        paired.to_csv(out / name / "paired_scores.csv", index=False)
        summary[name] = {"raw_100": native[3], "K20_100": result[3], "delta_100": result[3] - native[3],
                         "scenarios": 272, "raw_dir": str(raw_dir), "K20_dir": str(refined_dir),
                         "improved": int((paired.delta_100 > 1e-8).sum()),
                         "worsened": int((paired.delta_100 < -1e-8).sum()),
                         "tied": int((paired.delta_100.abs() <= 1e-8).sum())}
        write_json(out / name / "comparison.json", summary[name])
        write_json(out / "summary.json", summary)
        print(f"PAIRED {name}: " + json.dumps(summary[name]), flush=True)
    print("ALL COMPLETE", flush=True)


if __name__ == "__main__":
    main()
