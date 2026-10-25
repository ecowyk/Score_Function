#!/usr/bin/env python3
"""Twelve paired non-DP evaluations on the completed DP Test14-Random NR cohort.

The controller uses only the standard library. Each planner is run in its own
existing Python environment, so PlanTF/PLUTO's identically named ``src`` packages
cannot contaminate one another. A simulation owns one of eight GPUs.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import traceback

CODE = Path(__file__).resolve().parent.parent
VARIANTS = ("urban_driver", "plantf_epoch17", "plantf_epoch24", "pdm_open", "gc_pgp", "pluto_wo_refine")
CHALLENGE = "closed_loop_nonreactive_agents"
SPLIT = "test14-random"
GPU_IDS = "0,1,2,3,4,5,6,7"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def keys(raw):
    require(isinstance(raw, list) and bool(raw), "The cohort must be a nonempty list")
    result = []
    for row in raw:
        require(isinstance(row, (list, tuple)) and len(row) == 2, "Invalid (log, token) row")
        log, token = row
        require(isinstance(log, str) and bool(log) and isinstance(token, str)
                and re.fullmatch(r"[0-9a-f]{16}", token), "Invalid scenario identity")
        result.append((log, token))
    require(len(set(result)) == len(result), "Duplicate scenario identities")
    return sorted(result)


def shared():
    # Dependencies are imported only inside the appropriate planner environment.
    import evaluate_native_score_queue
    return evaluate_native_score_queue


def native_planner(name, source):
    planner = copy.deepcopy(source["planner"])
    if "native_score" in planner:
        wrapper = planner["native_score"]
        require(wrapper.get("steps") == 0, "Reference must be native K0")
        planner = wrapper["native_planner_config"]
    if name == "urban_driver":
        require(list(planner) == ["ml_planner"] and
                "urban_driver" in str(planner["ml_planner"]["model_config"]).lower(),
                "UrbanDriver source is not the native UrbanDriver model")
    elif name.startswith("plantf_"):
        require(list(planner) == ["imitation_planner"] and
                planner["imitation_planner"]["_target_"] == "src.planners.imitation_planner.ImitationPlanner",
                "PlanTF source must use official ImitationPlanner")
        wanted = "last.ckpt" if name.endswith("24") else "epoch=17-val_minFDE=0.000.ckpt"
        require(Path(planner["imitation_planner"]["planner_ckpt"]).name == wanted,
                "PlanTF checkpoint selection differs from the completed hard_NR experiment")
    elif name == "pdm_open":
        require(list(planner) == ["pdm_open_planner"], "PDM-Open native source mismatch")
    elif name == "gc_pgp":
        require(list(planner) == ["ml_planner"] and
                "pgp" in str(planner["ml_planner"]["model_config"]).lower(), "GC-PGP native source mismatch")
    elif name == "pluto_wo_refine":
        require(list(planner) == ["pluto_score"] and planner["pluto_score"].get("steps") == 0
                and planner["pluto_score"].get("_target_") == "pluto_score.planner.PlutoScorePlanner",
                "PLUTO must be the previously tested w/o rule refinement K0 variant")
    else:
        raise ValueError("Unexpected planner; this queue never schedules Diffusion Planner")
    return planner


def checkpoint_paths(planner):
    result = []
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("checkpoint_path", "planner_ckpt"):
                    result.append(Path(item).resolve())
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(planner)
    require(len(set(result)) == 1, "Require one unambiguous native checkpoint per planner")
    return sorted(set(result))


def source_checkpoint_provenance(source, checkpoint):
    """Honor checkpoint hashes already recorded by the successful hard_NR queues."""
    paths = []
    checkpoint = Path(checkpoint).resolve()
    for folder in (source, source.parent, source.parent.parent):
        path = folder / "protocol.json"
        if not path.is_file():
            continue
        protocol = read(path)
        expected = protocol.get("hashes", {}).get(str(checkpoint))
        if protocol.get("checkpoint") and Path(protocol["checkpoint"]).resolve() == checkpoint:
            expected = protocol.get("checkpoint_sha256", expected)
        if expected:
            require(digest(checkpoint) == expected, f"Native checkpoint changed since hard_NR: {checkpoint}")
            paths.append(path)
    return paths


def make_config(name, base, source, output, cohort, steps, score_config, score_checkpoint, seed):
    require(name in VARIANTS and steps in (0, 20), "Only the specified six native/K20 pairs are allowed")
    cfg = shared().execution_config(base, Path(output), 8)
    cfg.pop("defaults", None)
    cfg["seed"] = seed
    cfg["group"] = str(Path(output).parent)
    cfg["job_name"] = name + ("_native" if steps == 0 else "_S10G_K20")
    cfg["experiment_uid"] = SPLIT + "/" + name
    # Models sometimes reference ${model}; source configs are already resolved.
    if "model" in source:
        cfg["model"] = copy.deepcopy(source["model"])
    else:
        cfg.pop("model", None)
    original = native_planner(name, source)
    common = dict(_recursive_=False, _convert_="all", steps=steps, gamma=.1, sigma=.1,
                  heading_projection=True, device="cuda",
                  score_config=str(score_config) if steps else "unused",
                  score_checkpoint=str(score_checkpoint) if steps else "unused",
                  audit_dir=str(Path(output) / "refinement_audit") if steps else None)
    if name == "urban_driver":
        native = original["ml_planner"]
        cfg["planner"] = {"urban_driver_score": dict(common,
            _target_="urban_driver_score.planner.UrbanDriverScorePlanner",
            model_config=native["model_config"], checkpoint_path=native["checkpoint_path"])}
    elif name.startswith("plantf_"):
        cfg["planner"] = {"plantf_score": dict(common,
            _target_="plantf_score.planner.PlanTFScorePlanner", native_planner_config=original)}
    elif name == "pluto_wo_refine":
        wrapper = original["pluto_score"]
        wrapper.update(common)
        cfg["planner"] = {"pluto_score": wrapper}
    else:
        cfg["planner"] = {"native_score": dict(common,
            _target_="native_planner_score.planner.NativePlannerScorePlanner",
            native_planner_config=original, planner_name=name,
            native_device="cpu" if name == "pdm_open" else "cuda")}
    # Freeze exact identities rather than rerunning the random subset selection.
    cfg["scenario_filter"].update(scenario_tokens=sorted({t for _, t in cohort}),
        log_names=sorted({log for log, _ in cohort}), num_scenarios_per_type=None,
        limit_total_scenarios=None, shuffle=False)
    return cfg


def candidates(exp, name):
    exp = Path(exp)
    patterns = {
        "urban_driver": ["exp/simulation/closed_loop_nonreactive_agents/urbandriver/test14-hard/raw_8gpu_*"],
        "plantf_epoch17": ["plantf_native_hard_NR_*/full"],
        "plantf_epoch24": ["plantf_last_hard_NR_*/full"],
        "pdm_open": ["native_score_queue_*/pdm_open/raw"],
        "gc_pgp": ["native_score_queue_*/gc_pgp/raw"],
        "pluto_wo_refine": ["pluto_score_queue_*/pluto_wo_refine/raw"],
    }
    result = [path.resolve() for pattern in patterns[name] for path in exp.glob(pattern) if path.is_dir()]
    # Native queues may have reused an earlier baseline rather than creating raw/.
    if name in ("pdm_open", "gc_pgp"):
        for summary in exp.glob("native_score_queue_*/summary.json"):
            record = read(summary).get(name)
            if record and record.get("raw_dir"):
                result.append(Path(record["raw_dir"]).resolve())
    require(bool(result), f"No prior native result for {name}; provide --sources JSON mapping names to result directories")
    return sorted(set(result), key=lambda p: p.stat().st_mtime, reverse=True)


def family(name):
    return "plantf" if name.startswith("plantf_") else "pluto" if name == "pluto_wo_refine" else "urbandriver"


def runtime(root, name):
    root = Path(root)
    kind = family(name)
    checkout = root / {"plantf": "planTF", "pluto": "pluto", "urbandriver": "tuplan_garage"}[kind]
    entry = checkout / "run_simulation.py" if kind == "plantf" else root / "nuplan-devkit/nuplan/planning/script/run_simulation.py"
    return dict(python=str(root / "envs" / kind / "bin/python"), cwd=str(checkout), entry=str(entry))


def environment(plan, name):
    root = Path(plan["root"])
    paths = [Path(plan["frozen_score_code"]), CODE, Path(runtime(root, name)["cwd"]),
             root / "Diffusion-Planner", root / "nuplan-devkit"]
    if family(name) != "urbandriver":
        paths.append(root / "tuplan_garage")
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join(map(str, paths)), CUDA_VISIBLE_DEVICES=GPU_IDS,
        NUPLAN_DEVKIT_ROOT=str(root / "nuplan-devkit"), NUPLAN_DATA_ROOT=str(root / "dataset/nuplan"),
        NUPLAN_MAPS_ROOT=str(root / "dataset/nuplan/maps"), NUPLAN_EXP_ROOT=str(root / "dataset/nuplan/exp"),
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", HYDRA_FULL_ERROR="1",
        NODE_RANK="0", NUM_NODES="1", PYTHONUNBUFFERED="1")
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                "GROUP_RANK", "ROLE_RANK", "RAY_ADDRESS", "NUPLAN_HYDRA_CONFIG_PATH"):
        env.pop(key, None)
    return env


def prepare(args):
    out, root = args.output.resolve(), args.root.resolve()
    plan_path = out / "plan.json"
    if plan_path.exists():
        plan = read(plan_path)
        require(plan["root"] == str(root) and plan["seed"] == args.seed,
                "Existing queue root/seed differs; use its original arguments")
        verify_hashes(plan)
        return plan
    exp = root / "dataset/nuplan/exp"
    matrix_path = (args.matrix or exp / "score_G_seed_k_1015_v1/plan.json").resolve()
    matrix = read(matrix_path)
    cohort = keys(matrix["scenario_keys"][SPLIT])
    hard = keys(matrix["scenario_keys"]["test14-hard"])
    anchor_path = matrix_path.parent / "jobs" / f"random_NR_seed{args.seed}_K0.json"
    anchor = read(anchor_path)
    require(anchor.get("state") == "complete" and anchor.get("K") == 0,
            "The selected DP random_NR K0 reference must already be complete; this queue will not rerun DP")
    require(keys(anchor["scenario_keys"]) == cohort, "DP reference cohort differs from the matrix")
    require(anchor.get("seed") == args.seed and anchor.get("setting") == "random_NR",
            "DP reference seed/setting mismatch")
    baseline = Path(anchor["result_dir"]).resolve()
    base, _, actual, _ = shared().read_result(baseline, len(cohort))
    require(actual == cohort and str(base["observation"]["_target_"]).endswith(".TracksObservation"),
            "DP reference must contain the exact completed random NR cohort")
    require(Path(base["scenario_builder"]["data_root"]).resolve() == root / "dataset/nuplan/nuplan-v1.1/test",
            "The DP reference must use formal test databases")
    frozen = (args.frozen_score_code or exp / "score_G_transfer_1014_v1/code/lg").resolve()
    score_cfg = (args.score_config or exp / "score_G_transfer_1014_v1/models/S10-G/config.json").resolve()
    score_ckpt = (args.score_checkpoint or exp / "score_G_transfer_1014_v1/models/S10-G/final_ema.pt").resolve()
    score_settings = read(score_cfg)
    require(score_settings["training"]["sigma"] == .1 and
            score_settings["model"].get("parameterization", "score") == "score",
            "The baseline queue requires the existing direct-score S10-G, not the new Energy model")
    explicit = read(args.sources) if args.sources else {}
    require(set(explicit) <= set(VARIANTS), "Unknown planner in --sources; DP is not scheduled")
    sources, configs, input_files = {}, {}, [matrix_path, anchor_path, baseline / "code/hydra/config.yaml", score_cfg, score_ckpt]
    for key in ("planner_checkpoint", "planner_args"):
        path = Path(score_settings["paths"][key]).expanduser()
        input_files.append(path if path.is_absolute() else Path(score_settings["paths"]["root"]) / path)
    for name in VARIANTS:
        source = Path(explicit[name]).resolve() if name in explicit else candidates(exp, name)[0]
        cfg, _, source_keys, score = shared().read_result(source, len(hard))
        require(source_keys == hard and str(cfg["observation"]["_target_"]).endswith(".TracksObservation"),
                f"{name}: native source is not the completed hard_NR cohort")
        original = native_planner(name, cfg)
        inputs = checkpoint_paths(original)
        input_files.extend(inputs + [source / "code/hydra/config.yaml"])
        input_files.extend(source_checkpoint_provenance(source, inputs[0]))
        for path in runtime(root, name).values():
            require(Path(path).exists(), f"Missing runtime prerequisite for {name}: {path}")
        sources[name] = dict(directory=str(source), hard_NR_score_100=score,
                             checkpoint=str(inputs[0]), runtime=runtime(root, name))
        configs[name] = {str(k): make_config(name, base, cfg, out / name / ("native" if k == 0 else "S10G_K20"),
                            cohort, k, score_cfg, score_ckpt, args.seed) for k in (0, 20)}
        print(f"SOURCE {name}: {score:.4f}; {source}", flush=True)
    # Pin relevant inference source; no environments, datasets or credentials are copied.
    for directory in [frozen / "score_function", CODE / "urban_driver_score", CODE / "native_planner_score",
                      CODE / "plantf_score", CODE / "pluto_score", root / "Diffusion-Planner/diffusion_planner",
                      root / "nuplan-devkit/nuplan",
                      root / "planTF/src", root / "pluto/src", root / "tuplan_garage/tuplan_garage"]:
        require(directory.is_dir(), f"Missing code directory: {directory}")
        input_files.extend(directory.rglob("*.py"))
    input_files.extend([Path(__file__).resolve(), CODE / "scripts/evaluate_native_score_queue.py"])
    hashes = {str(path): digest(path) for path in sorted(set(input_files))}
    plan = dict(schema_version=1, root=str(root), host=socket.gethostname(), split=SPLIT, challenge=CHALLENGE,
        seed=args.seed, gamma=.1, sigma=.1, ks=[0, 20], variants=list(VARIANTS), scenario_keys=cohort,
        reference_dp=str(baseline), matrix=str(matrix_path), frozen_score_code=str(frozen),
        score_config=str(score_cfg), score_checkpoint=str(score_ckpt), sources=sources, configs=configs,
        workers=8, gpus=list(range(8)), hashes=hashes)
    out.mkdir(parents=True, exist_ok=True)
    write(plan_path, plan)
    write(out / "status.json", dict(state="prepared", full_evaluations=12, scenarios=len(cohort)))
    print(f"READY: 12 evaluations; {len(cohort)} exact Test14-Random NR scenarios; eight GPUs; DP reused only", flush=True)
    return plan


def verify_hashes(plan):
    require(plan.get("variants") == list(VARIANTS) and plan.get("ks") == [0, 20]
            and plan.get("workers") == 8 and plan.get("gpus") == list(range(8))
            and plan.get("gamma") == .1 and plan.get("sigma") == .1, "Queue protocol changed")
    require(keys(plan["scenario_keys"]) == keys(read(plan["matrix"])["scenario_keys"][SPLIT]),
            "Queue cohort differs from the frozen DP random cohort")
    for name in VARIANTS:
        for steps in (0, 20):
            cfg = plan["configs"][name][str(steps)]
            wrapper = next(iter(cfg["planner"].values()))
            require(wrapper.get("steps") == steps and wrapper.get("gamma") == .1
                    and wrapper.get("sigma") == .1 and cfg.get("seed") == plan["seed"],
                    "Job seed/refinement settings changed")
            require(cfg.get("number_of_gpus_allocated_per_simulation") == 1.0
                    and str(cfg["worker"]["_target_"]).endswith(".RayDistributed"),
                    "Full simulations must reserve one GPU under Ray")
    for path, expected in plan["hashes"].items():
        require(digest(path) == expected, "Input changed since prepare: " + path)


def preflight(plan, name):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from urban_driver_score.runtime import load_cpu
    torch.set_num_threads(1)
    shared().score_preflight(Path(plan["score_config"]), Path(plan["score_checkpoint"]), Path(plan["frozen_score_code"]))
    wrapper = next(iter(plan["configs"][name]["0"]["planner"].values()))
    checkpoint = plan["sources"][name]["checkpoint"]
    if name.startswith("plantf_"):
        from plantf_score.planner import validated_native_config
        from src.planners.planner_utils import load_checkpoint
        state = load_cpu(checkpoint)
        expected = 17 if name.endswith("17") else 24
        require(state.get("epoch") == expected, "PlanTF checkpoint epoch differs from its selected row")
        if expected == 24:
            require(state.get("global_step", -1) >= 179674, "PlanTF last.ckpt training is incomplete")
        original = validated_native_config(wrapper["native_planner_config"])["imitation_planner"]
        model = instantiate(OmegaConf.create(original["planner"]))
        model.load_state_dict(load_checkpoint(checkpoint), strict=True)
        require(all(torch.isfinite(t).all() for t in model.state_dict().values()),
                "Nonfinite PlanTF checkpoint weights")
        del state, model
    elif name == "pluto_wo_refine":
        from evaluate_pluto_score_queue import verify_pluto_source
        from pluto_score.planner import preflight_model
        verify_pluto_source(Path(plan["root"]) / "pluto")
        preflight_model(wrapper["model_config"], checkpoint)
    else:
        from nuplan.planning.script.builders.planner_builder import build_planners
        # The official builder strictly restores the native Lightning model.
        # ModelLoader CUDA initialization is deferred until simulation initialize().
        # PDMOpen's own constructor loads its original model on CPU.
        if name == "urban_driver":
            original = {"ml_planner": {"_target_": "nuplan.planning.simulation.planner.ml_planner.ml_planner.MLPlanner",
                "_convert_": "all", "model_config": wrapper["model_config"], "checkpoint_path": checkpoint}}
        else:
            original = wrapper["native_planner_config"]
        models = build_planners(OmegaConf.create(original), None)
        require(len(models) == 1, "Native checkpoint preflight must instantiate exactly one original planner")
        del models
    print(f"PREFLIGHT {name}: native checkpoint loaded; eight GPUs and S10-G final EMA529 verified", flush=True)


def validate_stage(directory, cfg, cohort, name, steps):
    result = shared().validate_reuse(Path(directory), cfg, cohort)
    if steps:
        shared().check_audits(Path(directory), len(cohort))
        expected_poses = 80 if name.startswith("plantf_") or name == "pluto_wo_refine" else 16
        for path in (Path(directory) / "refinement_audit").glob("*.json"):
            audit = read(path)
            require(audit.get("native_pose_count") == expected_poses and audit.get("score_pose_count") == 80,
                    "Refinement trajectory sampling audit mismatch")
    return result


def stage(plan, out, name, steps, smoke, retry_failed):
    cohort = keys(plan["scenario_keys"])
    cfg = copy.deepcopy(plan["configs"][name][str(steps)])
    directory = Path(cfg["output_dir"])
    env = environment(plan, name)
    if smoke:
        directory = out / name / ("smoke_K" + str(steps))
        cfg = shared().execution_config(cfg, directory, 1)
        cfg["scenario_filter"].update(scenario_tokens=[cohort[0][1]], log_names=[cohort[0][0]])
        cfg["worker"] = {"_target_": "nuplan.planning.utils.multithreading.worker_sequential.Sequential", "_convert_": "all"}
        cfg["number_of_cpus_allocated_per_simulation"] = 1
        cohort = cohort[:1]
        env["CUDA_VISIBLE_DEVICES"] = "0"
    verify_hashes(plan)
    if directory.exists():
        try:
            result = validate_stage(directory, cfg, cohort, name, steps)
            print(f"REUSE {name}/{directory.name}: {result[3]:.4f}", flush=True)
            return result
        except (ValueError, FileNotFoundError, KeyError):
            if not retry_failed:
                raise RuntimeError(f"Incomplete or invalid {directory}; inspect logs, then rerun with --retry-failed to preserve and retry it")
            archived = directory.with_name(directory.name + ".failed." + str(time.time_ns()))
            directory.rename(archived)
            print(f"PRESERVED failed attempt: {archived}", flush=True)
    shared().run_simulation(cfg, directory, Path(plan["sources"][name]["runtime"]["entry"]), env)
    result = validate_stage(directory, cfg, cohort, name, steps)
    verify_hashes(plan)
    print(f"COMPLETE {name}/{directory.name}: {result[3]:.4f}; {len(cohort)}/{len(cohort)}", flush=True)
    return result


def worker(args):
    plan = read(args.output / "plan.json")
    verify_hashes(plan)
    require(socket.gethostname() == plan["host"], "Resume this queue on its original evaluation node")
    preflight(plan, args.variant)
    if args.command == "preflight":
        return
    for k in (0, 20):
        stage(plan, args.output, args.variant, k, True, args.retry_failed)
    native = stage(plan, args.output, args.variant, 0, False, args.retry_failed)
    refined = stage(plan, args.output, args.variant, 20, False, args.retry_failed)
    paired = native[1][["log_name", "scenario", "score"]].merge(
        refined[1][["log_name", "scenario", "score"]], on=["log_name", "scenario"],
        suffixes=("_native", "_K20"), validate="one_to_one")
    paired["delta_100"] = 100 * (paired.score_K20.astype(float) - paired.score_native.astype(float))
    paired.to_csv(args.output / args.variant / "paired_scores.csv", index=False)
    comparison = dict(variant=args.variant, native_100=native[3], K20_100=refined[3],
        delta_100=refined[3] - native[3], scenarios=len(paired), seed=plan["seed"],
        improved=int((paired.delta_100 > 1e-8).sum()), worsened=int((paired.delta_100 < -1e-8).sum()),
        tied=int((paired.delta_100.abs() <= 1e-8).sum()))
    write(args.output / args.variant / "comparison.json", comparison)
    print("PAIRED " + json.dumps(comparison), flush=True)


def summarize(out):
    rows = [read(out / name / "comparison.json") for name in VARIANTS if (out / name / "comparison.json").exists()]
    write(out / "summary.json", {row["variant"]: row for row in rows})
    if rows:
        with (out / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run(args):
    plan = prepare(args)
    require(socket.gethostname() == plan["host"], "Use the original evaluation node for this queue")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", GPU_IDS) == GPU_IDS, "Use physical GPU IDs 0,1,2,3,4,5,6,7")
    with (args.output / "queue.lock").open("a") as lock, Path("/tmp/score_random_baselines_1025.lock").open("a") as node_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(node_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(args.output / "process.json", dict(host=socket.gethostname(), pid=os.getpid()))
        while subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
            print("Waiting for the eight GPUs to be free; no existing process will be stopped", flush=True)
            write(args.output / "status.json", dict(state="waiting_gpu"))
            time.sleep(30)
        # Check every environment/checkpoint before the first expensive full job.
        for command in ("preflight", "worker"):
            for name in VARIANTS:
                write(args.output / "status.json", dict(state="running", phase=command, variant=name))
                runtime_spec = plan["sources"][name]["runtime"]
                argv = [runtime_spec["python"], "-u", str(Path(__file__).resolve()), command,
                        "--output", str(args.output), "--variant", name]
                if args.retry_failed:
                    argv.append("--retry-failed")
                log = args.output / (name + "." + command + ".log")
                print(f"START {command} {name}; log={log}", flush=True)
                with log.open("a") as handle:
                    subprocess.run(argv, cwd=runtime_spec["cwd"], env=environment(plan, name),
                                   stdout=handle, stderr=subprocess.STDOUT, check=True)
                if command == "worker":
                    summarize(args.output)
        write(args.output / "status.json", dict(state="complete", full_evaluations=12))
        write(args.output / "COMPLETE.json", dict(full_evaluations=12, variants=list(VARIANTS),
                                                 scenarios=len(plan["scenario_keys"])))
        (args.output / "FAILED.json").unlink(missing_ok=True)
        print("ALL COMPLETE: six native/K20 pairs; Diffusion Planner was not rerun", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status", "worker", "preflight"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/home/data/wyk"))
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--sources", type=Path, help="JSON name -> completed hard_NR native result directory")
    parser.add_argument("--score-config", type=Path)
    parser.add_argument("--score-checkpoint", type=Path)
    parser.add_argument("--frozen-score-code", type=Path)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--retry-failed", action="store_true", help="Preserve incomplete attempts and retry them; never overwrite")
    args = parser.parse_args()
    args.output = args.output.resolve()
    try:
        if args.command == "status":
            print(json.dumps(read(args.output / "status.json"), indent=2))
            if (args.output / "summary.json").exists():
                print(json.dumps(read(args.output / "summary.json"), indent=2))
        elif args.command in ("worker", "preflight"):
            require(args.variant is not None, "--variant is required for worker/preflight")
            worker(args)
        elif args.command == "prepare":
            prepare(args)
        else:
            run(args)
    except BaseException as error:
        if args.command == "run" and args.output.exists():
            write(args.output / "FAILED.json", dict(error=str(error), traceback=traceback.format_exc()))
            write(args.output / "status.json", dict(state="failed", error=str(error)))
        raise


if __name__ == "__main__":
    main()
