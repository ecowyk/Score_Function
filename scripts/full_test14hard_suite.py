"""Frozen final L/G/DiT Test14-Hard suite; one eight-GPU Ray evaluation at a time.

Uses the existing sim_score_planner.sh, including its original nuPlan output,
metrics, and NuBoard logs. No training changes and no new scenario subset.
"""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

KS = (5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000)
NAMES = ("S05-L", "S05-G", "S05-DiT")
CHALLENGE = "closed_loop_nonreactive_agents"
METRICS = (
    "score",
    "no_ego_at_fault_collisions",
    "time_to_collision_within_bound",
    "ego_is_comfortable",
    "ego_progress_along_expert_route",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "speed_limit_compliance",
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def hashes(root, pattern="*.py"):
    return {str(p.relative_to(root)): digest(p) for p in sorted(Path(root).rglob(pattern))}


def env_for(root, code, cuda="0,1,2,3,4,5,6,7"):
    env = dict(os.environ)
    for k in list(env):
        if k in {
            "RAY_ADDRESS",
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "NUPLAN_HYDRA_CONFIG_PATH",
        } or k.startswith("TORCHELASTIC_"):
            env.pop(k)
    env.update(
        DP_DIR=str(root / "Diffusion-Planner"),
        NUPLAN_DEVKIT_ROOT=str(root / "nuplan-devkit"),
        NUPLAN_DATA_ROOT=str(root / "dataset/nuplan"),
        NUPLAN_MAPS_ROOT=str(root / "dataset/nuplan/maps"),
        NUPLAN_EXP_ROOT=str(root / "dataset/nuplan/exp"),
        CUDA_VISIBLE_DEVICES=cuda,
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        NODE_RANK="0",
        HYDRA_FULL_ERROR="1",
        PYTHONUNBUFFERED="1",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        map(str, (code, root / "Diffusion-Planner", root / "nuplan-devkit"))
    )
    if os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] += os.pathsep + os.environ["PYTHONPATH"]
    return env


def jobs():
    return [dict(id="DP_K0", model="S05-G", K=0)] + [
        dict(id=f"{name}_K{k}", model=name, K=k) for k in KS for name in NAMES
    ]


def model_paths(out, name):
    return out / "models" / name / "config.json", out / "models" / name / "final_ema.pt"


def inspect_model(args):
    # Fresh interpreter, with the model's own frozen implementation first.
    import torch

    from score_function.utils.checkpoint import load_selected
    from score_function.utils.planner_utils import planner_identity
    from score_function.utils.train_utils import load_tensor

    torch.set_num_threads(2)
    out = args.output
    reference = load_tensor(model_paths(out, "S05-G")[1])
    if args.name == "S05-DiT":
        source = out / "models/S05-DiT/training_final.pt"
        state = load_tensor(source)
        if not state.get("finished") or state.get("epoch") != 529 or state.get("step") != 244398:
            raise ValueError("DiT last.pt must be final: finished=True, epoch=529, step=244398")
        if state["config"]["model"]["architecture"] != "dit_score_function":
            raise ValueError("Not a DiT training checkpoint")
        selected = {
            k: copy.deepcopy(state[k])
            for k in (
                "schema_version",
                "method",
                "planner",
                "ego_normalizer",
                "sigma_score",
                "config",
                "epoch",
                "step",
            )
        }
        selected.update(
            score_branch=state["ema_branch"],
            weight_kind="ema",
            criterion="frozen final EMA for full Test14-Hard suite",
            source_last_sha256=digest(source),
        )
        config_path, checkpoint = model_paths(out, args.name)
        write(config_path, selected["config"])
        torch.save(selected, checkpoint)
    config_path, checkpoint = model_paths(out, args.name)
    config = read(config_path)
    branch, state = load_selected(config, checkpoint, "cpu")
    for k in ("planner", "ego_normalizer", "sigma_score"):
        if state[k] != reference[k]:
            raise ValueError(f"{args.name}: {k} differs from this run's S05-G")
    common = read(model_paths(out, "S05-G")[0])
    if config["reference_seed"] != common["reference_seed"]:
        raise ValueError(f"{args.name}: reference_seed differs; no silent seed change")
    if any(
        config["refinement"][k] != common["refinement"][k] for k in ("gamma", "heading_projection")
    ):
        raise ValueError(f"{args.name}: refinement settings differ")
    if state["weight_kind"] != "ema" or not all(
        torch.isfinite(t).all() for t in state["score_branch"].values()
    ):
        raise ValueError("Require finite final EMA")
    if state.get("epoch") != 529 or state.get("step") != 244398:
        raise ValueError("Require final 529-epoch / 244398-update checkpoint")
    if planner_identity(config) != state["planner"]:
        raise ValueError("Original DP checkpoint/config/source identity changed")
    with torch.no_grad():
        value = branch(torch.zeros(1, 80, 4), torch.zeros(1, 107, 192), torch.zeros(1, 192))
    if value.shape != (1, 80, 4) or not torch.isfinite(value).all():
        raise ValueError("Model forward preflight failed")
    if args.name == "S05-DiT":
        import importlib.util
        from types import SimpleNamespace

        from score_function.model.refinement import refine_ego

        spec = importlib.util.spec_from_file_location(
            "original_dit_refinement", out / "original_dit_refinement.py"
        )
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        normalizer = SimpleNamespace(
            **{
                k: torch.tensor(state["ego_normalizer"][k]).reshape(1, 1, 4)
                for k in ("mean", "std")
            }
        )
        rng = torch.Generator().manual_seed(20261009)
        x = torch.randn(1, 80, 4, generator=rng) * 0.25
        scene = torch.randn(1, 107, 192, generator=rng)
        route = torch.randn(1, 192, generator=rng)
        kw = dict(
            sigma=state["sigma_score"],
            gamma=common["refinement"]["gamma"],
            heading_projection=common["refinement"]["heading_projection"],
            record_trace=False,
        )
        for k in (0, 5, 20):
            a, _ = original.refine_ego(branch, x, scene, route, normalizer, steps=k, **kw)
            b, _ = refine_ego(branch, x, scene, route, normalizer, steps=k, **kw)
            if not torch.equal(a, b):
                raise ValueError("DiT no-trace optimization changed the update")
        print("DiT original/no-trace CPU outputs bitwise equal at K=0,5,20", flush=True)
    print(
        f"VERIFIED {args.name}: final EMA, same DP/sigma/normalizer/seed, CPU forward OK",
        flush=True,
    )


def validate_result(run, job, plan, out):
    import numpy as np
    import pandas as pd
    from omegaconf import OmegaConf

    run = Path(run)
    cfg = OmegaConf.load(run / "code/hydra/config.yaml")
    planner = cfg.planner.score_function_planner
    config_path, checkpoint = model_paths(out, job["model"])
    if read(Path(planner.score_config)) != read(config_path) or digest(
        planner.score_checkpoint
    ) != digest(checkpoint):
        raise ValueError("Saved evaluation weights/config differ from the frozen model")
    config = read(config_path)
    if int(planner.steps) != job["K"] or float(planner.gamma) != plan["gamma"]:
        raise ValueError("K/gamma mismatch")
    heading = planner.heading_projection
    if (config["refinement"]["heading_projection"] if heading is None else heading) != plan[
        "heading_projection"
    ]:
        raise ValueError("Heading projection mismatch")
    if planner.trace_dir is not None or str(planner.device) != "cuda":
        raise ValueError("Unexpected trace/device setting")
    if CHALLENGE not in run.parts or set(cfg.scenario_filter.scenario_tokens) != set(
        plan["tokens"]
    ):
        raise ValueError("Require full official Test14-Hard, nonreactive closed loop")
    if (
        cfg.scenario_filter.limit_total_scenarios is not None
        or cfg.scenario_filter.num_scenarios_per_type is not None
    ):
        raise ValueError("Unexpected scenario limits")
    if str(cfg.scenario_builder.data_root).rstrip("/") != str(
        Path(plan["root"]) / "dataset/nuplan/nuplan-v1.1/test"
    ):
        raise ValueError("Not the nuPlan test data root")
    if not str(cfg.worker._target_).endswith("RayDistributed"):
        raise ValueError("Expected Ray scenario parallelism")
    files = sorted((run / "aggregator_metric").glob("*.parquet"))
    if len(files) != 1:
        raise ValueError("Expected one finished official metric aggregate")
    frame = pd.read_parquet(files[0])
    detail = frame.loc[frame.log_name.notna()]
    if len(detail) != 272 or set(detail.scenario.astype(str)) != set(plan["tokens"]):
        raise ValueError(
            "Missing/extra/duplicate scenario metrics; not 272 complete test scenarios"
        )
    report = pd.read_parquet(run / str(cfg.runner_report_file))
    if len(report) != 272 or not report.succeeded.eq(True).all():
        raise ValueError("Some simulations failed; cannot mark the run complete")
    expected = set(zip(detail.log_name.astype(str), detail.scenario.astype(str)))
    actual = set(zip(report.log_name.astype(str), report.scenario_name.astype(str)))
    if actual != expected or len(actual) != 272:
        raise ValueError("Simulation report and metrics have different scenarios")
    final = frame.loc[frame.scenario.eq("final_score")]
    if len(final) != 1 or int(final.iloc[0].num_scenarios) != 272:
        raise ValueError("Wrong final scenario count")
    result = {k: float(final.iloc[0][k]) for k in METRICS}
    if (
        not np.isfinite(detail[list(METRICS)].to_numpy(dtype=float)).all()
        or not np.isfinite(list(result.values())).all()
    ):
        raise ValueError("Nonfinite or missing metrics")
    return result


def verify(out, plan):
    for path, sha in plan["files"].items():
        if digest(path) != sha:
            raise ValueError(f"Frozen input changed: {path}")


def prepare(args):
    import yaml

    root, out = args.root.resolve(), args.output.resolve()
    if (out / "plan.json").exists():
        verify(out, read(out / "plan.json"))
        print("Already prepared; frozen inputs verified.")
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Incomplete preparation exists: {out}; use --output with a new directory")
    source = root / "Score_Function_aligned_v1"
    snap = root / "outputs/score_aligned_1004/LG_closed_loop_k_1009_v1"
    profile = snap / "execution_profiles/no_trace_v1"
    trials = root / "outputs/score_aligned_1004/after_L_G_DiT_v1"
    training = read(trials / "plan.json")
    dit_job = next(j for j in training["jobs"] if j["name"] == "S05-DiT")
    dit_code = Path(args.dit_code or dit_job["code"]).resolve()
    dit_run = Path(args.dit_run or dit_job["run"]).resolve()
    metadata = read(profile / "manifest.json")
    if hashes(profile / "code/score_function") != metadata["sources"]:
        raise ValueError("Existing no_trace_v1 package was changed")
    original_refinement = dit_code / "score_function/model/refinement.py"
    if digest(original_refinement) != metadata["original_sources"]["model/refinement.py"]:
        raise ValueError(
            "DiT refinement differs from the original L/G update; inspect before running"
        )
    for name in (
        "planner/planner.py",
        "model/refined_planner.py",
        "utils/sampling.py",
        "utils/normalizer.py",
    ):
        if digest(dit_code / "score_function" / name) != metadata["sources"][name]:
            raise ValueError(f"DiT uses a different DP/seed/refinement adapter: {name}")
    # Prepare private immutable implementations. Original training checkouts stay untouched.
    for label, package in (
        ("lg", profile / "code/score_function"),
        ("dit", dit_code / "score_function"),
    ):
        code = out / "code" / label
        shutil.copytree(
            package, code / "score_function", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        (code / "configs").mkdir()
        shutil.copy2(source / "sim_score_planner.sh", code / "sim_score_planner.sh")
        shutil.copy2(source / "configs/nuplan_planner.yaml", code / "configs/nuplan_planner.yaml")
    shutil.copy2(
        profile / "code/score_function/model/refinement.py",
        out / "code/dit/score_function/model/refinement.py",
    )
    shutil.copy2(original_refinement, out / "original_dit_refinement.py")
    for name in ("S05-L", "S05-G"):
        (out / "models" / name).mkdir(parents=True)
        for file in ("config.json", "final_ema.pt"):
            shutil.copy2(snap / "models" / name / file, out / "models" / name / file)
    (out / "models/S05-DiT").mkdir()
    shutil.copy2(dit_run / "score/last.pt", out / "models/S05-DiT/training_final.pt")
    for name in NAMES:
        code = out / "code" / ("dit" if name == "S05-DiT" else "lg")
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "_inspect",
                "--output",
                str(out),
                "--name",
                name,
            ],
            env=env_for(root, code, ""),
            cwd=code,
            check=True,
        )
    g = read(model_paths(out, "S05-G")[0])
    if g["refinement"]["gamma"] != 0.1:
        raise ValueError("This suite expects the current gamma=0.1 protocol")
    split = root / "Diffusion-Planner/diffusion_planner/config/scenario_filter/test14-hard.yaml"
    tokens = yaml.safe_load(split.read_text())["scenario_tokens"]
    if len(tokens) != 272 or len(set(tokens)) != 272:
        raise ValueError("Expected the official 272 unique Test14-Hard tokens")
    plan = dict(
        root=str(root),
        tokens=tokens,
        gamma=0.1,
        heading_projection=g["refinement"]["heading_projection"],
        reference_seed=g["reference_seed"],
        seed_rule="existing SHA256(reference_seed:nuPlan_timestamp), unchanged CPU DP noise draw",
        jobs=jobs(),
        created_unix=time.time(),
        files={},
    )
    os.environ.update(env_for(root, out / "code/lg", ""))
    result_base = (
        root / "dataset/nuplan/exp/exp/simulation" / CHALLENGE / "score_function/test14-hard"
    )
    for k in (20, 50, 100):
        job = dict(id=f"S05-G_K{k}", model="S05-G", K=k)
        candidates = sorted(result_base.glob(f"S05-G_final_test14hard_g01_k{k}_*"), reverse=True)
        errors = []
        for run in candidates:
            try:
                metrics = validate_result(run, job, plan, out)
            except Exception as exc:
                errors.append(f"{run.name}: {exc}")
                continue
            write(
                out / "jobs" / (job["id"] + ".json"),
                dict(job, state="reused", result_dir=str(run), metrics=metrics),
            )
            print(f"REUSE {job['id']}: {100 * metrics['score']:.4f}, 272/272 succeeded", flush=True)
            break
        else:
            raise ValueError(
                f"Cannot verify existing {job['id']}; it will NOT be rerun automatically. "
                + str(errors)
            )
    tracked = [Path(__file__).resolve(), split]
    tracked += [p for p in (out / "models").rglob("*") if p.is_file()]
    tracked += [p for p in (out / "code").rglob("*") if p.suffix in (".py", ".yaml", ".sh")]
    for key in ("planner_checkpoint", "planner_args"):
        tracked.append(Path(g["paths"]["root"]) / g["paths"][key])
    tracked += list((root / "Diffusion-Planner/diffusion_planner").rglob("*.py"))
    tracked += list((root / "nuplan-devkit/nuplan/planning/script/config").rglob("*.yaml"))
    plan["files"] = {str(p.resolve()): digest(p) for p in tracked}
    write(out / "plan.json", plan)
    write(out / "status.json", dict(state="prepared", updated_unix=time.time()))
    print(
        f"READY: 40 results total; 3 G results reused; 37 new full evaluations. seed={plan['reference_seed']}",
        flush=True,
    )


def process_identity(pid):
    try:
        # Namespace-safe enough for our own direct child; killpg is only used on that child.
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def show(out):
    state_path = out / "status.json"
    print("QUEUE:", read(state_path) if state_path.exists() else "not prepared")
    for job in jobs():
        path = out / "jobs" / (job["id"] + ".json")
        record = read(path) if path.exists() else {}
        score = record.get("metrics", {}).get("score")
        print(
            f"{job['id']:20s} {record.get('state', 'pending'):12s} "
            + (f"{100 * score:.4f}" if score is not None else "")
        )
        if record.get("state") in ("running", "failed", "interrupted"):
            print("  log:", record.get("log"), record.get("error", ""))
            if record.get("state") == "running" and record.get("log"):
                try:
                    with Path(record["log"]).open("rb") as f:
                        f.seek(max(0, Path(record["log"]).stat().st_size - 4096))
                        lines = f.read().decode(errors="replace").replace("\r", "\n").splitlines()
                    for line in [line for line in lines if line.strip()][-2:]:
                        print(" ", line[-240:])
                except OSError:
                    pass


def summarize(out):
    rows = []
    for job in jobs():
        p = out / "jobs" / (job["id"] + ".json")
        r = read(p) if p.exists() else dict(job, state="pending")
        rows.append(
            dict(job, state=r["state"], result_dir=r.get("result_dir", ""), **r.get("metrics", {}))
        )
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "model", "K", "state", "result_dir", *METRICS])
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    out = args.output.resolve()
    plan = read(out / "plan.json")
    root = Path(plan["root"])
    with (out / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ.update(env_for(root, out / "code/lg"))
        verify(out, plan)
        failures = []
        for job in plan["jobs"]:
            path = out / "jobs" / (job["id"] + ".json")
            old = read(path) if path.exists() else {}
            if old.get("state") in ("complete", "reused"):
                validate_result(old["result_dir"], job, plan, out)
                print("SKIP verified", job["id"], flush=True)
                continue
            if old.get("state") == "running" and old.get("pid") and process_identity(old["pid"]):
                raise RuntimeError(
                    f"Previous child PID {old['pid']} may still be running; inspect before resuming"
                )
            verify(out, plan)
            label = f"{job['id']}_fullsuite_{time.time_ns()}"
            code = out / "code" / ("dit" if job["model"] == "S05-DiT" else "lg")
            config, checkpoint = model_paths(out, job["model"])
            log = out / "logs" / (label + ".log")
            log.parent.mkdir(exist_ok=True)
            record = dict(job, state="running", log=str(log), started_unix=time.time())
            command = [
                "bash",
                str(code / "sim_score_planner.sh"),
                "--python",
                sys.executable,
                "--root",
                str(root),
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--name",
                label,
                "--split",
                "test14-hard",
                "--challenge",
                CHALLENGE,
                "--steps",
                str(job["K"]),
                "--gamma",
                str(plan["gamma"]),
                "--gpus",
                "0,1,2,3,4,5,6,7",
                "--gpu-fraction",
                "0.15",
                "--threads",
                "128",
            ]
            print(f"START {job['id']} | GPUs=0..7 | log={log}", flush=True)
            with log.open("w") as f:
                child = subprocess.Popen(
                    command,
                    cwd=code,
                    env=env_for(root, code),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                record.update(pid=child.pid, command=command)
                write(path, record)
                try:
                    while child.poll() is None:
                        write(
                            out / "status.json",
                            dict(
                                state="running",
                                job=job["id"],
                                pid=child.pid,
                                log=str(log),
                                elapsed_minutes=round(
                                    (time.time() - record["started_unix"]) / 60, 1
                                ),
                                updated_unix=time.time(),
                            ),
                        )
                        time.sleep(2)
                except BaseException:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                    record.update(state="interrupted", error="Coordinator interrupted")
                    write(path, record)
                    write(
                        out / "status.json",
                        dict(state="interrupted", job=job["id"], updated_unix=time.time()),
                    )
                    raise
            try:
                lines = log.read_text(errors="replace").splitlines()
                result_dir = next(
                    line.split(": ", 1)[1] for line in lines if line.startswith("Results: ")
                )
                record["result_dir"] = result_dir
                if child.returncode != 0:
                    raise RuntimeError(f"nuPlan/launcher exit code {child.returncode}")
                record["metrics"] = validate_result(result_dir, job, plan, out)
                record["state"] = "complete"
                print(
                    f"DONE {job['id']}: {100 * record['metrics']['score']:.4f}, 272/272", flush=True
                )
            except Exception as exc:
                record.update(state="failed", error=str(exc))
                failures.append(job["id"])
                print(f"FAILED {job['id']}: {exc}; continuing remaining requested jobs", flush=True)
            record["finished_unix"] = time.time()
            write(path, record)
            summarize(out)
        write(
            out / "status.json",
            dict(
                state="complete" if not failures else "finished_with_failures",
                failed=failures,
                updated_unix=time.time(),
            ),
        )
        summarize(out)
        show(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "status", "_inspect"))
    parser.add_argument("--root", type=Path, default=Path("/home/data/wyk"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dit-code", type=Path)
    parser.add_argument("--dit-run", type=Path)
    parser.add_argument("--name", choices=NAMES)
    args = parser.parse_args()
    args.output = args.output or args.root / "dataset/nuplan/exp/score_full_k_1009_v1"
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "_inspect":
        inspect_model(args)
    elif args.mode == "status":
        show(args.output)
    else:

        def interrupt(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")

        signal.signal(signal.SIGTERM, interrupt)
        run(args)


if __name__ == "__main__":
    main()
