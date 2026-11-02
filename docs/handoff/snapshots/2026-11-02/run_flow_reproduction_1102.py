#!/usr/bin/env python3
"""Isolated Flow K0 parity check followed by training-consistent-noise hard NR."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from uuid import uuid4

# The single-file delivery embeds the small parity module here.
PARITY_SOURCE = '"""Compare an independent official planner with the K0 wrapper on real inputs.\n\nThis module does not change Flow sampling. Run it with the original official\npackage before testing a separately patched, training-consistent sampler.\n"""\n\nfrom dataclasses import replace\nimport json\nimport os\nfrom pathlib import Path\nimport random\nfrom tempfile import TemporaryDirectory\n\nimport numpy as np\nimport torch\nfrom omegaconf import OmegaConf\n\nfrom flow_score_cfg18.planner import FlowScorePlanner, TrajectorySampling\nfrom native_planner_score.planner import preserve_random_state\n\n\ndef _capture_rng(device):\n    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),\n            torch.cuda.get_rng_state().clone() if str(device) == "cuda" else None)\n\n\ndef _restore_rng(state):\n    random.setstate(state[0])\n    np.random.set_state(state[1])\n    torch.set_rng_state(state[2])\n    if state[3] is not None:\n        torch.cuda.set_rng_state(state[3])\n\n\ndef _same_rng(left, right):\n    numpy_equal = (left[1][0] == right[1][0]\n                   and np.array_equal(left[1][1], right[1][1])\n                   and left[1][2:] == right[1][2:])\n    cuda_equal = ((left[3] is None and right[3] is None)\n                  or (left[3] is not None and right[3] is not None\n                      and torch.equal(left[3], right[3])))\n    return left[0] == right[0] and numpy_equal and torch.equal(left[2], right[2]) and cuda_equal\n\n\ndef compare_trajectories(reference, actual):\n    """Compare nuPlan\'s sampled rear-axle states, including their timestamps."""\n    def extract(trajectory):\n        states = trajectory.get_sampled_trajectory()\n        times = np.asarray([state.time_point.time_us for state in states], dtype=np.int64)\n        poses = np.asarray([(state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading)\n                            for state in states], dtype=np.float64)\n        return times, poses\n\n    ref_time, ref_pose = extract(reference)\n    act_time, act_pose = extract(actual)\n    if ref_pose.shape != act_pose.shape or ref_pose.ndim != 2 or ref_pose.shape[1] != 3:\n        return dict(trajectories_equal=False, reason="Different or empty sampled-state shapes")\n    finite = bool(np.isfinite(ref_pose).all() and np.isfinite(act_pose).all())\n    if not finite:\n        return dict(trajectories_equal=False, reason="Nonfinite sampled states")\n    xy_error = float(np.max(np.abs(ref_pose[:, :2] - act_pose[:, :2])))\n    angle_difference = ref_pose[:, 2] - act_pose[:, 2]\n    heading_error = float(np.max(np.abs(np.arctan2(np.sin(angle_difference), np.cos(angle_difference)))))\n    timestamps_equal = bool(np.array_equal(ref_time, act_time))\n    return dict(trajectories_equal=bool(timestamps_equal and xy_error <= 1e-6 and heading_error <= 1e-6),\n                sampled_states=len(ref_time), timestamps_equal=timestamps_equal,\n                max_xy_abs_error_m=xy_error, max_heading_error_rad=heading_error,\n                absolute_tolerance=1e-6)\n\n\nclass ParityPlanner(FlowScorePlanner):\n    """One-frame, independent native-vs-wrapper numerical check; never refines."""\n\n    def __init__(self, parity_output, **kwargs):\n        if kwargs.get("steps", 0) != 0 or kwargs.get("cfg_weight") != 1.8:\n            raise ValueError("Parity requires Vanilla K0 and explicit CFG=1.8")\n        if kwargs.get("evaluation_seed") != 20260922:\n            raise ValueError("Parity requires the frozen seed=20260922")\n        super().__init__(**kwargs)\n        self.parity_output = Path(parity_output)\n        self._parity_checked = False\n\n    def initialize(self, initialization):\n        # Separate route lists: official feature processing can repair the route.\n        reference_initialization = replace(initialization,\n                                           route_roadblock_ids=list(initialization.route_roadblock_ids))\n        super().initialize(initialization)\n        wrapper_rng = _capture_rng(self.native_device)\n        from flow_planner.planner import FlowPlanner\n\n        with preserve_random_state(self.native_device):\n            random.seed(self.evaluation_seed)\n            np.random.seed(self.evaluation_seed)\n            torch.manual_seed(self.evaluation_seed)\n            with TemporaryDirectory(prefix="flow_parity_native_") as temporary:\n                config_path = Path(temporary) / "model_config.yaml"\n                OmegaConf.save(OmegaConf.create(self.model_config), config_path)\n                self._reference = FlowPlanner(\n                    config_path=str(config_path), ckpt_path=self.checkpoint_path,\n                    past_trajectory_sampling=TrajectorySampling(num_poses=20, time_horizon=2.),\n                    future_trajectory_sampling=TrajectorySampling(num_poses=80, time_horizon=8.),\n                    enable_ema=False, device=self.native_device, use_cfg=True, cfg_weight=1.8,\n                )\n            self._reference.initialize(reference_initialization)\n            self._initial_rng_equal = _same_rng(wrapper_rng, _capture_rng(self.native_device))\n        self._parity_checked = False\n\n    @torch.no_grad()\n    def compute_planner_trajectory(self, current_input):\n        if self._parity_checked:\n            return super().compute_planner_trajectory(current_input)\n        lights = list(current_input.traffic_light_data or [])\n        initial_rng = _capture_rng(self.native_device)\n        reference = self._reference.compute_planner_trajectory(\n            replace(current_input, traffic_light_data=list(lights)))\n        reference_rng = _capture_rng(self.native_device)\n        _restore_rng(initial_rng)\n        actual = super().compute_planner_trajectory(\n            replace(current_input, traffic_light_data=list(lights)))\n        actual_rng = _capture_rng(self.native_device)\n        result = compare_trajectories(reference, actual)\n        runtime_cfg = [float(planner._planner.flow_ode.cfg_weight)\n                       for planner in (self._reference, self._native)]\n        result.update(initial_rng_equal=self._initial_rng_equal,\n                      final_rng_equal=_same_rng(reference_rng, actual_rng),\n                      native_and_wrapper_sampler_cfg=runtime_cfg,\n                      seed=self.evaluation_seed,\n                      timestamp_us=current_input.iteration.time_point.time_us,\n                      scope="one_real_scene_first_frame_only")\n        passed = (result["trajectories_equal"] and result["initial_rng_equal"]\n                  and result["final_rng_equal"] and runtime_cfg == [1.8, 1.8])\n        result["state"] = "complete" if passed else "failed"\n        self.parity_output.parent.mkdir(parents=True, exist_ok=True)\n        temporary = self.parity_output.with_name(self.parity_output.name + f".{os.getpid()}.tmp")\n        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\\n")\n        temporary.replace(self.parity_output)\n        if not passed:\n            raise RuntimeError(f"Native/wrapper numerical parity failed: {self.parity_output}")\n        self._parity_checked = True\n        del self._reference\n        print(f"PARITY PASS: {self.parity_output}", flush=True)\n        return actual\n'
SEED = 20260922
PATCH_OLD = "        x_init = torch.randn((B, self.action_num, self.planner_params['action_len'], self.planner_params['state_dim']), device=self.device)"
PATCH_NEW = """        # Sample full-trajectory noise before overlapping chunking, as in training.
        x_full = torch.randn((B, 1, self.planner_params['future_len'], self.planner_params['state_dim']), device=self.device)
        x_init = torch.cat(traj_chunking(x_full, self.planner_params['action_len'], self.planner_params['action_overlap']), dim=1)"""


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def patch_noise(text):
    if text.count(PATCH_OLD) != 1:
        raise ValueError("Official noise initialization differs; no source was overwritten")
    updated = text.replace(PATCH_OLD, PATCH_NEW, 1)
    compile(updated, "flow_planner.py", "exec")
    return updated


def set_state(out, state, **kwargs):
    write(out / "status.json", dict(state=state, host=socket.gethostname(),
          pid=os.getpid(), updated_unix=time.time(), **kwargs))


def prepare(args):
    if (args.output / "manifest.json").exists():
        manifest = read(args.output / "manifest.json")
        for path, expected in manifest["files"].items():
            if digest(path) != expected:
                raise ValueError(f"Frozen reproduction input changed: {path}")
        return manifest
    plan = read(args.source / "plan.json")
    source = plan["sources"]["flow_planner"]
    cfg = copy.deepcopy(plan["configs"]["hard_NR"]["flow_planner"]["0"])
    planner = next(iter(cfg["planner"].values()))
    cohort = sorted(map(tuple, plan["cohorts"]["hard_NR"]))
    if not (len(cohort) == len(set(cohort)) == 272 and cfg["seed"] == SEED
            and planner["steps"] == 0 and planner["cfg_weight"] == 1.8
            and planner["evaluation_seed"] == SEED
            and cfg["observation"]["_target_"].endswith(".TracksObservation")):
        raise ValueError("Require completed-protocol hard NR / 272 / seed20260922 / CFG1.8 / K0")
    package = Path(source["code"]) / "flow_planner"
    adapter = Path(source["adapter"])
    repo = args.root / "Score_Function_TwoNode_1025"
    dependencies = [Path(source["checkpoint"]), Path(source["config_path"])]
    for directory in [package, adapter / "flow_score_cfg18", repo / "urban_driver_score", repo / "native_planner_score"]:
        dependencies.extend(directory.rglob("*.py"))
    for path in dependencies:
        expected = plan["inputs"].get(str(path))
        if expected is None or digest(path) != expected:
            raise ValueError(f"Source differs from the completed batch: {path}")
    noise_file = Path("flow_planner/model/flow_planner_model/flow_planner.py")
    patched = patch_noise((Path(source["code"]) / noise_file).read_text())
    code = args.output / "code"
    if code.exists():
        code.rename(args.output / f"code.incomplete.{time.time_ns()}")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".git")
    for mode in ("original", "aligned"):
        shutil.copytree(package, code / mode / "flow_planner", ignore=ignore)
    (code / "aligned" / noise_file).write_text(patched)
    runtime = code / "runtime"
    for name, origin in [("flow_score_cfg18", adapter), ("urban_driver_score", repo), ("native_planner_score", repo)]:
        shutil.copytree(origin / name, runtime / name, ignore=ignore)
    for name in ("evaluate_native_score_queue.py", "baseline_extension_config_1028.py", "baseline_job_cleanup_1028.py"):
        shutil.copy2(repo / "scripts" / name, runtime / name)
    helper = runtime / "flow_repro_1102"
    helper.mkdir()
    (helper / "__init__.py").write_text("")
    content = PARITY_SOURCE
    if content is None:
        content = (Path(__file__).resolve().parents[1] / "flow_repro_1102/planner.py").read_text()
    (helper / "planner.py").write_text(content)
    assets = args.output / "assets"
    assets.mkdir(exist_ok=True)
    for source_key, target_name, planner_key in [("checkpoint", "model.pth", "checkpoint_path"),
                                                ("config_path", "model_config.yaml", "config_path")]:
        shutil.copy2(source[source_key], assets / target_name)
        planner[planner_key] = str(assets / target_name)
    # Recheck the actual copied bytes, rather than assuming a shared source stayed unchanged.
    for source_key, target_name in [("checkpoint", "model.pth"), ("config_path", "model_config.yaml")]:
        if digest(assets / target_name) != plan["inputs"][source[source_key]]:
            raise ValueError("Source asset changed while copying")
    write(args.output / "base_config.json", cfg)
    write(args.output / "cohort.json", cohort)
    files = list(code.rglob("*.py")) + list(assets.iterdir()) + [args.output / "base_config.json", args.output / "cohort.json"]
    manifest = dict(seed=SEED, cfg_weight=1.8, scenarios=272, gpus=list(range(8)),
        source_queue=str(args.source), source_commit=source["source_commit"],
        source_plan_sha256=digest(args.source / "plan.json"),
        entry=source["runtime"]["entry"], frozen_score_code=plan["frozen_score_code"],
        patch="full_80_point_noise_then_overlapping_chunking; not yet author-confirmed",
        files={str(path): digest(path) for path in files})
    write(args.output / "manifest.json", manifest)
    print("Prepared: isolated source/weights; hard NR 272; CFG1.8; seed20260922; GPUs0-7", flush=True)
    return manifest


def controller_matches(argv, script, output):
    arguments = list(argv[1:])
    if arguments and arguments[0] == "-u":
        arguments.pop(0)
    if len(arguments) < 2 or Path(arguments[0]).resolve() != Path(script).resolve() or arguments[1] != "run":
        return False
    try:
        return Path(arguments[arguments.index("--output") + 1]).resolve() == Path(output).resolve()
    except (ValueError, IndexError):
        return False


def pause_baselines(args):
    record_path = args.source / "process.json"
    if not record_path.exists():
        return
    record = read(record_path)
    if record.get("host") != socket.gethostname():
        print("Source controller is recorded on another node; no remote process signalled", flush=True)
        return
    process = Path("/proc") / str(int(record["pid"]))
    try:
        argv = process.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0")
        before = process.joinpath("stat").read_text().rpartition(")")[2].split()[19]
    except FileNotFoundError:
        return
    script = args.root / "Score_Function_TwoNode_1025/scripts/evaluate_baselines_val_reactive_1028.py"
    if not controller_matches(argv, script, args.source):
        print("Recorded PID does not match the baseline controller; no signal sent", flush=True)
        return
    try:
        after = process.joinpath("stat").read_text().rpartition(")")[2].split()[19]
        current = process.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0")
        if before == after and controller_matches(current, script, args.source):
            os.kill(int(record["pid"]), signal.SIGINT)
            print("Paused the exact baseline controller; completed outputs preserved", flush=True)
    except (ProcessLookupError, FileNotFoundError):
        pass


def stage_environment(args, manifest, mode):
    env = os.environ.copy()
    root, code = args.root, args.output / "code"
    paths = [code / mode, code / "runtime", Path(manifest["frozen_score_code"]),
             root / "Diffusion-Planner", root / "nuplan-devkit"]
    env.update(PYTHONPATH=os.pathsep.join(map(str, paths)), CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        NUPLAN_DATA_ROOT=str(root / "dataset/nuplan"), NUPLAN_MAPS_ROOT=str(root / "dataset/nuplan/maps"),
        NUPLAN_EXP_ROOT=str(root / "dataset/nuplan/exp"), NUPLAN_DEVKIT_ROOT=str(root / "nuplan-devkit"),
        NODE_RANK="0", NUM_NODES="1", HYDRA_FULL_ERROR="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK", "RAY_ADDRESS", "NUPLAN_HYDRA_CONFIG_PATH"):
        env.pop(key, None)
    return env


def stage(args):
    manifest = read(args.output / "manifest.json")
    mode = "original" if args.phase == "parity" else "aligned"
    env = stage_environment(args, manifest, mode)
    os.environ.update(env)
    sys.path[:0] = env["PYTHONPATH"].split(os.pathsep)
    os.chdir(args.output / "code" / mode)
    from importlib.util import find_spec
    flow_root = (args.output / "code" / mode).resolve()
    for module_name in ("flow_planner.planner",
                        "flow_planner.model.flow_planner_model.flow_planner"):
        spec = find_spec(module_name)
        expected_file = flow_root.joinpath(*module_name.split(".")).with_suffix(".py")
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected_file:
            raise RuntimeError(f"Wrong Flow module imported: {module_name}")
    import evaluate_native_score_queue as shared
    from baseline_extension_config_1028 import order_metric_callbacks
    target = args.output / args.phase
    expected = sorted(map(tuple, read(args.output / "cohort.json")))
    cfg = order_metric_callbacks(shared.execution_config(read(args.output / "base_config.json"), target, 8))
    cfg.update(job_name="flow_repro_1102_" + args.phase, enable_simulation_progress_bar=True)
    if args.phase == "parity":
        expected = expected[:1]
        cfg["scenario_filter"].update(scenario_tokens=[expected[0][1]], log_names=[expected[0][0]])
        planner = next(iter(cfg["planner"].values()))
        planner.update(_target_="flow_repro_1102.planner.ParityPlanner", parity_output=str(args.output / "parity.json"))
        # No benchmark metrics are needed for a single real-scene check.
        cfg["run_metric"] = False
        cfg["main_callback"] = {key: value for key, value in cfg["main_callback"].items()
                                if "Metric" not in value.get("_target_", "")}
    if target.exists():
        target.rename(target.with_name(target.name + f".interrupted.{time.time_ns()}"))
    if args.phase == "parity" and (args.output / "parity.json").exists():
        (args.output / "parity.json").rename(args.output / f"parity.previous.{time.time_ns()}.json")
    shared.run_simulation(cfg, target, Path(manifest["entry"]), os.environ.copy())
    if args.phase == "parity":
        _, actual = shared.read_completed_run(target, 1)
        if actual != expected or read(args.output / "parity.json")["state"] != "complete":
            raise RuntimeError("Native/wrapper first-frame comparison did not pass")
        result = dict(state="complete", phase="parity", scenarios=1)
    else:
        _, _, actual, score = shared.read_result(target, 272)
        if actual != expected:
            raise RuntimeError("Actual scenario identities differ from frozen hard14 cohort")
        result = dict(state="complete", phase="aligned", scenarios=272, score_100=score,
                      paper_reference=76.47, delta_to_paper=score - 76.47)
        print(f"COMPLETE hard_NR Flow Vanilla aligned-noise: {score:.4f}/100; 272/272", flush=True)
    write(args.output / (args.phase + ".complete.json"), result)


def run(args):
    manifest = read(args.output / "manifest.json")
    sys.path.insert(0, str(args.output / "code/runtime"))
    from baseline_job_cleanup_1028 import cleanup_job
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        if args.pause_baselines:
            pause_baselines(args)
        with Path("/tmp/score_random_baselines_1025.lock").open("a") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    set_state(args.output, "waiting_node_lock")
                    print("WAIT: baseline controller is releasing its node lock", flush=True)
                    time.sleep(5)
            for phase in ("parity", "aligned"):
                receipt = args.output / (phase + ".complete.json")
                if receipt.exists() and read(receipt).get("state") == "complete":
                    print("SKIP complete", phase, flush=True)
                    continue
                while subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
                    set_state(args.output, "waiting_gpu", phase=phase)
                    print("WAIT GPUs0-7: other processes are still using GPUs", flush=True)
                    time.sleep(10)
                env = stage_environment(args, manifest, "original" if phase == "parity" else "aligned")
                token = uuid4().hex
                env.update(SCORE_BASELINE_QUEUE=str(args.output), SCORE_BASELINE_JOB_TOKEN=token)
                log = args.output / (phase + ".log")
                set_state(args.output, "running", phase=phase, log=str(log))
                print(f"START {phase}: {log}", flush=True)
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "stage", "--root", str(args.root),
                           "--source", str(args.source), "--output", str(args.output), "--phase", phase]
                child = None
                try:
                    with log.open("a") as stream:
                        child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                        code = child.wait()
                    if code or not receipt.exists() or read(receipt).get("state") != "complete":
                        raise RuntimeError(f"{phase} failed; inspect {log}")
                finally:
                    try:
                        cleanup = cleanup_job(args.output, token, process_group=child.pid if child else None)
                        if cleanup.get("errors"):
                            print("CLEANUP warning", cleanup, flush=True)
                    except Exception as error:
                        print(f"CLEANUP warning: {error}", flush=True)
                    if child is not None and child.poll() is None:
                        try:
                            child.terminate()
                            try:
                                child.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait(timeout=5)
                        except Exception as error:
                            print(f"CHILD cleanup warning: {error}", flush=True)
                print(f"COMPLETE {phase}: {read(receipt)}", flush=True)
            set_state(args.output, "complete", **{key: value for key, value in read(args.output / "aligned.complete.json").items() if key != "state"})
    except BaseException as error:
        set_state(args.output, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "run", "stage", "status"))
    parser.add_argument("--root", type=Path, default=Path("/home/data/wyk"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("parity", "aligned"))
    parser.add_argument("--pause-baselines", action="store_true")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.source = (args.source or args.root / "dataset/nuplan/exp/baselines8_flowcfg18_seed20260922_1028_v4").resolve()
    args.output = (args.output or args.root / "dataset/nuplan/exp/flow_repro_noisealigned_seed20260922_1102_v1").resolve()
    if args.action == "status":
        try:
            while True:
                if args.watch:
                    print("\033[2J\033[H", end="")
                status = read(args.output / "status.json") if (args.output / "status.json").exists() else {"state": "not_started"}
                print(json.dumps(status, ensure_ascii=False), flush=True)
                if status.get("log") and Path(status["log"]).exists():
                    with Path(status["log"]).open("rb") as stream:
                        stream.seek(max(0, Path(status["log"]).stat().st_size - 131072))
                        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", stream.read().decode(errors="replace"))
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    bars = [line for line in lines if "Ray objects:" in line]
                    if bars:
                        print(bars[-1])
                    print("\n".join(lines[-3:]))
                for name in ("parity.json", "aligned.complete.json"):
                    if (args.output / name).exists():
                        print(name, json.dumps(read(args.output / name), ensure_ascii=False))
                if not args.watch or status["state"] in ("complete", "failed", "interrupted", "not_started"):
                    break
                time.sleep(5)
        except KeyboardInterrupt:
            pass
        return
    if args.action == "launch":
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / "launcher.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            prepare(args)
            python = args.root / "envs/flow_planner/bin/python"
            if not python.is_file():
                raise FileNotFoundError(python)
            gpu_ids = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True).splitlines()
            if len(gpu_ids) < 8:
                raise RuntimeError("This reproduction requires eight visible GPUs on the node")
            command = [str(python), "-u", str(Path(__file__).resolve()), "run", "--root", str(args.root), "--source", str(args.source), "--output", str(args.output)]
            if args.pause_baselines:
                command.append("--pause-baselines")
            set_state(args.output, "launching")
            with (args.output / "queue.log").open("a") as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, pass_fds=(lock.fileno(),))
            print(f"Launched PID {child.pid}; log={args.output / 'queue.log'}", flush=True)
    elif args.action == "run":
        run(args)
    else:
        stage(args)


if __name__ == "__main__":
    main()
