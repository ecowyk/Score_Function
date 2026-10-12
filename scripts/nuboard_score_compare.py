#!/usr/bin/env python3
"""Open existing DP/G20/G50/G200 results together in official nuBoard.

Reads only the user's trusted nuPlan artifacts. No evaluation is rerun and no
saved result, planner, or nuPlan source file is modified. Logs load onto CPU.
"""

import argparse
import gc
import inspect
import io
import json
import lzma
import os
import pickle
import signal
import sys
from pathlib import Path

DEFAULT_SUITE = Path("/home/data/wyk/dataset/nuplan/exp/score_full_k_1009_v1")
JOBS = (("DP_K0", 0), ("S05-G_K20", 20), ("S05-G_K50", 50), ("S05-G_K200", 200))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def discover_runs(suite):
    """Resolve exactly four completed suite jobs, including reused old runs."""
    suite = Path(suite).expanduser().resolve()
    records = []
    for job_id, k in JOBS:
        path = suite / "jobs" / f"{job_id}.json"
        if not path.is_file():
            raise ValueError(f"Missing evaluation record: {path}")
        job = read_json(path)
        if job.get("state") not in ("complete", "reused"):
            raise ValueError(f"{job_id} is not complete/reused: {job.get('state')}")
        if job.get("id") != job_id or job.get("model") != "S05-G" or job.get("K") != k:
            raise ValueError(f"Wrong model/job/K in {path}")
        if not job.get("result_dir"):
            raise ValueError(f"Missing result directory in {path}")
        run = Path(job["result_dir"]).expanduser().resolve()
        if "closed_loop_nonreactive_agents" not in run.parts:
            raise ValueError(f"Expected the original nonreactive closed-loop run: {run}")
        files = sorted(run.glob("*.nuboard"))
        if len(files) != 1:
            raise ValueError(f"{job_id}: expected one .nuboard file, found {len(files)} in {run}")
        # A .nuboard file contains a pickled path dictionary, not simulation data.
        # Like nuBoard itself, this supports only the user's own trusted files.
        with files[0].open("rb") as source:
            header = pickle.load(source)
        folder = header.get("simulation_folder")
        if not folder:
            raise ValueError(f"{job_id}: the .nuboard header has no saved simulation folder")
        simulation = run / folder
        if not simulation.is_dir():
            simulation = Path(header["simulation_main_path"]) / folder
        if not simulation.is_dir():
            raise ValueError(f"{job_id}: saved simulation folder is unavailable: {simulation}")
        records.append(
            {
                "job_id": job_id,
                "K": k,
                "state": job["state"],
                "result_dir": str(run),
                "nuboard_file": str(files[0]),
                "simulation_dir": str(simulation.resolve()),
            }
        )
    # Official nuBoard keys include the simulation directory's parent basename.
    # With these actual suite run names, identical planner names remain distinct.
    labels = [Path(row["simulation_dir"]).parent.name for row in records]
    if len(set(labels)) != len(labels):
        raise ValueError(
            "Selected runs have colliding directory names in nuBoard; keep distinct result directories"
        )
    return records


def runtime_environment(root, suite, inherited=None):
    """Match DP's notebook paths while isolating a CPU-only viewer process."""
    root, suite = Path(root), Path(suite)
    env = dict(os.environ if inherited is None else inherited)
    distributed = {
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "RAY_ADDRESS",
        "NUPLAN_HYDRA_CONFIG_PATH",
    }
    for key in list(env):
        if key in distributed or key.startswith("TORCHELASTIC_"):
            env.pop(key)
    env.update(
        DP_DIR=str(root / "Diffusion-Planner"),
        NUPLAN_DEVKIT_ROOT=str(root / "nuplan-devkit"),
        NUPLAN_DATA_ROOT=str(root / "dataset/nuplan"),
        NUPLAN_MAPS_ROOT=str(root / "dataset/nuplan/maps"),
        NUPLAN_EXP_ROOT=str(root / "dataset/nuplan/exp"),
        NUPLAN_SIMULATION_ALLOW_ANY_BUILDER="1",
        CUDA_VISIBLE_DEVICES="",
        NODE_RANK="0",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        PYTHONUNBUFFERED="1",
        HYDRA_FULL_ERROR="1",
    )
    source_paths = [suite / "code/lg", root / "Diffusion-Planner", root / "nuplan-devkit"]
    env["PYTHONPATH"] = os.pathsep.join(map(str, source_paths)) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env, source_paths


def cpu_torch_bytes(blob):
    import torch

    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    return torch.load(io.BytesIO(blob), **options)


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return cpu_torch_bytes
        return super().find_class(module, name)


def load_log_cpu(file_path):
    """Official log decoding with CPU storage mapping for trusted saved models."""
    path = Path(file_path)
    with lzma.open(path, "rb") as source:
        if path.name.endswith(".msgpack.xz"):
            import msgpack

            log = CPUUnpickler(io.BytesIO(msgpack.unpackb(source.read()))).load()
        elif path.name.endswith(".pkl.xz"):
            log = CPUUnpickler(source).load()
        else:
            raise ValueError(f"Unsupported nuPlan simulation log: {path}")
    # nuBoard uses only scenario/history. The serialized planner is unnecessary.
    log.planner = None
    gc.collect()
    return log


def build_board(suite, records, port, frame_rate):
    plan = read_json(suite / "plan.json")
    env, source_paths = runtime_environment(plan["root"], suite)
    for key in set(os.environ) - set(env):
        os.environ.pop(key, None)
    os.environ.update(env)
    sys.path[:0] = list(map(str, source_paths))
    import torch
    from hydra.utils import instantiate
    from nuplan.planning.nuboard.nuboard import NuBoard
    from nuplan.planning.simulation.simulation_log import SimulationLog
    from omegaconf import OmegaConf

    torch.set_num_threads(2)
    # Process-local adapter only. Original logs and nuPlan source stay untouched.
    SimulationLog.load_data = staticmethod(load_log_cpu)
    cfg = OmegaConf.load(Path(records[0]["result_dir"]) / "code/hydra/config.yaml")
    # Reuse the saved test builder instead of the notebook's trainval default.
    builder = instantiate(cfg.scenario_builder)
    vehicle = instantiate(cfg.scenario_builder.vehicle_parameters)
    return NuBoard(
        nuboard_paths=[row["nuboard_file"] for row in records],
        scenario_builder=builder,
        vehicle_parameters=vehicle,
        port_number=port,
        async_scenario_rendering=True,
        scenario_rendering_frame_rate_cap_hz=frame_rate,
    )


def serve(board, address, port, extra_origins=()):
    """Serve the official UI without opening a browser on the remote server."""
    import nuplan.planning.nuboard.nuboard as official
    from bokeh.application import Application
    from bokeh.application.handlers import FunctionHandler
    from bokeh.server.server import Server
    from tornado.ioloop import IOLoop
    from tornado.web import StaticFileHandler

    resource = Path(official.__file__).parent / "resource"
    origins = list(dict.fromkeys([f"localhost:{port}", f"127.0.0.1:{port}", *extra_origins]))
    loop = IOLoop.current()
    server = Server(
        {"/": Application(FunctionHandler(board.main_page))},
        io_loop=loop,
        address=address,
        port=port,
        allow_websocket_origin=origins,
        extra_patterns=[(r"/resource/(.*)", StaticFileHandler, {"path": str(resource)})],
    )

    def stop(signum, frame):
        loop.add_callback(loop.stop)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    server.start()
    print(f"nuBoard ready: http://localhost:{port}/ (listen={address}; CPU only)", flush=True)
    print(f"VS Code: forward remote port {port}, then open the forwarded address.", flush=True)
    print(
        "Scenario tab: paste one token into 'Scenario token', then click 'Query Scenario'.",
        flush=True,
    )
    try:
        loop.start()
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument(
        "--list", action="store_true", help="Print the four runs without starting a server"
    )
    parser.add_argument("--port", type=int, default=6599)
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument(
        "--frame-rate", type=int, default=10, help="Browser rendering cap in Hz (1–60)"
    )
    parser.add_argument(
        "--allow-websocket-origin",
        action="append",
        default=[],
        help="Extra browser host:port for a proxy/forwarded address",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.frame_rate <= 60:
        parser.error("port must be 1–65535 and frame-rate must be 1–60")
    suite = args.suite.expanduser().resolve()
    records = discover_runs(suite)
    for row in records:
        print(
            f"{row['job_id']} [{row['state']}]\n  result: {row['result_dir']}\n  nuboard: {row['nuboard_file']}",
            flush=True,
        )
    if args.list:
        return
    board = build_board(suite, records, args.port, args.frame_rate)
    serve(board, args.address, args.port, args.allow_websocket_origin)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
