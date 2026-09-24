"""Resumable per-log preprocessing restricted to the official training allowlist."""

import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from score_function.utils.train_utils import (
    atomic_write,
    file_hash,
    fingerprint,
    planner_source_hashes,
    read_json,
    resolve_path,
)


def recording_name(path):
    return Path(path).stem.rsplit("_", 2)[0]


def build_split_plan(files, seed, validation_fraction, test_fraction):
    if not 0 < validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("Invalid split fractions")
    groups = sorted(
        {recording_name(p) for p in files},
        key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(),
    )
    nval = max(1, round(len(groups) * validation_fraction))
    ntest = max(1, round(len(groups) * test_fraction)) if test_fraction else 0
    if nval + ntest >= len(groups):
        raise ValueError("Not enough independent recordings for the requested splits")
    splits = {
        name: "val" if i < nval else "test" if i < nval + ntest else "train"
        for i, name in enumerate(groups)
    }
    return [
        {
            "db": str(p),
            "log": p.stem,
            "recording": recording_name(p),
            "split": splits[recording_name(p)],
        }
        for p in sorted(files)
    ]


def allowed_databases(config):
    allowed = read_json(resolve_path(config, "train_log_allowlist"))
    if not isinstance(allowed, list) or not all(isinstance(x, str) for x in allowed):
        raise ValueError("Expected the official nuplan_train.json list of log names")
    allowed = {Path(x).stem for x in allowed}
    files = sorted(resolve_path(config, "database_dir").rglob("*.db"))
    selected = [p for p in files if p.stem in allowed]
    if not selected or len({p.stem for p in selected}) != len(selected):
        raise ValueError("No official training DBs found, or duplicate extracted DB names")
    return selected, len(files) - len(selected)


class InvalidSample(ValueError):
    """Expected rejection of an incomplete/nonfinite expert target."""


def _prepare_log(task):
    config, row, identity = task
    from score_function.utils.config import configure_runtime

    configure_runtime(config, require_cuda=False)
    from diffusion_planner.data_process.data_processor import DataProcessor
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
        NuPlanScenarioBuilder,
    )
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    output = resolve_path(config, "data_output")
    manifest = output / "manifests" / f"{row['log']}.json"
    state = (
        read_json(manifest)
        if manifest.exists()
        else {"identity": identity, "complete": False, "records": [], "rejected": [], "errors": []}
    )
    if state["identity"] != identity:
        raise ValueError("Preprocessing identity changed; choose a new data_output")
    for item in state["records"]:
        if (
            not Path(item["feature"]).is_file()
            or file_hash(item["feature"]) != item["feature_sha256"]
        ):
            raise ValueError(f"Corrupt feature: {item['feature']}")
    if state["complete"]:
        return str(manifest)
    done = {item["token"] for item in state["records"] + state["rejected"]}
    state["errors"] = []
    features = output / "features" / row["log"]
    features.mkdir(parents=True, exist_ok=True)

    class CheckedProcessor(DataProcessor):
        def save_to_disk(self, directory, data):
            if data["ego_agent_future"].shape != (80, 3):
                raise InvalidSample("Incomplete future trajectory")
            for name, value in data.items():
                if (
                    isinstance(value, np.ndarray)
                    and value.dtype.kind in "fc"
                    and not np.isfinite(value).all()
                ):
                    raise InvalidSample(f"Nonfinite feature: {name}")
            super().save_to_disk(directory, data)

    processor = CheckedProcessor(
        SimpleNamespace(
            save_path=str(features),
            agent_num=32,
            static_objects_num=5,
            lane_num=70,
            lane_len=20,
            route_num=25,
            route_len=20,
        )
    )
    builder = NuPlanScenarioBuilder(
        str(resolve_path(config, "database_dir")),
        str(resolve_path(config, "maps_dir")),
        None,
        [row["db"]],
        config["data"]["map_version"],
        max_workers=1,
        verbose=False,
    )
    settings = config["data"]
    filter_ = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=settings["max_scenarios_per_db"],
        timestamp_threshold_s=settings["timestamp_spacing_s"],
        ego_displacement_minimum_m=None,
        expand_scenarios=settings["expand_scenarios"],
        remove_invalid_goals=settings["remove_invalid_goals"],
        shuffle=False,
    )
    for number, scenario in enumerate(builder.get_scenarios(filter_, Sequential())):
        if scenario.token in done:
            continue
        try:
            processor.work([scenario])
            path = features / f"{scenario._map_name}_{scenario.token}.npz"
            state["records"].append(
                {
                    **row,
                    "token": scenario.token,
                    "map": scenario._map_name,
                    "scenario_type": scenario.scenario_type,
                    "feature": str(path),
                    "feature_sha256": file_hash(path),
                    "start_time_us": scenario.start_time.time_us,
                }
            )
            done.add(scenario.token)
        except InvalidSample as exc:
            state["rejected"].append({"token": scenario.token, "reason": str(exc)})
            done.add(scenario.token)
        except Exception as exc:
            state["errors"].append({"token": scenario.token, "error": repr(exc)})
        if number % 32 == 0:
            atomic_write(manifest, state)
    state["complete"] = not state["errors"]
    atomic_write(manifest, state)
    return str(manifest)


def prepare(config):
    output = resolve_path(config, "data_output")
    output.mkdir(parents=True, exist_ok=True)
    files, excluded = allowed_databases(config)
    settings = config["data"]
    plan = build_split_plan(
        files, settings["split_seed"], settings["validation_fraction"], settings["test_fraction"]
    )
    scientific_settings = {
        key: value for key, value in settings.items() if key != "preprocess_workers"
    }
    identity = fingerprint(
        {
            "settings": scientific_settings,
            "logs": plan,
            "allowlist": file_hash(resolve_path(config, "train_log_allowlist")),
            "preprocessor_source": file_hash(__file__),
            "official_sources": planner_source_hashes(config),
        }
    )
    plan_path = output / "split_plan.json"
    if plan_path.exists() and read_json(plan_path)["fingerprint"] != identity:
        raise ValueError("Existing split/data selection differs; choose a new data_output")
    atomic_write(plan_path, {"fingerprint": identity, "logs": plan, "excluded_db_count": excluded})
    tasks = [(config, row, identity) for row in plan]
    workers = settings["preprocess_workers"]
    executor = (
        ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        if workers > 1
        else None
    )
    try:
        results = executor.map(_prepare_log, tasks) if executor else map(_prepare_log, tasks)
        manifests = []
        for path in results:
            manifests.append(path)
            status = {"state": "running", "logs_done": len(manifests), "logs_total": len(plan)}
            atomic_write(output / "preprocess_status.json", status)
            print(status, flush=True)
        records, errors, rejected = [], [], 0
        for path in manifests:
            state = read_json(path)
            records.extend(state["records"])
            errors.extend(state["errors"])
            rejected += len(state["rejected"])
        atomic_write(output / "preprocess_errors.json", errors)
        if errors or not records:
            raise RuntimeError("Preprocessing failed/empty; inspect per-log manifests and retry")
        # One final merge of metadata, never one whole-dataset rewrite per frame.
        atomic_write(resolve_path(config, "manifest"), records)
        atomic_write(
            output / "preprocess_status.json",
            {
                "state": "complete",
                "samples": len(records),
                "rejected": rejected,
                "excluded_db_count": excluded,
                "manifest_sha256": file_hash(resolve_path(config, "manifest")),
            },
        )
    except Exception as exc:
        atomic_write(output / "preprocess_status.json", {"state": "failed", "error": repr(exc)})
        raise
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
