"""Global official scenario selection followed by resumable parallel feature extraction."""

import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from score_function.data_process.scenario_selection import (
    freeze_selection,
    log_name,
    restore_scenario,
)
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
    allowed = {log_name(x) for x in allowed}
    files = sorted(resolve_path(config, "database_dir").rglob("*.db"))
    selected = [p for p in files if p.stem in allowed]
    if not selected or len({p.stem for p in selected}) != len(selected):
        raise ValueError("No official training DBs found, or duplicate extracted DB names")
    return selected, len(files) - len(selected)


class InvalidSample(ValueError):
    """Expected rejection of an incomplete/nonfinite expert target."""


def reusable_records(config, row, selected):
    """Reuse only selected, checksum-verified samples; never preferentially select old work."""
    if not config["paths"].get("reuse_features_from"):
        return {}
    source = resolve_path(config, "reuse_features_from")
    manifest = source / "manifests" / f"{row['log']}.json"
    if not manifest.is_file():
        return {}
    wanted = {item["token"]: item for item in selected}
    records = {}
    for item in read_json(manifest)["records"]:
        descriptor = wanted.get(item["token"])
        if descriptor is None:
            continue
        if (
            item["log"] != row["log"]
            or Path(item["db"]).resolve() != Path(row["db"]).resolve()
            or item["start_time_us"] != descriptor["start_time_us"]
            or item["map"] != descriptor["map"]
        ):
            raise ValueError("Old NPZ metadata does not match the selected DB/frame")
        path = Path(item["feature"])
        if not path.is_file() or file_hash(path) != item["feature_sha256"]:
            continue  # Recompute in the new output, leaving the old file untouched.
        with np.load(path, allow_pickle=False) as data:
            if data["ego_agent_future"].shape != (80, 3):
                continue
        records[item["token"]] = {
            **item,
            **{k: row[k] for k in ("db", "log", "recording", "split")},
        }
    return records


def _prepare_log(task):
    config, row, identity = task
    from score_function.utils.config import configure_runtime

    configure_runtime(config, require_cuda=False)
    from diffusion_planner.data_process.data_processor import DataProcessor

    output = resolve_path(config, "data_output")
    selection_path = output / "selection" / row["path"]
    if file_hash(selection_path) != row["sha256"]:
        raise ValueError("Selected scenario list changed")
    selected = read_json(selection_path)
    manifest = output / "manifests" / f"{row['log']}.json"
    state = (
        read_json(manifest)
        if manifest.exists()
        else {
            "identity": identity,
            "complete": False,
            "records": [],
            "rejected": [],
            "errors": [],
            "reused": 0,
            "selected": len(selected),
        }
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
    old = reusable_records(config, row, [item for item in selected if item["token"] not in done])
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
    for number, descriptor in enumerate(selected):
        token = descriptor["token"]
        if token in done:
            continue
        try:
            if token in old:
                state["records"].append(old[token])
                state["reused"] += 1
            else:
                scenario = restore_scenario(config, row, descriptor)
                processor.work([scenario])
                path = features / f"{descriptor['map']}_{token}.npz"
                state["records"].append(
                    {
                        **{k: row[k] for k in ("db", "log", "recording", "split")},
                        **{
                            k: descriptor[k]
                            for k in ("token", "map", "scenario_type", "start_time_us")
                        },
                        "feature": str(path),
                        "feature_sha256": file_hash(path),
                    }
                )
            done.add(token)
        except InvalidSample as exc:
            state["rejected"].append({"token": token, "reason": str(exc)})
            done.add(token)
        except Exception as exc:
            state["errors"].append({"token": token, "error": repr(exc)})
        if number % 32 == 0:
            atomic_write(manifest, state)
    state["complete"] = not state["errors"]
    atomic_write(manifest, state)
    return str(manifest)


def prepare(config):
    output = resolve_path(config, "data_output")
    output.mkdir(parents=True, exist_ok=True)
    if (
        config["paths"].get("reuse_features_from")
        and resolve_path(config, "reuse_features_from").resolve() == output.resolve()
    ):
        raise ValueError("Reuse source must differ from the new data_output directory")
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
            "selection_source": file_hash(Path(__file__).with_name("scenario_selection.py")),
            "reuse_features_from": config["paths"].get("reuse_features_from"),
            "official_sources": planner_source_hashes(config),
        }
    )
    plan_path = output / "split_plan.json"
    if plan_path.exists() and read_json(plan_path)["fingerprint"] != identity:
        raise ValueError("Existing split/data selection differs; choose a new data_output")
    atomic_write(plan_path, {"fingerprint": identity, "logs": plan, "excluded_db_count": excluded})
    try:
        selection = freeze_selection(config, plan, identity)
    except Exception as exc:
        atomic_write(
            output / "preprocess_status.json",
            {
                "state": "failed",
                "stage": "global_scenario_selection",
                "error": repr(exc),
            },
        )
        raise
    tasks = [(config, row, identity) for row in selection["logs"]]
    workers = settings["preprocess_workers"]
    executor = (
        ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        if workers > 1
        else None
    )
    try:
        results = (
            (
                future.result()
                for future in as_completed([executor.submit(_prepare_log, task) for task in tasks])
            )
            if executor
            else map(_prepare_log, tasks)
        )
        manifests = []
        for path in results:
            manifests.append(path)
            status = {
                "state": "running",
                "stage": "feature_extraction",
                "logs_done": len(manifests),
                "logs_total": len(tasks),
                "selected_scenarios": selection["selected_scenarios"],
            }
            atomic_write(output / "preprocess_status.json", status)
            print(status, flush=True)
        records, errors, rejected, reused = [], [], 0, 0
        for path in sorted(manifests):
            state = read_json(path)
            records.extend(state["records"])
            errors.extend(state["errors"])
            rejected += len(state["rejected"])
            reused += state["reused"]
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
                "selected_scenarios": selection["selected_scenarios"],
                "global_cap": settings["total_scenarios"],
                "reused_samples": reused,
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
