#!/usr/bin/env python3
"""Official nuPlan configuration and CPU-only cohort selection for baseline extensions."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
import re
import sys


SPLITS = {"val14": "nuplan", "test14-hard": "nuplan_challenge", "test14-random": "nuplan_challenge"}
CHALLENGES = ("closed_loop_nonreactive_agents", "closed_loop_reactive_agents")


def order_metric_callbacks(cfg):
    """Restore file -> aggregate -> summary order after sorted JSON serialization."""
    result = copy.deepcopy(cfg)
    items = list(result.get("main_callback", {}).items())
    ranks = {"MetricFileCallback": 0, "MetricAggregatorCallback": 1, "MetricSummaryCallback": 2,
             "metric_file_callback": 0, "metric_aggregator_callback": 1, "metric_summary_callback": 2}
    slots, selected = [], []
    for index, (name, callback) in enumerate(items):
        target = str(callback.get("_target_", "")).rsplit(".", 1)[-1] if isinstance(callback, dict) else ""
        rank = ranks.get(target, ranks.get(name))
        if rank is not None:
            slots.append(index)
            selected.append((rank, (name, callback)))
    for index, (_, item) in zip(slots, sorted(selected, key=lambda row: row[0])):
        items[index] = item
    if "main_callback" in result:
        result["main_callback"] = dict(items)
    return result


def compose_template(root, split, challenge, output, seed):
    """Compose an official split/challenge, resolving all application interpolations.

    Requires the existing nuPlan Python environment (Hydra/OmegaConf), the devkit,
    and Diffusion-Planner's scenario-filter YAMLs. Does not instantiate a planner
    or create a CUDA/Ray context. Caller replaces the simple-planner placeholder.
    """
    if split not in SPLITS or challenge not in CHALLENGES:
        raise ValueError(f"Unsupported split/challenge: {split}/{challenge}")
    root, output = Path(root).resolve(), Path(output).resolve()
    devkit, dp = root / "nuplan-devkit", root / "Diffusion-Planner"
    config_dir = devkit / "nuplan/planning/script/config/simulation"
    dp_config = dp / "diffusion_planner/config"
    filter_path = dp_config / "scenario_filter" / (split + ".yaml")
    if not filter_path.is_file():
        raise FileNotFoundError(f"Official scenario filter is missing: {filter_path}")
    for path in (devkit, dp):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.update(
        NUPLAN_DATA_ROOT=str(root / "dataset/nuplan"),
        NUPLAN_MAPS_ROOT=str(root / "dataset/nuplan/maps"),
        NUPLAN_EXP_ROOT=str(root / "dataset/nuplan/exp"),
        NUPLAN_DEVKIT_ROOT=str(devkit), DP_DIR=str(dp),
    )
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from hydra.core.utils import setup_globals
    from omegaconf import OmegaConf, open_dict

    setup_globals()  # Install ${now:...} and other Hydra resolvers outside @hydra.main.
    searchpath = ["file://" + str(dp_config / "scenario_filter"), "file://" + str(dp_config),
                  "pkg://nuplan.planning.script.config.common", "pkg://nuplan.planning.script.experiments"]
    overrides = [
        "+simulation=" + challenge, "scenario_builder=" + SPLITS[split], "scenario_filter=" + split,
        "planner=simple_planner", "seed=" + str(int(seed)), "verbose=true",
        "output_dir=" + json.dumps(str(output)), "experiment_uid=" + json.dumps(split + "/baseline_extension_1028"),
        "enable_simulation_progress_bar=true", "exit_on_failure=true", "run_metric=true",
        "hydra.searchpath=" + json.dumps(searchpath),
    ]
    with initialize_config_dir(config_dir=str(config_dir), job_name="baseline_extension_1028"):
        cfg = compose(config_name="default_simulation", overrides=overrides, return_hydra_config=True)
        with open_dict(cfg):
            cfg.hydra.runtime.output_dir = str(output)
        HydraConfig.instance().set_config(cfg)
        choices = dict(cfg.hydra.runtime.choices)
        result = OmegaConf.to_container(
            OmegaConf.masked_copy(cfg, [key for key in cfg if key != "hydra"]),
            resolve=True, throw_on_missing=True,
        )

    mapping = "nuplan_scenario_mapping" if split == "val14" else "nuplan_challenge_scenario_mapping"
    if choices.get("scenario_builder") != SPLITS[split] or choices.get("scenario_builder/scenario_mapping") != mapping:
        raise ValueError("Official scenario-builder/mapping selection differs from the requested split")
    data_root = root / "dataset/nuplan/nuplan-v1.1" / ("trainval" if split == "val14" else "test")
    if Path(result["scenario_builder"]["data_root"]).resolve() != data_root.resolve():
        raise ValueError(f"Unexpected database root for {split}: {result['scenario_builder']['data_root']}")
    if result["scenario_builder"].get("db_files") is not None:
        raise ValueError("Official template unexpectedly overrides data_root with db_files")
    observation = "IDMAgents" if challenge == "closed_loop_reactive_agents" else "TracksObservation"
    if not result["observation"]["_target_"].endswith("." + observation):
        raise ValueError("Official challenge observation differs from the requested NR/R mode")
    if {value.get("challenge_name") for value in result["metric_aggregator"].values()} != {challenge}:
        raise ValueError("Official metric aggregator differs from the requested challenge")
    return order_metric_callbacks(result)


def select_cohort(cfg, seed):
    """Return exact sorted (log_name, token) pairs using official filtering, without GPUs."""
    import numpy as np
    from omegaconf import OmegaConf
    from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
    from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
    from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor

    config = OmegaConf.create(copy.deepcopy(cfg))
    config.scenario_builder.max_workers = 16
    python_state, numpy_state = random.getstate(), np.random.get_state()
    worker = None
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2 ** 32))
        builder = build_scenario_builder(config)
        scenario_filter = build_scenario_filter(config.scenario_filter)
        worker = SingleMachineParallelExecutor(use_process_pool=False, max_workers=16)
        scenarios = builder.get_scenarios(scenario_filter, worker)
        cohort = sorted((scenario.log_name, scenario.token) for scenario in scenarios)
        if not cohort or len(cohort) != len(set(cohort)):
            raise ValueError("Official filter returned an empty cohort or duplicate scenario identities")
        if any(not isinstance(log, str) or not log or not re.fullmatch(r"[0-9a-f]{16}", token)
               for log, token in cohort):
            raise ValueError("Official filter returned an invalid log/token identity")
        return cohort
    finally:
        if worker is not None:
            # This devkit executor has no public shutdown method.
            worker._executor.shutdown(wait=True)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
