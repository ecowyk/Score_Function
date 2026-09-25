"""One global official nuPlan selection, frozen before expensive feature extraction."""

import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from score_function.utils.train_utils import atomic_write, file_hash, read_json, resolve_path


def log_name(value):
    """nuPlan names contain dots: strip only a literal .db suffix."""
    return Path(value).name.removesuffix(".db")


def official_scenarios(config, plan):
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
        NuPlanScenarioBuilder,
    )
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor

    settings = config["data"]
    builder = NuPlanScenarioBuilder(
        str(resolve_path(config, "database_dir")),
        str(resolve_path(config, "maps_dir")),
        None,
        [row["db"] for row in plan],
        settings["map_version"],
        max_workers=settings["preprocess_workers"],
        verbose=True,
    )
    filters = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=None,
        log_names=[row["log"] for row in plan],
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=settings["total_scenarios"],
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=True,
        remove_invalid_goals=settings["remove_invalid_goals"],
        shuffle=True,
    )
    worker = SingleMachineParallelExecutor(
        use_process_pool=True,
        max_workers=settings["preprocess_workers"],
    )
    rng = random.getstate()
    random.seed(settings["selection_seed"])
    try:
        # Apply the official global filter ONCE across all allowed databases.
        return builder.get_scenarios(filters, worker)
    finally:
        random.setstate(rng)
        # The pinned official worker has no public close/context-manager API.
        worker._executor.shutdown(wait=True)


def freeze_selection(config, plan, identity, loader=None):
    root = resolve_path(config, "data_output") / "selection"
    index = root / "index.json"
    if index.exists():
        state = read_json(index)
        if state["identity"] != identity:
            raise ValueError("Frozen scenario selection changed; choose a new data_output")
        for row in state["logs"]:
            if file_hash(root / row["path"]) != row["sha256"]:
                raise ValueError("Frozen scenario list was modified")
        return state
    atomic_write(
        root / "status.json",
        {
            "state": "running",
            "stage": "global_scenario_selection",
            "candidate_logs": len(plan),
            "global_cap": config["data"]["total_scenarios"],
        },
    )
    print(
        {
            "stage": "global_scenario_selection",
            "candidate_logs": len(plan),
            "global_cap": config["data"]["total_scenarios"],
        },
        flush=True,
    )
    scenarios = (loader or official_scenarios)(config, plan)
    if not scenarios or len(scenarios) > config["data"]["total_scenarios"]:
        raise ValueError("Official selector returned an empty set or exceeded the global cap")
    by_log = {row["log"]: row for row in plan}
    grouped, seen = defaultdict(list), set()
    for scenario in scenarios:
        name = log_name(scenario.log_name)
        if name not in by_log:
            raise ValueError("Selected a log outside the official training allowlist")
        if scenario.token in seen:
            raise ValueError(f"Duplicate globally selected token: {scenario.token}")
        seen.add(scenario.token)
        info = scenario._scenario_extraction_info
        grouped[name].append(
            {
                "token": scenario.token,
                # Already loaded by the builder; avoid one extra SQL query per sample.
                "start_time_us": int(scenario._initial_lidar_timestamp),
                "map": scenario._map_name,
                "scenario_type": scenario.scenario_type,
                "extraction": asdict(info) if info is not None else None,
            }
        )
    count = len(scenarios)
    del scenarios, seen
    logs = []
    for name in sorted(grouped):
        path = root / f"{name}.json"
        atomic_write(path, grouped[name])
        logs.append(
            {
                **by_log[name],
                "path": path.name,
                "count": len(grouped[name]),
                "sha256": file_hash(path),
            }
        )
    state = {
        "identity": identity,
        "global_cap": config["data"]["total_scenarios"],
        "selected_scenarios": count,
        "selection_seed": config["data"]["selection_seed"],
        "logs": logs,
    }
    # Completion index is published last; partial selection files are never consumed.
    atomic_write(index, state)
    atomic_write(root / "status.json", {"state": "complete", "selected_scenarios": count})
    print(
        {
            "stage": "global_scenario_selection",
            "state": "complete",
            "selected_scenarios": count,
            "selected_logs": len(logs),
        },
        flush=True,
    )
    return state


def restore_scenario(config, row, descriptor):
    """Reconstruct only a selected scenario using the official constructor."""
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
        ScenarioExtractionInfo,
    )

    info = descriptor["extraction"]
    return NuPlanScenario(
        data_root=str(resolve_path(config, "database_dir")),
        log_file_load_path=row["db"],
        initial_lidar_token=descriptor["token"],
        initial_lidar_timestamp=descriptor["start_time_us"],
        scenario_type=descriptor["scenario_type"],
        map_root=str(resolve_path(config, "maps_dir")),
        map_version=config["data"]["map_version"],
        map_name=descriptor["map"],
        scenario_extraction_info=ScenarioExtractionInfo(**info) if info is not None else None,
        ego_vehicle_parameters=get_pacifica_parameters(),
        sensor_root=None,
    )
