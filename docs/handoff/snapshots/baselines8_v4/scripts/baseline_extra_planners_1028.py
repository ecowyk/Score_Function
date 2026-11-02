#!/usr/bin/env python3
"""DP and published Flow Planner configuration for the eight-baseline extension."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess

EXTRA_VARIANTS = ("diffusion_planner", "flow_planner")
DP_TARGET = "score_function.planner.planner.ScoreFunctionPlanner"
FLOW_TARGET = "flow_score_cfg18.planner.FlowScorePlanner"
FLOW_CFG_WEIGHT = 1.8
FLOW_ADAPTER_PACKAGE = "flow_score_cfg18"
FLOW_ADAPTER_COMMIT = "73f4701"
FLOW_COMMIT = "366df7654d7b7fa77f9d2577279da63127c6d2f8"
FLOW_SEED_POLICY = "once_per_scenario_before_native_initialization"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _file(path, description):
    path = Path(path).expanduser().resolve()
    require(path.is_file(), f"Missing {description}: {path}")
    return path


def _one_planner(planner, target):
    require(isinstance(planner, dict) and len(planner) == 1,
            "Require exactly one planner configuration")
    value = next(iter(planner.values()))
    require(value.get("_target_") == target, f"Require planner target {target}")
    return value


def _config_path(config, key):
    path = Path(config["paths"][key]).expanduser()
    return path.resolve() if path.is_absolute() else (Path(config["paths"]["root"]) / path).resolve()


def resolve_extra_sources(root, prior, flow_adapter=None, flow_code=None,
                          flow_config=None, flow_checkpoint=None):
    """Resolve existing assets only; no downloads, environment installs or training."""
    import evaluate_native_score_queue as shared

    root = Path(root).resolve()
    entry = _file(root / "nuplan-devkit/nuplan/planning/script/run_simulation.py", "nuPlan entry")
    source_config = _file(Path(prior["reference_dp"]) / "code/hydra/config.yaml", "completed DP config")
    base = shared.load_saved_config(Path(prior["reference_dp"]))
    template = copy.deepcopy(base["planner"])
    dp_planner = _one_planner(template, DP_TARGET)
    require(dp_planner.get("steps") == 0, "DP template must be a completed K0 configuration")
    require(dp_planner["future_trajectory_sampling"].get("num_poses") == 80
            and dp_planner["future_trajectory_sampling"].get("time_horizon") == 8,
            "DP requires its original 80-pose / 8-second sampling")
    score = json.loads(Path(prior["score_config"]).read_text())
    checkpoint = _file(_config_path(score, "planner_checkpoint"), "original DP checkpoint")
    planner_args = _file(_config_path(score, "planner_args"), "original DP model configuration")
    dp_python = root / "envs/diffusion_planner/bin/python"
    _file(dp_python, "DP Python")
    dp = dict(checkpoint=str(checkpoint), source_config=str(source_config), planner_args=str(planner_args),
              planner_template=template,
              runtime=dict(python=str(dp_python),
                           cwd=str(root / "Diffusion-Planner"), entry=str(entry)))

    adapter = Path(flow_adapter or Path(__file__).resolve().parent.parent).expanduser().resolve()
    code = Path(flow_code or root / "Flow-Planner").expanduser().resolve()
    adapter_file = _file(adapter / FLOW_ADAPTER_PACKAGE / "planner.py", "bundled Flow CFG1.8 adapter")
    config = _file(flow_config or code / "checkpoints/model_config.yaml", "published Flow config")
    weights = _file(flow_checkpoint or code / "checkpoints/model.pth", "published Flow checkpoint")
    commit = subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()
    require(commit == FLOW_COMMIT, f"Flow source must use the existing pinned revision {FLOW_COMMIT}; found {commit}")
    dirty = subprocess.check_output(["git", "-C", str(code), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    require(not dirty, "Published Flow Planner tracked source has local changes")
    flow_python = root / "envs/flow_planner/bin/python"
    _file(flow_python, "Flow Python")
    flow = dict(checkpoint=str(weights), config_path=str(config), adapter=str(adapter),
                adapter_file=str(adapter_file), code=str(code), source_commit=commit,
                adapter_package=FLOW_ADAPTER_PACKAGE, adapter_source_commit=FLOW_ADAPTER_COMMIT,
                cfg_weight=FLOW_CFG_WEIGHT,
                seed_policy=FLOW_SEED_POLICY,
                runtime=dict(python=str(flow_python),
                             cwd=str(code), entry=str(entry)))
    return {"diffusion_planner": dp, "flow_planner": flow}


def configure_extra(name, source, score_cfg, score_ckpt, seed, k, out):
    require(name in EXTRA_VARIANTS and k in (0, 20), "Unexpected extra-planner variant or K")
    require(type(seed) is int and 0 <= seed < 2 ** 32, "Invalid inference seed")
    if name == "diffusion_planner":
        result = copy.deepcopy(source["planner_template"])
        planner = _one_planner(result, DP_TARGET)
        planner.update(score_config=str(score_cfg), score_checkpoint=str(score_ckpt),
                       steps=k, gamma=.1, device="cuda", trace_dir=None)
        # Keep the same projection and native trajectory sampler as the proven DP run.
        return result
    return {"flow_score": {
        "_target_": FLOW_TARGET, "_recursive_": False, "_convert_": "all",
        "config_path": source["config_path"], "checkpoint_path": source["checkpoint"],
        "score_config": str(score_cfg) if k else "unused",
        "score_checkpoint": str(score_ckpt) if k else "unused",
        "steps": k, "gamma": .1, "sigma": .1, "heading_projection": True,
        "evaluation_seed": seed, "device": "cuda", "native_device": "cuda",
        "cfg_weight": FLOW_CFG_WEIGHT,
        "audit_dir": str(Path(out) / "refinement_audit") if k else None,
    }}


def extra_environment(plan, name, defaultenv):
    require(name in EXTRA_VARIANTS, "Unknown extra planner environment")
    env = dict(defaultenv)
    root = Path(plan["root"])
    paths = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    if name == "flow_planner":
        source = plan["sources"][name]
        # The frozen S10-G package remains first. Flow's adapter and official
        # module have distinct names and do not replace the existing wrappers.
        paths.extend([source["adapter"], source["code"]])
    else:
        paths.append(str(root / "Diffusion-Planner"))
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    return env


def validate_extra(name, directory, cfg, cohort):
    import evaluate_native_score_queue as shared

    require(name in EXTRA_VARIANTS, "Unknown extra planner validation")
    result = shared.validate_reuse(Path(directory), cfg, cohort)
    if name == "flow_planner":
        planner = _one_planner(cfg["planner"], FLOW_TARGET)
        require(planner.get("evaluation_seed") == cfg["seed"], "Flow inference/global seeds differ")
        require(planner.get("cfg_weight") == FLOW_CFG_WEIGHT, "Flow must use CFG1.8 for K0 and K20")
        if planner["steps"] == 20:
            shared.check_audits(Path(directory), len(cohort))
            for path in (Path(directory) / "refinement_audit").glob("*.json"):
                audit = json.loads(path.read_text())
                require(audit.get("native_pose_count") == 80 and audit.get("score_pose_count") == 80,
                        f"Flow must refine native80 to score80: {path}")
                require(audit.get("evaluation_seed") == planner["evaluation_seed"]
                        and audit.get("seed_policy") == FLOW_SEED_POLICY,
                        f"Flow seed policy changed: {path}")
                require(audit.get("cfg_weight") == FLOW_CFG_WEIGHT
                        and audit.get("effective_cfg_weight") == FLOW_CFG_WEIGHT,
                        f"Flow sampler did not use CFG1.8: {path}")
    else:
        # DP's established adapter validates its checkpoint/normalizer at load
        # time. Its K0 bypasses refinement; it does not emit native-wrapper audits.
        planner = _one_planner(result[0]["planner"], DP_TARGET)
        expected = _one_planner(cfg["planner"], DP_TARGET)
        score = json.loads(Path(planner["score_config"]).read_text())
        require(score["reference_seed"] == cfg["seed"], "DP inference/global seeds differ")
        require(score["training"]["sigma"] == .1
                and score["model"].get("parameterization", "score") == "score",
                "DP must use direct-score S10-G sigma .1")
        require(planner["steps"] == expected["steps"] and planner["steps"] in (0, 20)
                and planner["gamma"] == .1 and planner["device"] == "cuda"
                and planner.get("trace_dir") is None,
                "DP evaluation settings changed")
    return result


def input_files(source):
    result = [Path(source["checkpoint"])]
    for key in ("source_config", "planner_args", "config_path"):
        if source.get(key):
            result.append(Path(source[key]))
    if source.get("adapter"):
        result.extend((Path(source["adapter"]) / source.get("adapter_package", FLOW_ADAPTER_PACKAGE)).rglob("*.py"))
    if source.get("code"):
        result.extend((Path(source["code"]) / "flow_planner").rglob("*.py"))
    return sorted(set(result))
