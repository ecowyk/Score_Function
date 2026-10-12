"""Bind a saved frame to exact, official per-scenario closed-loop scores.

These scores describe separate full closed-loop runs of the same scenario.
They do not score the fixed-input counterfactual trajectories drawn in a frame.
No simulation, model inference, or mutation of frozen suite files occurs here.
"""

import hashlib
import json
import math
from numbers import Integral
from pathlib import Path

DEFAULT_SUITE = Path("/home/data/wyk/dataset/nuplan/exp/score_full_k_1009_v1")
METRICS = (
    "no_ego_at_fault_collisions",
    "time_to_collision_within_bound",
    "ego_is_comfortable",
    "ego_is_making_progress",
    "ego_progress_along_expert_route",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "speed_limit_compliance",
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _unit_score(value, description):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numerical score for {description}: {value!r}") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"Expected a finite [0, 1] score for {description}, got {value!r}")
    return number


class ScenarioScores:
    """Read each completed job once; pair scenario rows by token/type/log identity."""

    def __init__(self, suite_path=None):
        self.suite = Path(suite_path or DEFAULT_SUITE).expanduser().resolve()
        self.plan_path = self.suite / "plan.json"
        self.plan = _read(self.plan_path)
        self.config_path = self.suite / "models/S05-G/config.json"
        self.checkpoint_path = self.suite / "models/S05-G/final_ema.pt"
        self.config = _read(self.config_path)
        self.provenance = {
            "plan_sha256": _digest(self.plan_path),
            "config_sha256": _digest(self.config_path),
            "checkpoint_sha256": _digest(self.checkpoint_path),
        }
        for path, key in (
            (self.config_path, "config_sha256"),
            (self.checkpoint_path, "checkpoint_sha256"),
        ):
            expected = self.plan.get("files", {}).get(str(path))
            if expected is not None and expected != self.provenance[key]:
                raise ValueError(f"Frozen G input no longer matches plan.json: {path}")
        if self.config["reference_seed"] != self.plan["reference_seed"]:
            raise ValueError("Frozen G config and suite reference seeds differ")
        self._jobs = {}
        self._config_cache = {str(self.config_path): self.config}
        self._checkpoint_cache = {str(self.checkpoint_path): self.provenance["checkpoint_sha256"]}

    def _validate_saved_planner(self, run, k):
        import yaml

        path = run / "code/hydra/config.yaml"
        saved = yaml.safe_load(path.read_text(encoding="utf-8"))
        try:
            planner = saved["planner"]["score_function_planner"]
            if float(planner["steps"]) != k or float(planner["gamma"]) != float(self.plan["gamma"]):
                raise ValueError(f"Saved planner K/gamma mismatch in {path}")
            config_path = str(Path(planner["score_config"]).expanduser().resolve())
            checkpoint_path = str(Path(planner["score_checkpoint"]).expanduser().resolve())
            if config_path not in self._config_cache:
                self._config_cache[config_path] = _read(config_path)
            if checkpoint_path not in self._checkpoint_cache:
                self._checkpoint_cache[checkpoint_path] = _digest(checkpoint_path)
            if self._config_cache[config_path] != self.config:
                raise ValueError(f"Saved planner config differs from frozen G in {path}")
            if self._checkpoint_cache[checkpoint_path] != self.provenance["checkpoint_sha256"]:
                raise ValueError(f"Saved planner checkpoint differs from frozen G in {path}")
            heading = planner.get("heading_projection")
            if heading is None:
                heading = self.config["refinement"]["heading_projection"]
            if heading != self.plan["heading_projection"]:
                raise ValueError(f"Saved heading projection differs in {path}")
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Incomplete saved score planner configuration in {path}") from exc

    def _job(self, k):
        job_id = "DP_K0" if k == 0 else f"S05-G_K{k}"
        if job_id in self._jobs:
            return self._jobs[job_id]
        import pandas as pd

        path = self.suite / "jobs" / f"{job_id}.json"
        if not path.is_file():
            raise ValueError(f"Missing completed evaluation record: {path}")
        job = _read(path)
        if job.get("state") not in ("complete", "reused"):
            raise ValueError(f"Evaluation {job_id} is not complete/reused: {job.get('state')}")
        if job.get("id") != job_id or job.get("model") != "S05-G" or job.get("K") != k:
            raise ValueError(f"Evaluation identity/model/K mismatch: {path}")
        if not job.get("result_dir"):
            raise ValueError(f"Evaluation record has no result directory: {path}")
        run = Path(job["result_dir"]).expanduser().resolve()
        if "closed_loop_nonreactive_agents" not in run.parts:
            raise ValueError(f"Expected a nonreactive closed-loop result directory: {run}")
        self._validate_saved_planner(run, k)
        files = sorted((run / "aggregator_metric").glob("*.parquet"))
        if len(files) != 1:
            raise ValueError(f"Expected one official aggregate for {job_id}, found {len(files)}")
        table = pd.read_parquet(files[0])
        required = {"scenario", "scenario_type", "log_name", "score"}
        if not required.issubset(table.columns):
            raise ValueError(
                f"Missing per-scenario metric columns: {sorted(required - set(table.columns))}"
            )
        # nuPlan also writes per-type and final_score aggregate rows. A genuine
        # scenario row has a recording log; summary rows must never substitute.
        detail = table.loc[table["log_name"].notna()].copy()
        detail = detail.loc[detail["log_name"].astype(str).str.len().gt(0)]
        result = {
            "job_id": job_id,
            "result_dir": str(run),
            "metric_file": str(files[0]),
            "table": detail,
        }
        self._jobs[job_id] = result
        return result

    def _row(self, k, token, scenario_type, expected_log=None):
        job = self._job(k)
        table = job["table"]
        rows = table.loc[
            table["scenario"].astype(str).eq(token)
            & table["scenario_type"].astype(str).eq(scenario_type)
        ]
        if len(rows) != 1:
            raise ValueError(
                f"Expected one scenario row for {job['job_id']}, token={token}, "
                f"type={scenario_type}; found {len(rows)} (summary rows excluded)"
            )
        row = rows.iloc[0]
        log_name = str(row["log_name"])
        if expected_log is not None and log_name != expected_log:
            raise ValueError(f"Recording mismatch for {token}: {expected_log!r} vs {log_name!r}")
        raw = _unit_score(row["score"], f"{job['job_id']}/{token}")
        metrics = {"score": raw}
        for key in METRICS:
            if key in rows.columns:
                metrics[key] = _unit_score(row[key], f"{job['job_id']}/{token}/{key}")
        return {
            "score": 100 * raw,
            "score_raw": raw,
            "log_name": log_name,
            "job_id": job["job_id"],
            "result_dir": job["result_dir"],
            "metric_file": job["metric_file"],
            "metrics": metrics,
        }

    def for_payload(self, payload, ks):
        """Return score points (0–100), with raw metrics and exact source paths."""
        normalized = []
        for k in ks:
            if isinstance(k, bool) or not isinstance(k, Integral) or k <= 0:
                raise ValueError("Requested score K values must be positive integers for S05-G")
            if int(k) not in normalized:
                normalized.append(int(k))
        if not normalized:
            raise ValueError("At least one positive S05-G K is required")
        metadata = payload.get("metadata", {})
        if "suite" in metadata and Path(metadata["suite"]).expanduser().resolve() != self.suite:
            raise ValueError("Capture and requested score suite paths differ")
        for key, value in self.provenance.items():
            if key in metadata and metadata[key] != value:
                raise ValueError(f"Capture provenance mismatch: {key}")
        if (
            "reference_seed" in metadata
            and metadata["reference_seed"] != self.plan["reference_seed"]
        ):
            raise ValueError("Capture reference seed differs from score suite")
        token, scenario_type = str(payload["token"]), str(payload["scenario_type"])
        if self.plan.get("tokens") is not None and token not in self.plan["tokens"]:
            raise ValueError(f"Token {token} is outside the frozen evaluation suite")
        baseline = self._row(0, token, scenario_type)
        if metadata.get("source_log"):
            source = Path(metadata["source_log"]).expanduser().resolve()
            if Path(baseline["result_dir"]) not in source.parents:
                raise ValueError("Capture source log is outside the DP_K0 reference run")
        by_k = {
            str(k): self._row(k, token, scenario_type, baseline["log_name"]) for k in normalized
        }
        return {
            "scope": "scenario_closed_loop",
            "scale": 100,
            "token": token,
            "scenario_type": scenario_type,
            "baseline": baseline,
            "by_k": by_k,
            "suite": str(self.suite),
            **self.provenance,
            "note": "Scores describe complete closed-loop scenario runs, not the plotted fixed-input frame.",
        }
