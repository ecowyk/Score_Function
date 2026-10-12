"""Replay integrity checks without nuPlan installation, server data, or a GPU."""

import importlib.util
import lzma
import pickle
import sys
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "static_score_context", Path(__file__).parents[1] / "scripts/static_score_context.py"
)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def test_world_to_ego_transform_preserves_metric_distances_and_signs():
    anchor = SimpleNamespace(x=300_000.0, y=4_000_000.0, heading=np.pi / 2)
    world = [[anchor.x, anchor.y], [anchor.x, anchor.y + 3], [anchor.x - 2, anchor.y]]
    local = helper.to_local(world, anchor)
    np.testing.assert_allclose(local, [[0, 0], [3, 0], [0, 2]], atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(local[1] - local[2]), np.sqrt(13))
    np.testing.assert_allclose(
        helper.to_local(world, SimpleNamespace(**{**vars(anchor), "heading": 0})),
        [[0, 0], [0, 3], [-2, 0]],
    )


def test_probe_indices_and_cohort_preserve_saved_sample_identity():
    saved = [object() for _ in range(19)]
    probes = helper.probe_frames(len(saved))
    assert probes == [0, 6, 12]
    assert [saved[i] for i in probes] == [saved[0], saved[6], saved[12]]
    assert helper.probe_frames(1) == [0]
    assert helper.probe_frames(19, [12, 0, 12]) == [0, 12]
    assert [c[0] for c in helper.selected_cases("W04,I01,W04")] == ["W04", "I01"]
    for count, requested in ((0, None), (19, [-1]), (19, [19]), (19, [])):
        with pytest.raises(ValueError, match="Frame indices"):
            helper.probe_frames(count, requested)
    with pytest.raises(ValueError, match="cohort"):
        helper.selected_cases("not_a_case")


@pytest.mark.parametrize("extension", ["pkl.xz", "msgpack.xz"])
def test_trusted_gpu_tagged_pickle_restores_storage_on_cpu_and_discards_planner(
    tmp_path, extension
):
    torch = pytest.importorskip("torch")
    original = SimpleNamespace(
        planner=SimpleNamespace(label="saved_planner_not_for_replay"),
        tensor=torch.arange(6, dtype=torch.float32).reshape(2, 3),
        scenario=SimpleNamespace(token="source_token"),
    )
    # Create authentic torch storage serialization tagged CUDA, using CPU
    # tensors only. It would require a CUDA device without map_location=cpu.
    with patch("torch.serialization.location_tag", return_value="cuda:7"):
        data = pickle.dumps(original, protocol=pickle.HIGHEST_PROTOCOL)
    assert b"cuda:7" in data
    if extension.startswith("msgpack"):
        msgpack = pytest.importorskip("msgpack")
        data = msgpack.packb(data, use_bin_type=True)
    path = tmp_path / ("trusted_fixture." + extension)
    with lzma.open(path, "wb") as stream:
        stream.write(data)
    restored = helper.load_log(path)
    assert restored.planner is None
    assert restored.tensor.device.type == "cpu"
    torch.testing.assert_close(restored.tensor, original.tensor)
    assert restored.scenario.token == "source_token"


def test_history_uses_recorded_closed_loop_states_and_observations(monkeypatch):
    history_module = ModuleType("nuplan.planning.simulation.history.simulation_history_buffer")
    observation_module = ModuleType("nuplan.planning.simulation.observation.observation_type")
    detections_type = type("DetectionsTracks", (), {})
    observation_module.DetectionsTracks = detections_type
    recorded = [SimpleNamespace(ego_state=object(), observation=object()) for _ in range(10)]
    past_ego, past_observation = object(), object()

    class History:
        @staticmethod
        def initialize_from_scenario(size, scenario, observation_type):
            assert observation_type is detections_type
            assert scenario.database_interval == 0.1
            result = History()
            result.ego_states = deque([past_ego] * size, maxlen=size)
            result.observations = deque([past_observation] * size, maxlen=size)
            return result

        def append(self, ego, observation):
            self.ego_states.append(ego)
            self.observations.append(observation)

    history_module.SimulationHistoryBuffer = History
    monkeypatch.setitem(sys.modules, history_module.__name__, history_module)
    monkeypatch.setitem(sys.modules, observation_module.__name__, observation_module)
    scenario = SimpleNamespace(database_interval=0.1)
    start = helper.make_history(scenario, recorded, 0, 0.2)
    assert list(start.ego_states)[-1] is recorded[0].ego_state
    assert sum(x is recorded[0].ego_state for x in start.ego_states) == 1
    assert list(start.ego_states)[:-1] == [past_ego] * (len(start.ego_states) - 1)
    late = helper.make_history(scenario, recorded, 7, 0.2)
    retained = len(late.ego_states)
    assert list(late.ego_states) == [s.ego_state for s in recorded[8 - retained : 8]]
    assert list(late.observations) == [s.observation for s in recorded[8 - retained : 8]]
    assert all(x is not recorded[8].ego_state for x in late.ego_states)


@pytest.mark.parametrize("directory", ["simulation", "simulation_log"])
def test_find_log_requires_unique_saved_scenario_across_planners_and_formats(tmp_path, directory):
    token = "cf87d58064425ee7"
    with pytest.raises(ValueError, match="found 0"):
        helper.find_log(tmp_path, token)
    selected = (
        tmp_path
        / directory
        / "score_planner"
        / "following_lane_with_lead"
        / (token + ".msgpack.xz")
    )
    selected.parent.mkdir(parents=True)
    selected.touch()
    unrelated = selected.with_name("other_token.msgpack.xz")
    unrelated.touch()
    assert helper.find_log(tmp_path, token) == selected
    other_directory = "simulation_log" if directory == "simulation" else "simulation"
    duplicate = tmp_path / other_directory / "second_planner" / (token + ".pkl.xz")
    duplicate.parent.mkdir(parents=True)
    duplicate.touch()
    with pytest.raises(ValueError, match="found 2"):
        helper.find_log(tmp_path, token)


def test_find_log_honors_external_callback_root_and_deduplicates_resolved_paths(tmp_path):
    run, external = tmp_path / "exact_run", tmp_path / "configured_logs"
    run.mkdir()
    external.mkdir()
    token = "cf87d58064425ee7"
    log = external / (token + ".msgpack.xz")
    log.touch()
    assert helper.find_log(run, token, log_root=external) == log.resolve()
    # The explicitly configured root and an in-run link identify one saved log.
    linked = run / "simulation_log" / log.name
    linked.parent.mkdir()
    linked.symlink_to(log)
    assert helper.find_log(run, token, log_root=external) == log.resolve()
    # Merely placing a second experiment nearby must not broaden the search.
    other = tmp_path / "different_run" / log.name
    other.parent.mkdir()
    other.touch()
    assert helper.find_log(run, token, log_root=external) == log.resolve()


def test_empty_failed_capture_manifest_can_be_updated_but_real_captures_are_preserved(tmp_path):
    old, new = {"capture_script_sha256": "old"}, {"capture_script_sha256": "new"}
    helper.ensure_manifest(tmp_path, old)
    helper.ensure_manifest(tmp_path, new)
    assert helper.read(tmp_path / "capture_manifest.json") == new
    for name in ("payload.json", "inputs.npz"):
        path = tmp_path / "captures" / "I01" / "frame_0000" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"existing capture must survive")
        before = (tmp_path / "capture_manifest.json").read_bytes()
        with pytest.raises(ValueError):
            helper.ensure_manifest(tmp_path, old)
        assert (tmp_path / "capture_manifest.json").read_bytes() == before
        assert path.read_bytes() == b"existing capture must survive"
        # Matching provenance stays resumable even with existing captures.
        helper.ensure_manifest(tmp_path, new)
        path.unlink()


def saved_sample():
    anchor = SimpleNamespace(x=50.0, y=70.0, heading=np.pi / 2)
    timestamp = 1_791_550_000_000_001
    states = [SimpleNamespace(rear_axle=anchor, time_point=SimpleNamespace(time_us=timestamp))]
    for i in range(1, 81):
        states.append(
            SimpleNamespace(
                rear_axle=SimpleNamespace(x=50.0, y=70.0 + i),
                time_point=SimpleNamespace(time_us=timestamp + i * 100000 - 1),
            )
        )

    class SavedTrajectory:
        def get_state_at_times(self, times):
            raise AssertionError("Do not interpolate beyond the truncated saved endpoint")

        def get_sampled_trajectory(self):
            return states

    sample = SimpleNamespace(
        ego_state=SimpleNamespace(rear_axle=anchor, time_point=SimpleNamespace(time_us=timestamp)),
        trajectory=SavedTrajectory(),
    )
    prediction = np.column_stack((np.arange(1, 81), np.zeros(80), np.ones(80), np.zeros(80)))
    return sample, prediction, states


def test_saved_trajectory_replay_accepts_one_microsecond_truncation_and_detects_drift():
    sample, prediction, _ = saved_sample()
    assert helper.replay_error(prediction, sample)["max_xy_m"] < 1e-9
    prediction[1, 1] += 0.5
    error = helper.replay_error(prediction, sample)
    assert error["max_xy_m"] == pytest.approx(0.5)
    assert error["rms_xy_m"] == pytest.approx(np.sqrt(0.25 / 80))


@pytest.mark.parametrize("fault", ["future_time", "horizon", "start_time"])
def test_saved_trajectory_rejects_incompatible_pose_counts_and_time_grid(fault):
    sample, prediction, states = saved_sample()
    if fault == "future_time":
        states[-1].time_point.time_us += 50
    elif fault == "horizon":
        states.pop()
    else:
        states[0].time_point.time_us += 1
    with pytest.raises(ValueError, match="timesteps|horizon|start time"):
        helper.replay_error(prediction, sample)


def test_chunked_snapshot_refinement_matches_uninterrupted_updates_with_heading_projection():
    torch = pytest.importorskip("torch")
    from score_function.model.refinement import refine_ego
    from score_function.utils.normalizer import denormalize_ego_future

    class FixedScore(torch.nn.Module):
        def forward(self, x, scene, route):
            target = torch.zeros_like(x)
            target[..., 0] = 1.2
            target[..., 1] = 0.3
            target[..., 2] = 0.8
            target[..., 3] = 0.6
            return (target - x) * 10.0

    branch = FixedScore().eval()
    normalizer = SimpleNamespace(
        mean=torch.zeros(1, 1, 4), std=torch.tensor([[[20.0, 20.0, 1.0, 1.0]]])
    )
    initial = torch.randn(1, 80, 4, generator=torch.Generator().manual_seed(23))
    scene, route = torch.zeros(1, 2, 3), torch.zeros(1, 3)
    kwargs = dict(sigma=0.05, gamma=0.1, heading_projection=True, record_trace=False)
    full, _ = refine_ego(branch, initial, scene, route, normalizer, steps=500, **kwargs)
    current, previous = initial, 0
    for k in helper.KS[1:]:
        current, _ = refine_ego(
            branch, current, scene, route, normalizer, steps=k - previous, **kwargs
        )
        if k == 20:
            independently_20, _ = refine_ego(
                branch, initial, scene, route, normalizer, steps=20, **kwargs
            )
            assert torch.equal(current, independently_20)
        previous = k
    assert torch.equal(current, full)
    headings = denormalize_ego_future(full, normalizer)[..., 2:]
    torch.testing.assert_close(torch.linalg.vector_norm(headings, dim=-1), torch.ones(1, 80))
