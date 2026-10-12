"""Complete actual paths keep their own clocks and never use predicted futures."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "closed_loop_context_overview", SCRIPTS / "closed_loop_context_overview.py"
)
OVERVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OVERVIEW)


class ActualSample:
    def __init__(self, x, y, timestamp, heading=0):
        self.ego_state = SimpleNamespace(
            rear_axle=SimpleNamespace(x=x, y=y, heading=heading),
            time_point=SimpleNamespace(time_us=timestamp),
        )

    @property
    def trajectory(self):
        raise AssertionError("Predicted trajectories must not enter a full actual-path overview")


def fixture():
    tracks = {}
    for k, count, length in ((0, 81, 85), (20, 93, 92), (50, 70, 74), (200, 101, 103)):
        times = 1_000_000 + np.arange(count, dtype=np.int64) * 50_000
        x = np.linspace(0, length, count)
        y = np.sin(x / 100 * np.pi) * (0.2 + k / 300)
        tracks[str(k)] = dict(
            xy=np.column_stack((x, y)).tolist(),
            timestamps_us=times.tolist(),
            frame_count=count,
            duration_s=(count - 1) * 0.05,
        )
    return dict(
        case_id="K01",
        token="SYNTHETIC_QA_NOT_REAL_RESULTS",
        scenario_type="following_lane_with_lead",
        tracks=tracks,
        metadata={"kind": "actual_closed_loop_full_history"},
        map_lines={"boundaries": [[[0, -2], [110, -2]], [[0, 2], [110, 2]]]},
        map_polygons={"lane": [[[0, -2], [110, -2], [110, 2], [0, 2]]]},
        ego_polygon=[[-1, -0.9], [4, -0.9], [4, 0.9], [-1, 0.9]],
        agents=[{"kind": "vehicle", "polygon": [[20, -1], [24, -1], [24, 1], [20, 1]]}],
        closed_loop_scores={
            "scope": "scenario_closed_loop",
            "scale": 100,
            "baseline": {"score": 99.89},
            "by_k": {"20": {"score": 100.0}, "50": {"score": 0.0}, "200": {"score": 98.62}},
        },
    )


class ClosedLoopOverviewTest(unittest.TestCase):
    def test_all_actual_states_use_dp_anchor_without_prediction_or_time_resampling(self):
        samples = [
            ActualSample(10, 20, 1_000_000),
            ActualSample(10, 23, 1_070_000),
            ActualSample(8, 23, 1_190_000),
        ]
        anchor = SimpleNamespace(x=10, y=20, heading=np.pi / 2)
        result = OVERVIEW.history_track(samples, anchor)
        np.testing.assert_allclose(result["xy"], [[0, 0], [3, 0], [3, 2]], atol=1e-12)
        self.assertEqual(result["timestamps_us"], [1_000_000, 1_070_000, 1_190_000])
        self.assertEqual(result["frame_count"], 3)
        self.assertEqual(result["duration_s"], 0.19)
        self.assertEqual(OVERVIEW.validate_track(result).shape, (3, 2))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            OVERVIEW.history_track([samples[0], samples[0]], anchor)
        result["frame_count"] = 4
        with self.assertRaisesRegex(ValueError, "frame count"):
            OVERVIEW.validate_track(result)

    def test_extract_keeps_unequal_runs_and_removes_initial_dynamic_context(self):
        case = ("I01", "token123", "following_lane_with_lead")
        score_bundle = {
            "baseline": {"score": 80, "log_name": "recording_A", "result_dir": "run0"},
            "by_k": {
                str(k): {"score": 85, "log_name": "recording_A", "result_dir": f"run{k}"}
                for k in (20, 50, 200)
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            logs, paths = {}, {}
            for k, count in ((0, 3), (20, 4), (50, 2), (200, 5)):
                path = Path(tmp) / f"{k}.pkl.xz"
                path.write_bytes(f"trusted fake log {k}".encode())
                paths[f"run{k}"] = path
                logs[path] = SimpleNamespace(
                    scenario=SimpleNamespace(
                        token=case[1], scenario_type=case[2], log_name="recording_A"
                    ),
                    simulation_history=SimpleNamespace(
                        data=[
                            ActualSample(
                                10 + i, 20, 1_000_000 + i * (100_000 if k != 50 else 200_000)
                            )
                            for i in range(count)
                        ]
                    ),
                )
            environment = dict(
                map_lines={"lane_center": [[[0, 0], [5, 0]]], "red": [[[0, 0], [2, 0]]]},
                map_polygons={},
                route_lines=[],
                agents=[{"kind": "vehicle"}],
                ego_polygon=[],
            )
            with (
                patch.object(
                    OVERVIEW, "_log_path", side_effect=lambda row, token: paths[row["result_dir"]]
                ),
                patch.object(
                    OVERVIEW.capture_core, "load_log", side_effect=lambda path: logs[path]
                ),
                patch.object(
                    OVERVIEW.capture_core, "extract_environment", return_value=environment
                ),
            ):
                result = OVERVIEW.extract_case(case, score_bundle)
        self.assertEqual(
            {k: v["frame_count"] for k, v in result["tracks"].items()},
            {"0": 3, "20": 4, "50": 2, "200": 5},
        )
        self.assertEqual(result["tracks"]["50"]["timestamps_us"], [1_000_000, 1_200_000])
        self.assertNotIn("agents", result)
        self.assertNotIn("red", result["map_lines"])
        self.assertIn("lane_center", result["map_lines"])
        self.assertFalse(result["metadata"]["common_index_alignment"])

    def test_pages_show_full_paths_with_different_counts_and_actual_scope(self):
        class Pages:
            def __init__(self):
                self.values = []

            def savefig(self, fig):
                self.values.append(fig)

        pages = Pages()
        payload = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            result = OVERVIEW.render_overview(payload, tmp, shared_pdf=pages)
            self.assertEqual(len(result["png"]), 3)
            for path in [result["pdf"], *result["png"]]:
                self.assertTrue(Path(path).stat().st_size > 1000)
        for fig, k in zip(pages.values, (20, 50, 200)):
            text = "\n".join(t.get_text() for t in fig.texts)
            self.assertIn("全程实际行驶", text)
            self.assertIn("整个场景中车辆实际走过的路径", text)
            self.assertIn("不是某一帧预测", text)
            self.assertIn(f"G K={k}", text)
            self.assertEqual(len(fig.axes), 1)
            paths = [line for line in fig.axes[0].lines if line.get_marker() == "o"]
            self.assertEqual(
                [len(line.get_xdata()) for line in paths],
                [81, payload["tracks"][str(k)]["frame_count"]],
            )
            for line, key in zip(paths, ("0", str(k))):
                np.testing.assert_array_equal(
                    line.get_xdata(), np.array(payload["tracks"][key]["xy"])[:, 0]
                )
                self.assertEqual(line.axes.get_aspect(), 1)
            self.assertFalse(fig.axes[0].texts)  # no time-index pairing arrows


if __name__ == "__main__":
    unittest.main()
