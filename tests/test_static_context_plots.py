"""Geometry and rendering checks; no nuPlan, checkpoints, or GPU required."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location(
    "static_context_plots", SCRIPTS / "static_context_plots.py"
)
PLOTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOTS)


def fixture():
    t = np.linspace(0.1, 8.0, 80)
    dp = np.column_stack((t * 4, 0.01 * (t * 4) ** 2))
    snapshots = {"0": dp.tolist()}
    for k in PLOTS.OVERVIEW_KS:
        offset = (1 - np.exp(-k / 60.0)) * np.sin(t / 8 * np.pi)
        snapshots[str(k)] = (dp + np.column_stack((0.1 * offset, 0.6 * offset))).tolist()
    lane = np.vstack((dp + [0, 2], (dp - [0, 2])[::-1]))
    return {
        "case_id": "synthetic_geometry_QA",
        "token": "synthetic_not_a_nuplan_result",
        "scenario_type": "turning_demo",
        "frame_index": 12,
        "trajectories": snapshots,
        "future_time_s": t.tolist(),
        "map_polygons": {"lane": [lane.tolist()]},
        "map_lines": {"lane_boundaries": [(dp + [0, 2]).tolist(), (dp - [0, 2]).tolist()]},
        "route_lines": [dp.tolist()],
        "ego_polygon": [[-1, -0.9], [4, -0.9], [4, 0.9], [-1, 0.9]],
        "agents": [
            {"kind": "vehicle", "polygon": [[16, 3], [20, 3], [20, 5], [16, 5]]},
            {"kind": "pedestrian", "polygon": [[24, 8], [24.5, 8], [24.5, 8.5], [24, 8.5]]},
        ],
    }


class StaticContextPlotsTest(unittest.TestCase):
    def test_displacements_use_dp_tangent_and_stationary_fallback(self):
        vertical = np.column_stack((np.zeros(8), np.arange(8)))
        along, left = PLOTS.displacement_components(vertical, vertical + [-2, 3])
        np.testing.assert_allclose(along, 3)
        np.testing.assert_allclose(left, 2)
        stationary = np.zeros((8, 2))
        along, left = PLOTS.displacement_components(stationary, stationary + [4, -1])
        np.testing.assert_allclose(along, 4)
        np.testing.assert_allclose(left, -1)

    def test_mismatched_future_horizons_fail(self):
        with self.assertRaisesRegex(ValueError, "timestep count"):
            PLOTS._trajectories({"trajectories": {"0": [[0, 0], [1, 1]], "20": [[0, 0]]}})

    def test_context_and_detail_share_finite_physical_bounds(self):
        class SharedPages:
            def __init__(self):
                self.pages = []

            def savefig(self, figure):
                self.pages.append(figure)

        shared = SharedPages()
        with tempfile.TemporaryDirectory() as tmp:
            result = PLOTS.render_case(fixture(), tmp, shared_pdf=shared)
            self.assertEqual(len(shared.pages), 3)
            self.assertEqual(len(result["png"]), 3)
            for path in [result["pdf"], *result["png"]]:
                self.assertGreater(Path(path).stat().st_size, 1000)
            self.assertTrue(np.isfinite(result["limits"]).all())
            self.assertTrue(np.isfinite(result["zoom_limits"]).all())
            self.assertLess(
                result["zoom_limits"][1] - result["zoom_limits"][0],
                result["limits"][1] - result["limits"][0],
            )
            self.assertEqual(PLOTS._trajectories(fixture())[0].shape, (80, 2))
            for figure, k in zip(shared.pages, (20, 50, 200)):
                page_text = "\n".join(text.get_text() for text in figure.texts)
                self.assertIn("闭环得分未加载", page_text)
                self.assertNotIn("DP 0.00", page_text)
                paths = [
                    line
                    for ax in figure.axes
                    for line in ax.lines
                    if line.get_marker() == "o" and len(line.get_xdata()) == 80
                ]
                # Every page has an overview and a zoom, each showing exactly
                # the same DP and one selected K. Other K curves obscure changes.
                self.assertEqual(len(paths), 4)
                self.assertEqual(
                    {line.get_color() for line in paths}, {PLOTS.DP_COLOR, PLOTS.REFINED_COLOR}
                )
                for line in paths:
                    expected_k = 0 if line.get_color() == PLOTS.DP_COLOR else k
                    expected = np.asarray(fixture()["trajectories"][str(expected_k)])
                    np.testing.assert_array_equal(line.get_xdata(), expected[:, 0])
                    np.testing.assert_array_equal(line.get_ydata(), expected[:, 1])
                for line in paths:
                    self.assertLessEqual(line.get_linewidth(), 1.0)
                    self.assertLessEqual(line.get_markersize(), 1.5)
                    self.assertEqual(line.axes.get_aspect(), 1.0)

    def test_case_groups_are_selection_labels_and_do_not_decide_score_verdicts(self):
        for case, expected in (
            ("I01", "改善候选"),
            ("W04", "退化候选"),
            ("K02", "步数敏感候选"),
            ("C03", "对照"),
            ("QA_LONG", "诊断场景"),
        ):
            self.assertEqual(PLOTS.case_category(case), expected)
        payload = {
            "case_id": "I01",
            "closed_loop_scores": {
                "scope": "scenario_closed_loop",
                "scale": 100,
                "baseline": {"score": 100.0},
                "by_k": {"50": {"score": 0.0}},
            },
        }
        band = PLOTS._score_band(payload, 50)
        self.assertEqual(band["verdict"], "worsened")
        self.assertEqual(band["delta"], -100.0)
        self.assertIn("退步 -100.00 分", band["text"])

    def test_score_rounding_does_not_hide_small_changes_or_invent_missing_values(self):
        payload = {
            "closed_loop_scores": {
                "scope": "scenario_closed_loop",
                "scale": 100,
                "baseline": {"score": 76.99},
                "by_k": {},
            }
        }
        for delta, verdict, word in (
            (0.000004, "improved", "改善"),
            (-0.000004, "worsened", "退步"),
            (0.0000004, "tie", "持平"),
        ):
            payload["closed_loop_scores"]["by_k"]["20"] = {"score": 76.99 + delta}
            band = PLOTS._score_band(payload, 20)
            self.assertEqual(band["verdict"], verdict)
            self.assertIn(word, band["text"])
            if verdict != "tie":
                self.assertIn("<0.01 分", band["text"])
                self.assertNotIn("+0.00", band["text"])
                self.assertNotIn("-0.00", band["text"])
        self.assertEqual(PLOTS._score_band(payload, 50)["verdict"], "missing")
        self.assertEqual(PLOTS._score_band({}, 20)["text"], "闭环得分未加载")
        for score in (float("nan"), float("inf"), -1, 101):
            payload["closed_loop_scores"]["by_k"]["20"] = {"score": score}
            with self.assertRaises(ValueError):
                PLOTS._score_band(payload, 20)
        payload["closed_loop_scores"]["by_k"]["20"] = {"score": 77}
        payload["closed_loop_scores"]["scale"] = 1
        with self.assertRaisesRegex(ValueError, "scale=100"):
            PLOTS._score_band(payload, 20)

    def test_rendered_pages_show_the_corresponding_k_whole_scene_score(self):
        class SharedPages:
            def __init__(self):
                self.pages = []

            def savefig(self, figure):
                self.pages.append(figure)

        payload = fixture()
        payload["case_id"] = "K01"
        payload["closed_loop_scores"] = {
            "scope": "scenario_closed_loop",
            "scale": 100,
            "baseline": {"score": 99.89},
            "by_k": {"20": {"score": 100.0}, "50": {"score": 0.0}, "200": {"score": 98.62}},
        }
        expected = (
            "G K=20 100.00  |  改善 +0.11 分",
            "G K=50 0.00  |  退步 -99.89 分",
            "G K=200 98.62  |  退步 -1.27 分",
        )
        shared = SharedPages()
        with tempfile.TemporaryDirectory() as tmp:
            PLOTS.render_case(payload, tmp, shared_pdf=shared)
        for figure, score_text in zip(shared.pages, expected):
            page_text = "\n".join(text.get_text() for text in figure.texts)
            self.assertIn("K01（步数敏感候选）", page_text)
            self.assertIn("DP 99.89", page_text)
            self.assertIn(score_text, page_text)
            self.assertIn("整段场景得分，并非当前帧得分", page_text)
            self.assertIn("编号不代表排名", page_text)
            self.assertNotIn("闭环得分未加载", page_text)

    def test_limits_keep_long_turning_and_stationary_trajectories_in_metres(self):
        t = np.linspace(0.0, 1.0, 80)
        examples = (
            np.column_stack((180 * t, np.zeros_like(t))),
            np.column_stack((30 * np.sin(t * np.pi / 2), 30 * (1 - np.cos(t * np.pi / 2)))),
            np.column_stack((0.001 * t, 0.001 * t)),
        )
        for points in examples:
            limits = PLOTS._limits([points], min_span=8.0)
            self.assertTrue(np.isfinite(limits).all())
            self.assertGreaterEqual(limits[1] - limits[0], 8.0)
            self.assertGreaterEqual(limits[3] - limits[2], 8.0)
            self.assertTrue((points[:, 0] >= limits[0]).all())
            self.assertTrue((points[:, 0] <= limits[1]).all())
            self.assertTrue((points[:, 1] >= limits[2]).all())
            self.assertTrue((points[:, 1] <= limits[3]).all())


if __name__ == "__main__":
    unittest.main()
