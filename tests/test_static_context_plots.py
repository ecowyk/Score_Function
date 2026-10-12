"""Geometry and rendering checks; no nuPlan, checkpoints, or GPU required."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "static_context_plots", Path(__file__).parents[1] / "scripts/static_context_plots.py"
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
            self.assertEqual(len(shared.pages), 2)
            self.assertEqual(len(result["png"]), 2)
            for path in [result["pdf"], *result["png"]]:
                self.assertGreater(Path(path).stat().st_size, 1000)
            self.assertTrue(np.isfinite(result["limits"]).all())
            self.assertTrue(np.isfinite(result["zoom_limits"]).all())
            self.assertLess(
                result["zoom_limits"][1] - result["zoom_limits"][0],
                result["limits"][1] - result["limits"][0],
            )
            self.assertEqual(PLOTS._trajectories(fixture())[0].shape, (80, 2))


if __name__ == "__main__":
    unittest.main()
