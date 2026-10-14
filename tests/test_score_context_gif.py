"""GIF timing, actor provenance and physical displacement checks, without nuPlan."""

import copy
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import score_context_gif as gif


def box(x, y, length=4, width=1.8):
    return [[x - length / 2, y - width / 2], [x + length / 2, y - width / 2],
            [x + length / 2, y + width / 2], [x - length / 2, y + width / 2]]


def agents(x, prefix):
    return [
        {"kind": kind, "track_token": prefix + str(i), "polygon": box(x + dx, dy, size, 1),
         "center": [x + dx, dy], "heading": 0}
        for i, (kind, dx, dy, size) in enumerate([
            ("VEHICLE", 10, 3, 4), ("PEDESTRIAN", 6, -3, 0.6), ("BICYCLE", 18, -2, 1.5)
        ])
    ]


def refinement_fixture():
    t = np.arange(1, 81) / 10
    dp = np.column_stack((4 * t, 0.015 * (4 * t) ** 2))
    offset = np.column_stack((np.zeros(80), np.sin(t / 8 * np.pi)))
    return {
        "case_id": "SYNTHETIC_QA", "token": "NOT_A_NUPLAN_RESULT",
        "scenario_type": "following_lane_with_lead", "frame_index": 0,
        "future_time_s": t.tolist(),
        "trajectories": {"0": dp.tolist(), "5": (dp + 0.2 * offset).tolist(),
                         "20": (dp + 0.6 * offset).tolist()},
        "map_polygons": {"lane": [np.vstack((dp + [0, 2], (dp - [0, 2])[::-1])).tolist()]},
        "map_lines": {"lane_center": [dp.tolist()]}, "route_lines": [dp.tolist()],
        "agents": agents(0, "snapshot_"), "ego_polygon": box(1.5, 0),
    }


def closed_loop_fixture():
    payload = refinement_fixture()
    payload["metadata"] = {"kind": "timestamped_closed_loop_agents", "map_camera_radius_m": 35}
    payload["closed_loop_scores"] = {
        "scope": "scenario_closed_loop", "scale": 100,
        "baseline": {"score": 80}, "by_k": {"20": {"score": 85}},
    }
    payload["runs"] = {}
    # Different start, end and sample rates. Index alignment would be wrong.
    for k, times in (("0", [0, 100000, 200000]), ("20", [100000, 300000, 500000])):
        frames = []
        for i, time in enumerate(times):
            x, y = i * 3, i * (0.2 if k == "0" else 0.7)
            frames.append({
                "timestamp_us": 1_700_000_000_000_000 + time,
                "ego_xy": [x, y], "ego_polygon": box(x + 1.5, y),
                "agents": agents(x, f"run{k}_") if i != 1 else [],
                "traffic_lights": {"signal1": "green" if i else "red"},
            })
        payload["runs"][k] = {"frames": frames}
    payload["signal_connectors"] = {"signal1": [[10, 0], [15, 0]]}
    return payload


class ObservedSample:
    def __init__(self, x, y, time, kind="VEHICLE", token="track1"):
        def geometry(points):
            return SimpleNamespace(exterior=SimpleNamespace(coords=points))

        self.ego_state = SimpleNamespace(
            rear_axle=SimpleNamespace(x=x, y=y, heading=np.pi / 2),
            time_point=SimpleNamespace(time_us=time),
            car_footprint=SimpleNamespace(geometry=geometry(box(x, y))),
        )
        obj = SimpleNamespace(
            box=SimpleNamespace(center=SimpleNamespace(x=x + 2, y=y + 3, heading=np.pi),
                                geometry=geometry(box(x + 2, y + 3))),
            tracked_object_type=SimpleNamespace(name=kind), track_token=token,
        )
        self.observation = SimpleNamespace(tracked_objects=SimpleNamespace(tracked_objects=[obj]))
        self.traffic_light_status = [SimpleNamespace(lane_connector_id=12,
                                                     status=SimpleNamespace(name="RED"))]

    @property
    def trajectory(self):
        raise AssertionError("Predicted trajectories must not replace actual ego/agent history")


class ScoreContextGifTest(unittest.TestCase):
    def test_clocks_use_integer_timestamps_and_stop_at_run_end(self):
        payload = closed_loop_fixture()
        clocks = [gif.validate_clock(run["frames"]) for run in payload["runs"].values()]
        timeline = gif.playback_times(clocks, fps=10)
        self.assertEqual([gif.frame_at(clocks[0], t) for t in timeline], [0, 1, 2, None, None, None])
        self.assertEqual([gif.frame_at(clocks[1], t) for t in timeline], [None, 0, 0, 1, 1, 2])
        self.assertEqual(len(gif.playback_times(clocks, fps=10, speed=2)), 4)
        timeline_30fps = gif.playback_times([np.array([0, 999990])], fps=30)
        durations = gif.playback_durations(timeline_30fps, fps=30, speed=1)
        self.assertTrue({30, 40}.issubset(set(durations)))
        self.assertLess(abs(sum(durations) - (999.99 + 1000 / 30)), 10)
        duplicate = [payload["runs"]["0"]["frames"][0]] * 2
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            gif.validate_clock(duplicate)
        missing = copy.deepcopy(payload["runs"]["0"]["frames"])
        del missing[0]["agents"]
        with self.assertRaisesRegex(ValueError, "Missing observed agents"):
            gif.validate_clock(missing)

    def test_extraction_transforms_footprints_centers_and_headings_together(self):
        samples = [ObservedSample(10, 20, 100), ObservedSample(8, 23, 200, "PEDESTRIAN")]
        anchor = SimpleNamespace(x=10, y=20, heading=np.pi / 2)
        frames = gif.extract_frames(samples, anchor)
        np.testing.assert_allclose([f["ego_xy"] for f in frames], [[0, 0], [3, 2]], atol=1e-12)
        np.testing.assert_allclose(frames[0]["agents"][0]["center"], [3, -2], atol=1e-12)
        self.assertAlmostEqual(frames[0]["agents"][0]["heading"], np.pi / 2)
        np.testing.assert_allclose(
            frames[0]["agents"][0]["polygon"],
            gif.core.to_local(box(12, 23), anchor), atol=1e-12,
        )
        self.assertEqual(frames[1]["agents"][0]["kind"], "PEDESTRIAN")
        self.assertEqual(frames[0]["traffic_lights"], {"12": "red"})

    def test_closed_loop_extraction_queries_map_covering_both_runs(self):
        import closed_loop_context_overview as overview

        case = ("I01", "synthetic_token", "following_lane_with_lead")
        module = ModuleType("nuplan.common.maps.maps_datatypes")
        module.SemanticMapLayer = SimpleNamespace(LANE_CONNECTOR="connector")
        scenarios, logs, sources = [], {}, {}
        with tempfile.TemporaryDirectory() as tmp:
            for k, last_x in ((0, 12), (20, 120)):
                path = Path(tmp) / f"{k}.pkl.xz"
                path.write_bytes(f"synthetic log {k}".encode())
                scenario = SimpleNamespace(token=case[1], scenario_type=case[2], log_name="recording",
                                           map_api=SimpleNamespace(get_map_object=lambda *_: None))
                scenarios.append(scenario)
                sources[str(k)] = {"result_dir": str(path), "log_name": "recording"}
                logs[path] = SimpleNamespace(scenario=scenario, simulation_history=SimpleNamespace(data=[
                    ObservedSample(10, 20, 100, token=f"run{k}"),
                    ObservedSample(last_x, 23, 200, token=f"run{k}"),
                ]))
            scores = {"baseline": sources["0"], "by_k": {"20": sources["20"]}}
            with patch.dict(sys.modules, {module.__name__: module}), \
                    patch.object(overview, "_log_path", side_effect=lambda row, _: Path(row["result_dir"])), \
                    patch.object(gif.core, "load_log", side_effect=lambda path: logs[path]), \
                    patch.object(gif.core, "extract_environment", return_value={}) as environment:
                payload = gif.extract_closed_loop(case, scores, [20])
            np.testing.assert_allclose(payload["runs"]["20"]["frames"][-1]["ego_xy"],
                                       [3, -110], atol=1e-12)
            self.assertEqual(payload["runs"]["0"]["frames"][0]["agents"][0]["track_token"], "run0")
            self.assertEqual(payload["runs"]["20"]["frames"][0]["agents"][0]["track_token"], "run20")
            self.assertIs(environment.call_args.args[0], scenarios[0])
            corners = np.asarray(environment.call_args.args[2]["view_bounds"])
            self.assertLessEqual(corners[:, 1].min(), -145)
            self.assertEqual(payload["signal_connectors"], {"12": []})

    def test_refinement_gif_contains_only_saved_K_and_physical_update_arrows(self):
        from matplotlib.quiver import Quiver

        payload = refinement_fixture()
        snapshots = []
        original = gif.figure_image

        def inspect(figure):
            quivers = [a for a in figure.axes[0].collections if isinstance(a, Quiver)]
            snapshots.append({"arrows": [(q.U.copy(), q.V.copy(), q.scale) for q in quivers],
                              "bounds": figure.axes[0].get_xlim(),
                              "actors": len(figure.axes[0].patches),
                              "text": " ".join(t.get_text() for t in figure.texts)})
            return original(figure)

        with tempfile.TemporaryDirectory() as tmp, patch.object(gif, "figure_image", side_effect=inspect):
            output = Path(tmp) / "refinement.gif"
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message="Glyph .* missing from font")
                info = gif.render_refinement(payload, output, dpi=60, agent_ids=True)
            self.assertEqual(info["K"], [0, 5, 20])
            self.assertFalse(info["interpolated_K"])
            with Image.open(output) as im:
                self.assertEqual(im.n_frames, 3)
                for i in range(3):
                    im.seek(i)
                    im.convert("RGB").load()
                    self.assertEqual(im.info["duration"], 800)
            self.assertEqual(snapshots[0]["arrows"], [])
            self.assertEqual(len({s["bounds"] for s in snapshots}), 1)
            self.assertTrue(all(s["actors"] == 5 for s in snapshots))  # Lane, 3 agents, ego.
            u, v, scale = snapshots[2]["arrows"][0]
            expected = np.asarray(payload["trajectories"]["20"]) - payload["trajectories"]["5"]
            np.testing.assert_allclose(np.column_stack((u, v)), expected[::5])
            self.assertEqual(scale, 1)
            self.assertIn("K=5 → 20", snapshots[2]["text"])
            with self.assertRaisesRegex(ValueError, "Every K must exist"):
                gif.render_refinement(payload, output, ks=[7])

    def test_closed_loop_uses_each_runs_agents_and_does_not_freeze_ended_run(self):
        payload = closed_loop_fixture()
        rendered, signals = [], []
        original = gif.figure_image

        def inspect(figure):
            rendered.append([
                {"title": ax.get_title(loc="left"), "bounds": ax.get_xlim(),
                 "actors": len(ax.patches)} for ax in figure.axes
            ])
            signals.append([[line.get_color() for line in ax.lines] for ax in figure.axes])
            return original(figure)

        with tempfile.TemporaryDirectory() as tmp, patch.object(gif, "figure_image", side_effect=inspect), \
                patch.object(gif, "draw_actors", wraps=gif.draw_actors) as draw:
            output = Path(tmp) / "closed_loop.gif"
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message="Glyph .* missing from font")
                info = gif.render_closed_loop(payload, 20, output, dpi=60)
            self.assertEqual(info["frame_count"], 6)
            with Image.open(output) as im:
                self.assertEqual(im.n_frames, 6)
                for i in range(6):
                    im.seek(i)
                    self.assertEqual(im.info["duration"], 100)
                    im.convert("RGB").load()
            # 3 DP frames; 5 refine display frames, with causal holds at 0.2 and 0.4s.
            self.assertEqual(draw.call_count, 8)
            own_frames = [call.args[1] for call in draw.call_args_list]
            self.assertEqual([f["agents"][0]["track_token"] for f in own_frames if f["agents"]],
                             ["run0_0", "run20_0", "run0_0", "run20_0", "run20_0"])
            self.assertIn("已结束", rendered[3][0]["title"])
            self.assertEqual(rendered[3][0]["actors"], 1)  # Lane only, no stale ego/agents.
            self.assertIn("agent=0", rendered[3][1]["title"])
            self.assertTrue(all(row[0]["bounds"] == row[1]["bounds"] for row in rendered))
            self.assertIn("#c55252", signals[0][0])
            self.assertIn("#498864", signals[3][1])
            with self.assertRaisesRegex(ValueError, "re-extraction"):
                gif.render_closed_loop(payload, 20, output, radius=60)

    def test_failed_stream_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "existing.gif"
            output.write_bytes(b"previous artifact")
            with self.assertRaisesRegex(ValueError, "size changed"):
                with gif.GifStream(output) as writer:
                    writer.add(Image.new("RGB", (20, 20), "red"), 100)
                    writer.add(Image.new("RGB", (21, 20), "blue"), 100)
            self.assertEqual(output.read_bytes(), b"previous artifact")
            self.assertEqual(list(Path(tmp).glob("*.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
