"""Saved-result integrity and deterministic selection for the NAVSIM replay gallery."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

from score_function.navsim_dd import visualize as vis


def scene(token, log, base, refined):
    metrics = lambda score: {"score": score, "no_at_fault_collisions": 1.0,
                             "drivable_area_compliance": 1.0, "ego_progress": score,
                             "time_to_collision_within_bound": 1.0, "comfort": 1.0,
                             "driving_direction_compliance": 1.0}
    a, b = metrics(base), metrics(refined)
    return {"token": token, "log": log, "seed": 17, "top1_index": 3,
            "metrics": {"0": a, "20": b},
            "delta": {"0": {k: 0.0 for k in a}, "20": {k: b[k] - a[k] for k in a}},
            "trajectories": {"0": [[float(i), 0.0, 0.0] for i in range(1, 9)],
                             "20": [[float(i), 0.1, 0.0] for i in range(1, 9)]}}


def saved_evaluation(directory, records, shards=2):
    records = sorted(records, key=lambda r: r["token"])
    tokens = [r["token"] for r in records]
    run = {"split": "navtest", "count": len(records), "ks": [0, 20], "num_shards": shards,
           "token_digest": vis.digest_tokens(tokens), "data_root": "/data/navsim",
           "metric_cache": "/data/metrics", "score_checkpoint_sha256": "a" * 64}
    for i in range(shards):
        part = directory / f"part-{i:04d}-of-{shards:04d}"
        part.mkdir(parents=True)
        rows = records[i::shards]
        manifest = {"run": run, "shard_index": i, "shard_tokens": [r["token"] for r in rows]}
        vis.write_json(part / "run.json", manifest)
        vis.write_json(part / "complete.json", manifest)
        (part / "scenes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {"count": len(records), "by_k": {
        k: {"mean": {m: sum(r["metrics"][k][m] for r in records) / len(records)
                     for m in records[0]["metrics"][k]}} for k in ("0", "20")}}
    vis.write_json(directory / "summary.json", summary)
    vis.write_json(directory / "launcher.json", {"epochs": 500, "status": "complete",
                                               "checkpoint": "/run/e500/final_raw.pt"})
    return run


def file_digest_tree(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


class SelectionTest(unittest.TestCase):
    def test_five_each_distinct_logs_deterministic_and_ties_omitted(self):
        rows = [scene(f"i{i:02}", f"gain-log-{i}", 0.2, 0.9 - i * 0.05) for i in range(6)]
        rows += [scene("i-duplicate", "gain-log-0", 0.1, 0.95)]
        rows += [scene(f"d{i:02}", f"loss-log-{i}", 0.9, 0.1 + i * 0.05) for i in range(6)]
        rows += [scene("tie", "tie-log", 0.5, 0.5), scene("roundoff", "roundoff-log", 0.5, 0.5 + 1e-12)]
        expected = vis.select_cases(rows)
        shuffled = copy.deepcopy(rows)
        random.Random(91).shuffle(shuffled)
        self.assertEqual(expected, vis.select_cases(shuffled))
        self.assertEqual([c["case_id"] for c in expected],
                         [f"I{i:02d}" for i in range(1, 6)] + [f"D{i:02d}" for i in range(1, 6)])
        for group, sign in ((expected[:5], 1), (expected[5:], -1)):
            self.assertEqual(len({c["row"]["log"] for c in group}), 5)
            self.assertTrue(all(sign * c["row"]["delta"]["20"]["score"] > 0 for c in group))
        self.assertEqual(expected[0]["row"]["token"], "i-duplicate")
        self.assertNotIn("i00", [c["row"]["token"] for c in expected])
        self.assertNotIn("tie", [c["row"]["token"] for c in expected])
        self.assertNotIn("roundoff", [c["row"]["token"] for c in expected])

    def test_fewer_distinct_logs_fills_with_highest_remaining_without_duplicates(self):
        rows = [scene(f"i{i}", "one-gain-log", 0.1, 0.9 - i * 0.05) for i in range(7)]
        rows += [scene(f"d{i}", "one-loss-log", 0.9, 0.1 + i * 0.05) for i in range(7)]
        chosen = vis.select_cases(rows)
        self.assertEqual([c["row"]["token"] for c in chosen],
                         [f"i{i}" for i in range(5)] + [f"d{i}" for i in range(5)])
        self.assertEqual(len({c["row"]["token"] for c in chosen}), 10)

    def test_equal_delta_ties_use_log_then_token(self):
        rows = [scene("b", "a", 0.25, 0.5), scene("a", "a", 0.25, 0.5),
                scene("c", "b", 0.25, 0.5), scene("loss", "z", 0.5, 0.25)]
        self.assertEqual(vis.select_cases(rows, 1)[0]["row"]["token"], "a")

    def test_insufficient_sign_and_invalid_count_are_explicit(self):
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                vis.select_cases([], count)
        with self.assertRaisesRegex(ValueError, "worsened"):
            vis.select_cases([scene("only", "log", 0.1, 0.2)], 1)


class EvaluationIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.records = [scene(f"token{i}", f"log{i}", 0.25, 0.5 if i % 2 else 0.125) for i in range(6)]
        self.run = saved_evaluation(self.directory, self.records)

    def load(self):
        return vis.load_evaluation(self.directory, expected_count=6)

    def mutate_rows(self, fn):
        path = self.directory / "part-0000-of-0002/scenes.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        fn(rows)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_complete_load_is_read_only_and_hashes_actual_scene_bytes(self):
        before = file_digest_tree(self.directory)
        rows, run, provenance = self.load()
        self.assertEqual(len(rows), 6)
        self.assertEqual(run, self.run)
        self.assertEqual(file_digest_tree(self.directory), before)
        self.assertEqual(provenance["source"], str(self.directory.resolve()))
        self.assertEqual(len(provenance["scene_sha256"]), 2)
        for path, digest in provenance["scene_sha256"].items():
            self.assertEqual(digest, hashlib.sha256(Path(path).read_bytes()).hexdigest())

    def test_bad_delta_rejected_before_rendering(self):
        self.mutate_rows(lambda rows: rows[0]["delta"]["20"].update(score=0.99))
        with self.assertRaisesRegex(ValueError, "delta"):
            self.load()

    def test_missing_completion_rejected(self):
        (self.directory / "part-0001-of-0002/complete.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.load()

    def test_missing_shard_rejected_even_if_remaining_is_complete(self):
        import shutil
        shutil.rmtree(self.directory / "part-0001-of-0002")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.load()

    def test_summary_mismatch_rejected(self):
        summary = vis.read_json(self.directory / "summary.json")
        summary["by_k"]["20"]["mean"]["ego_progress"] += 0.01
        vis.write_json(self.directory / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "summary"):
            self.load()

    def test_duplicate_scene_rejected(self):
        self.mutate_rows(lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])))
        with self.assertRaisesRegex(ValueError, "duplicate token"):
            self.load()

    def test_nonfinite_or_wrong_shape_trajectory_rejected(self):
        self.mutate_rows(lambda rows: rows[0]["trajectories"]["20"][0].__setitem__(0, float("nan")))
        with self.assertRaisesRegex(ValueError, "trajectory"):
            self.load()

    def test_manifest_protocol_mismatch_rejected(self):
        directory = self.directory / "part-0001-of-0002"
        manifest = vis.read_json(directory / "run.json")
        manifest["run"]["score_checkpoint_sha256"] = "b" * 64
        for name in ("run.json", "complete.json"):
            vis.write_json(directory / name, manifest)
        with self.assertRaisesRegex(ValueError, "Mixed"):
            self.load()


class GalleryAndRunnerTest(unittest.TestCase):
    def test_gallery_escapes_metadata_and_only_links_complete_cases(self):
        gain = {"case_id": "I01", "group": "improved", "state": "complete",
                "row": scene("<token>", '<script>alert("x")</script>', 0.25, 0.5),
                "gif": 'I01_"test".gif', "poster": "I01.png"}
        loss = {"case_id": "D01", "group": "worsened", "state": "failed",
                "row": scene("loss", "log", 0.75, 0.5), "error": "bad <value> & reason"}
        state = {"status": "failed", "source": "/x/<source>", "cases": [gain, loss]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            vis.write_gallery(path, state)
            page = (path / "index.html").read_text()
        self.assertIn("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;", page)
        self.assertNotIn('<script>alert("x")</script>', page)
        self.assertIn("bad &lt;value&gt; &amp; reason", page)
        self.assertIn('data-gif="I01_&quot;test&quot;.gif"', page)
        self.assertEqual(page.count("<img "), 1)
        self.assertIn("+25.0000", page)
        self.assertIn("-25.0000", page)
        self.assertIn("不代表总体平均", page)

    def test_nonfinal_or_incomplete_launcher_never_creates_output(self):
        for epoch, status in ((100, "complete"), (500, "running"), (500, "failed")):
            with self.subTest(epoch=epoch, status=status), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                exp = root / "dataset/navsim/exp"
                source = exp / "eval"
                source.mkdir(parents=True)
                vis.write_json(source / "launcher.json", {"epochs": epoch, "status": status})
                with patch.object(vis, "setup_runtime") as setup, self.assertRaisesRegex(ValueError, "500"):
                    vis.run_visualization(root, source=source)
                setup.assert_not_called()
                self.assertFalse((exp / vis.VIS_POINTER).exists())
                self.assertEqual(list(exp.iterdir()), [source])

    def test_runtime_failure_records_failed_gallery_and_preserves_source(self):
        records = [scene("gain", "g", 0.25, 0.5), scene("loss", "d", 0.5, 0.25)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dataset/navsim/exp/eval"
            source.mkdir(parents=True)
            run = saved_evaluation(source, records)
            before = file_digest_tree(source)
            with patch.object(vis, "load_evaluation", return_value=(records, run, {})), \
                 patch.object(vis, "setup_runtime", side_effect=RuntimeError("missing map")), \
                 self.assertRaisesRegex(RuntimeError, "missing map"):
                vis.run_visualization(root, source=source, count=1)
            output = Path((source.parent / vis.VIS_POINTER).read_text().strip())
            state = vis.read_json(output / "selection.json")
            self.assertEqual(state["status"], "failed")
            self.assertIn("missing map", state["error"])
            self.assertEqual(file_digest_tree(source), before)
            self.assertIn("状态：failed", (output / "index.html").read_text())

    def test_runner_supplies_scene_expert_and_keeps_other_case_after_render_failure(self):
        import numpy as np

        records = [scene("gain", "g", 0.25, 0.5), scene("loss", "d", 0.5, 0.25)]
        expert = np.array([[i, 0.2, 0.0] for i in range(1, 9)], dtype=np.float64)
        expert_requests, replayed = [], []

        def get_scene(token):
            row = next(r for r in records if r["token"] == token)

            def future(n):
                expert_requests.append((token, n))
                return SimpleNamespace(poses=expert)

            return SimpleNamespace(
                scene_metadata=SimpleNamespace(initial_token=token, log_name=row["log"], num_history_frames=1),
                frames=[SimpleNamespace(ego_status=SimpleNamespace(ego_pose=np.array([10.0, 20.0, 0.25])))],
                get_future_trajectory=future,
            )

        metric = SimpleNamespace(ego_state=SimpleNamespace(rear_axle=SimpleNamespace(x=10.0, y=20.0, heading=0.25)))
        runtime = (SimpleNamespace(get_scene_from_token=get_scene),
                   SimpleNamespace(get_from_token=lambda token: metric), object(), object())
        backend, renderer = ModuleType("fake_replay"), ModuleType("fake_render")

        def replay(row, cache, simulator, scorer, expert_poses):
            np.testing.assert_array_equal(expert_poses, expert)
            self.assertIs(cache, metric)
            replayed.append(row["token"])
            return {"metadata": {}, "expert_xy": expert[:, :2].tolist()}

        def render(payload, gif, poster, **kwargs):
            if kwargs["case_id"] == "I01":
                raise RuntimeError("render I01 failed")
            gif.write_bytes(b"fake-test-gif")
            poster.write_bytes(b"fake-test-poster")
            return {"frames": 41, "duration_s": 4.0}

        backend.replay_case, renderer.render_case = replay, render
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dataset/navsim/exp/eval"
            source.mkdir(parents=True)
            run = saved_evaluation(source, records)
            before = file_digest_tree(source)
            with patch.object(vis, "load_evaluation", return_value=(records, run, {})), \
                 patch.object(vis, "setup_runtime", return_value=runtime), \
                 patch.dict("sys.modules", {"score_function.navsim_dd.visualization_replay": backend,
                                             "score_function.navsim_dd.visualization_render": renderer}), \
                 self.assertRaisesRegex(RuntimeError, "I01"):
                vis.run_visualization(root, source=source, count=1)
            output = Path((source.parent / vis.VIS_POINTER).read_text().strip())
            state = vis.read_json(output / "selection.json")
            self.assertEqual([c["state"] for c in state["cases"]], ["failed", "complete"])
            self.assertEqual(replayed, ["gain", "loss"])
            self.assertEqual(expert_requests, [("gain", 8), ("loss", 8)])
            self.assertEqual(file_digest_tree(source), before)
            self.assertEqual(vis.read_json(output / "D01_loss.json")["metadata"]["epoch"], 500)


if __name__ == "__main__":
    unittest.main()
