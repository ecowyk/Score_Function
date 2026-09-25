"""Opt-in small real-DB interface test, isolated from production training.

SCORE_FUNCTION_REAL_ROOT must contain Diffusion-Planner, nuplan-devkit and
dataset/nuplan/{nuplan-v1.1/mini,maps}. A temporary allowlist is used exclusively
for this engineering check; it is not the production training split.
"""

import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from score_function.utils.train_utils import atomic_write, file_hash, read_json

REAL_ROOT = os.environ.get("SCORE_FUNCTION_REAL_ROOT")


@unittest.skipUnless(REAL_ROOT, "Set SCORE_FUNCTION_REAL_ROOT for the small real-DB test")
class RealPreprocessingTests(unittest.TestCase):
    def test_global_cap_resume_and_reuse_on_two_real_databases(self):
        source = Path(REAL_ROOT)
        mini = source / "dataset/nuplan/nuplan-v1.1/mini"
        files = sorted(mini.glob("*.db"))[:2]
        self.assertEqual(len(files), 2)
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="score_global_real_") as tmp:
            root = Path(tmp)
            cfg = read_json(project / "configs/score_function.json")
            cfg["paths"].update(
                root=str(source),
                database_dir=str(mini),
                maps_dir=str(source / "dataset/nuplan/maps"),
                train_log_allowlist=str(root / "allow.json"),
                data_output=str(root / "first"),
                manifest=str(root / "first/features_manifest.json"),
            )
            cfg["data"].update(total_scenarios=6, preprocess_workers=2)
            cfg["runtime"]["device"] = "cpu"
            atomic_write(root / "allow.json", [path.stem for path in files])
            path = root / "config.json"
            atomic_write(path, cfg)

            def run(config_path):
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "score_function",
                        "prepare",
                        "--config",
                        str(config_path),
                    ],
                    cwd=project,
                    env={**os.environ, "TQDM_DISABLE": "1"},
                    check=True,
                )

            run(path)
            status = read_json(root / "first/preprocess_status.json")
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["selected_scenarios"], 6)
            self.assertEqual(status["global_cap"], 6)
            records = read_json(root / "first/features_manifest.json")
            self.assertGreater(len(records), 0)
            self.assertLessEqual(len(records), 6)
            features = {
                item["feature"]: (
                    file_hash(item["feature"]),
                    Path(item["feature"]).stat().st_mtime_ns,
                )
                for item in records
            }
            index = root / "first/selection/index.json"
            selection_hash = file_hash(index)
            run(path)
            self.assertEqual(file_hash(index), selection_hash)
            self.assertEqual(
                features,
                {name: (file_hash(name), Path(name).stat().st_mtime_ns) for name in features},
            )
            reuse = copy.deepcopy(cfg)
            reuse["paths"].update(
                data_output=str(root / "reuse"),
                manifest=str(root / "reuse/features_manifest.json"),
                reuse_features_from=str(root / "first"),
            )
            atomic_write(root / "reuse_config.json", reuse)
            run(root / "reuse_config.json")
            reused = read_json(root / "reuse/preprocess_status.json")
            self.assertEqual(reused["reused_samples"], len(records))
            self.assertEqual(reused["selected_scenarios"], 6)
            self.assertEqual(
                {row["token"] for row in read_json(root / "reuse/features_manifest.json")},
                {row["token"] for row in records},
            )
            # Verify the next production stage accepts real dotted log names too.
            from score_function.data_process.feature_cache import build_cache
            from score_function.utils.config import configure_runtime

            cfg["paths"]["cache"] = str(root / "cache/index.json")
            configure_runtime(cfg, require_cuda=False)
            build_cache(cfg)
            self.assertEqual(read_json(root / "cache/index.json")["metadata"]["samples"], 6)
            print(
                {
                    "real_DBs": 2,
                    "selected_global": 6,
                    "valid_features": len(records),
                    "reused_features": reused["reused_samples"],
                    "resume_rewrites": 0,
                },
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
