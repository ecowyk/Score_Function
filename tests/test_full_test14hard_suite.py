"""Protocol validation and real queue/restart tests using synthetic nuPlan outputs."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

SOURCE = Path(__file__).resolve().parents[1] / "scripts/full_test14hard_suite.py"
spec = importlib.util.spec_from_file_location("suite", SOURCE)
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def fake_result(root, out, job, run):
    config, checkpoint = suite.model_paths(out, job["model"])
    plan = suite.read(out / "plan.json")
    cfg = dict(
        planner={
            "score_function_planner": dict(
                score_config=str(config),
                score_checkpoint=str(checkpoint),
                steps=job["K"],
                gamma=0.1,
                heading_projection=None,
                trace_dir=None,
                device="cuda",
            )
        },
        scenario_filter=dict(
            scenario_tokens=plan["tokens"], limit_total_scenarios=None, num_scenarios_per_type=None
        ),
        scenario_builder=dict(data_root=str(root / "dataset/nuplan/nuplan-v1.1/test")),
        worker={"_target_": "nuplan.planning.utils.multithreading.worker_ray.RayDistributed"},
        runner_report_file="runner_report.parquet",
    )
    (run / "code/hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create(cfg), run / "code/hydra/config.yaml")
    rows = [
        dict(
            scenario=t,
            log_name="qa_log",
            planner_name="qa",
            scenario_type="qa",
            num_scenarios=None,
            **{k: 0.8 for k in suite.METRICS},
        )
        for t in plan["tokens"]
    ]
    final = dict(rows[0], scenario="final_score", log_name=None, num_scenarios=272)
    (run / "aggregator_metric").mkdir()
    pd.DataFrame(rows + [final]).to_parquet(run / "aggregator_metric/results.parquet")
    pd.DataFrame(
        [
            dict(scenario_name=t, log_name="qa_log", planner_name="qa", succeeded=True)
            for t in plan["tokens"]
        ]
    ).to_parquet(run / "runner_report.parquet")
    return cfg


class SuiteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = self.root / "suite"
        self.out.mkdir()
        self.plan = dict(
            root=str(self.root),
            gamma=0.1,
            heading_projection=True,
            tokens=[f"{i:016x}" for i in range(272)],
            files={},
            jobs=suite.jobs(),
        )
        suite.write(self.out / "plan.json", self.plan)
        for name in suite.NAMES:
            c, w = suite.model_paths(self.out, name)
            suite.write(c, dict(reference_seed=20260920, refinement=dict(heading_projection=True)))
            w.write_bytes(name.encode())

    def tearDown(self):
        self.tmp.cleanup()

    def test_order_and_counts(self):
        self.assertEqual(len(suite.jobs()), 40)
        self.assertEqual(suite.jobs()[0]["id"], "DP_K0")
        self.assertEqual([j["K"] for j in suite.jobs()], sorted(j["K"] for j in suite.jobs()))
        self.assertEqual([j["K"] for j in suite.jobs()[-6:]], [15000] * 3 + [20000] * 3)

    def test_reject_partial_or_changed_seed(self):
        job = dict(id="S05-G_K20", model="S05-G", K=20)
        run = self.root / suite.CHALLENGE / "qa"
        cfg = fake_result(self.root, self.out, job, run)
        self.assertEqual(suite.validate_result(run, job, self.plan, self.out)["score"], 0.8)
        report_path = run / "runner_report.parquet"
        report = pd.read_parquet(report_path)
        report.loc[0, "succeeded"] = False
        report.to_parquet(report_path)
        with self.assertRaisesRegex(ValueError, "Some simulations failed"):
            suite.validate_result(run, job, self.plan, self.out)
        report.loc[0, "succeeded"] = True
        report.to_parquet(report_path)
        other = self.root / "wrong_seed.json"
        data = suite.read(suite.model_paths(self.out, "S05-G")[0])
        data["reference_seed"] += 1
        suite.write(other, data)
        cfg["planner"]["score_function_planner"]["score_config"] = str(other)
        OmegaConf.save(OmegaConf.create(cfg), run / "code/hydra/config.yaml")
        with self.assertRaisesRegex(ValueError, "weights/config differ"):
            suite.validate_result(run, job, self.plan, self.out)

    def test_real_queue_reuse_failure_continue_resume(self):
        # Two synthetic launches are enough to verify serial execution, a failure
        # continuing to the next job, then resuming only the failed one.
        skip = dict(id="S05-G_K20", model="S05-G", K=20)
        failed = dict(id="S05-L_K5", model="S05-L", K=5)
        okay = dict(id="S05-DiT_K20000", model="S05-DiT", K=20000)
        self.plan["jobs"] = [skip, failed, okay]
        suite.write(self.out / "plan.json", self.plan)
        prior = self.root / suite.CHALLENGE / "existing_G20"
        fake_result(self.root, self.out, skip, prior)
        suite.write(
            self.out / "jobs/S05-G_K20.json",
            dict(
                skip,
                state="reused",
                result_dir=str(prior),
                metrics=suite.validate_result(prior, skip, self.plan, self.out),
            ),
        )
        runner = self.root / "fake_runner.py"
        runner.write_text(
            """
import sys, os, json
from pathlib import Path
sys.path.insert(0, """
            + repr(str(Path(__file__).resolve().parent))
            + """)
from test_full_test14hard_suite import fake_result
import full_suite_test_import_placeholder
""".replace("import full_suite_test_import_placeholder", "")
            + """
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
root=Path(args['--root']); out=root/'suite'
assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1,2,3,4,5,6,7'
assert args['--gpu-fraction']=='0.15' and args['--split']=='test14-hard'
assert args['--threads']=='128' and args['--gamma']=='0.1'
name=Path(args['--config']).parent.name; k=int(args['--steps'])
assert not (name=='S05-G' and k==20), 'must never rerun reused G20'
with (out/'starts.txt').open('a') as f: f.write(f'{name}:{k}\\n')
if name=='S05-L' and (out/'fail_once').exists(): sys.exit(2)
run=root/'closed_loop_nonreactive_agents'/args['--name']
fake_result(root, out, dict(model=name,K=k), run)
print('Results: '+str(run), flush=True)
"""
        )
        for label in ("lg", "dit"):
            code = self.out / "code" / label
            code.mkdir(parents=True)
            (code / "sim_score_planner.sh").write_text(
                f'#!/bin/bash\nexec "{sys.executable}" "{runner}" "$@"\n'
            )
        (self.out / "fail_once").touch()
        command = [sys.executable, str(SOURCE), "run", "--output", str(self.out)]
        first = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertEqual(suite.read(self.out / "status.json")["state"], "finished_with_failures")
        self.assertEqual(suite.read(self.out / "jobs/S05-DiT_K20000.json")["state"], "complete")
        (self.out / "fail_once").unlink()
        second = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual(suite.read(self.out / "status.json")["state"], "complete")
        self.assertEqual(
            (self.out / "starts.txt").read_text().splitlines(),
            ["S05-L:5", "S05-DiT:20000", "S05-L:5"],
        )


if __name__ == "__main__":
    unittest.main()
