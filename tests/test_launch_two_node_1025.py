"""Fresh-controller regressions for nuPlan path interpolation on both nodes."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import launch_two_node_1025 as launcher


RESOLVE_SAVED_CONFIG = r'''
import json
from omegaconf import OmegaConf
cfg = OmegaConf.create({
    "planner": "${oc.env:DP_DIR}",
    "devkit": "${oc.env:NUPLAN_DEVKIT_ROOT}",
    "data": "${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/test",
    "maps": "${oc.env:NUPLAN_MAPS_ROOT}",
    "experiments": "${oc.env:NUPLAN_EXP_ROOT}",
})
print(json.dumps(OmegaConf.to_container(cfg, resolve=True)))
'''


class LauncherEnvironmentTests(unittest.TestCase):
    def check_controller_environment(self, role, stale):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "server-root"
            environment_name = "urbandriver" if role == "baselines" else "diffusion_planner"
            python = root / "envs" / environment_name / "bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
            output = root / "queued" / role
            expected = {
                "planner": str(root / "Diffusion-Planner"),
                "devkit": str(root / "nuplan-devkit"),
                "data": str(root / "dataset/nuplan/nuplan-v1.1/test"),
                "maps": str(root / "dataset/nuplan/maps"),
                "experiments": str(root / "dataset/nuplan/exp"),
            }
            variables = ("DP_DIR", "NUPLAN_DEVKIT_ROOT", "NUPLAN_DATA_ROOT",
                         "NUPLAN_MAPS_ROOT", "NUPLAN_EXP_ROOT")
            parent_environment = {key: value for key, value in os.environ.items()
                                  if key not in variables}
            if stale:
                parent_environment.update({key: "/another-node/stale/" + key for key in variables})
            preparations = []

            def run_prepare(command, *, cwd, env, check):
                # Exercise the exact launch boundary where saved Hydra configs
                # were failing, using a real fresh OmegaConf interpreter. No
                # nuPlan data, GPU worker or actual queue is started.
                self.assertEqual(command[0], str(python))
                wanted_script = ("evaluate_random_baselines_1025.py" if role == "baselines"
                                 else "prepare_energy_aligned_1025.py")
                self.assertEqual(Path(command[2]).name, wanted_script)
                self.assertTrue(check)
                resolved = subprocess.run(
                    [sys.executable, "-c", RESOLVE_SAVED_CONFIG], cwd=cwd, env=env,
                    check=True, capture_output=True, text=True,
                )
                self.assertEqual(json.loads(resolved.stdout), expected)
                preparations.append(dict(env))
                return subprocess.CompletedProcess(command, 0)

            detached = mock.Mock(pid=43210)
            detached.poll.return_value = None
            popen = mock.Mock(return_value=detached)
            # Replace only the launcher's module reference; retain real
            # subprocess.Popen for the fresh interpreter above.
            subprocess_proxy = SimpleNamespace(
                run=run_prepare, Popen=popen,
                DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT,
            )
            with mock.patch.dict(os.environ, parent_environment, clear=True), \
                 mock.patch.object(launcher.sys, "argv", [
                     "launch_two_node_1025.py", role, "--root", str(root), "--output", str(output)
                 ]), \
                 mock.patch.object(launcher, "subprocess", subprocess_proxy), \
                 mock.patch.object(launcher.time, "sleep"), \
                 contextlib.redirect_stdout(io.StringIO()):
                launcher.main()
                self.assertEqual(dict(os.environ), parent_environment)

            self.assertEqual(len(preparations), 1)
            popen.assert_called_once()
            launch = popen.call_args.kwargs
            self.assertEqual(launch["env"], preparations[0])
            self.assertEqual(launch["env"]["CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")
            self.assertEqual(launch["stdin"], subprocess.DEVNULL)
            self.assertTrue(launch["start_new_session"])

    def test_baseline_prepare_resolves_paths_without_inherited_nuplan_environment(self):
        self.check_controller_environment("baselines", stale=False)

    def test_energy_prepare_resolves_paths_without_inherited_nuplan_environment(self):
        self.check_controller_environment("energy", stale=False)

    def test_baseline_prepare_replaces_stale_node_paths(self):
        self.check_controller_environment("baselines", stale=True)

    def test_energy_prepare_replaces_stale_node_paths(self):
        self.check_controller_environment("energy", stale=True)


if __name__ == "__main__":
    unittest.main()
