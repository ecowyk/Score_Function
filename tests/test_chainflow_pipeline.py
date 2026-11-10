"""Queue contract tests: no model downloads or GPUs required."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/run_navsim_chainflow.py'
spec = importlib.util.spec_from_file_location('chainflow_pipeline', SCRIPT)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.args = queue.parse_args(['run', '--root', str(self.root), '--python', sys.executable])
        self.args.output.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path, value):
        queue.atomic_json(path, value)

    def test_default_conservative_resource_and_training_plan(self):
        self.assertEqual(self.args.gpu_policy, 'wait-idle')
        self.assertEqual(self.args.gpu_budget_gb, 12)
        self.assertEqual(self.args.gpu_reserve_gb, 4)
        cfg = queue.effective_config(self.args)
        self.assertEqual((cfg['epochs'], cfg['checkpoint_every_epochs']), (500, 50))
        self.assertEqual(cfg['global_batch_size'] // (cfg['microbatch_size']*self.args.workers), 8)
        self.assertEqual(self.args.baseline, self.root/'dataset/navsim/exp/chainflow_native_navtest_v1')
        self.assertEqual(self.args.source_cache, self.root/'dataset/navsim/exp/dd_score_cache_v1')
        self.args.microbatch_size = 17
        with self.assertRaisesRegex(ValueError, 'divide'):
            queue.effective_config(self.args)

    def test_commands_preserve_global_batch_and_exact_final_ema(self):
        command = queue.train_command(self.args, 4)
        self.assertEqual(command[0], sys.executable)
        self.assertIn('torch.distributed.run', command)
        self.assertIn('--nproc_per_node=4', command)
        self.assertNotIn('--resume', command)
        self.write(self.args.training/'run.json', {})
        self.assertIn('--resume', queue.train_command(self.args, 1))
        evaluate = queue.eval_command(self.args, 4)
        self.assertIn('--gpu-lease-held', evaluate)
        self.assertIn(str(self.args.training/'final_ema.pt'), evaluate)
        i = evaluate.index('--ks')
        self.assertEqual(evaluate[i+1:i+6], ['5','10','15','20','30'])
        self.assertEqual(evaluate[evaluate.index('--mode')+1], 'refine')

    def test_plan_refuses_configuration_or_code_rebinding(self):
        path = self.args.output/'plan.json'
        plan = queue.plan_for(self.args)
        queue.bind_plan(path, plan)
        queue.bind_plan(path, plan)
        modified = {**plan, 'ks': [20]}
        with self.assertRaisesRegex(ValueError, 'changed'):
            queue.bind_plan(path, modified)
        self.assertEqual(queue.read_json(path), plan)

    def test_resumed_world_is_fixed_even_with_auto(self):
        self.write(self.args.training/'run.json', {'protocol': {'world_size': 2}, 'config': {'microbatch_size': 16}})
        with self.assertRaisesRegex(ValueError, '2 GPUs'):
            queue.cached_world(self.args, 'train')
        self.args.workers = 'auto'
        self.assertEqual(queue.cached_world(self.args, 'train'), 2)
        self.args.microbatch_size = 32
        with self.assertRaisesRegex(ValueError, 'microbatch'):
            queue.cached_world(self.args, 'train')

    def test_training_completion_requires_config_cache_and_checkpoint_hashes(self):
        self.assertFalse(queue.training_complete(self.args))
        metadata = {'x': 'same-cache'}
        fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        self.write(self.args.cache/'manifest.json', metadata)
        self.write(self.args.training/'run.json', {'signature': 'sig', 'config': queue.effective_config(self.args),
                                                   'protocol': {'cache_fingerprint': fingerprint}})
        marker = {'signature': 'sig', 'epochs': 500}
        for kind in ('raw', 'ema'):
            path = self.args.training/f'final_{kind}.pt'
            path.write_bytes(kind.encode())
            marker[f'final_{kind}_sha256'] = queue.sha256(path)
        self.write(self.args.training/'TRAINING_COMPLETE.json', marker)
        self.assertTrue(queue.training_complete(self.args))
        (self.args.training/'final_ema.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            queue.training_complete(self.args)

    def test_wait_native_never_acquires_gpu_and_validates_after_publication(self):
        runner = queue.Queue(self.args)
        def publish(_):
            self.write(self.args.baseline/'summary.json', {'complete': True})
        with mock.patch.object(queue.time, 'sleep', side_effect=publish), \
                mock.patch.object(runner, 'execute') as execute, mock.patch.object(runner, 'gpu') as gpu:
            runner.wait_baseline()
        gpu.assert_not_called()
        self.assertEqual(execute.call_args.args[0], 'verify_native')
        self.assertIn('_verify-native', execute.call_args.args[1])

    def test_order_and_external_baseline_no_native_resampling(self):
        runner = queue.Queue(self.args)
        stages = []
        def execute(stage, command, env=None):
            stages.append(stage)
            self.assertNotIn('baseline', command)
        def gpu(stage, builder, workers=None):
            stages.append(stage)
            builder(workers or 4)
        with mock.patch.object(runner, 'execute', side_effect=execute), \
                mock.patch.object(runner, 'gpu', side_effect=gpu), \
                mock.patch.object(runner, 'wait_baseline', side_effect=lambda: stages.append('wait_native')), \
                mock.patch.object(queue, 'training_complete', side_effect=[False, True]):
            runner.run()
        self.assertEqual(stages, ['check_data', 'wait_native', 'cache', 'preflight', 'train', 'evaluate'])
        self.assertEqual(runner.state['status'], 'complete')

    def test_completed_evaluation_is_verified_without_waiting_for_gpu(self):
        runner = queue.Queue(self.args)
        self.write(self.args.cache/'TRAINING_READY.json', {})
        self.write(self.args.evaluation/'summary.json', {})
        stages = []
        with mock.patch.object(runner, 'execute', side_effect=lambda stage, *a: stages.append(stage)), \
                mock.patch.object(runner, 'wait_baseline'), mock.patch.object(runner, 'gpu') as gpu, \
                mock.patch.object(queue, 'training_complete', return_value=True):
            runner.run()
        gpu.assert_not_called()
        self.assertEqual(stages, ['check_data', 'verify_cache', 'verify_evaluation'])

    def test_failed_child_preserves_log_and_clears_pid(self):
        runner = queue.Queue(self.args)
        with self.assertRaisesRegex(RuntimeError, 'exited 7'):
            runner.execute('test', [sys.executable, '-c', 'print("failure evidence", flush=True); raise SystemExit(7)'])
        self.assertIn('failure evidence', (self.args.output/'test.log').read_text())
        self.assertIsNone(runner.state['child_pid'])

    def test_interruption_signals_only_owned_process_group(self):
        runner = queue.Queue(self.args)
        child = mock.Mock(pid=7654321)
        child.poll.return_value = None
        with mock.patch.object(queue.subprocess, 'Popen', return_value=child), \
                mock.patch.object(queue.time, 'sleep', side_effect=KeyboardInterrupt), \
                mock.patch.object(queue.os, 'killpg') as kill:
            with self.assertRaises(KeyboardInterrupt):
                runner.execute('test', ['example'])
        kill.assert_called_once_with(child.pid, queue.signal.SIGTERM)
        child.wait.assert_called_once_with(timeout=20)
        self.assertIsNone(runner.state['child_pid'])

    def test_status_uses_only_standard_library(self):
        result = subprocess.run([sys.executable, '-S', str(SCRIPT), 'status', '--root', str(self.root)],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('not_started', result.stdout)
        self.assertIn('pipeline lock: idle', result.stdout)
        self.assertFalse((self.args.output/'pipeline.lock').exists())


if __name__ == '__main__':
    unittest.main()
