"""Queue contract tests: no model downloads or GPUs required."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
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

    def test_b64_recipe_isolates_training_and_evaluates_its_epoch_40_ema(self):
        args = queue.parse_args(['run', '--recipe', 'b64-e40', '--root', str(self.root)])
        cfg = queue.effective_config(args)
        self.assertEqual((args.workers, args.microbatch_size, cfg['global_batch_size']), (8, 8, 64))
        self.assertEqual((cfg['epochs'], cfg['checkpoint_start_epoch'], cfg['checkpoint_every_epochs']), (40, 5, 5))
        self.assertEqual(cfg['model'], queue.effective_config(self.args)['model'])
        for name in ('cache', 'source_cache', 'checkpoint', 'metric_cache'):
            self.assertEqual(getattr(args, name), getattr(self.args, name))
        for name in ('training', 'evaluation', 'output'):
            self.assertNotEqual(getattr(args, name), getattr(self.args, name))
        self.write(self.args.training/'run.json', {})
        command = queue.train_command(args, args.workers)
        self.assertNotIn('--resume', command)
        self.assertIn('--nproc_per_node=8', command)
        self.assertEqual(command[command.index('--config')+1], str(args.config))
        helper = queue.parse_args(queue.helper_command(args, '_preflight')[3:])
        self.assertEqual(queue.effective_config(helper), cfg)
        self.assertEqual((helper.recipe, helper.output), (args.recipe, args.output))
        evaluate = queue.eval_command(args, args.workers)
        for flag, expected in [('--mode', 'refine'), ('--weight-kind', 'ema'), ('--expected-epoch', '40'),
                               ('--score-checkpoint', str(args.training/'final_ema.pt')),
                               ('--baseline', str(args.baseline))]:
            self.assertEqual(evaluate[evaluate.index(flag)+1], expected)
        i = evaluate.index('--ks')
        self.assertEqual(evaluate[i+1:i+6], ['5', '10', '15', '20', '30'])

    def test_b64_queue_uses_prepared_cache_and_skips_old_asset_stages(self):
        args = queue.parse_args(['run', '--recipe', 'b64-e40', '--root', str(self.root),
                                 '--python', sys.executable])
        self.write(args.cache/'TRAINING_READY.json', {})
        runner = queue.Queue(args)
        stages = []
        with mock.patch.object(runner, 'execute', side_effect=lambda stage, *a: stages.append(stage)), \
                mock.patch.object(runner, 'gpu', side_effect=lambda stage, *a, **k: stages.append(stage)), \
                mock.patch.object(runner, 'wait_baseline', side_effect=lambda: stages.append('wait_native')), \
                mock.patch.object(queue, 'training_complete', side_effect=[False, True]):
            runner.run()
        self.assertEqual(stages, ['wait_native', 'verify_cache', 'preflight', 'train', 'evaluate', 'verify_evaluation'])
        self.assertEqual((runner.state['epochs'], runner.state['status']), (40, 'complete'))

    def test_b64_reports_a_missing_native_handoff_without_waiting(self):
        args = queue.parse_args(['run', '--recipe', 'b64-e40', '--root', str(self.root)])
        self.assertEqual(args.baseline, self.args.baseline/'run_native_stage2_navtrain')
        with mock.patch.object(queue.time, 'sleep') as sleep:
            with self.assertRaisesRegex(FileNotFoundError, 'native K0'):
                queue.Queue(args).wait_baseline()
        sleep.assert_not_called()

    def test_b64_inherits_asset_paths_only_from_the_legacy_plan(self):
        self.args.baseline = self.root/'external-native'/'run_native_stage2_navtrain'
        self.args.navtest_vlm_cache = self.root/'test-vlm-release'
        self.args.cache = self.root/'existing-training-cache'
        self.args.gpu_lock_dir = self.root/'shared-leases'
        plan = self.args.output/'plan.json'
        self.write(plan, queue.plan_for(self.args))
        override = self.root/'explicit-override'
        args = queue.parse_args(['run', '--root', str(self.root), '--recipe', 'b64-e40',
                                 '--assets-from-plan', str(plan), '--metric-cache', str(override)])
        for name in ('baseline', 'navtest_vlm_cache', 'cache', 'gpu_lock_dir'):
            self.assertEqual(getattr(args, name), getattr(self.args, name))
        for name in ('training', 'evaluation', 'output', 'config'):
            self.assertNotEqual(getattr(args, name), getattr(self.args, name))
        self.assertEqual((args.metric_cache, args.workers, args.microbatch_size), (override, 8, 8))

    def test_external_handoff_selects_existing_test_vlm_without_moving_it(self):
        baseline = self.root/'external-native'
        wrapper = self.root/'already-downloaded-test-wrapper'
        self.write(baseline/'reproduction_ready.json',
                   {'schema': 'chainflow-native-reproduction-v1', 'vlm_cache': str(wrapper)})
        args = queue.parse_args(['run', '--root', str(self.root), '--baseline', str(baseline)])
        self.assertEqual(args.navtest_vlm_cache, wrapper)
        self.assertNotEqual(args.vlm_cache, wrapper)
        command = queue.cache_command(args, 1, check=True)
        self.assertEqual(command[command.index('--vlm-cache')+1], str(args.vlm_cache))
        self.assertEqual(command[command.index('--navtest-vlm-cache')+1], str(wrapper))
        command = queue.eval_command(args, 4)
        self.assertEqual(command[command.index('--vlm-cache')+1], str(wrapper))
        self.assertFalse(wrapper.exists())

    def test_external_ready_marker_does_not_require_summary_json(self):
        self.write(self.args.baseline/'reproduction_ready.json', {'schema': 'chainflow-native-reproduction-v1'})
        runner = queue.Queue(self.args)
        with mock.patch.object(queue.time, 'sleep') as sleep, mock.patch.object(runner, 'execute') as execute:
            runner.wait_baseline()
        sleep.assert_not_called()
        self.assertEqual(execute.call_args.args[0], 'verify_native')

    def test_preparation_reuses_complete_training_features_and_never_reinstalls(self):
        from score_function.navsim_chainflow import assets
        complete = {'present': {'train': 85109, 'val': 18179}, 'complete': True, 'missing_count': 0}
        cache_module = types.ModuleType('score_function.navsim_chainflow.cache')
        cache_module.source_tasks = mock.Mock(return_value=({}, []))
        native_module = types.ModuleType('score_function.navsim_chainflow.native')
        native_module.conditioner_identity = mock.Mock(return_value={})
        self.write(self.args.source_cache/'manifest.json', {})
        with mock.patch.dict(sys.modules, {cache_module.__name__: cache_module, native_module.__name__: native_module}), \
                mock.patch.object(queue, 'trainval_vlm_coverage', return_value=complete), \
                mock.patch.object(assets, 'fetch') as fetch, mock.patch.object(queue.os, 'setpriority'):
            queue.prepare_assets(self.args)
        fetch.assert_not_called()
        native_module.conditioner_identity.assert_called_once()
        self.assertTrue(queue.read_json(self.args.output/'asset_preparation.json')['complete'])

    def test_preparation_fetches_only_trainval_and_checks_coverage_afterwards(self):
        from score_function.navsim_chainflow import assets
        missing = {'present': {'train': 0, 'val': 0}, 'complete': False, 'missing_count': 103288}
        complete = {'present': {'train': 85109, 'val': 18179}, 'complete': True, 'missing_count': 0}
        cache_module = types.ModuleType('score_function.navsim_chainflow.cache')
        cache_module.source_tasks = mock.Mock(return_value=({}, []))
        native_module = types.ModuleType('score_function.navsim_chainflow.native')
        native_module.conditioner_identity = mock.Mock(return_value={})
        self.write(self.args.source_cache/'manifest.json', {})
        with mock.patch.dict(sys.modules, {cache_module.__name__: cache_module, native_module.__name__: native_module}), \
                mock.patch.object(queue, 'trainval_vlm_coverage', side_effect=[missing, complete]), \
                mock.patch.object(assets, 'fetch') as fetch, mock.patch.object(queue.os, 'setpriority'):
            queue.prepare_assets(self.args)
        self.assertEqual(fetch.call_args.kwargs['only'], 'trainval')
        self.assertEqual(fetch.call_args.kwargs['scope'], 'all')
        self.assertEqual(fetch.call_args.args[1], self.args.root)

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
        for recipe in ('v1', 'b64-e40'):
            with self.subTest(recipe=recipe):
                args = queue.parse_args(['run', '--root', str(self.root), '--recipe', recipe])
                self.check_training_completion(args)

    def check_training_completion(self, args):
        self.assertFalse(queue.training_complete(args))
        metadata = {'x': 'same-cache'}
        fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        cfg = queue.effective_config(args)
        self.write(args.cache/'manifest.json', metadata)
        self.write(args.training/'run.json', {'signature': 'sig', 'config': cfg,
                                              'protocol': {'cache_fingerprint': fingerprint}})
        marker = {'signature': 'sig', 'epochs': cfg['epochs']}
        for kind in ('raw', 'ema'):
            path = args.training/f'final_{kind}.pt'
            path.write_bytes(kind.encode())
            marker[f'final_{kind}_sha256'] = queue.sha256(path)
        self.write(args.training/'TRAINING_COMPLETE.json', marker)
        self.assertTrue(queue.training_complete(args))
        (args.training/'final_ema.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            queue.training_complete(args)

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
        self.assertEqual(stages, ['wait_native', 'check_data', 'cache', 'preflight', 'train', 'evaluate'])
        self.assertEqual(runner.state['status'], 'complete')

    def test_optional_download_stage_runs_after_native_check_before_data_audit(self):
        self.args.prepare_assets = True
        runner = queue.Queue(self.args)
        stages = []
        with mock.patch.object(runner, 'execute', side_effect=lambda stage, *a: stages.append(stage)), \
                mock.patch.object(runner, 'gpu', side_effect=lambda stage, *a, **k: stages.append(stage)), \
                mock.patch.object(runner, 'wait_baseline', side_effect=lambda: stages.append('wait_native')), \
                mock.patch.object(queue, 'training_complete', side_effect=[False, True]):
            runner.run()
        self.assertEqual(stages[:3], ['wait_native', 'prepare_assets', 'check_data'])

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
