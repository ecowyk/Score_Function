"""Reader-specific evaluation guards, paired K0 reuse, and interrupted-shard recovery."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from score_function.navsim_dd.evaluate import (
    ReaderConditioner, digest_tokens, load_reusable_k0, load_score_refiner, resume_shard_prefix,
    score_pose_group, summarize, token_seed, validate_result_row, verify_reader_refiner,
)
from test_navsim_dd_reuse_k0 import baseline_fixture

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
SPEC = importlib.util.spec_from_file_location('reader_eval_runner', REPO / 'scripts/eval_navsim_dd_reader.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def result_row(token='scene-a'):
    arms = [0, *runner.KS]
    metrics = {str(k): {'score': 0.4 + k / 1000} for k in arms}
    return dict(token=token, log='log-a', seed=token_seed(20260922, token), top1_index=0,
                trajectories={str(k): [[k / 100.0, 0.0, 0.0]] * 8 for k in arms}, metrics=metrics,
                delta={str(k): {'score': metrics[str(k)]['score'] - metrics['0']['score']} for k in arms},
                native_ms=0.0, conditioner_ms=1.0, refine_ms=2.0, checks={'k0_reused': True})


def source_row(row):
    return {row['token']: dict(trajectory=row['trajectories']['0'], metrics=row['metrics']['0'],
                              seed=row['seed'], top1_index=row['top1_index'], log=row['log'])}


class ReaderEvaluationTests(unittest.TestCase):
    def test_full500_training_and_final_raw_are_required(self):
        cfg = runner.expected_config()
        history = [{'epoch': epoch} for epoch in range(1, 501)]
        run = dict(config=cfg, global_batch=512, protocol={'world_size': 8, 'device_type': 'cuda'})
        runner.validate_training(run, history)
        for replacement in (history[:-1], [{'epoch': 500}], history[::-1]):
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                runner.validate_training(run, replacement)
        bad_config = {**cfg, 'lr_schedule_epochs': 100}
        with self.assertRaisesRegex(ValueError, 'configuration'):
            runner.validate_training({**run, 'config': bad_config}, history)
        saved = dict(format='navsim-dd-reader-score-v1', epoch=500, weight_kind='raw', sigma=0.1,
                     gamma=0.1, training_precision='16-mixed', model=cfg['model'], reader_spec={'fixture': 1},
                     state_dict={'fixture': torch.ones(1)}, conditioner_identity={
                         'condition': 'dd-backbone512-status-reader-v1', 'reader_assets_sha256': 'asset'})
        runner.validate_checkpoint(saved)
        for change in ({'epoch': 450}, {'weight_kind': 'ema'}, {'format': 'navsim-dd-score-v1'},
                       {'reader_spec': None}, {'conditioner_identity': {}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.validate_checkpoint({**saved, **change})

    def test_worker_requests_all_five_arms_and_resume_with_saved_k0(self):
        command = runner.worker_command(Path('/root/project'), Path('/eval'), Path('/final_raw.pt'), Path('/k0'), 4)
        self.assertEqual(command[command.index('--ks') + 1:command.index('--gamma')], ['5', '10', '15', '20', '30'])
        self.assertEqual(command[command.index('--reuse-k0-from') + 1], '/k0')
        self.assertEqual(command[command.index('--device') + 1], 'cuda:4')
        self.assertIn('--resume', command)
        self.assertIn('--verify-native', command)

    def test_conditioner_exposes_uncompressed_latent_and_status_without_sampling(self):
        latent, status = torch.randn(1, 512, 8, 8), torch.randn(1, 256)
        class Agent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._transfuser_model = SimpleNamespace(
                    _backbone=lambda camera, lidar: (torch.zeros(1), latent, None),
                    _status_encoding=lambda state: status)
            def forward(self, *_):
                raise AssertionError('Native sampler must not run')
        c, g = ReaderConditioner(Agent()).encode_features(dict(camera_feature=None, lidar_feature=None, status_feature=None))
        self.assertIs(c, latent)
        self.assertIs(g, status)

    def test_versioned_checkpoint_dispatch_and_native_identity_check(self):
        fake = SimpleNamespace(require_native_conditioner=lambda identity: self.assertEqual(identity, {'native': 1}))
        with (patch('torch.load', return_value={'format': 'navsim-dd-reader-score-v1'}),
              patch('score_function.navsim_dd.reader_runtime.DDReaderRefiner.from_checkpoint', return_value=fake)):
            model, uses_reader = load_score_refiner('/checkpoint', {'native': 1}, 'cpu')
        self.assertIs(model, fake)
        self.assertTrue(uses_reader)
        with patch('torch.load', return_value={'format': 'unknown'}), self.assertRaisesRegex(ValueError, 'Unsupported'):
            load_score_refiner('/checkpoint', {}, 'cpu')

    def test_native_parity_checks_reader_geometry_and_weights(self):
        assets = {'spec': {'extent': 32}, 'state_dict': {'weight': torch.ones(1)}}
        reader = SimpleNamespace(spec=assets['spec'], state_dict=lambda: {'weight': torch.ones(1)})
        model = SimpleNamespace(reader_spec=assets['spec'], reader=reader)
        refiner = SimpleNamespace(model=model)
        with (patch('score_function.navsim_dd.reader.export_reader_assets', return_value=assets),
              patch('score_function.navsim_dd.reader.verify_native_reader', return_value={'error': 0}) as check):
            actual = verify_reader_refiner(None, refiner, None, None, None)
            self.assertTrue(actual['frozen_reader_weights_match_native'])
            check.assert_called_once()
            model.reader_spec = {'extent': 64}
            with self.assertRaisesRegex(ValueError, 'geometry'):
                verify_reader_refiner(None, refiner, None, None, None)
            model.reader_spec = assets['spec']
            reader.state_dict = lambda: {'weight': torch.zeros(1)}
            with self.assertRaisesRegex(ValueError, 'weights'):
                verify_reader_refiner(None, refiner, None, None, None)

    def test_all_new_arms_share_raw_and_saved_k0_is_not_rescored(self):
        seen = []
        class Refiner:
            def refine_many(self, poses, context, status, steps, gamma=None):
                self_steps = steps[:]
                self.asserted = True
                seen.append(('steps', self_steps))
                return {k: poses + k / 100 for k in steps}
        def score(poses):
            seen.append(('score', float(poses[0, 0])))
            return {'score': 0.9}
        trajectories, metrics, deltas, _ = score_pose_group(
            torch.zeros(1, 8, 3), [0, *runner.KS], Refiner(), None, None, .1, score, 'cpu',
            baseline_metrics={'score': .4})
        self.assertEqual(seen[0], ('steps', [0, *runner.KS]))
        self.assertEqual(len(seen), 6)
        self.assertEqual(trajectories[0], [[0., 0., 0.]] * 8)
        self.assertEqual(metrics[0], {'score': .4})
        self.assertEqual(deltas[30], {'score': .5})

    def test_partial_last_line_is_trimmed_after_valid_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = result_row()
            valid = json.dumps(row) + '\n'
            path = root / 'scenes.jsonl.tmp'
            path.write_text(valid + '{"token":')
            resumed = resume_shard_prefix(root, ['scene-a', 'scene-b'], {'scene-a': 'log-a'},
                                          [0, *runner.KS], 20260922, source_row(row))
            self.assertEqual(len(resumed), 1)
            self.assertEqual(path.read_text(), valid)

    def test_invalid_complete_row_is_rejected_without_mutating_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = result_row()
            for name, edit in (
                ('seed', lambda row: row.update(seed=0)),
                ('trajectory', lambda row: row['trajectories']['10'][0].append(3)),
                ('k0', lambda row: row['trajectories']['0'][0].__setitem__(0, 2)),
                ('delta', lambda row: row['delta']['5'].update(score=1)),
                ('nonfinite', lambda row: row['metrics']['30'].update(score=float('nan'))),
            ):
                row = copy.deepcopy(original)
                edit(row)
                path = root / 'scenes.jsonl.tmp'
                path.write_text(json.dumps(row) + '\n{"token":')
                before = path.read_bytes()
                with self.subTest(name=name), self.assertRaises(ValueError):
                    resume_shard_prefix(root, ['scene-a'], {'scene-a': 'log-a'},
                                        [0, *runner.KS], 20260922, source_row(original))
                self.assertEqual(path.read_bytes(), before)

    def test_crash_after_jsonl_rename_recovers_without_repeating_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = result_row()
            (root / 'scenes.jsonl').write_text(json.dumps(row) + '\n')
            resumed = resume_shard_prefix(root, ['scene-a'], {'scene-a': 'log-a'},
                                          [0, *runner.KS], 20260922, source_row(row))
            self.assertEqual(len(resumed), 1)
            self.assertFalse((root / 'scenes.jsonl').exists())
            self.assertTrue((root / 'scenes.jsonl.tmp').exists())

    def test_failed_training_preflight_starts_no_workers_or_eval_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (patch.object(runner, 'preflight', side_effect=ValueError('incomplete')),
                  patch.object(runner.subprocess, 'Popen') as spawn):
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    runner.run(root)
                spawn.assert_not_called()
            self.assertFalse((root / 'dataset/navsim/exp' / runner.LATEST).exists())
            self.assertFalse((root / 'dataset/navsim/exp' / runner.EVAL_DIRECTORY).exists())

    def test_completed_shard_reuse_checks_exact_k0_and_trajectories_not_just_means(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / 'baseline'
            baseline.mkdir()
            native, selected = baseline_fixture(baseline)
            _, provenance = load_reusable_k0(baseline, native, selected, needed=[])
            info = {'run': {key: native[key] for key in (
                'split', 'seed', 'identity', 'token_digest', 'count', 'data_root', 'max_scenes', 'metric_cache')},
                    'provenance': provenance}
            output = root / 'new'
            part = output / 'part-0000-of-0002'
            part.mkdir(parents=True)
            run = {**info['run'], 'ks': [0, *runner.KS], 'gamma': .1, 'num_shards': 2,
                   'score_checkpoint_sha256': 'new-checkpoint', 'k0_reuse': provenance,
                   'score_architecture': 'dd_scene_reader_score'}
            mark = {'run': run, 'shard_index': 0, 'shard_tokens': ['scene-a']}
            row = result_row()
            for name in ('run.json', 'complete.json'):
                (part / name).write_text(json.dumps(mark))
            (part / 'summary.json').write_text(json.dumps(summarize([row])))
            (part / 'scenes.jsonl').write_text(json.dumps(row) + '\n')
            before = {str(p): p.read_bytes() for p in baseline.rglob('*') if p.is_file()}
            with patch.object(runner.base, 'SHARDS', 2):
                self.assertTrue(runner.validate_completed_shard(output, 0, 'new-checkpoint', info))
                for key, edit in (
                    ('k0', lambda x: x['trajectories']['0'][0].__setitem__(0, 12)),
                    ('refined', lambda x: x['trajectories']['15'][0].append(4)),
                    ('delta', lambda x: x['delta']['20'].update(score=100)),
                ):
                    changed = copy.deepcopy(row)
                    edit(changed)
                    (part / 'scenes.jsonl').write_text(json.dumps(changed) + '\n')
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        runner.validate_completed_shard(output, 0, 'new-checkpoint', info)
                wrong = {**mark, 'shard_tokens': ['scene-b']}
                for name in ('run.json', 'complete.json'):
                    (part / name).write_text(json.dumps(wrong))
                with self.assertRaisesRegex(ValueError, 'token assignment'):
                    runner.validate_completed_shard(output, 0, 'new-checkpoint', info)
            self.assertEqual(before, {str(p): p.read_bytes() for p in baseline.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()
