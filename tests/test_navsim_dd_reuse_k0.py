"""Reuse a complete paired baseline and score only the newly refined arm."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from score_function.navsim_dd.evaluate import (
    digest_tokens, load_reusable_k0, score_pose_group, token_seed,
)


def baseline_fixture(root):
    selected = ['scene-a', 'scene-b']
    run = dict(split='navtest', seed=20260922, identity={'checkpoint_sha256': 'dd-fixture'},
               token_digest=digest_tokens(selected), count=2, data_root='/data', max_scenes=None,
               metric_cache='/metric', ks=[0, 20], gamma=0.1, num_shards=2,
               score_checkpoint_sha256='old-score')
    for index, token in enumerate(selected):
        directory = root / f'part-{index:04d}-of-0002'
        directory.mkdir()
        manifest = dict(run=run, shard_index=index, shard_tokens=[token])
        for name in ('run.json', 'complete.json'):
            (directory / name).write_text(json.dumps(manifest))
        row = dict(token=token, seed=token_seed(run['seed'], token), top1_index=index,
                   trajectories={'0': [[float(index), 0.0, 0.0]] * 8},
                   metrics={'0': {'score': 0.4 + 0.2 * index}})
        (directory / 'scenes.jsonl').write_text(json.dumps(row) + '\n')
    (root / 'summary.json').write_text(json.dumps({'count': 2, 'by_k': {'0': {'mean': {'score': 0.5}}}}))
    return run, selected


class ReuseK0Tests(unittest.TestCase):
    def test_compatible_new_score_can_reuse_same_raw_and_metrics_without_writing_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, tokens = baseline_fixture(root)
            before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            # K0 is independent of the score checkpoint, gamma and sharding.
            expected = {**run, 'score_checkpoint_sha256': 'new-score', 'gamma': 0.2, 'num_shards': 1}
            records, provenance = load_reusable_k0(root, expected, tokens, [tokens[1]])
            self.assertEqual(list(records), [tokens[1]])
            self.assertEqual(records[tokens[1]]['trajectory'], [[1.0, 0.0, 0.0]] * 8)
            self.assertAlmostEqual(records[tokens[1]]['metrics']['score'], 0.6)
            self.assertEqual(len(provenance['files_sha256']), 4)
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_changed_baseline_protocol_missing_shard_and_summary_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, tokens = baseline_fixture(root)
            for key, value in [('seed', 17), ('identity', {'checkpoint_sha256': 'other'}),
                               ('metric_cache', '/other'), ('token_digest', 'different')]:
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    load_reusable_k0(root, {**run, key: value}, tokens)
            summary = root / 'summary.json'
            summary.write_text(json.dumps({'count': 2, 'by_k': {'0': {'mean': {'score': 0.99}}}}))
            with self.assertRaisesRegex(ValueError, 'summary differs'):
                load_reusable_k0(root, run, tokens)
            (root / 'part-0001-of-0002/complete.json').unlink()
            with self.assertRaises(FileNotFoundError):
                load_reusable_k0(root, run, tokens)

    def test_changed_scene_seed_is_rejected_even_with_unchanged_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, tokens = baseline_fixture(root)
            path = root / 'part-0000-of-0002/scenes.jsonl'
            row = json.loads(path.read_text())
            row['seed'] += 1
            path.write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'seed mismatch'):
                load_reusable_k0(root, run, tokens)

    def test_k0_is_never_scored_again_and_delta_uses_saved_per_scene_score(self):
        class Refiner:
            def refine_many(self, poses, context, route, steps, gamma=None):
                return {k: poses + k / 100.0 for k in steps}

        calls = []
        def scorer(poses):
            calls.append(poses.copy())
            self.assertAlmostEqual(float(poses[0, 0]), 0.2)
            return {'score': 0.9}

        raw = torch.zeros(1, 8, 3)
        trajectories, scores, deltas, _ = score_pose_group(
            raw, [0, 20], Refiner(), None, None, 0.1, scorer, 'cpu',
            baseline_metrics={'score': 0.75},
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(trajectories[0], raw[0].tolist())
        self.assertEqual(scores[0]['score'], 0.75)
        self.assertAlmostEqual(deltas[20]['score'], 0.15)


if __name__ == '__main__':
    unittest.main()
