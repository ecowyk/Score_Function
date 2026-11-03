import json
from dataclasses import dataclass
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from score_function.navsim_dd.evaluate import (
    bounded_loader, digest_tokens, merge_shards, metric_index, native_top1, require_coverage, sampling_rng,
    score_pose_group, select_tokens, token_seed,
)


class FakeDecoder(torch.nn.Module):
    def forward(self, _features):
        # Mimics the stochastic candidates and the last-layer confidence output.
        poses = torch.randn(1, 3, 8, 3)
        logits = torch.tensor([[0., 1., 2.]])
        return [poses], [logits]


class FakePlanner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = FakeDecoder()
        self._transfuser_model = SimpleNamespace(_trajectory_head=SimpleNamespace(diff_decoder=self.decoder))
        self.call_count = 0

    def forward(self, features):
        self.call_count += 1
        poses, logits = self.decoder(features)
        return {'trajectory': poses[-1][:, logits[-1][0].argmax()]}


class FakeRefiner:
    def __init__(self):
        self.calls = 0

    def refine_many(self, poses, context, g, steps, gamma=None):
        self.calls += 1
        return {k: poses + k / 100. for k in steps}


class TestPairedEvaluation(unittest.TestCase):
    def test_state_independent_top1_and_single_rollout(self):
        planner = FakePlanner()
        state = torch.get_rng_state().clone()
        seed = token_seed(20260922, 'scene42')
        with sampling_rng(seed, 'cpu'):
            raw, mode = native_top1(planner, {})
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(mode, 2)
        torch.rand(81)  # Another token's scheduling/consumption must not affect this one.
        with sampling_rng(seed, 'cpu'):
            replica, _ = native_top1(planner, {})
        self.assertTrue(torch.equal(raw, replica))
        self.assertFalse(planner.decoder._forward_hooks)
        refiner = FakeRefiner()
        unchanged = raw.clone()
        paths, scores, delta, _ = score_pose_group(
            raw, [0, 5, 20], refiner, None, None, None,
            lambda array: {'score': array[0, 0]}, 'cpu',
        )
        self.assertEqual(refiner.calls, 1)
        self.assertEqual(planner.call_count, 2)  # Just the two explicit initializations above.
        self.assertTrue(torch.equal(unchanged, torch.tensor(paths[0]).unsqueeze(0)))
        self.assertAlmostEqual(delta[20]['score'], 0.20, places=5)
        self.assertLess(scores[0]['score'], scores[20]['score'])

    def test_stable_shards_and_missing_metric_coverage(self):
        selection, shard = select_tokens(['d', 'c', 'a', 'b'], 3, 1, 2)
        self.assertEqual(selection, ['a', 'b', 'c'])
        self.assertEqual(shard, ['b'])
        with self.assertRaisesRegex(ValueError, 'Missing 1'):
            require_coverage(selection, ['a', 'c'], 'metric caches')

    def test_bounded_smoke_uses_only_needed_present_official_logs(self):
        @dataclass
        class Filter:
            log_names: list
            max_scenes: int = 0

        class Loader:
            def __init__(self, names):
                self.scene_frames_dicts = dict.fromkeys(names)

            def __len__(self):
                return len(self.scene_frames_dicts)

        seen = []
        def load(filt):
            name = filt.log_names[0]
            seen.append(name)
            return Loader([name + str(i) for i in range(min(2, filt.max_scenes))])

        result = bounded_loader(Filter(['z', 'y', 'missing', 'x']), {'z', 'x', 'other', 'y'}, 3, load)
        self.assertEqual(seen, ['x', 'y'])
        self.assertEqual(list(result.scene_frames_dicts), ['x0', 'x1', 'y0'])
        with self.assertRaisesRegex(ValueError, 'fewer than 3'):
            bounded_loader(Filter(['x', 'missing']), {'x', 'other'}, 3, load)

    def test_metric_cache_index_uses_multiple_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'metadata'
            path.mkdir()
            for i, token in enumerate(['abc', 'def']):
                (path / f'node{i}.csv').write_text('file_name\n' + f'/cache/{token}/metric_cache.pkl\n')
            (path / 'node2.csv').write_text('\n')  # a node without scenes
            self.assertEqual(set(metric_index(path.parent)), {'abc', 'def'})
            (path / 'node3.csv').write_text('file_name\n/other/abc/metric_cache.pkl\n')
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                metric_index(path.parent)

    def test_merge_checks_complete_manifest_and_weights_scenes_equally(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = {'num_shards': 2, 'token_digest': digest_tokens(['a', 'b', 'c'])}
            for i, tokens in enumerate([['a', 'c'], ['b']]):
                shard = root / f'part-{i:04d}-of-0002'
                shard.mkdir()
                (shard/'complete.json').write_text(json.dumps(dict(run=config, shard_index=i, shard_tokens=tokens)))
                rows = [dict(token=t, metrics={'0': {'score': float(i)}}, delta={'0': {'score': 0.}}, refine_ms=0.) for t in tokens]
                (shard/'scenes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
                if i == 0:
                    with self.assertRaisesRegex(ValueError, 'coverage'):
                        merge_shards(root)
            report = merge_shards(root)
            self.assertAlmostEqual(report['by_k']['0']['mean']['score'], 1/3)
            (shard/'scenes.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError, 'Incomplete records'):
                merge_shards(root)
