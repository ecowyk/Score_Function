"""Renderer tests use explicit synthetic data, never real experiment claims."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from score_function.navsim_dd.visualization_render import (
    _EVENT_OUTLINES, polygon_path, poster_frame_index, render_case, validate_payload,
)


def synthetic_payload():
    def rectangle(x, y, length=4.5, width=2):
        return [[x-length/2, y-width/2], [x+length/2, y-width/2],
                [x+length/2, y+width/2], [x-length/2, y+width/2]]

    metrics = dict(no_at_fault_collisions=1., drivable_area_compliance=1.,
                   time_to_collision_within_bound=1., ego_progress=.8, comfort=1.)
    runs = {}
    for arm in ('0', '20'):
        frames = []
        for i, time_s in enumerate([0., .1, .2]):
            x, y = i * 2, (0. if arm == '0' else i * .6)
            events = dict(offroad=[], collision=[])
            if arm == '0' and i >= 1:
                events['offroad'] = [[x+2.25, y+1]]
            if arm == '20' and i == 2:
                events['collision'] = [dict(xy=[x+2, y], token='actor', at_fault=False)]
            frames.append(dict(time_s=time_s, ego_xy=[x, y], ego_polygon=rectangle(x, y),
                               actors=[dict(token='actor', kind='VEHICLE', polygon=rectangle(7, 1.5))],
                               events=events))
        runs[arm] = dict(planned_xy=[[0, 0], [10, 0 if arm == '0' else 1.2], [20, 0]],
                         frames=frames, metrics={**metrics, 'score': .8 if arm == '0' else .9})
    return dict(metadata=dict(token='SYNTHETIC_NOT_EXPERIMENT', log='renderer_test_fixture',
                              scores={'0': .8, '20': .9}, delta=.1),
                dt=.1, times=[0., .1, .2],
                drivable_polygons=[dict(exterior=[[-20, -8], [40, -8], [40, 8], [-20, 8]],
                                        holes=[[[15, 3], [18, 3], [18, 6], [15, 6]]])],
                lanes=[], expert_xy=[[0, 0], [10, .3], [20, .3]], runs=runs)


class RendererTests(unittest.TestCase):
    def test_timeline_and_required_context(self):
        payload = synthetic_payload()
        np.testing.assert_equal(validate_payload(payload), [0, .1, .2])
        for mutate in (
            lambda p: p.update(drivable_polygons=[]),
            lambda p: p.update(expert_xy=[]),
            lambda p: p['runs']['20']['frames'][1].update(time_s=.15),
        ):
            altered = copy.deepcopy(payload)
            mutate(altered)
            with self.assertRaises(ValueError):
                validate_payload(altered)

    def test_event_poster_uses_both_arms(self):
        payload = synthetic_payload()
        self.assertEqual(poster_frame_index(payload), 2)
        payload['runs']['20']['frames'][2]['events']['collision'] = []
        self.assertEqual(poster_frame_index(payload), 1)
        for run in payload['runs'].values():
            for frame in run['frames']:
                frame['events'] = {}
        self.assertEqual(poster_frame_index(payload), 2)

    def test_polygon_hole_has_opposite_winding(self):
        path = polygon_path(synthetic_payload()['drivable_polygons'][0])
        starts = np.flatnonzero(path.codes == path.MOVETO)
        areas = []
        for start, end in zip(starts, [*starts[1:], len(path.vertices)]):
            xy = path.vertices[start:end]
            areas.append(np.sum(xy[:-1, 0]*xy[1:, 1] - xy[1:, 0]*xy[:-1, 1]))
        self.assertGreater(areas[0], 0)
        self.assertLess(areas[1], 0)

    def test_gif_and_poster_include_full_timeline(self):
        self.assertEqual(set(_EVENT_OUTLINES), {'offroad', 'collision'})
        with tempfile.TemporaryDirectory() as directory:
            gif, png = Path(directory)/'synthetic.gif', Path(directory)/'synthetic.png'
            result = render_case(synthetic_payload(), gif, png, dpi=60, radius=18, case_id='TEST ONLY')
            self.assertEqual(result['frames'], 3)
            self.assertAlmostEqual(result['duration_s'], 1.3)
            with Image.open(gif) as image:
                self.assertEqual(image.n_frames, 3)
                durations = []
                for i in range(image.n_frames):
                    image.seek(i)
                    durations.append(image.info['duration'])
                self.assertEqual(durations, [500, 100, 700])
            with Image.open(png) as image:
                self.assertEqual(image.size, (960, 540))


if __name__ == '__main__':
    unittest.main()
