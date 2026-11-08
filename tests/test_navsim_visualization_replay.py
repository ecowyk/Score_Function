"""Geometry and event-fidelity tests without a NAVSIM installation or GPU."""

from enum import IntEnum
import math
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from score_function.navsim_dd.visualization_replay import (
    _collision_events, offroad_corners, polygon_records, require_matching_metrics, to_local,
)


def test_shared_anchor_rotation_and_empty_points():
    anchor = dict(x=10.0, y=20.0, heading=math.pi / 2)
    np.testing.assert_allclose(to_local([[10, 21], [9, 20]], anchor), [[1, 0], [0, 1]], atol=1e-12)
    assert to_local(np.empty((0, 2)), anchor).shape == (0, 2)


def test_replay_metrics_reject_changed_score_missing_key_and_nonfinite():
    require_matching_metrics({"score": 0.8, "comfort": 1}, {"score": 0.8, "comfort": 1})
    for reference in ({"score": 0.9, "comfort": 1}, {"score": 0.8}, {"score": float("nan"), "comfort": 1}):
        with pytest.raises(ValueError):
            require_matching_metrics({"score": 0.8, "comfort": 1}, reference)


def test_polygon_holes_survive_transform_with_opposite_winding():
    shapely = pytest.importorskip("shapely")
    polygon = shapely.Polygon([(0, 0), (0, 10), (10, 10), (10, 0)],
                              holes=[[(3, 3), (7, 3), (7, 7), (3, 7)]])
    records = polygon_records(polygon, dict(x=5, y=5, heading=math.pi / 2))
    assert len(records) == 1 and len(records[0]["holes"]) == 1
    result = shapely.Polygon(records[0]["exterior"], records[0]["holes"])
    assert result.area == pytest.approx(84.0)
    assert not result.contains(shapely.Point(0, 0))
    assert result.exterior.is_ccw and not result.interiors[0].is_ccw


def test_offroad_locations_use_same_map_union_and_exclude_center():
    shapely = pytest.importorskip("shapely")
    # Center may lie outside all polygons: official DAC checks four corners.
    shapes = [shapely.box(0, 0, 5, 5), shapely.box(5, 0, 10, 5)]
    class Map:
        def points_in_polygons(self, coordinates):
            return np.stack([shapely.contains_xy(shape, coordinates[..., 0], coordinates[..., 1])
                             for shape in shapes])
    coords = np.array([[[1, 1], [1, 4], [9, 4], [9, 1], [100, 100]],
                       [[-1, 1], [1, 4], [9, 4], [9, 1], [2, 2]]], dtype=float)
    points = offroad_corners(Map(), coords, [0, 1], [False, True])
    assert points[0].shape == (0, 2)
    np.testing.assert_array_equal(points[1], [[-1, 1]])
    with pytest.raises(ValueError, match="official PDM"):
        offroad_corners(Map(), coords, [0, 1], [False, False])


def _modules(attributes):
    result = {}
    for name, attrs in attributes.items():
        parts = name.split(".")
        for count in range(1, len(parts) + 1):
            path = ".".join(parts[:count])
            result.setdefault(path, ModuleType(path))
        result[name].__dict__.update(attrs)
    return result


def test_collision_marks_keep_ignored_contacts_but_verify_official_fault_score():
    shapely = pytest.importorskip("shapely")
    class Collision(IntEnum):
        ACTIVE_FRONT_COLLISION = 0
        STOPPED_TRACK_COLLISION = 1
        ACTIVE_LATERAL_COLLISION = 2
    class Area(IntEnum):
        MULTIPLE_LANES = 0
        NON_DRIVABLE_AREA = 1
    class Multi(IntEnum):
        NO_COLLISION = 0
    attributes = {
        "nuplan.common.actor_state.tracked_objects_types": {"AGENT_TYPES": {"vehicle"}},
        "nuplan.planning.metrics.utils.collision_utils": {"CollisionType": Collision},
        "navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils": {
            "get_collision_type": lambda state, polygon, actor, other: actor.collision_type},
        "navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums": {
            "EgoAreaIndex": Area, "MultiMetricIndex": Multi},
    }
    shape = shapely.box(-1, -1, 1, 1)
    class Occupancy:
        def __init__(self, tokens): self.tokens = tokens
        def intersects(self, polygon): return self.tokens
        def __getitem__(self, token): return shape
    class Observation:
        collided_track_ids = ["initial"]
        red_light_token = "red_light"
        unique_objects = {
            "initial": SimpleNamespace(tracked_object_type="vehicle", collision_type=Collision.ACTIVE_FRONT_COLLISION),
            "side": SimpleNamespace(tracked_object_type="vehicle", collision_type=Collision.ACTIVE_LATERAL_COLLISION),
            "front": SimpleNamespace(tracked_object_type="vehicle", collision_type=Collision.ACTIVE_FRONT_COLLISION),
        }
        def __getitem__(self, index):
            return Occupancy(["initial", "side", "red_light_foo"] + (["front"] if index else []))
    scorer = SimpleNamespace(_states=np.zeros((2, 3, 11)), _ego_polygons=np.array([[shape] * 3] * 2),
                             _ego_areas=np.zeros((2, 3, 2), dtype=bool),
                             _multi_metrics=np.zeros((1, 2)), _collision_time_idcs=np.array([math.inf, 1.0]))
    # A side contact on a later offroad frame stays ignored after its first non-fault contact.
    scorer._ego_areas[1, 1:, Area.NON_DRIVABLE_AREA] = True
    with patch.dict(sys.modules, _modules(attributes)):
        events = _collision_events(scorer, Observation(), dict(x=0, y=0, heading=0))
        assert [len(frame) for frame in events] == [2, 3, 3]
        assert events[0][0]["ignored_initial"] and events[0][0]["ignored_by_metric"]
        assert not events[0][1]["at_fault"] and not events[0][1]["ignored_by_metric"]
        assert events[1][1]["ignored_by_metric"] and not events[1][1]["at_fault"]
        assert events[1][2]["at_fault"]
        scorer._multi_metrics[0, 1] = 1.0
        with pytest.raises(ValueError, match="official PDM collision"):
            _collision_events(scorer, Observation(), dict(x=0, y=0, heading=0))
