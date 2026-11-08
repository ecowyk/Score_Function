"""Reconstruct saved NAVSIM evaluations for faithful, CPU-only visual replay.

The plotted ego motion is the official PDM simulator output, not a linear
interpolation of the submitted waypoints.  No DD or Score network is loaded.
All geometry uses one fixed frame: the initial ego rear axle is the origin.
"""

from dataclasses import asdict
import math

import numpy as np


def to_local(points, anchor):
    """Transform arbitrary [..., 2] world-coordinate arrays into the anchor frame."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1:] != (2,) or not np.isfinite(points).all():
        raise ValueError("Expected finite world coordinates [...,2]")
    x, y, heading = (float(anchor[name]) for name in ("x", "y", "heading"))
    c, s = math.cos(heading), math.sin(heading)
    return (points - [x, y]) @ np.array([[c, -s], [s, c]])


def polygon_records(geometry, anchor):
    """Preserve polygon holes and ring winding for matplotlib's nonzero fill rule."""
    from shapely.geometry.polygon import orient

    if geometry.is_empty:
        return []
    if geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        return [record for part in geometry.geoms for record in polygon_records(part, anchor)]
    if geometry.geom_type != "Polygon":
        return []
    polygon = orient(geometry, sign=1.0)
    return [{"exterior": to_local(np.asarray(polygon.exterior.coords)[:, :2], anchor).tolist(),
             "holes": [to_local(np.asarray(ring.coords)[:, :2], anchor).tolist()
                       for ring in polygon.interiors]}]


def require_matching_metrics(actual, saved, *, tolerance=1e-7):
    """Fail before publishing a replay whose simulation does not match its score."""
    if set(actual) != set(saved):
        raise ValueError(f"Replay metric keys differ: {set(actual) ^ set(saved)}")
    for name, value in actual.items():
        reference = float(saved[name])
        if (not math.isfinite(float(value)) or not math.isfinite(reference)
                or not math.isclose(float(value), reference, rel_tol=0.0, abs_tol=tolerance)):
            raise ValueError(f"Replay metric mismatch for {name}: {value} vs saved {reference}")


def offroad_corners(drivable_map, ego_coords, drivable_indices, expected_mask):
    """Locate exactly the corners used by the official NON_DRIVABLE_AREA mask."""
    ego_coords = np.asarray(ego_coords, dtype=np.float64)
    # PDM returns [polygon, proposal, time, point]; final point is the center.
    contains = drivable_map.points_in_polygons(ego_coords[None])
    inside = contains[np.asarray(drivable_indices, dtype=int), 0, :, :-1].any(axis=0)
    outside = ~inside
    if not np.array_equal(outside.any(axis=-1), np.asarray(expected_mask, dtype=bool)):
        raise ValueError("Located offroad corners disagree with the official PDM area mask")
    return [coords[:-1][mask] for coords, mask in zip(ego_coords, outside)]


def _collision_events(scorer, observation, anchor, proposal=1):
    """Mark every actual contact; separately reproduce official responsibility.

    Red-light occupancy polygons are virtual constraints, not physical actors.
    Contacts ignored by NC (initial or previously non-fault contacts) are still
    drawn, with an explicit ignored_by_metric flag. TTC projections are not
    drawn as actual collisions.
    """
    from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
    from nuplan.planning.metrics.utils.collision_utils import CollisionType
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils import get_collision_type
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex, MultiMetricIndex

    initial_ignored = set(observation.collided_track_ids)
    ignored = set(initial_ignored)
    collision_score, first_fault = 1.0, math.inf
    result = []
    for time_idx, polygon in enumerate(scorer._ego_polygons[proposal]):
        occupancy = observation[time_idx]
        events = []
        for token in occupancy.intersects(polygon):
            if observation.red_light_token in token:
                continue
            actor = observation.unique_objects[token]
            actor_polygon = occupancy[token]
            excluded = token in ignored
            at_fault = False
            collision_type_name = None
            if not excluded:
                collision_type = get_collision_type(
                    scorer._states[proposal, time_idx], polygon, actor, actor_polygon
                )
                collision_type_name = collision_type.name
                crossing = bool(scorer._ego_areas[proposal, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                                or scorer._ego_areas[proposal, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA])
                at_fault = (collision_type in (CollisionType.ACTIVE_FRONT_COLLISION,
                                               CollisionType.STOPPED_TRACK_COLLISION)
                            or (crossing and collision_type == CollisionType.ACTIVE_LATERAL_COLLISION))
                if at_fault:
                    collision_score = min(collision_score, 0.0 if actor.tracked_object_type in AGENT_TYPES else 0.5)
                    first_fault = min(first_fault, time_idx)
                else:
                    ignored.add(token)
            contact = polygon.intersection(actor_polygon)
            point = contact.representative_point() if not contact.is_empty else actor_polygon.centroid
            events.append({"xy": to_local([point.x, point.y], anchor).tolist(), "token": token,
                           "at_fault": bool(at_fault), "ignored_initial": token in initial_ignored,
                           "ignored_by_metric": excluded, "collision_type": collision_type_name})
        result.append(events)
    official = float(scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, proposal])
    if collision_score != official or first_fault != float(scorer._collision_time_idcs[proposal]):
        raise ValueError("Replayed collision events disagree with official PDM collision score/time")
    return result


def _actors_at(observation, time_idx, anchor):
    occupancy = observation[time_idx]
    actors = []
    for token in occupancy.tokens:
        if observation.red_light_token in token:
            continue
        actor = observation.unique_objects[token]
        records = polygon_records(occupancy[token], anchor)
        for polygon in records:
            actors.append({"token": token, "kind": actor.tracked_object_type.name,
                           "polygon": polygon["exterior"]})
    return actors


def replay_case(row, metric_cache, simulator, scorer, expert_poses=None):
    """Replay saved K0/K20 waypoints and return renderer-neutral JSON data.

    ``expert_poses`` must be the actual human future from Scene metadata, in
    initial-ego-relative coordinates. metric_cache.trajectory is a PDM reference
    and deliberately is never presented as the human expert.
    """
    from shapely.ops import unary_union
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        EgoAreaIndex, StateIndex, MultiMetricIndex,
    )

    sampling = simulator.proposal_sampling
    if sampling != scorer.proposal_sampling:
        raise ValueError("PDM simulator and scorer sampling differ")
    if sampling.num_poses != 40 or not math.isclose(sampling.interval_length, 0.1):
        raise ValueError("This replay requires NAVSIM v1's 4-second, 10-Hz PDM simulation")
    initial = metric_cache.ego_state.rear_axle
    anchor = {name: float(getattr(initial, name)) for name in ("x", "y", "heading")}
    drivable_map = metric_cache.drivable_area_map
    area_indices = drivable_map.get_indices_of_map_type([
        SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA,
    ])
    if not area_indices:
        raise ValueError("Metric cache has no official drivable-area polygons")
    road = unary_union([drivable_map[drivable_map.tokens[index]] for index in area_indices])
    lanes = []
    for index in drivable_map.get_indices_of_map_type([SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]):
        lanes.extend(polygon_records(drivable_map[drivable_map.tokens[index]], anchor))
    dt = float(sampling.interval_length)
    times = (np.arange(sampling.num_poses + 1) * dt).tolist()
    actors = [_actors_at(metric_cache.observation, index, anchor) for index in range(len(times))]
    expert_xy = []
    if expert_poses is not None:
        expert = np.asarray(expert_poses, dtype=np.float64)
        if expert.shape != (8, 3) or not np.isfinite(expert).all():
            raise ValueError("Expected eight finite human expert future poses")
        expert_xy = np.concatenate([np.zeros((1, 2)), expert[:, :2]]).tolist()
    payload = {"metadata": {"token": row["token"], "log": row["log"], "anchor": anchor,
                            "metrics": row["metrics"],
                            "scores": {k: float(row["metrics"][k]["score"]) for k in ("0", "20")},
                            "delta": float(row["metrics"]["20"]["score"] - row["metrics"]["0"]["score"]),
                            "motion_source": "official PDM simulated states",
                            "collision_marks": "all physical contacts; at_fault is recorded separately",
                            "expert_source": "human Scene future" if expert_poses is not None else None},
               "dt": dt, "times": times, "drivable_polygons": polygon_records(road, anchor),
               "lanes": lanes, "expert_xy": expert_xy, "runs": {}}
    for k in ("0", "20"):
        # Original evaluation handed float32 predictions to Trajectory. Recover
        # that same dtype before the official transform converts to float64.
        poses = np.asarray(row["trajectories"][k], dtype=np.float32)
        if poses.shape != (8, 3) or not np.isfinite(poses).all():
            raise ValueError(f"Invalid saved trajectory for K={k}")
        actual = {name: float(value) for name, value in asdict(pdm_score(
            metric_cache, Trajectory(poses), sampling, simulator, scorer
        )).items()}
        require_matching_metrics(actual, row["metrics"][k])
        coords = scorer._ego_coords[1]
        mask = scorer._ego_areas[1, :, EgoAreaIndex.NON_DRIVABLE_AREA]
        corners = offroad_corners(drivable_map, coords, area_indices, mask)
        if float(not mask.any()) != float(scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, 1]):
            raise ValueError("Drivable-area events disagree with the recomputed metric")
        collisions = _collision_events(scorer, metric_cache.observation, anchor)
        frames = []
        for index, time_s in enumerate(times):
            state = scorer._states[1, index]
            frames.append({"time_s": time_s,
                           "ego_xy": to_local(state[StateIndex.POINT], anchor).tolist(),
                           "ego_polygon": to_local(coords[index, :-1], anchor).tolist(),
                           "actors": actors[index],
                           "events": {"offroad": to_local(corners[index], anchor).tolist(),
                                      "collision": collisions[index]}})
        payload["runs"][k] = {"planned_xy": np.concatenate([np.zeros((1, 2)), poses[:, :2]]).tolist(),
                                 "frames": frames, "metrics": actual}
    if not np.allclose(payload["runs"]["0"]["frames"][0]["ego_polygon"],
                       payload["runs"]["20"]["frames"][0]["ego_polygon"], rtol=0, atol=1e-9):
        raise ValueError("Replayed K0 and K20 do not share the same initial footprint")
    return payload
