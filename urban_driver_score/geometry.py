"""Resample the native nuPlan trajectory in its rear-axle ego frame."""

import numpy as np


def relative_features(global_poses, anchor):
    poses = np.asarray(global_poses, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    if poses.shape != (80, 3) or anchor.shape != (3,):
        raise ValueError("Expected 80 global rear-axle poses and one SE2 anchor")
    if not np.isfinite(poses).all() or not np.isfinite(anchor).all():
        raise ValueError("Nonfinite trajectory")
    delta = poses[:, :2] - anchor[:2]
    c, s = np.cos(anchor[2]), np.sin(anchor[2])
    heading = poses[:, 2] - anchor[2]
    return np.column_stack((delta[:, 0] * c + delta[:, 1] * s,
                            -delta[:, 0] * s + delta[:, 1] * c,
                            np.cos(heading), np.sin(heading))).astype(np.float32)


def sample_native(trajectory, ego):
    from nuplan.common.actor_state.state_representation import TimePoint

    start = ego.time_point.time_us
    times = start + np.arange(1, 81, dtype=np.int64) * 100000
    # nuPlan's float-second -> integer-microsecond conversion can truncate 1 us.
    if trajectory.start_time.time_us > start or times[-1] > trajectory.end_time.time_us + 1:
        raise ValueError("UrbanDriver must cover the full next 8 seconds")
    times = np.minimum(times, trajectory.end_time.time_us)
    poses = [trajectory.get_state_at_time(TimePoint(int(t))).rear_axle.serialize()
             for t in times]
    return relative_features(poses, ego.rear_axle.serialize())


def physical_to_poses(physical):
    values = np.asarray(physical, dtype=np.float64)
    if values.shape != (80, 4) or not np.isfinite(values).all():
        raise ValueError("Expected finite physical ego80x4")
    return np.column_stack((values[:, :2], np.arctan2(values[:, 3], values[:, 2])))
