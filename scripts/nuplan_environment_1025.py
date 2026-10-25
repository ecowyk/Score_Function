"""Root-derived paths needed while resolving saved nuPlan Hydra configs."""

from pathlib import Path


def nuplan_paths(root):
    root = Path(root).expanduser().resolve()
    return dict(
        DP_DIR=str(root / "Diffusion-Planner"),
        NUPLAN_DEVKIT_ROOT=str(root / "nuplan-devkit"),
        NUPLAN_DATA_ROOT=str(root / "dataset/nuplan"),
        NUPLAN_MAPS_ROOT=str(root / "dataset/nuplan/maps"),
        NUPLAN_EXP_ROOT=str(root / "dataset/nuplan/exp"),
    )
