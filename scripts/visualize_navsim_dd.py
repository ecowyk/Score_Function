#!/usr/bin/env python3
"""Replay five gains and five losses from the aligned 500-epoch evaluation."""
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]


def main():
    # Configure paths before NAVSIM imports capture their environment variables.
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", type=Path, default=Path("/home/data/wyk"))
    args, _ = parser.parse_known_args()
    root = args.root.resolve()
    sys.path[:0] = list(map(str, [REPO, root / "DiffusionDrive", root / "nuplan-devkit"]))
    os.environ.update(OPENSCENE_DATA_ROOT=str(root / "dataset/navsim"),
                      NAVSIM_DEVKIT_ROOT=str(root / "DiffusionDrive"),
                      NUPLAN_MAPS_ROOT=str(root / "dataset/nuplan/maps"),
                      NUPLAN_MAP_VERSION="nuplan-maps-v1.0", MPLBACKEND="Agg",
                      CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", PYTHONNOUSERSITE="1")
    from score_function.navsim_dd.visualize import main as visualize
    visualize()


if __name__ == "__main__":
    main()
