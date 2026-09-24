"""Fetch missing official sources/checkpoints without changing existing checkouts."""

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from score_function.tools.experiment_suite import make_plan, parser

SOURCES = (
    (
        "planner_dir",
        "https://github.com/ZhengYinan-AIR/Diffusion-Planner.git",
        "a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df",
    ),
    (
        "devkit_dir",
        "https://github.com/motional/nuplan-devkit.git",
        "e9241677997dd86bfc0bcd44817ab04fe631405b",
    ),
)


def main():
    args = parser().parse_args()
    config = make_plan(args)["experiments"][0]["config"]
    paths = config["paths"]
    for key in ("database_dir", "maps_dir"):
        if not Path(paths[key]).is_dir():
            raise FileNotFoundError(f"Provide extracted nuPlan {key}: {paths[key]}")
    for key, url, revision in SOURCES:
        path = Path(paths[key])
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", url, str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", revision], check=True)
        expected = "diffusion_planner" if key == "planner_dir" else "nuplan"
        if not (path / expected).is_dir():
            raise FileNotFoundError(f"Not an official source checkout: {path}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                "-e",
                str(path),
            ],
            check=True,
        )
    for key, filename, supplied in (
        ("planner_args", "args.json", args.planner_args),
        ("planner_checkpoint", "model.pth", args.checkpoint),
    ):
        path = Path(paths[key])
        if path.exists():
            continue
        if supplied:
            raise FileNotFoundError(f"Explicit checkpoint path is missing: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        url = f"https://huggingface.co/ZhengYinan2001/Diffusion-Planner/resolve/main/{filename}"
        print(f"Downloading official {filename} to {path}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
        os.replace(temporary, path)


if __name__ == "__main__":
    main()
