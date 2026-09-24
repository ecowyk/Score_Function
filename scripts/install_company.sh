#!/usr/bin/env bash
# Run in an existing compatible official Diffusion Planner conda environment.
set -euo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python -c 'import sys,numpy,torch; assert sys.version_info >= (3,9); assert int(numpy.__version__.split(".")[0]) < 2; assert torch.cuda.is_available(); print(torch.__version__, numpy.__version__)'
python -m pip install --no-deps --no-build-isolation -e "$CODE"
echo 'Installed. Edit configs/score_function.json paths, then use train_tmux.sh.'
