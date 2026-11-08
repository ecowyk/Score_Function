#!/usr/bin/env bash
# Start the fixed eight-GPU, 100-epoch DD-aligned score experiment.
set -euo pipefail
check=0
resume=0
for argument in "$@"; do
    case "$argument" in
        --check) check=1 ;;
        --resume) resume=1 ;;
        *) echo 'Usage: bash scripts/train_navsim_dd_aligned.sh [--check] [--resume]' >&2; exit 2 ;;
    esac
done

w=${WYK_ROOT:-/home/data/wyk}
sf=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cache="$w/dataset/navsim/exp/dd_score_cache_v1"
output="$w/dataset/navsim/exp/dd_score_ddaligned_v1"
python="$w/envs/diffusion_drive/bin/python"
config="$sf/configs/navsim_dd_s10_g_ddaligned.json"
[[ "$w" = /* && -x "$python" && -f "$config" && -f "$cache/TRAINING_READY.json" ]] || {
    echo "Missing aligned config, prepared cache, or Python environment under $w" >&2
    exit 1
}
export PATH="$(dirname "$python"):$PATH" PYTHONPATH="$sf:$w/DiffusionDrive:$w/nuplan-devkit"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$sf"
# The launcher's lock is adjacent to output, preserving the trainer's empty-directory check.
exec 9>"$output.launcher.lock"
flock -n 9 || { echo "Another aligned launcher owns $output" >&2; exit 1; }

"$python" - "$config" "$cache" "$output" "$resume" "$w" <<'PY'
import os
from pathlib import Path
import re
import sys

config, cache, output = map(Path, sys.argv[1:4])
resume = sys.argv[4] == "1"
root = Path(sys.argv[5])
old_output = (root / "dataset/navsim/exp/dd_score_s10g_v1_e200").resolve()
blocked = []
# Identify this user's exact local trainer argv before initializing CUDA.
for process in Path("/proc").iterdir():
    if not process.name.isdigit():
        continue
    try:
        if process.stat().st_uid != os.getuid():
            continue
        argv = [os.fsdecode(p) for p in (process / "cmdline").read_bytes().split(b"\0") if p]
        if not argv or not re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", Path(argv[0]).name):
            continue
        invocation = argv[1:]
        while invocation and invocation[0] in ("-u", "-B", "-E", "-s", "-S", "-I"):
            invocation = invocation[1:]
        if (len(invocation) < 2 or invocation[0] != "-m"
                or invocation[1] not in ("torch.distributed.run", "score_function.navsim_dd.train")):
            continue
        destination = next((argv[i + 1] for i, word in enumerate(argv[:-1]) if word == "--output"), None)
        destination = next((a.split("=", 1)[1] for a in argv if a.startswith("--output=")), destination)
        if destination is not None:
            path = Path(destination)
            if not path.is_absolute():
                path = (process / "cwd").resolve() / path
            if path.resolve() == old_output:
                blocked.append(process.name)
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
if blocked:
    sys.exit(f"Existing e200 training is active locally, PID(s) {', '.join(blocked)}. "
             "Stop or finish that run before launching this experiment.")

import torch
from score_function.model.score_branch import build_model
from score_function.navsim_dd.cache import NavsimScoreDataset, load_cache_metadata
from score_function.navsim_dd.train import _check_config, _seed
from score_function.train_epoch import train_epoch
from score_function.utils.dataset import collate_cpu
from score_function.utils.train_utils import file_hash, fingerprint, read_json

cfg = read_json(config)
_check_config(cfg)
expected = dict(epochs=100, microbatch_size=64, learning_rate=6e-4, weight_decay=1e-4,
                warmup_epochs=3, min_learning_rate=1e-6, lr_schedule="dd_warmup_cosine",
                precision="16-mixed", gradient_clip=0.0, ema_decay=0.0, drop_last=False,
                seed=20260922, sigma=0.1, gamma=0.1)
if any(cfg.get(key) != value for key, value in expected.items()):
    sys.exit("The DD-aligned configuration differs from the fixed experiment.")
ready = read_json(cache / "TRAINING_READY.json")
metadata = load_cache_metadata(cache)
if (ready.get("validation_version") != 1
        or ready.get("manifest_sha256") != file_hash(cache / "manifest.json")
        or ready.get("cache_fingerprint") != fingerprint(metadata)
        or metadata["identity"].get("max_per_split") is not None):
    sys.exit("TRAINING_READY does not validate the current full cache. Re-run score preparation.")
training = NavsimScoreDataset(cache, "train")
validation = NavsimScoreDataset(cache, "val")
counts = {"train": len(training), "val": len(validation)}
if ready.get("records") != counts or sum(counts.values()) != 103288 or len(training) < 64:
    sys.exit(f"Unexpected prepared NAVSIM sample counts: {counts}")
protocol = {
    "config": {k: v for k, v in cfg.items() if k != "num_workers"},
    "world_size": 8, "device_type": "cuda", "cache_fingerprint": fingerprint(metadata),
}
if resume:
    run = read_json(output / "run.json")
    if run.get("config") != cfg or run.get("signature") != fingerprint(protocol):
        sys.exit("Resume requires the same DD-aligned configuration, cache and eight-GPU protocol.")
    if (output / "history.json").exists() and not (output / "last.pt").is_file():
        sys.exit("The output has training history but last.pt is missing.")
elif output.exists() and any(p.name != ".train.lock" for p in output.iterdir()):
    sys.exit(f"Output already contains a run: {output}. Use --resume for this same experiment.")

if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
    sys.exit(f"Eight visible CUDA devices are required, found {torch.cuda.device_count()}.")
for index in range(8):
    device = torch.device("cuda", index)
    value = torch.ones(1, device=device)
    if value.sum().item() != 1:
        sys.exit(f"CUDA check failed on visible device {index}")
    print(f"CUDA {index}: {torch.cuda.get_device_name(index)}", flush=True)
    del value
device = torch.device("cuda", 0)
torch.cuda.set_device(device)
_seed(cfg["seed"])
model = build_model(cfg, device)
batch = collate_cpu([training[i] for i in range(64)])
if not all(torch.isfinite(batch[key]).all() for key in ("target", "context", "route")):
    sys.exit("The real cached batch contains nonfinite tensors.")
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
scaler = torch.cuda.amp.GradScaler()
# Reuse the production AMP, DSM and optimizer path. Initial GradScaler overflow
# can legitimately lower the scale, so retry this one real batch until an update succeeds.
for attempt in range(8):
    _seed(cfg["seed"] + 1)
    result = next(train_epoch([batch], model, optimizer, cfg, None, device, start_step=0,
                             start_update=0, updates_per_epoch=1, accumulation=1,
                             next_lr=cfg["learning_rate"], scaler=scaler))
    if not result["optimizer_step_skipped"]:
        break
else:
    sys.exit("FP16 preflight skipped every optimizer update after eight scale adjustments.")
torch.cuda.synchronize()
if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
    sys.exit("FP16 preflight produced nonfinite parameters.")
print(f"CHECK OK: cache {counts}, real batch=64, fp16 DSM={result['train_dsm']:.6f}, "
      f"GradScaler={result['amp_scale']:g}, visible GPUs={torch.cuda.device_count()}.", flush=True)
PY

(( check == 0 )) || exit 0
args=()
(( resume == 0 )) || args+=(--resume)
exec "$python" -m torch.distributed.run --standalone --nproc_per_node=8 \
    -m score_function.navsim_dd.train --cache-dir "$cache" --output "$output" \
    --config "$config" "${args[@]}"
