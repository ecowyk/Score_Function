#!/usr/bin/env python3
"""Evaluate the final full DD-reader model at K=5,10,15,20,30 with saved K0."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import eval_navsim_dd_aligned as base

sys.path.insert(0, str(base.REPO))

KS = (5, 10, 15, 20, 30)
EPOCHS = 500
LATEST = "dd_score_reader_eval_latest.txt"
TRAIN_DIRECTORY = "dd_score_reader_v1"
EVAL_DIRECTORY = "dd_score_reader_navtest_v1"


def expected_config():
    return base.read_json(base.REPO / "configs/navsim_dd_s10_g_reader.json")


def validate_training(run, history):
    expected = expected_config()
    if expected.get("epochs") != EPOCHS or expected.get("lr_schedule_epochs", EPOCHS) != EPOCHS:
        raise ValueError("Reader evaluation requires the full 500-epoch cosine training configuration")
    if [row.get("epoch") for row in history] != list(range(1, EPOCHS + 1)):
        raise ValueError("Training is incomplete: require all 500 completed epochs")
    strip = lambda cfg: {k: v for k, v in cfg.items() if k != "num_workers"}
    if strip(run.get("config", {})) != strip(expected):
        raise ValueError("Training configuration differs from the full DD-reader experiment")
    if (run.get("global_batch") != 512 or run.get("protocol", {}).get("world_size") != 8
            or run.get("protocol", {}).get("device_type") != "cuda"):
        raise ValueError("Require eight-GPU training with global batch 512")


def validate_checkpoint(saved):
    expected = dict(format="navsim-dd-reader-score-v1", epoch=EPOCHS, weight_kind="raw",
                    sigma=0.1, gamma=0.1, training_precision="16-mixed", model=expected_config()["model"])
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f"Reader checkpoint {key}: expected {value!r}, got {saved.get(key)!r}")
    if not saved.get("reader_spec") or not saved.get("state_dict"):
        raise ValueError("Reader checkpoint has no portable reader specification or weights")
    identity = saved.get("conditioner_identity", {})
    if identity.get("condition") != "dd-backbone512-status-reader-v1" or not identity.get("reader_assets_sha256"):
        raise ValueError("Reader checkpoint has no frozen reader provenance")


def paths(root, training=None, output=None):
    exp = root / "dataset/navsim/exp"
    return (Path(training).resolve() if training is not None else exp / TRAIN_DIRECTORY,
            Path(output).resolve() if output is not None else exp / EVAL_DIRECTORY)


def reference_info(root, baseline):
    """Verify every reused K0 record before any new GPU workers are launched."""
    from score_function.navsim_dd.cache import conditioner_identity
    from score_function.navsim_dd.evaluate import digest_tokens, load_reusable_k0
    manifests = [base.read_json(path) for path in sorted(baseline.glob("part-*-of-*/complete.json"))]
    if len(manifests) != base.SHARDS:
        raise ValueError("K0 source needs all eight completed evaluation shards")
    selected = sorted(token for mark in manifests for token in mark["shard_tokens"])
    if len(selected) != base.TOTAL or len(set(selected)) != base.TOTAL:
        raise ValueError("K0 source must cover exactly 12146 distinct navtest scenes")
    weights = root / "DiffusionDrive/checkpoints"
    identity = conditioner_identity(weights / "diffusiondrive_navsim_88p1_PDMS",
                                    weights / "kmeans_navsim_traj_20.npy", weights / "pytorch_model.bin")
    expected = dict(split="navtest", seed=20260922, identity=identity,
                    token_digest=digest_tokens(selected), count=base.TOTAL,
                    data_root=str((root / "dataset/navsim").resolve()), max_scenes=None,
                    metric_cache=str((root / "dataset/navsim/exp/metric_cache").resolve()))
    _, provenance = load_reusable_k0(baseline, expected, selected, needed=[])
    return {"run": expected, "provenance": provenance}


def preflight(root, training, baseline=None):
    checkpoint = training / "final_raw.pt"
    for name in ("run.json", "history.json", "final_raw.pt"):
        if not (training / name).is_file():
            raise ValueError(f"Training has not completed: missing {training / name}")
    validate_training(base.read_json(training / "run.json"), base.read_json(training / "history.json"))
    if baseline is None:
        pointer = root / "dataset/navsim/exp/dd_score_e200_eval_latest.txt"
        if not pointer.is_file():
            raise ValueError(f"Missing K0 pointer: {pointer}; K0 will not be recomputed")
        baseline = Path(pointer.read_text().strip())
    if not baseline.is_absolute():
        raise ValueError("K0 source must be an absolute path")
    baseline = baseline.resolve()
    required = [root / "DiffusionDrive/checkpoints" / name for name in
                ("diffusiondrive_navsim_88p1_PDMS", "kmeans_navsim_traj_20.npy", "pytorch_model.bin")]
    required += [root / "dataset/nuplan/maps/nuplan-maps-v1.0.json",
                 root / "dataset/navsim/exp/metric_cache", root / "dataset/navsim/navsim_logs/test",
                 root / "dataset/navsim/sensor_blobs/test", root / "nuplan-devkit", root / "DiffusionDrive/navsim"]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing evaluation dependency: {path}")
    if Path(sys.executable).resolve() != (root / "envs/diffusion_drive/bin/python").resolve():
        raise ValueError(f"Run with {root}/envs/diffusion_drive/bin/python")
    import torch
    validate_checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=True))
    if not torch.cuda.is_available() or torch.cuda.device_count() != base.SHARDS:
        raise ValueError("Require exactly eight visible CUDA GPUs")
    info = reference_info(root, baseline)
    return checkpoint, baseline, info


def worker_command(root, output, checkpoint, baseline, shard):
    command = base.worker_command(root, output, checkpoint, baseline, shard)
    start, end = command.index("--ks") + 1, command.index("--gamma")
    command[start:end] = [str(k) for k in KS]
    command.append("--resume")
    return command


def validate_completed_shard(output, index, checkpoint_hash, info):
    """A skipped shard must have intact records for this exact run and K0 source."""
    from score_function.navsim_dd.evaluate import load_reusable_k0, summarize, validate_result_row
    directory = output / f"part-{index:04d}-of-{base.SHARDS:04d}"
    if not (directory / "complete.json").is_file():
        return False
    mark = base.read_json(directory / "complete.json")
    if mark != base.read_json(directory / "run.json") or mark.get("shard_index") != index:
        raise ValueError(f"Invalid completed reader shard {index}")
    run = mark["run"]
    expected = {**info["run"], "ks": [0, *KS], "gamma": 0.1, "num_shards": base.SHARDS,
                "score_checkpoint_sha256": checkpoint_hash, "k0_reuse": info["provenance"],
                "score_architecture": "dd_scene_reader_score"}
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Completed shard {index} belongs to a different reader evaluation")
    source = Path(info["provenance"]["source"])
    selected = sorted(token for path in source.glob("part-*-of-*/complete.json")
                      for token in base.read_json(path)["shard_tokens"])
    needed = selected[index::base.SHARDS]
    if mark["shard_tokens"] != needed:
        raise ValueError(f"Incorrect token assignment in completed reader shard {index}")
    baseline, provenance = load_reusable_k0(source, info["run"], selected, needed)
    if provenance != info["provenance"]:
        raise ValueError("K0 source changed while validating completed shards")
    rows = [json.loads(line) for line in (directory / "scenes.jsonl").read_text().splitlines()]
    if [row["token"] for row in rows] != needed:
        raise ValueError(f"Invalid records in completed reader shard {index}")
    for row in rows:
        token = row["token"]
        validate_result_row(row, token, baseline[token].get("log", row.get("log")),
                            [0, *KS], 20260922, baseline)
    actual = summarize(rows)
    recorded = base.read_json(directory / "summary.json")
    if any(recorded.get(key) != value for key, value in actual.items()):
        raise ValueError(f"Summary mismatch in completed reader shard {index}")
    return True


def print_summary(output):
    summary = base.read_json(output / "summary.json")
    if summary.get("count") != base.TOTAL or set(summary.get("by_k", {})) != {str(k) for k in [0, *KS]}:
        raise ValueError("Reader evaluation summary has incomplete scenes or K values")
    print(f"DD reader C, final RAW epoch {EPOCHS}; {summary['count']} paired scenes; "
          "seed=20260922; gamma=0.1; sigma=0.1", flush=True)
    print("K    origin       PDMS      delta K0    improved / worsened / tied", flush=True)
    rows = []
    for k in (0, *KS):
        item = summary["by_k"][str(k)]
        row = dict(K=k, origin="reused" if k == 0 else "computed", PDMS=100 * item["mean"]["score"],
                   delta_K0=100 * item["mean_delta"]["score"], improved=item["improved"],
                   worsened=item["worsened"], tied=item["tied"])
        rows.append(row)
        print(f"{k:<4} {row['origin']:<9} {row['PDMS']:9.4f} {row['delta_K0']:+12.4f}    "
              f"{row['improved']} / {row['worsened']} / {row['tied']}", flush=True)
    return rows


def save_table(output):
    rows = print_summary(output)
    temporary = output / "summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "summary.csv")


def run(root, training=None, output=None, baseline=None):
    from score_function.utils.train_utils import file_hash
    training, output = paths(root, training, output)
    exp = root / "dataset/navsim/exp"
    exp.mkdir(parents=True, exist_ok=True)
    with (exp / "dd_score_ddaligned_eval.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another eight-GPU DD evaluation runner is active") from None
        checkpoint, baseline, info = preflight(root, training, baseline)
        if output in (training, baseline) or baseline in output.parents or output in baseline.parents:
            raise ValueError("Evaluation output must be separate from training and reused K0 results")
        protocol = dict(checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
                        training=str(training), baseline=str(baseline), reference=info,
                        epochs=EPOCHS, computed_ks=list(KS), reused_ks=[0], seed=20260922, gamma=0.1, sigma=0.1)
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".launcher.lock").open("a") as output_lock:
            try:
                fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Another launcher owns {output}") from None
            if (output / "launcher.json").exists():
                previous = base.read_json(output / "launcher.json")
                if previous.get("protocol") != protocol:
                    raise ValueError("Reader evaluation output already belongs to another protocol")
            elif any(path.name != ".launcher.lock" for path in output.iterdir()):
                raise ValueError("Reader evaluation output is nonempty and has no launcher provenance")
            state = dict(status="running", pid=os.getpid(), host=socket.gethostname(),
                         protocol=protocol, started=time.time())
            def save():
                base.atomic_write(output / "launcher.json", json.dumps(state, indent=2) + "\n")
            save()
            base.atomic_write(exp / LATEST, str(output) + "\n")
            children, handlers = [], {}
            def interrupted(signum, frame):
                raise InterruptedError(f"Received signal {signum}")
            try:
                for signum in (signal.SIGINT, signal.SIGTERM):
                    handlers[signum] = signal.signal(signum, interrupted)
                print(f"Reader evaluation: {output}\nCheckpoint: {checkpoint}\nReused K0: {baseline}", flush=True)
                print(f"Compute K={list(KS)} along one shared path through K={max(KS)}", flush=True)
                env, active = base.child_env(root), {}
                for shard in range(base.SHARDS):
                    if validate_completed_shard(output, shard, protocol["checkpoint_sha256"], info):
                        print(f"Shard {shard}: verified complete, reused", flush=True)
                        continue
                    with (output / f"shard-{shard}.log").open("a") as log:
                        log.write(f"\n=== reader evaluation attempt {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\n")
                        log.flush()
                        child = subprocess.Popen(worker_command(root, output, checkpoint, baseline, shard),
                                                 cwd=base.REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
                    children.append(child)
                    active[shard] = child
                last_report = 0
                while active:
                    for shard, child in list(active.items()):
                        code = child.poll()
                        if code is not None:
                            del active[shard]
                            print(f"Shard {shard} exited {code}; log: {output}/shard-{shard}.log", flush=True)
                            if code:
                                raise RuntimeError(f"Shard {shard} failed (exit {code}); merge skipped")
                    if time.monotonic() - last_report >= 30:
                        base.print_progress(output)
                        last_report = time.monotonic()
                    if active:
                        time.sleep(1)
                if reference_info(root, baseline) != info or file_hash(checkpoint) != protocol["checkpoint_sha256"]:
                    raise ValueError("The reused K0 source or final checkpoint changed during evaluation")
                state["status"] = "merging"
                save()
                merge = subprocess.Popen([sys.executable, "-m", "score_function.navsim_dd.evaluate",
                                          "--merge", "--output", str(output)], cwd=base.REPO, env=env)
                children.append(merge)
                if merge.wait() != 0:
                    raise RuntimeError("Reader evaluation merge failed")
                save_table(output)
                state["status"] = "complete"
            except BaseException as error:
                for signum in handlers:
                    signal.signal(signum, signal.SIG_IGN)
                base.stop_children(children)
                state.update(status="interrupted" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                             error=str(error))
                raise
            finally:
                state["finished"] = time.time()
                save()
                for signum, handler in handlers.items():
                    signal.signal(signum, handler)
    return output


def status(root, output=None):
    if output is None:
        pointer = root / "dataset/navsim/exp" / LATEST
        if not pointer.is_file():
            print("尚未启动 DD reader 评测")
            return
        output = Path(pointer.read_text().strip())
    print(f"Reader evaluation: {output}")
    state = base.read_json(output / "launcher.json")
    print(f"State: {state['status']}; final RAW epoch {EPOCHS}; computed K={list(KS)}; reused K0")
    if state.get("error"):
        print(state["error"])
    base.print_progress(output)
    if state["status"] == "complete":
        print_summary(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status"))
    parser.add_argument("--root", type=Path, default=Path("/home/data/wyk"))
    parser.add_argument("--train-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "run":
            run(args.root.resolve(), args.train_output, args.output, args.baseline)
        else:
            status(args.root.resolve(), args.output)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
