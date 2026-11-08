"""Select and replay saved NAVSIM K0/K20 comparisons without neural inference."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import math
from pathlib import Path
import tempfile
import traceback

TOTAL = 12146
EVAL_POINTER = "dd_score_ddaligned_e500_eval_latest.txt"
VIS_POINTER = "dd_score_ddaligned_e500_visuals_latest.txt"
RENDER_VERSION = 1


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_text(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest_tokens(tokens):
    return hashlib.sha256("\n".join(tokens).encode()).hexdigest()


def load_evaluation(source, expected_count=TOTAL):
    """Require complete, unique, same-protocol shards and matching saved means."""
    source = Path(source).resolve()
    summary = read_json(source / "summary.json")
    if summary.get("count") != expected_count:
        raise ValueError(f"Expected {expected_count} evaluated scenes: {source}")
    means = [summary.get("by_k", {}).get(k, {}).get("mean", {}) for k in ("0", "20")]
    if any("score" not in mean for mean in means) or set(means[0]) != set(means[1]):
        raise ValueError("Both summary arms must contain matching metric sets including score")
    records, tokens, shard_ids, hashes, run = [], set(), set(), {}, None
    for directory in sorted(source.glob("part-*-of-*")):
        complete = read_json(directory / "complete.json")
        if read_json(directory / "run.json") != complete:
            raise ValueError(f"Run/completion metadata differ: {directory}")
        if run is None:
            run = complete["run"]
        if complete["run"] != run or complete["shard_index"] in shard_ids:
            raise ValueError(f"Mixed or duplicate shards: {directory}")
        shard_ids.add(complete["shard_index"])
        path = directory / "scenes.jsonl"
        digest, found = hashlib.sha256(), []
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                row = json.loads(line)
                token = row["token"]
                if not isinstance(token, str) or not token or token in tokens:
                    raise ValueError(f"Invalid/duplicate token: {token}")
                if not isinstance(row.get("log"), str) or not row["log"]:
                    raise ValueError(f"Missing log identity: {token}")
                for k in ("0", "20"):
                    poses = row["trajectories"][k]
                    if (len(poses) != 8 or any(len(p) != 3 for p in poses)
                            or not all(math.isfinite(float(v)) for p in poses for v in p)):
                        raise ValueError(f"Invalid K{k} trajectory: {token}")
                    metrics = row["metrics"][k]
                    if set(metrics) != set(means[0]) or not all(
                        math.isfinite(float(v)) and 0 <= float(v) <= 1 for v in metrics.values()
                    ):
                        raise ValueError(f"Invalid K{k} metrics: {token}")
                delta = float(row["metrics"]["20"]["score"]) - float(row["metrics"]["0"]["score"])
                if not math.isclose(delta, row["delta"]["20"]["score"], abs_tol=1e-12, rel_tol=0):
                    raise ValueError(f"Inconsistent saved delta: {token}")
                tokens.add(token)
                found.append(token)
                records.append(row)
        if sorted(found) != complete["shard_tokens"]:
            raise ValueError(f"Incomplete shard: {directory}")
        hashes[str(path)] = digest.hexdigest()
    if (run is None or run.get("split") != "navtest" or run.get("count") != expected_count
            or not {0, 20}.issubset(run.get("ks", []))
            or shard_ids != set(range(run["num_shards"]))
            or len(records) != expected_count
            or digest_tokens(sorted(tokens)) != run["token_digest"]):
        raise ValueError("Incomplete or incompatible navtest evaluation")
    for k in ("0", "20"):
        for metric, saved_mean in summary["by_k"][k]["mean"].items():
            actual = math.fsum(float(row["metrics"][k][metric]) for row in records) / expected_count
            if not math.isclose(actual, saved_mean, abs_tol=1e-10, rel_tol=0):
                raise ValueError(f"Saved summary disagrees with scene rows: K{k} {metric}")
    return records, run, {"source": str(source), "scene_sha256": hashes,
                          "summary_sha256": hashlib.sha256((source / "summary.json").read_bytes()).hexdigest()}


def select_cases(records, count=5):
    """Largest gains/losses, one scene per log when possible; deterministic ties."""
    if type(count) is not int or count < 1:
        raise ValueError("count must be positive")
    selected = []
    for name, prefix, sign in (("improved", "I", 1), ("worsened", "D", -1)):
        ranked = [r for r in records if sign * (r["metrics"]["20"]["score"] - r["metrics"]["0"]["score"]) > 1e-9]
        ranked.sort(key=lambda r: (-sign * (r["metrics"]["20"]["score"] - r["metrics"]["0"]["score"]), r["log"], r["token"]))
        if len(ranked) < count:
            raise ValueError(f"Only {len(ranked)} {name} scenes; need {count}")
        chosen, logs = [], set()
        for row in ranked:
            if row["log"] not in logs:
                chosen.append(row)
                logs.add(row["log"])
            if len(chosen) == count:
                break
        used = {r["token"] for r in chosen}
        chosen.extend(r for r in ranked if r["token"] not in used and len(chosen) < count)
        for i, row in enumerate(chosen, 1):
            selected.append({"case_id": f"{prefix}{i:02d}", "group": name, "row": row})
    return selected


def write_gallery(output, state):
    """Local, self-contained index referencing colocated GIF/PNG files."""
    esc = lambda value: html.escape(str(value), quote=True)
    cards = []
    for case in state["cases"]:
        row, status = case["row"], case["state"]
        a, b = (100 * row["metrics"][k]["score"] for k in ("0", "20"))
        body = f'<p class="status">{esc(status)}</p>'
        if status == "complete":
            gif, poster = esc(case["gif"]), esc(case["poster"])
            body = (f'<a href="{gif}"><img src="{poster}" data-gif="{gif}" alt="{esc(case["case_id"])} K0 versus K20" loading="lazy"></a>'
                    f'<p><button onclick="toggle(this)">播放 / 暂停</button> · <a href="{gif}">打开动图</a> · <a href="{poster}">静态图</a></p>')
        elif case.get("error"):
            body += f'<pre>{esc(case["error"])}</pre>'
        metrics = "".join(f'<tr><td>{esc(m)}</td><td>{row["metrics"]["0"][m]:.4f}</td><td>{row["metrics"]["20"][m]:.4f}</td></tr>'
                          for m in row["metrics"]["0"] if m != "score")
        cards.append(f'<section><h2>{esc(case["case_id"])} · {"改善" if case["group"] == "improved" else "退化"} · {b-a:+.4f} 分</h2>'
                     f'<p>K0 {a:.4f} → K20 {b:.4f} / 100 · <code>{esc(row["token"])}</code></p><p class="log">{esc(row["log"])}</p>{body}'
                     f'<details><summary>分项（0–1）</summary><table><tr><th>指标</th><th>K0</th><th>K20</th></tr>{metrics}</table></details></section>')
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>DD + Score · 500 epoch · K0 / K20</title><style>'
            'body{font-family:system-ui,sans-serif;background:#eef2f5;color:#172735;max-width:1500px;margin:24px auto;padding:0 16px}'
            'section{background:white;padding:20px;margin:22px 0;border-radius:12px}h1{font-size:28px}h2{font-size:21px}'
            'img{width:100%;height:auto}table{border-collapse:collapse}td,th{padding:5px 18px;text-align:left;border-bottom:1px solid #ddd}'
            '.log,small{color:#586775}button{padding:6px 18px;cursor:pointer}pre{white-space:pre-wrap}</style>'
            '<h1>DD + Score · 500 epoch · K0 / K20</h1>'
            '<p>左：原始 DD；右：K20 refine。相同时间、相同比例。绿色为可行驶区域；红圈标注各自发生的越界和碰撞。黑色虚线为记录中的专家轨迹。</p>'
            '<p>按 PDMS 分差排序，优先每个日志选一个场景；这些是定向选出的案例，不代表总体平均。动图展示官方控制器执行结果，细线保留规划轨迹。</p>'
            '<p>碰撞红圈表示实际接触；是否扣除 NC 分数按官方责任判定。TTC 风险不作为已发生的碰撞标注。</p>'
            f'<p>状态：{esc(state["status"])} · 来源：<code>{esc(state["source"])}</code></p>' + ''.join(cards) +
            '<script>function toggle(b){let i=b.closest("section").querySelector("img");if(!i.dataset.poster)i.dataset.poster=i.src;'
            'let play=i.dataset.play!=="1";i.src=play?i.dataset.gif:i.dataset.poster;i.dataset.play=play?"1":"0";b.textContent=play?"暂停":"播放"}</script></html>')
    atomic_text(output / "index.html", page)


def setup_runtime(root, run, cases):
    """Read metadata and metric caches only; never construct a neural planner."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import MetricCacheLoader, SceneLoader
    from navsim.common.dataclasses import SensorConfig
    from .cache import verified_devkit_root
    from .evaluate import metric_index

    devkit = verified_devkit_root(root / "DiffusionDrive")
    config_root = devkit / "navsim/planning/script/config"
    scoring = OmegaConf.load(config_root / "pdm_scoring/default_scoring_parameters.yaml")
    simulator, scorer = instantiate(scoring.simulator), instantiate(scoring.scorer)
    scene_filter = instantiate(OmegaConf.load(config_root / "common/train_test_split/scene_filter/navtest.yaml"))
    wanted = [c["row"]["token"] for c in cases]
    if not set(wanted).issubset(scene_filter.tokens):
        raise ValueError("Selected tokens are not in the official navtest split")
    selected_filter = replace(scene_filter, log_names=sorted({c["row"]["log"] for c in cases}), tokens=wanted, max_scenes=None)
    data = Path(run["data_root"])
    loader = SceneLoader(data / "navsim_logs/test", data / "sensor_blobs/test", selected_filter, SensorConfig.build_no_sensors())
    if set(loader.tokens) != set(wanted):
        raise ValueError("Selected scene metadata are missing")

    class AllNodeMetricCache(MetricCacheLoader):
        _load_metric_cache_paths = staticmethod(metric_index)

    cache = AllNodeMetricCache(Path(run["metric_cache"]))
    if not set(wanted).issubset(cache.tokens):
        raise ValueError("Selected metric caches are missing")
    return loader, cache, simulator, scorer


def run_visualization(root, source=None, count=5, radius=35.0, dpi=100):
    root = Path(root).resolve()
    exp = root / "dataset/navsim/exp"
    source = Path(source or (exp / EVAL_POINTER).read_text().strip()).resolve()
    launcher = read_json(source / "launcher.json")
    if launcher.get("epochs") != 500 or launcher.get("status") != "complete":
        raise ValueError("Require the completed aligned 500-epoch evaluation")
    records, run, provenance = load_evaluation(source)
    provenance.update(launcher=launcher, evaluation_protocol=run)
    cases = select_cases(records, count)
    del records
    output = Path(tempfile.mkdtemp(prefix="dd_score_ddaligned_e500_visuals_", dir=exp))
    state = {"render_version": RENDER_VERSION, "source": str(source), "status": "running",
             "created": datetime.now(timezone.utc).isoformat(), "selection": "largest signed PDMS changes; distinct logs preferred",
             "provenance": provenance, "cases": [{**c, "state": "pending"} for c in cases]}
    write_json(output / "selection.json", state)
    write_gallery(output, state)
    atomic_text(exp / VIS_POINTER, str(output) + "\n")
    print(f"Output: {output}\nGallery: {output / 'index.html'}", flush=True)
    for case in cases:
        row = case["row"]
        a, b = (100 * row["metrics"][k]["score"] for k in ("0", "20"))
        print(f"Selected {case['case_id']}: {row['token']} K0={a:.4f} K20={b:.4f} delta={b-a:+.4f} {row['log']}", flush=True)
    try:
        loader, cache, simulator, scorer = setup_runtime(root, run, cases)
        from .visualization_replay import replay_case
        from .visualization_render import render_case
        for case in state["cases"]:
            row, case_id = case["row"], case["case_id"]
            print(f"Replaying {case_id} {row['token']}", flush=True)
            try:
                scene = loader.get_scene_from_token(row["token"])
                if scene.scene_metadata.initial_token != row["token"] or scene.scene_metadata.log_name != row["log"]:
                    raise ValueError("Scene token/log identity mismatch")
                metric = cache.get_from_token(row["token"])
                # Both expert and cached simulation must use exactly this rear-axle anchor.
                import numpy as np
                anchor = scene.frames[scene.scene_metadata.num_history_frames - 1].ego_status.ego_pose
                rear = metric.ego_state.rear_axle
                if not np.allclose(anchor[:2], [rear.x, rear.y], rtol=0, atol=1e-4) or abs(math.atan2(math.sin(anchor[2]-rear.heading), math.cos(anchor[2]-rear.heading))) > 1e-5:
                    raise ValueError("Scene and metric cache have different initial ego poses")
                payload = replay_case(row, metric, simulator, scorer, expert_poses=scene.get_future_trajectory(8).poses)
                payload["metadata"].update(case_id=case_id, epoch=500, group=case["group"])
                stem = f"{case_id}_{row['token']}"
                write_json(output / f"{stem}.json", payload)
                result = render_case(payload, output / f"{stem}.gif", output / f"{stem}.png", radius=radius, dpi=dpi, case_id=case_id)
                case.update(state="complete", gif=f"{stem}.gif", poster=f"{stem}.png", payload=f"{stem}.json", rendering=result)
                print(f"Ready: {output / case['gif']}", flush=True)
            except Exception as error:
                case.update(state="failed", error=f"{type(error).__name__}: {error}")
                traceback.print_exc()
            write_json(output / "selection.json", state)
            write_gallery(output, state)
        failures = [c["case_id"] for c in state["cases"] if c["state"] != "complete"]
        state["status"] = "failed" if failures else "complete"
        if failures:
            raise RuntimeError(f"Failed cases: {', '.join(failures)}; inspect {output / 'selection.json'}")
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "selection.json", state)
        write_gallery(output, state)
    print(f"COMPLETE: {len(cases)} cases\nOpen: {output / 'index.html'}", flush=True)
    return output


def status(root):
    pointer = Path(root) / "dataset/navsim/exp" / VIS_POINTER
    if not pointer.is_file():
        print("尚未启动可视化")
        return
    output = Path(pointer.read_text().strip())
    state = read_json(output / "selection.json")
    print(f"{state['status']}: {sum(c['state'] == 'complete' for c in state['cases'])}/{len(state['cases'])} cases")
    print(f"Gallery: {output / 'index.html'}")
    for c in state["cases"]:
        row = c["row"]
        a, b = (100 * row["metrics"][k]["score"] for k in ("0", "20"))
        print(f"{c['case_id']} {c['state']:8} {row['token']} K0={a:.4f} K20={b:.4f} delta={b-a:+.4f}" + (f" {c['error']}" if c.get('error') else ""))
    if state.get("error"):
        print(state["error"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=["run", "status"], default="run")
    parser.add_argument("--root", type=Path, default=Path("/home/data/wyk"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--count", type=int, default=5, help="cases per improvement/degradation group")
    parser.add_argument("--radius", type=float, default=35.0)
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_args(argv)
    if args.count < 1 or not math.isfinite(args.radius) or args.radius < 10 or args.dpi < 50:
        parser.error("count>=1, radius>=10m and dpi>=50 required")
    if args.action == "status":
        status(args.root)
        return
    exp = args.root / "dataset/navsim/exp"
    with (exp / "dd_score_ddaligned_e500_visuals.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, "Another visualization process is active\n")
        run_visualization(args.root, args.source, args.count, args.radius, args.dpi)


if __name__ == "__main__":
    main()
