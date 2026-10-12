#!/usr/bin/env python3
"""Render Chinese reports from existing captures, without model/data loading."""

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_SOURCE = Path("/home/data/wyk/dataset/nuplan/exp/score_G_static_context_1012_v1")
PRIORITY = ("I01", "W01", "K01", "K02", "K04", "W04")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, help="Default: SOURCE/report_cn_v2")
    parser.add_argument("--cases", nargs="+", default=list(PRIORITY), help="Case IDs or all")
    parser.add_argument("--frames", nargs="+", type=int, help="Optional selected history indices")
    parser.add_argument("--ks", nargs="+", type=int, default=[20, 50, 200])
    parser.add_argument(
        "--include-curves", action="store_true", help="Separate optional displacement appendix"
    )
    parser.add_argument(
        "--font", type=Path, help="Optional full Chinese font; bundled subset used by default"
    )
    args = parser.parse_args()
    os.environ["MPLBACKEND"] = "Agg"
    from matplotlib.backends.backend_pdf import PdfPages
    from static_context_plots import render_case

    root = args.source.resolve() / "captures"
    cases = (
        sorted(p.name for p in root.iterdir() if p.is_dir())
        if args.cases == ["all"]
        else list(dict.fromkeys(args.cases))
    )
    paths = []
    missing = []
    for case in cases:
        if Path(case).name != case or case in (".", ".."):
            parser.error("Case IDs must be directory names, not paths")
        files = sorted((root / case).glob("frame_*/payload.json"))
        if args.frames is not None:
            files = [p for p in files if json.loads(p.read_text())["frame_index"] in args.frames]
        if not files:
            missing.append(case)
        paths.extend(files)
    if not paths:
        parser.error(f"No existing captures found under {root}; selected cases: {cases}")
    if missing:
        print("尚无可用数据的场景：" + ", ".join(missing), flush=True)
    ks = tuple(sorted(set(args.ks)))
    if not ks or any(k <= 0 for k in ks):
        parser.error("--ks must contain positive refinement counts")
    output = (args.output or args.source / "report_cn_v2").resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = output / "trajectory_report_cn.pdf"
    temporary = output / f"trajectory_report_cn.tmp.{os.getpid()}.pdf"
    index = []
    # Keep completed original reports intact if rendering fails midway.
    try:
        with PdfPages(
            temporary, metadata={"Title": "S05-G 最终模型：轨迹推动", "Author": "Score Function"}
        ) as pdf:
            for number, path in enumerate(paths, 1):
                payload = json.loads(path.read_text())
                absent = [k for k in ks if str(k) not in payload["trajectories"]]
                if absent:
                    raise ValueError(
                        f"{path} has no captured K={absent}; do not rerun inference for layout changes"
                    )
                case = payload["case_id"]
                frame = int(payload["frame_index"])
                folder = output / "figures" / case / f"frame_{frame:04d}"
                print(f"[{number}/{len(paths)}] 场景 {case} / 帧 {frame} / K={ks}", flush=True)
                result = render_case(
                    payload,
                    folder,
                    shared_pdf=pdf,
                    ks=ks,
                    include_curves=args.include_curves,
                    font_path=args.font,
                )
                index.append(dict(case_id=case, frame_index=frame, source=str(path), **result))
        temporary.replace(report)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    (output / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"完成：{len(paths)} 帧\n合并 PDF：{report}\n单页 PNG / 分帧 PDF：{output / 'figures'}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
