"""Readable reports for offline corruption diagnostics."""

from pathlib import Path


def format_metric(value):
    return "undefined" if value is None else f"{value:.6g}"


def write_diagnostic_report(path, summary):
    cosine, diagnostics = summary["cosine"], summary["diagnostics"]
    lines = [
        "# Score Function: offline corruption diagnostics",
        "",
        f"Split: `{summary['split']}`; unique frames: {summary['samples']}; "
        f"fixed corruptions per frame: {summary['noise_repeats']}; sigma: {summary['sigma']}.",
        "",
        f"DSM: {format_metric(diagnostics['dsm']['mean'])}. "
        f"Direction cosine mean/median: {format_metric(cosine['mean'])}/{format_metric(cosine['median'])}; "
        f"positive fraction: {format_metric(cosine['fraction_positive'])}; "
        f"undefined (zero norm): {cosine['undefined_count']}.",
        "",
        f"Score D1/D2: {format_metric(diagnostics['score_d1']['mean'])}/"
        f"{format_metric(diagnostics['score_d2']['mean'])}. These are diagnostics only.",
        "",
        "All one-step recovery below is **unprojected**. Gamma=1 is the Tweedie-style estimate.",
        "",
        "| gamma | normalized MSE | ADE (m) | FDE (m) | heading MAE (rad) | MSE improved fraction |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    before = summary["corrupted_baseline"]
    columns = ("mse_normalized", "ade_m", "fde_m", "heading_mae_rad")
    lines.append(
        "| before | " + " | ".join(format_metric(before[key]["mean"]) for key in columns) + " | — |"
    )
    for gamma, metrics in summary["recovery"].items():
        lines.append(
            f"| {gamma} | "
            + " | ".join(format_metric(metrics[key]["mean"]) for key in columns)
            + f" | {format_metric(metrics['improved_mse_normalized']['mean'])} |"
        )
    lines += [
        "",
        *[f"- {note}" for note in summary["interpretation"]],
        "",
        "Per-sample CSV/JSONL include improvement fractions for all four errors. "
        "The plot subset is fixed by cache ordering. No nuPlan evaluation was run by this command.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
