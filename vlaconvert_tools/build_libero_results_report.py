#!/usr/bin/env python3
"""Build CSV/Markdown/HTML plots from archived LIBERO result folders."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Any


SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_LABELS = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "Long",
}
MODE_ORDER = ["fp16", "fake", "real"]
MODE_COLORS = {
    "fp16": "#6b7280",
    "fake": "#2563eb",
    "real": "#dc2626",
    "unknown": "#9333ea",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results", help="Archived results root.")
    parser.add_argument("--output-dir", default=None, help="Output dir. Defaults to results/plots.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_nested(data: dict[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def infer_suite_from_name(name: str) -> str | None:
    for suite in SUITE_ORDER:
        if suite in name:
            return suite
    return None


def infer_mode(path: Path, summary: dict[str, Any] | None, suite_dir: Path) -> str:
    if summary and summary.get("result_tag"):
        tag = str(summary["result_tag"]).lower()
        if tag in MODE_ORDER:
            return tag

    text = "/".join(part.lower() for part in path.parts)
    for mode in MODE_ORDER:
        if f"_{mode}" in path.name.lower() or f"/{mode}_" in text or f"/{mode}/" in text:
            return mode

    for report in suite_dir.glob("quantvla_converted_*_replacement_report.json"):
        match = re.search(r"quantvla_converted_([^_]+)_", report.name)
        if match:
            return match.group(1)

    return "unknown"


def parse_success_from_log(log_path: Path) -> tuple[int | None, int | None, float | None]:
    if not log_path.exists():
        return None, None, None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    successes = re.findall(r"# successes:\s+(\d+)\s+\(([\d.]+)%\)", text)
    episodes = re.findall(r"# episodes completed so far:\s+(\d+)", text)
    if not successes:
        return None, int(episodes[-1]) if episodes else None, None
    return (
        int(successes[-1][0]),
        int(episodes[-1]) if episodes else None,
        float(successes[-1][1]),
    )


def find_log_for_summary(summary_path: Path, suite: str, mode: str) -> Path:
    candidates = [
        summary_path.with_name(f"libero_eval_{suite}_{mode}.log"),
        summary_path.with_name(f"libero_eval_{suite}.log"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def collect_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for summary_path in sorted(results_dir.rglob("*_latency_summary.json")):
        suite = infer_suite_from_name(summary_path.name)
        if suite is None:
            suite = infer_suite_from_name(str(summary_path.parent))
        if suite is None:
            continue

        summary = read_json(summary_path) or {}
        mode = infer_mode(summary_path, summary, summary_path.parent)
        run = summary_path.relative_to(results_dir).parts[0]
        key = (run, mode, suite, str(summary_path))
        if key in seen:
            continue
        seen.add(key)

        log_successes, log_episodes, log_success_pct = parse_success_from_log(
            find_log_for_summary(summary_path, suite, mode)
        )
        success_rate = summary.get("success_rate")
        success_pct = float(success_rate) * 100.0 if success_rate is not None else log_success_pct

        report_path = summary_path.with_name(
            f"quantvla_converted_{mode}_{suite}_replacement_report.json"
        )
        report = read_json(report_path) or {}

        row = {
            "run": run,
            "mode": mode,
            "suite": suite,
            "suite_label": SUITE_LABELS.get(suite, suite),
            "success_rate_pct": success_pct,
            "num_successes": summary.get("num_successes", log_successes),
            "num_episodes": summary.get("num_episodes", log_episodes),
            "num_action_steps": get_nested(summary, "overall", "num_action_steps"),
            "step_total_ms_mean": get_nested(summary, "overall", "step_total_ms", "mean"),
            "step_total_ms_median": get_nested(summary, "overall", "step_total_ms", "median"),
            "step_total_ms_p90": get_nested(summary, "overall", "step_total_ms", "p90"),
            "step_total_ms_p99": get_nested(summary, "overall", "step_total_ms", "p99"),
            "client_roundtrip_ms_mean": get_nested(
                summary, "overall", "client_roundtrip_ms", "mean"
            ),
            "policy_total_ms_mean": get_nested(summary, "overall", "policy_total_ms", "mean"),
            "policy_model_get_action_ms_mean": get_nested(
                summary, "overall", "policy_model_get_action_ms", "mean"
            ),
            "policy_apply_transforms_ms_mean": get_nested(
                summary, "overall", "policy_apply_transforms_ms", "mean"
            ),
            "env_step_ms_mean": get_nested(summary, "overall", "env_step_ms", "mean"),
            "successfully_replaced": report.get("successfully_replaced"),
            "fallback_fp16_layers_count": report.get("fallback_fp16_layers_count"),
            "summary_path": str(summary_path),
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            MODE_ORDER.index(row["mode"]) if row["mode"] in MODE_ORDER else 99,
            SUITE_ORDER.index(row["suite"]) if row["suite"] in SUITE_ORDER else 99,
            row["run"],
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "run",
        "mode",
        "suite",
        "suite_label",
        "success_rate_pct",
        "num_successes",
        "num_episodes",
        "num_action_steps",
        "step_total_ms_mean",
        "step_total_ms_median",
        "step_total_ms_p90",
        "step_total_ms_p99",
        "client_roundtrip_ms_mean",
        "policy_total_ms_mean",
        "policy_model_get_action_ms_mean",
        "policy_apply_transforms_ms_mean",
        "env_step_ms_mean",
        "successfully_replaced",
        "fallback_fp16_layers_count",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "mode",
        "suite_label",
        "success_rate_pct",
        "step_total_ms_mean",
        "step_total_ms_p90",
        "step_total_ms_p99",
        "policy_model_get_action_ms_mean",
        "client_roundtrip_ms_mean",
        "env_step_ms_mean",
        "successfully_replaced",
        "fallback_fp16_layers_count",
    ]
    lines = ["# LIBERO Accuracy And Latency Summary", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    lines.extend(
        [
            "",
            "## How To Read",
            "",
            "- `success_rate_pct`: LIBERO success rate.",
            "- `step_total_ms_*`: end-to-end action step latency.",
            "- `policy_model_get_action_ms_mean`: model-side action generation time.",
            "- `client_roundtrip_ms_mean`: client/server request overhead.",
            "- `env_step_ms_mean`: simulator step time.",
            "- Compare FakeQuant vs RealQuant mainly with `policy_model_get_action_ms_mean` and `step_total_ms_mean/p90/p99`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bar_svg(rows: list[dict[str, Any]], metric: str, title: str, ylabel: str) -> str:
    data = [row for row in rows if row.get(metric) is not None]
    if not data:
        return f"<section><h2>{html.escape(title)}</h2><p>No data.</p></section>"
    width, height = 920, 360
    margin_left, margin_bottom, margin_top = 72, 68, 42
    plot_w = width - margin_left - 24
    plot_h = height - margin_top - margin_bottom
    max_value = max(float(row[metric]) for row in data) or 1.0

    labels = []
    for suite in SUITE_ORDER:
        for mode in MODE_ORDER:
            match = next((r for r in data if r["suite"] == suite and r["mode"] == mode), None)
            if match:
                labels.append(match)
    labels.extend([row for row in data if row not in labels])
    duplicate_counts: dict[tuple[str, str], int] = {}
    for row in data:
        duplicate_counts[(row["suite"], row["mode"])] = duplicate_counts.get((row["suite"], row["mode"]), 0) + 1

    bar_gap = 5
    bar_w = max(10, (plot_w - bar_gap * max(0, len(labels) - 1)) / max(1, len(labels)))
    parts = [
        f"<section><h2>{html.escape(title)}</h2>",
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>",
        f"<text x='{width/2}' y='22' text-anchor='middle' class='chart-title'>{html.escape(title)}</text>",
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{height-margin_bottom}' class='axis'/>",
        f"<line x1='{margin_left}' y1='{height-margin_bottom}' x2='{width-18}' y2='{height-margin_bottom}' class='axis'/>",
        f"<text x='18' y='{margin_top + plot_h/2}' transform='rotate(-90 18 {margin_top + plot_h/2})' text-anchor='middle' class='axis-label'>{html.escape(ylabel)}</text>",
    ]
    for idx, row in enumerate(labels):
        value = float(row[metric])
        x = margin_left + idx * (bar_w + bar_gap)
        h = (value / max_value) * plot_h
        y = margin_top + plot_h - h
        color = MODE_COLORS.get(row["mode"], MODE_COLORS["unknown"])
        label = f"{SUITE_LABELS.get(row['suite'], row['suite'])} {row['mode']}"
        if duplicate_counts.get((row["suite"], row["mode"]), 0) > 1:
            label = f"{label} {row['run']}"
        parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{h:.1f}' fill='{color}'/>")
        parts.append(f"<text x='{x + bar_w/2:.1f}' y='{y - 4:.1f}' text-anchor='middle' class='value'>{value:.1f}</text>")
        parts.append(f"<text x='{x + bar_w/2:.1f}' y='{height - 45}' text-anchor='end' transform='rotate(-35 {x + bar_w/2:.1f} {height - 45})' class='tick'>{html.escape(label)}</text>")
    parts.append("</svg></section>")
    return "\n".join(parts)


def horizontal_chart(rows: list[dict[str, Any]], metric: str, title: str, unit: str) -> str:
    data = [row for row in rows if row.get(metric) is not None]
    if not data:
        return f"<section class='quick-chart'><h2>{html.escape(title)}</h2><p>No data.</p></section>"

    max_value = max(float(row[metric]) for row in data) or 1.0
    row_html = []
    for suite in SUITE_ORDER:
        suite_rows = [row for row in data if row["suite"] == suite]
        if not suite_rows:
            continue
        row_html.append(
            f"<div class='suite-label'>{html.escape(SUITE_LABELS.get(suite, suite))}</div>"
        )
        for row in sorted(
            suite_rows,
            key=lambda item: MODE_ORDER.index(item["mode"]) if item["mode"] in MODE_ORDER else 99,
        ):
            value = float(row[metric])
            width_pct = max(1.0, value / max_value * 100.0)
            color = MODE_COLORS.get(row["mode"], MODE_COLORS["unknown"])
            row_html.append(
                "<div class='bar-row'>"
                f"<div class='bar-name'>{html.escape(row['mode'])}</div>"
                "<div class='bar-track'>"
                f"<div class='bar-fill' style='width:{width_pct:.2f}%; background:{color}'></div>"
                "</div>"
                f"<div class='bar-value'>{value:.2f} {html.escape(unit)}</div>"
                "</div>"
            )
    return (
        "<section class='quick-chart'>"
        f"<h2>{html.escape(title)}</h2>"
        + "".join(row_html)
        + "</section>"
    )


def write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    quick_sections = [
        horizontal_chart(rows, "success_rate_pct", "Accuracy", "%"),
        horizontal_chart(rows, "policy_model_get_action_ms_mean", "Model Latency Mean", "ms"),
        horizontal_chart(rows, "step_total_ms_mean", "End-to-End Step Latency Mean", "ms"),
    ]
    sections = [
        bar_svg(rows, "success_rate_pct", "Accuracy By Suite", "Success (%)"),
        bar_svg(rows, "policy_model_get_action_ms_mean", "Model Latency Mean", "ms"),
        bar_svg(rows, "step_total_ms_mean", "End-to-End Step Latency Mean", "ms"),
        bar_svg(rows, "step_total_ms_p90", "End-to-End Step Latency P90", "ms"),
        bar_svg(rows, "step_total_ms_p99", "End-to-End Step Latency P99", "ms"),
        bar_svg(rows, "client_roundtrip_ms_mean", "Client Roundtrip Mean", "ms"),
        bar_svg(rows, "env_step_ms_mean", "Environment Step Mean", "ms"),
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(fmt(row.get(col)))}</td>"
                for col in [
                    "mode",
                    "suite_label",
                    "success_rate_pct",
                    "step_total_ms_mean",
                    "step_total_ms_p90",
                    "step_total_ms_p99",
                    "policy_model_get_action_ms_mean",
                    "client_roundtrip_ms_mean",
                    "env_step_ms_mean",
                ]
            )
            + "</tr>"
        )
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>LIBERO QuantVLA Converted Results</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
    h1 {{ margin-bottom: 4px; }}
    .note {{ color: #4b5563; max-width: 920px; line-height: 1.45; }}
    .quick-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; margin: 24px 0 36px; }}
    .quick-chart {{ border: 1px solid #d1d5db; padding: 16px; background: #ffffff; }}
    .quick-chart h2 {{ margin: 0 0 14px; font-size: 18px; }}
    .suite-label {{ font-weight: 700; margin: 12px 0 6px; color: #111827; }}
    .bar-row {{ display: grid; grid-template-columns: 54px 1fr 92px; gap: 8px; align-items: center; margin: 5px 0; }}
    .bar-name {{ font-size: 12px; color: #374151; }}
    .bar-track {{ height: 14px; background: #e5e7eb; overflow: hidden; }}
    .bar-fill {{ height: 100%; }}
    .bar-value {{ font-size: 12px; text-align: right; color: #111827; }}
    section {{ margin: 28px 0 42px; }}
    svg {{ width: 100%; max-width: 1040px; height: auto; border: 1px solid #e5e7eb; }}
    .axis {{ stroke: #374151; stroke-width: 1; }}
    .chart-title {{ font-size: 16px; font-weight: 700; }}
    .axis-label, .tick {{ font-size: 11px; fill: #374151; }}
    .value {{ font-size: 10px; fill: #111827; }}
    table {{ border-collapse: collapse; margin-top: 20px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; font-size: 13px; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>LIBERO QuantVLA Converted Results</h1>
  <p class="note">
    Use <code>policy_model_get_action_ms_mean</code> to compare FakeQuant vs RealQuant model-side speed.
    Use <code>step_total_ms_mean/p90/p99</code> for end-to-end robot action-step latency.
  </p>
  <div class="quick-grid">
    {''.join(quick_sections)}
  </div>
  {''.join(sections)}
  <h2>Summary Table</h2>
  <table>
    <thead>
      <tr>
        <th>mode</th><th>suite</th><th>success %</th><th>step mean</th><th>step p90</th><th>step p99</th>
        <th>model mean</th><th>roundtrip mean</th><th>env mean</th>
      </tr>
    </thead>
    <tbody>
      {''.join(table_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(results_dir)
    write_csv(output_dir / "libero_summary.csv", rows)
    write_markdown(output_dir / "libero_summary.md", rows)
    write_html(output_dir / "libero_report.html", rows)

    print(f"Found {len(rows)} result rows")
    print(f"CSV: {output_dir / 'libero_summary.csv'}")
    print(f"Markdown: {output_dir / 'libero_summary.md'}")
    print(f"HTML: {output_dir / 'libero_report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
