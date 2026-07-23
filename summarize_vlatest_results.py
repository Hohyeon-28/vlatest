"""Summarize vlatest LIBERO accuracy, latency, and DiT MLP probe outputs.

This is a local reporting helper for experiment folders copied from the server.
It reads:
  - eval_driver.log
  - results/*_latency_summary.json
  - dit_mlp_probe/dit_mlp_up_probe*.csv
  - dit_mlp_probe/dit_mlp_probe*.csv

and writes:
  - accuracy_latency_summary.csv
  - episode_steps_summary.csv
  - task_steps_summary.csv
  - mlp_probe_summary.csv
  - vlatest_experiment_report.html
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path


SUITE_ORDER = {"libero_spatial": 0, "libero_object": 1, "libero_goal": 2}
METHOD_ORDER = {"Fake W4A8": 0, "Fake W4A16": 1, "Real W4A16 Marlin": 2}


def mean(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return sum(values) / len(values) if values else float("nan")


def fmt(value: float, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def suite_label(suite: str) -> str:
    return {
        "libero_spatial": "Spatial",
        "libero_object": "Object",
        "libero_goal": "Goal",
        "libero_10": "Long",
    }.get(suite, suite)


def infer_method(path: Path) -> str:
    text = str(path).lower()
    if "fake_w4a8" in text or "fakew4a8" in text:
        return "Fake W4A8"
    if "fake_w4a16" in text or "fakew4a16" in text:
        return "Fake W4A16"
    if "real" in text or "marlin" in text:
        return "Real W4A16 Marlin"
    return "Unknown"


def parse_eval_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"# episodes completed so far:\s*(\d+).*?# successes:\s*(\d+)\s*\(([\d.]+)%\)", text, re.S)
    if matches:
        episodes, successes, pct = matches[-1]
        return {
            "episodes_from_log": int(episodes),
            "successes_from_log": int(successes),
            "success_rate_from_log": float(pct) / 100.0,
        }
    return {}


def parse_episode_successes(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    successes = re.findall(r"Success:\s*(True|False)", text)
    out: list[dict] = []
    for idx, value in enumerate(successes, start=1):
        out.append({"global_episode": idx, "success": value == "True"})
    return out


def load_replacement(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        "target_linear_layers": data.get("target_linear_layers"),
        "successfully_replaced": data.get("successfully_replaced"),
        "fallback_fp16_layers_count": data.get("fallback_fp16_layers_count"),
        "unreplaced_target_layers_count": data.get("unreplaced_target_layers_count"),
    }


def find_experiment_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("eval_driver.log"):
        exp_dir = path.parent
        if "YYYYMMDD" in str(exp_dir):
            continue
        if list((exp_dir / "results").glob("*_latency_summary.json")):
            out.append(exp_dir)
    return out


def read_accuracy_latency(exp_dir: Path) -> dict:
    summary_path = next((exp_dir / "results").glob("*_latency_summary.json"))
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    method = infer_method(exp_dir)
    suite = data.get("task_suite", "")
    overall = data.get("overall", {})
    row = {
        "method": method,
        "suite": suite,
        "suite_label": suite_label(suite),
        "exp_dir": str(exp_dir),
        "summary_path": str(summary_path),
        "episodes": data.get("num_episodes", ""),
        "successes": data.get("num_successes", ""),
        "success_rate": float(data.get("success_rate", float("nan"))),
        "num_action_steps": overall.get("num_action_steps", ""),
        "policy_model_get_action_ms_mean": overall.get("policy_model_get_action_ms", {}).get("mean", float("nan")),
        "policy_total_ms_mean": overall.get("policy_total_ms", {}).get("mean", float("nan")),
        "client_get_action_ms_mean": overall.get("client_get_action_ms", {}).get("mean", float("nan")),
        "env_step_ms_mean": overall.get("env_step_ms", {}).get("mean", float("nan")),
        "step_total_ms_mean": overall.get("step_total_ms", {}).get("mean", float("nan")),
        "step_total_ms_p90": overall.get("step_total_ms", {}).get("p90", float("nan")),
        "step_total_ms_p99": overall.get("step_total_ms", {}).get("p99", float("nan")),
    }
    row.update(parse_eval_log(exp_dir / "eval_driver.log"))
    row.update(load_replacement(exp_dir / "replacement_report.json"))
    return row


def read_episode_steps(exp_dir: Path, method: str, suite: str) -> list[dict]:
    csv_paths = list((exp_dir / "results").glob("*_latency_steps.csv"))
    if not csv_paths:
        return []
    success_by_episode = {
        row["global_episode"]: row["success"] for row in parse_episode_successes(exp_dir / "eval_driver.log")
    }
    grouped: dict[int, dict] = {}
    with csv_paths[0].open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                global_episode = int(row["global_episode"])
                task_id = int(row["task_id"])
            except Exception:
                continue
            item = grouped.setdefault(
                global_episode,
                {
                    "method": method,
                    "suite": suite,
                    "suite_label": suite_label(suite),
                    "global_episode": global_episode,
                    "task_id": task_id,
                    "task_description": row.get("task_description", ""),
                    "steps": 0,
                    "step_total_ms_values": [],
                    "policy_model_get_action_ms_values": [],
                },
            )
            item["steps"] += 1
            for key in ["step_total_ms", "policy_model_get_action_ms"]:
                try:
                    item[f"{key}_values"].append(float(row[key]))
                except Exception:
                    pass
    out: list[dict] = []
    for global_episode, item in sorted(grouped.items()):
        step_values = item.pop("step_total_ms_values")
        model_values = item.pop("policy_model_get_action_ms_values")
        success = success_by_episode.get(global_episode)
        item.update(
            {
                "success": success,
                "step_total_ms_mean": mean(step_values),
                "policy_model_get_action_ms_mean": mean(model_values),
            }
        )
        out.append(item)
    return out


def summarize_episode_steps(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["suite"])].append(row)

    out: list[dict] = []
    for (method, suite), group in grouped.items():
        success_rows = [r for r in group if r.get("success") is True]
        fail_rows = [r for r in group if r.get("success") is False]
        out.append(
            {
                "method": method,
                "suite": suite,
                "suite_label": suite_label(suite),
                "episodes": len(group),
                "success_episodes": len(success_rows),
                "failed_episodes": len(fail_rows),
                "steps_mean_all": mean([float(r["steps"]) for r in group]),
                "steps_mean_success": mean([float(r["steps"]) for r in success_rows]),
                "steps_mean_fail": mean([float(r["steps"]) for r in fail_rows]),
                "step_total_ms_mean_success": mean([float(r["step_total_ms_mean"]) for r in success_rows]),
                "policy_model_get_action_ms_mean_success": mean(
                    [float(r["policy_model_get_action_ms_mean"]) for r in success_rows]
                ),
            }
        )
    return sorted(out, key=lambda r: (METHOD_ORDER.get(r["method"], 99), SUITE_ORDER.get(r["suite"], 99)))


def summarize_task_steps(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["suite"], row["task_id"], row["task_description"])].append(row)

    out: list[dict] = []
    for (method, suite, task_id, task_description), group in grouped.items():
        success_rows = [r for r in group if r.get("success") is True]
        out.append(
            {
                "method": method,
                "suite": suite,
                "suite_label": suite_label(suite),
                "task_id": task_id,
                "task_description": task_description,
                "episodes": len(group),
                "successes": len(success_rows),
                "success_rate": len(success_rows) / len(group) if group else float("nan"),
                "steps_mean_all": mean([float(r["steps"]) for r in group]),
                "steps_mean_success": mean([float(r["steps"]) for r in success_rows]),
            }
        )
    return sorted(
        out,
        key=lambda r: (METHOD_ORDER.get(r["method"], 99), SUITE_ORDER.get(r["suite"], 99), int(r["task_id"])),
    )


def read_mlp_probe(exp_dir: Path) -> list[dict]:
    rows: list[dict] = []
    csv_paths = list((exp_dir / "dit_mlp_probe").glob("dit_mlp_up_probe*.csv"))
    csv_paths.extend((exp_dir / "dit_mlp_probe").glob("dit_mlp_probe*.csv"))
    for csv_path in sorted(set(csv_paths)):
        if csv_path.stat().st_size <= 101:
            continue
        if "pair" in csv_path.name:
            continue
        method = infer_method(exp_dir)
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append(
                        {
                            "method": method,
                            "suite": row.get("task", ""),
                            "suite_label": suite_label(row.get("task", "")),
                            "exp_dir": str(exp_dir),
                            "target": row.get("target", "up_proj"),
                            "layer_index": int(row["layer_index"]),
                            "denoise_iter": int(row["denoise_iter"]),
                            "iter_label": row.get("iter_label", f"iter_{row['denoise_iter']}"),
                            "channel_bin": int(row["channel_bin"]),
                            "calls": int(float(row.get("calls", "0") or 0)),
                            "mean": float(row["mean"]),
                            "abs_mean": float(row["abs_mean"]),
                            "std": float(row["std"]),
                            "rms": float(row["rms"]),
                            "min": float(row["min"]),
                            "max": float(row["max"]),
                        }
                    )
                except Exception:
                    continue
    return rows


def summarize_mlp(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["suite"], row.get("target", "up_proj"))].append(row)

    out: list[dict] = []
    for (method, suite, target), group in grouped.items():
        out.append(
            {
                "method": method,
                "suite": suite,
                "suite_label": suite_label(suite),
                "target": target,
                "probe_rows": len(group),
                "layers_seen": len({r["layer_index"] for r in group}),
                "iters_seen": ",".join(str(x) for x in sorted({r["denoise_iter"] for r in group})),
                "calls_mean": mean([float(r["calls"]) for r in group]),
                "abs_mean_avg": mean([r["abs_mean"] for r in group]),
                "rms_avg": mean([r["rms"] for r in group]),
                "std_avg": mean([r["std"] for r in group]),
                "min_seen": min((r["min"] for r in group), default=float("nan")),
                "max_seen": max((r["max"] for r in group), default=float("nan")),
            }
        )
    return sorted(
        out,
        key=lambda r: (METHOD_ORDER.get(r["method"], 99), SUITE_ORDER.get(r["suite"], 99), r.get("target", "")),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def chart_accuracy(rows: list[dict]) -> str:
    bars = []
    width = 760
    row_h = 26
    left = 150
    top = 24
    max_w = 560
    sorted_rows = sorted(rows, key=lambda r: (SUITE_ORDER.get(r["suite"], 99), METHOD_ORDER.get(r["method"], 99)))
    height = top * 2 + len(sorted_rows) * row_h
    for i, row in enumerate(sorted_rows):
        y = top + i * row_h
        value = float(row["success_rate"]) * 100.0
        bar_w = max(0.0, min(100.0, value)) / 100.0 * max_w
        label = f"{row['suite_label']} / {row['method']}"
        bars.append(f'<text x="4" y="{y + 17}" class="axis">{html.escape(label)}</text>')
        bars.append(f'<rect x="{left}" y="{y + 5}" width="{bar_w:.1f}" height="15" class="bar"></rect>')
        bars.append(f'<text x="{left + bar_w + 6:.1f}" y="{y + 17}" class="value">{value:.1f}%</text>')
    return f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Accuracy by method and suite">{"".join(bars)}</svg>'


def chart_latency(rows: list[dict], metric: str, title: str) -> str:
    values = [float(r.get(metric, float("nan"))) for r in rows if math.isfinite(float(r.get(metric, float("nan"))))]
    vmax = max(values) if values else 1.0
    bars = []
    width = 760
    row_h = 26
    left = 190
    top = 24
    max_w = 520
    sorted_rows = sorted(rows, key=lambda r: (SUITE_ORDER.get(r["suite"], 99), METHOD_ORDER.get(r["method"], 99)))
    height = top * 2 + len(sorted_rows) * row_h
    for i, row in enumerate(sorted_rows):
        y = top + i * row_h
        value = float(row.get(metric, float("nan")))
        bar_w = 0 if not math.isfinite(value) else value / vmax * max_w
        label = f"{row['suite_label']} / {row['method']}"
        bars.append(f'<text x="4" y="{y + 17}" class="axis">{html.escape(label)}</text>')
        bars.append(f'<rect x="{left}" y="{y + 5}" width="{bar_w:.1f}" height="15" class="bar latency"></rect>')
        bars.append(f'<text x="{left + bar_w + 6:.1f}" y="{y + 17}" class="value">{fmt(value, 1)} ms</text>')
    return f'<h2>{html.escape(title)}</h2><svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">{"".join(bars)}</svg>'


def profile_series(rows: list[dict], method: str, suite: str, iter_label: str, metric: str = "abs_mean") -> list[tuple[int, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] == method and row["suite"] == suite and row["iter_label"] == iter_label:
            grouped[row["channel_bin"]].append(float(row[metric]))
    return [(k, mean(v)) for k, v in sorted(grouped.items())]


def chart_mlp_profiles(rows: list[dict], metric: str = "abs_mean") -> str:
    methods = sorted({r["method"] for r in rows}, key=lambda m: METHOD_ORDER.get(m, 99))
    suites = sorted({r["suite"] for r in rows}, key=lambda s: SUITE_ORDER.get(s, 99))
    labels = ["first", "mid", "last"]
    colors = {
        "Fake W4A8": "#4763ff",
        "Fake W4A16": "#12a878",
        "Real W4A16 Marlin": "#df4f4f",
    }
    all_values = [float(r[metric]) for r in rows if math.isfinite(float(r[metric]))]
    ymax = max(all_values) if all_values else 1.0
    panels = []
    panel_w, panel_h = 330, 190
    pad_l, pad_t, pad_r, pad_b = 36, 24, 10, 28
    for suite in suites:
        for iter_label in labels:
            paths = []
            legend = []
            for method in methods:
                series = profile_series(rows, method, suite, iter_label, metric)
                if not series:
                    continue
                max_bin = max((x for x, _ in series), default=1)
                pts = []
                for x, y in series:
                    px = pad_l + (x / max(1, max_bin)) * (panel_w - pad_l - pad_r)
                    py = pad_t + (1 - min(y / ymax, 1)) * (panel_h - pad_t - pad_b)
                    pts.append(f"{px:.1f},{py:.1f}")
                color = colors.get(method, "#666")
                paths.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.8"></polyline>')
                legend.append(f'<span><i style="background:{color}"></i>{html.escape(method)}</span>')
            if not paths:
                continue
            title = f"{suite_label(suite)} / {iter_label}"
            panels.append(
                f'<div class="panel"><h3>{html.escape(title)}</h3>'
                f'<svg viewBox="0 0 {panel_w} {panel_h}" role="img" aria-label="{html.escape(title)}">'
                f'<line x1="{pad_l}" y1="{panel_h-pad_b}" x2="{panel_w-pad_r}" y2="{panel_h-pad_b}" class="grid"></line>'
                f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{panel_h-pad_b}" class="grid"></line>'
                f'{"".join(paths)}'
                f'<text x="{pad_l}" y="{panel_h-7}" class="axis">channel bin</text>'
                f'<text x="{pad_l}" y="14" class="axis">{html.escape(metric)}</text>'
                f'</svg><div class="legend">{"".join(legend)}</div></div>'
            )
    return '<div class="profile-grid">' + "".join(panels) + "</div>"


def table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in columns)
    body = []
    for row in rows:
        cells = []
        for key, _, kind in columns:
            value = row.get(key, "")
            if kind == "pct":
                text = f"{float(value) * 100:.1f}%" if value != "" and math.isfinite(float(value)) else ""
            elif kind == "ms":
                text = f"{float(value):.1f}" if value != "" and math.isfinite(float(value)) else ""
            elif kind == "float":
                text = f"{float(value):.3f}" if value != "" and math.isfinite(float(value)) else ""
            else:
                text = str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html_report(
    path: Path,
    acc_rows: list[dict],
    episode_step_summary: list[dict],
    task_step_summary: list[dict],
    mlp_summary: list[dict],
    mlp_rows: list[dict],
) -> None:
    acc_rows = sorted(acc_rows, key=lambda r: (METHOD_ORDER.get(r["method"], 99), SUITE_ORDER.get(r["suite"], 99)))
    mlp_summary = sorted(mlp_summary, key=lambda r: (METHOD_ORDER.get(r["method"], 99), SUITE_ORDER.get(r["suite"], 99)))
    acc_table = table(
        acc_rows,
        [
            ("method", "Method", "text"),
            ("suite_label", "Suite", "text"),
            ("successes", "Successes", "text"),
            ("episodes", "Episodes", "text"),
            ("success_rate", "Accuracy", "pct"),
            ("policy_model_get_action_ms_mean", "Model get_action mean ms", "ms"),
            ("policy_total_ms_mean", "Policy total mean ms", "ms"),
            ("step_total_ms_mean", "Step total mean ms", "ms"),
            ("env_step_ms_mean", "Env step mean ms", "ms"),
            ("successfully_replaced", "Replaced layers", "text"),
        ],
    )
    mlp_table = table(
        mlp_summary,
        [
            ("method", "Method", "text"),
            ("suite_label", "Suite", "text"),
            ("probe_rows", "Probe rows", "text"),
            ("layers_seen", "Layers", "text"),
            ("iters_seen", "Denoise iters", "text"),
            ("calls_mean", "Calls mean", "float"),
            ("abs_mean_avg", "Abs mean avg", "float"),
            ("rms_avg", "RMS avg", "float"),
            ("std_avg", "STD avg", "float"),
            ("min_seen", "Min seen", "float"),
            ("max_seen", "Max seen", "float"),
        ],
    )
    episode_step_table = table(
        episode_step_summary,
        [
            ("method", "Method", "text"),
            ("suite_label", "Suite", "text"),
            ("episodes", "Episodes", "text"),
            ("success_episodes", "Success episodes", "text"),
            ("failed_episodes", "Failed episodes", "text"),
            ("steps_mean_success", "Mean steps on success", "float"),
            ("steps_mean_fail", "Mean steps on failure", "float"),
            ("steps_mean_all", "Mean steps all", "float"),
            ("step_total_ms_mean_success", "Success step total ms", "ms"),
            ("policy_model_get_action_ms_mean_success", "Success model ms", "ms"),
        ],
    )
    task_step_table = table(
        task_step_summary,
        [
            ("method", "Method", "text"),
            ("suite_label", "Suite", "text"),
            ("task_id", "Task id", "text"),
            ("successes", "Successes", "text"),
            ("episodes", "Episodes", "text"),
            ("success_rate", "Accuracy", "pct"),
            ("steps_mean_success", "Mean success steps", "float"),
            ("steps_mean_all", "Mean all steps", "float"),
            ("task_description", "Task", "text"),
        ],
    )
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>vlatest LIBERO Experiment Summary</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #111827; }}
    h1 {{ margin-bottom: 4px; }}
    h2 {{ margin-top: 34px; }}
    p {{ color: #4b5563; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 22px; font-size: 14px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 7px 9px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ background: #f3f4f6; }}
    .chart {{ width: 100%; max-width: 980px; display: block; margin: 10px 0 28px; }}
    .axis {{ font-size: 12px; fill: #374151; }}
    .value {{ font-size: 12px; fill: #111827; }}
    .bar {{ fill: #4f6bed; }}
    .bar.latency {{ fill: #14a37f; }}
    .profile-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); gap: 18px; }}
    .panel h3 {{ margin: 0 0 4px; font-size: 15px; }}
    .panel svg {{ width: 100%; height: auto; border: 1px solid #e5e7eb; }}
    .grid {{ stroke: #d1d5db; stroke-width: 1; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 8px; font-size: 12px; margin-top: 4px; color: #374151; }}
    .legend i {{ display: inline-block; width: 10px; height: 10px; margin-right: 4px; vertical-align: -1px; }}
    code {{ background: #f3f4f6; padding: 1px 4px; }}
  </style>
</head>
<body>
  <h1>vlatest LIBERO Experiment Summary</h1>
  <p>
    Accuracy is read from each completed <code>latency_summary.json</code>.
    DiT MLP distributions are channel-bin statistics from
    <code>action_head.model.transformer_blocks.*.ff.net.0.proj</code>
    at denoising iterations first/mid/last.
  </p>

  <h2>Accuracy and Latency Table</h2>
  {acc_table}

  <h2>Accuracy Graph</h2>
  {chart_accuracy(acc_rows)}

  {chart_latency(acc_rows, "policy_model_get_action_ms_mean", "Model get_action Latency")}

  {chart_latency(acc_rows, "step_total_ms_mean", "Step Total Latency")}

  <h2>Episode Step Summary</h2>
  <p>
    Step counts are grouped from <code>latency_steps.csv</code> by
    <code>global_episode</code>. Success/failure is joined from the matching
    <code>eval_driver.log</code> sequence.
  </p>
  {episode_step_table}

  <h2>Task-level Step Summary</h2>
  <p>
    Each LIBERO short suite has 10 tasks and 5 episodes per task. This table
    shows per-task success count and the average number of action steps.
  </p>
  {task_step_table}

  <h2>DiT MLP Probe Summary</h2>
  {mlp_table}

  <h2>DiT MLP Channel-bin Profiles</h2>
  <p>
    Lines show the average <code>abs_mean</code> across DiT MLP layers for each channel bin.
    This is a channel profile, not a raw activation-value histogram. If a red
    RealQuant line hides the blue/green FakeQuant lines, the profiles are almost
    numerically overlapping on this shared y-axis; missing legends indicate
    missing or empty probe CSVs for that suite.
  </p>
  {chart_mlp_profiles(mlp_rows, "abs_mean")}
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=r"C:\Users\zadac\Desktop\vlatest_results_backup")
    parser.add_argument("--out-dir", default=r"C:\Users\zadac\Desktop\vlatest_results_backup\summary_report")
    args = parser.parse_args()

    root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_dirs = find_experiment_dirs(root)
    acc_rows = [read_accuracy_latency(exp_dir) for exp_dir in exp_dirs]
    acc_rows = [r for r in acc_rows if r["method"] in METHOD_ORDER and r["suite"] in SUITE_ORDER]

    # Keep the latest/most complete row for duplicate method-suite pairs.
    best: dict[tuple[str, str], dict] = {}
    for row in acc_rows:
        key = (row["method"], row["suite"])
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        if int(row.get("episodes", 0) or 0) > int(current.get("episodes", 0) or 0):
            best[key] = row
        elif str(row["exp_dir"]) > str(current["exp_dir"]):
            best[key] = row
    acc_rows = sorted(best.values(), key=lambda r: (METHOD_ORDER.get(r["method"], 99), SUITE_ORDER.get(r["suite"], 99)))

    mlp_rows: list[dict] = []
    episode_rows: list[dict] = []
    for row in acc_rows:
        exp_dir = Path(row["exp_dir"])
        mlp_rows.extend(read_mlp_probe(exp_dir))
        episode_rows.extend(read_episode_steps(exp_dir, row["method"], row["suite"]))
    mlp_summary = summarize_mlp(mlp_rows)
    episode_step_summary = summarize_episode_steps(episode_rows)
    task_step_summary = summarize_task_steps(episode_rows)

    write_csv(out_dir / "accuracy_latency_summary.csv", acc_rows)
    write_csv(out_dir / "episode_steps.csv", episode_rows)
    write_csv(out_dir / "episode_steps_summary.csv", episode_step_summary)
    write_csv(out_dir / "task_steps_summary.csv", task_step_summary)
    write_csv(out_dir / "mlp_probe_summary.csv", mlp_summary)
    write_html_report(
        out_dir / "vlatest_experiment_report.html",
        acc_rows,
        episode_step_summary,
        task_step_summary,
        mlp_summary,
        mlp_rows,
    )

    print(f"Wrote {out_dir / 'accuracy_latency_summary.csv'}")
    print(f"Wrote {out_dir / 'episode_steps.csv'}")
    print(f"Wrote {out_dir / 'episode_steps_summary.csv'}")
    print(f"Wrote {out_dir / 'task_steps_summary.csv'}")
    print(f"Wrote {out_dir / 'mlp_probe_summary.csv'}")
    print(f"Wrote {out_dir / 'vlatest_experiment_report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
