"""Plot overlaid DiT MLP probe channel distributions.

The probe CSVs saved by vlatest contain channel-bin statistics for the
DiT MLP up-projection output, not raw activation samples. This script plots
those binned channel profiles as overlaid curves so Fake/Real or W4A8/W4A16
can be compared visually.

Examples:
  python plot_dit_mlp_probe_distributions.py ^
    "C:\Users\zadac\Desktop\vlatest_results_backup\short_fake_w4a8_20260722_132210\**\dit_mlp_probe\*.csv" ^
    --out-dir C:\Users\zadac\Desktop\dit_mlp_plots

  python plot_dit_mlp_probe_distributions.py fake.csv real.csv --labels fake,real --per-layer
"""

from __future__ import annotations

import argparse
import csv
import glob
import html
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


METRIC_COLUMNS = {
    "mean": ("mean",),
    "abs_mean": ("abs_mean", "fake_abs_mean", "real_abs_mean"),
    "std": ("std",),
    "rms": ("rms", "fake_rms", "real_rms"),
    "min": ("min",),
    "max": ("max",),
    "diff_abs_mean": ("diff_abs_mean",),
    "diff_rms": ("diff_rms",),
    "mse": ("mse",),
}


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            paths.extend(Path(p) for p in matches)
        else:
            paths.append(Path(pattern))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists() and path.suffix.lower() == ".csv":
            unique.append(path)
    return unique


def guess_label(path: Path, fallback: str) -> str:
    text = str(path).lower()
    if "fake_w4a8" in text:
        return "Fake W4A8"
    if "fake_w4a16" in text:
        return "Fake W4A16"
    if "real" in text or "marlin" in text:
        return "Real W4A16 Marlin"
    if "fake" in text:
        return "Fake"
    return fallback


def read_probe_csv(path: Path, label: str, metric: str) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows

        fields = set(reader.fieldnames)
        paired = {"fake_abs_mean", "real_abs_mean"}.issubset(fields)

        for row in reader:
            if paired:
                task = row.get("task") or path.stem
                common = {
                    "source": str(path),
                    "task": task,
                    "layer_index": int(row["layer_index"]),
                    "denoise_iter": int(row["denoise_iter"]),
                    "iter_label": row.get("iter_label", f"iter_{row['denoise_iter']}"),
                    "channel_bin": int(row["channel_bin"]),
                    "calls": int(float(row.get("calls", 0) or 0)),
                }
                metric_name = metric
                if metric == "abs_mean":
                    rows.append({**common, "label": "Fake paired", "value": float(row["fake_abs_mean"])})
                    rows.append({**common, "label": "Real paired", "value": float(row["real_abs_mean"])})
                elif metric == "rms":
                    rows.append({**common, "label": "Fake paired", "value": float(row["fake_rms"])})
                    rows.append({**common, "label": "Real paired", "value": float(row["real_rms"])})
                elif metric_name in row and row[metric_name] != "":
                    rows.append({**common, "label": label, "value": float(row[metric_name])})
                continue

            candidate_cols = METRIC_COLUMNS.get(metric, (metric,))
            value_col = next((col for col in candidate_cols if col in fields), None)
            if value_col is None:
                continue
            value_raw = row.get(value_col, "")
            if value_raw == "":
                continue
            rows.append(
                {
                    "source": str(path),
                    "label": row.get("mode") or label,
                    "task": row.get("task") or path.stem,
                    "layer_index": int(row["layer_index"]),
                    "denoise_iter": int(row["denoise_iter"]),
                    "iter_label": row.get("iter_label", f"iter_{row['denoise_iter']}"),
                    "channel_bin": int(row["channel_bin"]),
                    "calls": int(float(row.get("calls", 0) or 0)),
                    "value": float(value_raw),
                }
            )
    return rows


def average_rows(rows: list[dict], by_layer: bool) -> list[dict]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    calls: dict[tuple, int] = defaultdict(int)
    for row in rows:
        layer_key = row["layer_index"] if by_layer else "all_layers"
        key = (
            row["label"],
            row["task"],
            layer_key,
            row["denoise_iter"],
            row["iter_label"],
            row["channel_bin"],
        )
        if math.isfinite(row["value"]):
            grouped[key].append(row["value"])
            calls[key] += row["calls"]

    out: list[dict] = []
    for key, values in grouped.items():
        label, task, layer_index, denoise_iter, iter_label, channel_bin = key
        out.append(
            {
                "label": label,
                "task": task,
                "layer_index": layer_index,
                "denoise_iter": denoise_iter,
                "iter_label": iter_label,
                "channel_bin": channel_bin,
                "calls": calls[key],
                "value": sum(values) / len(values),
            }
        )
    return out


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "plot"


def write_plot(group_rows: list[dict], output: Path, title: str, metric: str) -> None:
    import matplotlib.pyplot as plt

    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in group_rows:
        by_label[row["label"]].append(row)

    plt.figure(figsize=(9, 4.8), dpi=150)
    colors = {
        "fake": "#4968ff",
        "Fake": "#4968ff",
        "Fake W4A8": "#4968ff",
        "Fake W4A16": "#17a673",
        "real": "#e05656",
        "Real W4A16 Marlin": "#e05656",
        "Fake paired": "#4968ff",
        "Real paired": "#e05656",
    }

    plotted = False
    for label, rows in sorted(by_label.items()):
        rows = sorted(rows, key=lambda r: r["channel_bin"])
        x = [r["channel_bin"] for r in rows]
        y = [r["value"] for r in rows]
        if not x:
            continue
        color = colors.get(label, None)
        plt.plot(x, y, linewidth=1.8, label=label, color=color)
        plt.fill_between(x, y, alpha=0.20, color=color)
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.title(title)
    plt.xlabel("Channel bin")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    plt.close()


def write_index(plot_paths: list[Path], out_dir: Path, metric: str) -> None:
    items = []
    for path in plot_paths:
        rel = path.name
        items.append(
            f"<section><h2>{html.escape(path.stem)}</h2>"
            f'<img src="{html.escape(rel)}" style="max-width:100%;border:1px solid #ddd"></section>'
        )
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>DiT MLP Probe Distribution Plots</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
    h1 {{ margin-bottom: 4px; }}
    p {{ color: #4b5563; }}
    section {{ margin: 28px 0; }}
  </style>
</head>
<body>
  <h1>DiT MLP Probe Distribution Plots</h1>
  <p>
    Metric: <code>{html.escape(metric)}</code>. These plots show channel-bin
    activation profiles from DiT MLP up-projection probes. They are not raw
    activation-value histograms.
  </p>
  {''.join(items)}
</body>
</html>
"""
    (out_dir / "index.html").write_text(doc, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", help="Probe CSV path(s) or glob(s).")
    parser.add_argument("--labels", help="Comma-separated labels matching input order.")
    parser.add_argument("--metric", default="abs_mean", choices=sorted(METRIC_COLUMNS))
    parser.add_argument("--out-dir", default="dit_mlp_distribution_plots")
    parser.add_argument("--per-layer", action="store_true", help="Also save one plot per DiT block.")
    args = parser.parse_args()

    paths = expand_inputs(args.csv)
    if not paths:
        raise SystemExit("No CSV files found.")

    labels = [part.strip() for part in args.labels.split(",")] if args.labels else []
    rows: list[dict] = []
    for idx, path in enumerate(paths):
        label = labels[idx] if idx < len(labels) and labels[idx] else guess_label(path, f"input_{idx}")
        rows.extend(read_probe_csv(path, label, args.metric))

    if not rows:
        raise SystemExit(f"No usable rows found for metric={args.metric}.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []

    for by_layer in ([False, True] if args.per_layer else [False]):
        averaged = average_rows(rows, by_layer=by_layer)
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for row in averaged:
            key = (row["task"], row["layer_index"], row["denoise_iter"], row["iter_label"])
            groups[key].append(row)

        for (task, layer_index, denoise_iter, iter_label), group in sorted(groups.items()):
            layer_text = "all_layers" if layer_index == "all_layers" else f"L{int(layer_index):02d}"
            title = f"{task} / {layer_text} / {iter_label}({denoise_iter}) / {args.metric}"
            name = safe_name(f"{task}_{layer_text}_{iter_label}_{args.metric}.png")
            output = out_dir / name
            write_plot(group, output, title, args.metric)
            if output.exists():
                plot_paths.append(output)

    write_index(plot_paths, out_dir, args.metric)
    print(f"Wrote {len(plot_paths)} plot(s) to {out_dir}")
    print(f"Open {out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
