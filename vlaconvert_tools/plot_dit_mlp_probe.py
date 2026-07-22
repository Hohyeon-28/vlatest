"""Build HTML heatmaps from DiT MLP up-projection activation probe CSV files."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path


def _read_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if "fake_abs_mean" in row and "real_abs_mean" in row:
                    common = {
                        "_source": str(path),
                        "_paired": True,
                        "task": row["task"],
                        "layer": row["layer"],
                        "layer_index": int(row["layer_index"]),
                        "denoise_iter": int(row["denoise_iter"]),
                        "iter_label": row["iter_label"],
                        "channel_bin": int(row["channel_bin"]),
                        "calls": int(row["calls"]),
                    }
                    fake = {
                        **common,
                        "mode": "fake",
                        "mean": float(row["fake_mean"]),
                        "abs_mean": float(row["fake_abs_mean"]),
                        "std": float("nan"),
                        "rms": float(row["fake_rms"]),
                        "min": float("nan"),
                        "max": float("nan"),
                    }
                    real = {
                        **common,
                        "mode": "real",
                        "mean": float(row["real_mean"]),
                        "abs_mean": float(row["real_abs_mean"]),
                        "std": float("nan"),
                        "rms": float(row["real_rms"]),
                        "min": float("nan"),
                        "max": float("nan"),
                    }
                    diff = {
                        **common,
                        "mode": "paired_abs_diff",
                        "mean": float(row["diff_mean"]),
                        "abs_mean": float(row["diff_abs_mean"]),
                        "std": float("nan"),
                        "rms": float(row["diff_rms"]),
                        "min": 0.0,
                        "max": float(row["max_abs_diff"]),
                    }
                    rows.extend([fake, real, diff])
                    continue
                row["_source"] = str(path)
                row["_paired"] = False
                for key in ["layer_index", "denoise_iter", "channel_bin", "calls"]:
                    row[key] = int(row[key])
                for key in ["mean", "abs_mean", "std", "rms", "min", "max"]:
                    row[key] = float(row[key])
                rows.append(row)
    return rows


def _value_color(value: float, vmin: float, vmax: float) -> str:
    if not math.isfinite(value):
        return "#f3f4f6"
    if vmax <= vmin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    # Blue -> white -> red.
    if t < 0.5:
        k = t / 0.5
        r = int(59 + (248 - 59) * k)
        g = int(130 + (250 - 130) * k)
        b = int(246 + (252 - 246) * k)
    else:
        k = (t - 0.5) / 0.5
        r = int(248 + (220 - 248) * k)
        g = int(250 + (38 - 250) * k)
        b = int(252 + (38 - 252) * k)
    return f"rgb({r},{g},{b})"


def _render_heatmap(title: str, rows: list[dict], metric: str) -> str:
    if not rows:
        return ""
    y_keys = sorted({(r["layer_index"], r["denoise_iter"], r["iter_label"]) for r in rows})
    x_keys = sorted({r["channel_bin"] for r in rows})
    data = {(r["layer_index"], r["denoise_iter"], r["iter_label"], r["channel_bin"]): r[metric] for r in rows}
    values = [r[metric] for r in rows if math.isfinite(r[metric])]
    vmin = min(values) if values else 0.0
    vmax = max(values) if values else 1.0
    cells: list[str] = []
    for layer_idx, iter_idx, iter_label in y_keys:
        label = f"L{layer_idx:02d} {iter_label}({iter_idx})"
        cells.append(f'<div class="y-label">{html.escape(label)}</div>')
        for x in x_keys:
            value = data.get((layer_idx, iter_idx, iter_label, x), float("nan"))
            color = _value_color(value, vmin, vmax)
            tip = f"{label}, bin {x}, {metric}={value:.6g}"
            cells.append(f'<div class="cell" title="{html.escape(tip)}" style="background:{color}"></div>')
    grid_cols = "96px " + " ".join(["8px"] * len(x_keys))
    return f"""
<section>
  <h2>{html.escape(title)}</h2>
  <p>metric={html.escape(metric)}, min={vmin:.6g}, max={vmax:.6g}</p>
  <div class="heatmap" style="grid-template-columns:{grid_cols}">
    {''.join(cells)}
  </div>
</section>
"""


def _compare_rows(rows: list[dict], metric: str) -> list[dict]:
    by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        mode = row["mode"]
        key = (row["task"], row["layer_index"], row["denoise_iter"], row["iter_label"], row["channel_bin"])
        by_key[key][mode] = row

    out: list[dict] = []
    for key, modes in by_key.items():
        if "fake" not in modes or "real" not in modes:
            continue
        if not (modes["fake"].get("_paired") and modes["real"].get("_paired")):
            continue
        fake_value = modes["fake"][metric]
        real_value = modes["real"][metric]
        denom = abs(fake_value) if abs(fake_value) > 1e-9 else 1e-9
        task, layer_idx, iter_idx, iter_label, bin_idx = key
        out.append(
            {
                "mode": "real_minus_fake_pct",
                "task": task,
                "layer_index": layer_idx,
                "denoise_iter": iter_idx,
                "iter_label": iter_label,
                "channel_bin": bin_idx,
                "calls": min(modes["fake"]["calls"], modes["real"]["calls"]),
                metric: (real_value - fake_value) / denom * 100.0,
            }
        )
    return out


def _write_html(rows: list[dict], output: Path, metric: str) -> None:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], row["task"])].append(row)

    sections = []
    for (mode, task), group in sorted(grouped.items()):
        sections.append(_render_heatmap(f"{mode} / {task}", group, metric))

    compare = _compare_rows(rows, metric)
    if compare:
        compare_grouped: dict[str, list[dict]] = defaultdict(list)
        for row in compare:
            compare_grouped[row["task"]].append(row)
        for task, group in sorted(compare_grouped.items()):
            sections.append(_render_heatmap(f"Real - Fake % / {task}", group, metric))

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>DiT MLP Up Projection Probe</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
    h1 {{ margin-bottom: 8px; }}
    h2 {{ margin-top: 32px; }}
    p {{ color: #4b5563; }}
    .heatmap {{ display: grid; gap: 1px; align-items: stretch; overflow-x: auto; max-width: 100%; }}
    .y-label {{ font-size: 11px; line-height: 8px; white-space: nowrap; padding-right: 6px; }}
    .cell {{ width: 8px; height: 8px; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>DiT MLP Up Projection Probe</h1>
  <p>
    Rows are DiT block and denoising iteration. Columns are channel bins.
    Use <code>abs_mean</code> or <code>rms</code> to inspect activation magnitude,
    and the Real-Fake percent section to inspect distribution drift.
  </p>
  {''.join(sections)}
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(doc, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", help="Probe CSV file(s). Pass fake and real CSVs for comparison.")
    parser.add_argument("--metric", default="abs_mean", choices=["mean", "abs_mean", "std", "rms", "min", "max"])
    parser.add_argument("--output", default="dit_mlp_up_probe_report.html")
    args = parser.parse_args()

    rows = _read_rows([Path(p) for p in args.csv])
    if not rows:
        raise SystemExit("No rows found.")
    _write_html(rows, Path(args.output), args.metric)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
