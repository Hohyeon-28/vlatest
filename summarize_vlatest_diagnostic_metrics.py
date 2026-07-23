"""Summarize vlatest spatial diagnostic diff metrics.

This script is intentionally post-hoc: it reads files already produced by
LIBERO eval runs and does not touch model code.

Outputs:
  - diagnostic_pair_probe_summary.csv
  - diagnostic_action_diff_summary.csv
  - diagnostic_action_diff_steps.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path


ACTION_COLUMNS = [
    "action_x",
    "action_y",
    "action_z",
    "action_roll",
    "action_pitch",
    "action_yaw",
    "action_gripper",
]


def mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def infer_label(path: Path) -> str:
    text = str(path).lower()
    if "fake_w4a8" in text:
        return "Fake W4A8"
    if "fake_w4a16" in text:
        return "Fake W4A16"
    if "real" in text or "marlin" in text:
        return "Real W4A16 Marlin"
    if "quantvla" in text:
        return "QuantVLA baseline"
    return path.parent.parent.name


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def summarize_pair_probe(results_dir: Path) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for path in results_dir.glob("**/dit_mlp_probe/dit_mlp_probe_pair_*.csv"):
        if path.stat().st_size <= 100:
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    target = row.get("target", "up_proj")
                    key = (
                        row.get("task", ""),
                        target,
                        row.get("iter_label", ""),
                        row.get("layer_index", ""),
                    )
                    grouped[key].append(row)
                except Exception:
                    continue

    out: list[dict] = []
    for (suite, target, iter_label, layer_index), rows in sorted(grouped.items()):
        fake_abs = [float(r["fake_abs_mean"]) for r in rows]
        real_abs = [float(r["real_abs_mean"]) for r in rows]
        fake_abs_mean = mean(fake_abs)
        real_abs_mean = mean(real_abs)
        ratio = real_abs_mean / fake_abs_mean if abs(fake_abs_mean) > 1e-12 else float("nan")
        out.append(
            {
                "suite": suite,
                "target": target,
                "iter_label": iter_label,
                "layer_index": layer_index,
                "channel_bins": len(rows),
                "fake_abs_mean": fake_abs_mean,
                "real_abs_mean": real_abs_mean,
                "real_minus_fake_abs_mean": real_abs_mean - fake_abs_mean,
                "real_fake_abs_mean_ratio": ratio,
                "diff_abs_mean": mean([float(r["diff_abs_mean"]) for r in rows]),
                "diff_rms": mean([float(r["diff_rms"]) for r in rows]),
                "mse": mean([float(r["mse"]) for r in rows]),
                "per_channel_bin_max_abs_diff": max((float(r["max_abs_diff"]) for r in rows), default=float("nan")),
            }
        )
    return out


def read_action_trace(path: Path) -> dict[tuple[str, str, str], dict]:
    records: dict[tuple[str, str, str], dict] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (row["task_id"], row["episode_idx"], row["action_step_index"])
                action = [float(row[col]) for col in ACTION_COLUMNS]
                records[key] = {
                    "task_id": row["task_id"],
                    "episode_idx": row["episode_idx"],
                    "action_step_index": row["action_step_index"],
                    "action": action,
                }
            except Exception:
                continue
    return records


def compare_action_traces(left_path: Path, right_path: Path) -> tuple[list[dict], dict]:
    left = read_action_trace(left_path)
    right = read_action_trace(right_path)
    common_keys = sorted(set(left) & set(right), key=lambda k: (int(k[0]), int(k[1]), int(k[2])))
    step_rows: list[dict] = []
    for key in common_keys:
        left_action = left[key]["action"]
        right_action = right[key]["action"]
        diffs = [r - l for l, r in zip(left_action, right_action)]
        abs_diffs = [abs(v) for v in diffs]
        left_norm = math.sqrt(sum(v * v for v in left_action))
        right_norm = math.sqrt(sum(v * v for v in right_action))
        dot = sum(l * r for l, r in zip(left_action, right_action))
        cosine = dot / (left_norm * right_norm) if left_norm > 1e-12 and right_norm > 1e-12 else float("nan")
        row = {
            "left": infer_label(left_path),
            "right": infer_label(right_path),
            "task_id": key[0],
            "episode_idx": key[1],
            "action_step_index": key[2],
            "mean_abs_diff": mean(abs_diffs),
            "rms_diff": math.sqrt(mean([v * v for v in diffs])),
            "max_abs_diff": max(abs_diffs),
            "cosine_similarity": cosine,
        }
        for col, diff in zip(ACTION_COLUMNS, diffs):
            row[f"{col}_diff"] = diff
        step_rows.append(row)

    summary = {
        "left": infer_label(left_path),
        "right": infer_label(right_path),
        "left_path": str(left_path),
        "right_path": str(right_path),
        "matched_steps": len(step_rows),
        "mean_abs_diff": mean([r["mean_abs_diff"] for r in step_rows]),
        "rms_diff": mean([r["rms_diff"] for r in step_rows]),
        "max_abs_diff": max((r["max_abs_diff"] for r in step_rows), default=float("nan")),
        "cosine_similarity_mean": mean([r["cosine_similarity"] for r in step_rows]),
    }
    return step_rows, summary


def find_action_traces(results_dir: Path) -> list[Path]:
    paths = sorted(results_dir.glob("**/*_action_trace.csv"))
    return [path for path in paths if path.stat().st_size > 100]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="vlatest result directory to summarize.")
    parser.add_argument("--out-dir", help="Output directory. Defaults to <results-dir>/diagnostics.")
    parser.add_argument(
        "--action-pair",
        action="append",
        default=[],
        metavar="LEFT_SUBSTR,RIGHT_SUBSTR",
        help="Compare only action traces whose paths contain these substrings. Can be repeated.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "diagnostics"

    pair_rows = summarize_pair_probe(results_dir)
    write_csv(out_dir / "diagnostic_pair_probe_summary.csv", pair_rows)

    traces = find_action_traces(results_dir)
    comparisons: list[tuple[Path, Path]] = []
    if args.action_pair:
        for spec in args.action_pair:
            left_key, right_key = [part.strip() for part in spec.split(",", 1)]
            left = next((p for p in traces if left_key in str(p)), None)
            right = next((p for p in traces if right_key in str(p)), None)
            if left is None or right is None:
                raise SystemExit(f"Could not match action pair: {spec}")
            comparisons.append((left, right))
    else:
        comparisons = list(combinations(traces, 2))

    action_step_rows: list[dict] = []
    action_summary_rows: list[dict] = []
    for left, right in comparisons:
        step_rows, summary = compare_action_traces(left, right)
        action_step_rows.extend(step_rows)
        action_summary_rows.append(summary)

    write_csv(out_dir / "diagnostic_action_diff_steps.csv", action_step_rows)
    write_csv(out_dir / "diagnostic_action_diff_summary.csv", action_summary_rows)

    print(f"Wrote diagnostics to {out_dir}")
    print(f"Pair probe summaries: {len(pair_rows)}")
    print(f"Action trace comparisons: {len(action_summary_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
