#!/usr/bin/env bash
# Archive the current /tmp/logs LIBERO outputs into results/<run>/<suite>/.
#
# Usage:
#   bash vlaconvert_tools/archive_current_libero_logs.sh <fake|real|fp16> [run_name]
#
# Notes:
# - Eval files may be untagged from older runs:
#     /tmp/logs/libero_eval_libero_spatial.log
# - New eval files can be tagged:
#     /tmp/logs/libero_eval_libero_spatial_fake.log
# - This script prefers tagged files when present and falls back to untagged.

set -euo pipefail

MODE="${1:-}"
RUN_NAME="${2:-}"
LOG_DIR="${LOG_DIR:-/tmp/logs}"
RESULTS_DIR="${RESULTS_DIR:-results}"
SUITES=(libero_spatial libero_object libero_goal libero_10)

if [[ -z "$MODE" ]]; then
    echo "Usage: $0 <fake|real|fp16> [run_name]" >&2
    exit 1
fi

case "$MODE" in
    fake|real|fp16)
        ;;
    *)
        echo "Unknown mode: $MODE (expected fake, real, or fp16)" >&2
        exit 1
        ;;
esac

if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="${MODE}_$(date +%Y%m%d_%H%M%S)"
fi

copy_prefer_tagged() {
    local suite="$1"
    local suffix="$2"
    local dst_dir="$3"
    local tagged="${LOG_DIR}/libero_eval_${suite}_${MODE}${suffix}"
    local plain="${LOG_DIR}/libero_eval_${suite}${suffix}"

    if [[ -f "$tagged" ]]; then
        cp "$tagged" "$dst_dir/"
        echo "copied $(basename "$tagged")"
    elif [[ -f "$plain" ]]; then
        cp "$plain" "$dst_dir/"
        echo "copied $(basename "$plain")"
    else
        echo "missing libero_eval_${suite}${suffix}"
    fi
}

mkdir -p "$RESULTS_DIR/$RUN_NAME"

for suite in "${SUITES[@]}"; do
    dst="$RESULTS_DIR/$RUN_NAME/$suite"
    mkdir -p "$dst"

    copy_prefer_tagged "$suite" ".log" "$dst"
    copy_prefer_tagged "$suite" "_latency_summary.json" "$dst"
    copy_prefer_tagged "$suite" "_latency_steps.csv" "$dst"
    copy_prefer_tagged "$suite" "_latency_steps.jsonl" "$dst"

    report="${LOG_DIR}/quantvla_converted_${MODE}_${suite}_replacement_report.json"
    if [[ -f "$report" ]]; then
        cp "$report" "$dst/"
        echo "copied $(basename "$report")"
    elif [[ "$MODE" != "fp16" ]]; then
        echo "missing $(basename "$report")"
    fi

    {
        echo "mode=$MODE"
        echo "suite=$suite"
        echo "source_log_dir=$LOG_DIR"
        echo "archived_at=$(date --iso-8601=seconds)"
    } > "$dst/archive_manifest.txt"
done

echo "Archived current $MODE logs to $RESULTS_DIR/$RUN_NAME"
