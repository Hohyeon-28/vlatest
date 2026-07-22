#!/usr/bin/env bash
# Run only the LIBERO evals that match a vlatest server-only batch.
#
# Usage:
#   bash run_vlatest_evals_only.sh short_fake_w4a8 <RUN_ID>
#   bash run_vlatest_evals_only.sh short_real <RUN_ID>
#   bash run_vlatest_evals_only.sh short_fake_w4a16 <RUN_ID>
#   bash run_vlatest_evals_only.sh long <RUN_ID>
#
# RUN_ID can be omitted. In that case this script uses the newest matching
# vlatest_results/<batch>_* directory created by run_vlatest_servers_only.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BATCH="${1:-}"
RUN_ID="${2:-}"
RESULTS_PARENT="${VLA_RESULTS_PARENT:-$HOME/private/vlatest/vlatest_results}"

mode_label() {
    case "$1" in
        fake_w4a8) echo "exp1_fake_w4a8" ;;
        real) echo "exp2_real_w4a16_marlin" ;;
        fake) echo "exp3_fake_w4a16" ;;
        *) echo "$1" ;;
    esac
}

default_run_prefix() {
    case "$BATCH" in
        short_fake_w4a8|short_real|short_fake_w4a16|long)
            echo "$BATCH"
            ;;
        *)
            echo "$BATCH"
            ;;
    esac
}

find_latest_run_dir() {
    local prefix
    local latest
    prefix="$(default_run_prefix)"
    latest="$(find "$RESULTS_PARENT" -maxdepth 1 -type d -name "${prefix}_*" ! -name "*YYYYMMDD*" -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {print $2}')"
    if [[ -z "$latest" ]]; then
        echo "No matching server result directory found under $RESULTS_PARENT for batch $BATCH" >&2
        echo "Start servers first, for example: ENABLE_DIT_PROBE=1 bash run_vlatest_servers_only.sh $BATCH" >&2
        return 1
    fi
    echo "$latest"
}

specs_for_batch() {
    case "$BATCH" in
        short_fake_w4a8)
            printf '%s\n' \
                "fake_w4a8:libero_spatial:0:5556" \
                "fake_w4a8:libero_goal:1:5557" \
                "fake_w4a8:libero_object:2:5558"
            ;;
        short_real)
            printf '%s\n' \
                "real:libero_spatial:3:5560" \
                "real:libero_goal:4:5561" \
                "real:libero_object:5:5562"
            ;;
        short_fake_w4a16)
            printf '%s\n' \
                "fake:libero_spatial:0:5556" \
                "fake:libero_goal:1:5557" \
                "fake:libero_object:2:5558"
            ;;
        long)
            printf '%s\n' \
                "fake_w4a8:libero_10:0:5556" \
                "real:libero_10:1:5557" \
                "fake:libero_10:2:5558"
            ;;
        *)
            cat >&2 <<EOF
Usage:
  bash run_vlatest_evals_only.sh short_fake_w4a8 <RUN_ID>
  bash run_vlatest_evals_only.sh short_real <RUN_ID>
  bash run_vlatest_evals_only.sh short_fake_w4a16 <RUN_ID>
  bash run_vlatest_evals_only.sh long <RUN_ID>
EOF
            return 2
            ;;
    esac
}

validate_batch() {
    case "$BATCH" in
        short_fake_w4a8|short_real|short_fake_w4a16|long)
            ;;
        *)
            specs_for_batch >/dev/null
            ;;
    esac
}

if [[ -z "$BATCH" ]]; then
    specs_for_batch >/dev/null
    exit 2
fi

validate_batch

if [[ -z "$RUN_ID" ]]; then
    BASE_RESULT_DIR="$(find_latest_run_dir)"
    RUN_ID="$(basename "$BASE_RESULT_DIR")"
elif [[ -n "${VLA_RESULTS_DIR:-}" ]]; then
    BASE_RESULT_DIR="$VLA_RESULTS_DIR"
else
    BASE_RESULT_DIR="$RESULTS_PARENT/$RUN_ID"
fi
mkdir -p "$BASE_RESULT_DIR"

server_log_for() {
    local mode="$1"
    local suite="$2"
    local label
    local tag
    label="$(mode_label "$mode")"
    tag="${label}_${suite}"
    echo "$BASE_RESULT_DIR/$tag/server.log"
}

assert_server_ready() {
    local mode="$1"
    local suite="$2"
    local port="$3"
    local log
    log="$(server_log_for "$mode" "$suite")"
    if [[ ! -f "$log" ]]; then
        echo "Server log not found for $suite on port $port: $log" >&2
        echo "Start servers first: ENABLE_DIT_PROBE=1 bash run_vlatest_servers_only.sh $BATCH" >&2
        return 1
    fi
    if ! grep -q "Server is ready" "$log"; then
        echo "Server for $suite on port $port is not ready yet. Check: $log" >&2
        tail -n 40 "$log" >&2 || true
        return 1
    fi
}

run_eval_job() (
    set -euo pipefail

    local mode="$1"
    local suite="$2"
    local gpu="$3"
    local port="$4"
    local label
    local tag
    local job_dir
    local result_dir
    local eval_log

    label="$(mode_label "$mode")"
    tag="${label}_${suite}"
    job_dir="$BASE_RESULT_DIR/$tag"
    result_dir="$job_dir/results"
    eval_log="$job_dir/eval_driver.log"

    mkdir -p "$job_dir" "$result_dir"

    echo "[$tag] eval gpu=$gpu port=$port"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        bash "$SCRIPT_DIR/run_libero_eval.sh" \
            "$suite" \
            --headless \
            --port "$port" \
            --result-tag "$tag" \
            --result-dir "$result_dir"
    ) >"$eval_log" 2>&1
    echo "[$tag] eval done"
)

pids=()

echo "Batch: $BATCH"
echo "RUN_ID: $RUN_ID"
echo "Results root: $BASE_RESULT_DIR"
echo

while IFS=: read -r mode suite gpu port; do
    assert_server_ready "$mode" "$suite" "$port"
done < <(specs_for_batch)

while IFS=: read -r mode suite gpu port; do
    run_eval_job "$mode" "$suite" "$gpu" "$port" &
    pids+=("$!")
done < <(specs_for_batch)

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

if [[ "$failed" != "0" ]]; then
    echo "One or more evals failed. Check $BASE_RESULT_DIR/*/eval_driver.log" >&2
    exit 1
fi

echo "All evals finished."
echo "Results root: $BASE_RESULT_DIR"
