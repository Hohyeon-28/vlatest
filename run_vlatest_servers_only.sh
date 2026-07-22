#!/usr/bin/env bash
# Start only the QuantVLA-converted servers for a vlatest batch.
#
# Usage:
#   RUN_ID=my_run bash run_vlatest_servers_only.sh short_fake_w4a8
#   RUN_ID=my_run bash run_vlatest_servers_only.sh short_real
#   RUN_ID=my_run bash run_vlatest_servers_only.sh short_fake_w4a16
#   RUN_ID=my_run bash run_vlatest_servers_only.sh long
#
# In another terminal, run:
#   bash run_vlatest_evals_only.sh <same_batch> <same_RUN_ID>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BATCH="${1:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BASE_RESULT_DIR="${VLA_RESULTS_DIR:-/tmp/logs/vlatest_runs/$RUN_ID}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-900}"
SERVER_READY_PATTERN="${SERVER_READY_PATTERN:-Server is ready}"

mkdir -p "$BASE_RESULT_DIR"

mode_label() {
    case "$1" in
        fake_w4a8) echo "exp1_fake_w4a8" ;;
        real) echo "exp2_real_w4a16_marlin" ;;
        fake) echo "exp3_fake_w4a16" ;;
        *) echo "$1" ;;
    esac
}

default_probe_kind() {
    if [[ "${ENABLE_DIT_PROBE:-1}" == "0" ]]; then
        echo "none"
    else
        echo "active"
    fi
}

checkpoint_for_suite() {
    local suite="$1"
    echo "$SCRIPT_DIR/outputs/${suite}_quantvla_full_gptq_like"
}

wait_for_server() {
    local pid="$1"
    local log="$2"
    local tag="$3"
    local deadline=$((SECONDS + READY_TIMEOUT_SEC))
    while [[ $SECONDS -lt $deadline ]]; do
        if grep -q "$SERVER_READY_PATTERN" "$log" 2>/dev/null; then
            echo "[$tag] ready"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[$tag] server exited before ready. Last log lines:" >&2
            tail -n 80 "$log" >&2 || true
            return 1
        fi
        sleep 5
    done
    echo "[$tag] timed out waiting for server readiness. Last log lines:" >&2
    tail -n 80 "$log" >&2 || true
    return 1
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
  RUN_ID=my_run bash run_vlatest_servers_only.sh short_fake_w4a8
  RUN_ID=my_run bash run_vlatest_servers_only.sh short_real
  RUN_ID=my_run bash run_vlatest_servers_only.sh short_fake_w4a16
  RUN_ID=my_run bash run_vlatest_servers_only.sh long
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

pids=()
logs=()
tags=()

cleanup() {
    local pid
    for pid in "${pids[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT INT TERM

probe_kind="$(default_probe_kind)"
validate_batch

echo "Batch: $BATCH"
echo "RUN_ID: $RUN_ID"
echo "Results root: $BASE_RESULT_DIR"
echo "DiT active probe: $probe_kind"
echo

while IFS=: read -r mode suite gpu port; do
    label="$(mode_label "$mode")"
    tag="${label}_${suite}"
    ckpt="$(checkpoint_for_suite "$suite")"
    job_dir="$BASE_RESULT_DIR/$tag"
    server_log="$job_dir/server.log"
    report="$job_dir/replacement_report.json"

    if [[ ! -d "$ckpt" ]]; then
        echo "[$tag] converted checkpoint not found: $ckpt" >&2
        echo "[$tag] run first: CUDA_VISIBLE_DEVICES=$gpu bash run_quantvla_convert_full.sh $suite" >&2
        exit 1
    fi

    mkdir -p "$job_dir"
    echo "[$tag] starting server gpu=$gpu port=$port"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export QUANTVLA_CONVERTED_REPORT="$report"
        export QUANTVLA_FAKE_WEIGHT_SOURCE="${QUANTVLA_FAKE_WEIGHT_SOURCE:-raw_w4}"
        export QUANTVLA_FAKE_ACT_BITS="${QUANTVLA_FAKE_ACT_BITS:-8}"
        export GR00T_TIMING="${GR00T_TIMING:-1}"
        if [[ "$probe_kind" == "active" ]]; then
            export GR00T_DIT_MLP_PROBE=1
            export GR00T_DIT_MLP_PROBE_DIR="$job_dir/dit_mlp_probe"
            export GR00T_DIT_MLP_PROBE_BINS="${GR00T_DIT_MLP_PROBE_BINS:-128}"
            export GR00T_DIT_MLP_PROBE_ITERS="${GR00T_DIT_MLP_PROBE_ITERS:-first,mid,last}"
        fi
        bash "$SCRIPT_DIR/run_quantvla_converted_server.sh" "$mode" "$suite" "$ckpt" "$port"
    ) >"$server_log" 2>&1 &

    pids+=("$!")
    logs+=("$server_log")
    tags+=("$tag")
done < <(specs_for_batch)

for i in "${!pids[@]}"; do
    wait_for_server "${pids[$i]}" "${logs[$i]}" "${tags[$i]}"
done

echo
echo "All servers are ready."
echo "Run evals in another terminal:"
echo "  bash run_vlatest_evals_only.sh $BATCH $RUN_ID"
echo
echo "Press Ctrl+C here after evals finish to stop the servers."

wait
