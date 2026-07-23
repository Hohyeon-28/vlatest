#!/usr/bin/env bash
# Dedicated runner for the slow MLP probe / diagnostic experiments.
#
# Results are intentionally separated from normal accuracy/latency runs:
#   $HOME/private/vlatest/vlatest_mlp_results/<RUN_ID>
#
# Default GPU/port map for spatial MLP diagnostics:
#   Fake W4A8              GPU 0  port 5571
#   Real W4A16 Marlin      GPU 1  port 5572
#   Fake W4A16             GPU 2  port 5573
#   QuantVLA baseline      GPU 6  port 5574

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ACTION="${1:-}"
ARG_RUN_ID="${2:-}"

RESULTS_PARENT="${VLA_MLP_RESULTS_PARENT:-$HOME/private/vlatest/vlatest_mlp_results}"
RUN_PREFIX="${VLA_MLP_RUN_PREFIX:-mlp_spatial}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-900}"
SERVER_READY_PATTERN="${SERVER_READY_PATTERN:-Server is ready}"

FAKE_W4A8_GPU="${VLA_MLP_FAKE_W4A8_GPU:-0}"
REAL_GPU="${VLA_MLP_REAL_GPU:-1}"
FAKE_W4A16_GPU="${VLA_MLP_FAKE_W4A16_GPU:-2}"
BASELINE_GPU="${VLA_MLP_BASELINE_GPU:-6}"

FAKE_W4A8_PORT="${VLA_MLP_FAKE_W4A8_PORT:-5571}"
REAL_PORT="${VLA_MLP_REAL_PORT:-5572}"
FAKE_W4A16_PORT="${VLA_MLP_FAKE_W4A16_PORT:-5573}"
BASELINE_PORT="${VLA_MLP_BASELINE_PORT:-5574}"

COMPARE_PRIMARY_PORT="${VLA_MLP_COMPARE_PRIMARY_PORT:-$REAL_PORT}"
COMPARE_SECONDARY_PORT="${VLA_MLP_COMPARE_SECONDARY_PORT:-$FAKE_W4A16_PORT}"

mkdir -p "$RESULTS_PARENT"

usage() {
    cat <<EOF
Usage:
  bash run_vlatest_mlp_probe.sh servers [RUN_ID]
      Start all 4 spatial MLP diagnostic servers:
        Fake W4A8             GPU ${FAKE_W4A8_GPU} port ${FAKE_W4A8_PORT}
        Real W4A16 Marlin     GPU ${REAL_GPU} port ${REAL_PORT}
        Fake W4A16            GPU ${FAKE_W4A16_GPU} port ${FAKE_W4A16_PORT}
        QuantVLA baseline     GPU ${BASELINE_GPU} port ${BASELINE_PORT}

  bash run_vlatest_mlp_probe.sh evals [RUN_ID]
      Run all 4 matching spatial evals with --record-actions.

  bash run_vlatest_mlp_probe.sh compare-real-fake [RUN_ID]
      Query real(port ${COMPARE_PRIMARY_PORT}) and fake_w4a16(port ${COMPARE_SECONDARY_PORT})
      on the same observation, execute only real action, and save action diffs.

  bash run_vlatest_mlp_probe.sh summarize [RUN_ID]
      Summarize pair-probe and action-diff CSVs.

Compatibility aliases:
  baseline-server, baseline-eval

Environment overrides:
  VLA_MLP_RESULTS_PARENT, VLA_MLP_RUN_PREFIX,
  VLA_MLP_FAKE_W4A8_GPU, VLA_MLP_REAL_GPU, VLA_MLP_FAKE_W4A16_GPU, VLA_MLP_BASELINE_GPU,
  VLA_MLP_FAKE_W4A8_PORT, VLA_MLP_REAL_PORT, VLA_MLP_FAKE_W4A16_PORT, VLA_MLP_BASELINE_PORT
EOF
}

latest_run_id() {
    local latest
    latest="$(
        find "$RESULTS_PARENT" -maxdepth 1 -type d -name "${RUN_PREFIX}_*" ! -name "*YYYYMMDD*" \
            -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {print $2}'
    )"
    if [[ -z "$latest" ]]; then
        echo "No existing MLP run found under $RESULTS_PARENT" >&2
        echo "Start one first: bash run_vlatest_mlp_probe.sh servers" >&2
        return 1
    fi
    basename "$latest"
}

run_id_for_create() {
    echo "${ARG_RUN_ID:-${RUN_ID:-${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)}}"
}

run_id_for_use() {
    if [[ -n "$ARG_RUN_ID" ]]; then
        echo "$ARG_RUN_ID"
    elif [[ -n "${RUN_ID:-}" ]]; then
        echo "$RUN_ID"
    else
        latest_run_id
    fi
}

base_dir_for() {
    local run_id="$1"
    echo "$RESULTS_PARENT/$run_id"
}

mode_label() {
    case "$1" in
        fake_w4a8) echo "exp1_fake_w4a8" ;;
        real) echo "exp2_real_w4a16_marlin" ;;
        fake) echo "exp3_fake_w4a16" ;;
        quantvla_baseline) echo "exp0_quantvla_baseline" ;;
        *) echo "$1" ;;
    esac
}

checkpoint_for_suite() {
    local suite="$1"
    echo "$SCRIPT_DIR/outputs/${suite}_quantvla_full_gptq_like"
}

converted_specs() {
    printf '%s\n' \
        "fake_w4a8:libero_spatial:${FAKE_W4A8_GPU}:${FAKE_W4A8_PORT}" \
        "real:libero_spatial:${REAL_GPU}:${REAL_PORT}" \
        "fake:libero_spatial:${FAKE_W4A16_GPU}:${FAKE_W4A16_PORT}"
}

all_eval_specs() {
    converted_specs
    printf '%s\n' "quantvla_baseline:libero_spatial:${BASELINE_GPU}:${BASELINE_PORT}"
}

server_log_for() {
    local base_dir="$1"
    local mode="$2"
    local suite="$3"
    local tag
    tag="$(mode_label "$mode")_${suite}"
    echo "$base_dir/$tag/server.log"
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

kill_process_tree() {
    local pid="${1:-}"
    local signal="${2:-TERM}"
    local child
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        kill_process_tree "$child" "$signal"
    done
    kill -"$signal" "$pid" 2>/dev/null || true
}

stop_server_tree() {
    local pid="${1:-}"
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    kill_process_tree "$pid" TERM
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
        kill_process_tree "$pid" KILL
    fi
    wait "$pid" 2>/dev/null || true
}

start_converted_server() {
    local base_dir="$1"
    local mode="$2"
    local suite="$3"
    local gpu="$4"
    local port="$5"
    local label tag ckpt job_dir server_log report pid

    label="$(mode_label "$mode")"
    tag="${label}_${suite}"
    ckpt="$(checkpoint_for_suite "$suite")"
    job_dir="$base_dir/$tag"
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
        export GR00T_DIT_MLP_PROBE=1
        export GR00T_DIT_MLP_PROBE_DIR="$job_dir/dit_mlp_probe"
        export GR00T_DIT_MLP_PROBE_BINS="${GR00T_DIT_MLP_PROBE_BINS:-128}"
        export GR00T_DIT_MLP_PROBE_ITERS="${GR00T_DIT_MLP_PROBE_ITERS:-first,mid,last}"
        if [[ "$mode" == "real" && "${ENABLE_DIT_PAIR:-1}" != "0" ]]; then
            export GR00T_DIT_MLP_PROBE_PAIR=1
        fi
        bash "$SCRIPT_DIR/run_quantvla_converted_server.sh" "$mode" "$suite" "$ckpt" "$port"
    ) >"$server_log" 2>&1 &
    pid="$!"
    pids+=("$pid")
    logs+=("$server_log")
    tags+=("$tag")
    wait_for_server "$pid" "$server_log" "$tag"
}

start_baseline_server() {
    local base_dir="$1"
    local gpu="$2"
    local port="$3"
    local tag job_dir server_log pid

    tag="exp0_quantvla_baseline_libero_spatial"
    job_dir="$base_dir/$tag"
    server_log="$job_dir/server.log"

    mkdir -p "$job_dir"
    echo "[$tag] starting server gpu=$gpu port=$port"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PORT="$port"
        export GR00T_DUQUANT_ASSUME_YES=1
        export GR00T_TIMING=1
        export GR00T_DIT_MLP_PROBE=1
        export GR00T_DIT_MLP_PROBE_MODE="quantvla_baseline"
        export GR00T_DIT_MLP_PROBE_TASK="libero_spatial"
        export GR00T_DIT_MLP_PROBE_DIR="$job_dir/dit_mlp_probe"
        export GR00T_DIT_MLP_PROBE_BINS="${GR00T_DIT_MLP_PROBE_BINS:-128}"
        export GR00T_DIT_MLP_PROBE_ITERS="${GR00T_DIT_MLP_PROBE_ITERS:-first,mid,last}"
        bash "$SCRIPT_DIR/run_quantvla.sh" libero_spatial "$port"
    ) >"$server_log" 2>&1 &
    pid="$!"
    pids+=("$pid")
    logs+=("$server_log")
    tags+=("$tag")
    wait_for_server "$pid" "$server_log" "$tag"
}

assert_server_ready() {
    local base_dir="$1"
    local mode="$2"
    local suite="$3"
    local port="$4"
    local log
    log="$(server_log_for "$base_dir" "$mode" "$suite")"
    if [[ ! -f "$log" ]]; then
        echo "Server log not found for $suite on port $port: $log" >&2
        return 1
    fi
    if ! grep -q "$SERVER_READY_PATTERN" "$log"; then
        echo "Server for $suite on port $port is not ready yet. Check: $log" >&2
        tail -n 40 "$log" >&2 || true
        return 1
    fi
}

run_eval_job() (
    set -euo pipefail

    local base_dir="$1"
    local mode="$2"
    local suite="$3"
    local gpu="$4"
    local port="$5"
    local label tag job_dir result_dir eval_log

    label="$(mode_label "$mode")"
    tag="${label}_${suite}"
    job_dir="$base_dir/$tag"
    result_dir="$job_dir/results"
    eval_log="$job_dir/eval_driver.log"

    mkdir -p "$job_dir" "$result_dir"
    echo "[$tag] eval gpu=$gpu port=$port"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        bash "$SCRIPT_DIR/run_libero_eval.sh" "$suite" \
            --headless \
            --port "$port" \
            --result-tag "$tag" \
            --result-dir "$result_dir" \
            --record-actions
    ) >"$eval_log" 2>&1
    echo "[$tag] eval done"
)

case "$ACTION" in
    servers)
        RUN_ID="$(run_id_for_create)"
        BASE_DIR="$(base_dir_for "$RUN_ID")"
        mkdir -p "$BASE_DIR"
        echo "MLP probe results root: $BASE_DIR"
        echo "RUN_ID: $RUN_ID"
        echo

        pids=()
        logs=()
        tags=()
        cleanup() {
            local pid
            for pid in "${pids[@]:-}"; do
                stop_server_tree "$pid"
            done
        }
        trap cleanup EXIT INT TERM

        while IFS=: read -r mode suite gpu port; do
            start_converted_server "$BASE_DIR" "$mode" "$suite" "$gpu" "$port"
        done < <(converted_specs)
        start_baseline_server "$BASE_DIR" "$BASELINE_GPU" "$BASELINE_PORT"

        echo
        echo "All MLP probe servers are ready."
        echo "Run evals in another terminal:"
        echo "  bash run_vlatest_mlp_probe.sh evals $RUN_ID"
        echo
        echo "Press Ctrl+C here after evals finish to stop all 4 servers."
        wait
        ;;

    evals)
        RUN_ID="$(run_id_for_use)"
        BASE_DIR="$(base_dir_for "$RUN_ID")"
        echo "MLP probe results root: $BASE_DIR"
        echo "RUN_ID: $RUN_ID"
        echo

        while IFS=: read -r mode suite gpu port; do
            assert_server_ready "$BASE_DIR" "$mode" "$suite" "$port"
        done < <(all_eval_specs)

        pids=()
        eval_logs=()
        while IFS=: read -r mode suite gpu port; do
            run_eval_job "$BASE_DIR" "$mode" "$suite" "$gpu" "$port" &
            pids+=("$!")
            eval_logs+=("$(server_log_for "$BASE_DIR" "$mode" "$suite" | sed 's/server\.log$/eval_driver.log/')")
        done < <(all_eval_specs)

        failed=0
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                failed=1
            fi
        done
        if [[ "$failed" != "0" ]]; then
            echo "One or more evals failed. Check $BASE_DIR/*/eval_driver.log" >&2
            for log in "${eval_logs[@]}"; do
                if [[ -f "$log" ]]; then
                    echo "========== $log =========="
                    tail -n 80 "$log" || true
                fi
            done
            exit 1
        fi
        echo "All MLP probe evals finished."
        echo "Results root: $BASE_DIR"
        ;;

    baseline-server)
        RUN_ID="$(run_id_for_create)"
        BASE_DIR="$(base_dir_for "$RUN_ID")"
        pids=()
        logs=()
        tags=()
        trap 'stop_server_tree "${pids[0]:-}"' EXIT INT TERM
        start_baseline_server "$BASE_DIR" "$BASELINE_GPU" "$BASELINE_PORT"
        echo "Baseline server is ready. Press Ctrl+C after eval finishes."
        wait
        ;;

    baseline-eval)
        RUN_ID="$(run_id_for_use)"
        BASE_DIR="$(base_dir_for "$RUN_ID")"
        run_eval_job "$BASE_DIR" quantvla_baseline libero_spatial "$BASELINE_GPU" "$BASELINE_PORT"
        ;;

    compare-real-fake)
        RUN_ID="$(run_id_for_use)"
        BASE_DIR="$(base_dir_for "$RUN_ID")"
        JOB_DIR="$BASE_DIR/exp4_real_vs_fake_w4a16_same_obs_libero_spatial"
        RESULT_DIR="$JOB_DIR/results"
        mkdir -p "$RESULT_DIR"
        echo "MLP probe same-observation action diff root: $BASE_DIR"
        bash "$SCRIPT_DIR/run_libero_eval.sh" libero_spatial \
            --headless \
            --port "$COMPARE_PRIMARY_PORT" \
            --compare-port "$COMPARE_SECONDARY_PORT" \
            --compare-label fake_w4a16_same_obs \
            --result-tag exp4_real_vs_fake_w4a16_same_obs_libero_spatial \
            --result-dir "$RESULT_DIR" \
            --record-actions | tee "$JOB_DIR/eval_driver.log"
        ;;

    summarize)
        RUN_ID="$(run_id_for_use)"
        BASE_DIR="$(base_dir_for "$RUN_ID")"
        python "$SCRIPT_DIR/summarize_vlatest_diagnostic_metrics.py" --results-dir "$BASE_DIR"
        ;;

    *)
        usage >&2
        exit 2
        ;;
esac
