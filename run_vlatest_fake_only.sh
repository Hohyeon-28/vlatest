#!/usr/bin/env bash
# Dedicated runner for naive fake-only vlatest experiments.
#
# Default mapping:
#   Naive Fake W4A8   spatial/goal/object/long -> GPUs 0/1/2/3, ports 5581/5582/5583/5584
#   Naive Fake W4A16  spatial/goal/object/long -> GPUs 4/5/6/7, ports 5591/5592/5593/5594
#
# Usage:
#   ENABLE_DIT_PROBE=0 bash run_vlatest_fake_only.sh servers
#   bash run_vlatest_fake_only.sh evals
#
# Optional:
#   RUN_ID=my_run bash run_vlatest_fake_only.sh servers
#   bash run_vlatest_fake_only.sh evals my_run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ACTION="${1:-}"
ARG_RUN_ID="${2:-}"
CURRENT_SERVER_PIDS=()

RESULTS_PARENT="${VLA_RESULTS_PARENT:-$HOME/private/vlatest/vlatest_results}"
RUN_PREFIX="${VLA_FAKE_ONLY_RUN_PREFIX:-fake_only_naive}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-900}"
SERVER_READY_PATTERN="${SERVER_READY_PATTERN:-Server is ready}"

mkdir -p "$RESULTS_PARENT"

usage() {
    cat <<EOF
Usage:
  ENABLE_DIT_PROBE=0 bash run_vlatest_fake_only.sh servers [RUN_ID]
      Start 8 naive fake servers:
        Naive Fake W4A8   libero_spatial/libero_goal/libero_object/libero_10 -> GPUs 0/1/2/3
        Naive Fake W4A16  libero_spatial/libero_goal/libero_object/libero_10 -> GPUs 4/5/6/7

  bash run_vlatest_fake_only.sh evals [RUN_ID]
      Run matching evals in parallel. If RUN_ID is omitted, the newest
      ${RESULTS_PARENT}/${RUN_PREFIX}_* directory is used.

  bash run_vlatest_fake_only.sh tail [RUN_ID]
      Tail all eval logs for a run.

  bash run_vlatest_fake_only.sh stop [RUN_ID]
      Stop server PIDs recorded under the run directory.

Environment overrides:
  VLA_RESULTS_PARENT, VLA_FAKE_ONLY_RUN_PREFIX, ENABLE_DIT_PROBE,
  READY_TIMEOUT_SEC, GR00T_DENOISING_STEPS
EOF
}

run_id_for_create() {
    echo "${ARG_RUN_ID:-${RUN_ID:-${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)}}"
}

latest_run_id() {
    local latest
    latest="$(
        find "$RESULTS_PARENT" -maxdepth 1 -type d -name "${RUN_PREFIX}_*" ! -name "*YYYYMMDD*" \
            -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {print $2}'
    )"
    if [[ -z "$latest" ]]; then
        echo "No fake-only run found under $RESULTS_PARENT" >&2
        echo "Start servers first: ENABLE_DIT_PROBE=0 bash run_vlatest_fake_only.sh servers" >&2
        return 1
    fi
    basename "$latest"
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

mode_label() {
    case "$1" in
        naive_fake_w4a8) echo "exp4_naive_fake_w4a8" ;;
        naive_fake_w4a16) echo "exp5_naive_fake_w4a16" ;;
        *) echo "$1" ;;
    esac
}

checkpoint_for_suite() {
    local suite="$1"
    echo "$SCRIPT_DIR/outputs/${suite}_quantvla_full_gptq_like"
}

specs() {
    printf '%s\n' \
        "naive_fake_w4a8:libero_spatial:0:5581" \
        "naive_fake_w4a8:libero_goal:1:5582" \
        "naive_fake_w4a8:libero_object:2:5583" \
        "naive_fake_w4a8:libero_10:3:5584" \
        "naive_fake_w4a16:libero_spatial:4:5591" \
        "naive_fake_w4a16:libero_goal:5:5592" \
        "naive_fake_w4a16:libero_object:6:5593" \
        "naive_fake_w4a16:libero_10:7:5594"
}

base_dir_for() {
    local run_id="$1"
    echo "$RESULTS_PARENT/$run_id"
}

job_tag() {
    local mode="$1"
    local suite="$2"
    echo "$(mode_label "$mode")_${suite}"
}

server_log_for() {
    local base_dir="$1"
    local mode="$2"
    local suite="$3"
    echo "$base_dir/$(job_tag "$mode" "$suite")/server.log"
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

stop_pid() {
    local pid="$1"
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    kill_process_tree "$pid" TERM
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
        kill_process_tree "$pid" KILL
    fi
}

cleanup_current_servers() {
    local pid
    for pid in "${CURRENT_SERVER_PIDS[@]:-}"; do
        stop_pid "$pid"
    done
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

start_servers() {
    local run_id base_dir pids_file
    local mode suite gpu port tag ckpt job_dir server_log report pid
    local probe_enabled="${ENABLE_DIT_PROBE:-0}"

    run_id="$(run_id_for_create)"
    base_dir="$(base_dir_for "$run_id")"
    pids_file="$base_dir/server_pids.txt"
    mkdir -p "$base_dir"
    : >"$pids_file"
    trap cleanup_current_servers EXIT INT TERM

    echo "Fake-only run: $run_id"
    echo "Results root: $base_dir"
    echo "DiT probe: $probe_enabled"
    echo

    while IFS=: read -r mode suite gpu port; do
        tag="$(job_tag "$mode" "$suite")"
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
            export QUANTVLA_FAKE_WEIGHT_SOURCE="naive_w4"
            export QUANTVLA_FAKE_ACT_BITS="${QUANTVLA_FAKE_ACT_BITS:-8}"
            export GR00T_TIMING="${GR00T_TIMING:-1}"
            if [[ "$probe_enabled" != "0" ]]; then
                export GR00T_DIT_MLP_PROBE=1
                export GR00T_DIT_MLP_PROBE_DIR="$job_dir/dit_mlp_probe"
                export GR00T_DIT_MLP_PROBE_BINS="${GR00T_DIT_MLP_PROBE_BINS:-128}"
                export GR00T_DIT_MLP_PROBE_ITERS="${GR00T_DIT_MLP_PROBE_ITERS:-first,mid,last}"
            fi
            bash "$SCRIPT_DIR/run_quantvla_converted_server.sh" "$mode" "$suite" "$ckpt" "$port"
        ) >"$server_log" 2>&1 &
        pid="$!"
        CURRENT_SERVER_PIDS+=("$pid")
        echo "$pid $tag $gpu $port" >>"$pids_file"
        wait_for_server "$pid" "$server_log" "$tag"
    done < <(specs)

    echo
    echo "All fake-only servers are ready."
    echo "RUN_ID=$run_id"
    echo "Next:"
    echo "  bash run_vlatest_fake_only.sh evals $run_id"

    wait
}

run_evals() {
    local run_id base_dir failed pid
    local mode suite gpu port tag job_dir result_dir eval_log
    local pids=()
    local logs=()

    run_id="$(run_id_for_use)"
    base_dir="$(base_dir_for "$run_id")"
    mkdir -p "$base_dir"

    echo "Fake-only eval run: $run_id"
    echo "Results root: $base_dir"
    echo

    while IFS=: read -r mode suite gpu port; do
        assert_server_ready "$base_dir" "$mode" "$suite" "$port"
    done < <(specs)

    while IFS=: read -r mode suite gpu port; do
        tag="$(job_tag "$mode" "$suite")"
        job_dir="$base_dir/$tag"
        result_dir="$job_dir/results"
        eval_log="$job_dir/eval_driver.log"
        mkdir -p "$job_dir" "$result_dir"
        logs+=("$eval_log")
        (
            set -euo pipefail
            echo "[$tag] eval gpu=$gpu port=$port"
            export CUDA_VISIBLE_DEVICES="$gpu"
            bash "$SCRIPT_DIR/run_libero_eval.sh" \
                "$suite" \
                --headless \
                --port "$port" \
                --result-tag "$tag" \
                --result-dir "$result_dir"
            echo "[$tag] eval done"
        ) >"$eval_log" 2>&1 &
        pids+=("$!")
    done < <(specs)

    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done

    if [[ "$failed" != "0" ]]; then
        echo "One or more fake-only evals failed. Check $base_dir/*/eval_driver.log" >&2
        for eval_log in "${logs[@]}"; do
            if [[ -f "$eval_log" ]]; then
                echo "========== $eval_log =========="
                tail -n 80 "$eval_log" || true
            fi
        done
        exit 1
    fi

    echo "All fake-only evals finished."
    echo "Results root: $base_dir"
}

tail_logs() {
    local run_id base_dir files=()
    local mode suite gpu port tag eval_log
    run_id="$(run_id_for_use)"
    base_dir="$(base_dir_for "$run_id")"
    while IFS=: read -r mode suite gpu port; do
        tag="$(job_tag "$mode" "$suite")"
        eval_log="$base_dir/$tag/eval_driver.log"
        if [[ -f "$eval_log" ]]; then
            files+=("$eval_log")
        fi
    done < <(specs)
    if [[ "${#files[@]}" == "0" ]]; then
        echo "No eval logs found under $base_dir" >&2
        exit 1
    fi
    tail -f "${files[@]}"
}

stop_servers() {
    local run_id base_dir pids_file pid tag gpu port
    run_id="$(run_id_for_use)"
    base_dir="$(base_dir_for "$run_id")"
    pids_file="$base_dir/server_pids.txt"
    if [[ ! -f "$pids_file" ]]; then
        echo "No server PID file found: $pids_file" >&2
        exit 1
    fi
    while read -r pid tag gpu port; do
        if [[ -n "${pid:-}" ]]; then
            echo "Stopping $tag pid=$pid gpu=$gpu port=$port"
            stop_pid "$pid"
        fi
    done <"$pids_file"
}

case "$ACTION" in
    servers)
        start_servers
        ;;
    evals)
        run_evals
        ;;
    tail)
        tail_logs
        ;;
    stop)
        stop_servers
        ;;
    ""|-h|--help|help)
        usage
        ;;
    *)
        echo "Unknown action: $ACTION" >&2
        usage >&2
        exit 2
        ;;
esac
