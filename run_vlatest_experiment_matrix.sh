#!/usr/bin/env bash
# Run vlatest LIBERO experiment batches with isolated ports, GPUs, tags, logs,
# and result directories.
#
# Usage:
#   bash run_vlatest_experiment_matrix.sh short
#   bash run_vlatest_experiment_matrix.sh long
#   bash run_vlatest_experiment_matrix.sh dit_short
#   bash run_vlatest_experiment_matrix.sh dit_long
#   bash run_vlatest_experiment_matrix.sh dit_one libero_goal 0 5556
#
# Assumptions:
#   - Converted checkpoints already exist at ./outputs/<suite>_quantvla_full_gptq_like
#   - FakeQuant defaults to raw W4 dense reference.
#   - RealQuant uses vLLM GPTQ-Marlin.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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
        dit_pair) echo "dit_pair_real_vs_fake" ;;
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
    local label="$3"
    local deadline=$((SECONDS + READY_TIMEOUT_SEC))
    while [[ $SECONDS -lt $deadline ]]; do
        if grep -q "$SERVER_READY_PATTERN" "$log" 2>/dev/null; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[$label] server exited before ready. Last log lines:" >&2
            tail -n 80 "$log" >&2 || true
            return 1
        fi
        sleep 5
    done
    echo "[$label] timed out waiting for server readiness. Last log lines:" >&2
    tail -n 80 "$log" >&2 || true
    return 1
}

run_one_job() (
    set -euo pipefail

    local mode="$1"
    local suite="$2"
    local gpu="$3"
    local port="$4"
    local probe_kind="${5:-none}"
    local label
    local tag
    local ckpt
    local job_dir
    local result_dir
    local server_log
    local eval_log
    local report
    local server_pid=""

    if [[ "$probe_kind" == "1" ]]; then
        probe_kind="paired"
    elif [[ "$probe_kind" == "0" ]]; then
        probe_kind="none"
    fi

    if [[ "$probe_kind" == "paired" ]]; then
        label="$(mode_label dit_pair)"
    else
        label="$(mode_label "$mode")"
    fi
    tag="${label}_${suite}"
    ckpt="$(checkpoint_for_suite "$suite")"
    job_dir="$BASE_RESULT_DIR/$tag"
    result_dir="$job_dir/results"
    server_log="$job_dir/server.log"
    eval_log="$job_dir/eval_driver.log"
    report="$job_dir/replacement_report.json"

    mkdir -p "$job_dir" "$result_dir"

    if [[ ! -d "$ckpt" ]]; then
        echo "[$tag] converted checkpoint not found: $ckpt" >&2
        echo "[$tag] run first: CUDA_VISIBLE_DEVICES=$gpu bash run_quantvla_convert_full.sh $suite" >&2
        exit 1
    fi

    cleanup() {
        if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
            kill "$server_pid" 2>/dev/null || true
            wait "$server_pid" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT INT TERM

    echo "[$tag] starting server on gpu=$gpu port=$port"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export QUANTVLA_CONVERTED_REPORT="$report"
        export QUANTVLA_FAKE_WEIGHT_SOURCE="${QUANTVLA_FAKE_WEIGHT_SOURCE:-raw_w4}"
        export QUANTVLA_FAKE_ACT_BITS="${QUANTVLA_FAKE_ACT_BITS:-8}"
        export GR00T_TIMING="${GR00T_TIMING:-1}"
        if [[ "$probe_kind" == "active" || "$probe_kind" == "paired" ]]; then
            export GR00T_DIT_MLP_PROBE=1
            export GR00T_DIT_MLP_PROBE_DIR="$job_dir/dit_mlp_probe"
            export GR00T_DIT_MLP_PROBE_BINS="${GR00T_DIT_MLP_PROBE_BINS:-128}"
            export GR00T_DIT_MLP_PROBE_ITERS="${GR00T_DIT_MLP_PROBE_ITERS:-first,mid,last}"
        fi
        if [[ "$probe_kind" == "paired" ]]; then
            export GR00T_DIT_MLP_PROBE_PAIR=1
        fi
        bash "$SCRIPT_DIR/run_quantvla_converted_server.sh" "$mode" "$suite" "$ckpt" "$port"
    ) >"$server_log" 2>&1 &
    server_pid="$!"

    wait_for_server "$server_pid" "$server_log" "$tag"

    echo "[$tag] running eval"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        bash "$SCRIPT_DIR/run_libero_eval.sh" \
            "$suite" \
            --headless \
            --port "$port" \
            --result-tag "$tag" \
            --result-dir "$result_dir"
    ) >"$eval_log" 2>&1

    echo "[$tag] done"
    echo "[$tag] job dir: $job_dir"
)

run_parallel_jobs() {
    local pids=()
    local pid
    for spec in "$@"; do
        # spec format: mode:suite:gpu:port:probe_kind
        # probe_kind: none, active, paired
        IFS=: read -r mode suite gpu port probe_kind <<<"$spec"
        run_one_job "$mode" "$suite" "$gpu" "$port" "$probe_kind" &
        pids+=("$!")
    done

    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    return "$failed"
}

run_short() {
    local probe_kind
    probe_kind="$(default_probe_kind)"
    echo "[short] results: $BASE_RESULT_DIR"
    echo "[short] active DiT MLP probe: $probe_kind"
    echo "[short] wave A: exp1 fake_w4a8 on GPUs 0,1,2 and exp2 real on GPUs 3,4,5"
    run_parallel_jobs \
        "fake_w4a8:libero_spatial:0:5556:$probe_kind" \
        "fake_w4a8:libero_goal:1:5557:$probe_kind" \
        "fake_w4a8:libero_object:2:5558:$probe_kind" \
        "real:libero_spatial:3:5560:$probe_kind" \
        "real:libero_goal:4:5561:$probe_kind" \
        "real:libero_object:5:5562:$probe_kind"

    echo "[short] wave B: exp3 fake_w4a16 on GPUs 0,1,2"
    run_parallel_jobs \
        "fake:libero_spatial:0:5556:$probe_kind" \
        "fake:libero_goal:1:5557:$probe_kind" \
        "fake:libero_object:2:5558:$probe_kind"
}

run_long() {
    local probe_kind
    probe_kind="$(default_probe_kind)"
    echo "[long] results: $BASE_RESULT_DIR"
    echo "[long] active DiT MLP probe: $probe_kind"
    echo "[long] exp1/2/3 all together on GPUs 0,1,2"
    run_parallel_jobs \
        "fake_w4a8:libero_10:0:5556:$probe_kind" \
        "real:libero_10:1:5557:$probe_kind" \
        "fake:libero_10:2:5558:$probe_kind"
}

run_dit_short() {
    echo "[dit_short] paired DiT probe, separated from latency experiments"
    run_parallel_jobs \
        "real:libero_spatial:0:5556:paired" \
        "real:libero_goal:1:5557:paired" \
        "real:libero_object:2:5558:paired"
}

run_dit_long() {
    echo "[dit_long] paired DiT probe for long suite"
    run_parallel_jobs "real:libero_10:0:5556:paired"
}

run_dit_one() {
    local suite="${1:-libero_goal}"
    local gpu="${2:-0}"
    local port="${3:-5556}"
    echo "[dit_one] suite=$suite gpu=$gpu port=$port"
    run_parallel_jobs "real:${suite}:${gpu}:${port}:paired"
}

case "${1:-}" in
    short)
        run_short
        ;;
    long)
        run_long
        ;;
    dit_short)
        run_dit_short
        ;;
    dit_long)
        run_dit_long
        ;;
    dit_one)
        shift || true
        run_dit_one "$@"
        ;;
    *)
        cat <<EOF
Usage:
  bash run_vlatest_experiment_matrix.sh short
  bash run_vlatest_experiment_matrix.sh long
  bash run_vlatest_experiment_matrix.sh dit_short
  bash run_vlatest_experiment_matrix.sh dit_long
  bash run_vlatest_experiment_matrix.sh dit_one <suite> <gpu> <port>

Notes:
  short:
    wave A: fake_w4a8 spatial/goal/object on GPU 0/1/2, real on GPU 3/4/5
    wave B: fake_w4a16 spatial/goal/object on GPU 0/1/2
  long:
    fake_w4a8, real, fake_w4a16 for libero_10 on GPU 0/1/2 together
  DiT active probe:
    enabled by default for short/long and saved under each job's dit_mlp_probe.
    disable for pure latency with ENABLE_DIT_PROBE=0.
  dit_*:
    separate paired DiT probe runs; do not mix these with latency runs.

Results:
  $BASE_RESULT_DIR
EOF
        exit 2
        ;;
esac

echo "All requested jobs finished."
echo "Results root: $BASE_RESULT_DIR"
