#!/usr/bin/env bash
# Dedicated runner for the slow MLP probe / diagnostic experiments.
#
# Results are intentionally separated from the normal accuracy/latency runs:
#   $HOME/private/vlatest/vlatest_mlp_results/<RUN_ID>
#
# Usage:
#   bash run_vlatest_mlp_probe.sh servers [RUN_ID]
#   bash run_vlatest_mlp_probe.sh evals [RUN_ID]
#   bash run_vlatest_mlp_probe.sh baseline-server [RUN_ID]
#   bash run_vlatest_mlp_probe.sh baseline-eval [RUN_ID]
#   bash run_vlatest_mlp_probe.sh compare-real-fake [RUN_ID]
#   bash run_vlatest_mlp_probe.sh summarize [RUN_ID]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ACTION="${1:-}"
ARG_RUN_ID="${2:-}"
RESULTS_PARENT="${VLA_MLP_RESULTS_PARENT:-$HOME/private/vlatest/vlatest_mlp_results}"
RUN_PREFIX="${VLA_MLP_RUN_PREFIX:-mlp_spatial}"
BASELINE_GPU="${VLA_MLP_BASELINE_GPU:-3}"
BASELINE_PORT="${VLA_MLP_BASELINE_PORT:-5573}"
COMPARE_PRIMARY_PORT="${VLA_MLP_COMPARE_PRIMARY_PORT:-5571}"
COMPARE_SECONDARY_PORT="${VLA_MLP_COMPARE_SECONDARY_PORT:-5572}"

mkdir -p "$RESULTS_PARENT"

usage() {
    cat <<EOF
Usage:
  bash run_vlatest_mlp_probe.sh servers [RUN_ID]
      Start converted spatial diagnostic servers:
        fake_w4a8: GPU0 port 5570
        real:      GPU1 port 5571
        fake_w4a16:GPU2 port 5572

  bash run_vlatest_mlp_probe.sh evals [RUN_ID]
      Run evals for those converted servers with --record-actions.

  bash run_vlatest_mlp_probe.sh baseline-server [RUN_ID]
      Start original QuantVLA baseline spatial server:
        quantvla baseline: GPU${BASELINE_GPU} port ${BASELINE_PORT}

  bash run_vlatest_mlp_probe.sh baseline-eval [RUN_ID]
      Run eval for the original QuantVLA baseline server with --record-actions.

  bash run_vlatest_mlp_probe.sh compare-real-fake [RUN_ID]
      Query real(port ${COMPARE_PRIMARY_PORT}) and fake_w4a16(port ${COMPARE_SECONDARY_PORT})
      on the same observation, execute only real action, and save action diffs.

  bash run_vlatest_mlp_probe.sh summarize [RUN_ID]
      Summarize pair-probe and action-diff CSVs.

Environment overrides:
  VLA_MLP_RESULTS_PARENT, VLA_MLP_RUN_PREFIX, VLA_MLP_BASELINE_GPU,
  VLA_MLP_BASELINE_PORT, VLA_MLP_COMPARE_PRIMARY_PORT, VLA_MLP_COMPARE_SECONDARY_PORT
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

case "$ACTION" in
    servers)
        RUN_ID="$(run_id_for_create)"
        BASE_DIR="$(base_dir_for "$RUN_ID")"
        mkdir -p "$BASE_DIR"
        echo "MLP probe results root: $BASE_DIR"
        RUN_ID="$RUN_ID" \
        VLA_RESULTS_DIR="$BASE_DIR" \
        ENABLE_DIT_PROBE=1 \
        ENABLE_DIT_PAIR=1 \
        bash "$SCRIPT_DIR/run_vlatest_servers_only.sh" diag_spatial
        ;;

    evals)
        RUN_ID="$(run_id_for_use)"
        BASE_DIR="$(base_dir_for "$RUN_ID")"
        echo "MLP probe results root: $BASE_DIR"
        RECORD_ACTION_TRACE=1 \
        VLA_RESULTS_DIR="$BASE_DIR" \
        bash "$SCRIPT_DIR/run_vlatest_evals_only.sh" diag_spatial "$RUN_ID"
        ;;

    baseline-server)
        RUN_ID="$(run_id_for_create)"
        BASE_DIR="$(base_dir_for "$RUN_ID")"
        JOB_DIR="$BASE_DIR/exp0_quantvla_baseline_libero_spatial"
        mkdir -p "$JOB_DIR"
        echo "MLP probe baseline results root: $BASE_DIR"
        echo "[exp0_quantvla_baseline_libero_spatial] starting server gpu=$BASELINE_GPU port=$BASELINE_PORT"
        (
            export CUDA_VISIBLE_DEVICES="$BASELINE_GPU"
            export PORT="$BASELINE_PORT"
            export GR00T_DUQUANT_ASSUME_YES=1
            export GR00T_TIMING=1
            export GR00T_DIT_MLP_PROBE=1
            export GR00T_DIT_MLP_PROBE_MODE="quantvla_baseline"
            export GR00T_DIT_MLP_PROBE_TASK="libero_spatial"
            export GR00T_DIT_MLP_PROBE_DIR="$JOB_DIR/dit_mlp_probe"
            export GR00T_DIT_MLP_PROBE_BINS="${GR00T_DIT_MLP_PROBE_BINS:-128}"
            export GR00T_DIT_MLP_PROBE_ITERS="${GR00T_DIT_MLP_PROBE_ITERS:-first,mid,last}"
            bash "$SCRIPT_DIR/run_quantvla.sh" libero_spatial "$BASELINE_PORT"
        ) >"$JOB_DIR/server.log" 2>&1
        ;;

    baseline-eval)
        RUN_ID="$(run_id_for_use)"
        BASE_DIR="$(base_dir_for "$RUN_ID")"
        JOB_DIR="$BASE_DIR/exp0_quantvla_baseline_libero_spatial"
        RESULT_DIR="$JOB_DIR/results"
        mkdir -p "$RESULT_DIR"
        echo "MLP probe baseline results root: $BASE_DIR"
        bash "$SCRIPT_DIR/run_libero_eval.sh" libero_spatial \
            --headless \
            --port "$BASELINE_PORT" \
            --result-tag exp0_quantvla_baseline_libero_spatial \
            --result-dir "$RESULT_DIR" \
            --record-actions | tee "$JOB_DIR/eval_driver.log"
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
