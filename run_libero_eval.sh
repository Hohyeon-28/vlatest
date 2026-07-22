#!/bin/bash
# Script to run Libero evaluation
# Usage: ./run_libero_eval.sh [task_suite_name] [extra args...]
# task_suite_name: libero_spatial (default), libero_goal, libero_object, libero_90, libero_10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK=${1:-libero_10}
shift || true
EXTRA_ARGS=("$@")

HEADLESS_FLAG="no"
HAS_PORT="no"
RESULT_TAG="${LIBERO_RESULT_TAG:-}"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
    arg="${EXTRA_ARGS[$i]}"
    if [[ "$arg" == "--headless" ]]; then
        HEADLESS_FLAG="yes"
    fi
    if [[ "$arg" == "--port" || "$arg" == --port=* ]]; then
        HAS_PORT="yes"
    fi
    if [[ "$arg" == "--result-tag" && $((i + 1)) -lt ${#EXTRA_ARGS[@]} ]]; then
        RESULT_TAG="${EXTRA_ARGS[$((i + 1))]}"
    elif [[ "$arg" == --result-tag=* ]]; then
        RESULT_TAG="${arg#--result-tag=}"
    fi
done
PORT_VALUE="${PORT:-5556}"
RESULT_SUFFIX=""
if [[ -n "$RESULT_TAG" ]]; then
    RESULT_SUFFIX="_${RESULT_TAG}"
fi

# Reuse the user's active venv. Only activate conda when it exists.
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "${LIBERO_CONDA_ENV:-libero_test}" || true
fi

# Add this GR00T checkout and optional LIBERO checkout to Python path.
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
if [[ -n "${LIBERO_ROOT:-}" ]]; then
    export PYTHONPATH="$LIBERO_ROOT:$PYTHONPATH"
elif [[ -d "$HOME/private/LIBERO" ]]; then
    export PYTHONPATH="$HOME/private/LIBERO:$PYTHONPATH"
fi

# LIBERO asks for a dataset path during import when ~/.libero/config.yaml is
# missing. Background eval jobs have no stdin, so create the same config up
# front and keep the eval fully non-interactive.
LIBERO_CHECKOUT="${LIBERO_ROOT:-$HOME/private/LIBERO}"
LIBERO_CONFIG="${LIBERO_CONFIG:-$HOME/.libero/config.yaml}"
LIBERO_DATASET_DIR="${LIBERO_DATASET_DIR:-$LIBERO_CHECKOUT/datasets}"
if [[ -d "$LIBERO_CHECKOUT/libero/libero" && ! -f "$LIBERO_CONFIG" ]]; then
    mkdir -p "$(dirname "$LIBERO_CONFIG")"
    cat >"$LIBERO_CONFIG" <<EOF
benchmark_root: $LIBERO_CHECKOUT/libero/libero
bddl_files: $LIBERO_CHECKOUT/libero/libero/bddl_files
init_states: $LIBERO_CHECKOUT/libero/libero/init_files
datasets: $LIBERO_DATASET_DIR
assets: $LIBERO_CHECKOUT/libero/libero/assets
EOF
fi

echo "=========================================="
echo "Running Libero evaluation for $TASK"
echo "Headless mode: $HEADLESS_FLAG"
if [[ -n "$RESULT_TAG" ]]; then
    echo "Result tag: $RESULT_TAG"
fi
if [[ "$HAS_PORT" == "yes" ]]; then
    echo "Port: provided by extra args"
else
    echo "Port: $PORT_VALUE"
fi
echo "=========================================="
echo ""
echo "Make sure the inference server is running in another terminal!"
echo "Run: bash run_quantvla_converted_server.sh real $TASK <converted_checkpoint> $PORT_VALUE"
echo ""
echo "Results will be saved to:"
echo "  - Log: /tmp/logs/libero_eval_${TASK}${RESULT_SUFFIX}.log"
echo "  - Latency JSONL: /tmp/logs/libero_eval_${TASK}${RESULT_SUFFIX}_latency_steps.jsonl"
echo "  - Latency CSV: /tmp/logs/libero_eval_${TASK}${RESULT_SUFFIX}_latency_steps.csv"
echo "  - Latency summary: /tmp/logs/libero_eval_${TASK}${RESULT_SUFFIX}_latency_summary.json"
echo "  - Videos: disabled by default; pass --save-videos to write MP4 rollouts"
echo "=========================================="
echo ""

cd "$SCRIPT_DIR/examples/Libero/eval"

CMD=(python -u run_libero_eval.py --task_suite_name "$TASK")
if [[ "$HAS_PORT" == "no" ]]; then
    CMD+=(--port "$PORT_VALUE")
fi
if [[ -n "${LIBERO_RESULT_TAG:-}" ]]; then
    CMD+=(--result-tag "$LIBERO_RESULT_TAG")
fi
CMD+=("${EXTRA_ARGS[@]}")
"${CMD[@]}"
