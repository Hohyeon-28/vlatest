#!/bin/bash
# Run GR00T LIBERO inference server with converted QuantVLA GPTQ-like weights.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_quantvla_converted_server.sh <real|fake> <task> <converted_checkpoint> [port]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-real}"
TASK="${2:-libero_10}"
CONVERTED_CHECKPOINT="${3:-}"
PORT="${4:-${PORT:-5556}}"

case "$MODE" in
    real|real_quant|marlin|gptq_marlin)
        MODE="real"
        ;;
    fake|fake_quant|torch|dequant|dequant_torch)
        MODE="fake"
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        echo "Valid modes: real, fake" >&2
        exit 1
        ;;
esac

if [[ -z "$CONVERTED_CHECKPOINT" ]]; then
    echo "Usage: $0 <real|fake> <task> <converted_checkpoint> [port]" >&2
    exit 1
fi

if [[ ! -d "$CONVERTED_CHECKPOINT" ]]; then
    echo "Converted checkpoint directory not found: $CONVERTED_CHECKPOINT" >&2
    exit 1
fi
CONVERTED_CHECKPOINT="$(cd "$(dirname "$CONVERTED_CHECKPOINT")" && pwd)/$(basename "$CONVERTED_CHECKPOINT")"

# Reuse the user's active venv. Only activate conda when it exists.
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "${GR00T_CONDA_ENV:-groot_test}" || true
fi

GR00T_REPO="${GR00T_REPO:-$HOME/private/QuantVLA_marlin}"
if [[ ! -d "$GR00T_REPO" ]]; then
    GR00T_REPO="$HOME/private/QuantVLA"
fi
if [[ ! -d "$GR00T_REPO" ]]; then
    echo "Could not find GR00T repo. Set GR00T_REPO=/path/to/QuantVLA_marlin or QuantVLA." >&2
    exit 1
fi

# Avoid accidentally enabling the old DuQuant or GPTQ replacement path while this server patches explicitly.
for name in $(env | cut -d= -f1 | grep -E '^(GR00T_DUQUANT_|GR00T_GPTQ_)' || true); do
    unset "$name"
done

mkdir -p /tmp/logs
REPORT="${QUANTVLA_REPORT:-/tmp/logs/quantvla_converted_${MODE}_${TASK}_replacement_report.json}"
DENOISING_STEPS="${GR00T_DENOISING_STEPS:-8}"

export PYTHONPATH="$SCRIPT_DIR:$GR00T_REPO:${PYTHONPATH:-}"

echo "=========================================="
echo "Starting QuantVLA-converted LIBERO server"
echo "Mode: $MODE"
echo "Task: $TASK"
echo "Converted checkpoint: $CONVERTED_CHECKPOINT"
echo "GR00T repo: $GR00T_REPO"
echo "Port: $PORT"
echo "Report: $REPORT"
echo "=========================================="

python "$SCRIPT_DIR/run_quantvla_converted_server.py" "$TASK" \
    --converted-checkpoint "$CONVERTED_CHECKPOINT" \
    --mode "$MODE" \
    --port "$PORT" \
    --denoising-steps "$DENOISING_STEPS" \
    --replacement-report "$REPORT"
