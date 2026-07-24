#!/bin/bash
# Run GR00T LIBERO inference server with converted QuantVLA GPTQ-like weights.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_quantvla_converted_server.sh <real|fake|fake_w4a8|naive_fake_w4a16|naive_fake_w4a8> <task> [converted_checkpoint] [port]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-real}"
TASK="${2:-libero_10}"
CONVERTED_CHECKPOINT="${3:-}"
PORT="${4:-${PORT:-5556}}"

if [[ -n "$CONVERTED_CHECKPOINT" && -z "${4:-}" && "$CONVERTED_CHECKPOINT" =~ ^[0-9]+$ ]]; then
    PORT="$CONVERTED_CHECKPOINT"
    CONVERTED_CHECKPOINT=""
fi

case "$MODE" in
    real|real_quant|marlin|gptq_marlin)
        MODE="real"
        ;;
    fake|fake_quant|fake_w4a16|w4a16|torch|dequant|dequant_torch)
        MODE="fake"
        ;;
    fake_w4a8|w4a8|torch_w4a8)
        MODE="fake_w4a8"
        ;;
    naive|naive_fake|naive_fake_w4a16|naive_w4a16|dense_w4a16|original_w4a16)
        MODE="fake"
        export QUANTVLA_FAKE_WEIGHT_SOURCE="naive_w4"
        ;;
    naive_fake_w4a8|naive_w4a8|dense_w4a8|original_w4a8)
        MODE="fake_w4a8"
        export QUANTVLA_FAKE_WEIGHT_SOURCE="naive_w4"
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        echo "Valid modes: real, fake, fake_w4a8, naive_fake_w4a16, naive_fake_w4a8" >&2
        exit 1
        ;;
esac

if [[ -z "$CONVERTED_CHECKPOINT" ]]; then
    CONVERTED_CHECKPOINT="$SCRIPT_DIR/outputs/${TASK}_quantvla_full_gptq_like"
fi

if [[ ! -d "$CONVERTED_CHECKPOINT" ]]; then
    echo "Converted checkpoint directory not found: $CONVERTED_CHECKPOINT" >&2
    echo "Run conversion first, for example: bash run_quantvla_convert_full.sh $TASK" >&2
    exit 1
fi
CONVERTED_CHECKPOINT="$(cd "$(dirname "$CONVERTED_CHECKPOINT")" && pwd)/$(basename "$CONVERTED_CHECKPOINT")"

# Reuse the user's active venv. Only activate conda when it exists.
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "${GR00T_CONDA_ENV:-groot_test}" || true
fi

case "$TASK" in
    libero_spatial)
        MODEL_PATH="youliangtan/gr00t-n1.5-libero-spatial-posttrain"
        DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
        ;;
    libero_goal)
        MODEL_PATH="youliangtan/gr00t-n1.5-libero-goal-posttrain"
        DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfigMeanStd"
        ;;
    libero_object)
        MODEL_PATH="youliangtan/gr00t-n1.5-libero-object-posttrain"
        DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
        ;;
    libero_90)
        MODEL_PATH="youliangtan/gr00t-n1.5-libero-90-posttrain"
        DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
        ;;
    libero_10)
        MODEL_PATH="youliangtan/gr00t-n1.5-libero-long-posttrain"
        DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
        ;;
    *)
        echo "Unknown task: $TASK" >&2
        echo "Available tasks: libero_spatial, libero_goal, libero_object, libero_90, libero_10" >&2
        exit 1
        ;;
esac

# Avoid accidentally enabling the old DuQuant or GPTQ replacement path.
for name in $(env | cut -d= -f1 | grep -E '^(GR00T_DUQUANT_|GR00T_GPTQ_)' || true); do
    unset "$name"
done

mkdir -p /tmp/logs
REPORT="${QUANTVLA_CONVERTED_REPORT:-/tmp/logs/quantvla_converted_${MODE}_${TASK}_replacement_report.json}"
DENOISING_STEPS="${GR00T_DENOISING_STEPS:-8}"

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
export QUANTVLA_CONVERTED_CHECKPOINT="$CONVERTED_CHECKPOINT"
export QUANTVLA_CONVERTED_MODE="$MODE"
export QUANTVLA_CONVERTED_TASK="$TASK"
export QUANTVLA_CONVERTED_REPORT="$REPORT"
export QUANTVLA_CONVERTED_STRICT="${QUANTVLA_CONVERTED_STRICT:-1}"
export QUANTVLA_CONVERTED_DTYPE="${QUANTVLA_CONVERTED_DTYPE:-bfloat16}"
export GR00T_DIT_MLP_PROBE_DENOISING_STEPS="$DENOISING_STEPS"
export GR00T_TIMING="${GR00T_TIMING:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

echo "=========================================="
echo "Starting QuantVLA-converted LIBERO server"
echo "Mode: $MODE"
echo "Task: $TASK"
echo "Model: $MODEL_PATH"
echo "Converted checkpoint: $CONVERTED_CHECKPOINT"
echo "Data config: $DATA_CONFIG"
echo "Port: $PORT"
echo "Report: $REPORT"
echo "GR00T timing: $GR00T_TIMING"
if [[ "$MODE" == fake* ]]; then
    echo "Fake weight source: ${QUANTVLA_FAKE_WEIGHT_SOURCE:-raw_w4}"
fi
if [[ "$MODE" == "fake_w4a8" ]]; then
    echo "Fake activation bits: ${QUANTVLA_FAKE_ACT_BITS:-8}"
fi
echo "=========================================="

python -u scripts/inference_service.py \
    --model_path "$MODEL_PATH" \
    --server \
    --data_config "$DATA_CONFIG" \
    --denoising-steps "$DENOISING_STEPS" \
    --port "$PORT" \
    --embodiment-tag new_embodiment
