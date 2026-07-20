#!/bin/bash
# Experimental GR00T server path: replace language-model Linear layers with
# GPTQ-Marlin wrappers, while keeping the original GR00T policy and DiT head.
#
# Usage:
#   ./run_inference_server_gptq_marlin_llm.sh <task_suite> <gptq_checkpoint> [port]
#
# Example:
#   CUDA_VISIBLE_DEVICES=0 ./run_inference_server_gptq_marlin_llm.sh \
#       libero_10 ./marlin_outputs/libero_10_llm_gptq4_marlin 5556

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK="${1:-libero_10}"
GPTQ_CHECKPOINT="${2:-}"
PORT="${3:-${PORT:-5556}}"

if [[ -z "$GPTQ_CHECKPOINT" ]]; then
    echo "Usage: $0 <task_suite> <gptq_checkpoint> [port]"
    exit 1
fi

if [[ ! -d "$GPTQ_CHECKPOINT" ]]; then
    echo "GPTQ checkpoint directory not found: $GPTQ_CHECKPOINT"
    exit 1
fi
GPTQ_CHECKPOINT="$(cd "$(dirname "$GPTQ_CHECKPOINT")" && pwd)/$(basename "$GPTQ_CHECKPOINT")"

# Keep the user's active venv if conda is unavailable.
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "${GR00T_CONDA_ENV:-groot_test}"
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
        echo "Unknown task: $TASK"
        echo "Available tasks: libero_spatial, libero_goal, libero_object, libero_90, libero_10"
        exit 1
        ;;
esac

export GR00T_GPTQ_QUANT_MODE="${GR00T_GPTQ_QUANT_MODE:-real}"
export GR00T_GPTQ_CHECKPOINT="$GPTQ_CHECKPOINT"
export GR00T_GPTQ_MARLIN_CHECKPOINT="$GPTQ_CHECKPOINT"
export GR00T_GPTQ_INCLUDE="${GR00T_GPTQ_INCLUDE:-.*backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$}"
export GR00T_GPTQ_EXCLUDE="${GR00T_GPTQ_EXCLUDE:-(?:^|\.)(vision|radio|embed|lm_head|action_head)(?:\.|$)}"
export GR00T_GPTQ_STRICT="${GR00T_GPTQ_STRICT:-1}"
export GR00T_GPTQ_REPORT="${GR00T_GPTQ_REPORT:-/tmp/logs/gptq_${GR00T_GPTQ_QUANT_MODE}_${TASK}_replacement_report.json}"
export GR00T_GPTQ_MARLIN_INCLUDE="$GR00T_GPTQ_INCLUDE"
export GR00T_GPTQ_MARLIN_EXCLUDE="$GR00T_GPTQ_EXCLUDE"
export GR00T_GPTQ_MARLIN_STRICT="$GR00T_GPTQ_STRICT"
export GR00T_GPTQ_MARLIN_REPORT="$GR00T_GPTQ_REPORT"

# Avoid accidentally enabling DuQuant in this path.
for name in $(env | cut -d= -f1 | grep '^GR00T_DUQUANT_' || true); do
    unset "$name"
done

DENOISING_STEPS="${GR00T_DENOISING_STEPS:-8}"

cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

echo "=========================================="
echo "Starting GR00T GPTQ-Marlin LLM server"
echo "Task: $TASK"
echo "Model: $MODEL_PATH"
echo "GPTQ checkpoint: $GPTQ_CHECKPOINT"
echo "Data Config: $DATA_CONFIG"
echo "Port: $PORT"
echo "Denoising Steps: $DENOISING_STEPS"
echo "=========================================="

python scripts/inference_service.py \
    --model_path "$MODEL_PATH" \
    --server \
    --data_config "$DATA_CONFIG" \
    --denoising-steps "$DENOISING_STEPS" \
    --port "$PORT" \
    --embodiment-tag new_embodiment
