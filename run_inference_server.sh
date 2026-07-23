#!/bin/bash
# Script to run GR00T inference server for Libero evaluation
# Usage: ./run_inference_server.sh [task_suite_name] [port]
# task_suite_name: libero_spatial (default), libero_goal, libero_object, libero_90, libero_10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK=${1:-libero_10}
PORT=${2:-${PORT:-5556}}

# Reuse the user's active venv. Only activate conda when it exists.
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "${GR00T_CONDA_ENV:-groot_test}" || true
fi

# Set model path and data config based on task
case $TASK in
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

# Allow override of denoising steps via environment variable
DENOISING_STEPS=${GR00T_DENOISING_STEPS:-8}

echo "=========================================="
echo "Starting GR00T inference server for $TASK"
echo "Model: $MODEL_PATH"
echo "Data Config: $DATA_CONFIG"
echo "Port: $PORT"
echo "Denoising Steps: $DENOISING_STEPS"
echo "=========================================="

cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

python -u scripts/inference_service.py \
    --model_path "$MODEL_PATH" \
    --server \
    --data_config "$DATA_CONFIG" \
    --denoising-steps "$DENOISING_STEPS" \
    --port "$PORT" \
    --embodiment-tag new_embodiment
