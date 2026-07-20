#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

# Prepare a GPTQ/Marlin-format LLM checkpoint from a GR00T LIBERO checkpoint.
# Usage: ./run_marlin_quant.sh [libero_suite] [output_root]
# Example: CUDA_VISIBLE_DEVICES=0 ./run_marlin_quant.sh libero_10

TASK_SUITE="${1:-libero_10}"
OUT_ROOT="${2:-${MARLIN_OUT_DIR:-./marlin_outputs}}"
BITS="${GPTQ_BITS:-4}"
GROUP_SIZE="${GPTQ_GROUP_SIZE:-128}"
BATCH_SIZE="${GPTQ_BATCH_SIZE:-1}"
CALIB_FILE="${GPTQ_CALIB_FILE:-}"

case "$TASK_SUITE" in
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
        echo "Unknown task suite: $TASK_SUITE" >&2
        echo "Valid options: libero_spatial, libero_goal, libero_object, libero_90, libero_10" >&2
        exit 1
        ;;
esac

FP16_LLM_DIR="$OUT_ROOT/${TASK_SUITE}_llm_fp16"
GPTQ_DIR="$OUT_ROOT/${TASK_SUITE}_llm_gptq${BITS}_marlin"

mkdir -p "$OUT_ROOT"

echo "=========================================="
echo "Preparing GPTQ/Marlin LLM for $TASK_SUITE"
echo "GR00T checkpoint: $MODEL_PATH"
echo "Data config:      $DATA_CONFIG"
echo "FP16 LLM out:     $FP16_LLM_DIR"
echo "GPTQ out:         $GPTQ_DIR"
echo "Bits/group:       W${BITS}, group_size=${GROUP_SIZE}"
echo "=========================================="

python marlin_tools/extract_gr00t_llm.py \
    --model-path "$MODEL_PATH" \
    --data-config "$DATA_CONFIG" \
    --output-dir "$FP16_LLM_DIR"

QUANT_ARGS=(
    --model "$FP16_LLM_DIR"
    --output "$GPTQ_DIR"
    --bits "$BITS"
    --group-size "$GROUP_SIZE"
    --batch-size "$BATCH_SIZE"
)

if [[ -n "$CALIB_FILE" ]]; then
    QUANT_ARGS+=(--calibration-file "$CALIB_FILE")
fi

python marlin_tools/quantize_llm_gptq_marlin.py "${QUANT_ARGS[@]}"
python marlin_tools/inspect_gptq_checkpoint.py "$GPTQ_DIR"

echo ""
echo "Done. GPTQ checkpoint is at: $GPTQ_DIR"
echo "Try vLLM smoke test with:"
echo "  python marlin_tools/smoke_test_vllm.py $GPTQ_DIR"


