#!/usr/bin/env bash
# Convert QuantVLA/DuQuant packed weights into a GPTQ-like checkpoint for
# transform-aware FakeQuant or vLLM GPTQ-Marlin RealQuant execution.
#
# Default scope matches run_quantvla.sh:
#   - GR00T language-model attention/MLP Linear layers
#   - GR00T action_head DiT MLP Linear layers
#
# Usage:
#   bash run_quantvla_convert_full.sh <task> [pack_dir] [output_dir] [base_checkpoint]
#
# Example:
#   BASE_CKPT=$(ls -d ~/.cache/huggingface/hub/models--youliangtan--gr00t-n1.5-libero-long-posttrain/snapshots/* | head -n 1)
#   bash run_quantvla_convert_full.sh libero_10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

TASK="${1:-libero_10}"
PACK_DIR="${2:-}"
OUTPUT_DIR="${3:-./outputs/${TASK}_quantvla_full_gptq_like}"
BASE_CHECKPOINT="${4:-}"

case "$TASK" in
    libero_spatial)
        MODEL_CACHE_NAME="models--youliangtan--gr00t-n1.5-libero-spatial-posttrain"
        PACK_SUFFIX="spatial"
        ;;
    libero_goal)
        MODEL_CACHE_NAME="models--youliangtan--gr00t-n1.5-libero-goal-posttrain"
        PACK_SUFFIX="goal"
        ;;
    libero_object)
        MODEL_CACHE_NAME="models--youliangtan--gr00t-n1.5-libero-object-posttrain"
        PACK_SUFFIX="object"
        ;;
    libero_90)
        MODEL_CACHE_NAME="models--youliangtan--gr00t-n1.5-libero-90-posttrain"
        PACK_SUFFIX="90"
        ;;
    libero_10)
        MODEL_CACHE_NAME="models--youliangtan--gr00t-n1.5-libero-long-posttrain"
        PACK_SUFFIX="long"
        ;;
    *)
        echo "Unknown task: $TASK" >&2
        echo "Available tasks: libero_spatial, libero_goal, libero_object, libero_90, libero_10" >&2
        exit 1
        ;;
esac

if [[ -z "$PACK_DIR" ]]; then
    QUANTVLA_REPO="${QUANTVLA_REPO:-$HOME/private/QuantVLA}"
    PACK_DIR="$QUANTVLA_REPO/duquant_packed_full_llm_dit_mlp_w4a8_b64c32ls015_${PACK_SUFFIX}_0"
fi

if [[ ! -d "$PACK_DIR" ]]; then
    echo "QuantVLA pack directory not found: $PACK_DIR" >&2
    echo "Run QuantVLA first, or pass the pack directory as the 2nd argument." >&2
    exit 1
fi

if [[ -z "$BASE_CHECKPOINT" ]]; then
    CACHE_ROOT="${HF_HOME:-$HOME/.cache/huggingface}/hub/$MODEL_CACHE_NAME/snapshots"
    BASE_CHECKPOINT="$(find "$CACHE_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1 || true)"
fi

if [[ -z "$BASE_CHECKPOINT" || ! -e "$BASE_CHECKPOINT" ]]; then
    echo "Could not locate base checkpoint. Pass it as the 4th argument." >&2
    exit 1
fi

echo "=========================================="
echo "Converting QuantVLA full default scope"
echo "Task:            $TASK"
echo "Base checkpoint: $BASE_CHECKPOINT"
echo "Pack dir:        $PACK_DIR"
echo "Output:          $OUTPUT_DIR"
echo "Scope:           LLM + DiT MLP"
echo "Bits/group:      W4, group_size=${GPTQ_GROUP_SIZE:-128}"
echo "=========================================="

python vlaconvert_tools/convert_quantvla_to_gptq_like.py \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --pack-dir "$PACK_DIR" \
    --output "$OUTPUT_DIR" \
    --bits 4 \
    --group-size "${GPTQ_GROUP_SIZE:-128}" \
    --scale-source "${QUANTVLA_SCALE_SOURCE:-mse}" \
    --row-rot-mode "${QUANTVLA_ROW_ROT_MODE:-restore}"

echo ""
echo "Done. Converted checkpoint is at: $OUTPUT_DIR"
echo "Use it with:"
echo "  CUDA_VISIBLE_DEVICES=0 bash run_quantvla_converted_server.sh real $TASK $OUTPUT_DIR 5556"
