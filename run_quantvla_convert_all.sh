#!/usr/bin/env bash
# Convert all four LIBERO QuantVLA/DuQuant packs into GPTQ-like checkpoints.
#
# Assumes QuantVLA packs exist under:
#   ${QUANTVLA_REPO:-$HOME/private/QuantVLA}/duquant_packed_full_llm_dit_mlp_w4a8_b64c32ls015_<suite>_0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASKS=(
    libero_spatial
    libero_object
    libero_goal
    libero_10
)

for task in "${TASKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Converting $task"
    echo "=========================================="
    bash "$SCRIPT_DIR/run_quantvla_convert_full.sh" "$task"
done

echo ""
echo "All conversions complete. Outputs:"
find "$SCRIPT_DIR/outputs" -maxdepth 1 -type d -name "libero_*_quantvla_full_gptq_like" | sort
