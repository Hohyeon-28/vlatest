"""Inspect a QuantVLA/DuQuant pack directory for Marlin conversion feasibility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from quantvla_marlin_utils import has_integer_weight_keys, iter_pack_files, load_quantvla_pack


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", required=True, help="Directory containing QuantVLA/DuQuant .npz pack files")
    parser.add_argument("--max-layers", type=int, default=20)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    files = iter_pack_files(args.pack_dir)
    if not files:
        raise SystemExit(f"No .npz pack files found: {args.pack_dir}")

    layers = []
    has_int_count = 0
    has_scale_count = 0
    has_input_transform_count = 0
    has_output_restore_count = 0

    for file in files:
        pack = load_quantvla_pack(file)
        has_int = has_integer_weight_keys(pack.raw_keys)
        has_scale = pack.weight_scale.size > 0
        has_input_transform = pack.perm is not None or bool(pack.r_in_blocks)
        has_output_restore = bool(pack.r_out_blocks)
        has_int_count += int(has_int)
        has_scale_count += int(has_scale)
        has_input_transform_count += int(has_input_transform)
        has_output_restore_count += int(has_output_restore)
        layers.append(
            {
                "layer": pack.layer_name,
                "file": str(file),
                "raw_keys": list(pack.raw_keys),
                "has_integer_code": has_int,
                "has_weight_scale": has_scale,
                "has_input_transform": has_input_transform,
                "has_output_restore": has_output_restore,
                "weight_scale_shape": list(pack.weight_scale.shape),
                "in_features": pack.meta.get("in_features"),
                "out_features": pack.meta.get("out_features"),
                "block_size": pack.meta.get("block_size"),
                "block_out_size": pack.meta.get("block_out_size"),
            }
        )

    summary = {
        "pack_dir": str(Path(args.pack_dir).resolve()),
        "num_pack_files": len(files),
        "layers_with_integer_code": has_int_count,
        "layers_with_weight_scale": has_scale_count,
        "layers_with_input_transform": has_input_transform_count,
        "layers_with_output_restore": has_output_restore_count,
        "direct_gptq_marlin_compatible": False,
        "requires_base_weight_to_reconstruct_int4": has_int_count == 0,
        "conversion_possible": has_scale_count > 0,
        "conclusion": (
            "Pack metadata can be converted only by reconstructing QuantVLA transformed weights from the base model. "
            "The pack does not directly look like a GPTQ/Marlin checkpoint."
            if has_int_count == 0
            else "Pack appears to contain integer codes; direct repacking may be possible after key/layout validation."
        ),
        "layers": layers,
    }

    print("========== QuantVLA Pack Inspection ==========")
    print(f"pack_dir: {summary['pack_dir']}")
    print(f"pack files: {summary['num_pack_files']}")
    print(f"integer-code layers: {has_int_count}")
    print(f"scale layers: {has_scale_count}")
    print(f"input-transform layers: {has_input_transform_count}")
    print(f"output-restore layers: {has_output_restore_count}")
    print(f"direct GPTQ/Marlin compatible: {summary['direct_gptq_marlin_compatible']}")
    print(f"requires base weight: {summary['requires_base_weight_to_reconstruct_int4']}")
    print(f"conversion possible: {summary['conversion_possible']}")
    print("\nSample layers:")
    for item in layers[: args.max_layers]:
        print(
            f"- {item['layer']} keys={item['raw_keys']} "
            f"scale={item['weight_scale_shape']} transform={item['has_input_transform']} restore={item['has_output_restore']}"
        )

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()