"""Convert QuantVLA/DuQuant pack artifacts into a GPTQ-like INT4 checkpoint.

This is an experiment bridge, not a claim that QuantVLA packs are already
native Marlin checkpoints.  Current QuantVLA/DuQuant packs store transform
metadata and scales, but not the final integer qweight tensor.  Therefore this
script reconstructs the transformed weight from the original FP/BF weight,
requantizes it to signed INT4, and writes GPTQ-like qweight/scales/qzeros.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from quantvla_marlin_utils import (
    apply_quantvla_weight_transform,
    candidate_weight_keys,
    choose_scales,
    dequantize_gptq_like,
    has_integer_weight_keys,
    iter_pack_files,
    load_quantvla_pack,
    load_weight_tensors,
    make_symmetric_qzeros,
    output_prefix_for_layer,
    pack_int4_qweight,
    quantize_signed_int,
    repeat_scales_for_groups,
    sanitize_name,
)


DEFAULT_LAYER_REGEX = (
    r".*("
    r"backbone\.eagle_model\.language_model\..*\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    r"|"
    r"action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)"
    r")$"
)

DEFAULT_EXCLUDE_REGEX = r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)"


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _make_quantize_config(bits: int, group_size: int) -> dict[str, Any]:
    return {
        "bits": bits,
        "group_size": group_size,
        "sym": True,
        "desc_act": False,
        "quant_method": "gptq",
        "checkpoint_format": "gptq",
        "lm_head": False,
        "source": "quantvla_duquant_repacked_experiment",
        "note": (
            "qweight/scales/qzeros were reconstructed from QuantVLA/DuQuant "
            "transform metadata plus the original dense checkpoint weights."
        ),
    }


def _find_weight_key(
    layer_name: str,
    tensors: dict[str, torch.Tensor],
    strip_prefixes: list[str],
) -> str | None:
    for key in candidate_weight_keys(layer_name, strip_prefixes):
        if key in tensors:
            return key
    return None


def _store_transform_arrays(
    transform_arrays: dict[str, np.ndarray],
    prefix: str,
    pack,
) -> dict[str, Any]:
    safe = sanitize_name(prefix)
    keys: dict[str, Any] = {}
    if pack.perm is not None:
        key = f"{safe}.perm"
        transform_arrays[key] = pack.perm.astype(np.int64, copy=False)
        keys["perm"] = key
    if pack.r_in_blocks:
        rin_keys = {}
        for block, value in sorted(pack.r_in_blocks.items()):
            key = f"{safe}.Rin_{block}"
            transform_arrays[key] = value.astype(np.float32, copy=False)
            rin_keys[str(block)] = key
        keys["r_in_blocks"] = rin_keys
    if pack.r_out_blocks:
        rout_keys = {}
        for block, value in sorted(pack.r_out_blocks.items()):
            key = f"{safe}.Rout_{block}"
            transform_arrays[key] = value.astype(np.float32, copy=False)
            rout_keys[str(block)] = key
        keys["r_out_blocks"] = rout_keys
    return keys


def _layer_report_header() -> str:
    return (
        "Target Linear layers: {target}\n"
        "Successfully converted: {converted}\n"
        "Unmatched checkpoint keys: {unmatched}\n"
        "Skipped by regex: {skipped}\n"
        "Fallback FP16 layers: 0"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True, help="Dense checkpoint dir/file containing original weights")
    parser.add_argument("--pack-dir", required=True, help="QuantVLA/DuQuant .npz pack directory")
    parser.add_argument("--output", required=True, help="Output directory for GPTQ-like tensors and metadata")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--layer-regex", default=DEFAULT_LAYER_REGEX)
    parser.add_argument(
        "--exclude-regex",
        default=DEFAULT_EXCLUDE_REGEX,
        help="Layers matching this regex are skipped after --layer-regex. Empty string disables it.",
    )
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=["backbone.eagle_model.language_model.", "language_model."],
        help="Prefix to strip from QuantVLA layer names when writing GPTQ-like keys. Can be repeated.",
    )
    parser.add_argument("--scale-source", choices=["mse", "pack", "maxabs"], default="mse")
    parser.add_argument("--row-rot-mode", choices=["restore", "propagate", "0"], default="restore")
    parser.add_argument("--save-bias", choices=["none", "original"], default="none")
    parser.add_argument("--max-layers", type=int, default=0, help="Debug limit. 0 means all matching layers.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-report", default=None)
    args = parser.parse_args()

    if args.bits != 4:
        raise SystemExit("Only W4 packing is implemented in this experiment.")
    if args.group_size <= 0:
        raise SystemExit("--group-size must be positive")

    pack_files = iter_pack_files(args.pack_dir)
    if not pack_files:
        raise SystemExit(f"No .npz pack files found: {args.pack_dir}")

    print("Loading dense checkpoint tensors...")
    dense_tensors = load_weight_tensors(args.base_checkpoint)
    if not dense_tensors:
        raise SystemExit(f"No weight tensors found in {args.base_checkpoint}")

    layer_re = re.compile(args.layer_regex)
    exclude_re = re.compile(args.exclude_regex) if args.exclude_regex else None
    out_dir = Path(args.output)
    output_tensors: dict[str, torch.Tensor] = {}
    transform_arrays: dict[str, np.ndarray] = {}
    layers: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    skipped: list[str] = []
    converted = 0
    target = 0

    for pack_file in pack_files:
        pack = load_quantvla_pack(pack_file)
        layer_name = pack.layer_name
        if not layer_re.match(layer_name):
            skipped.append(layer_name)
            continue
        if exclude_re is not None and exclude_re.search(layer_name):
            skipped.append(layer_name)
            continue
        target += 1
        if args.max_layers and converted >= args.max_layers:
            skipped.append(layer_name)
            continue

        weight_key = _find_weight_key(layer_name, dense_tensors, args.strip_prefix)
        if weight_key is None:
            unmatched.append(
                {
                    "layer": layer_name,
                    "tried": candidate_weight_keys(layer_name, args.strip_prefix),
                }
            )
            continue

        weight = dense_tensors[weight_key]
        if weight.ndim != 2:
            unmatched.append({"layer": layer_name, "weight_key": weight_key, "reason": f"weight ndim is {weight.ndim}"})
            continue

        apply_row_rot = args.row_rot_mode != "0"
        transformed = apply_quantvla_weight_transform(weight, pack, apply_row_rot=apply_row_rot)
        out_features, in_features = transformed.shape
        scales_out = choose_scales(transformed, pack, args.bits, args.scale_source).to(torch.float32)
        if scales_out.numel() != out_features:
            unmatched.append(
                {
                    "layer": layer_name,
                    "weight_key": weight_key,
                    "reason": f"scale shape {list(scales_out.shape)} does not match out_features={out_features}",
                }
            )
            continue

        q_signed = quantize_signed_int(transformed, scales_out, args.bits)
        qweight = pack_int4_qweight(q_signed)
        groups = math.ceil(in_features / args.group_size)
        scales = repeat_scales_for_groups(scales_out, in_features, args.group_size).to(torch.float16)
        qzeros = make_symmetric_qzeros(groups, out_features)
        deq = dequantize_gptq_like(qweight, scales.to(torch.float32), in_features, args.group_size)
        quant_ref = q_signed.to(torch.float32) * scales_out[:, None]
        pack_mse = torch.mean((deq - quant_ref) ** 2).item()
        original_mse = torch.mean((deq - transformed.to(torch.float32)) ** 2).item()
        mean_abs = torch.mean(torch.abs(deq - transformed.to(torch.float32))).item()
        max_abs = torch.max(torch.abs(deq - transformed.to(torch.float32))).item()

        out_prefix = output_prefix_for_layer(layer_name, args.strip_prefix)
        output_tensors[f"{out_prefix}.qweight"] = qweight.cpu()
        output_tensors[f"{out_prefix}.scales"] = scales.cpu()
        output_tensors[f"{out_prefix}.qzeros"] = qzeros.cpu()
        if args.save_bias == "original":
            bias_key = weight_key[: -len(".weight")] + ".bias"
            if bias_key in dense_tensors:
                output_tensors[f"{out_prefix}.bias"] = dense_tensors[bias_key].to(torch.float16).cpu()

        transform_keys = _store_transform_arrays(transform_arrays, out_prefix, pack)
        layer_info = {
            "source_layer": layer_name,
            "source_pack_file": str(pack_file),
            "source_weight_key": weight_key,
            "output_prefix": out_prefix,
            "in_features": in_features,
            "out_features": out_features,
            "bits": args.bits,
            "group_size": args.group_size,
            "groups": groups,
            "scale_source": args.scale_source,
            "row_rot_mode": args.row_rot_mode,
            "pack_meta": pack.meta,
            "pack_has_integer_codes": has_integer_weight_keys(pack.raw_keys),
            "requires_base_weight_reconstruction": not has_integer_weight_keys(pack.raw_keys),
            "has_input_transform": pack.perm is not None or bool(pack.r_in_blocks),
            "has_output_restore": bool(pack.r_out_blocks),
            "transform_keys": transform_keys,
            "qweight_shape": list(qweight.shape),
            "scales_shape": list(scales.shape),
            "qzeros_shape": list(qzeros.shape),
            "validation": {
                "pack_roundtrip_mse": pack_mse,
                "mse_vs_quantvla_transformed_weight": original_mse,
                "mean_abs_error_vs_quantvla_transformed_weight": mean_abs,
                "max_abs_error_vs_quantvla_transformed_weight": max_abs,
            },
        }
        if layer_info["has_input_transform"]:
            layer_info["note"] = "Runtime must apply QuantVLA input transform before this linear."
        if layer_info["has_output_restore"]:
            layer_info["output_restore_note"] = "Runtime must apply QuantVLA output restore after this linear."
        layers.append(layer_info)
        converted += 1

    report = {
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "pack_dir": str(Path(args.pack_dir).resolve()),
        "output": str(out_dir.resolve()),
        "bits": args.bits,
        "group_size": args.group_size,
        "scale_source": args.scale_source,
        "row_rot_mode": args.row_rot_mode,
        "layer_regex": args.layer_regex,
        "exclude_regex": args.exclude_regex,
        "target_linear_layers": target,
        "successfully_converted": converted,
        "unmatched_checkpoint_keys": len(unmatched),
        "skipped_by_regex": len(skipped),
        "fallback_fp16_layers": 0,
        "conversion_kind": "quantvla_transformed_weight_to_gptq_like_int4",
        "direct_gptq_marlin_compatible_input": False,
        "requires_transform_aware_runtime": any(
            item["has_input_transform"] or item["has_output_restore"] for item in layers
        ),
        "layers": layers,
        "unmatched": unmatched,
        "skipped": skipped,
    }

    print("========== QuantVLA -> GPTQ-like Conversion ==========")
    print(
        _layer_report_header().format(
            target=target,
            converted=converted,
            unmatched=len(unmatched),
            skipped=len(skipped),
        )
    )
    if unmatched:
        print("\nUnmatched layers:")
        for item in unmatched[:20]:
            print(f"- {item['layer']}: {item.get('reason', 'no matching dense weight key')}")
    if converted:
        avg_mse = sum(item["validation"]["mse_vs_quantvla_transformed_weight"] for item in layers) / converted
        print(f"\nAverage MSE vs QuantVLA-transformed dense weight: {avg_mse:.6e}")

    if args.dry_run:
        print("\nDry run only; no files written.")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        from safetensors.torch import save_file

        save_file(output_tensors, out_dir / "model.safetensors", metadata={"format": "pt"})
        _save_json(out_dir / "quantize_config.json", _make_quantize_config(args.bits, args.group_size))
        _save_json(out_dir / "conversion_report.json", report)
        _save_json(out_dir / "quantvla_marlin_meta.json", {"layers": layers})
        if transform_arrays:
            np.savez_compressed(out_dir / "quantvla_transforms.npz", **transform_arrays)
        print(f"\nWrote {out_dir}")

    if args.json_report:
        _save_json(Path(args.json_report), report)
        print(f"Wrote report {args.json_report}")


if __name__ == "__main__":
    main()
