"""Reference runtime for converted QuantVLA -> GPTQ-like INT4 layers.

This module deliberately uses Torch dequantization instead of a Marlin kernel.
Its job is to verify semantics: QuantVLA input transforms, GPTQ-like INT4
weight reconstruction, and optional QuantVLA output restore are all applied in
the correct order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from quantvla_marlin_utils import (
    QuantVLAPack,
    apply_quantvla_input_transform,
    apply_quantvla_output_restore,
    dequantize_gptq_like,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _pack_from_transform_arrays(layer_info: dict[str, Any], arrays: np.lib.npyio.NpzFile) -> QuantVLAPack:
    transform_keys = layer_info.get("transform_keys", {})
    perm = arrays[transform_keys["perm"]] if "perm" in transform_keys else None

    r_in_blocks = None
    if "r_in_blocks" in transform_keys:
        r_in_blocks = {int(block): arrays[key] for block, key in transform_keys["r_in_blocks"].items()}

    r_out_blocks = None
    if "r_out_blocks" in transform_keys:
        r_out_blocks = {int(block): arrays[key] for block, key in transform_keys["r_out_blocks"].items()}

    return QuantVLAPack(
        layer_name=layer_info["output_prefix"],
        weight_scale=np.array([], dtype=np.float32),
        meta=layer_info.get("pack_meta", {}),
        perm=perm,
        r_in_blocks=r_in_blocks,
        r_out_blocks=r_out_blocks,
        raw_keys=(),
    )


class QuantVLAGPTQLikeReferenceLinear(torch.nn.Module):
    """Torch reference layer for a converted QuantVLA/GPTQ-like linear."""

    def __init__(
        self,
        *,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        in_features: int,
        out_features: int,
        group_size: int,
        pack: QuantVLAPack,
        row_rot_mode: str,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.row_rot_mode = row_rot_mode
        self.pack = pack
        self.register_buffer("qweight", qweight.to(torch.int32).contiguous(), persistent=False)
        self.register_buffer("scales", scales.to(torch.float32).contiguous(), persistent=False)
        self.register_buffer("qzeros", qzeros.to(torch.int32).contiguous(), persistent=False)
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.to(torch.float32).contiguous(), persistent=False)

    @torch.inference_mode()
    def dequantized_weight(self) -> torch.Tensor:
        return dequantize_gptq_like(self.qweight, self.scales, self.in_features, self.group_size)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected last dim {self.in_features}, got {x.shape[-1]}")

        work = x.to(torch.float32)
        work = apply_quantvla_input_transform(work, self.pack)
        weight = self.dequantized_weight().to(work.device)
        bias = self.bias.to(work.device) if self.bias is not None else None
        out = F.linear(work, weight, bias)
        if self.row_rot_mode == "restore":
            out = apply_quantvla_output_restore(out, self.pack)
        return out


def load_reference_layer(
    checkpoint_dir: str | Path,
    output_prefix: str,
    *,
    device: str | torch.device = "cpu",
) -> QuantVLAGPTQLikeReferenceLinear:
    checkpoint_dir = Path(checkpoint_dir)
    from safetensors.torch import load_file

    tensors = load_file(checkpoint_dir / "model.safetensors", device="cpu")
    report = _load_json(checkpoint_dir / "conversion_report.json")
    layer_info = next((item for item in report["layers"] if item["output_prefix"] == output_prefix), None)
    if layer_info is None:
        raise KeyError(f"Layer not found in conversion_report.json: {output_prefix}")

    transforms_path = checkpoint_dir / "quantvla_transforms.npz"
    arrays = np.load(transforms_path, allow_pickle=False) if transforms_path.exists() else {}

    pack = _pack_from_transform_arrays(layer_info, arrays)
    bias = tensors.get(f"{output_prefix}.bias")
    layer = QuantVLAGPTQLikeReferenceLinear(
        qweight=tensors[f"{output_prefix}.qweight"],
        scales=tensors[f"{output_prefix}.scales"],
        qzeros=tensors[f"{output_prefix}.qzeros"],
        in_features=layer_info["in_features"],
        out_features=layer_info["out_features"],
        group_size=layer_info["group_size"],
        pack=pack,
        row_rot_mode=layer_info["row_rot_mode"],
        bias=bias,
    )
    return layer.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--layer", required=True, help="Converted output_prefix, e.g. model.layers.0.self_attn.q_proj")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    layer = load_reference_layer(args.checkpoint_dir, args.layer, device=args.device)
    x = torch.randn(args.batch, args.seq, layer.in_features, device=args.device)
    y = layer(x)
    print("========== QuantVLA GPTQ-like Reference Layer ==========")
    print(f"layer: {args.layer}")
    print(f"input: {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}")
    print(f"device: {args.device}")
    print(f"row_rot_mode: {layer.row_rot_mode}")
    print(f"weight: {tuple(layer.dequantized_weight().shape)}")


if __name__ == "__main__":
    main()
