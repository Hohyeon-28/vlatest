"""FakeQuant and RealQuant execution modes for converted QuantVLA weights.

Definitions used here:

- FakeQuant: converted QuantVLA GPTQ-like qweight/scales are dequantized once
  into a dense torch weight, then executed with torch.nn.functional.linear.
- RealQuant: the same qweight/scales are executed by vLLM's GPTQ-Marlin Linear
  path. QuantVLA input/output transforms are preserved around the matmul.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from quantvla_marlin_utils import (
    QuantVLAPack,
    apply_quantvla_input_transform,
    apply_quantvla_output_restore,
    dequantize_gptq_like,
)


GPTQ_QUANT_MODE_FAKE = "fake"
GPTQ_QUANT_MODE_REAL = "real"


@dataclass
class ConvertedLayerPack:
    prefix: str
    qweight: torch.Tensor
    scales: torch.Tensor
    qzeros: torch.Tensor | None
    bias: torch.Tensor | None
    layer_info: dict[str, Any]
    transform_pack: QuantVLAPack

    @property
    def in_features(self) -> int:
        return int(self.layer_info["in_features"])

    @property
    def out_features(self) -> int:
        return int(self.layer_info["out_features"])

    @property
    def group_size(self) -> int:
        return int(self.layer_info["group_size"])

    @property
    def row_rot_mode(self) -> str:
        return str(self.layer_info.get("row_rot_mode", "restore"))


def dtype_from_name(name: str) -> torch.dtype:
    value = name.strip().lower()
    if value in ("bf16", "bfloat16"):
        return torch.bfloat16
    if value in ("fp16", "float16"):
        return torch.float16
    if value in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _pack_from_transform_arrays(layer_info: dict[str, Any], arrays: Any) -> QuantVLAPack:
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


class ConvertedQuantVLACheckpoint:
    """Small loader for outputs from convert_quantvla_to_gptq_like.py."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Converted checkpoint not found: {self.root}")

        from safetensors.torch import load_file

        self.tensors = load_file(self.root / "model.safetensors", device="cpu")
        self.report = _load_json(self.root / "conversion_report.json")
        qcfg = self.root / "quantize_config.json"
        self.quant_config = _load_json(qcfg) if qcfg.exists() else {}

        transforms_path = self.root / "quantvla_transforms.npz"
        self.transforms = np.load(transforms_path, allow_pickle=False) if transforms_path.exists() else {}
        self.layers = list(self.report.get("layers", []))
        self._by_prefix = {item["output_prefix"]: item for item in self.layers}

    def prefixes(self) -> list[str]:
        return sorted(self._by_prefix)

    def load_pack(self, prefix: str) -> ConvertedLayerPack:
        if prefix not in self._by_prefix:
            raise KeyError(f"Layer not found in converted checkpoint report: {prefix}")
        info = self._by_prefix[prefix]
        qweight_key = f"{prefix}.qweight"
        scales_key = f"{prefix}.scales"
        qzeros_key = f"{prefix}.qzeros"
        bias_key = f"{prefix}.bias"
        if qweight_key not in self.tensors or scales_key not in self.tensors:
            raise KeyError(f"Missing qweight/scales tensors for {prefix}")
        return ConvertedLayerPack(
            prefix=prefix,
            qweight=self.tensors[qweight_key],
            scales=self.tensors[scales_key],
            qzeros=self.tensors.get(qzeros_key),
            bias=self.tensors.get(bias_key),
            layer_info=info,
            transform_pack=_pack_from_transform_arrays(info, self.transforms),
        )


class QuantVLAFakeQuantLinear(nn.Module):
    """FakeQuant/reference path: same W4 tensors, dense torch F.linear."""

    def __init__(self, pack: ConvertedLayerPack, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.name = pack.prefix
        self.in_features = pack.in_features
        self.out_features = pack.out_features
        self.group_size = pack.group_size
        self.row_rot_mode = pack.row_rot_mode
        self.transform_pack = pack.transform_pack
        weight = dequantize_gptq_like(pack.qweight, pack.scales.to(torch.float32), self.in_features, self.group_size)
        self.register_buffer("weight", weight.to(dtype=dtype).contiguous(), persistent=False)
        if pack.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", pack.bias.to(dtype=dtype).contiguous(), persistent=False)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        work = apply_quantvla_input_transform(x.to(torch.float32), self.transform_pack)
        out = F.linear(work.to(dtype=self.weight.dtype), self.weight, self.bias)
        if self.row_rot_mode == "restore":
            out = apply_quantvla_output_restore(out.to(torch.float32), self.transform_pack).to(dtype=self.weight.dtype)
        return out


class QuantVLARealQuantMarlinLinear(nn.Module):
    """RealQuant path: same W4 tensors, vLLM GPTQ-Marlin matmul."""

    def __init__(self, pack: ConvertedLayerPack, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.name = pack.prefix
        self.in_features = pack.in_features
        self.out_features = pack.out_features
        self.input_dtype = dtype
        self.group_size = pack.group_size
        self.row_rot_mode = pack.row_rot_mode
        self.transform_pack = pack.transform_pack
        self._processed = False

        if pack.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(pack.bias.detach().clone().to(dtype=dtype), requires_grad=False)

        self._init_vllm_method(pack)
        self._copy_packed_tensors(pack)

    def _init_vllm_method(self, pack: ConvertedLayerPack) -> None:
        try:
            from vllm.model_executor.layers.quantization.gptq_marlin import (
                GPTQMarlinConfig,
                GPTQMarlinLinearMethod,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "vLLM with GPTQ-Marlin support is required for RealQuant. "
                "Install vllm in this environment, or run only FakeQuant."
            ) from exc

        cfg = {
            "quant_method": "gptq",
            "bits": int(pack.layer_info.get("bits", 4)),
            "group_size": int(pack.layer_info.get("group_size", 128)),
            "desc_act": False,
            "sym": True,
            "lm_head": False,
        }
        self.quant_config = GPTQMarlinConfig.from_config(cfg)
        self.quant_method = GPTQMarlinLinearMethod(self.quant_config)
        self.quant_method.create_weights(
            self,
            input_size_per_partition=self.in_features,
            output_partition_sizes=[self.out_features],
            input_size=self.in_features,
            output_size=self.out_features,
            params_dtype=self.input_dtype,
        )
        if hasattr(self, "qzeros"):
            self.qzeros.data.zero_()
        if hasattr(self, "g_idx"):
            self.g_idx.data.copy_(torch.arange(self.g_idx.numel(), dtype=self.g_idx.dtype).view_as(self.g_idx))

    def _copy_parameter(self, name: str, tensor: torch.Tensor | None, required: bool = False) -> None:
        if tensor is None:
            if required:
                raise ValueError(f"Missing required GPTQ tensor for {self.name}: {name}")
            return
        param = getattr(self, name)
        if tuple(param.shape) != tuple(tensor.shape):
            raise ValueError(
                f"Shape mismatch for {self.name}.{name}: checkpoint {tuple(tensor.shape)} "
                f"vs vLLM expected {tuple(param.shape)}"
            )
        param.data.copy_(tensor.to(dtype=param.dtype))

    def _copy_packed_tensors(self, pack: ConvertedLayerPack) -> None:
        self._copy_parameter("qweight", pack.qweight, required=True)
        self._copy_parameter("scales", pack.scales, required=True)
        self._copy_parameter("qzeros", pack.qzeros, required=False)

    def _process_once(self) -> None:
        if self._processed:
            return
        self.quant_method.process_weights_after_loading(self)
        self._processed = True

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._process_once()
        work = apply_quantvla_input_transform(x.to(torch.float32), self.transform_pack).to(dtype=self.input_dtype)
        original_shape = work.shape[:-1]
        work_2d = work.reshape(-1, work.shape[-1])
        out_2d = self.quant_method.apply(self, work_2d, self.bias)
        out = out_2d.reshape(*original_shape, self.out_features)
        if self.row_rot_mode == "restore":
            out = apply_quantvla_output_restore(out.to(torch.float32), self.transform_pack).to(dtype=self.input_dtype)
        return out


def mode_definitions() -> dict[str, str]:
    return {
        GPTQ_QUANT_MODE_FAKE: "QuantVLA-converted GPTQ-like qweight/scales -> dense dequant -> torch F.linear",
        GPTQ_QUANT_MODE_REAL: "same qweight/scales -> vLLM GPTQ-Marlin Linear; QuantVLA transforms preserved",
    }
