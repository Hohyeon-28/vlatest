"""Utilities for the QuantVLA-weight-to-Marlin-format experiment.

These helpers intentionally mirror the QuantVLA/DuQuant implementation in
QuantVLA/gr00t/quantization/duquant_preprocess.py without importing GR00T.
The goal is to inspect and convert DuQuant pack artifacts in isolation.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch


@dataclass
class QuantVLAPack:
    layer_name: str
    weight_scale: np.ndarray
    meta: dict[str, Any]
    perm: Optional[np.ndarray] = None
    r_in_blocks: Optional[dict[int, np.ndarray]] = None
    r_out_blocks: Optional[dict[int, np.ndarray]] = None
    raw_keys: tuple[str, ...] = ()


def sanitize_name(name: str) -> str:
    import re

    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return name.replace("..", ".")


def qmax(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def load_quantvla_pack(path: str | os.PathLike[str]) -> QuantVLAPack:
    p = Path(path)
    with np.load(p, allow_pickle=False) as f:
        raw_keys = tuple(f.files)
        weight_scale = f["weight_scale"] if "weight_scale" in f.files else np.array([], dtype=np.float32)
        perm = f["perm"] if "perm" in f.files else None

        r_in_blocks = None
        if "R_in_blocks" in f.files:
            r_in_blocks = {}
            for b in f["R_in_blocks"]:
                r_in_blocks[int(b)] = f[f"Rin_{int(b)}"]

        r_out_blocks = None
        if "R_out_blocks" in f.files:
            r_out_blocks = {}
            for b in f["R_out_blocks"]:
                r_out_blocks[int(b)] = f[f"Rout_{int(b)}"]

        meta = {}
        if "meta" in f.files:
            try:
                meta = json.loads(f["meta"].tolist())
            except Exception:
                meta = {}

    return QuantVLAPack(
        layer_name=p.stem,
        weight_scale=weight_scale.astype(np.float32, copy=False),
        meta=meta,
        perm=perm.astype(np.int64, copy=False) if perm is not None else None,
        r_in_blocks=r_in_blocks,
        r_out_blocks=r_out_blocks,
        raw_keys=raw_keys,
    )


def iter_pack_files(pack_dir: str | os.PathLike[str]) -> list[Path]:
    return sorted(Path(pack_dir).glob("*.npz"))


def has_integer_weight_keys(keys: Iterable[str]) -> bool:
    needles = ("qweight", "int_weight", "q_weight", "packed_weight", "weight_int", "w_int")
    return any(any(n in key.lower() for n in needles) for key in keys)


def apply_quantvla_weight_transform(
    weight: torch.Tensor,
    pack: QuantVLAPack,
    *,
    apply_row_rot: bool = True,
) -> torch.Tensor:
    """Return the transformed W' used by DuQuant forward before fake quantization.

    Input and output are shaped like torch.nn.Linear.weight: [out_features, in_features].
    """
    w_t = weight.detach().to(dtype=torch.float32, device="cpu").clone()
    out_features, in_features = w_t.shape

    if pack.perm is not None:
        perm = torch.from_numpy(pack.perm).long()
        w_t = w_t.index_select(dim=1, index=perm)

    if pack.r_in_blocks:
        block_size = int(pack.meta.get("block_size", next(iter(pack.r_in_blocks.values())).shape[0]))
        w_next = w_t.clone()
        n_blocks = math.ceil(in_features / block_size)
        for b in range(n_blocks):
            if b not in pack.r_in_blocks:
                continue
            start = b * block_size
            end = min((b + 1) * block_size, in_features)
            r = torch.from_numpy(pack.r_in_blocks[b][: end - start, : end - start]).to(w_t)
            w_next[:, start:end] = w_t[:, start:end] @ r
        w_t = w_next

    if apply_row_rot and pack.r_out_blocks:
        block_out_size = int(pack.meta.get("block_out_size", next(iter(pack.r_out_blocks.values())).shape[0]))
        w_next = w_t.clone()
        n_blocks = math.ceil(out_features / block_out_size)
        for b in range(n_blocks):
            if b not in pack.r_out_blocks:
                continue
            start = b * block_out_size
            end = min((b + 1) * block_out_size, out_features)
            r = torch.from_numpy(pack.r_out_blocks[b][: end - start, : end - start]).to(w_t)
            w_next[start:end, :] = r @ w_t[start:end, :]
        w_t = w_next

    return w_t.contiguous()


def apply_quantvla_input_transform(x: torch.Tensor, pack: QuantVLAPack) -> torch.Tensor:
    y = x
    in_features = y.shape[-1]
    if pack.perm is not None:
        perm = torch.from_numpy(pack.perm).to(device=y.device, dtype=torch.long)
        y = y.index_select(dim=-1, index=perm)
    if pack.r_in_blocks:
        block_size = int(pack.meta.get("block_size", next(iter(pack.r_in_blocks.values())).shape[0]))
        y2 = y.reshape(-1, in_features)
        out = y2.clone()
        n_blocks = math.ceil(in_features / block_size)
        for b in range(n_blocks):
            if b not in pack.r_in_blocks:
                continue
            start = b * block_size
            end = min((b + 1) * block_size, in_features)
            r = torch.from_numpy(pack.r_in_blocks[b][: end - start, : end - start]).to(y2)
            out[:, start:end] = y2[:, start:end] @ r
        y = out.reshape(*y.shape)
    return y


def apply_quantvla_output_restore(y: torch.Tensor, pack: QuantVLAPack) -> torch.Tensor:
    if not pack.r_out_blocks:
        return y
    out_features = y.shape[-1]
    block_out_size = int(pack.meta.get("block_out_size", next(iter(pack.r_out_blocks.values())).shape[0]))
    y2 = y.reshape(-1, out_features)
    out = y2.clone()
    n_blocks = math.ceil(out_features / block_out_size)
    for b in range(n_blocks):
        if b not in pack.r_out_blocks:
            continue
        start = b * block_out_size
        end = min((b + 1) * block_out_size, out_features)
        r = torch.from_numpy(pack.r_out_blocks[b][: end - start, : end - start]).to(y2)
        out[:, start:end] = y2[:, start:end] @ r
    return out.reshape(*y.shape)


def compute_mse_scales(weight: torch.Tensor, bits: int) -> torch.Tensor:
    if bits <= 0:
        return torch.ones(weight.shape[0], dtype=weight.dtype, device=weight.device)
    max_q = qmax(bits)
    max_abs = torch.amax(torch.abs(weight), dim=1).clamp_min(1e-8)
    alphas = torch.tensor([0.5, 0.75, 1.0, 1.25, 1.5], dtype=weight.dtype, device=weight.device)
    scales = (max_abs[:, None] / max_q) / alphas[None, :]
    q = torch.round(weight[:, None, :] / scales[:, :, None]).clamp(-max_q - 1, max_q)
    rec = q * scales[:, :, None]
    mse = torch.mean((rec - weight[:, None, :]) ** 2, dim=2)
    idx = torch.argmin(mse, dim=1)
    return scales[torch.arange(weight.shape[0], device=weight.device), idx].contiguous()


def choose_scales(weight: torch.Tensor, pack: QuantVLAPack, bits: int, source: str) -> torch.Tensor:
    source = source.lower()
    if source == "pack":
        if pack.weight_scale.size == 0:
            raise ValueError(f"Pack for {pack.layer_name} has no weight_scale")
        return torch.from_numpy(pack.weight_scale).to(dtype=weight.dtype, device=weight.device).contiguous()
    if source == "maxabs":
        return (torch.amax(torch.abs(weight), dim=1).clamp_min(1e-8) / qmax(bits)).contiguous()
    if source == "mse":
        return compute_mse_scales(weight, bits)
    raise ValueError(f"Unknown scale source: {source}")


def quantize_signed_int(weight: torch.Tensor, scales: torch.Tensor, bits: int) -> torch.Tensor:
    max_q = qmax(bits)
    q = torch.round(weight / scales[:, None]).clamp(-max_q - 1, max_q)
    return q.to(torch.int16).contiguous()


def pack_int4_qweight(q_signed_out_in: torch.Tensor) -> torch.Tensor:
    """Pack signed [-8, 7] [out, in] codes as GPTQ-like uint nibbles [ceil(in/8), out]."""
    q_unsigned_in_out = (q_signed_out_in.to(torch.int32) + 8).t().contiguous()
    in_features, out_features = q_unsigned_in_out.shape
    rows = math.ceil(in_features / 8)
    packed = torch.zeros((rows, out_features), dtype=torch.int32)
    for offset in range(8):
        src = torch.zeros((rows, out_features), dtype=torch.int32)
        valid = torch.arange(rows) * 8 + offset
        mask = valid < in_features
        if mask.any():
            src[mask] = q_unsigned_in_out[valid[mask]]
        packed |= (src & 0xF) << (4 * offset)
    return packed.contiguous()


def pack_int4_cols(values: torch.Tensor, out_features: int) -> torch.Tensor:
    """Pack [groups, out] unsigned nibble values along output columns."""
    values = values.to(torch.int32).contiguous()
    groups = values.shape[0]
    cols = math.ceil(out_features / 8)
    packed = torch.zeros((groups, cols), dtype=torch.int32)
    for offset in range(8):
        idx = torch.arange(cols) * 8 + offset
        mask = idx < out_features
        if mask.any():
            src = torch.zeros((groups, cols), dtype=torch.int32)
            src[:, mask] = values[:, idx[mask]]
            packed |= (src & 0xF) << (4 * offset)
    return packed.contiguous()


def repeat_scales_for_groups(scales_out: torch.Tensor, in_features: int, group_size: int) -> torch.Tensor:
    groups = math.ceil(in_features / group_size)
    return scales_out.reshape(1, -1).repeat(groups, 1).contiguous()


def make_symmetric_qzeros(groups: int, out_features: int) -> torch.Tensor:
    # Existing GPTQ dequant code interprets unpack(qzeros) + 1 as the zero value.
    # Pack 7 so the effective zero point is 8, matching q_unsigned = q_signed + 8.
    logical = torch.full((groups, out_features), 7, dtype=torch.int32)
    return pack_int4_cols(logical, out_features)


def unpack_int4_qweight(qweight: torch.Tensor, in_features: int) -> torch.Tensor:
    shifts = torch.arange(8, device=qweight.device, dtype=torch.int32) * 4
    unpacked = ((qweight.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF).permute(0, 2, 1).reshape(
        -1, qweight.shape[1]
    )
    return unpacked[:in_features, :].contiguous()


def dequantize_gptq_like(qweight: torch.Tensor, scales: torch.Tensor, in_features: int, group_size: int) -> torch.Tensor:
    q_unsigned = unpack_int4_qweight(qweight, in_features).to(torch.float32)
    scales = scales.to(device=q_unsigned.device, dtype=torch.float32)
    groups = scales.shape[0]
    group_ids = (torch.arange(in_features, device=scales.device) // group_size).clamp(0, groups - 1)
    weight_in_out = (q_unsigned - 8.0) * scales[group_ids]
    return weight_in_out.t().contiguous()


def load_weight_tensors(checkpoint: str | os.PathLike[str]) -> dict[str, torch.Tensor]:
    path = Path(checkpoint)
    tensors: dict[str, torch.Tensor] = {}
    if path.is_dir():
        files = sorted(path.glob("*.safetensors"))
        if files:
            from safetensors import safe_open

            for file in files:
                with safe_open(file, framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        if key.endswith(".weight") or key.endswith(".bias"):
                            tensors[key] = handle.get_tensor(key)
            return tensors
        for file in sorted(path.glob("pytorch_model*.bin")):
            state = torch.load(file, map_location="cpu")
            state = state.get("state_dict", state)
            for key, value in state.items():
                if torch.is_tensor(value) and (key.endswith(".weight") or key.endswith(".bias")):
                    tensors[key] = value.cpu()
            return tensors
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.endswith(".weight") or key.endswith(".bias"):
                    tensors[key] = handle.get_tensor(key)
        return tensors
    state = torch.load(path, map_location="cpu")
    state = state.get("state_dict", state)
    for key, value in state.items():
        if torch.is_tensor(value) and (key.endswith(".weight") or key.endswith(".bias")):
            tensors[key] = value.cpu()
    return tensors


def candidate_weight_keys(layer_name: str, strip_prefixes: Iterable[str]) -> list[str]:
    names = [layer_name]
    for prefix in strip_prefixes:
        if prefix and layer_name.startswith(prefix):
            names.append(layer_name[len(prefix) :])
    if "language_model." in layer_name:
        names.append(layer_name.split("language_model.", 1)[1])
    if "model.layers." in layer_name:
        names.append("model.layers." + layer_name.split("model.layers.", 1)[1])
    result = []
    for name in names:
        result.append(name if name.endswith(".weight") else f"{name}.weight")
    return list(dict.fromkeys(result))


def output_prefix_for_layer(layer_name: str, strip_prefixes: Iterable[str]) -> str:
    out = layer_name
    for prefix in strip_prefixes:
        if prefix and out.startswith(prefix):
            out = out[len(prefix) :]
    if out.endswith(".weight"):
        out = out[: -len(".weight")]
    return out
