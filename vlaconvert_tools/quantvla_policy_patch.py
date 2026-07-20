"""Patch GR00T language-model Linear layers with converted QuantVLA weights."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from quantvla_gptq_modes import (
    ConvertedLayerPack,
    ConvertedQuantVLACheckpoint,
    QuantVLAFakeQuantLinear,
    QuantVLARealQuantMarlinLinear,
    dtype_from_name,
)


DEFAULT_INCLUDE = (
    r".*("
    r"backbone\.eagle_model\.language_model\..*\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    r"|"
    r"action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)"
    r")$"
)
DEFAULT_EXCLUDE = r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)"


@dataclass
class ReplacementIssue:
    layer: str
    reason: str


@dataclass
class QuantVLAReplacementReport:
    quant_mode: str
    checkpoint: str
    target_linear_layers: int
    successfully_replaced: int
    unmatched_checkpoint_keys: list[str] = field(default_factory=list)
    unreplaced_target_layers: list[ReplacementIssue] = field(default_factory=list)
    fallback_fp16_layers: list[ReplacementIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["unmatched_checkpoint_keys_count"] = len(self.unmatched_checkpoint_keys)
        data["unreplaced_target_layers_count"] = len(self.unreplaced_target_layers)
        data["fallback_fp16_layers_count"] = len(self.fallback_fp16_layers)
        return data

    def print_summary(self, prefix: str = "[QuantVLA-GPTQ]") -> None:
        print(f"{prefix} Quant mode: {self.quant_mode}")
        print(f"Target Linear layers: {self.target_linear_layers}")
        print(f"Successfully replaced: {self.successfully_replaced}")
        print(f"Unmatched checkpoint keys: {len(self.unmatched_checkpoint_keys)}")
        print(f"Unreplaced target layers: {len(self.unreplaced_target_layers)}")
        print(f"Fallback FP16 layers: {len(self.fallback_fp16_layers)}")
        for key in self.unmatched_checkpoint_keys:
            print(f"{prefix} UNMATCHED_CHECKPOINT_KEY {key}")
        for issue in self.unreplaced_target_layers:
            print(f"{prefix} UNREPLACED_TARGET_LAYER {issue.layer} :: {issue.reason}")
        for issue in self.fallback_fp16_layers:
            print(f"{prefix} FALLBACK_FP16_LAYER {issue.layer} :: {issue.reason}")

    def write_json(self, path: str | os.PathLike[str] | None) -> None:
        if not path:
            return
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        print(f"[QuantVLA-GPTQ] Replacement report written to {out}")


def normalize_quant_mode(mode: str | None) -> str:
    value = (mode or "real").strip().lower().replace("-", "_")
    aliases = {
        "real": "real",
        "real_quant": "real",
        "marlin": "real",
        "gptq_marlin": "real",
        "fake": "fake",
        "fake_quant": "fake",
        "torch": "fake",
        "dequant": "fake",
        "dequant_torch": "fake",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported quant mode {mode!r}; use real or fake")
    return aliases[value]


def _candidate_checkpoint_prefixes(module_name: str) -> list[str]:
    prefixes = [module_name]
    markers = [
        "backbone.eagle_model.language_model.",
        "eagle_model.language_model.",
        "language_model.",
    ]
    for marker in markers:
        if marker in module_name:
            prefixes.append(module_name.split(marker, 1)[1])
    if "model.layers." in module_name:
        prefixes.append("model.layers." + module_name.split("model.layers.", 1)[1])
    return list(dict.fromkeys(prefixes))


def _get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _iter_target_linears(model: nn.Module, include_regex: str, exclude_regex: str) -> list[tuple[str, nn.Linear]]:
    include = re.compile(include_regex)
    exclude = re.compile(exclude_regex) if exclude_regex else None
    targets: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not include.search(name):
            continue
        if exclude is not None and exclude.search(name):
            continue
        targets.append((name, module))
    return targets


def _load_matching_pack(
    index: ConvertedQuantVLACheckpoint,
    module_name: str,
) -> tuple[str, ConvertedLayerPack] | None:
    for prefix in _candidate_checkpoint_prefixes(module_name):
        if prefix in index.prefixes():
            return prefix, index.load_pack(prefix)
    return None


def _validate_shape(name: str, base: nn.Linear, pack: ConvertedLayerPack) -> None:
    if base.in_features != pack.in_features or base.out_features != pack.out_features:
        raise ValueError(
            f"shape mismatch for {name}: module=({base.in_features}, {base.out_features}) "
            f"checkpoint=({pack.in_features}, {pack.out_features})"
        )


def _make_wrapper(
    mode: str,
    base: nn.Linear,
    pack: ConvertedLayerPack,
    dtype: torch.dtype,
) -> nn.Module:
    _validate_shape(pack.prefix, base, pack)
    if mode == "fake":
        wrapper = QuantVLAFakeQuantLinear(pack, dtype=dtype)
    elif mode == "real":
        wrapper = QuantVLARealQuantMarlinLinear(pack, dtype=dtype)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return wrapper.to(device=base.weight.device)


def replace_quantvla_converted_linears(
    model: nn.Module,
    checkpoint: str | os.PathLike[str],
    *,
    mode: str = "real",
    include_regex: str = DEFAULT_INCLUDE,
    exclude_regex: str = DEFAULT_EXCLUDE,
    strict: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> QuantVLAReplacementReport:
    mode = normalize_quant_mode(mode)
    index = ConvertedQuantVLACheckpoint(checkpoint)
    targets = _iter_target_linears(model, include_regex, exclude_regex)
    matched_prefixes: set[str] = set()
    replaced = 0
    unreplaced: list[ReplacementIssue] = []

    for name, module in targets:
        matched = _load_matching_pack(index, name)
        if matched is None:
            unreplaced.append(ReplacementIssue(name, "missing converted qweight/scales tensors"))
            continue
        prefix, pack = matched
        matched_prefixes.add(prefix)
        try:
            wrapper = _make_wrapper(mode, module, pack, dtype=dtype)
        except Exception as exc:
            unreplaced.append(
                ReplacementIssue(name, f"failed to initialize {mode} wrapper: {type(exc).__name__}: {exc}")
            )
            continue
        parent, child_name = _get_parent_module(model, name)
        setattr(parent, child_name, wrapper)
        replaced += 1

    unmatched = sorted(set(index.prefixes()) - matched_prefixes)
    report = QuantVLAReplacementReport(
        quant_mode=mode,
        checkpoint=str(checkpoint),
        target_linear_layers=len(targets),
        successfully_replaced=replaced,
        unmatched_checkpoint_keys=unmatched,
        unreplaced_target_layers=unreplaced,
        fallback_fp16_layers=[],
    )
    model._quantvla_converted_replacement_report = report  # type: ignore[attr-defined]
    if strict and (report.unreplaced_target_layers or report.fallback_fp16_layers):
        report.print_summary()
        raise RuntimeError("QuantVLA converted replacement was not exact; see report above")
    return report


def replace_quantvla_converted_from_env(model: nn.Module) -> QuantVLAReplacementReport | None:
    checkpoint = os.environ.get("QUANTVLA_CONVERTED_CHECKPOINT")
    if not checkpoint:
        return None
    mode = normalize_quant_mode(os.environ.get("QUANTVLA_QUANT_MODE", "real"))
    include = os.environ.get("QUANTVLA_INCLUDE", DEFAULT_INCLUDE)
    exclude = os.environ.get("QUANTVLA_EXCLUDE", DEFAULT_EXCLUDE)
    strict = os.environ.get("QUANTVLA_STRICT", "1") not in ("0", "false", "False")
    dtype = dtype_from_name(os.environ.get("QUANTVLA_DTYPE", "bfloat16"))
    report = replace_quantvla_converted_linears(
        model,
        checkpoint,
        mode=mode,
        include_regex=include,
        exclude_regex=exclude,
        strict=strict,
        dtype=dtype,
    )
    report.print_summary()
    report.write_json(os.environ.get("QUANTVLA_REPORT"))
    return report
