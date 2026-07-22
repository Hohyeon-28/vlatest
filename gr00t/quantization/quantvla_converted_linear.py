"""FakeQuant and RealQuant execution modes for converted QuantVLA weights.

Definitions used here:

- FakeQuant: converted QuantVLA GPTQ-like qweight/scales are dequantized once
  into a dense torch weight, then executed with torch.nn.functional.linear.
- RealQuant: the same qweight/scales are executed by vLLM's GPTQ-Marlin Linear
  path. QuantVLA input/output transforms are preserved around the matmul.
"""

from __future__ import annotations

import json
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .quantvla_converted_utils import (
    QuantVLAPack,
    apply_quantvla_input_transform,
    apply_quantvla_output_restore,
    dequantize_gptq_like,
)

import os
import re
from dataclasses import asdict, dataclass, field


GPTQ_QUANT_MODE_FAKE = "fake"
GPTQ_QUANT_MODE_REAL = "real"


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False", "")


def _load_vllm_gptq_marlin_symbols() -> tuple[type, type, str]:
    """Load vLLM's GPTQ-Marlin Linear symbols with actionable diagnostics.

    vLLM keeps GPTQ-Marlin behind internal APIs. Those paths have changed
    across releases, so a plain "vLLM is required" error is not enough when
    the package is installed but its layout differs from what this wrapper
    expects.
    """

    module_names = ["vllm.model_executor.layers.quantization.gptq_marlin"]
    try:
        import vllm

        package_file = getattr(vllm, "__file__", None)
        if package_file:
            package_root = Path(package_file).resolve().parent
            for path in package_root.rglob("*.py"):
                lower_full = str(path).lower()
                if "gptq" not in lower_full or "marlin" not in lower_full:
                    continue
                rel = path.relative_to(package_root).with_suffix("")
                module_names.append("vllm." + ".".join(rel.parts))
    except Exception as exc:  # pragma: no cover - environment dependent
        module_names.append(f"<vllm discovery failed: {type(exc).__name__}: {exc}>")
    module_names = list(dict.fromkeys(module_names))

    errors: list[str] = []
    for module_name in module_names:
        if module_name.startswith("<"):
            errors.append(module_name)
            continue
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            errors.append(f"{module_name}: import failed: {type(exc).__name__}: {exc}")
            continue

        config_cls = getattr(module, "GPTQMarlinConfig", None)
        method_cls = getattr(module, "GPTQMarlinLinearMethod", None)
        if config_cls is not None and method_cls is not None:
            return config_cls, method_cls, module_name

        marlin_exports = sorted(name for name in dir(module) if "Marlin" in name or "marlin" in name)
        errors.append(
            f"{module_name}: missing GPTQMarlinConfig/GPTQMarlinLinearMethod; "
            f"marlin exports={marlin_exports}"
        )

    raise RuntimeError(
        "Could not load vLLM GPTQ-Marlin Linear API. "
        "This usually means the installed vLLM build has a different internal API "
        "or was built without the required Marlin components.\n"
        + "\n".join(errors)
    )


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
            GPTQMarlinConfig, GPTQMarlinLinearMethod, module_name = _load_vllm_gptq_marlin_symbols()
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "vLLM is installed only if Python can import it, but RealQuant also "
                "needs vLLM's internal GPTQ-Marlin Linear API. "
                f"Probe it with: python vlaconvert_tools/probe_vllm_marlin.py. Details: {exc}"
            ) from exc

        cfg = {
            "quant_method": "gptq",
            "bits": int(pack.layer_info.get("bits", 4)),
            "group_size": int(pack.layer_info.get("group_size", 128)),
            "desc_act": False,
            "sym": True,
            "lm_head": False,
        }
        self.vllm_marlin_module = module_name
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


class QuantVLAPairedRealFakeLinear(nn.Module):
    """Run RealQuant for policy output and record FakeQuant on the same input.

    This is a diagnostic wrapper for paired DiT activation probes. The returned
    tensor is always the RealQuant result, so LIBERO behavior stays RealQuant.
    FakeQuant is computed only when the active probe asks for a sampled call.
    """

    def __init__(self, pack: ConvertedLayerPack, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.name = pack.prefix
        self.in_features = pack.in_features
        self.out_features = pack.out_features
        self.real = QuantVLARealQuantMarlinLinear(pack, dtype=dtype)
        self.fake = QuantVLAFakeQuantLinear(pack, dtype=dtype)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real_out = self.real(x)
        if not _env_enabled("GR00T_DIT_MLP_PROBE_PAIR"):
            return real_out
        try:
            from .dit_mlp_probe import get_active_dit_mlp_probe

            probe = get_active_dit_mlp_probe()
            if probe is None:
                return real_out
            iter_idx, should_record = probe.reserve_pair_call(self.name)
            if not should_record:
                return real_out
            fake_out = self.fake(x)
            probe.record_pair(self.name, iter_idx, fake_out, real_out)
        except Exception as exc:
            if _env_enabled("GR00T_DIT_MLP_PROBE_STRICT"):
                raise
            print(f"[DiT-MLP-PROBE] paired real/fake record skipped for {self.name}: {type(exc).__name__}: {exc}")
        return real_out


def mode_definitions() -> dict[str, str]:
    return {
        GPTQ_QUANT_MODE_FAKE: "QuantVLA-converted GPTQ-like qweight/scales -> dense dequant -> torch F.linear",
        GPTQ_QUANT_MODE_REAL: "same qweight/scales -> vLLM GPTQ-Marlin Linear; QuantVLA transforms preserved",
    }


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
class QuantVLAConvertedReplacementReport:
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

    def print_summary(self, prefix: str = "[QuantVLA-CONVERTED]") -> None:
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

    def write_json_if_requested(self) -> None:
        path = os.environ.get("QUANTVLA_CONVERTED_REPORT") or os.environ.get("QUANTVLA_REPORT")
        if not path:
            return
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        print(f"[QuantVLA-CONVERTED] Replacement report written to {out}")


def normalize_quantvla_converted_mode(mode: str | None) -> str:
    value = (mode or GPTQ_QUANT_MODE_REAL).strip().lower().replace("-", "_")
    aliases = {
        "real": GPTQ_QUANT_MODE_REAL,
        "real_quant": GPTQ_QUANT_MODE_REAL,
        "marlin": GPTQ_QUANT_MODE_REAL,
        "gptq_marlin": GPTQ_QUANT_MODE_REAL,
        "fake": GPTQ_QUANT_MODE_FAKE,
        "fake_quant": GPTQ_QUANT_MODE_FAKE,
        "torch": GPTQ_QUANT_MODE_FAKE,
        "dequant": GPTQ_QUANT_MODE_FAKE,
        "dequant_torch": GPTQ_QUANT_MODE_FAKE,
    }
    if value not in aliases:
        raise ValueError(f"Unsupported QUANTVLA_CONVERTED_MODE={mode!r}; use real or fake")
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


def _load_matching_pack(index: ConvertedQuantVLACheckpoint, module_name: str) -> tuple[str, ConvertedLayerPack] | None:
    available = set(index.prefixes())
    for prefix in _candidate_checkpoint_prefixes(module_name):
        if prefix in available:
            return prefix, index.load_pack(prefix)
    return None


def _validate_shape(name: str, base: nn.Linear, pack: ConvertedLayerPack) -> None:
    if base.in_features != pack.in_features or base.out_features != pack.out_features:
        raise ValueError(
            f"shape mismatch for {name}: module=({base.in_features}, {base.out_features}) "
            f"checkpoint=({pack.in_features}, {pack.out_features})"
        )


def replace_quantvla_converted_linears(
    model: nn.Module,
    checkpoint: str | os.PathLike[str],
    include_regex: str = DEFAULT_INCLUDE,
    exclude_regex: str = DEFAULT_EXCLUDE,
    mode: str = GPTQ_QUANT_MODE_REAL,
    strict: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> QuantVLAConvertedReplacementReport:
    mode = normalize_quantvla_converted_mode(mode)
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
            _validate_shape(name, module, pack)
            if mode == GPTQ_QUANT_MODE_REAL:
                if _env_enabled("GR00T_DIT_MLP_PROBE_PAIR"):
                    wrapper = QuantVLAPairedRealFakeLinear(pack, dtype=dtype)
                else:
                    wrapper = QuantVLARealQuantMarlinLinear(pack, dtype=dtype)
            else:
                wrapper = QuantVLAFakeQuantLinear(pack, dtype=dtype)
        except Exception as exc:
            unreplaced.append(
                ReplacementIssue(name, f"failed to initialize {mode} wrapper: {type(exc).__name__}: {exc}")
            )
            continue
        parent, child_name = _get_parent_module(model, name)
        setattr(parent, child_name, wrapper)
        replaced += 1

    unmatched = sorted(set(index.prefixes()) - matched_prefixes)
    report = QuantVLAConvertedReplacementReport(
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
        raise RuntimeError("QuantVLA converted replacement was not exact; see layer report above")
    return report


def enable_quantvla_converted_if_configured(model: nn.Module) -> QuantVLAConvertedReplacementReport | None:
    checkpoint = os.environ.get("QUANTVLA_CONVERTED_CHECKPOINT")
    if not checkpoint:
        return None
    mode = normalize_quantvla_converted_mode(os.environ.get("QUANTVLA_CONVERTED_MODE", os.environ.get("QUANTVLA_MODE")))
    include = os.environ.get("QUANTVLA_CONVERTED_INCLUDE", os.environ.get("QUANTVLA_INCLUDE", DEFAULT_INCLUDE))
    exclude = os.environ.get("QUANTVLA_CONVERTED_EXCLUDE", os.environ.get("QUANTVLA_EXCLUDE", DEFAULT_EXCLUDE))
    strict = os.environ.get("QUANTVLA_CONVERTED_STRICT", os.environ.get("QUANTVLA_STRICT", "1")) not in (
        "0",
        "false",
        "False",
    )
    dtype = dtype_from_name(os.environ.get("QUANTVLA_CONVERTED_DTYPE", os.environ.get("QUANTVLA_DTYPE", "bfloat16")))

    print("[QuantVLA-CONVERTED] mode:", mode)
    print("[QuantVLA-CONVERTED] checkpoint:", checkpoint)
    print("[QuantVLA-CONVERTED] include:", include)
    print("[QuantVLA-CONVERTED] exclude:", exclude)
    report = replace_quantvla_converted_linears(
        model,
        checkpoint,
        include_regex=include,
        exclude_regex=exclude,
        mode=mode,
        strict=strict,
        dtype=dtype,
    )
    report.print_summary()
    report.write_json_if_requested()
    return report
