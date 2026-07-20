"""Experimental GPTQ-Marlin Linear replacement for the GR00T language model.

This module is intentionally opt-in. Set GR00T_GPTQ_MARLIN_CHECKPOINT to a
GPTQModel/vLLM-compatible checkpoint directory and GR00T will replace matching
language-model nn.Linear modules with GPTQMarlinLinear wrappers during policy
loading.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from torch import nn


DEFAULT_INCLUDE = (
    r".*backbone\.eagle_model\.language_model\..*\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)
DEFAULT_EXCLUDE = r"(?:^|\.)(vision|radio|embed|lm_head|action_head)(?:\.|$)"
GPTQ_QUANT_MODE_REAL = "real"
GPTQ_QUANT_MODE_FAKE = "fake"


def _load_vllm_gptq_marlin_symbols() -> tuple[type, type, str]:
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
class ReplacementIssue:
    layer: str
    reason: str


@dataclass
class GPTQMarlinReplacementReport:
    target_linear_layers: int
    successfully_replaced: int
    unmatched_checkpoint_keys: list[str] = field(default_factory=list)
    unreplaced_target_layers: list[ReplacementIssue] = field(default_factory=list)
    fallback_fp16_layers: list[ReplacementIssue] = field(default_factory=list)
    dry_run: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        data["unmatched_checkpoint_keys_count"] = len(self.unmatched_checkpoint_keys)
        data["unreplaced_target_layers_count"] = len(self.unreplaced_target_layers)
        data["fallback_fp16_layers_count"] = len(self.fallback_fp16_layers)
        return data

    def print_summary(self, prefix: str = "[GR00T-GPTQ-MARLIN]") -> None:
        if self.dry_run:
            print(f"{prefix} DRY RUN: Successfully replaced means checkpoint-compatible target layers.")
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
        path = os.environ.get("GR00T_GPTQ_REPORT") or os.environ.get("GR00T_GPTQ_MARLIN_REPORT")
        if not path:
            return
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        print(f"[GR00T-GPTQ-MARLIN] Replacement report written to {out}")


@dataclass
class GPTQMarlinLayerPack:
    checkpoint_prefix: str
    qweight: torch.Tensor
    scales: torch.Tensor
    qzeros: Optional[torch.Tensor] = None
    g_idx: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None


class GPTQCheckpointIndex:
    """Small safetensors index for GPTQ checkpoints."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"GPTQ checkpoint not found: {self.root}")

        qcfg_path = self.root / "quantize_config.json"
        if qcfg_path.exists():
            self.quant_config = json.loads(qcfg_path.read_text(encoding="utf-8"))
        else:
            self.quant_config = {}

        self._tensor_to_file: Dict[str, Path] = {}
        self._scan_safetensors()

    def _scan_safetensors(self) -> None:
        files = sorted(self.root.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No .safetensors files found in {self.root}")

        try:
            from safetensors import safe_open
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("safetensors is required to load GPTQ checkpoints") from exc

        for file in files:
            with safe_open(file, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    self._tensor_to_file[key] = file

    @property
    def tensor_keys(self) -> Iterable[str]:
        return self._tensor_to_file.keys()

    def qweight_prefixes(self) -> set[str]:
        return {key[: -len(".qweight")] for key in self._tensor_to_file if key.endswith(".qweight")}

    def load_tensor(self, key: str) -> torch.Tensor:
        from safetensors import safe_open

        file = self._tensor_to_file[key]
        with safe_open(file, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    def load_pack(self, module_name: str) -> Optional[GPTQMarlinLayerPack]:
        for prefix in _candidate_checkpoint_prefixes(module_name):
            qweight_key = f"{prefix}.qweight"
            scales_key = f"{prefix}.scales"
            if qweight_key not in self._tensor_to_file or scales_key not in self._tensor_to_file:
                continue

            qzeros_key = f"{prefix}.qzeros"
            g_idx_key = f"{prefix}.g_idx"
            bias_key = f"{prefix}.bias"
            return GPTQMarlinLayerPack(
                checkpoint_prefix=prefix,
                qweight=self.load_tensor(qweight_key),
                scales=self.load_tensor(scales_key),
                qzeros=self.load_tensor(qzeros_key) if qzeros_key in self._tensor_to_file else None,
                g_idx=self.load_tensor(g_idx_key) if g_idx_key in self._tensor_to_file else None,
                bias=self.load_tensor(bias_key) if bias_key in self._tensor_to_file else None,
            )
        return None


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


class GPTQMarlinLinear(nn.Module):
    """nn.Linear-compatible wrapper backed by vLLM GPTQ-Marlin kernels."""

    def __init__(self, base: nn.Linear, pack: GPTQMarlinLayerPack, quant_config: dict, name: str):
        super().__init__()
        self.name = name
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.input_dtype = base.weight.dtype
        self._processed = False
        self._allow_fp_fallback = os.environ.get("GR00T_GPTQ_MARLIN_ALLOW_FP_FALLBACK", "0") not in (
            "0",
            "false",
            "False",
        )

        self._fp_fallback_weight = None
        if self._allow_fp_fallback:
            self.register_buffer("_fp_fallback_weight", base.weight.detach().clone(), persistent=False)

        bias_tensor = pack.bias if pack.bias is not None else base.bias
        if bias_tensor is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias_tensor.detach().clone().to(dtype=base.weight.dtype), requires_grad=False)

        self._init_vllm_method(quant_config)
        self._copy_packed_tensors(pack)

    @property
    def is_fp16_fallback(self) -> bool:
        return self._allow_fp_fallback and self._fp_fallback_weight is not None

    def _init_vllm_method(self, quant_config: dict) -> None:
        try:
            GPTQMarlinConfig, GPTQMarlinLinearMethod, module_name = _load_vllm_gptq_marlin_symbols()
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "vLLM is installed only if Python can import it, but RealQuant also "
                "needs vLLM's internal GPTQ-Marlin Linear API. "
                f"Probe it with: python vlaconvert_tools/probe_vllm_marlin.py. Details: {exc}"
            ) from exc

        cfg = dict(quant_config)
        cfg.setdefault("quant_method", "gptq")
        cfg.setdefault("bits", 4)
        cfg.setdefault("group_size", 128)
        cfg.setdefault("desc_act", False)
        cfg.setdefault("sym", True)
        cfg.setdefault("lm_head", False)

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

    def _copy_parameter(self, name: str, tensor: Optional[torch.Tensor], required: bool = False) -> None:
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

    def _copy_packed_tensors(self, pack: GPTQMarlinLayerPack) -> None:
        self._copy_parameter("qweight", pack.qweight, required=True)
        self._copy_parameter("scales", pack.scales, required=True)
        self._copy_parameter("qzeros", pack.qzeros, required=False)
        self._copy_parameter("g_idx", pack.g_idx, required=False)

    def _process_once(self, x: torch.Tensor) -> None:
        if self._processed:
            return
        self.quant_method.process_weights_after_loading(self)
        self._processed = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_fp16_fallback:
            return torch.nn.functional.linear(x, self._fp_fallback_weight.to(x.dtype), self.bias)

        self._process_once(x)
        original_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        y_2d = self.quant_method.apply(self, x_2d, self.bias)
        return y_2d.reshape(*original_shape, self.out_features)


class GPTQDequantLinear(nn.Module):
    """Reference Linear using the same GPTQ tensors, dequantized to torch F.linear."""

    def __init__(self, base: nn.Linear, pack: GPTQMarlinLayerPack, quant_config: dict, name: str, dtype: torch.dtype):
        super().__init__()
        self.name = name
        self.in_features = base.in_features
        self.out_features = base.out_features
        weight = dequantize_gptq_weight(pack, quant_config, base.in_features, base.out_features)
        self.register_buffer("weight", weight.to(device=base.weight.device, dtype=dtype), persistent=False)
        bias_tensor = pack.bias if pack.bias is not None else base.bias
        if bias_tensor is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias_tensor.detach().clone().to(device=base.weight.device, dtype=dtype), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight.to(dtype=x.dtype), self.bias)


class GPTQFakeQuantLinear(GPTQDequantLinear):
    """Fake/reference quant path: GPTQ weight, no Marlin kernel, torch F.linear."""


GPTQRealQuantMarlinLinear = GPTQMarlinLinear

def _unpack_int4_rows(packed: torch.Tensor, rows: int) -> torch.Tensor:
    values_per_word = 8
    packed_i64 = packed.to(torch.int64)
    shifts = torch.arange(values_per_word, device=packed.device, dtype=torch.int64) * 4
    unpacked = ((packed_i64.unsqueeze(-1) >> shifts) & 0xF).reshape(-1, packed.shape[1])
    return unpacked[:rows, :]


def _unpack_int4_cols(packed: torch.Tensor, cols: int) -> torch.Tensor:
    values_per_word = 8
    packed_i64 = packed.to(torch.int64)
    shifts = torch.arange(values_per_word, device=packed.device, dtype=torch.int64) * 4
    unpacked = ((packed_i64.unsqueeze(-1) >> shifts) & 0xF).reshape(packed.shape[0], -1)
    return unpacked[:, :cols]


def _normalized_scales(scales: torch.Tensor, groups: int, out_features: int) -> torch.Tensor:
    scales = scales.to(torch.float32)
    if scales.shape == (groups, out_features):
        return scales
    if scales.shape == (out_features, groups):
        return scales.t().contiguous()
    raise ValueError(f"Unsupported GPTQ scales shape {tuple(scales.shape)} for groups={groups}, out={out_features}")


def dequantize_gptq_weight(
    pack: GPTQMarlinLayerPack,
    quant_config: dict,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    """Best-effort GPTQ int4 dequantization to a [out, in] torch weight."""

    bits = int(quant_config.get("bits", quant_config.get("w_bit", 4)))
    if bits != 4:
        raise ValueError(f"Only 4-bit GPTQ dequantization is supported, got bits={bits}")
    if pack.qweight.shape[1] != out_features:
        raise ValueError(
            f"Unsupported qweight shape {tuple(pack.qweight.shape)} for out_features={out_features}; "
            "expected [in_features / 8, out_features]"
        )

    group_size = int(quant_config.get("group_size", 128))
    if group_size <= 0:
        group_size = in_features
    groups = (in_features + group_size - 1) // group_size

    qweight = _unpack_int4_rows(pack.qweight.cpu(), in_features).to(torch.float32)
    scales = _normalized_scales(pack.scales.cpu(), groups, out_features)

    if pack.qzeros is not None:
        qzeros = _unpack_int4_cols(pack.qzeros.cpu(), out_features).to(torch.float32) + 1.0
    else:
        zero_value = float(2 ** (bits - 1)) if bool(quant_config.get("sym", True)) else 0.0
        qzeros = torch.full((groups, out_features), zero_value, dtype=torch.float32)

    if pack.g_idx is not None and pack.g_idx.numel() >= in_features:
        group_ids = pack.g_idx[:in_features].cpu().to(torch.long).clamp_(0, groups - 1)
    else:
        group_ids = (torch.arange(in_features, dtype=torch.long) // group_size).clamp_(0, groups - 1)

    weight_in_out = (qweight - qzeros[group_ids]) * scales[group_ids]
    return weight_in_out.t().contiguous()


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


def _make_unmatched_prefixes(index: GPTQCheckpointIndex, matched_prefixes: set[str]) -> list[str]:
    return sorted(index.qweight_prefixes() - matched_prefixes)


def replace_gptq_marlin_linears(
    model: nn.Module,
    checkpoint: str | os.PathLike[str],
    include_regex: str = DEFAULT_INCLUDE,
    exclude_regex: str = DEFAULT_EXCLUDE,
    dry_run: bool = False,
    strict: bool = True,
) -> GPTQMarlinReplacementReport:
    index = GPTQCheckpointIndex(checkpoint)
    targets = _iter_target_linears(model, include_regex, exclude_regex)
    matched_prefixes: set[str] = set()
    replaced = 0
    unreplaced: list[ReplacementIssue] = []
    fallback: list[ReplacementIssue] = []
    allow_layer_fallback = os.environ.get("GR00T_GPTQ_MARLIN_ALLOW_LAYER_FALLBACK", "0") not in (
        "0",
        "false",
        "False",
    )
    allow_fp_fallback = os.environ.get("GR00T_GPTQ_MARLIN_ALLOW_FP_FALLBACK", "0") not in (
        "0",
        "false",
        "False",
    )

    for name, module in targets:
        pack = index.load_pack(name)
        if pack is None:
            unreplaced.append(ReplacementIssue(name, "missing qweight/scales tensors in GPTQ checkpoint"))
            continue
        matched_prefixes.add(pack.checkpoint_prefix)
        if dry_run:
            replaced += 1
            continue
        try:
            wrapper = GPTQMarlinLinear(module, pack, index.quant_config, name)
        except Exception as exc:
            reason = f"failed to initialize GPTQMarlinLinear: {type(exc).__name__}: {exc}"
            if allow_layer_fallback:
                fallback.append(ReplacementIssue(name, reason))
                continue
            unreplaced.append(ReplacementIssue(name, reason))
            continue
        parent, child_name = _get_parent_module(model, name)
        setattr(parent, child_name, wrapper)
        replaced += 1
        if allow_fp_fallback:
            fallback.append(ReplacementIssue(name, "GR00T_GPTQ_MARLIN_ALLOW_FP_FALLBACK=1"))

    report = GPTQMarlinReplacementReport(
        target_linear_layers=len(targets),
        successfully_replaced=replaced,
        unmatched_checkpoint_keys=_make_unmatched_prefixes(index, matched_prefixes),
        unreplaced_target_layers=unreplaced,
        fallback_fp16_layers=fallback,
        dry_run=dry_run,
    )
    model._gptq_marlin_replacement_report = report  # type: ignore[attr-defined]

    if strict and (report.unreplaced_target_layers or report.fallback_fp16_layers):
        report.print_summary()
        raise RuntimeError("GPTQ-Marlin replacement was not exact; see layer report above")
    return report


def replace_gptq_dequantized_linears(
    model: nn.Module,
    checkpoint: str | os.PathLike[str],
    include_regex: str = DEFAULT_INCLUDE,
    exclude_regex: str = DEFAULT_EXCLUDE,
    strict: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> GPTQMarlinReplacementReport:
    index = GPTQCheckpointIndex(checkpoint)
    targets = _iter_target_linears(model, include_regex, exclude_regex)
    matched_prefixes: set[str] = set()
    replaced = 0
    unreplaced: list[ReplacementIssue] = []

    for name, module in targets:
        pack = index.load_pack(name)
        if pack is None:
            unreplaced.append(ReplacementIssue(name, "missing qweight/scales tensors in GPTQ checkpoint"))
            continue
        matched_prefixes.add(pack.checkpoint_prefix)
        try:
            wrapper = GPTQFakeQuantLinear(module, pack, index.quant_config, name, dtype=dtype)
        except Exception as exc:
            unreplaced.append(ReplacementIssue(name, f"failed to dequantize GPTQ tensor: {type(exc).__name__}: {exc}"))
            continue
        parent, child_name = _get_parent_module(model, name)
        setattr(parent, child_name, wrapper)
        replaced += 1

    report = GPTQMarlinReplacementReport(
        target_linear_layers=len(targets),
        successfully_replaced=replaced,
        unmatched_checkpoint_keys=_make_unmatched_prefixes(index, matched_prefixes),
        unreplaced_target_layers=unreplaced,
        fallback_fp16_layers=[],
        dry_run=False,
    )
    model._gptq_dequant_replacement_report = report  # type: ignore[attr-defined]
    if strict and report.unreplaced_target_layers:
        report.print_summary(prefix="[GR00T-GPTQ-DEQUANT]")
        raise RuntimeError("GPTQ dequant replacement was not exact; see layer report above")
    return report



def replace_gptq_real_quant_linears(
    model: nn.Module,
    checkpoint: str | os.PathLike[str],
    include_regex: str = DEFAULT_INCLUDE,
    exclude_regex: str = DEFAULT_EXCLUDE,
    dry_run: bool = False,
    strict: bool = True,
) -> GPTQMarlinReplacementReport:
    """RealQuant: GPTQ packed tensors executed through vLLM GPTQ-Marlin Linear."""
    return replace_gptq_marlin_linears(
        model,
        checkpoint,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
        dry_run=dry_run,
        strict=strict,
    )


def replace_gptq_fake_quant_linears(
    model: nn.Module,
    checkpoint: str | os.PathLike[str],
    include_regex: str = DEFAULT_INCLUDE,
    exclude_regex: str = DEFAULT_EXCLUDE,
    strict: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> GPTQMarlinReplacementReport:
    """FakeQuant/reference: same GPTQ tensors dequantized, then torch F.linear."""
    return replace_gptq_dequantized_linears(
        model,
        checkpoint,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
        strict=strict,
        dtype=dtype,
    )


def normalize_gptq_quant_mode(mode: str | None) -> str:
    value = (mode or GPTQ_QUANT_MODE_REAL).strip().lower().replace("-", "_")
    aliases = {
        "real": GPTQ_QUANT_MODE_REAL,
        "real_quant": GPTQ_QUANT_MODE_REAL,
        "marlin": GPTQ_QUANT_MODE_REAL,
        "gptq_marlin": GPTQ_QUANT_MODE_REAL,
        "fake": GPTQ_QUANT_MODE_FAKE,
        "fake_quant": GPTQ_QUANT_MODE_FAKE,
        "torch": GPTQ_QUANT_MODE_FAKE,
        "gptq_torch": GPTQ_QUANT_MODE_FAKE,
        "dequant": GPTQ_QUANT_MODE_FAKE,
        "dequant_torch": GPTQ_QUANT_MODE_FAKE,
    }
    if value not in aliases:
        raise ValueError(f"Unsupported GR00T_GPTQ_QUANT_MODE={mode!r}; use real or fake")
    return aliases[value]


def _dtype_from_name(name: str) -> torch.dtype:
    value = name.strip().lower()
    if value in ("bf16", "bfloat16"):
        return torch.bfloat16
    if value in ("fp16", "float16"):
        return torch.float16
    if value in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported GPTQ fake dtype: {name}")


def enable_gptq_quant_if_configured(model: nn.Module) -> None:
    ckpt = os.environ.get("GR00T_GPTQ_CHECKPOINT") or os.environ.get("GR00T_GPTQ_MARLIN_CHECKPOINT")
    if not ckpt:
        return

    mode = normalize_gptq_quant_mode(os.environ.get("GR00T_GPTQ_QUANT_MODE"))
    include = os.environ.get("GR00T_GPTQ_INCLUDE", os.environ.get("GR00T_GPTQ_MARLIN_INCLUDE", DEFAULT_INCLUDE))
    exclude = os.environ.get("GR00T_GPTQ_EXCLUDE", os.environ.get("GR00T_GPTQ_MARLIN_EXCLUDE", DEFAULT_EXCLUDE))
    dry_run = os.environ.get("GR00T_GPTQ_DRYRUN", os.environ.get("GR00T_GPTQ_MARLIN_DRYRUN", "0")) not in (
        "0",
        "false",
        "False",
    )
    strict = os.environ.get("GR00T_GPTQ_STRICT", os.environ.get("GR00T_GPTQ_MARLIN_STRICT", "1")) not in (
        "0",
        "false",
        "False",
    )

    print("[GR00T-GPTQ] mode:", mode)
    print("[GR00T-GPTQ] checkpoint:", ckpt)
    print("[GR00T-GPTQ] include:", include)
    print("[GR00T-GPTQ] exclude:", exclude)

    if mode == GPTQ_QUANT_MODE_REAL:
        report = replace_gptq_real_quant_linears(
            model,
            ckpt,
            include_regex=include,
            exclude_regex=exclude,
            dry_run=dry_run,
            strict=strict,
        )
        prefix = "[GR00T-GPTQ-REAL]"
    else:
        dtype_name = os.environ.get("GR00T_GPTQ_FAKE_DTYPE", os.environ.get("GR00T_GPTQ_DTYPE", "bfloat16"))
        report = replace_gptq_fake_quant_linears(
            model,
            ckpt,
            include_regex=include,
            exclude_regex=exclude,
            strict=strict,
            dtype=_dtype_from_name(dtype_name),
        )
        prefix = "[GR00T-GPTQ-FAKE]"

    report.print_summary(prefix=prefix)
    report.write_json_if_requested()

def enable_gptq_marlin_if_configured(model: nn.Module) -> None:
    ckpt = os.environ.get("GR00T_GPTQ_MARLIN_CHECKPOINT")
    if not ckpt:
        return

    include = os.environ.get("GR00T_GPTQ_MARLIN_INCLUDE", DEFAULT_INCLUDE)
    exclude = os.environ.get("GR00T_GPTQ_MARLIN_EXCLUDE", DEFAULT_EXCLUDE)
    dry_run = os.environ.get("GR00T_GPTQ_MARLIN_DRYRUN", "0") not in ("0", "false", "False")
    strict = os.environ.get("GR00T_GPTQ_MARLIN_STRICT", "1") not in ("0", "false", "False")

    print("[GR00T-GPTQ-MARLIN] checkpoint:", ckpt)
    print("[GR00T-GPTQ-MARLIN] include:", include)
    print("[GR00T-GPTQ-MARLIN] exclude:", exclude)
    report = replace_gptq_marlin_linears(
        model,
        ckpt,
        include_regex=include,
        exclude_regex=exclude,
        dry_run=dry_run,
        strict=strict,
    )
    report.print_summary()
    report.write_json_if_requested()
