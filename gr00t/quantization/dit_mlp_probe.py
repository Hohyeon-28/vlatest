"""Activation probes for GR00T DiT MLP up-projection outputs."""

from __future__ import annotations

import atexit
import csv
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn


DEFAULT_LAYER_REGEX = r"action_head\.model\.transformer_blocks\.(\d+)\.ff\.net\.0\.proj$"


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False", "")


def _parse_int_set(value: str | None) -> set[int] | None:
    if not value:
        return None
    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _selected_iters(num_steps: int) -> set[int]:
    raw = os.environ.get("GR00T_DIT_MLP_PROBE_ITERS")
    if raw:
        aliases = {
            "first": 0,
            "start": 0,
            "mid": max(0, num_steps // 2),
            "middle": max(0, num_steps // 2),
            "last": max(0, num_steps - 1),
            "end": max(0, num_steps - 1),
        }
        selected: set[int] = set()
        for part in raw.split(","):
            key = part.strip().lower()
            if not key:
                continue
            selected.add(aliases.get(key, int(key) if key.isdigit() else 0))
        return {i for i in selected if 0 <= i < num_steps}
    return {0, max(0, num_steps // 2), max(0, num_steps - 1)}


def _iter_label(iter_idx: int, num_steps: int) -> str:
    if iter_idx == 0:
        return "first"
    if iter_idx == max(0, num_steps - 1):
        return "last"
    if iter_idx == max(0, num_steps // 2):
        return "mid"
    return f"iter_{iter_idx}"


@dataclass
class _BinAccum:
    calls: int = 0
    mean_sum: float = 0.0
    abs_mean_sum: float = 0.0
    std_sum: float = 0.0
    rms_sum: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")

    def update(self, mean: float, abs_mean: float, std: float, rms: float, min_value: float, max_value: float) -> None:
        self.calls += 1
        self.mean_sum += mean
        self.abs_mean_sum += abs_mean
        self.std_sum += std
        self.rms_sum += rms
        self.min_value = min(self.min_value, min_value)
        self.max_value = max(self.max_value, max_value)


@dataclass
class _PairBinAccum:
    calls: int = 0
    fake_mean_sum: float = 0.0
    fake_abs_mean_sum: float = 0.0
    fake_rms_sum: float = 0.0
    real_mean_sum: float = 0.0
    real_abs_mean_sum: float = 0.0
    real_rms_sum: float = 0.0
    diff_mean_sum: float = 0.0
    diff_abs_mean_sum: float = 0.0
    diff_rms_sum: float = 0.0
    mse_sum: float = 0.0
    max_abs_diff: float = 0.0

    def update(
        self,
        fake_mean: float,
        fake_abs_mean: float,
        fake_rms: float,
        real_mean: float,
        real_abs_mean: float,
        real_rms: float,
        diff_mean: float,
        diff_abs_mean: float,
        diff_rms: float,
        mse: float,
        max_abs_diff: float,
    ) -> None:
        self.calls += 1
        self.fake_mean_sum += fake_mean
        self.fake_abs_mean_sum += fake_abs_mean
        self.fake_rms_sum += fake_rms
        self.real_mean_sum += real_mean
        self.real_abs_mean_sum += real_abs_mean
        self.real_rms_sum += real_rms
        self.diff_mean_sum += diff_mean
        self.diff_abs_mean_sum += diff_abs_mean
        self.diff_rms_sum += diff_rms
        self.mse_sum += mse
        self.max_abs_diff = max(self.max_abs_diff, max_abs_diff)


_ACTIVE_DIT_MLP_PROBE: "DiTMLPProbe | None" = None


@dataclass
class DiTMLPProbe:
    num_steps: int
    output_dir: Path
    mode: str
    task: str
    bins: int = 128
    selected_layers: set[int] | None = None
    layer_regex: re.Pattern[str] = field(default_factory=lambda: re.compile(DEFAULT_LAYER_REGEX))

    def __post_init__(self) -> None:
        self.selected_iters = _selected_iters(self.num_steps)
        self.call_counts: dict[str, int] = {}
        self.pair_call_counts: dict[str, int] = {}
        self.layer_indices: dict[str, int] = {}
        self.accum: dict[tuple[str, int, int], _BinAccum] = {}
        self.pair_accum: dict[tuple[str, int, int], _PairBinAccum] = {}
        self.hooks: list[Any] = []
        self.lock = threading.RLock()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / f"dit_mlp_up_probe_{self.mode}_{self.task}.csv"
        self.pair_csv_path = self.output_dir / f"dit_mlp_up_probe_pair_{self.mode}_{self.task}.csv"
        self.meta_path = self.output_dir / f"dit_mlp_up_probe_{self.mode}_{self.task}_meta.txt"
        self.flush_every = int(os.environ.get("GR00T_DIT_MLP_PROBE_FLUSH_EVERY", "2000"))
        self.total_observed_calls = 0
        self.total_pair_calls = 0

    def register(self, model: nn.Module) -> int:
        for name, module in model.named_modules():
            match = self.layer_regex.match(name)
            if not match:
                continue
            layer_idx = int(match.group(1))
            if self.selected_layers is not None and layer_idx not in self.selected_layers:
                continue
            self.layer_indices[name] = layer_idx
            self.call_counts[name] = 0
            self.hooks.append(module.register_forward_hook(self._make_hook(name)))
        self._write_meta()
        atexit.register(self.dump)
        return len(self.hooks)

    def _write_meta(self) -> None:
        text = "\n".join(
            [
                f"mode={self.mode}",
                f"task={self.task}",
                f"num_steps={self.num_steps}",
                f"selected_iters={sorted(self.selected_iters)}",
                f"selected_layers={sorted(self.selected_layers) if self.selected_layers is not None else 'all'}",
                f"bins={self.bins}",
                f"csv={self.csv_path}",
                f"paired_csv={self.pair_csv_path}",
                f"paired_enabled={_env_enabled('GR00T_DIT_MLP_PROBE_PAIR')}",
            ]
        )
        self.meta_path.write_text(text + "\n", encoding="utf-8")

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            if not torch.is_tensor(output):
                return
            call_idx = self.call_counts[name]
            self.call_counts[name] = call_idx + 1
            iter_idx = call_idx % self.num_steps
            if iter_idx not in self.selected_iters:
                return

            with torch.no_grad():
                y = output.detach().float()
                if y.ndim < 2:
                    return
                channels = y.shape[-1]
                flat = y.reshape(-1, channels)
                per_channel_mean = flat.mean(dim=0)
                per_channel_abs = flat.abs().mean(dim=0)
                per_channel_std = flat.std(dim=0, unbiased=False)
                per_channel_rms = flat.pow(2).mean(dim=0).sqrt()
                per_channel_min = flat.min(dim=0).values
                per_channel_max = flat.max(dim=0).values

                bin_count = min(self.bins, channels)
                edges = torch.linspace(0, channels, steps=bin_count + 1, device=flat.device).round().long()
                rows: list[tuple[int, float, float, float, float, float, float]] = []
                for bin_idx in range(bin_count):
                    start = int(edges[bin_idx].item())
                    end = int(edges[bin_idx + 1].item())
                    if end <= start:
                        continue
                    rows.append(
                        (
                            bin_idx,
                            float(per_channel_mean[start:end].mean().item()),
                            float(per_channel_abs[start:end].mean().item()),
                            float(per_channel_std[start:end].mean().item()),
                            float(per_channel_rms[start:end].mean().item()),
                            float(per_channel_min[start:end].min().item()),
                            float(per_channel_max[start:end].max().item()),
                        )
                    )

            with self.lock:
                for bin_idx, mean, abs_mean, std, rms, min_value, max_value in rows:
                    key = (name, iter_idx, bin_idx)
                    self.accum.setdefault(key, _BinAccum()).update(mean, abs_mean, std, rms, min_value, max_value)
                self.total_observed_calls += 1
                if self.flush_every > 0 and self.total_observed_calls % self.flush_every == 0:
                    self.dump()

        return hook

    def reserve_pair_call(self, name: str) -> tuple[int, bool]:
        match = self.layer_regex.match(name)
        if not match:
            return 0, False
        layer_idx = int(match.group(1))
        if self.selected_layers is not None and layer_idx not in self.selected_layers:
            return 0, False
        with self.lock:
            self.layer_indices.setdefault(name, layer_idx)
            call_idx = self.pair_call_counts.get(name, 0)
            self.pair_call_counts[name] = call_idx + 1
            iter_idx = call_idx % self.num_steps
            return iter_idx, iter_idx in self.selected_iters

    def record_pair(self, name: str, iter_idx: int, fake_output: torch.Tensor, real_output: torch.Tensor) -> None:
        if iter_idx not in self.selected_iters:
            return
        with torch.no_grad():
            fake = fake_output.detach().float()
            real = real_output.detach().float()
            if fake.ndim < 2 or real.ndim < 2 or fake.shape != real.shape:
                return
            channels = fake.shape[-1]
            fake_flat = fake.reshape(-1, channels)
            real_flat = real.reshape(-1, channels)
            diff_flat = real_flat - fake_flat

            fake_mean = fake_flat.mean(dim=0)
            fake_abs = fake_flat.abs().mean(dim=0)
            fake_rms = fake_flat.pow(2).mean(dim=0).sqrt()
            real_mean = real_flat.mean(dim=0)
            real_abs = real_flat.abs().mean(dim=0)
            real_rms = real_flat.pow(2).mean(dim=0).sqrt()
            diff_mean = diff_flat.mean(dim=0)
            diff_abs = diff_flat.abs().mean(dim=0)
            diff_rms = diff_flat.pow(2).mean(dim=0).sqrt()
            mse = diff_flat.pow(2).mean(dim=0)
            max_abs_diff = diff_flat.abs().max(dim=0).values

            bin_count = min(self.bins, channels)
            edges = torch.linspace(0, channels, steps=bin_count + 1, device=fake_flat.device).round().long()
            rows: list[tuple[int, float, float, float, float, float, float, float, float, float, float, float]] = []
            for bin_idx in range(bin_count):
                start = int(edges[bin_idx].item())
                end = int(edges[bin_idx + 1].item())
                if end <= start:
                    continue
                rows.append(
                    (
                        bin_idx,
                        float(fake_mean[start:end].mean().item()),
                        float(fake_abs[start:end].mean().item()),
                        float(fake_rms[start:end].mean().item()),
                        float(real_mean[start:end].mean().item()),
                        float(real_abs[start:end].mean().item()),
                        float(real_rms[start:end].mean().item()),
                        float(diff_mean[start:end].mean().item()),
                        float(diff_abs[start:end].mean().item()),
                        float(diff_rms[start:end].mean().item()),
                        float(mse[start:end].mean().item()),
                        float(max_abs_diff[start:end].max().item()),
                    )
                )

        with self.lock:
            for row in rows:
                bin_idx = row[0]
                key = (name, iter_idx, bin_idx)
                self.pair_accum.setdefault(key, _PairBinAccum()).update(*row[1:])
            self.total_pair_calls += 1
            if self.flush_every > 0 and self.total_pair_calls % self.flush_every == 0:
                self.dump()

    def dump(self) -> None:
        with self.lock:
            fieldnames = [
                "mode",
                "task",
                "layer",
                "layer_index",
                "denoise_iter",
                "iter_label",
                "channel_bin",
                "calls",
                "mean",
                "abs_mean",
                "std",
                "rms",
                "min",
                "max",
            ]
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for (layer, iter_idx, bin_idx), acc in sorted(
                    self.accum.items(), key=lambda item: (self.layer_indices[item[0][0]], item[0][1], item[0][2])
                ):
                    calls = max(1, acc.calls)
                    writer.writerow(
                        {
                            "mode": self.mode,
                            "task": self.task,
                            "layer": layer,
                            "layer_index": self.layer_indices[layer],
                            "denoise_iter": iter_idx,
                            "iter_label": _iter_label(iter_idx, self.num_steps),
                            "channel_bin": bin_idx,
                            "calls": acc.calls,
                            "mean": acc.mean_sum / calls,
                            "abs_mean": acc.abs_mean_sum / calls,
                            "std": acc.std_sum / calls,
                            "rms": acc.rms_sum / calls,
                            "min": acc.min_value,
                            "max": acc.max_value,
                        }
                    )
            if self.pair_accum:
                pair_fieldnames = [
                    "task",
                    "active_mode",
                    "reference_mode",
                    "layer",
                    "layer_index",
                    "denoise_iter",
                    "iter_label",
                    "channel_bin",
                    "calls",
                    "fake_mean",
                    "fake_abs_mean",
                    "fake_rms",
                    "real_mean",
                    "real_abs_mean",
                    "real_rms",
                    "diff_mean",
                    "diff_abs_mean",
                    "diff_rms",
                    "mse",
                    "max_abs_diff",
                    "real_minus_fake_abs_mean_pct",
                ]
                with self.pair_csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=pair_fieldnames)
                    writer.writeheader()
                    for (layer, iter_idx, bin_idx), acc in sorted(
                        self.pair_accum.items(),
                        key=lambda item: (self.layer_indices[item[0][0]], item[0][1], item[0][2]),
                    ):
                        calls = max(1, acc.calls)
                        fake_abs_mean = acc.fake_abs_mean_sum / calls
                        real_abs_mean = acc.real_abs_mean_sum / calls
                        denom = abs(fake_abs_mean) if abs(fake_abs_mean) > 1e-9 else 1e-9
                        writer.writerow(
                            {
                                "task": self.task,
                                "active_mode": "real",
                                "reference_mode": "fake",
                                "layer": layer,
                                "layer_index": self.layer_indices[layer],
                                "denoise_iter": iter_idx,
                                "iter_label": _iter_label(iter_idx, self.num_steps),
                                "channel_bin": bin_idx,
                                "calls": acc.calls,
                                "fake_mean": acc.fake_mean_sum / calls,
                                "fake_abs_mean": fake_abs_mean,
                                "fake_rms": acc.fake_rms_sum / calls,
                                "real_mean": acc.real_mean_sum / calls,
                                "real_abs_mean": real_abs_mean,
                                "real_rms": acc.real_rms_sum / calls,
                                "diff_mean": acc.diff_mean_sum / calls,
                                "diff_abs_mean": acc.diff_abs_mean_sum / calls,
                                "diff_rms": acc.diff_rms_sum / calls,
                                "mse": acc.mse_sum / calls,
                                "max_abs_diff": acc.max_abs_diff,
                                "real_minus_fake_abs_mean_pct": (real_abs_mean - fake_abs_mean) / denom * 100.0,
                            }
                        )


def get_active_dit_mlp_probe() -> DiTMLPProbe | None:
    return _ACTIVE_DIT_MLP_PROBE


def enable_dit_mlp_probe_if_configured(model: nn.Module) -> DiTMLPProbe | None:
    global _ACTIVE_DIT_MLP_PROBE
    if not _env_enabled("GR00T_DIT_MLP_PROBE"):
        _ACTIVE_DIT_MLP_PROBE = None
        return None
    action_head = getattr(model, "action_head", None)
    num_steps = int(
        os.environ.get(
            "GR00T_DIT_MLP_PROBE_DENOISING_STEPS",
            os.environ.get("GR00T_DENOISING_STEPS", getattr(action_head, "num_inference_timesteps", 8)),
        )
    )
    mode = os.environ.get("QUANTVLA_CONVERTED_MODE", os.environ.get("GR00T_DIT_MLP_PROBE_MODE", "unknown"))
    task = os.environ.get("QUANTVLA_CONVERTED_TASK", os.environ.get("GR00T_DIT_MLP_PROBE_TASK", "unknown"))
    output_dir = Path(os.environ.get("GR00T_DIT_MLP_PROBE_DIR", "/tmp/logs/dit_mlp_probe"))
    bins = int(os.environ.get("GR00T_DIT_MLP_PROBE_BINS", "128"))
    layers = _parse_int_set(os.environ.get("GR00T_DIT_MLP_PROBE_LAYERS"))
    probe = DiTMLPProbe(
        num_steps=num_steps,
        output_dir=output_dir,
        mode=mode,
        task=task,
        bins=bins,
        selected_layers=layers,
    )
    count = probe.register(model)
    _ACTIVE_DIT_MLP_PROBE = probe
    model._dit_mlp_probe = probe  # type: ignore[attr-defined]
    print(
        "[DiT-MLP-PROBE] enabled: "
        f"layers={count}, iters={sorted(probe.selected_iters)}, bins={bins}, output={probe.csv_path}"
    )
    if _env_enabled("GR00T_DIT_MLP_PROBE_PAIR"):
        print(f"[DiT-MLP-PROBE] paired real/fake output enabled: output={probe.pair_csv_path}")
    return probe
