"""Probe GPTQ FakeQuant vs RealQuant on the same GPTQ checkpoint.

This is not a PPL or task-success evaluator. It is a low-level definition and
sanity probe for the two quant execution modes used in this repository:

- fake: same GPTQ qweight/scales, dequantized to a dense torch weight, then
  torch.nn.functional.linear.
- real: same GPTQ qweight/scales, executed by the vLLM GPTQ-Marlin Linear path.

The script runs selected Linear layers with synthetic activations and reports
latency, memory, and numerical agreement between fake and real outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

from gr00t.quantization.gptq_marlin_linear import (
    GPTQCheckpointIndex,
    GPTQFakeQuantLinear,
    GPTQMarlinLinear,
)

DEFAULT_LAYER_REGEX = r"model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"


def dtype_from_name(name: str) -> torch.dtype:
    value = name.strip().lower()
    if value in ("bf16", "bfloat16"):
        return torch.bfloat16
    if value in ("fp16", "float16"):
        return torch.float16
    if value in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def infer_shape(pack) -> tuple[int, int]:
    # GPTQ qweight is packed along the input dimension: [in_features / 8, out_features].
    in_features = int(pack.qweight.shape[0]) * 8
    out_features = int(pack.qweight.shape[1])
    return in_features, out_features


def make_base_linear(in_features: int, out_features: int, has_bias: bool, dtype: torch.dtype) -> nn.Linear:
    base = nn.Linear(in_features, out_features, bias=has_bias, device="cpu", dtype=dtype)
    base.requires_grad_(False)
    return base


def cuda_peak_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn: Callable[[], torch.Tensor], warmup: int, rep: int) -> tuple[str, float, float | None]:
    try:
        import triton

        for _ in range(max(1, warmup)):
            fn()
        synchronize()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        ms = triton.testing.do_bench(fn, warmup=0, rep=rep)
        if isinstance(ms, (tuple, list)):
            ms = float(ms[0])
        synchronize()
        return "triton.testing.do_bench", float(ms), cuda_peak_mb()
    except Exception:
        for _ in range(max(1, warmup)):
            fn()
        synchronize()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(max(1, rep)):
                fn()
            end.record()
            synchronize()
            return "torch.cuda.Event", start.elapsed_time(end) / max(1, rep), cuda_peak_mb()

        import time

        start_time = time.perf_counter()
        for _ in range(max(1, rep)):
            fn()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0 / max(1, rep)
        return "time.perf_counter", elapsed_ms, None


def compare_outputs(fake_y: torch.Tensor, real_y: torch.Tensor) -> dict:
    fake = fake_y.detach().float().reshape(-1)
    real = real_y.detach().float().reshape(-1)
    diff = real - fake
    denom = fake.abs().clamp_min(1e-6)
    rel = diff.abs() / denom
    return {
        "mse": diff.pow(2).mean().item(),
        "mae": diff.abs().mean().item(),
        "max_abs": diff.abs().max().item(),
        "mean_relative_error": rel.mean().item(),
        "max_relative_error": rel.max().item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(fake, real, dim=0).item(),
    }


def quant_config_summary(index: GPTQCheckpointIndex) -> dict:
    cfg = index.quant_config
    return {
        "bits": cfg.get("bits", cfg.get("w_bit", 4)),
        "group_size": cfg.get("group_size", 128),
        "desc_act": cfg.get("desc_act", None),
        "sym": cfg.get("sym", None),
        "quant_method": cfg.get("quant_method", cfg.get("checkpoint_format", "gptq")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gptq_checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--layer-regex", default=DEFAULT_LAYER_REGEX)
    parser.add_argument("--max-layers", type=int, default=8, help="0 means all matching layers")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    dtype = dtype_from_name(args.dtype)
    index = GPTQCheckpointIndex(args.gptq_checkpoint)
    layer_re = re.compile(args.layer_regex)
    prefixes = sorted(prefix for prefix in index.qweight_prefixes() if layer_re.search(prefix))
    if args.max_layers > 0:
        prefixes = prefixes[: args.max_layers]
    if not prefixes:
        raise SystemExit(f"No GPTQ qweight prefixes matched regex: {args.layer_regex}")

    print("========== GPTQ Quant Mode Probe ==========")
    print(f"checkpoint: {args.gptq_checkpoint}")
    print("definition.fake: GPTQ qweight/scales -> dense dequant -> torch F.linear")
    print("definition.real: GPTQ qweight/scales -> vLLM GPTQ-Marlin Linear")
    print(f"layers: {len(prefixes)}")
    print(f"dtype: {args.dtype}, batch={args.batch_size}, seq_len={args.seq_len}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    layer_results = []
    failures = []

    for prefix in prefixes:
        pack = index.load_pack(prefix)
        if pack is None:
            failures.append({"layer": prefix, "reason": "missing qweight/scales"})
            continue
        in_features, out_features = infer_shape(pack)
        print(f"\n[layer] {prefix} in={in_features} out={out_features}")
        base = make_base_linear(in_features, out_features, pack.bias is not None, dtype=dtype)
        x = torch.randn(args.batch_size, args.seq_len, in_features, device=args.device, dtype=dtype)

        try:
            fake = GPTQFakeQuantLinear(base, pack, index.quant_config, prefix, dtype=dtype).to(args.device)
            real = GPTQMarlinLinear(base, pack, index.quant_config, prefix).to(args.device)
            fake.eval()
            real.eval()

            with torch.inference_mode():
                fake_y = fake(x)
                real_y = real(x)
                synchronize()
                error_metrics = compare_outputs(fake_y, real_y)

                fake_backend, fake_ms, fake_peak_mb = bench(lambda: fake(x), args.warmup, args.rep)
                real_backend, real_ms, real_peak_mb = bench(lambda: real(x), args.warmup, args.rep)

            tokens = args.batch_size * args.seq_len
            result = {
                "layer": prefix,
                "in_features": in_features,
                "out_features": out_features,
                "fake_backend": "torch F.linear after GPTQ dequant",
                "real_backend": "vLLM GPTQ-Marlin Linear",
                "benchmark_backend_fake": fake_backend,
                "benchmark_backend_real": real_backend,
                "fake_latency_ms": fake_ms,
                "real_latency_ms": real_ms,
                "real_speedup_vs_fake": fake_ms / real_ms if real_ms > 0 else math.inf,
                "fake_tokens_per_s": tokens * 1000.0 / fake_ms if fake_ms > 0 else math.inf,
                "real_tokens_per_s": tokens * 1000.0 / real_ms if real_ms > 0 else math.inf,
                "fake_peak_memory_mb": fake_peak_mb,
                "real_peak_memory_mb": real_peak_mb,
                **error_metrics,
            }
            layer_results.append(result)
            print(
                f"fake={fake_ms:.4f} ms, real={real_ms:.4f} ms, "
                f"speedup={result['real_speedup_vs_fake']:.3f}x, "
                f"mse={result['mse']:.4e}, max_abs={result['max_abs']:.4e}"
            )
        except Exception as exc:
            failures.append({"layer": prefix, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED: {type(exc).__name__}: {exc}")
        finally:
            del x, base
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "checkpoint": args.gptq_checkpoint,
        "definition": {
            "fake": "same GPTQ qweight/scales, no Marlin kernel, dense dequant + torch F.linear",
            "real": "same GPTQ qweight/scales, vLLM GPTQ-Marlin Linear kernel path",
        },
        "quant_config": quant_config_summary(index),
        "layers_requested": len(prefixes),
        "layers_succeeded": len(layer_results),
        "layers_failed": len(failures),
        "failures": failures,
        "metrics": {
            "latency_ms": "per-layer forward latency for synthetic activations",
            "tokens_per_s": "batch_size * seq_len / latency",
            "peak_memory_mb": "CUDA max_memory_allocated during measured forwards, if CUDA is available",
            "mse_mae_max_abs": "numerical agreement between fake output and real output",
            "relative_error": "absolute output error divided by fake output magnitude",
            "cosine_similarity": "directional agreement between fake output and real output",
        },
        "layers": layer_results,
    }
    if layer_results:
        summary["aggregate"] = {
            "mean_fake_latency_ms": statistics.mean(r["fake_latency_ms"] for r in layer_results),
            "mean_real_latency_ms": statistics.mean(r["real_latency_ms"] for r in layer_results),
            "median_fake_latency_ms": statistics.median(r["fake_latency_ms"] for r in layer_results),
            "median_real_latency_ms": statistics.median(r["real_latency_ms"] for r in layer_results),
            "mean_real_speedup_vs_fake": statistics.mean(r["real_speedup_vs_fake"] for r in layer_results),
            "max_mse": max(r["mse"] for r in layer_results),
            "max_abs_error": max(r["max_abs"] for r in layer_results),
            "min_cosine_similarity": min(r["cosine_similarity"] for r in layer_results),
        }

    print("\n========== Summary ==========")
    print(json.dumps({k: v for k, v in summary.items() if k != "layers"}, indent=2))

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[probe] JSON written to {out}")


if __name__ == "__main__":
    main()