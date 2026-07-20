"""Benchmark FakeQuant vs RealQuant for converted QuantVLA GPTQ-like weights.

This script is intentionally layer-level. It does not run LIBERO episodes and
does not measure PPL. It measures whether the same converted QuantVLA W4
qweight/scales execute through:

1. FakeQuant: dense dequant + torch F.linear
2. RealQuant: vLLM GPTQ-Marlin Linear

with QuantVLA input/output transforms preserved around the matmul.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch

from quantvla_gptq_modes import (
    ConvertedQuantVLACheckpoint,
    QuantVLAFakeQuantLinear,
    QuantVLARealQuantMarlinLinear,
    dtype_from_name,
    mode_definitions,
)


DEFAULT_LAYER_REGEX = (
    r".*("
    r"model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
    r"|"
    r"action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)"
    r")$"
)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_peak_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)


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
    except Exception as exc:
        print(f"[bench] Triton unavailable or failed ({type(exc).__name__}: {exc}); falling back")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("converted_checkpoint", help="Output directory from convert_quantvla_to_gptq_like.py")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--layer-regex", default=DEFAULT_LAYER_REGEX)
    parser.add_argument("--max-layers", type=int, default=8, help="0 means all matching layers")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fake-only", action="store_true", help="Skip vLLM GPTQ-Marlin RealQuant")
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    dtype = dtype_from_name(args.dtype)
    checkpoint = ConvertedQuantVLACheckpoint(args.converted_checkpoint)
    layer_re = re.compile(args.layer_regex)
    prefixes = [prefix for prefix in checkpoint.prefixes() if layer_re.search(prefix)]
    if args.max_layers > 0:
        prefixes = prefixes[: args.max_layers]
    if not prefixes:
        raise SystemExit(f"No converted qweight prefixes matched regex: {args.layer_regex}")

    print("========== QuantVLA GPTQ-like Quant Mode Benchmark ==========")
    print(f"checkpoint: {args.converted_checkpoint}")
    print(f"definition.fake: {mode_definitions()['fake']}")
    print(f"definition.real: {mode_definitions()['real']}")
    print(f"layers: {len(prefixes)}")
    print(f"dtype: {args.dtype}, batch={args.batch_size}, seq_len={args.seq_len}, device={args.device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    layer_results = []
    failures = []

    for prefix in prefixes:
        layer_pack = checkpoint.load_pack(prefix)
        print(f"\n[layer] {prefix} in={layer_pack.in_features} out={layer_pack.out_features}")
        x = torch.randn(args.batch_size, args.seq_len, layer_pack.in_features, device=args.device, dtype=dtype)

        try:
            fake = QuantVLAFakeQuantLinear(layer_pack, dtype=dtype).to(args.device).eval()
            with torch.inference_mode():
                fake_y = fake(x)
                fake_backend, fake_ms, fake_peak_mb = bench(lambda: fake(x), args.warmup, args.rep)

            result = {
                "layer": prefix,
                "in_features": layer_pack.in_features,
                "out_features": layer_pack.out_features,
                "has_input_transform": bool(layer_pack.layer_info.get("has_input_transform")),
                "has_output_restore": bool(layer_pack.layer_info.get("has_output_restore")),
                "fake_backend": "QuantVLA GPTQ-like dense dequant + torch F.linear",
                "benchmark_backend_fake": fake_backend,
                "fake_latency_ms": fake_ms,
                "fake_tokens_per_s": args.batch_size * args.seq_len * 1000.0 / fake_ms if fake_ms > 0 else math.inf,
                "fake_peak_memory_mb": fake_peak_mb,
            }

            if args.fake_only:
                result["real_skipped"] = "fake-only"
                layer_results.append(result)
                print(f"fake={fake_ms:.4f} ms")
                continue

            try:
                real = QuantVLARealQuantMarlinLinear(layer_pack, dtype=dtype).to(args.device).eval()
                with torch.inference_mode():
                    real_y = real(x)
                    synchronize()
                    error_metrics = compare_outputs(fake_y, real_y)
                    real_backend, real_ms, real_peak_mb = bench(lambda: real(x), args.warmup, args.rep)
                result.update(
                    {
                        "real_backend": "vLLM GPTQ-Marlin Linear with QuantVLA transforms",
                        "benchmark_backend_real": real_backend,
                        "real_latency_ms": real_ms,
                        "real_tokens_per_s": args.batch_size * args.seq_len * 1000.0 / real_ms
                        if real_ms > 0
                        else math.inf,
                        "real_peak_memory_mb": real_peak_mb,
                        "real_speedup_vs_fake": fake_ms / real_ms if real_ms > 0 else math.inf,
                        **error_metrics,
                    }
                )
                print(
                    f"fake={fake_ms:.4f} ms, real={real_ms:.4f} ms, "
                    f"speedup={result['real_speedup_vs_fake']:.3f}x, "
                    f"mse={result['mse']:.4e}, max_abs={result['max_abs']:.4e}"
                )
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                result["real_failed"] = reason
                failures.append({"layer": prefix, "mode": "real", "reason": reason})
                print(f"real FAILED: {reason}")

            layer_results.append(result)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            failures.append({"layer": prefix, "mode": "fake", "reason": reason})
            print(f"fake FAILED: {reason}")
        finally:
            del x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    real_success = [item for item in layer_results if "real_latency_ms" in item]
    summary = {
        "checkpoint": args.converted_checkpoint,
        "definition": mode_definitions(),
        "layers_requested": len(prefixes),
        "layers_succeeded_fake": len(layer_results),
        "layers_succeeded_real": len(real_success),
        "layers_failed": len(failures),
        "failures": failures,
        "metrics": {
            "latency_ms": "per-layer forward latency for synthetic activations",
            "tokens_per_s": "batch_size * seq_len / latency",
            "peak_memory_mb": "CUDA max_memory_allocated during measured forwards, if CUDA is available",
            "mse_mae_max_abs": "numerical agreement between FakeQuant output and RealQuant output",
            "relative_error": "absolute output error divided by FakeQuant output magnitude",
            "cosine_similarity": "directional agreement between FakeQuant output and RealQuant output",
        },
        "layers": layer_results,
    }
    if layer_results:
        summary["aggregate_fake"] = {
            "mean_fake_latency_ms": statistics.mean(r["fake_latency_ms"] for r in layer_results),
            "median_fake_latency_ms": statistics.median(r["fake_latency_ms"] for r in layer_results),
        }
    if real_success:
        summary["aggregate_real"] = {
            "mean_real_latency_ms": statistics.mean(r["real_latency_ms"] for r in real_success),
            "median_real_latency_ms": statistics.median(r["real_latency_ms"] for r in real_success),
            "mean_real_speedup_vs_fake": statistics.mean(r["real_speedup_vs_fake"] for r in real_success),
            "max_mse": max(r["mse"] for r in real_success),
            "max_abs_error": max(r["max_abs"] for r in real_success),
            "min_cosine_similarity": min(r["cosine_similarity"] for r in real_success),
        }

    print("\n========== Summary ==========")
    print(json.dumps({k: v for k, v in summary.items() if k != "layers"}, indent=2))

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[bench] JSON written to {out}")


if __name__ == "__main__":
    main()
