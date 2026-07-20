"""Benchmark GR00T language-model Linear paths.

Compares:
1. BF16/FP16 torch Linear baseline
2. GPTQ dequantized weights + torch F.linear
3. GPTQ-Marlin Linear wrappers
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from typing import Callable

import torch

from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy
from gr00t.quantization.gptq_marlin_linear import (
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    replace_gptq_dequantized_linears,
)


TASKS = {
    "libero_spatial": (
        "youliangtan/gr00t-n1.5-libero-spatial-posttrain",
        "examples.Libero.custom_data_config:LiberoDataConfig",
    ),
    "libero_goal": (
        "youliangtan/gr00t-n1.5-libero-goal-posttrain",
        "examples.Libero.custom_data_config:LiberoDataConfigMeanStd",
    ),
    "libero_object": (
        "youliangtan/gr00t-n1.5-libero-object-posttrain",
        "examples.Libero.custom_data_config:LiberoDataConfig",
    ),
    "libero_90": (
        "youliangtan/gr00t-n1.5-libero-90-posttrain",
        "examples.Libero.custom_data_config:LiberoDataConfig",
    ),
    "libero_10": (
        "youliangtan/gr00t-n1.5-libero-long-posttrain",
        "examples.Libero.custom_data_config:LiberoDataConfig",
    ),
}


@contextmanager
def quant_env(**updates: str):
    keys = [k for k in os.environ if k.startswith("GR00T_GPTQ_MARLIN_") or k.startswith("GR00T_DUQUANT_")]
    saved = {k: os.environ[k] for k in keys}
    for key in keys:
        os.environ.pop(key, None)
    os.environ.update({k: v for k, v in updates.items() if v is not None})
    try:
        yield
    finally:
        for key in [k for k in os.environ if k.startswith("GR00T_GPTQ_MARLIN_") or k.startswith("GR00T_DUQUANT_")]:
            os.environ.pop(key, None)
        os.environ.update(saved)


def dtype_from_name(name: str) -> torch.dtype:
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_policy(task: str, device: str) -> Gr00tPolicy:
    model_path, data_config = TASKS[task]
    cfg = load_data_config(data_config)
    return Gr00tPolicy(
        model_path=model_path,
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=8,
        device=device,
    )


def language_model_from_policy(policy: Gr00tPolicy, include_lm_head: bool):
    lm = policy.model.backbone.eagle_model.language_model
    if include_lm_head or not hasattr(lm, "model"):
        return lm
    return lm.model


def hidden_size_of(module) -> int:
    cfg = getattr(module, "config", None)
    if cfg is not None and hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    if hasattr(module, "embed_tokens"):
        return int(module.embed_tokens.embedding_dim)
    raise RuntimeError("Could not infer language model hidden size")


def make_runner(module, batch_size: int, seq_len: int, dtype: torch.dtype, device: str) -> Callable[[], object]:
    hidden_size = hidden_size_of(module)
    torch.manual_seed(1234)
    fixed_inputs_embeds = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    fixed_attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)

    @torch.inference_mode()
    def run_llm():
        return module(
            inputs_embeds=fixed_inputs_embeds,
            attention_mask=fixed_attention_mask,
            use_cache=False,
        )

    return run_llm


def bench_with_triton(fn: Callable[[], object], warmup: int, rep: int) -> tuple[str, float]:
    try:
        import triton

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        if isinstance(ms, (tuple, list)):
            ms = float(ms[0])
        return "triton.testing.do_bench", float(ms)
    except Exception as exc:
        print(f"[bench] Triton benchmark unavailable ({type(exc).__name__}: {exc}); falling back")
        if not torch.cuda.is_available():
            start = time.perf_counter()
            for _ in range(max(1, rep)):
                fn()
            elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(1, rep)
            return "time.perf_counter", elapsed_ms

        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(rep):
            fn()
        end_event.record()
        torch.cuda.synchronize()
        return "torch.cuda.Event", start_event.elapsed_time(end_event) / rep


def unload(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_case(args, case_name: str) -> dict:
    dtype = dtype_from_name(args.dtype)
    strict = not args.no_strict
    report = None
    with quant_env():
        if case_name == "bf16_linear":
            policy = load_policy(args.task, args.device)
        elif case_name == "gptq_dequant_torch_linear":
            policy = load_policy(args.task, args.device)
            report = replace_gptq_dequantized_linears(
                policy.model,
                args.gptq_checkpoint,
                include_regex=args.include,
                exclude_regex=args.exclude,
                strict=strict,
                dtype=dtype,
            )
            report.print_summary(prefix="[GR00T-GPTQ-DEQUANT]")
        elif case_name == "gptq_marlin_linear":
            env_updates = {
                "GR00T_GPTQ_MARLIN_CHECKPOINT": args.gptq_checkpoint,
                "GR00T_GPTQ_MARLIN_INCLUDE": args.include,
                "GR00T_GPTQ_MARLIN_EXCLUDE": args.exclude,
                "GR00T_GPTQ_MARLIN_STRICT": "0" if args.no_strict else "1",
                "GR00T_GPTQ_MARLIN_REPORT": args.replacement_report or "",
            }
            with quant_env(**env_updates):
                policy = load_policy(args.task, args.device)
        else:
            raise ValueError(case_name)

        module = language_model_from_policy(policy, include_lm_head=args.include_lm_head)
        module.eval()
        runner = make_runner(module, args.batch_size, args.seq_len, dtype, args.device)
        runner()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        backend, latency_ms = bench_with_triton(runner, args.warmup, args.rep)
        tokens_per_s = args.batch_size * args.seq_len * 1000.0 / latency_ms
        result = {
            "case": case_name,
            "benchmark_backend": backend,
            "latency_ms": latency_ms,
            "tokens_per_s": tokens_per_s,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "dtype": args.dtype,
            "include_lm_head": args.include_lm_head,
        }
        if report is not None:
            result["replacement_report"] = report.to_dict()
        unload(runner, module, policy)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=sorted(TASKS), nargs="?", default="libero_10")
    parser.add_argument("gptq_checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--include", default=DEFAULT_INCLUDE)
    parser.add_argument("--exclude", default=DEFAULT_EXCLUDE)
    parser.add_argument("--include-lm-head", action="store_true", help="Benchmark full CausalLM including lm_head")
    parser.add_argument("--no-strict", action="store_true", help="Do not fail on unreplaced target layers")
    parser.add_argument("--replacement-report", default=None, help="Optional JSON report path for Marlin replacement")
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    results = []
    for case_name in ["bf16_linear", "gptq_dequant_torch_linear", "gptq_marlin_linear"]:
        print(f"\n========== Benchmark: {case_name} ==========")
        result = run_case(args, case_name)
        results.append(result)
        print(
            f"{case_name}: {result['latency_ms']:.4f} ms "
            f"({result['tokens_per_s']:.2f} tokens/s, backend={result['benchmark_backend']})"
        )

    baseline_ms = results[0]["latency_ms"]
    print("\n========== Summary ==========")
    for result in results:
        speedup = baseline_ms / result["latency_ms"]
        result["speedup_vs_bf16"] = speedup
        print(
            f"{result['case']}: latency_ms={result['latency_ms']:.4f}, "
            f"speedup_vs_bf16={speedup:.3f}x, tokens_per_s={result['tokens_per_s']:.2f}"
        )

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[bench] JSON written to {out}")


if __name__ == "__main__":
    main()

