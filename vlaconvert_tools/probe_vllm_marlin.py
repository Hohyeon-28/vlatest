#!/usr/bin/env python3
"""Probe the installed vLLM package for GPTQ-Marlin Linear support."""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path


def discover_candidate_modules(vllm_module) -> list[str]:
    """Find likely GPTQ-Marlin modules without importing the whole vLLM tree."""

    candidates = ["vllm.model_executor.layers.quantization.gptq_marlin"]
    package_file = getattr(vllm_module, "__file__", None)
    if not package_file:
        return candidates

    package_root = Path(package_file).resolve().parent
    for path in package_root.rglob("*.py"):
        lower_full = str(path).lower()
        if "gptq" not in lower_full or "marlin" not in lower_full:
            continue
        rel = path.relative_to(package_root).with_suffix("")
        module_name = "vllm." + ".".join(rel.parts)
        candidates.append(module_name)
    return list(dict.fromkeys(candidates))


def scan_vllm_sources(vllm_module) -> list[tuple[str, list[str]]]:
    """Search vLLM source files for GPTQ-Marlin related symbols."""

    package_file = getattr(vllm_module, "__file__", None)
    if not package_file:
        return []

    package_root = Path(package_file).resolve().parent
    needles = [
        "GPTQMarlinConfig",
        "GPTQMarlinLinearMethod",
        "gptq_marlin",
        "GPTQ",
        "Marlin",
    ]
    hits: list[tuple[str, list[str]]] = []
    for path in package_root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        found = [needle for needle in needles if needle in text]
        if found:
            hits.append((str(path.relative_to(package_root)), found))
    return hits


def main() -> int:
    try:
        import vllm
    except Exception as exc:
        print(f"vLLM import failed: {type(exc).__name__}: {exc}")
        return 1

    print(f"vLLM version: {getattr(vllm, '__version__', '<unknown>')}")
    print(f"vLLM package: {getattr(vllm, '__file__', '<unknown>')}")

    source_hits = scan_vllm_sources(vllm)
    print("Source symbol hits:")
    for rel_path, symbols in source_hits[:80]:
        print(f"  - {rel_path}: {', '.join(symbols)}")
    if len(source_hits) > 80:
        print(f"  ... {len(source_hits) - 80} more files omitted")
    if not source_hits:
        print("  <none>")

    candidates = discover_candidate_modules(vllm)
    print("Candidate modules:")
    for name in candidates:
        print(f"  - {name}")

    usable = False
    for name in candidates:
        print(f"\n[{name}]")
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            print(f"IMPORT_ERROR {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=4)
            continue

        marlin_exports = sorted(item for item in dir(module) if "Marlin" in item or "marlin" in item)
        print(f"marlin exports: {marlin_exports}")
        has_config = hasattr(module, "GPTQMarlinConfig")
        has_method = hasattr(module, "GPTQMarlinLinearMethod")
        print(f"GPTQMarlinConfig: {has_config}")
        print(f"GPTQMarlinLinearMethod: {has_method}")
        usable = usable or (has_config and has_method)

    if usable:
        print("\nRESULT: GPTQ-Marlin Linear API is available for RealQuant.")
        return 0

    print("\nRESULT: GPTQ-Marlin Linear API was not found in this vLLM install.")
    print("FakeQuant can still run; RealQuant needs a vLLM build exposing GPTQMarlinConfig and GPTQMarlinLinearMethod.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
