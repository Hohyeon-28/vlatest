"""Inspect whether a checkpoint has GPTQ-style files/keys expected by vLLM."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python marlin_tools/inspect_gptq_checkpoint.py <checkpoint_dir>")

    root = Path(sys.argv[1])
    if not root.exists():
        raise SystemExit(f"Not found: {root}")

    print(f"[inspect] {root}")
    qcfg = root / "quantize_config.json"
    if qcfg.exists():
        print("[inspect] quantize_config.json: yes")
        try:
            print(json.dumps(json.loads(qcfg.read_text(encoding="utf-8")), indent=2)[:2000])
        except Exception as exc:
            print(f"[inspect] failed to parse quantize_config.json: {exc}")
    else:
        print("[inspect] quantize_config.json: NO")

    st_files = sorted(root.glob("*.safetensors"))
    bin_files = sorted(root.glob("*.bin"))
    print(f"[inspect] safetensors files: {len(st_files)}")
    print(f"[inspect] bin files: {len(bin_files)}")

    found_keys = set()
    if st_files:
        try:
            from safetensors import safe_open

            for file in st_files[:3]:
                with safe_open(file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        for marker in ("qweight", "qzeros", "scales", "g_idx"):
                            if marker in key:
                                found_keys.add(marker)
                if found_keys:
                    break
        except Exception as exc:
            print(f"[inspect] safetensors key scan failed: {exc}")

    print(f"[inspect] GPTQ-like tensor markers: {sorted(found_keys) if found_keys else 'none found'}")
    if qcfg.exists() and {"qweight", "scales"}.issubset(found_keys):
        print("[inspect] Looks GPTQ-like for vLLM. Marlin use still depends on vLLM version, GPU, group size, and dtype.")
    else:
        print("[inspect] This does not yet look like a GPTQ-Marlin-ready checkpoint.")


if __name__ == "__main__":
    main()
