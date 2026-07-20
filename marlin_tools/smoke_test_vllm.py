"""Minimal vLLM load/generate smoke test for a GPTQ checkpoint."""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python marlin_tools/smoke_test_vllm.py <gptq_checkpoint_dir> [prompt]")
    model_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Pick up the mug and place it on the plate."

    from vllm import LLM, SamplingParams

    print(f"[vllm] Loading {model_path}")
    llm = LLM(model=model_path, trust_remote_code=True)
    outputs = llm.generate([prompt], SamplingParams(max_tokens=16, temperature=0.0))
    for out in outputs:
        print("[vllm] prompt:", out.prompt)
        print("[vllm] output:", out.outputs[0].text)


if __name__ == "__main__":
    main()
