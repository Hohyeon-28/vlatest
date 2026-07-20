"""Quantize a standalone HF LLM checkpoint into GPTQ format for vLLM Marlin.

vLLM does not produce the GPTQ checkpoint here; GPTQModel does. vLLM later
loads the resulting GPTQ checkpoint and can dispatch to Marlin when compatible.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List


DEFAULT_CALIBRATION_TEXTS = [
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "pick up the book and place it in the back compartment of the caddy",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both moka pots on the stove",
    "put the yellow and white mug in the microwave and close it",
] * 32


def load_calibration_texts(path: str | None, limit: int) -> List[str]:
    if path:
        texts = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                texts.append(line)
    else:
        texts = list(DEFAULT_CALIBRATION_TEXTS)
    if limit > 0:
        texts = texts[:limit]
    if not texts:
        raise ValueError("Calibration text list is empty")
    return texts


def make_quant_config(quant_config_cls, required: Dict[str, object], optional: Dict[str, object]):
    """Build QuantizeConfig while tolerating GPTQModel API differences."""
    kwargs = dict(required)
    for key, value in optional.items():
        trial = dict(kwargs)
        trial[key] = value
        try:
            quant_config_cls(**trial)
        except TypeError as exc:
            print(f"[gptq] QuantizeConfig does not accept {key}={value!r}; skipping ({exc})")
        else:
            kwargs[key] = value
    return quant_config_cls(**kwargs), kwargs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Standalone HF LLM path or repo id")
    parser.add_argument("--output", required=True, help="Output GPTQ checkpoint directory")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--calibration-file", default=None, help="Optional newline-delimited calibration prompts")
    parser.add_argument("--calibration-limit", type=int, default=256)
    parser.add_argument("--desc-act", action="store_true", help="Enable GPTQ desc_act if supported by GPTQModel")
    parser.add_argument("--asym", action="store_true", help="Prefer asymmetric quantization if supported")
    args = parser.parse_args()

    from gptqmodel import GPTQModel, QuantizeConfig

    calibration_dataset = load_calibration_texts(args.calibration_file, args.calibration_limit)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    required = {
        "bits": args.bits,
        "group_size": args.group_size,
    }
    optional = {}
    if args.desc_act:
        optional["desc_act"] = True
    if not args.asym:
        optional["sym"] = True

    quant_config, accepted_kwargs = make_quant_config(QuantizeConfig, required, optional)

    print(f"[gptq] Loading {args.model}")
    print(f"[gptq] QuantizeConfig={accepted_kwargs}")
    model = GPTQModel.load(args.model, quant_config)

    print(f"[gptq] Calibrating with {len(calibration_dataset)} text samples")
    model.quantize(calibration_dataset, batch_size=args.batch_size)

    print(f"[gptq] Saving GPTQ checkpoint to {out}")
    model.save(str(out))
    print("[gptq] Done")


if __name__ == "__main__":
    main()
