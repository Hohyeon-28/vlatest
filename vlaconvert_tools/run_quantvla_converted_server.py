"""Run a GR00T inference server with converted QuantVLA GPTQ-like weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from quantvla_policy_patch import DEFAULT_EXCLUDE, DEFAULT_INCLUDE, replace_quantvla_converted_linears
from quantvla_gptq_modes import dtype_from_name


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=sorted(TASKS), nargs="?", default="libero_10")
    parser.add_argument("--converted-checkpoint", required=True)
    parser.add_argument("--mode", default="real", choices=["real", "fake"])
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--include", default=DEFAULT_INCLUDE)
    parser.add_argument("--exclude", default=DEFAULT_EXCLUDE)
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument("--replacement-report", default=None)
    parser.add_argument("--http-server", action="store_true")
    parser.add_argument("--api-token", default=None)
    args = parser.parse_args()

    from gr00t.eval.robot import RobotInferenceServer
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    model_path, data_config_name = TASKS[args.task]
    data_config = load_data_config(data_config_name)
    policy_kwargs = {}
    if args.device is not None:
        policy_kwargs["device"] = args.device

    print("==========================================")
    print("Starting GR00T QuantVLA-converted server")
    print(f"Task: {args.task}")
    print(f"Model: {model_path}")
    print(f"Converted checkpoint: {args.converted_checkpoint}")
    print(f"Quant mode: {args.mode}")
    print(f"Data config: {data_config_name}")
    print(f"Port: {args.port}")
    print(f"Denoising steps: {args.denoising_steps}")
    print("==========================================")

    policy = Gr00tPolicy(
        model_path=model_path,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        **policy_kwargs,
    )

    report = replace_quantvla_converted_linears(
        policy.model,
        args.converted_checkpoint,
        mode=args.mode,
        include_regex=args.include,
        exclude_regex=args.exclude,
        strict=not args.no_strict,
        dtype=dtype_from_name(args.dtype),
    )
    report.print_summary()
    report.write_json(args.replacement_report)

    if args.http_server:
        from gr00t.eval.http_server import HTTPInferenceServer

        server = HTTPInferenceServer(policy, port=args.port, host=args.host, api_token=args.api_token)
    else:
        server = RobotInferenceServer(policy, port=args.port, api_token=args.api_token)
    server.run()


if __name__ == "__main__":
    main()
