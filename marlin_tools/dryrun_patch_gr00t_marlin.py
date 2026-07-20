"""Dry-run GR00T GPTQ-Marlin layer replacement without starting the server."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy


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
    parser.add_argument("gptq_checkpoint")
    parser.add_argument("--strict", action="store_true", help="fail if any matched layer is missing")
    args = parser.parse_args()

    model_path, data_config = TASKS[args.task]
    os.environ["GR00T_GPTQ_MARLIN_CHECKPOINT"] = args.gptq_checkpoint
    os.environ["GR00T_GPTQ_MARLIN_DRYRUN"] = "1"
    os.environ["GR00T_GPTQ_MARLIN_STRICT"] = "1" if args.strict else "0"

    cfg = load_data_config(data_config)
    Gr00tPolicy(
        model_path=model_path,
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=8,
    )


if __name__ == "__main__":
    main()

