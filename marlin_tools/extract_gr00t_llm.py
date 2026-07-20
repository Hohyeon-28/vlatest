"""Extract the LLM backbone from a GR00T checkpoint as a standalone HF model.

This script deliberately avoids instantiating the full GR00T/EAGLE model. The
vision side can import flash-attn extensions, which are unnecessary when we only
need the language model weights for GPTQ/Marlin preparation. Instead, it reads
GR00T checkpoint shards directly and rewrites keys under
`backbone.eagle_model.language_model.*` into a standalone causal-LM directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

LLM_PREFIX_CANDIDATES = (
    "backbone.eagle_model.language_model.",
    "model.backbone.eagle_model.language_model.",
    "backbone.language_model.",
    "language_model.",
)


def snapshot_or_path(model_path: str) -> Path:
    src = Path(model_path)
    if src.exists():
        return src
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_path, repo_type="model"))


def iter_weight_files(model_dir: Path) -> Iterable[Path]:
    files = sorted(model_dir.glob("*.safetensors"))
    if files:
        return files
    return sorted(model_dir.glob("pytorch_model*.bin"))


def read_eagle_text_config(repo_root: Path) -> dict:
    eagle_config = repo_root / "gr00t" / "model" / "backbone" / "eagle2_hg_model" / "config.json"
    if not eagle_config.exists():
        raise FileNotFoundError(f"Could not find local EAGLE config: {eagle_config}")
    data = json.loads(eagle_config.read_text(encoding="utf-8"))
    text_config = dict(data["text_config"])
    text_config.pop("_attn_implementation_autoset", None)
    text_config["_attn_implementation"] = "sdpa"
    text_config["torch_dtype"] = "bfloat16"
    return text_config


def copy_tokenizer_sidecars(src_dirs: list[Path], out: Path) -> None:
    patterns = [
        "tokenizer*",
        "vocab*",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.json",
    ]
    for src in src_dirs:
        if not src.exists():
            continue
        for pattern in patterns:
            for path in src.glob(pattern):
                target = out / path.name
                if path.is_file() and not target.exists():
                    shutil.copy2(path, target)


def extract_safetensors(model_dir: Path, out: Path) -> tuple[int, str | None]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {}
    used_prefix = None
    for shard in iter_weight_files(model_dir):
        if shard.suffix != ".safetensors":
            continue
        print(f"[extract] Scanning {shard.name}")
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                for prefix in LLM_PREFIX_CANDIDATES:
                    if key.startswith(prefix):
                        used_prefix = used_prefix or prefix
                        tensors[key.removeprefix(prefix)] = f.get_tensor(key)
                        break

    if not tensors:
        return 0, used_prefix

    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
    return len(tensors), used_prefix


def extract_torch_bins(model_dir: Path, out: Path) -> tuple[int, str | None]:
    import torch
    from safetensors.torch import save_file

    tensors = {}
    used_prefix = None
    for shard in iter_weight_files(model_dir):
        if shard.suffix == ".safetensors":
            continue
        print(f"[extract] Scanning {shard.name}")
        state = torch.load(shard, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        for key, value in state.items():
            for prefix in LLM_PREFIX_CANDIDATES:
                if key.startswith(prefix):
                    used_prefix = used_prefix or prefix
                    tensors[key.removeprefix(prefix)] = value
                    break

    if not tensors:
        return 0, used_prefix

    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
    return len(tensors), used_prefix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="GR00T HF repo id or local checkpoint path")
    parser.add_argument("--data-config", required=False, default="", help="Unused compatibility argument")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--llm-attr", default=None, help="Compatibility argument; direct extraction uses known prefixes")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--denoising-steps", type=int, default=8)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_dir = snapshot_or_path(args.model_path)
    print(f"[extract] Reading GR00T checkpoint shards from: {model_dir}")
    print("[extract] Direct shard extraction avoids EAGLE/flash-attn imports.")

    text_config = read_eagle_text_config(repo_root)
    (out / "config.json").write_text(json.dumps(text_config, indent=2), encoding="utf-8")

    count, used_prefix = extract_safetensors(model_dir, out)
    if count == 0:
        count, used_prefix = extract_torch_bins(model_dir, out)
    if count == 0:
        raise SystemExit(
            "No LLM tensors found. Expected keys like "
            "backbone.eagle_model.language_model.model.layers.0.self_attn.q_proj.weight"
        )

    eagle_dir = repo_root / "gr00t" / "model" / "backbone" / "eagle2_hg_model"
    copy_tokenizer_sidecars([model_dir, eagle_dir], out)

    manifest = {
        "source_model_path": args.model_path,
        "source_snapshot_path": str(model_dir),
        "data_config": args.data_config,
        "llm_prefix": used_prefix,
        "tensor_count": count,
        "note": "Standalone Qwen LLM backbone extracted directly from GR00T shards for GPTQ/Marlin quantization.",
    }
    (out / "gr00t_llm_extract_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[extract] Saved {count} LLM tensors with prefix {used_prefix}")
    print(f"[extract] Wrote {out}")


if __name__ == "__main__":
    main()
