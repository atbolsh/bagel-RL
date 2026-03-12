#!/usr/bin/env python3
"""
Merge LoRA adapters with a base model and save the result.
Reads config from the JSON file saved during training (e.g. outputs/run_name/config.json).
"""

import argparse
import json
import torch
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

try:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForVision2Seq


def load_config(config_path: Path):
    """Load config JSON. Accept path to config.json or to directory containing it."""
    p = Path(config_path)
    if p.is_dir():
        p = p / "config.json"
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, "r") as f:
        config = json.load(f)
    return config, p.parent


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters with a base model and save the result."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.json (or output directory containing it, e.g. outputs/position_qa_dpo)"
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        required=True,
        help="Checkpoint step number (e.g. 28000 for checkpoint-28000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save merged model (default: {config_dir}/merged_checkpoint_{step})"
    )
    parser.add_argument(
        "--no_safetensors",
        action="store_true",
        help="Don't use safetensors format for saving"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    config, output_dir = load_config(args.config)
    model_cfg = config.get("model", {})
    base_model_name = model_cfg.get("name")
    if not base_model_name:
        raise ValueError("Config must have model.name")

    is_vlm = model_cfg.get("model_type") == "vlm"
    trust_remote_code = model_cfg.get("trust_remote_code", False)

    # dtype from config (e.g. "bfloat16") or default for VLMs
    dtype_str = model_cfg.get("torch_dtype", "bfloat16" if is_vlm else "float32")
    if dtype_str in ("bfloat16", "bf16"):
        dtype = torch.bfloat16
    elif dtype_str in ("float16", "fp16"):
        dtype = torch.float16
    else:
        dtype = torch.float32

    adapter_path = output_dir / f"checkpoint-{args.checkpoint}"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {adapter_path}")

    save_dir = args.output_dir
    if save_dir is None:
        save_dir = output_dir / f"merged_checkpoint_{args.checkpoint}"
    save_dir = Path(save_dir)

    print(f"Base model: {base_model_name}")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {save_dir}")
    print(f"VLM: {is_vlm}, dtype: {dtype}")

    if is_vlm:
        print("Loading base VLM...")
        base_model = AutoModelForVision2Seq.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )
    else:
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )

    print("Loading adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    print("Merging adapter with base model...")
    model = model.merge_and_unload()

    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to {save_dir}")
    model.save_pretrained(
        save_dir,
        safe_serialization=not args.no_safetensors
    )

    if is_vlm:
        print("Saving processor...")
        processor = AutoProcessor.from_pretrained(
            str(adapter_path),
            trust_remote_code=trust_remote_code,
        )
        processor.save_pretrained(save_dir)
    else:
        print("Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=trust_remote_code,
        )
        tokenizer.save_pretrained(save_dir)

    print(f"✓ Model successfully merged and saved to {save_dir}")


if __name__ == "__main__":
    main()
