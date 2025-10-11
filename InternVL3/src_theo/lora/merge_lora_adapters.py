"""
Merge LoRA adapters with InternVL3.5 base model
"""

import argparse
import os
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModel


def merge_lora_adapters(
    adapter_path: str,
    output_path: str,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """
    Merge LoRA adapters with base model and save the merged model

    Args:
        adapter_path: Path to the LoRA adapters (checkpoint directory)
        output_path: Path to save the merged model
        device_map: Device map for loading model
        torch_dtype: Torch dtype for model
    """

    print(f"Loading LoRA adapters from {adapter_path}...")

    # Load PEFT config to get base model info
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_name = peft_config.base_model_name_or_path

    print(f"Loading base model: {base_model_name}")

    # Load the base model using AutoModel (InternVL uses AutoModel, not AutoModelForCausalLM)
    base_model = AutoModel.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print("Loading LoRA adapters...")

    # Load the PEFT model with our base model
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )

    print("Merging LoRA adapters with base model...")

    # Merge the adapters with the base model
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")

    # Save the merged model
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="5GB"
    )

    # Load and save the tokenizer
    print("Saving tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
    except:
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            tokenizer.save_pretrained(output_path)
        except:
            print("Warning: Could not save tokenizer. You may need to copy it manually.")

    print("Merge completed successfully!")
    print(f"Merged model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters with InternVL3.5 base model")
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the LoRA adapters (checkpoint directory)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the merged model"
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for loading model (default: auto)"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model (default: bfloat16)"
    )

    args = parser.parse_args()

    # Convert string dtype to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    merge_lora_adapters(
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
    )


if __name__ == "__main__":
    main()
