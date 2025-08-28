"""
Merge LoRA adapters with Ovis2.5 base model
"""

import argparse
import torch
from peft import AutoPeftModelForCausalLM
from ovis.model.modeling_ovis2_5 import Ovis2_5


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
    
    print(f"Loading LoRA model from {adapter_path}...")
    
    # Load the PEFT model with adapters
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
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
    
    # Also save the tokenizer
    tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
    if tokenizer is None:
        # Try to get tokenizer from the original model config
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        except:
            print("Warning: Could not save tokenizer. You may need to copy it manually.")
    
    if tokenizer is not None:
        tokenizer.save_pretrained(output_path)
    
    print("Merge completed successfully!")
    print(f"Merged model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters with Ovis2.5 base model")
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