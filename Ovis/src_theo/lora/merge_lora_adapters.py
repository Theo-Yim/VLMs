"""
Merge LoRA adapters with Ovis2.5 base model
"""

import argparse
import os
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer
from ovis.model.modeling_ovis2_5 import Ovis2_5
from ovis.model.configuration_ovis2_5 import Ovis2_5_Config


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
    
    # Load the base model using our local Ovis2_5 class
    config = Ovis2_5_Config.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = Ovis2_5.from_pretrained(
        base_model_name,
        config=config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
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
    # progressbar=False to avoid tqdm issues in some environments
    merged_model = model.merge_and_unload(progressbar=False)

    # IMPORTANT: Check if there are modules_to_save (e.g., embeddings, lm_head)
    # These are fully trained (not LoRA) and need to be copied over
    if hasattr(peft_config, 'modules_to_save') and peft_config.modules_to_save:
        print(f"Copying fully-trained modules: {peft_config.modules_to_save}")

        # The modules_to_save are already in the merged_model after merge_and_unload()
        # but let's verify they were properly transferred
        for module_name in peft_config.modules_to_save:
            if hasattr(model, module_name):
                print(f"  ✓ {module_name} will be saved")
            else:
                # Try to find it in nested attributes
                parts = module_name.split('.')
                obj = model
                found = True
                for part in parts:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        found = False
                        break
                if found:
                    print(f"  ✓ {module_name} found and will be saved")
                else:
                    print(f"  ⚠ Warning: {module_name} not found in model")

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

    # Copy additional config files that are NOT auto-generated
    # Note: generation_config.json is automatically created by save_pretrained()
    print("Copying additional config files...")
    import shutil

    # Only copy files that are needed but not automatically saved
    config_files = [
        "preprocessor_config.json",  # Required for image processor
        "chat_template.json"          # Useful for chat formatting
    ]

    for config_file in config_files:
        # Try adapter path first (from training checkpoint)
        source = os.path.join(adapter_path, config_file)
        if os.path.exists(source):
            dest = os.path.join(output_path, config_file)
            shutil.copy2(source, dest)
            print(f"  ✓ Copied {config_file} from adapter")
        else:
            # Try base model path (from original model)
            source = os.path.join(base_model_name, config_file)
            if os.path.exists(source):
                dest = os.path.join(output_path, config_file)
                shutil.copy2(source, dest)
                print(f"  ✓ Copied {config_file} from base model")
            # If file doesn't exist anywhere, that's OK - it's optional

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