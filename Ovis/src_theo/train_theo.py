"""
Modified version of your train_theo.py to just load pretrained model
Minimal changes from your original code
"""

import pathlib
import sys

import torch
from Ovis.HF_Repo.modeling_ovis2_5 import Ovis2_5


class Ovis25Train:
    """
    Comprehensive training wrapper for Ovis2.5 VL model
    """

    def __init__(
        self,
        model_path: str = "AIDC-AI/Ovis2.5-9B",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the Ovis2.5 model

        Args:
            model_path: Path to the model (9B or 2B version)
            device: Device to run the model on
            torch_dtype: Torch dtype for inference
        """
        self.device = device
        self.torch_dtype = torch_dtype

        # Load model using custom implementation
        print("Loading Ovis2.5 model...")
        self.model = Ovis2_5.from_pretrained(
            model_path, torch_dtype=torch_dtype, trust_remote_code=True
        ).to(device)

        self.text_tokenizer = self.model.text_tokenizer
        print(f"Model loaded successfully on {device}")



if __name__ == "__main__":
    print("Starting training...")
