"""
PyTorch-style DataLoader with multiprocessing workers.
Uses standard PyTorch DataLoader with num_workers for true parallel loading.
"""

import torch
from torch.utils.data import DataLoader, Dataset

from InternVL3.utils.preprocess import load_image


class LHDataset(Dataset):
    """
    PyTorch Dataset wrapper for LHDataLoader.
    Enables use of standard PyTorch DataLoader with multiprocessing.
    """

    def __init__(self, base_loader, device, max_num=12, start_idx=None, end_idx=None):
        """
        Args:
            base_loader: Base LHDataLoader instance
            device: Target device (e.g., "cuda:0")
            max_num: Max number of image patches (passed to load_image)
            start_idx: Optional starting index for data slice
            end_idx: Optional ending index for data slice
        """
        self.base_loader = base_loader
        self.device = device
        self.max_num = max_num

        # Support for index slicing (parallel processing)
        self.start_idx = start_idx if start_idx is not None else 0
        self.end_idx = end_idx if end_idx is not None else len(base_loader)

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        """
        Load and preprocess a single item.
        Called by worker processes in parallel.
        """
        # Adjust index if we're working with a slice
        actual_idx = self.start_idx + idx

        # Get item from base loader
        item = self.base_loader[actual_idx]

        try:
            # Load image (preprocessing happens in worker process)
            image_path = item["image_file"]
            pixels = load_image(image_path, max_num=self.max_num).to(torch.bfloat16)

            return {**item, "pixels": pixels, "error": None, "index": actual_idx}
        except Exception as e:
            # Return error info instead of crashing worker
            return {
                **item,
                "pixels": torch.zeros((3, 224, 224), dtype=torch.bfloat16),  # Dummy tensor
                "error": str(e),
                "index": actual_idx,
            }


def collate_fn(batch):
    """
    Custom collate function that doesn't stack tensors.
    Returns a list of items since pixels have different shapes.
    """
    return batch


def create_dataloader(
    base_loader, device, max_num=12, num_workers=4, start_idx=None, end_idx=None, pin_memory=True
):
    """
    Create a PyTorch DataLoader with multiprocessing workers.

    Args:
        base_loader: Base LHDataLoader instance
        device: Target device (e.g., "cuda:0")
        max_num: Max number of image patches
        num_workers: Number of worker processes for parallel loading (default: 4)
        start_idx: Optional starting index for data slice
        end_idx: Optional ending index for data slice
        pin_memory: Use pinned memory for faster CPU-to-GPU transfer

    Returns:
        PyTorch DataLoader instance
    """
    dataset = LHDataset(base_loader, device, max_num, start_idx, end_idx)

    return DataLoader(
        dataset,
        batch_size=1,  # Process one image at a time
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,  # Keep workers alive between iterations
        prefetch_factor=2 if num_workers > 0 else None,  # Each worker prefetches 2 batches
    )
