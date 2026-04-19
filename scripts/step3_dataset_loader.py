"""
STEP 3: PYTORCH DATASET & DATALOADERS
=======================================
Reads the extracted frames and returns tensors ready for model training.
"""

import json
import torch                                         # PyTorch library for tensors
import numpy as np
from pathlib import Path
from PIL import Image                                # Open image files
from torch.utils.data import Dataset, DataLoader     # Create dataset & loader
import torchvision.transforms as T                   # Image transformations

# ── Project path ──────────────────────────────────────────────────────
PROJECT_DIR   = Path(r"C:\Hope AI\11. Deep Learning\Shoplifting_Project")
MANIFEST_PATH = PROJECT_DIR / "data" / "split_manifest.json"
FRAMES_ROOT   = PROJECT_DIR / "data" / "frames"

LABEL_MAP = {"normal": 0, "shoplifting": 1}    # Convert labels to numbers


# ── Transformations (preprocessing + augmentation) ──────────────────────────
def get_transforms(split: str):
    mean = [0.485, 0.456, 0.406]   # Standard mean for normalization
    std  = [0.229, 0.224, 0.225]   # Standard deviation for normalization

    if split == "train":
        return T.Compose([
            T.Resize((224, 224)),                                         # Resize image
            T.RandomHorizontalFlip(p=0.5),                                # Random flip for augmentation
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # Change colors slightly
            T.RandomRotation(degrees=10),                                 # Rotate image slightly
            T.ToTensor(),                                                 # Convert image to tensor
            T.Normalize(mean, std),                                       # Normalize values
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),        # Resize image
            T.ToTensor(),                # Convert to tensor
            T.Normalize(mean, std),      # Normalize
        ])


# ── Custom Dataset ─────────────────────────────────────────────────────────────
class ShopliftingDataset(Dataset):
    def __init__(self, split: str):
        if not MANIFEST_PATH.exists():
            raise FileNotFoundError(
                f"\n❌ Manifest not found: {MANIFEST_PATH}\n"
                "Please run step2_preprocess.py first!"
            )                                               # Stop if dataset split file not found

        with open(str(MANIFEST_PATH)) as f:
            manifest = json.load(f)                 # Load dataset split info

        self.clips     = manifest[split]            # Get clips for train/val/test
        self.split     = split                      # Store current split name
        self.transform = get_transforms(split)      # Get transforms for this split

    def __len__(self):
        return len(self.clips)             # Return number of video clips

    def __getitem__(self, idx):
        item      = self.clips[idx]         # Get one video entry
        label_str = item["label"]           # Get label name (normal/shoplifting)
        label     = LABEL_MAP[label_str]    # Convert label to number

        clip_name  = Path(item["path"]).stem   # Get video name without extension
        frames_dir = FRAMES_ROOT / self.split / label_str / clip_name  # Path where frames of this video are stored

        if not frames_dir.exists():
            raise FileNotFoundError(
                f"\n❌ Frames folder not found: {frames_dir}\n"
                "Please run step2_preprocess.py first!"
            )

        frame_paths = sorted(frames_dir.glob("*.jpg"))   # Get all frame images

        if len(frame_paths) == 0:
            raise ValueError(f"No frames found in: {frames_dir}")

        frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")   # Open image
            img = self.transform(img)             # Apply transformations
            frames.append(img)                    # Add to list

        clip_tensor = torch.stack(frames, dim=0).permute(1, 0, 2, 3)  # (C, T, H, W)
        # Convert list of frames into tensor (C, T, H, W)

        return clip_tensor, torch.tensor(label, dtype=torch.long)
        # Return video tensor and label


# ── DataLoader factory ─────────────────────────────────────────────────────────
def get_dataloaders(batch_size: int = 4, num_workers: int = 0) -> dict:
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = ShopliftingDataset(split)      # Create dataset
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,        # Number of samples per batch
            shuffle=(split == "train"),   # Shuffle only training data
            num_workers=num_workers,      # Parallel loading (Keep 0 on Windows to avoid errors)
            pin_memory=False,             # Memory optimization (not needed here)
        )
        print(f"{split:5s} → {len(ds)} clips, {len(loaders[split])} batches")   # Show dataset info
    return loaders          # Return all dataloaders


if __name__ == "__main__":
    print("Testing dataloaders...")                 # Check if everything works
    loaders = get_dataloaders(batch_size=4)         # Create dataloaders
    clips, labels = next(iter(loaders["train"]))    # Get one batch from training data
    print(f"✅ Batch shape : {clips.shape}")       # Show tensor shape
    print(f"✅ Labels      : {labels}")            # Show labels
