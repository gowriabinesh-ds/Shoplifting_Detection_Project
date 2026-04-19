"""
STEP 3: PYTORCH DATASET & DATALOADERS
=======================================
Reads the extracted frames and returns tensors ready for model training.
"""

import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ── Absolute project path ──────────────────────────────────────────────────────
PROJECT_DIR   = Path(r"C:\Hope AI\11. Deep Learning\Shoplifting_Project")
MANIFEST_PATH = PROJECT_DIR / "data" / "split_manifest.json"
FRAMES_ROOT   = PROJECT_DIR / "data" / "frames"

LABEL_MAP = {"normal": 0, "shoplifting": 1}


# ── Augmentation pipelines ────────────────────────────────────────────────────
def get_transforms(split: str):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.RandomRotation(degrees=10),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


# ── Custom Dataset ─────────────────────────────────────────────────────────────
class ShopliftingDataset(Dataset):
    def __init__(self, split: str):
        if not MANIFEST_PATH.exists():
            raise FileNotFoundError(
                f"\n❌ Manifest not found: {MANIFEST_PATH}\n"
                "Please run step2_preprocess.py first!"
            )

        with open(str(MANIFEST_PATH)) as f:
            manifest = json.load(f)

        self.clips     = manifest[split]
        self.split     = split
        self.transform = get_transforms(split)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        item      = self.clips[idx]
        label_str = item["label"]
        label     = LABEL_MAP[label_str]

        clip_name  = Path(item["path"]).stem
        frames_dir = FRAMES_ROOT / self.split / label_str / clip_name

        if not frames_dir.exists():
            raise FileNotFoundError(
                f"\n❌ Frames folder not found: {frames_dir}\n"
                "Please run step2_preprocess.py first!"
            )

        frame_paths = sorted(frames_dir.glob("*.jpg"))

        if len(frame_paths) == 0:
            raise ValueError(f"No frames found in: {frames_dir}")

        frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            img = self.transform(img)
            frames.append(img)

        clip_tensor = torch.stack(frames, dim=0).permute(1, 0, 2, 3)  # (C, T, H, W)
        return clip_tensor, torch.tensor(label, dtype=torch.long)


# ── DataLoader factory ─────────────────────────────────────────────────────────
def get_dataloaders(batch_size: int = 4, num_workers: int = 0) -> dict:
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = ShopliftingDataset(split)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,  # Keep 0 on Windows to avoid errors
            pin_memory=False,
        )
        print(f"{split:5s} → {len(ds)} clips, {len(loaders[split])} batches")
    return loaders


if __name__ == "__main__":
    print("Testing dataloaders...")
    loaders = get_dataloaders(batch_size=4)
    clips, labels = next(iter(loaders["train"]))
    print(f"✅ Batch shape : {clips.shape}")
    print(f"✅ Labels      : {labels}")
