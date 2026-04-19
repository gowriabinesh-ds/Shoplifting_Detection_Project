"""
STEP 2: DATA PREPROCESSING & FRAME EXTRACTION
===============================================
Converts raw video clips into frame sequences ready for model training.
Each video becomes a folder of uniformly-sampled, resized JPEG frames.
"""

import cv2
import os
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ── Config — ALL paths are absolute, nothing relative ─────────────────────────
PROJECT_DIR = Path(r"C:\Hope AI\11. Deep Learning\Shoplifting_Project")

RAW_VIDEO_DIRS = {
    "shoplifting": PROJECT_DIR / "data" / "raw" / "Shoplifting",
    "normal":      PROJECT_DIR / "data" / "raw" / "Normal",
}

FRAMES_DIR    = PROJECT_DIR / "data" / "frames"
MANIFEST_PATH = PROJECT_DIR / "data" / "split_manifest.json"

IMG_SIZE        = (224, 224)
FRAMES_PER_CLIP = 16
SEED            = 42
VAL_SIZE        = 0.15
TEST_SIZE       = 0.15


# ── Create all needed folders if they don't exist yet ─────────────────────────
def create_folders():
    (PROJECT_DIR / "data").mkdir(parents=True, exist_ok=True)
    (PROJECT_DIR / "data" / "raw").mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    print("✅ Folders ready")


# ── Extract evenly-spaced frames from one video clip ──────────────────────────
def extract_frames(video_path: Path, output_dir: Path, n_frames: int = 16):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n_frames:
        n_frames = total

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_resized = cv2.resize(frame, IMG_SIZE)
        out_path = output_dir / f"frame_{saved:04d}.jpg"
        cv2.imwrite(str(out_path), frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved += 1

    cap.release()
    return saved > 0


# ── Improve contrast of dark CCTV frames ──────────────────────────────────────
def apply_clahe(img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        return
    lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl      = clahe.apply(l)
    lab     = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    cv2.imwrite(img_path, enhanced)


# ── Build train / val / test split and save as JSON ───────────────────────────
def build_split_manifest():
    all_clips = []

    for label, folder in RAW_VIDEO_DIRS.items():
        if not folder.exists():
            print(f"⚠ WARNING: Folder not found: {folder}")
            print(f"  Make sure your videos are inside that folder.")
            continue

        found = 0
        for ext in ["*.mp4", "*.avi", "*.mov"]:
            for clip in folder.glob(ext):
                all_clips.append({"path": str(clip), "label": label})
                found += 1

        print(f"  Found {found} {label} videos")

    if len(all_clips) == 0:
        raise ValueError(
            "\n❌ No videos found!\n"
            "Check that your folders exist and contain .mp4/.avi/.mov files:\n"
            f"  {RAW_VIDEO_DIRS['shoplifting']}\n"
            f"  {RAW_VIDEO_DIRS['normal']}"
        )

    random.seed(SEED)
    random.shuffle(all_clips)

    train_val, test = train_test_split(all_clips, test_size=TEST_SIZE, random_state=SEED)
    train, val      = train_test_split(train_val, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=SEED)

    manifest = {"train": train, "val": val, "test": test}

    with open(str(MANIFEST_PATH), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Split complete:")
    print(f"   Train : {len(train)} clips")
    print(f"   Val   : {len(val)} clips")
    print(f"   Test  : {len(test)} clips")
    print(f"   Saved : {MANIFEST_PATH}")
    return manifest


# ── Main: extract frames for every clip ───────────────────────────────────────
def preprocess_all():
    create_folders()
    manifest = build_split_manifest()

    for split, clips in manifest.items():
        print(f"\nExtracting [{split}] ...")
        for item in tqdm(clips, desc=split):
            clip_path = Path(item["path"])
            label     = item["label"]
            out_dir   = FRAMES_DIR / split / label / clip_path.stem

            success = extract_frames(clip_path, out_dir, FRAMES_PER_CLIP)
            if not success:
                print(f"  ⚠ Skipped: {clip_path.name}")
                continue

            for jpg in out_dir.glob("*.jpg"):
                apply_clahe(str(jpg))

    print("\n✅ Preprocessing complete!")
    print(f"   Frames saved to: {FRAMES_DIR}")


if __name__ == "__main__":
    preprocess_all()
