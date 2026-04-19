"""
STEP 2: DATA PREPROCESSING & FRAME EXTRACTION
===============================================
Converts raw video clips into frame sequences ready for model training.
Each video becomes a folder of uniformly-sampled, resized JPEG frames.
"""

import cv2  # Used to read videos and process images
import os   # Helps interact with the operating system (files/folders)
import json  # Used to save and load data in JSON format
import random  # Used to shuffle data randomly
import numpy as np  # Used for numerical operations
from pathlib import Path  # Used to handle file paths easily
from tqdm import tqdm  # Shows progress bar while processing files
from sklearn.model_selection import train_test_split   # Splits dataset into train/validation/test sets

# ── Main project folder path ─────────────────────────────────────────
PROJECT_DIR = Path(r"C:\Hope AI\11. Deep Learning\Shoplifting_Project")   

# Store paths for raw videos for each class (shoplifting & normal)
RAW_VIDEO_DIRS = {
    "shoplifting": PROJECT_DIR / "data" / "raw" / "Shoplifting",
    "normal":      PROJECT_DIR / "data" / "raw" / "Normal",
}

FRAMES_DIR    = PROJECT_DIR / "data" / "frames"      # Folder where extracted frames will be stored
MANIFEST_PATH = PROJECT_DIR / "data" / "split_manifest.json"   # File path where dataset split info (train/val/test) will be saved

IMG_SIZE        = (224, 224)   # Resize all frames to this size
FRAMES_PER_CLIP = 16           # Number of frames to extract per video
SEED            = 42           # Fix randomness for reproducibility
VAL_SIZE        = 0.15         # 15% for validation
TEST_SIZE       = 0.15         # 15% for testing


# ── Create all needed folders if they don't exist yet ─────────────────────────
def create_folders():                                                  
    (PROJECT_DIR / "data").mkdir(parents=True, exist_ok=True)          # Create data folder
    (PROJECT_DIR / "data" / "raw").mkdir(parents=True, exist_ok=True)  # Create raw folder
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)                      # Create frames folder
    print("✅ Folders ready")                                         # Confirm creation


# ── Extract evenly-spaced frames from one video clip ──────────────────────────
def extract_frames(video_path: Path, output_dir: Path, n_frames: int = 16):
    
    cap = cv2.VideoCapture(str(video_path))           # Open video file
    
    if not cap.isOpened():
        return False          # Stop if video cannot be opened


    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    
    if total < n_frames:
        n_frames = total                # Adjust if video has fewer frames

    indices = np.linspace(0, total - 1, n_frames, dtype=int)  # Select evenly spaced frames
    output_dir.mkdir(parents=True, exist_ok=True)   # Create output folder for this video

    saved = 0   # Counter for saved frames
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)    # Move to specific frame
        ret, frame = cap.read()                  # Read frame
        if not ret:
            continue   # Skip if reading failed
        frame_resized = cv2.resize(frame, IMG_SIZE)   # Resize frame
        out_path = output_dir / f"frame_{saved:04d}.jpg"   # Create filename
        cv2.imwrite(str(out_path), frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])  # Save frame as image
        saved += 1   # Increase count

    cap.release()      # Close video
    return saved > 0   # Return True if frames saved


# ── Improve contrast of dark CCTV frames ──────────────────────────────────────
def apply_clahe(img_path: str):
    img = cv2.imread(img_path)   # Read image
    if img is None:
        return         # Stop if image not found
    lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)   # Convert to LAB color space
    l, a, b = cv2.split(lab)                         # Split channels
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Create CLAHE object
    cl      = clahe.apply(l)      # Apply CLAHE on brightness channel
    lab     = cv2.merge((cl, a, b))    # Merge channels back
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)       # Convert back to normal color
    cv2.imwrite(img_path, enhanced)       # Save enhanced image


# ── Build train / val / test split and save as JSON ───────────────────────────
def build_split_manifest():
    all_clips = []      # List to store all video paths

    for label, folder in RAW_VIDEO_DIRS.items():   # Loop through classes
        if not folder.exists():
            print(f"⚠ WARNING: Folder not found: {folder}")
            print(f"  Make sure your videos are inside that folder.")
            continue

        found = 0
        for ext in ["*.mp4", "*.avi", "*.mov"]:         # Check video formats
            for clip in folder.glob(ext):               # Find videos
                all_clips.append({"path": str(clip), "label": label})
                found += 1

        print(f"  Found {found} {label} videos")       # Show count

    if len(all_clips) == 0:
        raise ValueError(                 
            "\n❌ No videos found!\n"
            "Check that your folders exist and contain .mp4/.avi/.mov files:\n"
            f"  {RAW_VIDEO_DIRS['shoplifting']}\n"
            f"  {RAW_VIDEO_DIRS['normal']}"
        )                                               # Stop if no data

    random.seed(SEED)             # Fix randomness
    random.shuffle(all_clips)     # Shuffle dataset
    
    # Split data into train+val and test to keep test set completely unseen for final evaluation
    train_val, test = train_test_split(all_clips, test_size=TEST_SIZE, random_state=SEED)    
    
    # Further split train+val into train and validation to tune model without touching test data
    train, val      = train_test_split(train_val, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=SEED)   

    # Store all dataset splits in a dictionary for easy access later
    manifest = {"train": train, "val": val, "test": test}    

    with open(str(MANIFEST_PATH), "w") as f:
        json.dump(manifest, f, indent=2)           # Save split info as JSON

    print(f"\n✅ Split complete:")                 # Print summary
    print(f"   Train : {len(train)} clips")
    print(f"   Val   : {len(val)} clips")
    print(f"   Test  : {len(test)} clips")
    print(f"   Saved : {MANIFEST_PATH}")
    return manifest                            # Return dataset split


# ── Main: extract frames for every clip ───────────────────────────────────────
def preprocess_all():
    create_folders()           # Create required folders
    manifest = build_split_manifest()     # Create dataset split

    for split, clips in manifest.items():     # Loop through train/val/test
        print(f"\nExtracting [{split}] ...")
        for item in tqdm(clips, desc=split):    # Loop through videos
            clip_path = Path(item["path"])      # Get video path
            label     = item["label"]           # Get label
            out_dir   = FRAMES_DIR / split / label / clip_path.stem   # Define output folder

            success = extract_frames(clip_path, out_dir, FRAMES_PER_CLIP)  # Extract frames
            if not success:
                print(f"  ⚠ Skipped: {clip_path.name}")
                continue                     # Skip failed videos

            for jpg in out_dir.glob("*.jpg"):  # Loop through frames
                apply_clahe(str(jpg))          # Enhance each frame

    print("\n✅ Preprocessing complete!")    # Done message
    print(f"   Frames saved to: {FRAMES_DIR}")   # Show location


# Run preprocessing only when this script is executed directly (not when imported)
if __name__ == "__main__":
    preprocess_all()        
