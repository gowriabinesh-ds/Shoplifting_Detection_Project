"""

STEP 1: DATA COLLECTION — DOCUMENTATION                  
Shoplifting Detection System                             
UCF-Crime Dataset — University of Central Florida        

═══════════════════════════════════════════════════════════════════
DATASET USED
═══════════════════════════════════════════════════════════════════

  Name     : UCF-Crime Dataset
  Source   : University of North Carolina Charlotte (UNC Charlotte)
  URL      : https://www.crcv.ucf.edu/research/real-world-anomaly-detection/
  Format   : MP4 / AVI video clips (real CCTV surveillance footage)

  Full dataset contains 1,900 videos across 13 crime categories:
    Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting,
    RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism

═══════════════════════════════════════════════════════════════════
WHAT WAS DOWNLOADED FOR THIS PROJECT
═══════════════════════════════════════════════════════════════════

  Only two specific classes were downloaded — NOT the full dataset.
  This keeps storage manageable and creates a clean binary
  classification problem (shoplifting vs normal).

  ┌─────────────────────────────────────────────────────────────┐
  │  File              : Anomaly-Videos-Part-4.zip  (6.11 GB)  │
  │  Contains          : Shoplifting folder extracted from it  │
  │  Clips used        : 50 shoplifting video clips            │
  │  Stored at         : data\raw\Shoplifting\                 │
  ├─────────────────────────────────────────────────────────────┤
  │  File              : Normal_Videos_for_Event_Recognition   │
  │                      .zip  (1 GB)                          │
  │  Clips used        : 50 normal (non-theft) video clips     │
  │  Stored at         : data\raw\Normal\                      │
  └─────────────────────────────────────────────────────────────┘

  Total clips collected  : 100  (50 shoplifting + 50 normal)
  Total storage used     : ~6 GB (raw videos)
  Class balance          : Perfectly balanced — 50/50 split

═══════════════════════════════════════════════════════════════════
WHY THIS DATASET?
═══════════════════════════════════════════════════════════════════

  1. Real surveillance footage — not acted or staged. Clips are
     genuine CCTV recordings from real-world environments.

  2. Explicit shoplifting class — unlike most video datasets
     which focus on fighting or accidents, UCF-Crime has a
     dedicated shoplifting category.

  3. Freely available — academic use licence, no cost, no
     special permissions needed beyond registration.

  4. Well established — widely cited in academic papers on
     anomaly detection and video classification.


═══════════════════════════════════════════════════════════════════
DATASET LIMITATIONS
═══════════════════════════════════════════════════════════════════

  - Only 50 clips per class — small by production standards
    (production systems typically use 500+ clips per class)

  - All footage from a single dataset source — model may
    struggle on footage from different cameras or environments
    (domain shift problem — see step5 training notes)

  - No variation in store type, lighting condition, or
    camera angle within training data


═══════════════════════════════════════════════════════════════════
PROJECT FOLDER STRUCTURE (after data collection)
═══════════════════════════════════════════════════════════════════

  Shoplifting_Project\
  │
  ├── data\
  │   └── raw\
  │       ├── Shoplifting\    ← 50 .mp4 clips (UCF-Crime)
  │       └── Normal\         ← 50 .mp4 clips (UCF-Crime)
  │
  ├── scripts\
  │   ├── step1_collect_dataset.py   ← this file (documentation)
  │   ├── step2_preprocess.py        ← extract frames
  │   ├── step3_dataset_loader.py    ← PyTorch dataloader
  │   ├── step4_model.py             ← R3D-18 model
  │   ├── step5_train.py             ← training loop
  │   ├── step6_export_model.py      ← export to ONNX/TorchScript
  │   ├── step7_inference.py         ← live inference + alarm
  │   └── step8_app.py               ← FastAPI web service
  │
  ├── checkpoints\
  │   └── best_model.pth             ← saved trained model
  │
  └── alerts\                        ← evidence clips saved here

═══════════════════════════════════════════════════════════════════
"""
