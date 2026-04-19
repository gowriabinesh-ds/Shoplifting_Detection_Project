# 🚨 Shoplifting Detection System
### Real-Time AI-Powered CCTV Analysis for UK Retail Security

---

## 📌 Overview

UK retailers lose **£4.7 billion annually** to shoplifting. Traditional CCTV systems only record, they don't watch. Staff cannot monitor every aisle, and UK law limits physical intervention.

This project builds an **end-to-end deep learning pipeline** that:
- Analyses CCTV footage in real time using a 3D convolutional neural network
- Detects shoplifting behaviour (concealment gestures) across video frames
- Triggers an **instant audio alarm** when high-confidence detection occurs
- Serves predictions through a **live REST API dashboard**
- Saves evidence clips automatically for security review

> **Test Result:** 86.1% confidence detection on real UCF-Crime surveillance footage · 80% overall test accuracy

---

## 🎯 Demo

| Normal Activity | Shoplifting Detected |
|----------------|---------------------|
| ✅ Green label — no alarm | 🚨 Red label — beep alarm fires |
| Confidence bar low | Confidence bar fills red |
| Alert: `false` | Alert: `true` — evidence clip saved |

**API Response Example:**
```json
{
  "label": "shoplifting",
  "confidence": 0.861,
  "prob_normal": 0.139,
  "prob_shoplifting": 0.861,
  "alert": true,
  "filename": "Shoplifting001_x264.mp4"
}
```

---

## 🗂 Project Structure

```
Shoplifting_Project/
│
├── data/
│   ├── raw/
│   │   ├── Shoplifting/          # 50 UCF-Crime shoplifting clips
│   │   └── Normal/               # 50 UCF-Crime normal clips
│   ├── frames/                   # Extracted JPEG frames (post-preprocessing)
│   └── split_manifest.json       # Train/val/test split record
│
├── scripts/
│   ├── step1_collect_dataset.py  # Data collection documentation
│   ├── step2_preprocess.py       # Frame extraction + CLAHE enhancement
│   ├── step3_dataset_loader.py   # PyTorch Dataset + augmentation
│   ├── step4_model.py            # R3D-18 model definition
│   ├── step5_train.py            # Training loop + early stopping
│   ├── step6_export_model.py     # Export to ONNX + TorchScript
│   ├── step7_inference.py        # Live inference + alarm system
│   └── step8_app.py              # FastAPI web service + dashboard
│
├── checkpoints/
│   ├── best_model.pth            # Saved best model weights
│   └── training_log.csv          # Epoch-by-epoch accuracy/loss log
│
├── exports/
│   ├── shoplifting_detector.onnx # ONNX format (hardware-agnostic)
│   └── shoplifting_detector.pt   # TorchScript format
│
├── alerts/                       # Auto-saved evidence clips
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

```
Input (B, 3, 16, 112, 112)
        │
    ┌───▼────┐
    │  Stem  │  ← Frozen (Kinetics-400 pretrained)
    └───┬────┘
    ┌───▼────┐
    │Layer 1 │  ← Frozen
    └───┬────┘
    ┌───▼────┐
    │Layer 2 │  ← Trainable
    └───┬────┘
    ┌───▼────┐
    │Layer 3 │  ← Trainable
    └───┬────┘
    ┌───▼────┐
    │Layer 4 │  ← Trainable
    └───┬────┘
    ┌───▼─────┐
    │ AvgPool │
    └───┬─────┘
    ┌───▼──────────────┐
    │ FC Head (2-class)│  ← Custom — Normal / Shoplifting
    └──────────────────┘
```

**Why R3D-18?**
- 3D convolutions process **time AND space** simultaneously — understands motion, not just static frames
- Pretrained on **Kinetics-400** (400 human action classes) — motion priors transfer directly to shoplifting gestures
- Transfer learning allows strong performance with only 100 training clips
- Production-friendly — exportable to ONNX, runs on CPU

---

## 📊 Results

### Training Curve
| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 55.1% | 43.8% |
| 2 | 62.3% | **81.2%** ← Best saved |
| 3 | 66.7% | 68.8% |
| ... | ... | ... |
| 9 | 71.0% | 43.8% |

*Early stopping triggered at epoch 9 (patience=7)*

### Classification Report (Test Set)

```
              precision    recall  f1-score   support

      Normal       0.88      0.78      0.82         9
 Shoplifting       0.71      0.83      0.77         6

    accuracy                           0.80        15
   macro avg       0.79      0.81      0.80        15
weighted avg       0.81      0.80      0.80        15
```

### Confusion Matrix

```
                 Pred: Normal    Pred: Shoplifting
Actual Normal        7 ✓              2 ✗
Actual Shoplifting   1 ✗              5 ✓
```

**Test Accuracy: 80.0%** — trained on CPU, 100 clips, no GPU

---

## ⚙️ Technical Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Deep Learning | PyTorch 2.2 + torchvision |
| Model | R3D-18 (Kinetics-400 pretrained) |
| Video Processing | OpenCV |
| Augmentation | torchvision.transforms |
| API Framework | FastAPI + Uvicorn |
| Model Export | ONNX + TorchScript |
| Environment | Anaconda |
| Training Hardware | CPU (no GPU) |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/shoplifting-detection.git
cd shoplifting-detection
```

### 2. Create Conda Environment
```bash
conda create -n shoplifting python=3.11
conda activate shoplifting
```

### 3. Install Dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4. Download Dataset
Register and download from [UCF-Crime Dataset](https://www.crcv.ucf.edu/research/real-world-anomaly-detection/)

Place videos in:
```
data/raw/Shoplifting/    ← shoplifting clips
data/raw/Normal/         ← normal clips
```

### 5. Run the Pipeline
```bash
# Step 2 — Extract frames
python scripts/step2_preprocess.py

# Step 3 — Verify dataloader
python scripts/step3_dataset_loader.py

# Step 4 — Verify model
python scripts/step4_model.py

# Step 5 — Train (leave running 1-2 hours on CPU)
python scripts/step5_train.py

# Step 6 — Export model
python scripts/step6_export_model.py

# Step 7 — Live inference (webcam)
python scripts/step7_inference.py

# Step 8 — Launch web app
uvicorn step8_app:app --host 0.0.0.0 --port 8000
```

### 6. Open the Dashboard
```
http://localhost:8000
```

---

## 🌐 API Usage

**Start the server:**
```bash
conda activate shoplifting
cd scripts
uvicorn step8_app:app --host 0.0.0.0 --port 8000
```

**Upload a video via curl:**
```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@your_video.mp4"
```

**Response:**
```json
{
  "label": "shoplifting",
  "confidence": 0.861,
  "prob_normal": 0.139,
  "prob_shoplifting": 0.861,
  "alert": true,
  "filename": "your_video.mp4"
}
```

**Interactive dashboard:** `http://localhost:8000`
**Swagger UI docs:** `http://localhost:8000/docs`

---

## ⚠️ Limitations

| Limitation | Severity | Notes |
|-----------|---------|-------|
| Small dataset (100 clips) | High | Production systems need 500+ clips per class |
| CPU training only | Medium | GPU would allow VideoSwin Transformer |
| 22% false positive rate | Medium | Would reduce with more diverse training data |
| Single camera stream | Medium | Real deployment needs multi-camera support |
| Domain shift on compressed video | Medium | WhatsApp-compressed footage reduces accuracy |
| No GDPR compliance layer | Low | Required for UK production deployment |

---

## 🔮 Future Improvements

- [ ] Add Stealing class from UCF-Crime to increase training data to ~150 clips
- [ ] Integrate DCSASS dataset for additional theft footage diversity
- [ ] Upgrade to VideoSwin Transformer with GPU access for higher accuracy
- [ ] Build multi-camera async inference queue
- [ ] Implement GDPR-compliant pipeline (ICO guidelines — 30-day retention)
- [ ] Deploy on NVIDIA Jetson edge device for on-premise inference
- [ ] Add precision-recall threshold tuning to reduce false positive rate
- [ ] Integrate with store staff mobile alert system

---

## 📁 Dataset

| Dataset | Classes Used | Clips | Source |
|---------|-------------|-------|--------|
| UCF-Crime | Shoplifting, Normal | 50 + 50 | [UNC Charlotte](https://www.crcv.ucf.edu/research/real-world-anomaly-detection/) |

> **Note:** Dataset requires free academic registration. Videos are real CCTV surveillance footage — for research use only.

---

## 🛠 Requirements

```
torch>=2.2.0
torchvision>=0.17.0
opencv-python>=4.9.0
numpy>=1.26.0
scikit-learn>=1.4.0
tqdm>=4.66.0
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
Pillow>=10.3.0
```

---

## 📄 License

This project is for academic and educational purposes only.
The UCF-Crime dataset is subject to its own academic use licence — see the [dataset page](https://www.crcv.ucf.edu/research/real-world-anomaly-detection/) for terms.

---

## 🟨 Acknowledgements

- **UCF-Crime Dataset** — University of North Carolina Charlotte (Chen Chen et al.)
- **R3D-18 Architecture** — Hara et al., "Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?"
- **Kinetics-400** — DeepMind, for pretraining weights

---

<div align="center">

**Built with Python · PyTorch · FastAPI · OpenCV**

*Deep Learning Project — UK Retail Security*

</div>
