"""
STEP 7: REAL-TIME INFERENCE
=============================
"""

import cv2
import time
import torch
import threading                    # Run sound in background
import numpy as np
from pathlib import Path
from collections import deque       # Store fixed number of frames (buffer)
from datetime import datetime       # Timestamp for saving alerts
import torchvision.transforms as T  # Image preprocessing

from step4_model import get_model   # Load model architecture

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR   = Path(r"C:\Hope AI\11. Deep Learning\Shoplifting_Project")  # Project folder
CHECKPOINT    = PROJECT_DIR / "checkpoints" / "best_model.pth"             # Trained model path
ALERTS_DIR    = PROJECT_DIR / "alerts"                                     # Folder to save alert clips
ALERTS_DIR.mkdir(parents=True, exist_ok=True)                              # Create folder if needed

# ── Settings ───────────────────────────────────────────────────────────────────
NUM_FRAMES      = 16     # Frames per model input
STEP            = 8      # Run model every 8 frames (sliding window)
IMG_SIZE        = 112    # Must match training size for R3D-18
CONF_THRESHOLD  = 0.75   # Fire alarm if confidence >= 75%
ALARM_COOLDOWN  = 15     # Time gap between alerts (seconds) to avoid spam

LABEL_NAMES  = {0: "Normal", 1: "SHOPLIFTING DETECTED"}  # Class labels
LABEL_COLORS = {0: (0, 200, 0), 1: (0, 0, 255)}          # Green (safe), Red (alert)

# ── Preprocessing (same as validation pipeline) ────────────────────────────────
transform = T.Compose([
    T.ToPILImage(),                            # Convert frame to PIL image
    T.Resize((IMG_SIZE, IMG_SIZE)),            # Resize to model input size
    T.ToTensor(),                              # Convert to tensor
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize pixel values
])


# ── Alert sound ────────────────────────────────────────────────────────────────
def play_alert_sound():
    """Plays 3 beeps when shoplifting is detected. Runs in background so video doesn't freeze."""
    def _beep():
        try:
            import winsound
            for _ in range(3):
                winsound.Beep(1000, 350)  # 1000Hz tone, 350ms - Play beep sound
                time.sleep(0.15)
        except Exception as e:
            print(f"  (Sound unavailable: {e})")
    threading.Thread(target=_beep, daemon=True).start()  # Run sound in background thread

# ── Load trained model ───────────────────────────────
def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"\n❌ Model not found: {CHECKPOINT}\n"
            "Please run step5_train.py first!"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"       # Use GPU if available
    ckpt   = torch.load(str(CHECKPOINT), map_location=device)     # Load saved model
    model  = get_model(ckpt["config"]["num_classes"]).to(device)  # Rebuild model
    model.load_state_dict(ckpt["model_state"])                    # Load weights
    model.eval()                                                  # Set model to evaluation mode
    print(f"✅ Model loaded on {device}")
    return model, device

# ── Prediction function ─────────────────────────────
@torch.no_grad()
def predict(frames_list, model, device):
    tensors = [transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_list]     # Convert frame to tensor
    clip    = torch.stack(tensors, dim=0).permute(1, 0, 2, 3).unsqueeze(0).to(device)  # Convert frames into model input shape (1, C, T, H, W)
    logits  = model(clip)                      # Run model
    probs   = torch.softmax(logits, dim=1)[0]  # Convert to probabilities
    label   = probs.argmax().item()            # Get predicted class
    conf    = probs[label].item()              # Get confidence
    return label, conf

# ── Save alert video clip ───────────────────────────
def save_alert_clip(frames, fps=10.0):
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")  # Create timestamp
    out_path = str(ALERTS_DIR / f"alert_{ts}.avi")       # Define output file path
    h, w     = frames[0].shape[:2]                       # Get frame size
    writer   = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps, (w, h)
    )                                                    # Create video writer
    for f in frames:
        writer.write(f)                                  # Write frames to video
    writer.release()                                     # Save file
    print(f"  🚨 Alert clip saved: {out_path}")

# ── Main real-time loop ─────────────────────────────
def run_inference(source=0):
    """
    source=0          → laptop webcam
    source="C:/..."   → path to a video file
    """
    model, device = load_model()      # Load model
    cap = cv2.VideoCapture(source)    # Open webcam or video file

    if not cap.isOpened():
        print(f"❌ Cannot open video source: {source}")
        return

    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0  # Get video FPS
    buffer      = deque(maxlen=NUM_FRAMES)           # Store last 16 frames
    frame_count = 0
    last_alarm  = 0
    last_label  = 0
    last_conf   = 0.0

    print("Running... Press Q in the video window to quit.")

    while True:
        ret, frame = cap.read()  # Read frame
        if not ret:
            print("Video ended or camera disconnected.")
            break              # Stop if video ends

        buffer.append(frame.copy())  # Store frame
        frame_count += 1

        # Run model every STEP frames when buffer is full
        if len(buffer) == NUM_FRAMES and frame_count % STEP == 0:
            label, conf = predict(list(buffer), model, device)  # Predict action
            last_label, last_conf = label, conf

            # Trigger alarm if shoplifting detected
            if label == 1 and conf >= CONF_THRESHOLD:
                now = time.time()
                if now - last_alarm > ALARM_COOLDOWN:
                    last_alarm = now
                    print(f"🚨 SHOPLIFTING DETECTED — confidence: {conf:.1%}")
                    play_alert_sound()          # Play alarm sound
                    save_alert_clip(list(buffer), fps)

        # Draw label on frame
        display = frame.copy()
        color   = LABEL_COLORS[last_label]
        cv2.rectangle(display, (0, 0), (400, 50), (0, 0, 0), -1)  # Draw background box
        cv2.putText(
            display,
            f"{LABEL_NAMES[last_label]}  {last_conf:.0%}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, color, 2
        )          # Show label and confidence

        cv2.imshow("Shoplifting Detector", display)  # Show video
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break  # Exit on pressing Q

    cap.release()              # Release camera
    cv2.destroyAllWindows()    # Close windows
    print("Done.")

# ── Run program ─────────────────────────────────────
if __name__ == "__main__":
    # Set source = 0 for webcam
    # Set source = full path to a video file for recorded footage
    run_inference(source=0)  # Start real-time detection
