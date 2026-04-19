"""
STEP 7: REAL-TIME INFERENCE
=============================
"""

import cv2
import time
import torch
import threading
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime
import torchvision.transforms as T

from step4_model import get_model

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR   = Path(r"C:\Hope AI\11. Deep Learning\Shoplifting_Project")
CHECKPOINT    = PROJECT_DIR / "checkpoints" / "best_model.pth"
ALERTS_DIR    = PROJECT_DIR / "alerts"
ALERTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Settings ───────────────────────────────────────────────────────────────────
NUM_FRAMES      = 16     # Frames per model input
STEP            = 8      # Run model every 8 frames (sliding window)
IMG_SIZE        = 112    # Must match training size for R3D-18
CONF_THRESHOLD  = 0.75   # Fire alarm if confidence >= 75%
ALARM_COOLDOWN  = 15     # Seconds between alarms (avoid spam)

LABEL_NAMES  = {0: "Normal", 1: "SHOPLIFTING DETECTED"}
LABEL_COLORS = {0: (0, 200, 0), 1: (0, 0, 255)}   # Green / Red

# ── Preprocessing (same as validation pipeline) ────────────────────────────────
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Alert sound ────────────────────────────────────────────────────────────────
def play_alert_sound():
    """Plays 3 beeps when shoplifting is detected. Runs in background so video doesn't freeze."""
    def _beep():
        try:
            import winsound
            for _ in range(3):
                winsound.Beep(1000, 350)  # 1000Hz tone, 350ms
                time.sleep(0.15)
        except Exception as e:
            print(f"  (Sound unavailable: {e})")
    threading.Thread(target=_beep, daemon=True).start()


def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"\n❌ Model not found: {CHECKPOINT}\n"
            "Please run step5_train.py first!"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(str(CHECKPOINT), map_location=device)
    model  = get_model(ckpt["config"]["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"✅ Model loaded on {device}")
    return model, device


@torch.no_grad()
def predict(frames_list, model, device):
    tensors = [transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_list]
    clip    = torch.stack(tensors, dim=0).permute(1, 0, 2, 3).unsqueeze(0).to(device)
    logits  = model(clip)
    probs   = torch.softmax(logits, dim=1)[0]
    label   = probs.argmax().item()
    conf    = probs[label].item()
    return label, conf


def save_alert_clip(frames, fps=10.0):
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = str(ALERTS_DIR / f"alert_{ts}.avi")
    h, w     = frames[0].shape[:2]
    writer   = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps, (w, h)
    )
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"  🚨 Alert clip saved: {out_path}")


def run_inference(source=0):
    """
    source=0          → laptop webcam
    source="C:/..."   → path to a video file
    """
    model, device = load_model()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Cannot open video source: {source}")
        return

    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    buffer      = deque(maxlen=NUM_FRAMES)
    frame_count = 0
    last_alarm  = 0
    last_label  = 0
    last_conf   = 0.0

    print("Running... Press Q in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended or camera disconnected.")
            break

        buffer.append(frame.copy())
        frame_count += 1

        # Run model every STEP frames when buffer is full
        if len(buffer) == NUM_FRAMES and frame_count % STEP == 0:
            label, conf = predict(list(buffer), model, device)
            last_label, last_conf = label, conf

            # Trigger alarm
            if label == 1 and conf >= CONF_THRESHOLD:
                now = time.time()
                if now - last_alarm > ALARM_COOLDOWN:
                    last_alarm = now
                    print(f"🚨 SHOPLIFTING DETECTED — confidence: {conf:.1%}")
                    play_alert_sound()          # ← only new line added
                    save_alert_clip(list(buffer), fps)

        # Draw label on frame
        display = frame.copy()
        color   = LABEL_COLORS[last_label]
        cv2.rectangle(display, (0, 0), (400, 50), (0, 0, 0), -1)
        cv2.putText(
            display,
            f"{LABEL_NAMES[last_label]}  {last_conf:.0%}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, color, 2
        )

        cv2.imshow("Shoplifting Detector", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    # Set source = 0 for webcam
    # Set source = full path to a video file for recorded footage
    run_inference(source=0)
