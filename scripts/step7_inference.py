"""
STEP 7: REAL-TIME INFERENCE
=============================
Three modes available — can change SOURCE at the bottom:

  source = 0                        → live webcam
  source = NORMAL_VIDEO             → play normal sample video
  source = SHOPLIFTING_VIDEO        → play shoplifting sample video
  
When shoplifting is detected:
  - Red bounding box drawn around the frame
  - "SHOPLIFTING DETECTED" text with confidence shown
  - 3 beep alarm plays automatically
  - Evidence clip saved to alerts folder

Press Q in the video window to quit.

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


# ── Sample video paths ──────────────────────────────────────────────────────
SHOPLIFTING_VIDEO = str(PROJECT_DIR / "Sample_Video_Shoplifting.mp4")
NORMAL_VIDEO      = str(PROJECT_DIR / "Sample_Video_Normal.mp4")


# ── Settings ───────────────────────────────────────────────────────────────────
NUM_FRAMES      = 16     # Frames per model input
STEP            = 8      # Run model every 8 frames (sliding window)
IMG_SIZE        = 112    # Must match training size for R3D-18
CONF_THRESHOLD  = 0.60   # Alarm if confidence >= 60%
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

# ── Draw overlay on frame ──────────────────────────────────────────────────────
def draw_overlay(frame, label, conf, alert_flash):
    """
    Normal   → small green label top left
    Shoplifting → red bounding box around entire frame
                  large red alert text
                  flashing red border
    """
    h, w  = frame.shape[:2]
    color = LABEL_COLORS[label]

    if label == 0:
        # ── NORMAL — clean green label only ───────────────────
        cv2.rectangle(frame, (0, 0), (280, 45), (0, 0, 0), -1)
        cv2.putText(frame,
                    f"Normal  {conf:.0%}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (0, 200, 0), 2, cv2.LINE_AA)

    else:
        # ── SHOPLIFTING DETECTED ───────────────────────────────

        # 1. Thick red bounding box around entire frame
        thickness = 6
        cv2.rectangle(frame,
                      (thickness, thickness),
                      (w - thickness, h - thickness),
                      (0, 0, 255), thickness)

        # 2. Dark semi-transparent banner at top
        banner = frame.copy()
        cv2.rectangle(banner, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.addWeighted(banner, 0.7, frame, 0.3, 0, frame)

        # 3. "SHOPLIFTING DETECTED" text
        cv2.putText(frame,
                    "SHOPLIFTING DETECTED",
                    (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.75, (0, 0, 255), 2, cv2.LINE_AA)

        # 4. Confidence text below
        cv2.putText(frame,
                    f"Confidence: {conf:.1%}",
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. Flashing alert badge at bottom centre
        if alert_flash:
            badge_w = 320
            badge_x = (w - badge_w) // 2
            badge_y = h - 50
            cv2.rectangle(frame,
                          (badge_x, badge_y),
                          (badge_x + badge_w, badge_y + 38),
                          (0, 0, 200), -1)
            cv2.rectangle(frame,
                          (badge_x, badge_y),
                          (badge_x + badge_w, badge_y + 38),
                          (0, 0, 255), 2)
            cv2.putText(frame,
                        "!! ALERT — STAFF NOTIFIED !!",
                        (badge_x + 12, badge_y + 26),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


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
    alert_end   = 0       # When to stop showing alert badge

    print("Running... Press Q in the video window to quit.")

    while True:
        ret, frame = cap.read()

        # Loop video when it ends — keeps demo running continuously
        if not ret:
            if source != 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                buffer.clear()
                frame_count = 0
                continue
            else:
                print("Camera disconnected.")
                break

        # Upscale small videos so the overlay is more visible
        fh, fw = frame.shape[:2]
        if fw < 640:
            frame = cv2.resize(frame, (fw * 2, fh * 2))    
    
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
                    alert_end  = now + 4.0    # Show alert badge for 4 seconds
                    print(f"🚨 SHOPLIFTING DETECTED — confidence: {conf:.1%}")
                    play_alert_sound()          # Play alarm sound
                    save_alert_clip(list(buffer), fps)

        # Check if alert badge should still be showing
        alert_flash = time.time() < alert_end

        # Draw overlay
        display = draw_overlay(frame.copy(), last_label, last_conf, alert_flash)

        cv2.imshow("Shoplifting Detector", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

# ── Run program ─────────────────────────────────────
# source = 0                  → live webcam
# source = NORMAL_VIDEO       → sample normal video
# source = SHOPLIFTING_VIDEO  → sample shoplifting video
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    run_inference(source=SHOPLIFTING_VIDEO)
