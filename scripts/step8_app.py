"""
STEP 8: FASTAPI WEB SERVICE WITH ALERT SOUND
=============================================
Creates a web app where users upload videos and get shoplifting prediction.

To run:
  In Anaconda Prompt (shoplifting environment active):
  cd "C:\Hope AI\11. Deep Learning\Shoplifting_Project\scripts"
  uvicorn step8_app:app --host 0.0.0.0 --port 8000
 
Then open browser: http://localhost:8000
"""
 
import os
import cv2
import torch
import numpy as np
import tempfile                 # Temporary file storage
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException  # Web API framework
from fastapi.responses import JSONResponse, HTMLResponse      # API responses
import torchvision.transforms as T
 
from step4_model import get_model  # Load model architecture
 
# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(r"C:\Hope AI\11. Deep Learning\Shoplifting_Project") # Project folder
CHECKPOINT  = PROJECT_DIR / "checkpoints" / "best_model.pth"            # Trained model path
 
# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Shoplifting Detection API", # API name
    description="Upload a CCTV video clip to detect shoplifting.", 
    version="1.0.0",
)

# ── Settings ─────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 16
IMG_SIZE   = 112
 
# ── Load model once when app starts ─────────────────
print(f"Loading model on {DEVICE}...")
if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}\nRun step5_train.py first!")
 
ckpt   = torch.load(str(CHECKPOINT), map_location=DEVICE)    # Load saved model
_model = get_model(ckpt["config"]["num_classes"]).to(DEVICE) # Recreate model
_model.load_state_dict(ckpt["model_state"])                  # Load weights
_model.eval()                                                # Set model to evaluation mode
print("✅ Model ready.")


# ── Preprocessing ─────────────────────────────────── 
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
 
 
# ── Helper: extract frames from uploaded video ────────────────────────────────
def extract_frames_from_bytes(video_bytes: bytes) -> list:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
 
    cap   = cv2.VideoCapture(tmp_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
    if total == 0:
        cap.release()
        os.unlink(tmp_path)
        raise ValueError("Could not read frames from video.")
 
    indices = np.linspace(0, total - 1, min(NUM_FRAMES, total), dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
 
    cap.release()
    os.unlink(tmp_path)
    return frames
 
 
# ── Helper: run model prediction ──────────────────────────────────────────────
@torch.no_grad()
def run_prediction(frames: list) -> dict:
    tensors = [transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    clip    = torch.stack(tensors, dim=0).permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)
    logits  = _model(clip)
    probs   = torch.softmax(logits, dim=1)[0].cpu().tolist()
    label   = int(np.argmax(probs))
 
    return {
        "label":            "shoplifting" if label == 1 else "normal",
        "confidence":       round(probs[label], 4),
        "prob_normal":      round(probs[0], 4),
        "prob_shoplifting": round(probs[1], 4),
        "alert":            label == 1 and probs[1] >= 0.75,
    }
 
 
# ── Main dashboard page with sound alert ─────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard():
    """
    Returns a clean HTML dashboard page.
    Open http://localhost:8000 in your browser.
    When shoplifting is detected with alert=true, a beep plays automatically.
    """
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Shoplifting Detection System</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
 
    body {
      font-family: 'Segoe UI', sans-serif;
      background: #0d1117;
      color: #e6edf3;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 40px 20px;
    }
 
    .header {
      text-align: center;
      margin-bottom: 40px;
    }
 
    .header h1 {
      font-size: 2rem;
      font-weight: 700;
      color: #e8a020;
      letter-spacing: 2px;
    }
 
    .header p {
      color: #8896a5;
      margin-top: 8px;
      font-size: 0.95rem;
    }
 
    .card {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 32px;
      width: 100%;
      max-width: 600px;
      margin-bottom: 24px;
    }
 
    .upload-area {
      border: 2px dashed #30363d;
      border-radius: 8px;
      padding: 40px;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.2s;
      margin-bottom: 20px;
    }
 
    .upload-area:hover { border-color: #e8a020; }
 
    .upload-area input { display: none; }
 
    .upload-icon { font-size: 2.5rem; margin-bottom: 12px; }
 
    .upload-area p { color: #8896a5; font-size: 0.9rem; }
 
    .upload-area .filename {
      color: #e8a020;
      font-weight: 600;
      margin-top: 8px;
      font-size: 0.95rem;
    }
 
    .btn {
      width: 100%;
      padding: 14px;
      background: #e8a020;
      color: #0d1117;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      letter-spacing: 1px;
      transition: background 0.2s;
    }
 
    .btn:hover { background: #f5c842; }
    .btn:disabled { background: #30363d; color: #8896a5; cursor: not-allowed; }
 
    .result-card {
      display: none;
      border-radius: 12px;
      padding: 28px;
      width: 100%;
      max-width: 600px;
      text-align: center;
      margin-bottom: 24px;
      animation: fadeIn 0.4s ease;
    }
 
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
 
    .result-normal {
      background: #0d2818;
      border: 2px solid #27ae60;
    }
 
    .result-shoplifting {
      background: #2d0a0a;
      border: 2px solid #c0392b;
      animation: fadeIn 0.4s ease, flashRed 0.5s ease 3;
    }
 
    @keyframes flashRed {
      0%   { box-shadow: 0 0 0 0 rgba(192,57,43,0); }
      50%  { box-shadow: 0 0 30px 10px rgba(192,57,43,0.6); }
      100% { box-shadow: 0 0 0 0 rgba(192,57,43,0); }
    }
 
    .result-icon { font-size: 3.5rem; margin-bottom: 12px; }
 
    .result-label {
      font-size: 1.6rem;
      font-weight: 700;
      margin-bottom: 16px;
      letter-spacing: 2px;
    }
 
    .result-normal .result-label  { color: #27ae60; }
    .result-shoplifting .result-label { color: #e74c3c; }
 
    .confidence-bar-wrap {
      background: #30363d;
      border-radius: 100px;
      height: 10px;
      margin: 10px 0 6px;
      overflow: hidden;
    }
 
    .confidence-bar {
      height: 100%;
      border-radius: 100px;
      transition: width 0.8s ease;
    }
 
    .bar-normal      { background: #27ae60; }
    .bar-shoplifting { background: #e74c3c; }
 
    .confidence-label {
      font-size: 0.85rem;
      color: #8896a5;
      display: flex;
      justify-content: space-between;
    }
 
    .stats-row {
      display: flex;
      gap: 16px;
      margin-top: 20px;
    }
 
    .stat-box {
      flex: 1;
      background: #0d1117;
      border-radius: 8px;
      padding: 14px;
      text-align: center;
    }
 
    .stat-box .val {
      font-size: 1.4rem;
      font-weight: 700;
      color: #e8a020;
    }
 
    .stat-box .lbl {
      font-size: 0.75rem;
      color: #8896a5;
      margin-top: 4px;
    }
 
    .alert-badge {
      display: inline-block;
      padding: 8px 24px;
      border-radius: 100px;
      font-weight: 700;
      font-size: 0.9rem;
      letter-spacing: 1px;
      margin-top: 16px;
    }
 
    .badge-alert    { background: #c0392b; color: white; }
    .badge-safe     { background: #27ae60; color: white; }
 
    .loader {
      display: none;
      text-align: center;
      padding: 20px;
      color: #8896a5;
    }
 
    .spinner {
      border: 3px solid #30363d;
      border-top: 3px solid #e8a020;
      border-radius: 50%;
      width: 36px; height: 36px;
      animation: spin 0.8s linear infinite;
      margin: 0 auto 12px;
    }
 
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
 
  <div class="header">
    <h1>SHOPLIFTING DETECTION</h1>
    <p>Upload a CCTV video clip — AI analyses motion across 16 frames</p>
  </div>
 
  <div class="card">
    <div class="upload-area" onclick="document.getElementById('fileInput').click()">
      <div class="upload-icon">📹</div>
      <p>Click to choose a video file</p>
      <p>Supports .mp4 · .avi · .mov</p>
      <div class="filename" id="filename">No file selected</div>
      <input type="file" id="fileInput" accept=".mp4,.avi,.mov,.mkv">
    </div>
    <button class="btn" id="analyseBtn" onclick="analyse()" disabled>ANALYSE VIDEO</button>
  </div>
 
  <div class="loader" id="loader">
    <div class="spinner"></div>
    <p>Analysing video...</p>
  </div>
 
  <div class="result-card" id="resultCard">
    <div class="result-icon" id="resultIcon"></div>
    <div class="result-label" id="resultLabel"></div>
 
    <div style="text-align:left; margin-top: 16px;">
      <div style="font-size:0.85rem; color:#8896a5; margin-bottom:4px;">Shoplifting probability</div>
      <div class="confidence-bar-wrap">
        <div class="confidence-bar" id="confBar" style="width:0%"></div>
      </div>
      <div class="confidence-label">
        <span>0%</span>
        <span id="confPct"></span>
        <span>100%</span>
      </div>
    </div>
 
    <div class="stats-row">
      <div class="stat-box">
        <div class="val" id="statConf"></div>
        <div class="lbl">Confidence</div>
      </div>
      <div class="stat-box">
        <div class="val" id="statNormal"></div>
        <div class="lbl">Prob Normal</div>
      </div>
      <div class="stat-box">
        <div class="val" id="statShoplift"></div>
        <div class="lbl">Prob Shoplifting</div>
      </div>
    </div>
 
    <div id="alertBadge" class="alert-badge"></div>
  </div>
 
<script>
  // ── File selection ──────────────────────────────────────────────────────────
  document.getElementById('fileInput').addEventListener('change', function() {
    const file = this.files[0];
    if (file) {
      document.getElementById('filename').textContent = file.name;
      document.getElementById('analyseBtn').disabled = false;
    }
  });
 
  // ── Play beep sound using Web Audio API (no external files needed) ──────────
  function playAlarmBeep() {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
 
      // Play 3 short sharp beeps
      [0, 0.35, 0.7].forEach(startTime => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
 
        osc.connect(gain);
        gain.connect(ctx.destination);
 
        osc.type = 'square';          // Sharp alarm-style tone
        osc.frequency.value = 880;    // High pitch — 880 Hz
 
        gain.gain.setValueAtTime(0.4, ctx.currentTime + startTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + startTime + 0.25);
 
        osc.start(ctx.currentTime + startTime);
        osc.stop(ctx.currentTime + startTime + 0.25);
      });
    } catch(e) {
      console.log('Audio not available:', e);
    }
  }
 
  // ── Main analyse function ───────────────────────────────────────────────────
  async function analyse() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) return;
 
    // Show loader, hide previous result
    document.getElementById('loader').style.display = 'block';
    document.getElementById('resultCard').style.display = 'none';
    document.getElementById('analyseBtn').disabled = true;
 
    const formData = new FormData();
    formData.append('file', file);
 
    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData
      });
 
      const data = await response.json();
      document.getElementById('loader').style.display = 'none';
      showResult(data);
 
    } catch (err) {
      document.getElementById('loader').style.display = 'none';
      alert('Error: ' + err.message);
      document.getElementById('analyseBtn').disabled = false;
    }
  }
 
  // ── Display result ──────────────────────────────────────────────────────────
  function showResult(data) {
    const card      = document.getElementById('resultCard');
    const isShoplift = data.label === 'shoplifting';
    const pct       = Math.round(data.prob_shoplifting * 100);
 
    // Set card style
    card.className = 'result-card ' + (isShoplift ? 'result-shoplifting' : 'result-normal');
 
    document.getElementById('resultIcon').textContent  = isShoplift ? '🚨' : '✅';
    document.getElementById('resultLabel').textContent = isShoplift ? 'SHOPLIFTING DETECTED' : 'NORMAL ACTIVITY';
 
    // Confidence bar
    setTimeout(() => {
      document.getElementById('confBar').style.width = pct + '%';
      document.getElementById('confBar').className   = 'confidence-bar ' + (isShoplift ? 'bar-shoplifting' : 'bar-normal');
    }, 100);
 
    document.getElementById('confPct').textContent     = pct + '%';
    document.getElementById('statConf').textContent    = (data.confidence * 100).toFixed(1) + '%';
    document.getElementById('statNormal').textContent  = (data.prob_normal * 100).toFixed(1) + '%';
    document.getElementById('statShoplift').textContent = (data.prob_shoplifting * 100).toFixed(1) + '%';
 
    // Alert badge
    const badge = document.getElementById('alertBadge');
    if (data.alert) {
      badge.textContent  = '🔴  ALERT TRIGGERED — Staff Notified';
      badge.className    = 'alert-badge badge-alert';
      playAlarmBeep();   // Play 3 beeps when alert fires
    } else {
      badge.textContent  = '🟢  NO ALERT — Below Threshold';
      badge.className    = 'alert-badge badge-safe';
    }
 
    card.style.display = 'block';
    document.getElementById('analyseBtn').disabled = false;
  }
</script>
 
</body>
</html>
"""
    return HTMLResponse(content=html)
 
 
# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}  # Check if API is running
 
 
# ── Prediction API ──────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Please upload .mp4, .avi or .mov file.")  # Reject unsupported files
 
    video_bytes = await file.read()  # Read uploaded file
 
    try:
        frames = extract_frames_from_bytes(video_bytes)   # Extract frames
        if len(frames) < 4:
            raise HTTPException(status_code=422, detail="Video too short to analyse.")
        result = run_prediction(frames)      # Run model
        result["filename"] = file.filename   # Add filename
        return JSONResponse(content=result)  # Return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # Handle errors
 
# ── Run API server ──────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Start FastAPI server