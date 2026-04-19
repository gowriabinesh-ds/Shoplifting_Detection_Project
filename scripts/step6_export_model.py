"""
STEP 6: EXPORT MODEL TO ONNX
==============================
Converts the trained best_model.pth into ONNX format.
ONNX can run on any computer without needing PyTorch installed.
"""

import torch
from pathlib import Path
from step4_model import get_model   # Load model architecture

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR    = Path(r"C:\Hope AI\11. Deep Learning\Shoplifting_Project")    # Main project folder
CHECKPOINT     = PROJECT_DIR / "checkpoints" / "best_model.pth"               # Path to trained model
EXPORT_DIR     = PROJECT_DIR / "exports"                                      # Folder for exported models
EXPORT_DIR.mkdir(parents=True, exist_ok=True)                                 # Create folder if not exists

ONNX_PATH = EXPORT_DIR / "shoplifting_detector.onnx"                    # ONNX model path
TS_PATH   = EXPORT_DIR / "shoplifting_detector_torchscript.pt"          # TorchScript model path


def load_trained_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"\n❌ Checkpoint not found: {CHECKPOINT}\n"
            "Please run step5_train.py first!"
        )                                                      # Stop if model file is missing

    ckpt  = torch.load(str(CHECKPOINT), map_location="cpu")    # Load saved model checkpoint
    model = get_model(ckpt["config"]["num_classes"])           # Recreate model architecture
    model.load_state_dict(ckpt["model_state"])                 # Load trained weights into model
    model.eval()                                               # Set model to evaluation mode (disable training behavior)
    print(f"✅ Loaded model from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.3f})")  # Show model info
    return model  # Return ready model


def export_onnx(model):
    # R3D-18 uses 112x112 input (not 224)
    dummy = torch.randn(1, 3, 16, 112, 112)  # Create dummy input (same shape as real data)

    torch.onnx.export(
        model,                                     # Model to export
        dummy,                                     # Example input
        str(ONNX_PATH),                            # Save path
        export_params=True,                        # Include trained weights
        opset_version=17,                          # ONNX version
        do_constant_folding=True,                  # Optimize model graph
        input_names=["video_clip"],                # Input name
        output_names=["class_logits"],             # Output name
        dynamic_axes={
            "video_clip":   {0: "batch_size"},     # Allow variable batch size
            "class_logits": {0: "batch_size"},
        },
    )
    print(f"✅ ONNX model saved: {ONNX_PATH}")    # Confirm save


def export_torchscript(model):
    example  = torch.randn(1, 3, 16, 112, 112)       # Example input for tracing
    scripted = torch.jit.trace(model, example)       # Convert model into TorchScript format
    scripted.save(str(TS_PATH))                      # Save TorchScript model
    print(f"✅ TorchScript model saved: {TS_PATH}")  # Confirm save


def verify_onnx():
    try:
        import onnxruntime as ort   # ONNX runtime for testing
        import numpy as np

        session  = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])  # Load ONNX model
        dummy_np = np.random.randn(1, 3, 16, 112, 112).astype(np.float32)                    # Create dummy input
        outputs  = session.run(None, {"video_clip": dummy_np})                               # Run inference
        print(f"✅ ONNX verified — output shape: {outputs[0].shape}")                       # Confirm model works
    except ImportError:
        print("onnxruntime not installed — skipping verification")   # Skip if library not installed
        print("To verify: pip install onnxruntime")

# ── Run export ────────────────────────────────────────
if __name__ == "__main__":
    model = load_trained_model()  # Load trained model
    export_onnx(model)            # Convert to ONNX
    export_torchscript(model)     # Convert to TorchScript
    verify_onnx()                 # Test ONNX model

    print("\nExported files:")
    for f in EXPORT_DIR.iterdir():                 # Loop through exported files
        size_mb = f.stat().st_size / 1_048_576     # Convert size to MB
        print(f"  {f.name}  ({size_mb:.1f} MB)")   # Show file info
