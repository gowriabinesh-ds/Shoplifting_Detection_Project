"""
STEP 6: EXPORT MODEL TO ONNX
==============================
Converts the trained best_model.pth into ONNX format.
ONNX can run on any computer without needing PyTorch installed.
"""

import torch
from pathlib import Path
from step4_model import get_model

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR    = Path(r"C:\Hope AI\11. Deep Learning\Shoplifting_Project")
CHECKPOINT     = PROJECT_DIR / "checkpoints" / "best_model.pth"
EXPORT_DIR     = PROJECT_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

ONNX_PATH = EXPORT_DIR / "shoplifting_detector.onnx"
TS_PATH   = EXPORT_DIR / "shoplifting_detector_torchscript.pt"


def load_trained_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"\n❌ Checkpoint not found: {CHECKPOINT}\n"
            "Please run step5_train.py first!"
        )

    ckpt  = torch.load(str(CHECKPOINT), map_location="cpu")
    model = get_model(ckpt["config"]["num_classes"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"✅ Loaded model from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.3f})")
    return model


def export_onnx(model):
    # R3D-18 uses 112x112 input (not 224)
    dummy = torch.randn(1, 3, 16, 112, 112)

    torch.onnx.export(
        model,
        dummy,
        str(ONNX_PATH),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["video_clip"],
        output_names=["class_logits"],
        dynamic_axes={
            "video_clip":   {0: "batch_size"},
            "class_logits": {0: "batch_size"},
        },
    )
    print(f"✅ ONNX model saved: {ONNX_PATH}")


def export_torchscript(model):
    example  = torch.randn(1, 3, 16, 112, 112)
    scripted = torch.jit.trace(model, example)
    scripted.save(str(TS_PATH))
    print(f"✅ TorchScript model saved: {TS_PATH}")


def verify_onnx():
    try:
        import onnxruntime as ort
        import numpy as np

        session  = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
        dummy_np = np.random.randn(1, 3, 16, 112, 112).astype(np.float32)
        outputs  = session.run(None, {"video_clip": dummy_np})
        print(f"✅ ONNX verified — output shape: {outputs[0].shape}")
    except ImportError:
        print("onnxruntime not installed — skipping verification")
        print("To verify: pip install onnxruntime")


if __name__ == "__main__":
    model = load_trained_model()
    export_onnx(model)
    export_torchscript(model)
    verify_onnx()

    print("\nExported files:")
    for f in EXPORT_DIR.iterdir():
        size_mb = f.stat().st_size / 1_048_576
        print(f"  {f.name}  ({size_mb:.1f} MB)")
