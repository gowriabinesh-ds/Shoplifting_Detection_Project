"""
STEP 5: TRAINING
=================
Trains the model, saves the best version, and shows final results.
"""

import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

from step3_dataset_loader import get_dataloaders
from step4_model import get_model

# ── Config ─────────────────────────────────────────────────────────────────────
PROJECT_DIR     = Path(r"C:\Hope AI\11. Deep Learning\Shoplifting_Project")
CHECKPOINT_DIR  = PROJECT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"   # One name used everywhere
LOG_PATH        = CHECKPOINT_DIR / "training_log.csv"

BATCH_SIZE   = 2
NUM_WORKERS  = 0
EPOCHS       = 20
LR           = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 7       # Increased — gives model more time before stopping
NUM_CLASSES  = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Compute class weights ──────────────────────────────────────────────────────
def compute_class_weights(loaders):
    counts = torch.zeros(NUM_CLASSES)
    for _, labels in loaders["train"]:
        for lbl in labels:
            counts[lbl.item()] += 1
    weights = counts.sum() / (NUM_CLASSES * counts)
    print(f"Class weights: normal={weights[0]:.3f}, shoplifting={weights[1]:.3f}")
    return weights.to(DEVICE)


# ── One training epoch ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scheduler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for clips, labels in loader:
        clips, labels = clips.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(clips)
        loss    = criterion(outputs, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * clips.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += clips.size(0)

    return total_loss / total, correct / total


# ── Validation / test pass ─────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for clips, labels in loader:
        clips, labels = clips.to(DEVICE), labels.to(DEVICE)
        outputs = model(clips)
        loss    = criterion(outputs, labels)
        preds   = outputs.argmax(dim=1)

        total_loss += loss.item() * clips.size(0)
        correct    += (preds == labels).sum().item()
        total      += clips.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


# ── Main training loop ─────────────────────────────────────────────────────────
def train():
    print(f"Training on: {DEVICE}")
    print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}\n")

    loaders = get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    model   = get_model(NUM_CLASSES).to(DEVICE)

    class_weights = compute_class_weights(loaders)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(loaders["train"]),
        epochs=EPOCHS,
    )

    log_file   = open(str(LOG_PATH), "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    # Track best by ACCURACY — more stable than loss for small datasets
    best_val_acc     = 0.0
    patience_counter = 0

    print("Starting training...\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scheduler
        )
        val_loss, val_acc, _, _ = evaluate(model, loaders["val"], criterion)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f} | "
            f"{elapsed:.1f}s"
        )

        log_writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        # Save best model based on validation ACCURACY
        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0

            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "val_acc":     val_acc,
                "config": {
                    "arch":        "r3d",
                    "num_classes": NUM_CLASSES,
                },
            }, str(BEST_MODEL_PATH))           # Always saves as best_model.pth

            print(f"  ✅ Best model saved (val_acc = {val_acc:.3f})")

        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"\n⏹ Early stopping at epoch {epoch}.")
                break

    log_file.close()

    # ── Final test evaluation ──────────────────────────────────────────────────
    print("\n" + "="*50)
    print("FINAL TEST EVALUATION")
    print("="*50)

    if not BEST_MODEL_PATH.exists():
        print("❌ No saved model found — something went wrong during training.")
        return

    checkpoint = torch.load(str(BEST_MODEL_PATH), map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded best model from epoch {checkpoint['epoch']} "
          f"(val_acc={checkpoint['val_acc']:.3f})\n")

    _, test_acc, test_preds, test_labels = evaluate(model, loaders["test"], criterion)

    print(f"Test Accuracy: {test_acc:.4f}\n")
    print(classification_report(
        test_labels, test_preds,
        target_names=["Normal", "Shoplifting"]
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))
    print(f"\n✅ Best model saved at : {BEST_MODEL_PATH}")
    print(f"✅ Training log saved at: {LOG_PATH}")


if __name__ == "__main__":
    train()
