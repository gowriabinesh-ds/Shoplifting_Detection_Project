"""
STEP 5: TRAINING
=================
Trains the model, saves the best version, and shows final results.
"""

import os
import csv
import time                    # Track time per epoch
import torch
import torch.nn as nn          # Neural network functions
import torch.optim as optim    # Optimizers (model learning)
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

from step3_dataset_loader import get_dataloaders   # Load dataset
from step4_model import get_model                  # Load model

# ── Config ─────────────────────────────────────────────────────────────────────
PROJECT_DIR     = Path(r"C:\Hope AI\11. Deep Learning\Shoplifting_Project")   # Project path
CHECKPOINT_DIR  = PROJECT_DIR / "checkpoints"                                 # Folder to save models
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)                             # Create folder if not exists

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"       # Path to save best model
LOG_PATH        = CHECKPOINT_DIR / "training_log.csv"     # Path to save training logs

BATCH_SIZE   = 2       # Number of samples per batch
NUM_WORKERS  = 0       # Number of parallel workers (0 for Windows)
EPOCHS       = 20      # Total training cycles
LR           = 3e-4    # Learning rate (how fast model learns)
WEIGHT_DECAY = 1e-4    # Regularization to prevent overfitting
PATIENCE     = 7       # Early stopping patience
NUM_CLASSES  = 2       # Number of output classes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # Use GPU if available


# ── Compute class weights ──────────────────────────────────────────────────────
def compute_class_weights(loaders):
    counts = torch.zeros(NUM_CLASSES)      # Count samples per class
    for _, labels in loaders["train"]:     # Loop through training data
        for lbl in labels:
            counts[lbl.item()] += 1        # Count each label
    weights = counts.sum() / (NUM_CLASSES * counts)   # Give higher weight to minority class
    print(f"Class weights: normal={weights[0]:.3f}, shoplifting={weights[1]:.3f}")
    return weights.to(DEVICE)    # Move weights to device


# ── One training epoch ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scheduler):
    model.train()   # Set model to training mode
    total_loss, correct, total = 0.0, 0, 0   # Initialize metrics

    for clips, labels in loader:   # Loop through batches
        clips, labels = clips.to(DEVICE), labels.to(DEVICE)   # Move to GPU/CPU

        optimizer.zero_grad()                    # Clear previous gradients
        outputs = model(clips)                   # Forward pass (prediction)
        loss    = criterion(outputs, labels)     # Compute loss
        loss.backward()                          # Backpropagation (learn from error)

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # Prevent exploding gradients

        optimizer.step()  # Update model weights
        scheduler.step()  # Adjust learning rate

        total_loss += loss.item() * clips.size(0)      # Accumulate loss
        preds       = outputs.argmax(dim=1)            # Get predicted class
        correct    += (preds == labels).sum().item()   # Count correct predictions
        total      += clips.size(0)                    # Total samples

    return total_loss / total, correct / total         # Return avg loss & accuracy


# ── Evaluation (val/test) ─────────────────────────────────────────────────────
@torch.no_grad()   # Disable gradient calculation (faster)
def evaluate(model, loader, criterion):
    model.eval()   # Set model to evaluation mode
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for clips, labels in loader:
        clips, labels = clips.to(DEVICE), labels.to(DEVICE)
        outputs = model(clips)   # Forward pass
        loss    = criterion(outputs, labels)  # Compute loss
        preds   = outputs.argmax(dim=1)   # Get predictions

        total_loss += loss.item() * clips.size(0)
        correct    += (preds == labels).sum().item()
        total      += clips.size(0)
        all_preds.extend(preds.cpu().tolist())    # Store predictions
        all_labels.extend(labels.cpu().tolist())  # Store true labels

    return total_loss / total, correct / total, all_preds, all_labels


# ── Main training loop ─────────────────────────────────────────────────────────
def train():
    print(f"Training on: {DEVICE}")    # Show device
    print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}\n")

    loaders = get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)  # Load data
    model   = get_model(NUM_CLASSES).to(DEVICE)  # Load model

    class_weights = compute_class_weights(loaders)  # Handle imbalance
    criterion     = nn.CrossEntropyLoss(weight=class_weights)  # Loss function with class weights

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # Train only unfrozen layers
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(loaders["train"]),
        epochs=EPOCHS,
    )                              # Adjust learning rate dynamically

    log_file   = open(str(LOG_PATH), "w", newline="")   # Open log file
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])   # Write header

    
    best_val_acc     = 0.0  # Track best validation accuracy
    patience_counter = 0    # Track early stopping

    print("Starting training...\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()   # Start timer

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scheduler
        )  # Train model
        
        val_loss, val_acc, _, _ = evaluate(model, loaders["val"], criterion)  # Validate model

        elapsed = time.time() - t0   # Time taken
        
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f} | "
            f"{elapsed:.1f}s"
        )

        log_writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])  # Save logs

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

    checkpoint = torch.load(str(BEST_MODEL_PATH), map_location=DEVICE)   # Load best model
    model.load_state_dict(checkpoint["model_state"])                     # Restore weights
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

# ── Run training ─────────────────────────────────────
if __name__ == "__main__":
    train()
