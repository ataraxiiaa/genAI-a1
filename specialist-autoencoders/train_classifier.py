"""
Training and Evaluation Pipeline for Task 2: 4-Class Corruption Classifier
Oxford-IIIT Pet Dataset (128x128 RGB)

Features:
- Balanced 4-class training loop (25% Clean, 25% Salt & Pepper, 25% Blur, 25% Occlusion)
- Comprehensive evaluation:
    * Multi-class Accuracy
    * Per-class Precision, Recall, and F1-score
    * Full 4x4 Confusion Matrix
- Checkpoint saving: Saves best model to checkpoints/best_corruption_classifier.pt
- Visual reporting: Generates annotated confusion matrix heatmap
- CSV logging: Automatically records epoch metrics for IEEE paper reporting
"""

import os
import sys
import argparse
import json
from typing import Dict, Tuple, List, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent and local directories to sys.path
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(PARENT_DIR, "universal-restoration"))

from classifier import CorruptionClassifier, CLASS_NAMES, SHORT_NAMES, NUM_CLASSES
from loader import get_dataloaders
try:
    from track_specialists_excel import append_classifier_epoch
except ImportError:
    append_classifier_epoch = None



def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Runs one training epoch and returns (mean_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Train Classifier]", leave=False)
    for corrupted, clean, labels, meta in pbar:
        corrupted = corrupted.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(corrupted)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * corrupted.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{(correct / total) * 100:.2f}%",
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluates the classifier on the validation set.
    Computes Loss, Overall Accuracy, 4x4 Confusion Matrix, and Per-Class Metrics.
    """
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0

    # 4x4 Confusion Matrix: rows = Ground Truth, cols = Prediction
    confusion_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for corrupted, clean, labels, meta in dataloader:
        corrupted = corrupted.to(device)
        labels = labels.to(device)

        logits = model(corrupted)
        loss = criterion(logits, labels)

        running_loss += loss.item() * corrupted.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        y_true = labels.cpu().numpy()
        y_pred = preds.cpu().numpy()
        for t, p in zip(y_true, y_pred):
            confusion_mat[t, p] += 1

    val_loss = running_loss / total
    overall_acc = correct / total

    # Compute Per-Class Precision, Recall, and F1
    per_class_metrics = {}
    precisions = []
    recalls = []
    f1s = []

    for c in range(NUM_CLASSES):
        tp = confusion_mat[c, c]
        fp = confusion_mat[:, c].sum() - tp
        fn = confusion_mat[c, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        per_class_metrics[CLASS_NAMES[c]] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "total_samples": int(confusion_mat[c, :].sum()),
        }

    # Macro-averaged metrics (unweighted mean across all classes)
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    # Row-normalized 4-class confusion matrix
    row_sums = confusion_mat.sum(axis=1, keepdims=True)
    out_zeros = np.zeros_like(confusion_mat, dtype=float)
    norm_mat = np.divide(confusion_mat.astype(float), row_sums, out=out_zeros, where=row_sums != 0)

    return {
        "val_loss": val_loss,
        "val_acc": overall_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": confusion_mat,
        "normalized_confusion_matrix": norm_mat,
        "per_class": per_class_metrics,
    }



def plot_confusion_matrix(
    confusion_mat: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Corruption Classifier Confusion Matrix",
):
    """Plots and saves an annotated confusion matrix heatmap."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    # Normalize by row (Ground truth)
    row_sums = confusion_mat.sum(axis=1, keepdims=True)
    out_zeros = np.zeros_like(confusion_mat, dtype=float)
    norm_mat = np.divide(confusion_mat.astype(float), row_sums, out=out_zeros, where=row_sums != 0)


    im = ax.imshow(norm_mat, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Normalized Rate", rotation=-90, va="bottom", fontsize=10)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10, fontweight="bold")
    ax.set_yticklabels(class_names, fontsize=10, fontweight="bold")

    ax.set_xlabel("Predicted Corruption", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Ground Truth Corruption", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    # Annotate cells with count and percentage
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            raw_cnt = confusion_mat[i, j]
            pct = norm_mat[i, j] * 100
            color = "white" if norm_mat[i, j] > 0.5 else "black"
            text = f"{raw_cnt}\n({pct:.1f}%)"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def train(args):
    # If a config JSON is provided (or if checkpoints/best_classifier_hyperparameters.json exists), load it
    if hasattr(args, "config") and args.config and os.path.exists(args.config):
        print(f"Loading optimal classifier hyperparameters from Optuna config: {args.config}")
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            params = cfg.get("best_params", cfg)
            for k, v in params.items():
                if hasattr(args, k):
                    setattr(args, k, v)
                    print(f"  Applied {k}: {v}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")


    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    manifest_dir = os.path.join(base_dir, "manifests")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)

    # 1. Dataloaders
    print(f"Loading balanced 4-class dataset from: {base_dir}")
    train_loader, val_loader, _ = get_dataloaders(
        base_dir=base_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        manifest_dir=manifest_dir,
        seed=args.seed,
    )

    # 2. Build Classifier Model
    print(f"Initializing CorruptionClassifier (base_channels={args.base_channels}, dropout={args.dropout})")
    model = CorruptionClassifier(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base_channels=args.base_channels,
        dropout_rate=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_checkpoint_path = os.path.join(args.save_dir, "best_corruption_classifier.pt")
    log_csv_path = os.path.join(args.save_dir, "classifier_training_log.csv")

    # Initialize CSV log file
    with open(log_csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,lr,train_loss,train_acc,val_loss,val_acc,clean_f1,noise_f1,blur_f1,occlusion_f1\n")

    print("\n" + "=" * 70)
    print(f"Starting Classifier Training for {args.epochs} Epochs")
    print("=" * 70)

    best_confusion_matrix = None

    for epoch in range(1, args.epochs + 1):
        # A. Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)

        # B. Validate
        val_res = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        val_loss = val_res["val_loss"]
        val_acc = val_res["val_acc"]
        macro_p = val_res["macro_precision"]
        macro_r = val_res["macro_recall"]
        macro_f1 = val_res["macro_f1"]
        per_class = val_res["per_class"]
        curr_lr = optimizer.param_groups[0]["lr"]

        # C. Console Report
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} (Acc: {train_acc * 100:.2f}%) | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"Macro-F1: {macro_f1:.4f} (P: {macro_p * 100:.1f}%, R: {macro_r * 100:.1f}%)"
        )

        f1_summary = " | ".join([
            f"{s_name}: F1={per_class[c_name]['f1']:.3f} (P={per_class[c_name]['precision']:.2f}, R={per_class[c_name]['recall']:.2f})"
            for s_name, c_name in zip(SHORT_NAMES, CLASS_NAMES)
        ])
        print(f"           Per-Class -> {f1_summary}")

        # D. Save to CSV
        with open(log_csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{curr_lr:.6f},{train_loss:.4f},{train_acc:.4f},"
                f"{val_loss:.4f},{val_acc:.4f},"
                f"{per_class[CLASS_NAMES[0]]['f1']:.4f},"
                f"{per_class[CLASS_NAMES[1]]['f1']:.4f},"
                f"{per_class[CLASS_NAMES[2]]['f1']:.4f},"
                f"{per_class[CLASS_NAMES[3]]['f1']:.4f}\n"
            )

        # E. Save Best Checkpoint & Complete Metrics JSON
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_confusion_matrix = val_res["confusion_matrix"]
            best_norm_matrix = val_res["normalized_confusion_matrix"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "macro_precision": macro_p,
                    "macro_recall": macro_r,
                    "macro_f1": macro_f1,
                    "per_class": per_class,
                    "confusion_matrix": best_confusion_matrix.tolist(),
                    "normalized_confusion_matrix": best_norm_matrix.tolist(),
                    "config": vars(args),
                },
                best_checkpoint_path,
            )
            print(f"  --> Saved new best classifier checkpoint! (Val Acc: {val_acc * 100:.2f}%, Macro-F1: {macro_f1:.4f})")

            # Save full metrics JSON for IEEE research paper tables
            metrics_json_path = os.path.join(args.save_dir, "classifier_evaluation_metrics.json")
            with open(metrics_json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "best_epoch": epoch,
                    "overall_accuracy": val_acc,
                    "macro_precision": macro_p,
                    "macro_recall": macro_r,
                    "macro_f1": macro_f1,
                    "per_class": per_class,
                    "raw_confusion_matrix": best_confusion_matrix.tolist(),
                    "normalized_confusion_matrix": best_norm_matrix.tolist(),
                    "class_names": CLASS_NAMES,
                    "short_names": SHORT_NAMES,
                    "config": vars(args),
                }, f, indent=2)

            # Update confusion matrix plot
            cm_plot_path = os.path.join(args.samples_dir, "classifier_confusion_matrix.png")
            plot_confusion_matrix(
                best_confusion_matrix,
                SHORT_NAMES,
                cm_plot_path,
                title=f"Classifier Confusion Matrix (Epoch {epoch}, Acc: {val_acc * 100:.1f}%)"
            )

        # F. Log to Excel Tracker
        if append_classifier_epoch is not None:
            excel_path = os.path.join(CURRENT_DIR, "task2_specialists_tracker.xlsx")
            try:
                append_classifier_epoch(
                    epoch=epoch,
                    lr=curr_lr,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    per_class=per_class,
                    is_best=is_best,
                    notes="New Best Accuracy" if is_best else "",
                    excel_path=excel_path,
                )
            except Exception as e:
                print(f"  Warning: Failed to log epoch to Excel: {e}")


    print("\n" + "=" * 70)
    print(f"Classifier Training Complete! Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Best checkpoint saved at: {best_checkpoint_path}")
    print(f"Confusion matrix plot saved at: {os.path.join(args.samples_dir, 'classifier_confusion_matrix.png')}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 4-Class Corruption Classifier (Task 2)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--base_channels", type=int, default=32, help="Base convolutional channels")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--num_workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--samples_dir", type=str, default="visual_samples", help="Directory to save plots")
    parser.add_argument("--config", type=str, default=None, help="Path to best_classifier_hyperparameters.json from Optuna")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")


    args = parser.parse_args()
    train(args)
